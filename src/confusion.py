"""Cross-subclass confusion: five counters over the same runs.

The project's headline measure. There is no single agreed way to count "answered
using the wrong visa's rules", so all five defensible definitions are computed side
by side and the headline is chosen after seeing the numbers rather than before:

  (a) cites_wrong_subclass   the answer's [Source: ...] names a wrong-subclass document
  (b) fact_from_wrong_subclass   the answer states a fact found only in a wrong-subclass chunk
  (c) wrong_subclass_at_1    the rank-1 retrieved chunk is a wrong-subclass chunk
  (d) cites_absent_source    the answer cites a document that was never in its context
  (e) fact_from_wrong_stream the fact came from the wrong stream of the *right* subclass

(c) already existed in eval_configs.py and keeps that implementation.

(a) and (d) read the citation. (b) and (e) need to know where a fact came from, which
cannot be read off the answer text -- so facts are matched back to chunks by their
discriminating numbers, and "wrong subclass" comes from the human judgements in T02
rather than from a filename heuristic.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rag  # noqa: E402  -- reuses the pipeline's own retrieval, not a copy of it

ROOT = Path(__file__).resolve().parent.parent
TESTSET = ROOT / "data" / "testset" / "test_set.json"
SNAPSHOTS = ROOT / "data" / "snapshots"
RESULTS = ROOT / "results"
OUT = RESULTS / "confusion_counters.json"

# The prompt asks for "[Source: filename]" and the model complies about one time in
# ten. Observed forms include "Source: [f]", "sourced from [f]", "in the source
# document f" and bare mentions, so a citation is instead any occurrence of a known
# corpus filename: the document set is closed, and the stems are distinctive enough
# that prose never produces one by accident.
CITATION_STYLES = [
    ("bracketed_source", re.compile(r"\[Source:\s*([^\]]+)\]", re.I)),
    ("source_bracketed", re.compile(r"Source:\s*\[([^\]]+)\]", re.I)),
]

# A fact is a number bound to the unit it qualifies. "between 2 and 3 years" yields
# (2, year) and (3, year); the 1-and-2-years stream yields (1, year) and (2, year), so
# the 3 discriminates between them and the 2 does not. Only discriminating facts count.
UNITS = r"hours?|fortnights?|weeks?|months?|years?|days?"
# "between 2 and 3 years" states two figures. Taken with a plain number-then-unit scan
# only the 2 is found, because the scanner has already consumed the 3 -- and the 3 is
# exactly what separates the Post-Higher Education Work stream from the Second one.
RANGE = re.compile(rf"(\d+)\s*(?:and|to|-|\u2013)\s*(\d+)\s*({UNITS})", re.I)
SIMPLE = re.compile(rf"(\d[\d,]*(?:\.\d+)?)\s*(?:\w+\s+){{0,2}}?({UNITS})", re.I)
CURRENCY = re.compile(r"AUD\s*(\d[\d,]*(?:\.\d+)?)", re.I)
# The subclass and condition numbers name the document. They are not claims about work
# rights, and counting them makes every 485 chunk look like the source of every 485
# answer.
IDENTIFIER = re.compile(r"\b(?:subclass|condition)\s*\d+", re.I)
LABEL_PREFIX = re.compile(r"^\[[^\]]*\]\s*")

def subclass_of(name: str) -> str | None:
    """The visa subclass a filename belongs to, or None for cross-cutting documents."""
    m = re.search(r"subclass_(\d+)", name)
    return m.group(1) if m else None


def normalise_number(raw: str) -> str:
    n = raw.replace(",", "")
    return n.rstrip("0").rstrip(".") if "." in n else n


def facts_in(text: str) -> set[tuple[str, str]]:
    """Number-unit pairs stated in a passage, as claims rather than identifiers."""
    text = IDENTIFIER.sub(" ", LABEL_PREFIX.sub("", text))
    out = set()
    for lo, hi, unit in RANGE.findall(text):
        u = unit.lower().rstrip("s")
        out.add((normalise_number(lo), u))
        out.add((normalise_number(hi), u))
    for num, unit in SIMPLE.findall(text):
        out.add((normalise_number(num), unit.lower().rstrip("s")))
    for amount in CURRENCY.findall(text):
        out.add((normalise_number(amount), "aud"))
    return out


def stream_of(chunk: dict) -> str | None:
    """The stream heading a chunk sits under, if it is a stream chunk.

    Streams inside one subclass are the hardest confusion to see: the headings share
    most of their words and every filename-level check scores them as correct.
    """
    body = re.sub(r"^\[[^\]]*\]\s*", "", chunk["text"]).strip()
    head = body.split("\n", 1)[0].strip()
    return head if re.search(r"\bstream\b", head, re.I) else None


def load_store(snapshot: str, config: str) -> list[dict]:
    path = SNAPSHOTS / snapshot / f"vector_store__{config}.json"
    return json.loads(path.read_text())["chunks"]


def load_testset() -> list[dict]:
    data = json.loads(TESTSET.read_text())
    return data["questions"] if isinstance(data, dict) else data


def retrieved_chunk_ids(row: dict, chunks: list[dict], k: int) -> list[str] | None:
    """Recover which chunks a recorded run actually retrieved.

    The run files store source filenames and scores, not chunk ids, and a filename is
    ambiguous whenever its document contributed more than one chunk -- which is almost
    always. Retrieval is deterministic cosine over stored embeddings, so it is re-run
    here and accepted only when the source-and-score sequence reproduces the recorded
    one exactly. A near miss means the embeddings or the corpus moved underneath the
    run, and scoring it anyway would attribute facts to chunks the run never saw.

    Scores are compared as numbers, not as the strings the run file happens to hold:
    json writes round(0.810, 3) as "0.81", so a string comparison rejects runs that
    reproduced perfectly.
    """
    recorded = []
    for part in row["rag_retrieved_sources"].split(";"):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"(.+?)\s*\(([\d.]+)\)$", part)
        if not m:
            return None
        recorded.append((m.group(1).strip(), round(float(m.group(2)), 3)))
    recorded = recorded[:k]

    hits = rag.retrieve(row["question"], chunks, k=max(k, len(recorded)))[: len(recorded)]
    replayed = [(h["source"], round(h["score"], 3)) for h in hits]
    if replayed != recorded:
        return None
    return [h["id"] for h in hits]


def cited_sources(answer: str, known: set[str]) -> list[str]:
    """Every corpus document the answer names, in any of the forms the model uses."""
    found = []
    for src in sorted(known):
        stem = src[:-4] if src.endswith(".txt") else src
        if re.search(re.escape(stem), answer, re.I):
            found.append(src)
    return found


def citation_style(answer: str) -> str:
    for name, pat in CITATION_STYLES:
        if pat.search(answer):
            return name
    return "freeform_or_absent"


def score_row(row: dict, task: dict, chunks: list[dict], k: int) -> dict:
    by_id = {c["id"]: c for c in chunks}
    answer = row["rag_answer"]
    gold_source = task.get("gold_source", "")
    gold_sc = subclass_of(gold_source) if gold_source not in ("multiple", "none", "") else None

    retrieved_sources = [
        s.split("(")[0].strip() for s in row["rag_retrieved_sources"].split(";") if s.strip()
    ][:k]
    known = {c["source"] for c in chunks}
    cited = cited_sources(answer, known)

    # (a) a citation naming a document from another subclass.
    a = any(
        gold_sc and subclass_of(c) and subclass_of(c) != gold_sc for c in cited
    )
    # (d) a citation naming a document that was never in the context window.
    d = any(c not in retrieved_sources for c in cited)

    # (b) and (e) work on facts, so they need the chunk ids behind those filenames.
    wrong_sc_chunks = set(task.get("wrong_subclass_chunks") or [])
    gold_chunks = set(task.get("gold_chunk_ids") or [])
    ids = retrieved_chunk_ids(row, chunks, k)

    b = False
    e = False
    b_facts: list[str] = []
    e_facts: list[str] = []
    resolvable = ids is not None

    if resolvable:
        ans_facts = facts_in(answer)

        # A fact counts as coming from a chunk only if it discriminates -- present in
        # that chunk and absent from the gold ones. A figure both passages state
        # equally cannot be evidence of where the answer looked.
        gold_facts: set[tuple[str, str]] = set()
        for cid in gold_chunks:
            if cid in by_id:
                gold_facts |= facts_in(by_id[cid]["text"])

        for cid in ids:
            if cid not in by_id or cid not in wrong_sc_chunks:
                continue
            distinctive = (facts_in(by_id[cid]["text"]) & ans_facts) - gold_facts
            if distinctive:
                b = True
                b_facts += [f"{n} {u} <- {cid}" for n, u in sorted(distinctive)]

        # (e) the same test, restricted to the right subclass and different streams.
        #
        # The stream the question asked about is taken from the top-graded chunks, not
        # from everything graded relevant. For q02 the annotator marked all five 485
        # chunks 1 or above and only the Post-Higher Education Work chunk 3 -- so the
        # binary gold set contains the very stream this counter exists to catch, and
        # reading the target off it makes (e) unable to fire at all. This is the case
        # for graded judgements rather than binary ones, stated concretely.
        grades = {k: v for k, v in (task.get("gold_chunk_grades") or {}).items() if v}
        if grades:
            top = max(grades.values())
            target_ids = {cid for cid, g in grades.items() if g == top}
        else:
            target_ids = gold_chunks
        target_streams = {
            stream_of(by_id[cid]) for cid in target_ids if cid in by_id
        } - {None}
        if target_streams:
            for cid in ids:
                chunk = by_id.get(cid)
                if chunk is None or cid in wrong_sc_chunks:
                    continue
                stream = stream_of(chunk)
                if stream is None or stream in target_streams:
                    continue
                other = facts_in(chunk["text"]) & ans_facts
                shared = set()
                for tid in target_ids:
                    if tid in by_id:
                        shared |= facts_in(by_id[tid]["text"])
                distinctive = other - shared
                if distinctive:
                    e = True
                    e_facts += [f"{n} {u}".strip() + f" <- {stream}" for n, u in sorted(distinctive)]

    # (c) keeps eval_configs.py's definition: rank-1 chunk from another subclass.
    c = bool(gold_sc and retrieved_sources and subclass_of(retrieved_sources[0])
             and subclass_of(retrieved_sources[0]) != gold_sc)

    return {
        "id": row["id"],
        "category": row.get("category", ""),
        "a_cites_wrong_subclass": a,
        "b_fact_from_wrong_subclass": b,
        "c_wrong_subclass_at_1": c,
        "d_cites_absent_source": d,
        "e_fact_from_wrong_stream": e,
        "b_evidence": b_facts,
        "e_evidence": e_facts,
        "cited": cited,
        "citation_style": citation_style(answer),
        "chunks_resolvable": resolvable,
    }


def score_run(run_path: Path, testset: list[dict]) -> dict | None:
    data = json.loads(run_path.read_text())
    rows = data["rows"] if isinstance(data, dict) else data
    meta = data.get("run_meta", {}) if isinstance(data, dict) else {}

    snapshot = meta.get("snapshot_id")
    config = meta.get("config_id")
    k = meta.get("top_k")
    if snapshot is None:
        # Pre-fix runs predate run_meta. They are kept as regression fixtures, and
        # their corpus is the superseded snapshot by definition.
        legacy = sorted(p.name for p in SNAPSHOTS.iterdir() if p.is_dir())
        snapshot, config, k = legacy[0], "structure_aware_meta", 3

    try:
        chunks = load_store(snapshot, config)
    except FileNotFoundError:
        return None

    by_qid = {t["id"]: t for t in testset}
    scored = [score_row(r, by_qid[r["id"]], chunks, k) for r in rows if r["id"] in by_qid]

    counters = ["a_cites_wrong_subclass", "b_fact_from_wrong_subclass",
                "c_wrong_subclass_at_1", "d_cites_absent_source",
                "e_fact_from_wrong_stream"]
    return {
        "run": run_path.name,
        "snapshot_id": snapshot,
        "config_id": config,
        "top_k": k,
        "corpus_current": meta.get("corpus_current"),
        "n_questions": len(scored),
        "n_unresolvable": sum(1 for r in scored if not r["chunks_resolvable"]),
        "counts": {c: sum(1 for r in scored if r[c]) for c in counters},
        "citation_styles": {
            st: sum(1 for r in scored if r["citation_style"] == st)
            for st in sorted({r["citation_style"] for r in scored})
        },
        "rates": {c: round(sum(1 for r in scored if r[c]) / len(scored), 3) for c in counters},
        "per_question": scored,
    }


def main() -> None:
    testset = load_testset()
    runs = sorted(RESULTS.glob("b0_vs_rag__*_k*.json")) + [RESULTS / "b0_vs_rag_comparison.json"]
    out = [r for r in (score_run(p, testset) for p in runs if p.exists()) if r]

    header = f"{'run':<44}{'k':>3}{'  a':>5}{'  b':>5}{'  c':>5}{'  d':>5}{'  e':>5}"
    print(header)
    print("-" * len(header))
    for r in out:
        c = r["counts"]
        tag = "" if r["corpus_current"] else "  (superseded corpus)"
        print(f"{r['run']:<44}{r['top_k']:>3}"
              f"{c['a_cites_wrong_subclass']:>5}{c['b_fact_from_wrong_subclass']:>5}"
              f"{c['c_wrong_subclass_at_1']:>5}{c['d_cites_absent_source']:>5}"
              f"{c['e_fact_from_wrong_stream']:>5}{tag}")
        if r["n_unresolvable"]:
            print(f"{'':<47}{r['n_unresolvable']} question(s) had ambiguous chunk ids")

    OUT.write_text(json.dumps({"runs": out}, indent=2))
    print(f"\nWrote {OUT.relative_to(ROOT)}  (counts are questions, n={out[0]['n_questions']})")
    print("(a) cites wrong subclass  (b) fact only in wrong-subclass chunk")
    print("(c) rank-1 chunk wrong subclass  (d) cites source absent from context")
    print("(e) fact from wrong stream of the right subclass")


if __name__ == "__main__":
    main()
