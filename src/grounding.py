"""Faithfulness and attribution correctness: is the answer grounded in what was retrieved?

Dimensions 2 and 3.

  Attribution (3) -- does the answer cite a document that was actually in its context, and
  the right one? A citation naming a document the system never retrieved manufactures
  trust that was not earned, which in this domain is what makes someone act on an answer.

  Faithfulness (2) -- is every checkable claim traceable to the retrieved context, or did
  the model supply it from training?

Both reuse src/confusion.py rather than reimplementing anything. cited_sources matches the
five known corpus filenames instead of a citation format, which matters because the model
largely ignores the format its prompt asks for: across the 100 answers in the current run,
24 use the instructed [Source: f] form, 8 use Source: [f], and 68 cite freeform or not at
all. A parser built on the documented format scores most of the set as uncited and every
number below it is then wrong in the direction that looks fine. T06's first version did
exactly that. The low compliance rate is itself a finding and is reported.

DECISIONS ENCODED HERE:

  Context, for faithfulness, is the retrieved chunks PLUS the question. Two of the three
  facts that fail a chunks-only test are the model repeating a figure the user supplied
  (q13 "8 months", q73 "7 months"). Those are grounded in the prompt, not invented, and
  counting them as hallucinations measures the wrong thing -- it would put the rate at
  roughly three times its true value.

  Questions with gold_source "none" stay in the fabrication denominator and leave the
  wrong-subclass one. Fabrication is cited-but-not-retrieved, which is defined on every
  row because retrieval returns k chunks whether or not the corpus answers the question.
  Wrong-subclass needs a gold subclass to compare against and there is none. Rows whose
  gold_source is "multiple" leave it for the same reason, following confusion.score_row.

  The one remaining unsupported fact is kept in the count and annotated, not excluded.
  See REVIEWED_UNSUPPORTED.

LIMITATIONS, all carried into the output file:

  This is NUMERIC faithfulness. A fabricated qualitative claim carrying no figure is
  invisible to it. "Faithfulness 0.99" means no invented numbers, not no invented content.

  Exact-match fact checking cannot tell a derivation from a fabrication. q78 states "36
  months" where the three retrieved chunks each say "Stay: 12 months" for the first,
  second and third Working Holiday visa -- correct arithmetic the check cannot recognise.
  It stays flagged, because the detector genuinely cannot see the difference, and the
  review is recorded so a reader knows it was checked.

  cited_sources matches a filename stem anywhere in the answer, so a document mentioned in
  passing counts as cited. That is the right trade for this corpus -- the five stems are
  distinctive enough that prose does not produce one by accident -- but it measures
  "names the document", not "cites it in a citation".

  The before/after on fabricated citations is not a controlled comparison. See
  fabrication_before_after in the output.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Read-only reuse. A second citation parser or fact extractor would silently diverge from
# the one T06's and T09's numbers were built on.
import confusion as cf  # noqa: E402

TESTSET = ROOT / "data" / "testset" / "test_set.json"
RESULTS = ROOT / "results"
CURRENT_RUN = RESULTS / "b0_vs_rag__structure_aware_meta_k3.json"
PREFIX_RUN = RESULTS / "b0_vs_rag_comparison.json"
OUT = RESULTS / "grounding.json"

INSTRUCTED_STYLE = "bracketed_source"

# Unsupported facts that have been read and judged by hand. Recorded rather than removed:
# excluding a row after looking at it is how a measure gets fitted to its data, and the
# count below still includes them.
REVIEWED_UNSUPPORTED = {
    "q78": (
        "Correct derivation, not a fabrication. The answer states 36 months; the three "
        "retrieved chunks (subclass_417_working_holiday::1, ::2, ::3) are the first, "
        "second and third Working Holiday visa and each states 'Stay: 12 months'. The "
        "model multiplied. Exact-match fact checking cannot recognise a derived aggregate, "
        "so this stays in the unsupported count with the arithmetic recorded."
    ),
}


def parse_retrieved_sources(row: dict, k: int) -> list[str]:
    """The source filenames a recorded run retrieved, in rank order, truncated to k."""
    parts = [s.split("(")[0].strip() for s in row["rag_retrieved_sources"].split(";") if s.strip()]
    return parts[:k]


def attribution_for_row(row: dict, known: set[str], gold_source: str, k: int) -> dict:
    """Attribution correctness for one answer."""
    answer = row["rag_answer"]
    cited = cf.cited_sources(answer, known)
    retrieved = parse_retrieved_sources(row, k)

    fabricated = [c for c in cited if c not in retrieved]

    # Wrong-subclass is only defined when a single gold document exists. "multiple" spans
    # documents and "none" has no answer in the corpus at all.
    gold_sc = cf.subclass_of(gold_source) if gold_source not in ("multiple", "none", "") else None
    if gold_sc is None:
        wrong_subclass = None
    else:
        wrong_subclass = [c for c in cited if cf.subclass_of(c) not in (None, gold_sc)]

    return {
        "cited": cited,
        "retrieved": retrieved,
        "citation_style": cf.citation_style(answer),
        "cites_anything": bool(cited),
        "fabricated": fabricated,
        "wrong_subclass": wrong_subclass,
        "wrong_subclass_scoreable": gold_sc is not None,
    }


def faithfulness_for_row(row: dict, chunks: list[dict], k: int) -> dict:
    """Numeric faithfulness for one answer.

    Context is the retrieved chunks plus the question: a figure the user supplied and the
    model repeated is grounded in the prompt, not invented.
    """
    by_id = {c["id"]: c for c in chunks}
    ids = cf.retrieved_chunk_ids(row, chunks, k)
    if ids is None:
        return {"resolvable": False}

    answer_facts = cf.facts_in(row["rag_answer"])
    context_facts: set = set()
    for cid in ids:
        if cid in by_id:
            context_facts |= cf.facts_in(by_id[cid]["text"])
    question_facts = cf.facts_in(row["question"])
    unsupported = answer_facts - context_facts - question_facts

    return {
        "resolvable": True,
        "retrieved_chunk_ids": ids,
        "n_answer_facts": len(answer_facts),
        "n_unsupported": len(unsupported),
        "unsupported": sorted(f"{n} {u}" for n, u in unsupported),
    }


def score_run(rows, chunks, testset, k) -> dict:
    known = {c["source"] for c in chunks}
    per_question = {}
    for row in rows:
        gold_source = (testset.get(row["id"]) or {}).get("gold_source", "")
        entry = attribution_for_row(row, known, gold_source, k)
        entry["faithfulness"] = faithfulness_for_row(row, chunks, k)
        entry["gold_source"] = gold_source
        per_question[row["id"]] = entry
    return per_question


def aggregate(per_question: dict, rows) -> dict:
    n = len(rows)
    entries = list(per_question.values())

    styles = Counter(e["citation_style"] for e in entries)
    citing = [e for e in entries if e["cites_anything"]]
    fabricating = [e for e in entries if e["fabricated"]]
    subclass_scoreable = [e for e in entries if e["wrong_subclass_scoreable"]]
    wrong_sc = [e for e in subclass_scoreable if e["wrong_subclass"]]

    resolvable = [e for e in entries if e["faithfulness"]["resolvable"]]
    total_facts = sum(e["faithfulness"]["n_answer_facts"] for e in resolvable)
    unsupported = sum(e["faithfulness"]["n_unsupported"] for e in resolvable)

    return {
        "attribution": {
            "n": n,
            "citation_rate": {
                "answers_naming_a_corpus_document": len(citing),
                "n": n,
                "rate": round(len(citing) / n, 3),
                "note": "Just over half cite nothing, on a system whose prompt instructs it to cite every fact.",
            },
            "format_compliance": {
                "styles": dict(styles),
                "instructed_form": styles.get(INSTRUCTED_STYLE, 0),
                "of_n": n,
                "rate_of_all": round(styles.get(INSTRUCTED_STYLE, 0) / n, 3),
                "rate_of_citing": round(styles.get(INSTRUCTED_STYLE, 0) / len(citing), 3) if citing else None,
                "note": (
                    "The prompt asks for [Source: filename]. A parser built on that format "
                    "alone would score the rest of the set as uncited."
                ),
            },
            "fabricated_citations": {
                "answers_with_a_fabricated_citation": len(fabricating),
                "n": n,
                "rate": round(len(fabricating) / n, 3),
                "ids": sorted(qid for qid, e in per_question.items() if e["fabricated"]),
            },
            "wrong_subclass_citations": {
                "answers_citing_another_subclass": len(wrong_sc),
                "n": len(subclass_scoreable),
                "rate": round(len(wrong_sc) / len(subclass_scoreable), 3) if subclass_scoreable else None,
                "ids": sorted(qid for qid, e in per_question.items()
                              if e["wrong_subclass_scoreable"] and e["wrong_subclass"]),
                "denominator_note": (
                    "Scoreable only where the gold document names a subclass. 61 questions "
                    "have a single gold document; 7 of those point at "
                    "whm_6_month_work_limitation.txt, which is a condition rather than a "
                    "subclass and yields no gold subclass to compare against, leaving 54. "
                    "'multiple' spans documents and 'none' has no corpus answer, so neither "
                    "defines one either. This n is therefore smaller than the attribution n "
                    "and smaller again than the 61 used by the document-level retrieval "
                    "metrics in config_comparison.json."
                ),
            },
        },
        "faithfulness": {
            "n_questions_resolvable": len(resolvable),
            "n_questions": n,
            "n_answer_facts": total_facts,
            "n_unsupported_facts": unsupported,
            "supported_rate": round(1 - unsupported / total_facts, 3) if total_facts else None,
            "one_in": round(total_facts / unsupported, 1) if unsupported else None,
            "unsupported_detail": [
                {
                    "id": qid,
                    "facts": e["faithfulness"]["unsupported"],
                    "review": REVIEWED_UNSUPPORTED.get(qid, "not yet reviewed"),
                }
                for qid, e in sorted(per_question.items())
                if e["faithfulness"]["resolvable"] and e["faithfulness"]["n_unsupported"]
            ],
            "context_definition": (
                "Retrieved chunks at top-k, plus the question. A figure the user supplied "
                "and the model repeated is grounded in the prompt rather than invented."
            ),
            "measures_only": (
                "Numeric facts. A fabricated qualitative claim carrying no figure is "
                "invisible to this measure."
            ),
        },
    }


def b0_citation_contrast(rows, known) -> dict:
    """Whether the no-retrieval baseline invents citations.

    B0 receives no context, so any corpus document it named would be fabricated by
    construction. Reported as the contrast that shows citation behaviour is a product of
    the RAG prompt rather than of the model.
    """
    naming = [r["id"] for r in rows if cf.cited_sources(r["b0_answer"], known)]
    return {
        "b0_answers_naming_a_corpus_document": len(naming),
        "n": len(rows),
        "ids": naming,
        "note": (
            "B0 has no retrieved context, so any citation would be fabricated by "
            "construction. It cites nothing at all: the baseline does not invent "
            "citations, it simply never cites."
        ),
    }


def main() -> None:
    testset = {t["id"]: t for t in json.loads(TESTSET.read_text())}

    current = json.loads(CURRENT_RUN.read_text())
    meta, rows = current["run_meta"], current["rows"]
    chunks = cf.load_store(meta["snapshot_id"], meta["config_id"])
    known = {c["source"] for c in chunks}
    k = meta["top_k"]

    per_question = score_run(rows, chunks, testset, k)
    agg = aggregate(per_question, rows)

    # The positive control. This run predates the corpus correction and the temperature
    # pin, and is a bare list with no run_meta, so its snapshot, config and k are not
    # recorded anywhere.
    prefix_raw = json.loads(PREFIX_RUN.read_text())
    prefix_rows = prefix_raw["rows"] if isinstance(prefix_raw, dict) else prefix_raw
    prefix_fab = []
    for row in prefix_rows:
        cited = cf.cited_sources(row["rag_answer"], known)
        retrieved = parse_retrieved_sources(row, k)
        bad = [c for c in cited if c not in retrieved]
        if bad:
            prefix_fab.append({"id": row["id"], "cited": bad, "retrieved": retrieved})

    out = {
        "task": "T07 -- faithfulness and attribution correctness",
        "config": {
            "snapshot_id": meta["snapshot_id"],
            "config_id": meta["config_id"],
            "top_k": k,
            "gen_model": meta["gen_model"],
            "run_file": CURRENT_RUN.name,
        },
        "headline": (
            "Zero fabricated citations across 100 answers, and zero invented figures -- "
            "but only 49 of 100 answers cite anything at all, on a system instructed to "
            "cite every fact."
        ),
        **agg,
        "b0_contrast": b0_citation_contrast(rows, known),
        "fabrication_before_after": {
            "before": {
                "run_file": PREFIX_RUN.name,
                "n": len(prefix_rows),
                "answers_with_a_fabricated_citation": len(prefix_fab),
                "detail": prefix_fab,
                "run_meta": None,
            },
            "after": {
                "run_file": CURRENT_RUN.name,
                "n": len(rows),
                "answers_with_a_fabricated_citation": len(agg["attribution"]["fabricated_citations"]["ids"]),
            },
            "not_a_controlled_comparison": (
                "The before run carries no run_meta, so its snapshot, config and top_k are "
                "unrecorded; it predates both the August corpus correction and the "
                "27 Sep temperature pin; and it is 13 questions against 100 on a different "
                "corpus. The change is real and worth reporting, but it is a before-and-after "
                "across two different setups, not a controlled experiment. k was assumed to "
                "be the current run's top_k when parsing it, because the file does not say."
            ),
        },
        "limitations": [
            "Numeric faithfulness only. A fabricated qualitative claim carrying no figure is not detected.",
            "Exact-match fact checking cannot distinguish a derivation from a fabrication: q78's '36 months' is correct arithmetic over three chunks each stating 'Stay: 12 months', and is still counted as unsupported.",
            "Context for faithfulness is the retrieved chunks plus the question. A chunks-only definition would report three times as many unsupported facts, two of them the model repeating a figure the user supplied.",
            "cited_sources matches a filename stem anywhere in the answer, so a document named in passing counts as cited. It measures 'names the document', not 'cites it in the instructed form'.",
            "Wrong-subclass citations are scored only on rows with a single gold document, so that figure carries a smaller n than the others.",
            "The fabrication before/after spans two corpora, two sample sizes and an unrecorded config. See not_a_controlled_comparison.",
            "Human rag_correct exists for q01-q13 only; rag_correct_manual is empty on all 100 rows, so none of these measures are validated against human judgement.",
        ],
        "per_question": per_question,
    }

    OUT.write_text(json.dumps(out, indent=2))

    a, f = agg["attribution"], agg["faithfulness"]
    print(f"run {CURRENT_RUN.name}  snapshot {meta['snapshot_id']}  {meta['config_id']}  k={k}  n={len(rows)}")
    print()
    print(f"  citation rate          {a['citation_rate']['answers_naming_a_corpus_document']}/{a['citation_rate']['n']}"
          f"  ({a['citation_rate']['rate']:.0%})")
    print(f"  instructed format      {a['format_compliance']['instructed_form']}/{a['format_compliance']['of_n']}"
          f"  ({a['format_compliance']['rate_of_all']:.0%} of all, "
          f"{a['format_compliance']['rate_of_citing']:.0%} of those that cite)")
    print(f"  styles                 {a['format_compliance']['styles']}")
    print(f"  fabricated citations   {a['fabricated_citations']['answers_with_a_fabricated_citation']}/{a['fabricated_citations']['n']}")
    print(f"  wrong-subclass cites   {a['wrong_subclass_citations']['answers_citing_another_subclass']}/{a['wrong_subclass_citations']['n']}"
          f"   (n smaller: single-gold-document rows only)")
    print()
    print(f"  numeric faithfulness   {f['n_answer_facts'] - f['n_unsupported_facts']}/{f['n_answer_facts']} facts supported"
          f"  ({f['supported_rate']:.3f}), 1 unsupported in {f['one_in']}")
    for d in f["unsupported_detail"]:
        print(f"      {d['id']}: {d['facts']}")
    print()
    print(f"  B0 naming a corpus doc {out['b0_contrast']['b0_answers_naming_a_corpus_document']}/{out['b0_contrast']['n']}")
    print(f"  fabrication before     {len(prefix_fab)}/{len(prefix_rows)} in {PREFIX_RUN.name} (no run_meta)")
    print(f"\nWrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
