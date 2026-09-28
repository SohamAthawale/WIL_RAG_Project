"""T12 -- the full comparison table: four baselines against eight dimensions.

The report's central artefact. Its job is to be readable without being misleading, which
here takes more care than computing it does.

FOUR CELL STATES, because "blank" hides three different situations:

  a value        measured, and it carries the n it was measured on
  not measured   the task that produces it has not been done
  no run         the system exists but generation was never run for it
  n/a            the dimension does not apply to that system

B1 and B3 carry "no run" on every generation dimension. rag.retrieve is dense cosine,
so the only generated answers in the project are B0 (no retrieval) and B2 (dense). BM25
and RRF are scored at retrieval only. A table that left those cells blank would read as
zero or as pending; they are neither.

THREE DENOMINATORS, all correct, none interchangeable:

  n=95  chunk-level retrieval metrics, over the qrels, minus five questions with no
        relevant chunk
  n=61  document-level retrieval, over questions with a single gold document
  n=54  wrong-subclass citations: of those 61, seven point at the condition 8547 file,
        which names no subclass to compare against

and n=100 for anything computed from answers alone, n=13 where a human grade is needed.
Every cell prints its own n for this reason.

RECALL IS REPORTED AS SHARE OF ITS CEILING. Recall@1 cannot exceed 1/|relevant|, and
these questions carry 5.58 relevant chunks on average, so the ceiling is 0.192. A raw
0.182 reads as near-total failure and is in fact 95% of the achievable maximum. The raw
figure is also not comparable across question groups whose relevant-set sizes differ.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
OUT_CSV = R / "full_comparison.csv"
OUT_MD = R / "full_comparison.md"

NOT_MEASURED = "not yet measured"
NO_RUN = "no run"
NA = "n/a"

SYSTEMS = [
    ("B0", "no retrieval"),
    ("B1", "BM25"),
    ("B2", "dense"),
    ("B3", "hybrid RRF"),
]
# Which retriever key each baseline maps to in the retrieval result files.
RETRIEVER = {"B1": "bm25", "B2": "dense", "B3": "hybrid_rrf"}
# Generation was only ever run for B0 and B2.
HAS_GENERATION = {"B0", "B2"}


def load(name: str):
    path = R / name
    return json.loads(path.read_text()) if path.exists() else None


def cell(value, n=None, note=None) -> str:
    if value in (NOT_MEASURED, NO_RUN, NA) or value is None:
        return value or NOT_MEASURED
    s = f"{value}"
    if n is not None:
        s += f" (n={n})"
    if note:
        s += f" [{note}]"
    return s


def main() -> None:
    ans = load("answerability.json")
    grd = load("grounding.json")
    ret = load("retrieval_metrics.json")
    con = load("confusion_counters.json")

    rows = []
    for sid, label in SYSTEMS:
        r = {"baseline": sid, "retrieval": label}
        gen = sid in HAS_GENERATION

        # 1 Answerability -- the split, reported as two numbers, never one.
        if ans and gen:
            key = "B0" if sid == "B0" else "RAG"
            s = ans["splits"]["full_n100"][key]["split"]
            r["1_answerability"] = cell(
                f"correct_refusal {s.get('correct_refusal', 0)}, "
                f"answered_but_wrong {s.get('answered_but_wrong', 0)}", 100)
        else:
            r["1_answerability"] = cell(NO_RUN if not gen else NOT_MEASURED)

        # 2 Faithfulness -- numeric facts only.
        if grd and sid == "B2":
            f = grd["faithfulness"]
            r["2_faithfulness"] = cell(f"{f['supported_rate']}", f["n_questions"], "numeric facts only")
        elif sid == "B0":
            r["2_faithfulness"] = cell(NA, note=None)  # nothing retrieved to be faithful to
        else:
            r["2_faithfulness"] = cell(NO_RUN)

        # 3 Attribution correctness.
        if grd and sid == "B2":
            a = grd["attribution"]
            r["3_attribution"] = cell(
                f"cites {a['citation_rate']['answers_naming_a_corpus_document']}/100, "
                f"instructed form {a['format_compliance']['instructed_form']}/100, "
                f"fabricated {a['fabricated_citations']['answers_with_a_fabricated_citation']}/100, "
                f"wrong-subclass {a['wrong_subclass_citations']['answers_citing_another_subclass']}/"
                f"{a['wrong_subclass_citations']['n']}")
        elif sid == "B0" and grd:
            r["3_attribution"] = cell(
                f"cites {grd['b0_contrast']['b0_answers_naming_a_corpus_document']}/100", 100,
                "never cites")
        else:
            r["3_attribution"] = cell(NO_RUN)

        # 4 Retrieval quality -- chunk level, share of the recall ceiling.
        if sid == "B0":
            r["4_retrieval"] = cell(NA)
        elif ret:
            m = ret["configs"]["structure_aware_meta"]["combined"]["systems"][RETRIEVER[sid]]
            r["4_retrieval"] = cell(
                f"NDCG@5 {m['chunk_ndcg@5']}, MRR {m['chunk_mrr']}, "
                f"R@1 {m['chunk_recall@1_share_of_ceiling']} of ceiling", 95, "chunk level")
        else:
            r["4_retrieval"] = cell(NOT_MEASURED)

        # 5 Cross-subclass confusion -- the headline metric.
        if sid == "B0":
            r["5_cross_subclass"] = cell(NA)
        elif con and sid == "B2":
            run = next((x for x in con["runs"] if x.get("top_k") == 3 and x.get("corpus_current")), None)
            if run:
                c = run["counts"]
                r["5_cross_subclass"] = cell(
                    f"(a) {c['a_cites_wrong_subclass']}, (b) {c['b_fact_from_wrong_subclass']}, "
                    f"(c) {c['c_wrong_subclass_at_1']}, (d) {c['d_cites_absent_source']}, "
                    f"(e) {c['e_fact_from_wrong_stream']}", run["n_questions"], "5 counters")
            else:
                r["5_cross_subclass"] = cell(NOT_MEASURED)
        elif ret:
            # Retrieval-level proxy is available for every retriever.
            r["5_cross_subclass"] = cell(NO_RUN, note=None)
        else:
            r["5_cross_subclass"] = cell(NOT_MEASURED)

        # 6 Refusal compliance.
        if ans and gen:
            key = "B0_n100" if sid == "B0" else "RAG_n100"
            d = ans["diagnostics"][key]
            r["6_refusal"] = cell(
                f"declines {d['refused']}/{d['n_refusal_required']}, "
                f"hedged-then-advised {d['hedged_then_advised']}", 15)
        else:
            r["6_refusal"] = cell(NO_RUN)

        r["7_vocabulary_fairness"] = cell(NOT_MEASURED)
        r["8_temporal"] = cell(NOT_MEASURED)
        rows.append(r)

    fields = ["baseline", "retrieval", "1_answerability", "2_faithfulness", "3_attribution",
              "4_retrieval", "5_cross_subclass", "6_refusal", "7_vocabulary_fairness",
              "8_temporal"]
    with OUT_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    snapshot = (ret or {}).get("snapshot_id", "84636c1d6741")
    md = [
        "# Full comparison — four baselines × eight dimensions",
        "",
        f"Snapshot `{snapshot}` · config `structure_aware_meta` · k=3 · `qwen2.5:7b-instruct`,",
        "temperature 0, seed 97.",
        "",
        "**Cell states.** A value carries the n it was measured on. `not yet measured` means the",
        "task has not been done. `no run` means the system exists but generation was never run for",
        "it — `rag.retrieve` is dense cosine, so B1 and B3 are scored at retrieval only. `n/a` means",
        "the dimension does not apply.",
        "",
        "**Denominators differ by measure and all are correct:** n=95 chunk-level retrieval, n=61",
        "document-level, n=54 wrong-subclass citations, n=100 answer-only measures, n=13 where a",
        "human grade is required. Each cell states its own.",
        "",
        "**Recall is share of its ceiling.** Recall@1 cannot exceed 1/|relevant|; at 5.58 relevant",
        "chunks per question the ceiling is 0.192, so a raw 0.182 is 95% of the maximum, not a",
        "failure.",
        "",
    ]
    for f in fields[2:]:
        md.append(f"## {f.split('_', 1)[1].replace('_', ' ').title()}")
        md.append("")
        md.append("| Baseline | Retrieval | Result |")
        md.append("|---|---|---|")
        for r in rows:
            md.append(f"| {r['baseline']} | {r['retrieval']} | {r[f]} |")
        md.append("")
    OUT_MD.write_text("\n".join(md) + "\n")

    measured = sum(1 for f in fields[2:]
                   if any(rw[f] not in (NOT_MEASURED, NO_RUN, NA) for rw in rows))
    print(f"{measured} of 8 dimensions have at least one measured cell")
    for r in rows:
        filled = sum(1 for f in fields[2:] if r[f] not in (NOT_MEASURED, NO_RUN, NA))
        print(f"  {r['baseline']} ({r['retrieval']:<12}) {filled}/8 cells measured")
    print(f"\nWrote {OUT_CSV.relative_to(ROOT)} and {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
