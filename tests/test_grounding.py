"""Controls for faithfulness and attribution correctness.

Both headline numbers on this task are zero -- zero fabricated citations and zero invented
figures -- and a zero is the easiest result in the project to produce by accident. A
detector wired to nothing reports exactly the same clean table as a system that never errs.
So every check here is paired: the real run must read zero, and a synthetic failure built
from the same corpus must trip the same code path.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import confusion as cf  # noqa: E402
import grounding as gr  # noqa: E402

CURRENT = json.loads((ROOT / "results" / "b0_vs_rag__structure_aware_meta_k3.json").read_text())
META, ROWS = CURRENT["run_meta"], CURRENT["rows"]
CHUNKS = cf.load_store(META["snapshot_id"], META["config_id"])
KNOWN = {c["source"] for c in CHUNKS}
K = META["top_k"]
BY_ID = {r["id"]: r for r in ROWS}
TESTSET = {t["id"]: t for t in json.loads((ROOT / "data" / "testset" / "test_set.json").read_text())}

PREFIX_RAW = json.loads((ROOT / "results" / "b0_vs_rag_comparison.json").read_text())
PREFIX = {r["id"]: r for r in (PREFIX_RAW["rows"] if isinstance(PREFIX_RAW, dict) else PREFIX_RAW)}

failures: list[str] = []


def check(name: str, got, want) -> None:
    status = "ok " if got == want else "FAIL"
    print(f"  {status} {name}")
    if got != want:
        failures.append(f"{name}: got {got!r}, want {want!r}")


print("citation detection is format-independent, which is the whole point")
check("the instructed form is found",
      cf.cited_sources("The cap is 48 hours [Source: subclass_500_student.txt]", KNOWN),
      ["subclass_500_student.txt"])
check("the reversed form is found",
      cf.cited_sources("Source: [subclass_500_student.txt]", KNOWN),
      ["subclass_500_student.txt"])
check("freeform prose is found",
      cf.cited_sources("as set out in the subclass_500_student document", KNOWN),
      ["subclass_500_student.txt"])
check("an answer naming nothing cites nothing",
      cf.cited_sources("You may work 48 hours per fortnight.", KNOWN), [])
check("but the style checker still separates the instructed form from the rest",
      (cf.citation_style("x [Source: a.txt]"), cf.citation_style("as stated in the source document")),
      ("bracketed_source", "freeform_or_absent"))

print("the fabricated-citation fixture, named to its run file")
# b0_vs_rag_comparison.json (the pre-fix run, no run_meta) q01 cites subclass_500_student.txt
# while its retrieved list is 417, 485 and 482.
pre_q01 = gr.attribution_for_row(PREFIX["q01"], KNOWN, TESTSET["q01"]["gold_source"], K)
check("pre-fix q01 cites subclass_500_student.txt", pre_q01["cited"], ["subclass_500_student.txt"])
check("and that file is not in its retrieved list",
      "subclass_500_student.txt" in pre_q01["retrieved"], False)
check("so the detector flags it", pre_q01["fabricated"], ["subclass_500_student.txt"])

print("the zero on the current run is a real zero, not a broken join")
# The same row from the current run must come out clean, and a synthetic fabrication built
# from it must still fire. Without the second half, a detector wired to nothing passes.
cur_q01 = gr.attribution_for_row(BY_ID["q01"], KNOWN, TESTSET["q01"]["gold_source"], K)
check("current q01 has no fabricated citation", cur_q01["fabricated"], [])
absent = next(f for f in sorted(KNOWN) if f not in cur_q01["retrieved"])
synthetic = dict(BY_ID["q01"], rag_answer=BY_ID["q01"]["rag_answer"] + f" [Source: {absent}]")
check("injecting a citation to an unretrieved document fires the same check",
      gr.attribution_for_row(synthetic, KNOWN, TESTSET["q01"]["gold_source"], K)["fabricated"],
      [absent])
check("and injecting a citation to a retrieved document does not",
      gr.attribution_for_row(
          dict(BY_ID["q01"], rag_answer=BY_ID["q01"]["rag_answer"] + f" [Source: {cur_q01['retrieved'][0]}]"),
          KNOWN, TESTSET["q01"]["gold_source"], K)["fabricated"],
      [])

print("wrong-subclass scoring is skipped where no gold subclass exists")
check("a question whose gold document is a condition, not a subclass, is not scoreable",
      cf.subclass_of("whm_6_month_work_limitation.txt"), None)
none_row = next(qid for qid, t in TESTSET.items() if t["gold_source"] == "none")
check("gold_source 'none' yields wrong_subclass None",
      gr.attribution_for_row(BY_ID[none_row], KNOWN, "none", K)["wrong_subclass"], None)
check("but fabrication is still measured on that row",
      isinstance(gr.attribution_for_row(BY_ID[none_row], KNOWN, "none", K)["fabricated"], list), True)

print("faithfulness: the question counts as context")
# q13 repeats '8 months' from the question. Against chunks alone that reads as unsupported;
# it is the model echoing a figure the user supplied.
q13 = gr.faithfulness_for_row(BY_ID["q13"], CHUNKS, K)
check("q13 resolves to chunks", q13["resolvable"], True)
check("q13 has no unsupported fact once the question counts", q13["n_unsupported"], 0)
q13_chunks_only = cf.facts_in(BY_ID["q13"]["rag_answer"]) - set().union(
    *[cf.facts_in(c["text"]) for c in CHUNKS if c["id"] in q13["retrieved_chunk_ids"]])
check("and it would have one if the question did not count",
      ("8", "month") in q13_chunks_only, True)

print("faithfulness fires when a figure really is absent")
invented = dict(BY_ID["q13"], rag_answer=BY_ID["q13"]["rag_answer"] + " The cap is 99 fortnights.")
check("an invented figure is unsupported",
      "99 fortnight" in gr.faithfulness_for_row(invented, CHUNKS, K)["unsupported"], True)

print("the one unsupported fact is q78, and it is recorded as reviewed")
check("q78 is the only one", sorted(
    qid for qid in BY_ID
    if gr.faithfulness_for_row(BY_ID[qid], CHUNKS, K).get("n_unsupported")), ["q78"])
check("and it carries a review note rather than being excluded",
      "q78" in gr.REVIEWED_UNSUPPORTED, True)
check("its unsupported fact is the derived 36 months",
      gr.faithfulness_for_row(BY_ID["q78"], CHUNKS, K)["unsupported"], ["36 month"])

print("counts sum to n")
per_q = gr.score_run(ROWS, CHUNKS, TESTSET, K)
agg = gr.aggregate(per_q, ROWS)
styles = agg["attribution"]["format_compliance"]["styles"]
check("citation styles sum to 100", sum(styles.values()), len(ROWS))
check("per-question detail covers every row", len(per_q), len(ROWS))
check("every row resolved to chunk ids",
      sum(1 for e in per_q.values() if e["faithfulness"]["resolvable"]), len(ROWS))
check("citing + not citing = n",
      sum(1 for e in per_q.values() if e["cites_anything"])
      + sum(1 for e in per_q.values() if not e["cites_anything"]), len(ROWS))

print("the reported figures match what the brief predicted")
check("24 instructed / 8 reversed / 68 freeform",
      (styles.get("bracketed_source"), styles.get("source_bracketed"), styles.get("freeform_or_absent")),
      (24, 8, 68))
check("49 of 100 name a corpus document",
      agg["attribution"]["citation_rate"]["answers_naming_a_corpus_document"], 49)
check("zero fabricated citations on the current run",
      agg["attribution"]["fabricated_citations"]["answers_with_a_fabricated_citation"], 0)
check("wrong-subclass denominator is 54, not 100",
      agg["attribution"]["wrong_subclass_citations"]["n"], 54)

print("the baseline does not invent citations, it simply never cites")
check("B0 names no corpus document anywhere",
      gr.b0_citation_contrast(ROWS, KNOWN)["b0_answers_naming_a_corpus_document"], 0)

print()
if failures:
    print(f"{len(failures)} FAILED")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("all controls passed")
