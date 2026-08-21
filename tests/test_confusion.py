"""Positive and negative controls for the five confusion counters.

Every counter reads zero on the current runs. That is only informative if the counters
can be shown to fire when the failure is genuinely present -- a detector that never
fires produces the same clean-looking table as a system that never errs. Each counter
therefore gets a synthetic answer built from the real corpus that must trip it, and the
real answer that must not.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import confusion  # noqa: E402

SNAPSHOT = json.loads((ROOT / "data" / "documents" / "MANIFEST.json").read_text())["snapshot_id"]
CHUNKS = confusion.load_store(SNAPSHOT, "structure_aware_meta")
TESTSET = {t["id"]: t for t in confusion.load_testset()}
RUN = json.loads((ROOT / "results" / "b0_vs_rag__structure_aware_meta_k3.json").read_text())
ROWS = {r["id"]: r for r in RUN["rows"]}

failures: list[str] = []


def check(name: str, got, want) -> None:
    status = "ok " if got == want else "FAIL"
    if got != want:
        failures.append(f"{name}: got {got}, want {want}")
    print(f"  [{status}] {name}")


def score(qid: str, answer: str | None = None, sources: str | None = None) -> dict:
    row = dict(ROWS[qid])
    if answer is not None:
        row["rag_answer"] = answer
    if sources is not None:
        row["rag_retrieved_sources"] = sources
    return confusion.score_row(row, TESTSET[qid], CHUNKS, 3)


print("(e) wrong stream within the right subclass -- q02 asks about Post-Higher Education Work")
# The Second Post-Higher Education Work stream says 1 to 2 years; the asked-about stream
# says 2 to 3. Both chunks are retrieved for q02, so this is the confusion that matters.
r = score("q02", "The maximum stay is between 1 and 2 years. Source: [subclass_485_temporary_graduate.txt]")
check("fires on the Second-stream figure", r["e_fact_from_wrong_stream"], True)
check("  and is not counted as cross-subclass", r["b_fact_from_wrong_subclass"], False)
check("silent on the real answer", score("q02")["e_fact_from_wrong_stream"], False)

print("(b) fact drawn only from a wrong-subclass chunk -- q01 asks about Subclass 500")
# 417 is retrieved at rank 3 for q01. Its 6-month figure appears nowhere in the 500 gold.
r = score("q01", "You may work for 6 months with one employer.")
check("fires on the 417 figure", r["b_fact_from_wrong_subclass"], True)
check("silent on the real answer", score("q01")["b_fact_from_wrong_subclass"], False)

print("(a) citation naming a wrong-subclass document")
r = score("q02", "Stay is 2 to 3 years. [Source: subclass_500_student.txt]",
          sources="subclass_485_temporary_graduate.txt (0.868); subclass_500_student.txt (0.5); subclass_417_working_holiday.txt (0.4)")
check("fires when a 500 doc is cited for a 485 question", r["a_cites_wrong_subclass"], True)
check("silent on the real answer", score("q02")["a_cites_wrong_subclass"], False)

print("(d) citation naming a document that was never retrieved")
r = score("q02", "Stay is 2 to 3 years. [Source: subclass_500_student.txt]")
check("fires when the cited doc is absent from context", r["d_cites_absent_source"], True)
check("silent on the real answer", score("q02")["d_cites_absent_source"], False)

print("(c) rank-1 chunk from another subclass")
r = score("q02", sources="subclass_417_working_holiday.txt (0.9); subclass_485_temporary_graduate.txt (0.8); subclass_485_temporary_graduate.txt (0.7)")
check("fires when rank 1 is a 417 chunk", r["c_wrong_subclass_at_1"], True)
check("silent on the real answer", score("q02")["c_wrong_subclass_at_1"], False)

print("citation styles are recognised in the forms the model actually emits")
check("bracketed", confusion.citation_style("x [Source: a.txt]"), "bracketed_source")
check("reversed", confusion.citation_style("x Source: [a.txt]"), "source_bracketed")
check("freeform", confusion.citation_style("as stated in the source document a.txt"), "freeform_or_absent")
check("freeform still yields the citation",
      confusion.cited_sources("refer to the source document whm_6_month_work_limitation",
                              {"whm_6_month_work_limitation.txt"}),
      ["whm_6_month_work_limitation.txt"])

print()
if failures:
    print(f"{len(failures)} FAILED")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("all controls passed")
