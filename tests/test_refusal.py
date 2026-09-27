"""Controls for the three-state refusal detector and the bucket mapping.

The detector's whole claim is that it separates a decline from a decline-that-answers.
Hand-written strings establish that the three states are reachable at all; the real
fixtures establish that it holds on the answers this project actually produced, which is
where a boolean matcher fails. Every fixture is read from the file it lives in and named
here, because q13's text differs across the k1, k3 and k5 runs and the annotation sheet
-- an earlier note described it from one run and got the fixture wrong.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import refusal  # noqa: E402

GRADED = {r["id"]: r for r in refusal.load_graded_rows()}
RUN = json.loads((ROOT / "results" / "b0_vs_rag__structure_aware_meta_k3.json").read_text())
RUN_ROWS = {r["id"]: r for r in RUN["rows"]}

failures: list[str] = []


def check(name: str, got, want) -> None:
    status = "ok " if got == want else "FAIL"
    print(f"  {status} {name}")
    if got != want:
        failures.append(f"{name}: got {got!r}, want {want!r}")


CONFIGURED_NOT_ENOUGH = "I don't have enough information in my sources to answer that."
CONFIGURED_CASE_SPECIFIC = (
    "That's a case-specific question — please contact a registered migration agent or "
    "the Department of Home Affairs directly."
)

print("three states are reachable on hand-written strings")
check("a configured string alone is a refusal",
      refusal.classify_answer(CONFIGURED_NOT_ENOUGH), "refusal")
check("the other configured string alone is a refusal",
      refusal.classify_answer(CONFIGURED_CASE_SPECIFIC), "refusal")
check("a substantive answer with no decline is an answer",
      refusal.classify_answer(
          "Subclass 500 student visa holders may work 48 hours per fortnight while the "
          "course is in session."),
      "answer")
check("declining and then stating a figure is a hedge",
      refusal.classify_answer(
          CONFIGURED_NOT_ENOUGH + " That said, the cap is 48 hours per fortnight."),
      "hedged_answer")
check("declining and then naming a subclass is a hedge",
      refusal.classify_answer(
          CONFIGURED_CASE_SPECIFIC + " Subclass 417 carries condition 8547."),
      "hedged_answer")

print("the hedge signal, in isolation")
check("a figure is a concrete claim", refusal.has_concrete_claim("up to 6 months"), True)
check("a subclass code is a concrete claim", refusal.has_concrete_claim("the subclass 485 stream"), True)
check("apology and referral alone is not", refusal.has_concrete_claim("I'm sorry, I can't help with that."), False)

print("real hedges from the graded sheet -- a boolean matcher calls all three refusals")
check("q04 answers and then appends a migration-agent disclaimer",
      refusal.classify_answer(GRADED["q04"]["rag_answer"]), "hedged_answer")
check("q07 opens with the configured string and then answers about 417",
      refusal.classify_answer(GRADED["q07"]["rag_answer"]), "hedged_answer")
check("q13 answers and then appends the configured string",
      refusal.classify_answer(GRADED["q13"]["rag_answer"]), "hedged_answer")

print("the same questions in the k3 run, where the wording differs -- fixtures are per-run")
check("q07 hedges in k3 too", refusal.classify_answer(RUN_ROWS["q07"]["rag_answer"]), "hedged_answer")
check("q13 hedges in k3 too", refusal.classify_answer(RUN_ROWS["q13"]["rag_answer"]), "hedged_answer")
# q04 is a hedge in the graded sheet and a plain answer in k3: that run simply does not
# emit the disclaimer. Asserted rather than glossed, because it is the clearest evidence
# in the suite that a fixture without its run file named is not a fixture.
check("q04 carries no disclaimer in k3 and so is a plain answer",
      refusal.classify_answer(RUN_ROWS["q04"]["rag_answer"]), "answer")
check("the disclaimer is present in the graded text and absent in k3",
      (bool(refusal.which_markers(GRADED["q04"]["rag_answer"])[0]),
       bool(refusal.which_markers(RUN_ROWS["q04"]["rag_answer"])[0])),
      (True, False))

print("real refusals from the graded sheet")
for qid in ("q10", "q11", "q12"):
    check(f"{qid} is a clean refusal", refusal.classify_answer(GRADED[qid]["rag_answer"]), "refusal")

print("the baseline refuses in vocabulary the configured family does not contain")
b0_q12 = GRADED["q12"]["b0_answer"]
configured, natural = refusal.which_markers(b0_q12)
check("no configured marker fires on B0's decline", configured, [])
check("a natural-language marker does fire", bool(natural), True)
check("so B0's q12 is seen as a refusal", refusal.classify_answer(b0_q12), "refusal")

print("bucket mapping")
check("declined, corpus cannot answer -> success",
      refusal.bucket_for("refusal", "none", None), "correct_refusal")
check("declined, one document holds the answer -> coverage miss",
      refusal.bucket_for("refusal", "whm_6_month_work_limitation.txt", None), "missed_answerable")
check("declined, answerable only across documents -> over-caution",
      refusal.bucket_for("refusal", "multiple", None), "over_refused")
check("answered a question the corpus cannot answer -> wrong",
      refusal.bucket_for("answer", "none", None), "answered_but_wrong")
check("hedged a question the corpus cannot answer -> wrong",
      refusal.bucket_for("hedged_answer", "none", None), "answered_but_wrong")
check("graded correct -> correct",
      refusal.bucket_for("answer", "subclass_500_student.txt", "yes"), "answered_correctly")
check("graded partial counts as wrong, as the annotator did on q07",
      refusal.bucket_for("hedged_answer", "multiple", "partial"), "answered_but_wrong")
check("ungraded stays ungraded rather than being guessed",
      refusal.bucket_for("answer", "subclass_500_student.txt", None), "answered_ungraded")

print("the detector beats the boolean matcher it replaces")
testset = refusal.load_testset()
graded_rows = refusal.load_graded_rows()
agreement = refusal.agreement_with_human(graded_rows, testset)
naive = refusal.naive_boolean_baseline(graded_rows, testset)
check("state agreement is 13/13", agreement["state_agreement"]["matched"], 13)
check("the boolean matcher manages 10/13", naive["matched"], 10)
check("and all three of its misreads are hedges read as refusals",
      sorted(m["id"] for m in naive["misreads"]), ["q04", "q07", "q13"])
check("each one turns a wrong answer into a phantom refusal",
      sorted({m["human"] for m in naive["misreads"]}), ["answered_but_wrong"])

print("condition codes count as a concrete claim")
check("a condition code is a claim", refusal.has_concrete_claim("subject to condition 8547"), True)
check("the quantity extractor never saw it", bool(refusal.extract_quantities("condition 8547")), False)
check("nor did the subclass extractor", bool(refusal.extract_subclasses("condition 8547")), False)
check("declining and then naming a condition is a hedge",
      refusal.classify_answer(CONFIGURED_NOT_ENOUGH + " Condition 8547 may apply."), "hedged_answer")
# B0's q05 denies condition 8547 exists while a whole corpus document describes it. Before
# condition codes were part of the signal this read as a refusal in the k3 run and an answer
# in the graded sheet -- the project's only missed_answerable, and an artifact of which text
# was scored rather than a property of the system.
check("B0 q05 in k3 is a hedge, not a refusal",
      refusal.classify_answer(RUN_ROWS["q05"]["b0_answer"]), "hedged_answer")
# The two texts still differ -- the sheet's version carries no decline language at all,
# so it is a plain answer rather than a hedge. What the condition-code signal removes is
# the consequential divergence: neither text is a refusal any more, so the bucket no
# longer depends on which one was scored.
check("B0 q05 in the graded sheet carries no decline language",
      refusal.which_markers(GRADED["q05"]["b0_answer"]), ([], []))
check("so it is a plain answer there",
      refusal.classify_answer(GRADED["q05"]["b0_answer"]), "answer")
check("but neither text is a refusal, so the bucket is stable across both",
      {refusal.bucket_for(refusal.classify_answer(t), "whm_6_month_work_limitation.txt", "no")
       for t in (RUN_ROWS["q05"]["b0_answer"], GRADED["q05"]["b0_answer"])},
      {"answered_but_wrong"})

print("the hedge rule is measured, not asserted")
sens = refusal.hedge_rule_sensitivity(list(RUN_ROWS.values()), refusal.load_testset())
check("both rules are reported", sorted(sens["rules"]), ["narrow", "wide"])
check("RAG is unmoved by the choice",
      sens["rules"]["narrow"]["RAG"]["declined"] == sens["rules"]["wide"]["RAG"]["declined"], True)
check("B0 declines fewer under the broader rule, never more",
      sens["rules"]["wide"]["B0"]["declined"] <= sens["rules"]["narrow"]["B0"]["declined"], True)
check("and the gap does not reverse under either rule",
      all(sens["rules"][r]["RAG"]["declined"] > sens["rules"][r]["B0"]["declined"]
          for r in ("narrow", "wide")), True)

print("borderline hedges are named rather than folded into the count")
diag = refusal.refusal_required_diagnostic(list(RUN_ROWS.values()), refusal.load_testset(), "rag_answer")
check("q98 is flagged as a condition-code mention only",
      diag["hedged_on_condition_code_mention_only"], ["q98"])
check("and it is inside the hedged set it qualifies",
      "q98" in diag["hedged_ids"], True)

print()
if failures:
    print(f"{len(failures)} FAILED")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("all controls passed")
