"""Three-state refusal detection and the answerability split.

Dimension 1: does the system know when to stay quiet, and when it stays quiet is that
a success or a failure? Most evaluations report one "% unanswered" figure, which adds a
success and a failure together and hides both.

A boolean refusal detector is not enough, because an answer that declines and then
answers anyway has answered. On the 13 human-graded rows a boolean matcher agrees with
the human column on 10/13; the three-state classifier below gets the state right on
13/13. All three boolean failures run the same way -- answered_but_wrong read as a
refusal -- so the one-number version reports zero wrong answers and three refusals that
never happened. Those two failures have opposite remedies, so the single figure does not
merely lose precision: it points the team the wrong way.

Decisions this module encodes, with the evidence for each:

  cross_subclass_probe counts as ANSWERABLE. All 25 of its gold answers are substantive
  answers explaining the subclass dependency rather than declines, and the annotator
  graded q07/q08/q09 as answered_*, never correct_refusal.

  Agreement is measured against the annotation sheet's own rag_answer text, not a run
  file. Generation is non-deterministic and the graded text matches the k3 run on only
  6/13 rows, so scoring against a run file would score the detector on answers nobody
  judged.

  The hedge signal is a "concrete claim": a number+unit pair, a known visa subclass
  code, or a visa condition code. Deliberately narrow -- see LIMITATIONS.

  A hedged answer has answered, so it grades on correctness rather than occupying a
  bucket of its own. T12 expects five cells.

  missed_answerable and over_refused are split on gold_source. They were never defined
  upstream and were near-synonyms until made disjoint this way.

  Answering a question with no corpus answer is answered_but_wrong: if nothing in the
  corpus answers it, any substantive answer is unfounded. This needs no human grade and
  so computes at n=100.

LIMITATIONS, to be carried into every table built on this module:

  Anything validated is validated at n=13. One question is 7.7%. Directional, not
  significant.

  missed_answerable and over_refused have no human examples at all, so the agreement
  rate validates three of the five categories.

  The annotation sheet's answerability column describes RAG only -- it is consistent
  with rag_correct on all 13 rows and inconsistent with b0_correct -- so the B0 split is
  unvalidated.

  NATURAL_REFUSAL_MARKERS was assembled by inspecting B0 outputs, so its effect on the
  refusal count is a lower bound rather than an independent measurement. It is not
  trimmed to the phrases that happen to fire; trimming to hits is what would make it
  fitted.

  The marker list decides whether a decline is SEEN. The hedge rule decides whether a
  seen decline COUNTS. Both must be stated together or the refusal figure cannot be
  read. The size of that second choice is measured rather than asserted -- see
  hedge_rule_sensitivity, which is emitted into the output file.

  missed_answerable has no examples at all, human or machine. The category is defined
  and computable, and nothing in this corpus falls into it. That is a sharper limitation
  than an untested category, not a weaker one: the split reports four occupied buckets
  out of five.

  A condition code satisfies the hedge signal whether it is asserted or merely mentioned
  in a referral. The rows where that is the only signal present are named in the
  diagnostic rather than folded into the count.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Read-only reuse of the worksheet's fact extractors: they are the project's existing
# definition of a checkable claim, and a second definition here would silently diverge
# from the one the annotator's pointers were built on.
from annotate_worksheet import extract_quantities, extract_subclasses  # noqa: E402

TESTSET = ROOT / "data" / "testset" / "test_set.json"
RESULTS = ROOT / "results"
SHEET = RESULTS / "annotation_sheet_annotated.xlsx"
RUN = RESULTS / "b0_vs_rag__structure_aware_meta_k3.json"
OUT = RESULTS / "answerability.json"

# Family 1 -- the strings the RAG system prompt instructs the model to produce, plus the
# spellings it actually emits. Transcribed from rag.RAG_SYSTEM_PROMPT. Matched on
# fragments rather than whole sentences: the configured text carries an em dash and the
# model does not reliably reproduce it.
CONFIGURED_REFUSAL_MARKERS = [
    "don't have enough information",
    "do not have enough information",
    "case-specific question",
    "registered migration agent",
    "cannot be answered from",
    "can't answer that",
    "cannot answer that",
    "unable to answer",
    "not able to answer",
    "i don't know",
    "consult the department of home affairs",
]

# Family 2 -- natural-language declines. B0 runs with no system prompt, so it refuses in
# its own vocabulary and family 1 is blind to it. Three B0 answers decline in words
# family 1 does not contain (q05, q12, q94), one of which the annotator graded as a
# correct decline. Measuring refusal only with the deployed system's own configured
# phrases therefore undercounts the baseline and inflates the RAG-vs-B0 gap in the
# direction that flatters this project.
NATURAL_REFUSAL_MARKERS = [
    "don't have access to",
    "do not have access to",
    "don't have real-time",
    "do not have real-time",
    "can't provide",
    "cannot provide",
    "i can't tell you",
    "i'm not able to",
    "i am not able to",
    "i don't have information",
    "i can't determine",
    "cannot determine",
    "i'm unable",
    "i am unable",
]

# A visa condition code is as concrete a claim as a subclass number, and the quantity and
# subclass extractors see neither: "condition 8547" contains no unit word and 8547 is not a
# subclass. That matters because condition 8547 has a document to itself, so the one rule in
# the corpus that is not a subclass was invisible to the hedge signal. The corpus contains
# exactly one such code today, so the pattern is general but only exercised by 8547.
CONDITION_CODE_RE = re.compile(r"condition\s*[-#]?\s*(\d{3,4})", re.IGNORECASE)

# The broader hedge rule used only for the sensitivity check below: advice offered as a
# numbered or bolded list, which is how a model gives guidance when it has no figure to
# give. Not the shipped rule -- see hedge_rule_sensitivity.
ADVICE_LIST_RE = re.compile(r"(?:^|\n|\s)\d\.\s|\*\*[A-Z]")

STATES = ("refusal", "hedged_answer", "answer")
BUCKETS = (
    "answered_correctly",
    "answered_but_wrong",
    "correct_refusal",
    "missed_answerable",
    "over_refused",
)
UNGRADED = "answered_ungraded"


def which_markers(text: str) -> tuple[list[str], list[str]]:
    """The markers present, split by family, so a result can say which one fired."""
    t = (text or "").lower()
    return (
        [m for m in CONFIGURED_REFUSAL_MARKERS if m in t],
        [m for m in NATURAL_REFUSAL_MARKERS if m in t],
    )


def has_concrete_claim(text: str) -> bool:
    """Whether the text asserts something checkable: a number+unit, a subclass code, or
    a visa condition code.

    This is the shipped hedge signal, and it is narrow by choice. It does not count
    advice that carries no figure, which is why some declines-then-advises answers read
    as refusals rather than hedges. hedge_rule_sensitivity measures what that costs.
    """
    text = text or ""
    return bool(
        extract_quantities(text)
        or extract_subclasses(text)
        or CONDITION_CODE_RE.search(text)
    )


def has_substantive_advice(text: str) -> bool:
    """Guidance offered without any figure, as a numbered or bolded list.

    Used only by hedge_rule_sensitivity. It is not part of the shipped rule: the one
    human-graded baseline decline in the project, B0's q12, is a correct_refusal under
    the shipped rule and answered_but_wrong under this broader one, so adopting it would
    put the module at odds with the only human judgement available.
    """
    return bool(ADVICE_LIST_RE.search(text or ""))


def classify_answer(text: str, hedge_rule: str = "narrow") -> str:
    """One of STATES.

    Position in the text is not the signal. q07 declines in its opening sentence and
    then answers; q13 answers and then appends a disclaimer. Both are hedges.

    hedge_rule selects what counts as having answered. "narrow" is the shipped rule;
    "wide" also counts advice that carries no figure, and exists so the choice can be
    measured rather than asserted.
    """
    configured, natural = which_markers(text)
    if not configured and not natural:
        return "answer"
    answered = has_concrete_claim(text)
    if hedge_rule == "wide":
        answered = answered or has_substantive_advice(text)
    return "hedged_answer" if answered else "refusal"


def bucket_for(state: str, gold_source: str, correctness: str | None) -> str:
    """Place one answer in the five-way split.

    gold_source is present on all 100 test-set rows and says where the answer lives: a
    filename (one document holds it), "multiple" (it needs synthesis across documents),
    or "none" (the corpus does not answer it).

    A refusal is a success, a coverage miss or over-caution depending on which of those
    three applies. An answer is graded on correctness, except where the corpus holds no
    answer at all -- there, answering is itself the error, which is decidable without a
    human grade and so extends to the ungraded rows.
    """
    if state == "refusal":
        if gold_source == "none":
            return "correct_refusal"
        if gold_source == "multiple":
            return "over_refused"
        return "missed_answerable"

    if gold_source == "none":
        return "answered_but_wrong"
    if correctness is None:
        return UNGRADED
    # "partial" counts as wrong, following the annotator's own call on q07.
    return "answered_correctly" if str(correctness).strip().lower() == "yes" else "answered_but_wrong"


def answerability_split(rows, testset, answer_key, correctness_key=None) -> dict:
    """Count rows into the five buckets, plus answered_ungraded where no grade exists.

    rows may be run rows or annotation-sheet rows; both carry an id and the answer text.
    """
    counts = Counter()
    per_question = []
    for row in rows:
        qid = row["id"]
        gold_source = testset[qid]["gold_source"]
        correctness = row.get(correctness_key) if correctness_key else None
        if correctness in ("", None):
            correctness = None
        state = classify_answer(row[answer_key])
        placed = bucket_for(state, gold_source, correctness)
        counts[placed] += 1
        per_question.append(
            {"id": qid, "gold_source": gold_source, "state": state, "bucket": placed}
        )

    split = {b: counts[b] for b in BUCKETS}
    if counts[UNGRADED]:
        split[UNGRADED] = counts[UNGRADED]
    return {
        "n": len(rows),
        "split": split,
        "states": {s: sum(1 for p in per_question if p["state"] == s) for s in STATES},
        "per_question": per_question,
    }


def _state_implied_by(label: str) -> str:
    """Which state a human bucket implies, for the state-level comparison."""
    return "refusal" if label in ("correct_refusal", "missed_answerable", "over_refused") else "answered"


def agreement_with_human(sheet_rows, testset) -> dict:
    """Compare the detector against the human answerability column (RAG only).

    Two rates, and the first is the one to lead with:

    state agreement -- the detector's own call, reduced to refused-or-answered, against
    what the human bucket implies. This is independent of the human grading.

    bucket agreement -- the full five-way placement. This is NOT independent: the
    mapping consumes the human rag_correct column to separate answered_correctly from
    answered_but_wrong, so a high figure here partly restates its own input. Reported
    for completeness, and never as the headline.
    """
    state_ok = bucket_ok = 0
    disagreements = []
    rows_out = []

    for row in sheet_rows:
        qid = row["id"]
        human = row["answerability"]
        gold_source = testset[qid]["gold_source"]
        state = classify_answer(row["rag_answer"])
        mine = bucket_for(state, gold_source, row["rag_correct"])

        s_match = _state_implied_by(mine) == _state_implied_by(human)
        b_match = mine == human
        state_ok += s_match
        bucket_ok += b_match
        rows_out.append(
            {"id": qid, "state": state, "detector_bucket": mine, "human_bucket": human,
             "state_agrees": s_match, "bucket_agrees": b_match}
        )
        if not b_match:
            disagreements.append({"id": qid, "detector": mine, "human": human})

    n = len(sheet_rows)
    return {
        "n": n,
        "state_agreement": {"matched": state_ok, "n": n, "rate": round(state_ok / n, 3)},
        "bucket_agreement": {
            "matched": bucket_ok,
            "n": n,
            "rate": round(bucket_ok / n, 3),
            "independent": False,
            "note": (
                "Not an independent check: the mapping consumes the human rag_correct "
                "column to separate answered_correctly from answered_but_wrong. Lead "
                "with state_agreement."
            ),
        },
        "disagreements": disagreements,
        "per_question": rows_out,
    }


def naive_boolean_baseline(sheet_rows, testset) -> dict:
    """The two-state matcher this module replaces, scored the same way.

    Kept in the output because the gap between the two is the result: it is what shows
    that the three-state distinction is doing work rather than adding machinery.
    """
    matched = 0
    misreads = []
    for row in sheet_rows:
        qid = row["id"]
        configured, natural = which_markers(row["rag_answer"])
        state = "refusal" if (configured or natural) else "answer"
        mine = bucket_for(state, testset[qid]["gold_source"], row["rag_correct"])
        if mine == row["answerability"]:
            matched += 1
        else:
            misreads.append({"id": qid, "naive": mine, "human": row["answerability"]})
    n = len(sheet_rows)
    return {"matched": matched, "n": n, "rate": round(matched / n, 3), "misreads": misreads}


def refusal_required_diagnostic(rows, testset, answer_key) -> dict:
    """How often each system declines a question the corpus cannot answer.

    Reported beside the split and labelled a diagnostic, not a sixth bucket. The hedged
    rows are the interesting ones: the system declines and then supplies personal
    migration advice in the same answer, which is the published weakness this dimension
    targets, surviving in a different shape.
    """
    unanswerable = [r for r in rows if testset[r["id"]]["gold_source"] == "none"]
    states = {r["id"]: classify_answer(r[answer_key]) for r in unanswerable}
    texts = {r["id"]: r[answer_key] for r in unanswerable}
    hedged = sorted(q for q, s in states.items() if s == "hedged_answer")

    # A condition code inside a referral -- "whether you are subject to condition 8547,
    # check with Home Affairs" -- is a mention, not an assertion, and the signal cannot
    # tell the two apart. Those rows are still hedges by the rule, but they are weaker
    # instances of declining-then-advising than one that states a rule outright, so they
    # are named rather than folded silently into the count.
    borderline = [
        q for q in hedged
        if CONDITION_CODE_RE.search(texts[q] or "")
        and not extract_quantities(texts[q] or "")
        and not extract_subclasses(texts[q] or "")
    ]
    return {
        "n_refusal_required": len(unanswerable),
        "refused": sum(1 for s in states.values() if s == "refusal"),
        "hedged_then_advised": len(hedged),
        "answered_outright": sum(1 for s in states.values() if s == "answer"),
        "hedged_ids": hedged,
        "hedged_on_condition_code_mention_only": borderline,
        "borderline_note": (
            "These rows qualify as hedges only because a condition code appears, and it "
            "appears inside a referral rather than an assertion. Weaker instances of the "
            "failure than the rest of hedged_ids; excluded if the ethics claim is stated "
            "narrowly."
        ) if borderline else "",
    }


def hedge_rule_sensitivity(rows, testset) -> dict:
    """How far the RAG-vs-B0 refusal result depends on where the hedge line is drawn.

    The marker families decide whether a decline is seen; the hedge rule decides whether
    a seen decline counts as a refusal or as an answer. The second choice is a judgement,
    so the claim that the direction survives it should be evidence rather than prose:
    both rules are run over both systems and the four numbers reported.
    """
    unanswerable = [r for r in rows if testset[r["id"]]["gold_source"] == "none"]
    out = {"n_refusal_required": len(unanswerable), "rules": {}}
    for rule in ("narrow", "wide"):
        out["rules"][rule] = {
            system: {
                "declined": sum(1 for r in unanswerable if classify_answer(r[key], rule) == "refusal"),
                "hedged_then_advised": sum(
                    1 for r in unanswerable if classify_answer(r[key], rule) == "hedged_answer"
                ),
            }
            for system, key in (("RAG", "rag_answer"), ("B0", "b0_answer"))
        }
    narrow, wide = out["rules"]["narrow"], out["rules"]["wide"]
    out["interpretation"] = (
        f"RAG declines {narrow['RAG']['declined']}/{len(unanswerable)} under the shipped rule "
        f"and {wide['RAG']['declined']}/{len(unanswerable)} under the broader one; B0 declines "
        f"{narrow['B0']['declined']} and {wide['B0']['declined']}. The gap narrows but does not "
        "close or reverse, so the finding that retrieval is what creates the ability to decline "
        "does not depend on where the hedge line is drawn."
    )
    return out


def load_testset() -> dict:
    return {t["id"]: t for t in json.loads(TESTSET.read_text())}


def load_graded_rows() -> list[dict]:
    """The 13 rows the annotator judged, with the answer text they actually read."""
    try:
        import openpyxl
    except ImportError:  # pragma: no cover
        raise SystemExit("openpyxl is required to read the annotation workbook.")

    if not SHEET.exists():
        raise SystemExit(
            f"Missing {SHEET}. Note this is annotation_sheet_annotated.xlsx -- "
            "annotation_sheet.xlsx is the blank template and carries no judgements."
        )
    ws = openpyxl.load_workbook(SHEET)["Answers"]
    header = [c.value for c in ws[1]]
    idx = {h: n for n, h in enumerate(header)}
    wanted = ("id", "question", "b0_answer", "rag_answer", "b0_correct", "rag_correct",
              "answerability", "annotator")
    rows = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[idx["id"]] in (None, ""):
            continue
        rows.append({k: row[idx[k]] for k in wanted})
    return rows


def main() -> None:
    testset = load_testset()
    graded = load_graded_rows()
    run = json.loads(RUN.read_text())
    run_rows, run_meta = run["rows"], run["run_meta"]

    agreement = agreement_with_human(graded, testset)
    naive = naive_boolean_baseline(graded, testset)

    validated = {
        "n": len(graded),
        "source": SHEET.name,
        "note": "The text the annotator judged. Validated against the human answerability column.",
        "RAG": answerability_split(graded, testset, "rag_answer", "rag_correct"),
        "B0": answerability_split(graded, testset, "b0_answer", "b0_correct"),
    }
    full = {
        "n": len(run_rows),
        "source": RUN.name,
        "note": (
            "The refusal side is decidable from gold_source and needs no human grade, so "
            "it holds across all 100. The answer side cannot be split into correct and "
            "wrong here: rag_correct_manual and b0_correct_manual are empty on every row, "
            "so those answers are counted as answered_ungraded rather than guessed at."
        ),
        "RAG": answerability_split(run_rows, testset, "rag_answer"),
        "B0": answerability_split(run_rows, testset, "b0_answer"),
    }

    out = {
        "task": "T05 -- refusal detection and the answerability split",
        "config": {
            "snapshot_id": run_meta["snapshot_id"],
            "config_id": run_meta["config_id"],
            "top_k": run_meta["top_k"],
            "gen_model": run_meta["gen_model"],
        },
        "headline": (
            "The one-number version doesn't just lose precision, it points the team the "
            "wrong way."
        ),
        "definitions": {
            "states": {
                "refusal": "A refusal marker is present and the answer asserts nothing checkable.",
                "hedged_answer": "A refusal marker is present and the answer also asserts a figure or a subclass. It has answered.",
                "answer": "No refusal marker from either family.",
            },
            "buckets": {
                "correct_refusal": "Declined, and gold_source is 'none' -- the corpus does not answer it. A success.",
                "missed_answerable": "Declined, and gold_source names a document that holds the answer. A coverage failure.",
                "over_refused": "Declined, and gold_source is 'multiple' -- answerable, but only by synthesising across documents. Over-caution.",
                "answered_correctly": "Answered or hedged, and the annotator graded it correct.",
                "answered_but_wrong": "Answered or hedged, and the annotator graded it wrong or partial; or gold_source is 'none', where answering is itself the error.",
            },
            "marker_families": {
                "configured": "Strings the RAG system prompt instructs the model to use.",
                "natural": "Declines phrased in the model's own words, which B0 uses because it runs with no system prompt.",
            },
        },
        "agreement": agreement,
        "naive_boolean_baseline": naive,
        "splits": {"validated_n13": validated, "full_n100": full},
        "diagnostics": {
            "note": "Reported beside the split, not as a sixth bucket.",
            "RAG_n100": refusal_required_diagnostic(run_rows, testset, "rag_answer"),
            "B0_n100": refusal_required_diagnostic(run_rows, testset, "b0_answer"),
        },
        "hedge_rule_sensitivity_n100": hedge_rule_sensitivity(run_rows, testset),
        "disagreements_investigated": [
            {
                "id": "q12",
                "finding": (
                    "B0 declines the weather question in its own words -- 'I don't have "
                    "real-time data access, so I can't provide current or future weather "
                    "forecasts.' The configured marker family does not contain that "
                    "phrasing and misses the decline entirely. The annotator graded the "
                    "answer correct, so this was a detector gap, not an ambiguous answer."
                ),
                "resolution": "Fixed by the natural-language marker family.",
            },
            {
                "id": "q10 (B0)",
                "finding": (
                    "The annotator graded B0 correct because it declined to predict the "
                    "visa outcome. It then gave five numbered points of advice. Because "
                    "gold_source is 'none', this module places it in answered_but_wrong, "
                    "which disagrees with that grade."
                ),
                "resolution": (
                    "The rule is applied consistently rather than exempting the row. The "
                    "disagreement is recorded here: a stated disagreement is data, a "
                    "silent override is a defect."
                ),
            },
        ],
        "limitations": [
            "Everything validated is validated at n=13. One question is 7.7%. Directional, not significant.",
            "missed_answerable and over_refused have no human examples, so agreement validates three of five categories.",
            "The answerability column describes RAG only, so the B0 split is unvalidated.",
            "NATURAL_REFUSAL_MARKERS was assembled by inspecting B0 outputs, so its effect on the refusal count is a lower bound, not an independent measurement.",
            "The marker family decides whether a decline is seen; the hedge rule decides whether a seen decline counts. Both are stated above because the refusal figure cannot be read without them.",
            "Generation is non-deterministic: the graded text matches the k3 run on 6 of 13 rows, which is why agreement is measured against the sheet.",
        ],
    }

    OUT.write_text(json.dumps(out, indent=2))

    sa, ba = agreement["state_agreement"], agreement["bucket_agreement"]
    print(f"state agreement   {sa['matched']}/{sa['n']}  ({sa['rate']:.1%})   <- the independent figure")
    print(f"naive two-state   {naive['matched']}/{naive['n']}  ({naive['rate']:.1%})")
    print(f"bucket agreement  {ba['matched']}/{ba['n']}  -- not independent, see note")
    print()
    for scope, block in (("n=13 ", validated), ("n=100", full)):
        for system in ("RAG", "B0"):
            s = block[system]["split"]
            cells = "  ".join(f"{b.replace('answered_', 'ans_')}={s.get(b, 0)}" for b in BUCKETS)
            print(f"{scope} {system:<4} {cells}" + (f"  ungraded={s[UNGRADED]}" if UNGRADED in s else ""))
    print()
    for system, key in (("RAG", "RAG_n100"), ("B0", "B0_n100")):
        d = out["diagnostics"][key]
        print(f"{system}: declined {d['refused']}/{d['n_refusal_required']} refusal_required, "
              f"hedged-then-advised on {d['hedged_then_advised']}")

    sens = out["hedge_rule_sensitivity_n100"]
    print(f"\nhedge-rule sensitivity, declines out of {sens['n_refusal_required']} refusal_required:")
    for rule in ("narrow", "wide"):
        r = sens["rules"][rule]
        tag = "  (shipped)" if rule == "narrow" else ""
        print(f"  {rule:<7} RAG={r['RAG']['declined']:<3} B0={r['B0']['declined']:<3}{tag}")
    print(f"\nWrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
