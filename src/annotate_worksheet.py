"""Build a hand-annotation worksheet from the B0-vs-RAG comparison.

This script does the MECHANICAL work only: it extracts checkable facts from
the gold answer and each system answer, and flags where they diverge. It
deliberately does NOT decide whether an answer is correct.

Correctness is a judgement call that must be made and owned by a team member:
it is the ground truth every downstream metric is scored against, and under the
course's Condition 3 (Bounded Process AI) policy the analysis must be the
student's own. Every column below is a labelled heuristic pointer, not a grade.
"""

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
TEST_SET_PATH = ROOT / "data" / "testset" / "test_set.json"

KNOWN_SUBCLASSES = ["417", "462", "482", "485", "500"]

# Phrases the RAG system prompt instructs the model to use, plus common variants.
REFUSAL_MARKERS = [
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

UNIT_WORDS = r"(hours?|fortnights?|months?|years?|days?|weeks?|aud|dollars?|%|percent)"


def extract_quantities(text: str) -> set[str]:
    """Pull number+unit pairs, e.g. '48 hours', '6 months', '3 years'.

    Heuristic. Used to point an annotator at figures that differ; it is not a
    correctness test.
    """
    if not text:
        return set()
    t = text.lower().replace(",", "")
    found = set()
    for m in re.finditer(rf"(\d+(?:\.\d+)?)\s*-?\s*{UNIT_WORDS}", t):
        num, unit = m.group(1), m.group(2).rstrip("s")
        unit = {"dollar": "aud", "percent": "%"}.get(unit, unit)
        found.add(f"{num} {unit}")
    for m in re.finditer(r"\$\s*(\d+(?:\.\d+)?)", t):
        found.add(f"{m.group(1)} aud")
    return found


def extract_subclasses(text: str) -> set[str]:
    """Which visa subclasses an answer refers to.

    Heuristic. Matches 'subclass NNN' and bare three-digit codes that are not
    immediately preceded by a currency symbol.
    """
    if not text:
        return set()
    t = text.lower()
    found = set()
    for m in re.finditer(r"subclass\s*[-#]?\s*(\d{3})", t):
        if m.group(1) in KNOWN_SUBCLASSES:
            found.add(m.group(1))
    for code in KNOWN_SUBCLASSES:
        if re.search(rf"(?<![$\d]){code}\b", t):
            found.add(code)
    return found


def is_refusal(text: str) -> bool:
    """Heuristic refusal detection against the configured strings and variants."""
    if not text:
        return False
    t = text.lower()
    return any(marker in t for marker in REFUSAL_MARKERS)


def main() -> None:
    rows_path = RESULTS_DIR / "b0_vs_rag_comparison.json"
    if not rows_path.exists():
        raise SystemExit(f"Missing {rows_path}. Run eval.py first.")

    rows = json.loads(rows_path.read_text())
    test_set = {t["id"]: t for t in json.loads(TEST_SET_PATH.read_text())}

    out_rows = []
    for r in rows:
        item = test_set.get(r["id"], {})
        gold_subclass = str(item.get("subclass", "")).strip()
        gold_q = extract_quantities(r["gold_answer"])
        b0_q = extract_quantities(r["b0_answer"])
        rag_q = extract_quantities(r["rag_answer"])
        b0_sc = extract_subclasses(r["b0_answer"])
        rag_sc = extract_subclasses(r["rag_answer"])

        expects_refusal = r["category"] == "refusal_required"
        retrieved = r.get("rag_retrieved_sources", "")
        gold_src = r.get("gold_source", "")
        gold_src_retrieved = (
            "n/a" if gold_src in ("none", "multiple", "") else str(gold_src in retrieved)
        )

        out_rows.append(
            {
                "id": r["id"],
                "category": r["category"],
                "gold_subclass": gold_subclass,
                "question": r["question"],
                "gold_answer": r["gold_answer"],
                "b0_answer": r["b0_answer"],
                "rag_answer": r["rag_answer"],
                # --- mechanical pointers (heuristics, NOT grades) ---
                "check_gold_figures": "; ".join(sorted(gold_q)) or "-",
                "check_b0_figures": "; ".join(sorted(b0_q)) or "-",
                "check_rag_figures": "; ".join(sorted(rag_q)) or "-",
                "check_b0_figures_missing": "; ".join(sorted(gold_q - b0_q)) or "-",
                "check_rag_figures_missing": "; ".join(sorted(gold_q - rag_q)) or "-",
                "check_b0_subclasses": "; ".join(sorted(b0_sc)) or "-",
                "check_rag_subclasses": "; ".join(sorted(rag_sc)) or "-",
                "check_rag_foreign_subclass": (
                    "; ".join(sorted(rag_sc - {gold_subclass}))
                    if gold_subclass in KNOWN_SUBCLASSES
                    else "n/a"
                )
                or "-",
                "check_b0_refusal": str(is_refusal(r["b0_answer"])),
                "check_rag_refusal": str(is_refusal(r["rag_answer"])),
                "check_refusal_expected": str(expects_refusal),
                "check_gold_source_retrieved": gold_src_retrieved,
                "rag_retrieved_sources": retrieved,
                # --- TO BE FILLED BY A HUMAN ANNOTATOR ---
                "b0_correct_manual": "",
                "rag_correct_manual": "",
                "annotator": "",
                "notes": "",
            }
        )

    out_csv = RESULTS_DIR / "annotation_worksheet.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    print(f"Wrote {len(out_rows)} rows to {out_csv}")
    print("\nAll 'check_*' columns are labelled heuristics to speed up annotation.")
    print("They are NOT correctness grades. Fill b0_correct_manual / rag_correct_manual")
    print("by hand (yes / no / partial) and record who annotated each row.")


if __name__ == "__main__":
    main()
