"""Run the test set through B0 (bare LLM) and RAG, and write a comparison
table for manual annotation. This is the preliminary Milestone 1 run —
manual grading columns are left blank for the team to fill in."""

import csv
import json
from pathlib import Path

from rag import answer_b0, answer_rag, load_store

ROOT = Path(__file__).resolve().parent.parent
TEST_SET_PATH = ROOT / "data" / "testset" / "test_set.json"
RESULTS_DIR = ROOT / "results"


def main() -> None:
    test_set = json.loads(TEST_SET_PATH.read_text())
    store = load_store()
    RESULTS_DIR.mkdir(exist_ok=True)

    rows = []
    for i, item in enumerate(test_set, 1):
        print(f"[{i}/{len(test_set)}] {item['id']}: {item['question'][:70]}...")

        b0 = answer_b0(item["question"])
        rag = answer_rag(item["question"], store)

        rows.append(
            {
                "id": item["id"],
                "category": item["category"],
                "question": item["question"],
                "gold_answer": item["gold_answer"],
                "gold_source": item["gold_source"],
                "b0_answer": b0["answer"].replace("\n", " "),
                "rag_answer": rag["answer"].replace("\n", " "),
                "rag_retrieved_sources": "; ".join(
                    f"{r['source']} ({r['score']})" for r in rag["retrieved"]
                ),
                "b0_correct_manual": "",
                "rag_correct_manual": "",
                "notes": "",
            }
        )

    out_csv = RESULTS_DIR / "b0_vs_rag_comparison.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    out_json = RESULTS_DIR / "b0_vs_rag_comparison.json"
    out_json.write_text(json.dumps(rows, indent=2))

    print(f"\nWrote {len(rows)} rows to:\n  {out_csv}\n  {out_json}")
    print("\nNext step: fill in b0_correct_manual / rag_correct_manual (yes/no/partial) by hand,")
    print("then run summarize.py for the headline numbers.")


if __name__ == "__main__":
    main()
