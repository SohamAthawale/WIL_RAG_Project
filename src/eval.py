"""Run the test set through B0 (bare LLM) and a retrieval configuration.

Every output file is tagged with the snapshot and config that produced it, and
carries a run_meta block. Results from different chunkings are therefore never
silently comparable - which was a real defect: the generation path used to read
a stale flat store while the retrieval experiments read snapshots.

Usage:
    python eval.py                                   # structure_aware_meta, k=3
    python eval.py --config structure_aware_nometa   # D9 control
    python eval.py --k 1
"""

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from rag import DEFAULT_CONFIG, answer_b0, answer_rag, load_store, store_meta

ROOT = Path(__file__).resolve().parent.parent
TEST_SET_PATH = ROOT / "data" / "testset" / "test_set.json"
RESULTS_DIR = ROOT / "results"

FIELDS = [
    "id", "category", "question", "gold_answer", "gold_source",
    "b0_answer", "rag_answer", "rag_retrieved_sources",
    "b0_correct_manual", "rag_correct_manual", "notes",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--skip-b0", action="store_true",
                    help="reuse an earlier B0 run; B0 does not depend on the config")
    args = ap.parse_args()

    test_set = json.loads(TEST_SET_PATH.read_text())
    store = load_store(args.config)
    meta = store_meta(args.config)
    meta["top_k"] = args.k
    meta["run_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    RESULTS_DIR.mkdir(exist_ok=True)
    tag = f"{meta['config_id']}_k{args.k}"
    print(f"snapshot {meta['snapshot_id']} · config {meta['config_id']} · "
          f"{meta['n_chunks']} chunks · k={args.k}\n")

    rows = []
    for i, item in enumerate(test_set, 1):
        print(f"[{i}/{len(test_set)}] {item['id']}: {item['question'][:64]}...")
        b0 = {"answer": ""} if args.skip_b0 else answer_b0(item["question"])
        rag = answer_rag(item["question"], store, args.k)
        rows.append({
            "id": item["id"],
            "category": item["category"],
            "question": item["question"],
            "gold_answer": item["gold_answer"],
            "gold_source": item["gold_source"],
            "b0_answer": b0["answer"].replace("\n", " "),
            "rag_answer": rag["answer"].replace("\n", " "),
            "rag_retrieved_sources": "; ".join(
                f"{r['source']} ({r['score']})" for r in rag["retrieved"]),
            "b0_correct_manual": "",
            "rag_correct_manual": "",
            "notes": "",
        })

    out_csv = RESULTS_DIR / f"b0_vs_rag__{tag}.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    out_json = RESULTS_DIR / f"b0_vs_rag__{tag}.json"
    out_json.write_text(json.dumps({"run_meta": meta, "rows": rows}, indent=2))

    print(f"\nWrote {len(rows)} rows:\n  {out_csv.name}\n  {out_json.name}")
    print("\nManual grading columns are intentionally blank - correctness is the ground")
    print("truth every later metric is scored against and must be judged by a person.")


if __name__ == "__main__":
    main()
