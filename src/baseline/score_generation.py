"""Score every run file with the text-similarity measures.

Writes results/generation_metrics.csv. The proxy column is labelled in the
header so it cannot be mistaken for BERTScore downstream.
"""

import csv
import json
from pathlib import Path

from baseline.metrics_generation import PROXY_LABEL, embedding_cosine_proxy, rouge_1

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "results"
OUT = RESULTS / "generation_metrics.csv"


def main() -> None:
    rows = []
    for run in sorted(RESULTS.glob("b0_vs_rag__*.json")):
        payload = json.loads(run.read_text())
        meta = payload["run_meta"]
        for r in payload["rows"]:
            gold = r["gold_answer"]
            for system in ("b0", "rag"):
                ans = r.get(f"{system}_answer", "")
                if not ans.strip():
                    continue
                rg = rouge_1(ans, gold)
                rows.append({
                    "config_id": meta["config_id"], "top_k": meta["top_k"],
                    "id": r["id"], "category": r["category"], "system": system,
                    "rouge1_precision": rg["precision"], "rouge1_recall": rg["recall"],
                    "rouge1_f1": rg["f1"],
                    PROXY_LABEL: embedding_cosine_proxy(ans, gold),
                })
    if not rows:
        raise SystemExit("No run files found.")
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUT.name}")
    print(f"Proxy column header: {PROXY_LABEL!r}")

    import statistics as st
    for sysname in ("b0", "rag"):
        v = [r["rouge1_f1"] for r in rows if r["system"] == sysname]
        c = [r[PROXY_LABEL] for r in rows if r["system"] == sysname]
        if v:
            print(f"  {sysname:4} mean ROUGE-1 F1 {st.mean(v):.3f}   mean proxy {st.mean(c):.3f}   (n={len(v)})")


if __name__ == "__main__":
    main()
