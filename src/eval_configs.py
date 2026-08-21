"""Compare retrieval configurations against document-level gold sources.

Runs every snapshot in data/snapshots/<id>/ through dense, BM25 and RRF hybrid,
and reports Recall@k / MRR plus a wrong-subclass-at-rank-1 rate.

The wrong-subclass rate is D13 definition (c) - the automatable one. It measures
retrieval, not what the user is finally told, so it is a leading indicator for the
headline metric rather than the headline metric itself.

Document-level: uses gold_source from the test set. Chunk-level NDCG needs the
qrels and is not computed here.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import requests

from baseline import retrieval_bm25

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOTS = ROOT / "data" / "snapshots"
TESTSET = ROOT / "data" / "testset" / "test_set.json"
OUT = ROOT / "results" / "config_comparison.json"

OLLAMA = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"
KS = [1, 3, 5]
RRF_K = 60

_cache: dict[str, np.ndarray] = {}


def embed(text: str) -> np.ndarray:
    if text not in _cache:
        r = requests.post(OLLAMA, json={"model": EMBED_MODEL, "prompt": text}, timeout=60)
        r.raise_for_status()
        _cache[text] = np.array(r.json()["embedding"])
    return _cache[text]


def subclass_of(source: str) -> str | None:
    m = re.search(r"subclass_(\d{3})_", source)
    return m.group(1) if m else None


def rrf(orders, n):
    score = np.zeros(n)
    for o in orders:
        for pos, idx in enumerate(o, start=1):
            score[idx] += 1.0 / (RRF_K + pos)
    return np.argsort(-score)


def evaluate(store_path: Path, testset: list[dict]) -> dict:
    payload = json.loads(store_path.read_text())
    chunks = payload["chunks"]
    sources = [c["source"] for c in chunks]

    mat = np.array([c["embedding"] for c in chunks])
    mat = mat / np.linalg.norm(mat, axis=1, keepdims=True)
    bm = retrieval_bm25.BM25(chunks)

    scored = [t for t in testset if t["gold_source"] not in ("multiple", "none", "")]
    out = {}
    for name in ("dense", "bm25", "hybrid_rrf"):
        out[name] = {f"recall@{k}": [] for k in KS}
        out[name]["mrr"] = []
        out[name]["wrong_subclass@1"] = []

    for t in scored:
        qv = embed(t["question"])
        qv = qv / np.linalg.norm(qv)
        o_dense = np.argsort(-(mat @ qv))
        o_bm25 = np.argsort(-bm.scores(t["question"]))
        gold_sc = subclass_of(t["gold_source"])

        for name, order in (("dense", o_dense), ("bm25", o_bm25),
                            ("hybrid_rrf", rrf([o_dense, o_bm25], len(chunks)))):
            hits = [p for p, i in enumerate(order, start=1) if sources[i] == t["gold_source"]]
            first = hits[0] if hits else None
            for k in KS:
                out[name][f"recall@{k}"].append(1.0 if first and first <= k else 0.0)
            out[name]["mrr"].append(1.0 / first if first else 0.0)
            top1_sc = subclass_of(sources[order[0]])
            if gold_sc:
                out[name]["wrong_subclass@1"].append(
                    1.0 if top1_sc is not None and top1_sc != gold_sc else 0.0)

    return {
        # snapshot_id first: two snapshots can share a config_id, and without this
        # a row from a superseded corpus is indistinguishable from a current one.
        "snapshot_id": payload["snapshot_id"],
        "config_id": payload["config_id"],
        "strategy": payload["strategy"],
        "prepend_metadata": payload["prepend_metadata"],
        "n_chunks": payload["n_chunks"],
        "n_questions": len(scored),
        "systems": {n: {m: round(float(np.mean(v)), 3) for m, v in d.items()} for n, d in out.items()},
    }


def main() -> None:
    testset = json.loads(TESTSET.read_text())
    stores = sorted(SNAPSHOTS.glob("*/vector_store__*.json"))
    if not stores:
        sys.exit("No snapshots found. Run ingest.py first.")

    results = [evaluate(p, testset) for p in stores]

    hdr = f"{'snapshot':<14}{'config':<34}{'chunks':>7}  {'retriever':<12}" + "".join(f"{'R@'+str(k):>7}" for k in KS) + f"{'MRR':>7}{'wrong-sc@1':>12}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        for i, (name, m) in enumerate(r["systems"].items()):
            snap = r["snapshot_id"] if i == 0 else ""
            cfg = r["config_id"] if i == 0 else ""
            nch = str(r["n_chunks"]) if i == 0 else ""
            print(f"{snap:<14}{cfg:<34}{nch:>7}  {name:<12}"
                  + "".join(f"{m['recall@'+str(k)]:7.3f}" for k in KS)
                  + f"{m['mrr']:7.3f}{m['wrong_subclass@1']:12.3f}")
        print()

    OUT.write_text(json.dumps({"n_questions": results[0]["n_questions"], "configs": results}, indent=2))
    print(f"Wrote {OUT.relative_to(ROOT)}")
    print("\nwrong-sc@1 = rank-1 chunk belongs to a different subclass than the gold document")
    print("(D13 definition (c) - automatable, measures retrieval not final output)")


if __name__ == "__main__":
    main()
