"""Document-level retrieval comparison: dense (B2) vs BM25 (B1) vs RRF hybrid.

Uses `gold_source` from the test set as a document-level relevance signal, so
this runs WITHOUT the chunk-level qrels. It is therefore coarser than the
NDCG the framework will eventually report: it answers "was the right document
retrieved", not "was the right passage ranked well".

Questions whose gold_source is 'multiple' or 'none' are excluded - there is no
single document to score against.
"""

import json
from pathlib import Path

import numpy as np

import retrieval_bm25
from rag import embed, load_store

ROOT = Path(__file__).resolve().parent.parent
TESTSET = ROOT / "data" / "testset" / "test_set.json"
OUT = ROOT / "results" / "retrieval_comparison.json"

KS = [1, 3, 5]
RRF_K = 60


def rank_dense(qv, mat_norm):
    return np.argsort(-(mat_norm @ qv))


def rank_bm25(query, bm):
    return np.argsort(-bm.scores(query))


def rrf(rank_lists, n):
    """Reciprocal Rank Fusion over full orderings."""
    score = np.zeros(n)
    for order in rank_lists:
        for pos, idx in enumerate(order, start=1):
            score[idx] += 1.0 / (RRF_K + pos)
    return np.argsort(-score)


def main() -> None:
    store = load_store()
    testset = json.loads(TESTSET.read_text())
    bm = retrieval_bm25.BM25(store)

    mat = np.array([r["embedding"] for r in store])
    mat_norm = mat / np.linalg.norm(mat, axis=1, keepdims=True)
    sources = [r["source"] for r in store]

    scored = [t for t in testset if t["gold_source"] not in ("multiple", "none", "")]
    print(f"Scoring {len(scored)} of {len(testset)} questions "
          f"(excluded: gold_source is 'multiple' or 'none')\n")

    systems = {"dense": {}, "bm25": {}, "hybrid_rrf": {}}
    per_q = []

    for t in scored:
        qv = embed(t["question"])
        qv = qv / np.linalg.norm(qv)
        o_dense = rank_dense(qv, mat_norm)
        o_bm25 = rank_bm25(t["question"], bm)
        o_rrf = rrf([o_dense, o_bm25], len(store))

        row = {"id": t["id"], "gold_source": t["gold_source"]}
        for name, order in (("dense", o_dense), ("bm25", o_bm25), ("hybrid_rrf", o_rrf)):
            hit_positions = [p for p, i in enumerate(order, start=1)
                             if sources[i] == t["gold_source"]]
            first = hit_positions[0] if hit_positions else None
            row[f"{name}_first_rank"] = first
            for k in KS:
                systems[name].setdefault(f"recall@{k}", []).append(
                    1.0 if first is not None and first <= k else 0.0
                )
            systems[name].setdefault("mrr", []).append(1.0 / first if first else 0.0)
        per_q.append(row)

    print(f"{'system':<12} " + " ".join(f"{'R@'+str(k):>7}" for k in KS) + f" {'MRR':>7}")
    print("-" * 46)
    summary = {}
    for name, m in systems.items():
        vals = [np.mean(m[f"recall@{k}"]) for k in KS] + [np.mean(m["mrr"])]
        summary[name] = {f"recall@{k}": round(float(np.mean(m[f'recall@{k}'])), 3) for k in KS}
        summary[name]["mrr"] = round(float(np.mean(m["mrr"])), 3)
        print(f"{name:<12} " + " ".join(f"{v:7.3f}" for v in vals))

    print(f"\n{'q':<6} {'gold doc':<40} {'dense':>6} {'bm25':>6} {'rrf':>6}   (rank of gold doc)")
    print("-" * 76)
    for r in per_q:
        f = lambda v: str(v) if v else "miss"
        print(f"{r['id']:<6} {r['gold_source']:<40} "
              f"{f(r['dense_first_rank']):>6} {f(r['bm25_first_rank']):>6} {f(r['hybrid_rrf_first_rank']):>6}")

    OUT.write_text(json.dumps(
        {"n_questions": len(scored), "summary": summary, "per_question": per_q}, indent=2))
    print(f"\nWrote {OUT}")
    print("\nNOTE: document-level only. Chunk-level NDCG needs the qrels from the "
          "annotation workbook and is not computed here.")


if __name__ == "__main__":
    main()
