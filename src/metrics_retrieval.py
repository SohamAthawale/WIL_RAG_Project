"""Chunk-level retrieval metrics: Recall@k, MRR and NDCG@k over the T02/T13 qrels.

Dimension 4. These are the first chunk-level retrieval numbers in the project. Everything
reported before now -- results/retrieval_comparison.json and results/config_comparison.json
-- is DOCUMENT-level: it asks whether the right source file appeared, using gold_source,
over the 8 questions that carry a single gold document. See eval_configs.py, which says so
in its own docstring.

The two are not comparable and must not share a column. The metrics here are therefore
emitted as chunk_recall@k, chunk_mrr and chunk_ndcg@k, so a row lifted into the comparison
table cannot be mistaken for the document-level figure of the same name.

NDCG FORMULA -- stated because papers differ and an unstated choice is irreproducible:

    DCG@k  = sum over positions p=1..k of  (2**rel(p) - 1) / log2(p + 1)
    IDCG@k = the same sum over the ideal ranking (grades sorted descending)
    NDCG@k = DCG@k / IDCG@k

This is the exponential-gain form. The linear form, rel(p) / log2(p + 1), is also in
common use and gives different numbers on the same data. Grades run 0-3, so the choice
matters: exponential gain weights a grade-3 chunk at 7 against a grade-1 chunk at 1,
where linear gain weights them 3 against 1.

WHAT THE QRELS CLAIM, which differs by question:

  exhaustive_all_chunks (12 questions: q01-q11 and q13)
      every chunk in the store was judged, so a chunk absent from the grades was seen
      and rejected. Recall is a true recall.

  pooled_top5_union_dense_bm25_rrf (88 questions)
      only chunks that some retriever surfaced in its top 5 were judged. A chunk absent
      from the grades was never surfaced and is assumed irrelevant. Recall is an UPPER
      BOUND: a relevant chunk that nothing retrieved cannot count against it.

  The pool was built from the top-5 union of the same three retrievers scored here, so
  for pooled questions every relevant chunk is by construction inside some retriever's
  top 5. chunk_recall@5 on that group is close to circular. The size of that effect is
  computed rather than asserted -- see pooling_circularity in the output.

EXCLUSIONS:

  Five questions carry grades but no relevant chunk at all -- q12, q91, q98, q99, q100,
  every one refusal_required. Nothing in the corpus answers them, which is correct, not a
  gap. Recall@k is 0/0 there and NDCG's IDCG is 0: undefined, not zero. Scoring them zero
  would report a retrieval failure on the questions where the system is supposed to have
  nothing to retrieve. They are excluded from all three metrics and reported by name, so
  n=95 and not 100.

THE QRELS FREEZE THE CHUNKING. Chunk ids join 24/24 with structure_aware_meta and
structure_aware_nometa and 17/24 with paragraph_nometa, whose boundaries moved. Scoring
that config would drop seven chunks silently and read as poor retrieval rather than a
broken join, so the join is asserted and an incomplete one raises.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Read-only reuse: the same retrieval the rest of the project measures, rather than a
# second copy that can drift. Nothing in this module edits eval_configs.
import eval_configs  # noqa: E402
from baseline import retrieval_bm25  # noqa: E402

TESTSET = ROOT / "data" / "testset" / "test_set.json"
SNAPSHOTS = ROOT / "data" / "snapshots"
OUT = ROOT / "results" / "retrieval_metrics.json"

KS = (1, 3, 5)
SCOREABLE_CONFIGS = ("structure_aware_meta", "structure_aware_nometa")
SYSTEMS = ("dense", "bm25", "hybrid_rrf")


def _relevant_ids(qrels) -> set[str]:
    """The relevant set, from either an id collection or a grade mapping.

    gold_chunk_ids is everything graded 1 or above; gold_chunk_grades is the 0-3 scale.
    Accepting both means a caller cannot silently pass the wrong one and get a plausible
    number -- T06 lost a counter to exactly that.
    """
    if isinstance(qrels, dict):
        return {cid for cid, grade in qrels.items() if grade >= 1}
    return set(qrels)


def recall_at_k(ranked_ids, qrels, k: int) -> float | None:
    """Fraction of the relevant chunks that appear in the top k.

    Returns None when no relevant chunk exists: 0/0 is undefined, and reporting it as
    0.0 would claim a failure on a question that has nothing to retrieve. A question
    whose relevant chunks exist but are all missed scores 0.0, which is a real zero.
    """
    relevant = _relevant_ids(qrels)
    if not relevant:
        return None
    found = sum(1 for cid in ranked_ids[:k] if cid in relevant)
    return found / len(relevant)


def mrr(ranked_ids, qrels) -> float | None:
    """Reciprocal rank of the first relevant chunk; 0.0 if none is retrieved.

    Returns None when no relevant chunk exists, for the same reason as recall_at_k. The
    0.0 case means the relevant chunks were there to find and the ranking missed them.
    """
    relevant = _relevant_ids(qrels)
    if not relevant:
        return None
    for position, cid in enumerate(ranked_ids, start=1):
        if cid in relevant:
            return 1.0 / position
    return 0.0


def _dcg(grades) -> float:
    """Exponential-gain DCG over an already-ordered list of grades."""
    return sum((2 ** g - 1) / math.log2(p + 1) for p, g in enumerate(grades, start=1))


def ndcg_at_k(ranked_ids, qrels, k: int) -> float | None:
    """Normalised discounted cumulative gain at k, exponential gain.

    qrels must be the grade mapping: NDCG needs the 0-3 scale, not the binary id list.
    Returns None when the ideal DCG is zero, which happens when no chunk has a grade
    above 0 -- undefined rather than zero.
    """
    if not isinstance(qrels, dict):
        raise TypeError(
            "ndcg_at_k needs gold_chunk_grades (the 0-3 mapping), not gold_chunk_ids. "
            "The id list is everything graded 1 or above and loses the scale."
        )
    actual = _dcg([qrels.get(cid, 0) for cid in ranked_ids[:k]])
    ideal = _dcg(sorted(qrels.values(), reverse=True)[:k])
    if ideal == 0:
        return None
    return actual / ideal


# --- producing the rankings the metrics consume ------------------------------------


def load_testset() -> list[dict]:
    return json.loads(TESTSET.read_text())


def load_store(snapshot: str, config: str) -> dict:
    path = SNAPSHOTS / snapshot / f"vector_store__{config}.json"
    if not path.exists():
        raise SystemExit(f"No store at {path}")
    return json.loads(path.read_text())


def assert_qrels_join(store: dict, testset: list[dict], config: str) -> None:
    """Refuse to score a config whose chunk ids do not cover the qrels.

    A partial join is not a low score, it is a different corpus. Without this the
    paragraph_nometa store reads as poor retrieval when seven of the judged chunks
    simply do not exist under its boundaries.
    """
    store_ids = {c["id"] for c in store["chunks"]}
    qrels_ids = set()
    for t in testset:
        qrels_ids |= set((t.get("gold_chunk_grades") or {}).keys())
    missing = qrels_ids - store_ids
    if missing:
        raise SystemExit(
            f"Config {config!r} covers {len(qrels_ids) - len(missing)}/{len(qrels_ids)} "
            f"judged chunk ids. The qrels freeze the chunking, so this config cannot be "
            f"scored against them. Missing: {sorted(missing)[:5]}..."
        )


def rank_all(store: dict, testset: list[dict]) -> dict:
    """{question_id: {system: [chunk_id, ...] full ranking}}."""
    chunks = store["chunks"]
    ids = [c["id"] for c in chunks]
    mat = np.array([c["embedding"] for c in chunks])
    mat = mat / np.linalg.norm(mat, axis=1, keepdims=True)
    bm = retrieval_bm25.BM25(chunks)

    out = {}
    for t in testset:
        qv = eval_configs.embed(t["question"])
        qv = qv / np.linalg.norm(qv)
        o_dense = np.argsort(-(mat @ qv), kind="stable")
        o_bm25 = np.argsort(-bm.scores(t["question"]), kind="stable")
        o_rrf = eval_configs.rrf([o_dense, o_bm25], len(chunks))
        out[t["id"]] = {
            "dense": [ids[i] for i in o_dense],
            "bm25": [ids[i] for i in o_bm25],
            "hybrid_rrf": [ids[i] for i in o_rrf],
        }
    return out


def score_question(ranking: dict, question: dict) -> dict:
    """Every metric for one question, per system. None where undefined."""
    grades = question.get("gold_chunk_grades") or {}
    ids = question.get("gold_chunk_ids") or []
    per_system = {}
    for system in SYSTEMS:
        ranked = ranking[system]
        row = {f"chunk_recall@{k}": recall_at_k(ranked, ids, k) for k in KS}
        row["chunk_mrr"] = mrr(ranked, ids)
        for k in KS:
            row[f"chunk_ndcg@{k}"] = ndcg_at_k(ranked, grades, k)
        per_system[system] = row
    return per_system


def _mean(values) -> float | None:
    present = [v for v in values if v is not None]
    return round(float(np.mean(present)), 3) if present else None


def recall_ceilings(questions: list[dict]) -> dict:
    """The largest Recall@k any ranking could reach on these questions.

    Recall@k is capped at min(k, |relevant|) / |relevant|, and these questions carry 5.58
    relevant chunks on average. Recall@1 therefore cannot exceed about 0.19 no matter how
    good retrieval is. A bare "Recall@1 = 0.18" reads as near-total failure and is in fact
    within a few percent of the maximum, so the ceiling travels next to the number.
    """
    out = {}
    for k in KS:
        sizes = [len(q.get("gold_chunk_ids") or []) for q in questions]
        sizes = [s for s in sizes if s]
        out[f"chunk_recall@{k}"] = round(float(np.mean([min(k, s) / s for s in sizes])), 3) if sizes else None
    return out


def aggregate(scored: dict, questions: list[dict]) -> dict:
    """Mean of each metric over the questions given, per system, with n.

    Recall is reported three ways: the raw mean, the ceiling that the relevant-set sizes
    impose, and the fraction of that ceiling achieved. The third is what answers "is
    retrieval good here", and it is the only one of the three that is comparable across
    question groups with different relevant-set sizes.
    """
    metrics = [f"chunk_recall@{k}" for k in KS] + ["chunk_mrr"] + [f"chunk_ndcg@{k}" for k in KS]
    qids = [q["id"] for q in questions]
    ceilings = recall_ceilings(questions)
    systems = {}
    for system in SYSTEMS:
        row = {m: _mean([scored[qid][system][m] for qid in qids]) for m in metrics}
        for k in KS:
            raw, ceiling = row[f"chunk_recall@{k}"], ceilings[f"chunk_recall@{k}"]
            row[f"chunk_recall@{k}_share_of_ceiling"] = (
                round(raw / ceiling, 3) if raw is not None and ceiling else None
            )
        systems[system] = row
    return {
        "n": len(qids),
        "mean_relevant_chunks": round(
            float(np.mean([len(q.get("gold_chunk_ids") or []) for q in questions])), 2
        ),
        "recall_ceilings": ceilings,
        "ceiling_note": (
            "Recall@k cannot exceed min(k, |relevant|)/|relevant|. Read chunk_recall@k "
            "against its ceiling, or use the share_of_ceiling figure."
        ),
        "systems": systems,
    }


def pooling_circularity(rankings: dict, questions: list[dict]) -> dict:
    """How much of the pooled gold set sits inside the top 5 that built the pool.

    The pool was the top-5 union of these same three retrievers, so a relevant chunk in a
    pooled question is there because some retriever ranked it in its top 5. If that holds
    at 100%, chunk_recall@5 over pooled questions is measuring the pool's construction as
    much as the retrieval, and only k < 5 carries independent information.
    """
    inside = total = 0
    for q in questions:
        relevant = set(q.get("gold_chunk_ids") or [])
        if not relevant:
            continue
        union = set()
        for system in SYSTEMS:
            union |= set(rankings[q["id"]][system][:5])
        inside += len(relevant & union)
        total += len(relevant)
    return {
        "relevant_chunks": total,
        "inside_top5_union": inside,
        "share": round(inside / total, 3) if total else None,
        "note": (
            "Pooled qrels only. A share at or near 1.0 means chunk_recall@5 on this group "
            "is largely determined by how the pool was built, so read k=1 and k=3 as the "
            "informative columns."
        ),
    }


def main() -> None:
    testset = load_testset()
    by_id = {t["id"]: t for t in testset}

    zero_relevant = [t["id"] for t in testset if not (t.get("gold_chunk_ids") or [])]
    scoreable = [t for t in testset if t["id"] not in zero_relevant]
    exhaustive = [t for t in scoreable if t.get("qrels_method") == "exhaustive_all_chunks"]
    pooled = [t for t in scoreable if t.get("qrels_method") != "exhaustive_all_chunks"]

    snapshot = json.loads((ROOT / "data" / "documents" / "MANIFEST.json").read_text())["snapshot_id"]

    configs = {}
    for config in SCOREABLE_CONFIGS:
        store = load_store(snapshot, config)
        assert_qrels_join(store, testset, config)
        rankings = rank_all(store, testset)
        scored = {t["id"]: score_question(rankings[t["id"]], t) for t in scoreable}
        configs[config] = {
            "n_chunks": store["n_chunks"],
            "combined": aggregate(scored, scoreable),
            "exhaustive_qrels": aggregate(scored, exhaustive),
            "pooled_qrels": aggregate(scored, pooled),
            "pooling_circularity": pooling_circularity(rankings, pooled),
            "per_question": {qid: scored[qid] for qid in sorted(scored)},
        }

    out = {
        "task": "T03 -- chunk-level retrieval metrics (Recall@k, MRR, NDCG@k)",
        "snapshot_id": snapshot,
        "ndcg_formula": "exponential gain: (2**rel - 1) / log2(pos + 1), normalised by the same over the ideal ranking",
        "metric_names": {
            "note": (
                "Prefixed chunk_ deliberately. results/retrieval_comparison.json and "
                "results/config_comparison.json report recall@k and mrr at DOCUMENT level "
                "over 8 questions, matching gold_source. These are chunk level over the "
                "qrels at n=95. Same words, different measures; they must not share a column."
            ),
        },
        "denominator": {
            "n_questions_total": len(testset),
            "n_scored": len(scoreable),
            "excluded_zero_relevant": zero_relevant,
            "exclusion_reason": (
                "These carry grades but no chunk graded 1 or above, and all are "
                "refusal_required: nothing in the corpus answers them. Recall is 0/0 and "
                "IDCG is 0, so all three metrics are undefined rather than zero. Scoring "
                "them zero would report a retrieval failure where correct behaviour is to "
                "retrieve nothing."
            ),
            "n_exhaustive_qrels": len(exhaustive),
            "n_pooled_qrels": len(pooled),
        },
        "qrels_caveat": (
            "Recall over pooled qrels is an upper bound: a relevant chunk that no retriever "
            "surfaced was never judged and so cannot count against recall. The exhaustive "
            "group carries a true recall but is only 12 questions. Both are reported."
        ),
        "cross_config_comparison_warning": (
            "build_pooled_sheet.py imports rag, whose DEFAULT_CONFIG is structure_aware_meta, "
            "so the pool was built from that config's retrievals. Every relevant chunk in the "
            "pooled qrels sits inside meta's top-5 union (1.000) and only 0.794 of them sit "
            "inside nometa's. A chunk nometa would have surfaced and meta would not was never "
            "judged, so nometa cannot be credited for it. Comparing the two configs on the "
            "pooled group is therefore biased toward meta and should not be reported as a "
            "clean result. The 12 exhaustively judged questions carry no such bias: every "
            "chunk was judged under both. Read the meta-vs-nometa comparison there."
        ),
        "configs": configs,
    }
    OUT.write_text(json.dumps(out, indent=2))

    for config, block in configs.items():
        print(f"\n{config}  ({block['n_chunks']} chunks)")
        header = f"  {'group':<18}{'n':>4}  {'sys':<11}" + "".join(f"{'R@'+str(k):>8}" for k in KS) \
                 + f"{'MRR':>8}" + "".join(f"{'N@'+str(k):>8}" for k in KS)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for group in ("combined", "exhaustive_qrels", "pooled_qrels"):
            agg = block[group]
            for system in SYSTEMS:
                m = agg["systems"][system]
                cells = "".join(f"{m[f'chunk_recall@{k}']:>8.3f}" for k in KS)
                cells += f"{m['chunk_mrr']:>8.3f}"
                cells += "".join(f"{m[f'chunk_ndcg@{k}']:>8.3f}" for k in KS)
                label = group if system == SYSTEMS[0] else ""
                print(f"  {label:<18}{agg['n'] if system == SYSTEMS[0] else '':>4}  {system:<11}{cells}")
            ceil = agg["recall_ceilings"]
            print(f"  {'':<18}{'':>4}  {'(ceiling)':<11}"
                  + "".join(f"{ceil[f'chunk_recall@{k}']:>8.3f}" for k in KS)
                  + f"{'':>8}" + "".join(f"{'':>8}" for _ in KS)
                  + f"   mean relevant {agg['mean_relevant_chunks']}")
        c = block["pooling_circularity"]
        print(f"  pooled gold chunks inside the top-5 union that built the pool: "
              f"{c['inside_top5_union']}/{c['relevant_chunks']} = {c['share']}")

    print(f"\nExcluded {len(zero_relevant)} zero-relevant questions: {', '.join(zero_relevant)}")
    print(f"Wrote {OUT.relative_to(ROOT)}  (n={len(scoreable)}; "
          f"{len(exhaustive)} exhaustive, {len(pooled)} pooled)")


if __name__ == "__main__":
    main()
