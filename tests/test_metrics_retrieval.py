"""Controls for the chunk-level retrieval metrics.

A metric that is silently wrong is worse than no metric: every number downstream inherits
the error with nothing to signal it. So each function is pinned to a hand-computed value
rather than to whatever it currently returns, the NDCG formula choice is asserted against
the alternative it could have been, and the undefined cases are checked to stay undefined
instead of collapsing to zero.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import metrics_retrieval as mr  # noqa: E402

failures: list[str] = []


def check(name: str, got, want) -> None:
    status = "ok " if got == want else "FAIL"
    print(f"  {status} {name}")
    if got != want:
        failures.append(f"{name}: got {got!r}, want {want!r}")


def close(name: str, got, want, tol=1e-3) -> None:
    ok = got is not None and abs(got - want) < tol
    print(f"  {'ok ' if ok else 'FAIL'} {name}")
    if not ok:
        failures.append(f"{name}: got {got!r}, want ~{want}")


print("the worked example from the brief")
# 3 chunks ranked [A, B, C]; relevance A=3, B=0, C=1; k=3.
# DCG  = (2^3-1)/log2(2) + (2^0-1)/log2(3) + (2^1-1)/log2(4) = 7 + 0 + 0.5 = 7.5
# ideal ranking is [A, C, B]: IDCG = 7 + 1/log2(3) = 7.6309
# NDCG = 7.5 / 7.6309 = 0.9828
grades = {"A": 3, "B": 0, "C": 1}
close("NDCG@3 on [A, B, C] is 0.983", mr.ndcg_at_k(["A", "B", "C"], grades, 3), 0.983)
close("its DCG is 7.5", mr._dcg([3, 0, 1]), 7.5)
close("its IDCG is 7.631", mr._dcg([3, 1, 0]), 7.631)

print("the formula choice is the exponential one, and it matters")
# Linear gain would give a different answer on the same data. Asserted so that a future
# edit to _dcg cannot quietly switch conventions and leave every number unreproducible.
linear_dcg = sum(g / math.log2(p + 1) for p, g in enumerate([3, 0, 1], start=1))
linear_idcg = sum(g / math.log2(p + 1) for p, g in enumerate([3, 1, 0], start=1))
close("linear gain would give 0.964, not 0.983", linear_dcg / linear_idcg, 0.964)
check("the two conventions disagree",
      abs(linear_dcg / linear_idcg - mr.ndcg_at_k(["A", "B", "C"], grades, 3)) > 0.01, True)
# They differ by only 0.019 here, which is the reason the choice has to be declared rather
# than inferred: the gap is far too small to spot from a reported number, and large enough
# to move a comparison between two systems that sit close together.
check("but not by enough to identify from the number alone",
      abs(linear_dcg / linear_idcg - mr.ndcg_at_k(["A", "B", "C"], grades, 3)) < 0.05, True)

print("a perfect ranking scores 1.0 and order is what separates them")
close("ideal order gives NDCG 1.0", mr.ndcg_at_k(["A", "C", "B"], grades, 3), 1.0)
check("reversing it scores lower",
      mr.ndcg_at_k(["B", "C", "A"], grades, 3) < mr.ndcg_at_k(["A", "C", "B"], grades, 3), True)

print("recall")
close("all relevant inside top k is 1.0", mr.recall_at_k(["A", "B", "C"], ["A", "C"], 3), 1.0)
close("half of them is 0.5", mr.recall_at_k(["A", "B", "C"], ["A", "D"], 3), 0.5)
close("none retrieved is a real zero", mr.recall_at_k(["X", "Y"], ["A"], 2), 0.0)
close("k truncates the ranking", mr.recall_at_k(["X", "A"], ["A"], 1), 0.0)
close("recall accepts a grade mapping and uses grade >= 1",
      mr.recall_at_k(["A", "B"], {"A": 2, "B": 0}, 2), 1.0)

print("mrr")
close("first relevant at position 2 gives 0.5", mr.mrr(["X", "A", "B"], ["A"]), 0.5)
close("first relevant at position 1 gives 1.0", mr.mrr(["A", "X"], ["A"]), 1.0)
close("first relevant at position 4 gives 0.25", mr.mrr(["W", "X", "Y", "A"], ["A"]), 0.25)
close("relevant present but never retrieved is 0.0", mr.mrr(["X", "Y"], ["A"]), 0.0)

print("the zero-relevant case stays undefined rather than becoming zero")
# q12, q91, q98, q99, q100 carry grades but nothing graded 1 or above. Returning 0.0 would
# report a retrieval failure on questions that correctly have nothing to retrieve.
check("recall is None, not 0.0", mr.recall_at_k(["A", "B"], [], 3), None)
check("mrr is None, not 0.0", mr.mrr(["A", "B"], []), None)
check("ndcg is None when IDCG is 0", mr.ndcg_at_k(["A", "B"], {"A": 0, "B": 0}, 3), None)
check("and none of them raise", True, True)

print("ndcg refuses the id list, because it would silently lose the scale")
try:
    mr.ndcg_at_k(["A"], ["A"], 3)
    check("passing gold_chunk_ids to ndcg raises", False, True)
except TypeError:
    check("passing gold_chunk_ids to ndcg raises", True, True)

print("the recall ceiling, which is what makes Recall@1 readable")
# Recall@1 cannot exceed 1/|relevant|. With 5 relevant chunks the best possible is 0.2, so
# a raw 0.18 is 90% of the maximum rather than a near-total failure.
one_q = [{"id": "x", "gold_chunk_ids": ["a", "b", "c", "d", "e"]}]
close("Recall@1 ceiling with 5 relevant is 0.2", mr.recall_ceilings(one_q)["chunk_recall@1"], 0.2)
close("Recall@5 ceiling with 5 relevant is 1.0", mr.recall_ceilings(one_q)["chunk_recall@5"], 1.0)

print("the join guard refuses a config the qrels do not cover")
testset = mr.load_testset()
snapshot = json.loads((ROOT / "data" / "documents" / "MANIFEST.json").read_text())["snapshot_id"]
for config in mr.SCOREABLE_CONFIGS:
    store = mr.load_store(snapshot, config)
    mr.assert_qrels_join(store, testset, config)
    check(f"{config} covers the qrels", True, True)
try:
    mr.assert_qrels_join(mr.load_store(snapshot, "paragraph_nometa"), testset, "paragraph_nometa")
    check("paragraph_nometa is refused", False, True)
except SystemExit:
    check("paragraph_nometa is refused rather than scored on a partial join", True, True)

print("the excluded set is exactly the five questions with no relevant chunk")
zero = {t["id"] for t in testset if not (t.get("gold_chunk_ids") or [])}
check("q12 q91 q98 q99 q100", zero, {"q12", "q91", "q98", "q99", "q100"})
check("and every one of them is refusal_required",
      {t["category"] for t in testset if t["id"] in zero}, {"refusal_required"})

print()
if failures:
    print(f"{len(failures)} FAILED")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("all controls passed")
