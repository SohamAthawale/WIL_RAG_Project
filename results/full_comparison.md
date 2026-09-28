# Full comparison — four baselines × eight dimensions

Snapshot `84636c1d6741` · config `structure_aware_meta` · k=3 · `qwen2.5:7b-instruct`,
temperature 0, seed 97.

**Cell states.** A value carries the n it was measured on. `not yet measured` means the
task has not been done. `no run` means the system exists but generation was never run for
it — `rag.retrieve` is dense cosine, so B1 and B3 are scored at retrieval only. `n/a` means
the dimension does not apply.

**Denominators differ by measure and all are correct:** n=95 chunk-level retrieval, n=61
document-level, n=54 wrong-subclass citations, n=100 answer-only measures, n=13 where a
human grade is required. Each cell states its own.

**Recall is share of its ceiling.** Recall@1 cannot exceed 1/|relevant|; at 5.58 relevant
chunks per question the ceiling is 0.192, so a raw 0.182 is 95% of the maximum, not a
failure.

## Answerability

| Baseline | Retrieval | Result |
|---|---|---|
| B0 | no retrieval | correct_refusal 2, answered_but_wrong 13 (n=100) |
| B1 | BM25 | no run |
| B2 | dense | correct_refusal 8, answered_but_wrong 7 (n=100) |
| B3 | hybrid RRF | no run |

## Faithfulness

| Baseline | Retrieval | Result |
|---|---|---|
| B0 | no retrieval | n/a |
| B1 | BM25 | no run |
| B2 | dense | 0.987 (n=100) [numeric facts only] |
| B3 | hybrid RRF | no run |

## Attribution

| Baseline | Retrieval | Result |
|---|---|---|
| B0 | no retrieval | cites 0/100 (n=100) [never cites] |
| B1 | BM25 | no run |
| B2 | dense | cites 49/100, instructed form 24/100, fabricated 0/100, wrong-subclass 0/54 |
| B3 | hybrid RRF | no run |

## Retrieval

| Baseline | Retrieval | Result |
|---|---|---|
| B0 | no retrieval | n/a |
| B1 | BM25 | NDCG@5 0.835, MRR 0.963, R@1 0.911 of ceiling (n=95) [chunk level] |
| B2 | dense | NDCG@5 0.833, MRR 0.98, R@1 0.948 of ceiling (n=95) [chunk level] |
| B3 | hybrid RRF | NDCG@5 0.879, MRR 0.982, R@1 0.953 of ceiling (n=95) [chunk level] |

## Cross Subclass

| Baseline | Retrieval | Result |
|---|---|---|
| B0 | no retrieval | n/a |
| B1 | BM25 | no run |
| B2 | dense | (a) 0, (b) 0, (c) 1, (d) 0, (e) 0 (n=13) [5 counters] |
| B3 | hybrid RRF | no run |

## Refusal

| Baseline | Retrieval | Result |
|---|---|---|
| B0 | no retrieval | declines 2/15, hedged-then-advised 1 (n=15) |
| B1 | BM25 | no run |
| B2 | dense | declines 8/15, hedged-then-advised 7 (n=15) |
| B3 | hybrid RRF | no run |

## Vocabulary Fairness

| Baseline | Retrieval | Result |
|---|---|---|
| B0 | no retrieval | not yet measured |
| B1 | BM25 | not yet measured |
| B2 | dense | not yet measured |
| B3 | hybrid RRF | not yet measured |

## Temporal

| Baseline | Retrieval | Result |
|---|---|---|
| B0 | no retrieval | not yet measured |
| B1 | BM25 | not yet measured |
| B2 | dense | not yet measured |
| B3 | hybrid RRF | not yet measured |

