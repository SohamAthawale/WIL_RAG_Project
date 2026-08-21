# Provenance — what is reproduced, what is ours

Written so the distinction is verifiable rather than asserted. Reproducing published measures is
what gives our numbers a comparison point; the contribution is what sits on top.

## Layout

```
src/baseline/    reproduced measurement apparatus
src/             this project's system and its own measures
```

No external code or data is vendored anywhere in this repository. The baseline package contains
independent implementations of standard measures.

## Reproduced — published methodology

| Component | Where | Status |
|---|---|---|
| Okapi BM25 sparse retrieval (k₁=1.2, b=0.75) | `src/baseline/retrieval_bm25.py` | Done |
| ROUGE-1 precision / recall / F1 | `src/baseline/metrics_generation.py` | Done |
| Semantic similarity (labelled proxy, not BERTScore) | `src/baseline/metrics_generation.py` | Done |
| top-k sweep at k = 1, 3, 5 | `results/generation_metrics.csv` | Done |
| Dense retrieval condition | `src/rag.py` | Done |
| No-retrieval control (B0) | `src/rag.py` | Done |
| Answer-generation prompt with refusal instruction | `src/rag.py` | Done |
| Three-way question categorisation | `data/testset/test_set.json` | Done |
| NDCG | — | **Blocked on relevance judgements** |
| Significance testing | — | **Blocked on test set size** |

## Standard practice — neither borrowed nor novel

Worth separating out, because calling these "reproduced" would overstate the debt.

- **k₁=1.2, b=0.75** are the conventional Okapi defaults, used across the field.
- **A no-retrieval control** is ordinary experimental design.
- **Cosine similarity over embeddings** for dense retrieval is the standard formulation.
- **Instructing a model to decline when context is insufficient** is a widely used RAG pattern.

## Ours — original to this project

| Component | Where | Why it is not reproduction |
|---|---|---|
| Four chunking strategies behind one interface | `src/chunking.py` | The referenced work used pre-segmented records and had no chunking step |
| Rule/exemption binding rules | `src/chunking.py` | Addresses a failure mode specific to regulatory text |
| Subclass identifier embedded in chunk text | `src/chunking.py` | Original; measured as a controlled before/after |
| Corpus-hash snapshot and config tagging | `src/ingest.py` | Reproducibility discipline, original |
| Cross-subclass confusion rate | `src/eval_configs.py` | The referenced corpus was homogeneous, so this failure could not arise there |
| Chunking strategy sweep and its power analysis | `src/sweep_chunking.py` | Original |
| Split answerability, attribution correctness, temporal correctness | planned | Original dimensions |

## Measured results, and which side they belong to

| Result | Side |
|---|---|
| BM25 R@1 0.875 vs dense 0.625 | reproduction — comparing the published conditions |
| Quality rises with k; published finding did not replicate | reproduction — a refutation, which is a result |
| ROUGE-1 ranks a wrong answer above a correct one | reproduction — a limitation of the measure |
| RRF fusion only pays off once chunks carry subclass labels | **ours** |
| 8547 rule/exemption split in 30 of 34 configurations, fixed to 18 | **ours** |
| Chunking strategy differences sit below the noise floor at n=8 | **ours** |
| q01 retrieval miss and fabricated citation, both fixed | **ours** |
| q02 cross-stream confusion exposed after cross-subclass was eliminated | **ours** |

## Tag

`baseline-reproduction` marks the commit where the reproducible portion was complete. NDCG and
significance testing remain outstanding and are gated on the relevance judgements and test set
size respectively, not on effort.
