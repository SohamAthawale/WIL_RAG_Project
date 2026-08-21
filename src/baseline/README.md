# Baseline — reproduced methodology

Everything in this package reproduces evaluation apparatus described in the literature, so this
project's numbers have a published point of comparison. Everything **outside** this package is the
project's own work.

| Module | What it is |
|---|---|
| `retrieval_bm25.py` | Okapi BM25 sparse retrieval, conventional k₁=1.2 / b=0.75 |
| `metrics_generation.py` | ROUGE-1, and a similarity proxy explicitly labelled as not BERTScore |
| `score_generation.py` | Applies both across every run file |

No code or data from any external project is vendored here. These are independent
implementations of standard, published measures.

See `docs/provenance.md` for the component-by-component split.
