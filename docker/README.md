# Running the evaluation harness in Docker

Two services. `ollama` holds the models and owns the GPU; `harness` is the evaluation
code and is CPU-only. They are separate because most of this project does not need a GPU,
and pretending otherwise makes the whole thing look heavier than it is.

## What needs what

The useful boundary is not "offline versus GPU" — it is **which model a command touches**.

| Tier | Needs | Commands |
|---|---|---|
| none | nothing; run with `--network none` | `tests-offline`, `score-offline`, `manifest` |
| embeddings | `nomic-embed-text`, ~275 MB, no GPU | `tests`, `score`, `reproduce`, `confusion`, `retrieve` |
| generation | `qwen2.5:7b-instruct`, ~4.7 GB, wants the GPU | `ask`, `compare`, `eval` |

The middle tier is the surprising one. Retrieval metrics, grounding and the confusion
counters all *replay retrieval* against the stored corpus, so they have to embed the query
even though every document embedding is already on disk.

## First run

```bash
docker compose build harness
docker compose up -d ollama
docker compose run --rm harness models        # one-time, pulls ~5 GB
docker compose run --rm harness tests
```

`make help` lists the same things if you prefer that.

Check the GPU is actually attached before a generation run — if it is not, Ollama will
fall back to CPU and a full `eval` goes from minutes to hours:

```bash
docker compose exec ollama nvidia-smi
```

## Moving it to another machine

```bash
make save                        # -> dist/wil-rag-harness.tar.gz
WITH_OLLAMA=1 make save          # include the Ollama runtime too
```

On the target box:

```bash
gunzip -c wil-rag-harness.tar.gz | docker load
docker compose up -d ollama
docker compose run --rm harness models
docker compose run --rm --no-deps harness tests-offline
```

Copy `compose.yaml` across with the tarball. Model weights are deliberately **not** in the
image: they live in the `ollama-models` named volume, they are several GB, and pulling them
once on the target is faster than shipping them.

### Building for a different architecture

The image is architecture-specific. Build on, or for, the machine that will run it:

```bash
docker build --platform linux/amd64 -t wil-rag-harness:latest .
```

## Requirements on the host

- Docker with the Compose plugin
- NVIDIA driver and `nvidia-container-toolkit`, for the GPU tier only
- A Blackwell card (5060 Ti is `sm_120`) needs a CUDA 12.8+ Ollama build. `ollama/ollama:latest`
  is current enough; pin it once you have a tag that works:
  `OLLAMA_IMAGE=ollama/ollama:x.y.z docker compose up -d ollama`

## The results directory is baked in, on purpose

The published `results/` is **copied into the image**, not mounted from the host. A scoring
run rewrites files inside its own container and the host copy is never touched.

This matters more than it sounds. The human relevance grades in this project are attached to
the exact answer text in the stored runs. Regenerating those runs silently detaches every
judgement from the thing it was judging — which has already happened once on this project and
cost a week. Anything a run produces that is worth keeping is written to `/out`, which compose
maps to `./out`.

## `reproduce`

```bash
docker compose run --rm harness reproduce
```

It re-derives the scoring outputs from the stored runs and diffs them field by field against
the copies baked into the image at build time. Every field is expected to match: ranking sorts
are stable and generation is pinned to temperature 0 with a fixed seed, so the pipeline is
deterministic end to end and a difference means a real change rather than machine noise.

Verified across architectures — `retrieval_metrics.json` rebuilt on `linux/amd64` is identical
field for field to the same file rebuilt on `darwin/arm64`.

## A note on ranking determinism

Every `argsort` that produces a ranking passes `kind="stable"`. This matters most in
`eval_configs.rrf`: reciprocal rank fusion sums `1/(k + rank)` over full rankings, and two
chunks land on the same total whenever they swap ranks between the two retrievers. Ties are
therefore common rather than exotic. A stable sort settles them on store order, which is defined
and the same everywhere; the default quicksort settles them on whatever the local NumPy build
happens to do, which is not portable.
