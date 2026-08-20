# Requirements and setup

Everything runs **locally and free**. No API keys, no cloud services, no paid models.

## What you need

| | Requirement | Notes |
|---|---|---|
| **OS** | macOS, Linux or Windows | Ollama supports all three |
| **Python** | 3.10 or newer | Developed on 3.13 |
| **Ollama** | 0.32 or newer | The local model runner — https://ollama.com |
| **Git** | any recent version | |
| **RAM** | 8 GB minimum, 16 GB comfortable | The 7B model needs ~6 GB free while running |
| **Disk** | ~6 GB | 5 GB models, 52 MB virtual environment, ~3 MB repo |
| **GPU** | not required | Apple Silicon and NVIDIA are used automatically if present; CPU works, just slower |

No internet is needed once the models are pulled — the corpus is already in the repo.

## Models

| Model | Size | Used for |
|---|---|---|
| `nomic-embed-text` | 274 MB | Turning text into vectors for semantic search |
| `qwen2.5:7b-instruct` | 4.7 GB | Generating answers |

`qwen2.5:7b-instruct` stands in for the Falcon-7B-Instruct that the Walert paper used. Both are
open 7B instruction-tuned models, so the substitution is methodologically defensible — and it
should be stated as such in the report.

## Setup

**1. Install Ollama** from https://ollama.com, then pull the models:

```
ollama pull nomic-embed-text
ollama pull qwen2.5:7b-instruct
```

**2. Clone and configure git** — do this before your first commit, or your work will not count
toward your individual mark:

```
git clone https://github.com/SohamAthawale/WIL_RAG_Project.git
cd WIL_RAG_Project
git config user.name  "Your Full Name"
git config user.email "sXXXXXXX@student.rmit.edu.au"
```

**3. Create the virtual environment:**

```
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Three direct dependencies: `numpy`, `requests`, `openpyxl`. Deliberately minimal — no vector
database, no ML frameworks. At 26 chunks a brute-force similarity search is instantaneous, and a
vector DB would cost sprint time while earning nothing.

## Check it works

```
curl -s http://localhost:11434/api/tags
```

Should list both models. If it returns nothing, Ollama is not running — open the app, or run
`ollama serve`.

Then, from `src/`:

```
python -c "import rag; print(rag.store_meta()); print(len(rag.load_store()), 'chunks')"
```

Expect `structure_aware_meta`, 26 chunks. **If it reports 29 chunks or a `vector_store.json`
path, something has regressed** — the pipeline should be reading the snapshot stores.

Full smoke test, one question end to end:

```
python rag.py
```

## How long things take

| Operation | Time |
|---|---|
| Embedding one chunk or query | ~40 ms |
| Generating one answer | 20–35 seconds |
| Full test set through one config (13 questions) | 10–15 minutes |
| Re-ingesting the corpus (26 chunks) | ~2 seconds |

**Generation is the slow part.** Run full sweeps in the background rather than waiting on them,
and use `--skip-b0` on repeat runs — the no-retrieval baseline does not change with retrieval
settings, so re-running it wastes ten minutes each time.

Ollama processes one request at a time per model. If you start a second run while one is going,
it queues rather than failing — which looks like a hang but is not.

## Common problems

**"Connection refused" on port 11434** — Ollama is not running. Check `/api/tags` before blaming
the pipeline code.

**A run appears frozen with no output** — Python buffers stdout when piped. The process is
probably fine; check `~/.ollama/logs/server.log` for request timings to confirm progress.

**Low CPU during generation** — expected on Apple Silicon, where work is offloaded to the GPU.
Not a sign that anything is stuck.

**Results that do not match anyone else's** — check which config you ran. Every output file is
tagged (`b0_vs_rag__structure_aware_meta_k3.json`) and carries a `run_meta` block naming the
snapshot, config, models and k. Numbers from different configs are not comparable.
