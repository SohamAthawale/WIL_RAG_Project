#!/usr/bin/env bash
# Command dispatch for the evaluation harness.
#
# Three tiers, and the boundary between them is worth being precise about, because
# it is not the one you would guess:
#
#   offline   needs nothing. Refusal scoring and the comparison table read stored
#             answers and stored grades, so they run with the network cut.
#   embed     needs Ollama with nomic-embed-text (~275 MB). Retrieval metrics,
#             grounding and the confusion counters all replay retrieval against the
#             stored corpus, so they have to embed the query. No generation model,
#             and no GPU required for this tier.
#   generate  needs qwen2.5:7b-instruct (~4.7 GB) as well. Only answering questions.
set -euo pipefail

APP=/app
OUT=${OUT_DIR:-/out}
PY=python3
export PYTHONPATH="$APP/src:${PYTHONPATH:-}"

EMBED_MODEL=nomic-embed-text
GEN_MODEL=qwen2.5:7b-instruct

log()  { printf '\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33m    %s\033[0m\n' "$*"; }
die()  { printf '\033[31merror: %s\033[0m\n' "$*" >&2; exit 1; }

wait_for_ollama() {
  local url="${OLLAMA_HOST:-http://ollama:11434}" tries=${OLLAMA_WAIT:-90}
  for ((i=1; i<=tries; i++)); do
    curl -sf "$url/api/tags" >/dev/null 2>&1 && { log "Ollama is up at $url"; return 0; }
    [[ $i -eq 1 ]] && log "waiting for Ollama at $url"
    sleep 2
  done
  die "Ollama did not answer at $url after $((tries*2))s.
  Start it:   docker compose up -d ollama
  Pull models: docker compose run --rm harness models"
}

# Ask only for the models the command actually uses, so the scoring tier does not
# demand a 4.7 GB download it never touches.
require_models() {
  local url="${OLLAMA_HOST:-http://ollama:11434}" tags m
  tags=$(curl -sf "$url/api/tags" || echo '')
  for m in "$@"; do
    grep -q "${m%%:*}" <<<"$tags" || die "model '$m' is not present.
  Pull it once:  docker compose run --rm harness models"
  done
}

need_embed()    { wait_for_ollama; require_models "$EMBED_MODEL"; }
need_generate() { wait_for_ollama; require_models "$EMBED_MODEL" "$GEN_MODEL"; }

collect() {
  mkdir -p "$OUT"
  local f
  for f in "$@"; do
    [[ -e "$APP/results/$f" ]] && cp -f "$APP/results/$f" "$OUT/" && echo "  -> $OUT/$f"
  done
  return 0
}

run_suites() {
  local fail=0 t
  for t in "$@"; do
    echo; log "$(basename "$t")"
    "$PY" "$t" || fail=1
  done
  echo
  [[ $fail -eq 0 ]] || die "a suite failed"
  log "all suites passed"
}

cmd=${1:-tests}; shift || true

case "$cmd" in

  tests)
    need_embed
    log "all four control suites"
    warn "test_confusion and test_grounding replay retrieval, so they embed queries"
    run_suites "$APP"/tests/test_*.py
    ;;

  tests-offline)
    log "the suites that need nothing at all (network can be cut)"
    run_suites "$APP/tests/test_metrics_retrieval.py" "$APP/tests/test_refusal.py"
    ;;

  score)
    need_embed
    log "scoring the stored runs"
    cd "$APP/src"
    "$PY" metrics_retrieval.py
    "$PY" refusal.py
    "$PY" grounding.py
    "$PY" build_comparison.py
    log "collecting into $OUT"
    collect retrieval_metrics.json answerability.json grounding.json \
            full_comparison.csv full_comparison.md
    ;;

  score-offline)
    log "only the scoring that needs nothing (refusal and the comparison table)"
    cd "$APP/src"
    "$PY" refusal.py
    "$PY" build_comparison.py
    collect answerability.json full_comparison.csv full_comparison.md
    ;;

  confusion)
    need_embed
    log "cross-subclass confusion counters"
    cd "$APP/src" && "$PY" confusion.py
    collect confusion_counters.json
    ;;

  reproduce)
    need_embed
    log "re-deriving the scoring outputs, then diffing against the published copies"
    cd "$APP/src"
    "$PY" metrics_retrieval.py >/dev/null
    "$PY" refusal.py           >/dev/null
    "$PY" grounding.py         >/dev/null
    "$PY" build_comparison.py  >/dev/null
    cd "$APP" && "$PY" - <<'PYEOF'
import json, sys
from pathlib import Path

# Every scoring output is expected to match field for field. Ranking sorts are stable
# and generation is pinned, so a difference here is a real change, not machine noise.
FILES = ["retrieval_metrics.json", "answerability.json", "grounding.json"]
SKIP  = ("generated_at", "hostname", "python", "timestamp")

def leaves(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from leaves(v, f"{p}.{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from leaves(v, f"{p}[{i}]")
    else:
        yield p, o

def load(d, f):
    return dict(leaves(json.loads((d / f).read_text())))

now, was = Path("/app/results"), Path("/baseline-results")
bad = 0

for f in FILES:
    if not (was / f).exists():
        print("  --       %s: no published copy to compare against" % f)
        continue
    a, b = load(now, f), load(was, f)
    diff = []
    for k in a.keys() | b.keys():
        if any(s in k for s in SKIP):
            continue
        x, y = a.get(k), b.get(k)
        if isinstance(x, float) and isinstance(y, float) and abs(x - y) < 1e-9:
            continue
        if x != y:
            diff.append(k)
    if diff:
        bad += 1
        print("  DIFFERS  %s  (%d field(s))" % (f, len(diff)))
        for k in sorted(diff)[:8]:
            print("             %s: published=%r -> rebuilt=%r" % (k, b.get(k), a.get(k)))
    else:
        print("  matches  %s" % f)

print()
sys.exit(1 if bad else 0)
PYEOF
    log "every scoring output reproduces from the stored runs"
    ;;

  retrieve)
    [[ $# -ge 1 ]] || die 'usage: retrieve "your question"'
    need_embed
    log "dense vs BM25 vs reciprocal rank fusion, side by side"
    cd "$APP/src" && "$PY" - "$*" <<'PYEOF'
import sys, rag
from baseline import retrieval_bm25 as bm

q, K, RRF_K = sys.argv[1], 5, 60
store = rag.load_store()

dense  = [h["id"] for h in rag.retrieve(q, store, K)]
sparse = [h["id"] for h in bm.retrieve(q, store, K)]

ranks = {}
for lst in (dense, sparse):
    for pos, cid in enumerate(lst, start=1):
        ranks[cid] = ranks.get(cid, 0) + 1.0 / (RRF_K + pos)
fused = [c for c, _ in sorted(ranks.items(), key=lambda kv: kv[1], reverse=True)][:K]

# Every chunk opens with the same "[Subclass N ...]" metadata prefix, so a naive
# snippet shows the prefix and nothing that distinguishes one result from another.
import re as _re
def _snip(t):
    return _re.sub(r"^\s*\[[^\]]*\]\s*", "", t).replace("\n", " ")[:64]
info = {r["id"]: (r["source"], _snip(r["text"])) for r in store}
for name, ids in (("dense (cosine)", dense),
                  ("BM25 (k1=1.2, b=0.75)", sparse),
                  (f"RRF (k={RRF_K})", fused)):
    print(f"\n{name}")
    for i, cid in enumerate(ids, 1):
        src, snip = info[cid]
        print(f"  {i}. {src:<38} {snip}...")
PYEOF
    ;;

  ask)
    [[ $# -ge 1 ]] || die 'usage: ask "your question"'
    need_generate
    cd "$APP/src" && "$PY" - "$*" <<'PYEOF'
import sys, rag
q = sys.argv[1]
store, meta = rag.load_store(), rag.store_meta()
print(f"snapshot {meta['snapshot_id']} · config {meta['config_id']} · k={meta['top_k']} · "
      f"{meta['gen_model']} · temperature {rag.GEN_TEMPERATURE} seed {rag.GEN_SEED}\n")
r = rag.answer_rag(q, store)
print(r["answer"], "\n\nretrieved:")
for h in r["retrieved"]:
    print(f"  {h['score']:.3f}  {h['source']}")
PYEOF
    ;;

  compare)
    [[ $# -ge 1 ]] || die 'usage: compare "your question"'
    need_generate
    cd "$APP/src" && "$PY" - "$*" <<'PYEOF'
import sys, rag
q = sys.argv[1]
store = rag.load_store()
print("=== B0 — no retrieval ===\n")
print(rag.answer_b0(q)["answer"], "\n")
print("=== B2 — dense retrieval ===\n")
r = rag.answer_rag(q, store)
print(r["answer"], "\n")
print("retrieved:", ", ".join(f"{h['source']} ({h['score']:.3f})" for h in r["retrieved"]))
PYEOF
    ;;

  eval)
    need_generate
    log "full generation run over the test collection — this is the slow one"
    cd "$APP/src" && "$PY" eval.py "$@"
    mkdir -p "$OUT"
    cp -f "$APP"/results/b0_vs_rag__*.json "$APP"/results/b0_vs_rag__*.csv "$OUT/" 2>/dev/null || true
    log "runs collected into $OUT"
    ;;

  models)
    wait_for_ollama
    for m in "$EMBED_MODEL" "$GEN_MODEL"; do
      log "pulling $m"
      curl -S --fail-with-body -X POST "${OLLAMA_HOST}/api/pull" \
           -d "{\"model\":\"$m\",\"stream\":false}" && echo
    done
    log "models ready"
    ;;

  manifest)
    cd "$APP/src" && "$PY" manifest.py --verify
    ;;

  shell) exec /bin/bash ;;

  help|--help|-h)
    cat <<'TXT'
Evaluation harness — commands

  needs nothing (run with --no-deps, or --network none)
    tests-offline   the two suites that touch no service
    score-offline   refusal scoring and the comparison table
    manifest        verify corpus hashes against the manifest

  needs Ollama + nomic-embed-text (~275 MB, no GPU required)
    tests           all four control suites
    score           retrieval metrics, refusal, grounding, comparison table
    reproduce       re-derive those, then diff against the published copies
    confusion       cross-subclass confusion counters
    retrieve "..."  dense vs BM25 vs RRF for one question, side by side

  needs qwen2.5:7b-instruct as well (~4.7 GB, wants the GPU)
    ask "..."       answer one question through the RAG pipeline
    compare "..."   the same question with and without retrieval
    eval [--k N]    full generation run over the test collection (slow)

  models            pull both models into the Ollama volume (once)
  shell             interactive shell inside the image

Anything worth keeping is written to /out, which compose maps to ./out.
TXT
    ;;

  *) die "unknown command '$cmd'. Try: help" ;;
esac
