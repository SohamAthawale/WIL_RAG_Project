"""T09 — chunking strategy sweep.

Walert never faced this decision: their knowledge base was an FAQ, already
segmented one entry per answer. Our sources are unstructured government pages,
so chunking is a live design variable and an extension of their method.

Sweeps strategy x size x overlap x metadata, scores each with document-level
retrieval measures, and tracks one specific hazard: whether the condition 8547
rule ends up separated from its list of exemptions. A chunk stating a rule
without its exemptions is not merely incomplete, it is dangerous.

Sweep snapshots are written to data/sweeps/ (gitignored). They are large and
rebuildable - the CSV records the exact parameters for any config, so any row
can be regenerated with one ingest command.
"""

import csv
import json
from pathlib import Path

import requests

from chunking import chunk_document
from eval_configs import evaluate

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "data" / "documents"
SWEEPS = ROOT / "data" / "sweeps"
TESTSET = ROOT / "data" / "testset" / "test_set.json"
OUT = ROOT / "results" / "chunking_sweep.csv"

OLLAMA = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"

# The hazard: this rule and this exemption list must stay in the same chunk.
RULE = "maximum of 6 months"
EXEMPTIONS = "Exemptions to condition 8547"

GRID = (
    [dict(strategy="fixed_size", chunk_size=s, overlap=o)
     for s in (400, 600, 1000) for o in (0.0, 0.1, 0.2)]
    + [dict(strategy="paragraph", min_chars=m) for m in (150, 200, 400)]
    + [dict(strategy="structure_aware", max_chars=m) for m in (600, 900, 1400)]
    + [dict(strategy="sentence_window", window=w) for w in (3, 5)]
)


def embed(text: str) -> list[float]:
    r = requests.post(OLLAMA, json={"model": EMBED_MODEL, "prompt": text}, timeout=60)
    r.raise_for_status()
    return r.json()["embedding"]


def hazard_state(chunks: list[dict]) -> str:
    """Did the 8547 rule stay with its exemptions?"""
    rule_ids = {c["id"] for c in chunks if RULE in c["text"]}
    exempt_ids = {c["id"] for c in chunks if EXEMPTIONS in c["text"]}
    if not rule_ids or not exempt_ids:
        return "absent"
    return "together" if rule_ids & exempt_ids else "SPLIT"


def build(params: dict, prepend: bool) -> tuple[Path, dict]:
    strategy = params["strategy"]
    kw = {k: v for k, v in params.items() if k != "strategy"}
    cfg = "_".join([strategy] + [f"{k}{v}" for k, v in sorted(kw.items())]
                   + ["meta" if prepend else "nometa"])
    path = SWEEPS / f"vector_store__{cfg}.json"

    chunks = []
    for doc in sorted(DOCS.glob("*.txt")):
        chunks.extend(chunk_document(doc, strategy, prepend, **kw))

    haz = hazard_state(chunks)
    if not path.exists():
        records = [{**c, "embedding": embed(c["text"])} for c in chunks]
        path.write_text(json.dumps({
            "snapshot_id": "sweep", "config_id": cfg, "strategy": strategy,
            "prepend_metadata": prepend, "params": kw,
            "embed_model": EMBED_MODEL, "n_chunks": len(records), "chunks": records,
        }))
    lens = sorted(len(c["text"]) for c in chunks)
    return path, {
        "config_id": cfg, "strategy": strategy, "prepend_metadata": prepend,
        "params": json.dumps(kw), "n_chunks": len(chunks),
        "median_chars": lens[len(lens) // 2], "max_chars_actual": lens[-1],
        "cond8547": haz,
    }


def main() -> None:
    SWEEPS.mkdir(parents=True, exist_ok=True)
    testset = json.loads(TESTSET.read_text())
    rows = []
    total = len(GRID) * 2
    i = 0
    for params in GRID:
        for prepend in (False, True):
            i += 1
            path, meta = build(params, prepend)
            res = evaluate(path, testset)
            print(f"[{i}/{total}] {meta['config_id']:52} "
                  f"{meta['n_chunks']:>3} chunks  8547:{meta['cond8547']}")
            for retriever, m in res["systems"].items():
                rows.append({**meta, "retriever": retriever, **m})

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows ({total} configs x 3 retrievers) to {OUT.name}")
    print(f"Scored on {res['n_questions']} questions with a single gold document.")


if __name__ == "__main__":
    main()
