"""Chunk the source documents and embed them into a snapshot-tagged vector store.

Every store is written to data/snapshots/<snapshot_id>/vector_store__<config_id>.json
where snapshot_id is a hash of the corpus itself. If a source document changes, the
snapshot id changes and old relevance judgements are visibly no longer applicable -
which matters because chunk-level qrels (D5) are tied to a specific chunking.

Usage:
    python ingest.py                                     # default: structure_aware
    python ingest.py --strategy paragraph
    python ingest.py --strategy structure_aware --prepend-metadata
    python ingest.py --strategy fixed_size --chunk-size 400 --overlap 0.2
"""

import argparse
import hashlib
import json
from pathlib import Path

import requests

from chunking import STRATEGIES, chunk_document

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "data" / "documents"
SNAPSHOTS = ROOT / "data" / "snapshots"
LEGACY_OUT = ROOT / "data" / "vector_store.json"

OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"


def corpus_snapshot_id(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def config_id(strategy: str, prepend_metadata: bool, params: dict) -> str:
    bits = [strategy]
    for k in sorted(params):
        bits.append(f"{k}{params[k]}")
    bits.append("meta" if prepend_metadata else "nometa")
    return "_".join(str(b) for b in bits)


def embed(text: str) -> list[float]:
    r = requests.post(OLLAMA_URL, json={"model": EMBED_MODEL, "prompt": text}, timeout=60)
    r.raise_for_status()
    return r.json()["embedding"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="structure_aware", choices=sorted(STRATEGIES))
    ap.add_argument("--prepend-metadata", action="store_true",
                    help="D9: embed the subclass identifier into the chunk text")
    ap.add_argument("--chunk-size", type=int)
    ap.add_argument("--overlap", type=float)
    ap.add_argument("--max-chars", type=int)
    ap.add_argument("--min-chars", type=int)
    ap.add_argument("--also-write-legacy", action="store_true",
                    help="also overwrite data/vector_store.json for older scripts")
    args = ap.parse_args()

    params = {k: v for k, v in {
        "chunk_size": args.chunk_size, "overlap": args.overlap,
        "max_chars": args.max_chars, "min_chars": args.min_chars,
    }.items() if v is not None}

    doc_paths = sorted(DOCS_DIR.glob("*.txt"))
    snap = corpus_snapshot_id(doc_paths)
    cfg = config_id(args.strategy, args.prepend_metadata, params)
    out_dir = SNAPSHOTS / snap
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"vector_store__{cfg}.json"

    print(f"snapshot {snap} · config {cfg}")
    print(f"{len(doc_paths)} source documents\n")

    records = []
    for p in doc_paths:
        chunks = chunk_document(p, args.strategy, args.prepend_metadata, **params)
        print(f"  {p.name:44} {len(chunks):>3} chunks")
        for c in chunks:
            records.append({**c, "embedding": embed(c["text"])})

    payload = {
        "snapshot_id": snap,
        "config_id": cfg,
        "strategy": args.strategy,
        "prepend_metadata": args.prepend_metadata,
        "params": params,
        "embed_model": EMBED_MODEL,
        "n_chunks": len(records),
        "chunks": records,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {len(records)} chunks -> {out_path.relative_to(ROOT)}")

    if args.also_write_legacy:
        LEGACY_OUT.write_text(json.dumps(records, indent=2))
        print(f"Also wrote legacy flat store -> {LEGACY_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
