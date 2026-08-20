"""Chunk the source documents and embed them with a local Ollama model.
Output is a flat JSON vector store — no external DB needed for this scale."""

import json
import re
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "data" / "documents"
OUT_PATH = ROOT / "data" / "vector_store.json"

OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"

CHUNK_MIN_CHARS = 200


def chunk_text(text: str) -> list[str]:
    """Split on blank lines (paragraph/section boundaries), merging short ones."""
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buffer = ""
    for part in parts:
        buffer = f"{buffer}\n\n{part}".strip() if buffer else part
        if len(buffer) >= CHUNK_MIN_CHARS:
            chunks.append(buffer)
            buffer = ""
    if buffer:
        if chunks:
            chunks[-1] = f"{chunks[-1]}\n\n{buffer}"
        else:
            chunks.append(buffer)
    return chunks


def embed(text: str) -> list[float]:
    resp = requests.post(OLLAMA_URL, json={"model": EMBED_MODEL, "prompt": text}, timeout=60)
    resp.raise_for_status()
    return resp.json()["embedding"]


def main() -> None:
    records = []
    doc_paths = sorted(DOCS_DIR.glob("*.txt"))
    print(f"Found {len(doc_paths)} source documents.")

    for doc_path in doc_paths:
        raw = doc_path.read_text()
        chunks = chunk_text(raw)
        print(f"  {doc_path.name}: {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            records.append(
                {
                    "id": f"{doc_path.stem}::{i}",
                    "source": doc_path.name,
                    "text": chunk,
                    "embedding": embed(chunk),
                }
            )

    OUT_PATH.write_text(json.dumps(records, indent=2))
    print(f"\nWrote {len(records)} chunks with embeddings to {OUT_PATH}")


if __name__ == "__main__":
    main()
