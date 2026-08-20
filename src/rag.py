"""Retrieval + generation over the local vector store, plus a bare-LLM (B0)
baseline with no retrieval, for the RAG-vs-no-RAG comparison."""

import json
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOTS = ROOT / "data" / "snapshots"

# The generation path reads the same snapshot stores as the retrieval experiments.
# Anything else silently compares systems built on different chunkings.
DEFAULT_CONFIG = "structure_aware_meta"

OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
EMBED_MODEL = "nomic-embed-text"
GEN_MODEL = "qwen2.5:7b-instruct"

TOP_K = 3

RAG_SYSTEM_PROMPT = (
    "You are a visa information assistant for Australian temporary visa holders. "
    "Answer ONLY using the CONTEXT provided below — do not use any outside knowledge. "
    "Every answer must state which visa subclass it applies to, since rules differ by subclass. "
    "If the context does not contain the answer, say exactly: "
    "\"I don't have enough information in my sources to answer that.\" "
    "If the question asks for a prediction about a specific case, personal advice, or a legal/immigration "
    "opinion (e.g. whether an application will be approved, whether to accept a job offer), decline and say: "
    "\"That's a case-specific question — please contact a registered migration agent or the Department of "
    "Home Affairs directly.\" "
    "Cite the source document name for any fact you state."
)

B0_SYSTEM_PROMPT = "You are a helpful assistant. Answer the user's question as best you can."


def resolve_store(config: str = DEFAULT_CONFIG) -> Path:
    matches = sorted(SNAPSHOTS.glob(f"*/vector_store__{config}.json"))
    if not matches:
        raise FileNotFoundError(
            f"No snapshot for config {config!r} under {SNAPSHOTS}. "
            f"Run: python ingest.py --strategy structure_aware --prepend-metadata"
        )
    return matches[-1]


def load_store(config: str = DEFAULT_CONFIG) -> list[dict]:
    return json.loads(resolve_store(config).read_text())["chunks"]


def store_meta(config: str = DEFAULT_CONFIG) -> dict:
    """Provenance for the results file - which snapshot and config produced a run."""
    p = json.loads(resolve_store(config).read_text())
    return {
        "snapshot_id": p["snapshot_id"],
        "config_id": p["config_id"],
        "strategy": p["strategy"],
        "prepend_metadata": p["prepend_metadata"],
        "n_chunks": p["n_chunks"],
        "embed_model": p["embed_model"],
        "gen_model": GEN_MODEL,
        "top_k": TOP_K,
    }


def embed(text: str) -> np.ndarray:
    resp = requests.post(OLLAMA_EMBED_URL, json={"model": EMBED_MODEL, "prompt": text}, timeout=60)
    resp.raise_for_status()
    return np.array(resp.json()["embedding"])


def retrieve(query: str, store: list[dict], k: int = TOP_K) -> list[dict]:
    q_vec = embed(query)
    scored = []
    for rec in store:
        v = np.array(rec["embedding"])
        sim = float(np.dot(q_vec, v) / (np.linalg.norm(q_vec) * np.linalg.norm(v)))
        scored.append((sim, rec))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{"score": s, **{k: v for k, v in r.items() if k != "embedding"}} for s, r in scored[:k]]


def chat(system_prompt: str, user_prompt: str) -> str:
    resp = requests.post(
        OLLAMA_CHAT_URL,
        json={
            "model": GEN_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.1},
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def answer_rag(question: str, store: list[dict], k: int = TOP_K) -> dict:
    hits = retrieve(question, store, k)
    context = "\n\n---\n\n".join(f"[Source: {h['source']}]\n{h['text']}" for h in hits)
    prompt = f"CONTEXT:\n{context}\n\nQUESTION: {question}"
    answer = chat(RAG_SYSTEM_PROMPT, prompt)
    return {"answer": answer, "retrieved": [{"source": h["source"], "score": round(h["score"], 3)} for h in hits]}


def answer_b0(question: str) -> dict:
    answer = chat(B0_SYSTEM_PROMPT, question)
    return {"answer": answer, "retrieved": []}


if __name__ == "__main__":
    store = load_store()
    print("store:", store_meta())
    q = "How many hours a fortnight can I work on my visa?"
    print("=== B0 (no retrieval) ===")
    print(answer_b0(q)["answer"])
    print("\n=== RAG ===")
    result = answer_rag(q, store)
    print(result["answer"])
    print("\nRetrieved:", result["retrieved"])
