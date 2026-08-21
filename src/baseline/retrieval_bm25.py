"""Okapi BM25 sparse retrieval — the lexical baseline (B1).

k1=1.2 and b=0.75 are the conventional Okapi defaults. Implemented directly in
numpy: library implementations pull in a JVM or a heavy index, neither of which
earns anything on a corpus this size.

Interface matches the dense retriever in rag.py: retrieve(query, store, k).
"""

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
SNAPSHOTS = ROOT / "data" / "snapshots"
DEFAULT_CONFIG = "structure_aware_meta"

K1 = 1.2
B = 0.75

# Alphanumeric tokens, lowercased. Deliberately no stemming and no stopword
# removal: visa identifiers ("485", "8547", "48") are exactly the tokens that
# matter here, and aggressive normalisation is what blurs them.
TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, store: list[dict], k1: float = K1, b: float = B):
        self.store = store
        self.k1 = k1
        self.b = b
        self.docs = [tokenize(r["text"]) for r in store]
        self.doc_len = np.array([len(d) for d in self.docs], dtype=float)
        self.avgdl = float(self.doc_len.mean())
        self.N = len(self.docs)

        vocab: dict[str, int] = {}
        for d in self.docs:
            for t in set(d):
                vocab.setdefault(t, len(vocab))
        self.vocab = vocab

        # term-frequency matrix, docs x vocab
        self.tf = np.zeros((self.N, len(vocab)), dtype=float)
        for i, d in enumerate(self.docs):
            for t in d:
                self.tf[i, vocab[t]] += 1.0

        df = (self.tf > 0).sum(axis=0)
        # Robertson/Sparck Jones IDF with the +1 smoothing that keeps it non-negative
        self.idf = np.log(1.0 + (self.N - df + 0.5) / (df + 0.5))

        # denominator length-normalisation term, per document
        self.norm = self.k1 * (1.0 - self.b + self.b * self.doc_len / self.avgdl)

    def scores(self, query: str) -> np.ndarray:
        cols = [self.vocab[t] for t in tokenize(query) if t in self.vocab]
        if not cols:
            return np.zeros(self.N)
        total = np.zeros(self.N)
        for c in cols:
            f = self.tf[:, c]
            total += self.idf[c] * (f * (self.k1 + 1.0)) / (f + self.norm)
        return total


_cached: BM25 | None = None


def retrieve(query: str, store: list[dict], k: int = 3) -> list[dict]:
    """Same signature and return shape as rag.retrieve, so eval can swap them."""
    global _cached
    if _cached is None or _cached.store is not store:
        _cached = BM25(store)
    s = _cached.scores(query)
    order = np.argsort(-s)[:k]
    return [
        {"score": float(s[i]), **{kk: vv for kk, vv in store[i].items() if kk != "embedding"}}
        for i in order
    ]


if __name__ == "__main__":
    path = sorted(SNAPSHOTS.glob(f"*/vector_store__{DEFAULT_CONFIG}.json"))[-1]
    payload = json.loads(path.read_text())
    store = payload["chunks"]
    print(f"snapshot {payload['snapshot_id']} / {payload['config_id']}")
    bm = BM25(store)
    print(f"BM25 index: {bm.N} chunks, {len(bm.vocab)} terms, avgdl={bm.avgdl:.1f}, k1={K1}, b={B}\n")
    for q in [
        "How many hours a fortnight can a Subclass 500 student visa holder work?",
        "What is the maximum stay on a Subclass 485 Temporary Graduate visa?",
        "What is condition 8547?",
    ]:
        print(f"Q: {q}")
        for h in retrieve(q, store, 3):
            print(f"   {h['score']:6.3f}  {h['source']:42} {h['text'][:52].replace(chr(10),' ')}...")
        print()
