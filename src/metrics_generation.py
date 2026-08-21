"""Text-similarity measures for generated answers.

Both compare an answer to a reference answer as TEXT. Neither measures whether
the answer is true. That distinction matters here: an answer can share almost
all its vocabulary with the reference and still be wrong on the one figure that
determines whether someone breaches a visa condition.

Report these alongside the correctness judgements, and report where they
disagree - the disagreement is the interesting part.
"""

import re
from collections import Counter

import numpy as np
import requests

OLLAMA = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"

# Printed wherever the proxy appears, so it is never mistaken for BERTScore.
PROXY_LABEL = "cosine_proxy (NOT BERTScore)"

TOKEN_RE = re.compile(r"[a-z0-9]+")
_cache: dict[str, np.ndarray] = {}


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def rouge_1(candidate: str, reference: str) -> dict[str, float]:
    """Unigram overlap between candidate and reference.

    Uses clipped counts: a word occurring twice in the candidate and once in the
    reference counts once. Without clipping, repeating a word inflates the score.
    """
    cand, ref = tokenize(candidate), tokenize(reference)
    if not cand or not ref:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    overlap = sum((Counter(cand) & Counter(ref)).values())
    precision = overlap / len(cand)
    recall = overlap / len(ref)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def _embed(text: str) -> np.ndarray:
    if text not in _cache:
        r = requests.post(OLLAMA, json={"model": EMBED_MODEL, "prompt": text}, timeout=60)
        r.raise_for_status()
        v = np.array(r.json()["embedding"])
        _cache[text] = v / np.linalg.norm(v)
    return _cache[text]


def embedding_cosine_proxy(candidate: str, reference: str) -> float:
    """Cosine similarity between sentence embeddings of the two texts.

    This is NOT BERTScore. BERTScore does greedy token-level matching between
    contextual embeddings and reports precision/recall/F1. This is a single
    cosine over whole-text embeddings from the retrieval model already installed.

    It is a defensible stand-in for semantic similarity, but it must be labelled
    as a proxy in every table it appears in - see PROXY_LABEL. Reporting it under
    the name BERTScore would misrepresent the evaluation.
    """
    if not (candidate or "").strip() or not (reference or "").strip():
        return 0.0
    return round(float(_embed(candidate) @ _embed(reference)), 4)


if __name__ == "__main__":
    # Unit tests with hand-worked expectations.
    def check(name, got, want, tol=1e-4):
        ok = abs(got - want) < tol
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: got {got}, want {want}")
        return ok

    print("rouge_1")
    r = rouge_1("the cat sat", "the cat sat")
    passed = all([
        check("identical -> precision 1.0", r["precision"], 1.0),
        check("identical -> recall 1.0", r["recall"], 1.0),
        check("identical -> f1 1.0", r["f1"], 1.0),
    ])
    # candidate 2 tokens, both in reference; reference 6 tokens
    r = rouge_1("the cat", "the cat sat on the mat")
    passed &= check("subset -> precision 2/2", r["precision"], 1.0)
    passed &= check("subset -> recall 2/6", r["recall"], round(2 / 6, 4), 1e-3)
    # clipping: 'cat cat' has cat twice, reference once -> overlap 1, not 2
    r = rouge_1("cat cat", "cat dog")
    passed &= check("clipped counts -> precision 1/2", r["precision"], 0.5)
    r = rouge_1("", "anything")
    passed &= check("empty candidate -> f1 0", r["f1"], 0.0)

    print("\nembedding_cosine_proxy")
    try:
        p = embedding_cosine_proxy("the cat sat on the mat", "the cat sat on the mat")
        passed &= check("identical -> ~1.0", p, 1.0, tol=1e-3)
        near = embedding_cosine_proxy("a student may work 48 hours a fortnight",
                                      "students can work up to 48 hours per fortnight")
        far = embedding_cosine_proxy("a student may work 48 hours a fortnight",
                                     "the weather in Melbourne is mild")
        ok = near > far
        print(f"  {'PASS' if ok else 'FAIL'}  paraphrase ({near}) scores above unrelated ({far})")
        passed &= ok
        print(f"\n  label used in output: {PROXY_LABEL}")
    except Exception as e:
        print(f"  SKIP  embedding tests - Ollama unavailable: {e}")

    print(f"\n{'ALL TESTS PASSED' if passed else 'FAILURES ABOVE'}")
