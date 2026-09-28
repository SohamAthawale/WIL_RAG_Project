"""Build the pooled judging sheet for the 88 unjudged questions.

Exhaustive judging of 100 questions against 24 chunks is 2,400 rows, of which 2,088 are
new. That is roughly seven times T02 and nobody is resourced for it. Decision D2 already
anticipated this: exhaustive while the set is small, pooling once it grows.

Pooling is the standard Cranfield answer. Judge only the chunks some system actually
retrieved in its top-k, and treat everything unretrieved as not relevant. The known cost
is that recall is then measured against the pool rather than the corpus -- a relevant
chunk no system surfaced is never judged, so recall is an upper bound. At 24 chunks the
bias is small, and it belongs in the report rather than being discovered by a marker.

The pool is the union over three retrievers at top-k, so a chunk any one of them ranks
highly gets judged even when the other two miss it. Pooling over a single retriever would
bake that retriever's blind spots into the ground truth it is then scored against.

The 12 already-judged questions are excluded: they were judged exhaustively and re-judging
them would void qrels five results already depend on.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import rag  # noqa: E402
from baseline import retrieval_bm25  # noqa: E402

POOL_K = 5           # per retriever, before the union
OUT = ROOT / "results" / "pooled_judging_sheet.xlsx"
TESTSET = ROOT / "data" / "testset" / "test_set.json"

HDR = PatternFill("solid", fgColor="1F3864")
JUDGE = PatternFill("solid", fgColor="FFF2CC")
PEOPLE = ["Soham", "Glory", "Heet", "Manoj", "Vaishnavi", "Hitesh"]


def rrf(orders: list[np.ndarray], n: int, k: int = 60) -> np.ndarray:
    score = np.zeros(n)
    for order in orders:
        for rank, idx in enumerate(order, start=1):
            score[idx] += 1.0 / (k + rank)
    return np.argsort(-score, kind="stable")


def main() -> None:
    testset = json.loads(TESTSET.read_text())
    store = rag.load_store()
    chunks = [{k: v for k, v in c.items() if k != "embedding"} for c in store]
    mat = np.array([c["embedding"] for c in store])
    mat = mat / np.linalg.norm(mat, axis=1, keepdims=True)
    bm = retrieval_bm25.BM25(chunks)

    todo = [t for t in testset if not t.get("gold_chunk_ids")]
    print(f"{len(todo)} unjudged questions of {len(testset)}")

    rows = []
    for t in todo:
        qv = rag.embed(t["question"])
        qv = qv / np.linalg.norm(qv)
        o_dense = np.argsort(-(mat @ qv), kind="stable")
        o_bm25 = np.argsort(-bm.scores(t["question"]), kind="stable")
        o_rrf = rrf([o_dense, o_bm25], len(chunks))

        pool: dict[int, list[str]] = {}
        for name, order in (("dense", o_dense), ("bm25", o_bm25), ("hybrid_rrf", o_rrf)):
            for idx in order[:POOL_K]:
                pool.setdefault(int(idx), []).append(name)

        for idx, retrievers in sorted(pool.items()):
            c = chunks[idx]
            rows.append([
                t["id"], t["category"], t["question"], t["gold_answer"],
                c["id"], c["source"], c["text"][:600],
                "+".join(retrievers), "", "",
            ])

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Pool"
    headers = ["question_id", "category", "question", "gold_answer",
               "chunk_id", "source", "chunk_text", "retrieved_by",
               "relevance", "annotator"]
    ws.append(headers)
    for n, cell in enumerate(ws[1], start=1):
        cell.fill = HDR
        cell.font = Font(bold=True, color="FFFFFF", size=10)
    for r in rows:
        ws.append(r)
    for row in ws.iter_rows(min_row=2, min_col=9, max_col=10):
        for cell in row:
            cell.fill = JUDGE
    for col, w in zip("ABCDEFGHIJ", (12, 20, 46, 46, 34, 30, 70, 18, 11, 14)):
        ws.column_dimensions[col].width = w
    ws.freeze_panes = "A2"
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)

    # Blocks are whole questions, never row ranges: a judge holds one question in mind
    # across its pooled chunks, and splitting mid-question makes that impossible.
    guide = wb.create_sheet("How to judge")
    per = (len(todo) + len(PEOPLE) - 1) // len(PEOPLE)
    guide.append(["Pooled relevance judging"])
    guide.append([])
    guide.append(["Scale", "0 = not relevant · 1 = related · 2 = useful · 3 = directly answers"])
    guide.append(["Fill", "The two yellow columns only: relevance, annotator."])
    guide.append(["Rule", "Judge whole questions. Do not skip to the chunks you recognise."])
    guide.append(["Pool", f"Union of top-{POOL_K} from dense, BM25 and RRF. Unretrieved chunks count as 0."])
    guide.append([])
    guide.append(["Person", "Questions", "Rows"])
    for i, name in enumerate(PEOPLE):
        block = [t["id"] for t in todo[i * per:(i + 1) * per]]
        if not block:
            continue
        n_rows = sum(1 for r in rows if r[0] in set(block))
        guide.append([name, f"{block[0]}–{block[-1]} ({len(block)} questions)", n_rows])
    guide.column_dimensions["A"].width = 14
    guide.column_dimensions["B"].width = 42
    guide.column_dimensions["C"].width = 10
    guide["A1"].font = Font(bold=True, size=13)

    wb.save(OUT)
    print(f"{len(rows)} rows to judge, vs {len(todo) * len(chunks)} exhaustive "
          f"({100 * len(rows) / (len(todo) * len(chunks)):.0f}%)")
    print(f"mean pool size: {len(rows) / len(todo):.1f} chunks per question")
    print(f"-> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
