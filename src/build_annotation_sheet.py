"""Build the hand-annotation workbook.

Two judgement tasks in one file:
  Sheet 'Answers' - grade the 13 answers produced by B0 and the RAG system.
  Sheet 'Chunks'  - judge chunk relevance per question (the qrels / gold_chunk_ids).

Every judgement column is empty and every dropdown is a choice for the annotator.
This script only assembles and orders the evidence; it makes no judgements.
"""

import json
from pathlib import Path

import numpy as np
import requests
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOTS = ROOT / "data" / "snapshots"
# Judge against the no-metadata store: the D9 prefix changes the embedded text but
# not the underlying passage, so one set of judgements is valid for both variants.
STORE_GLOB = "*/vector_store__structure_aware_nometa.json"
TESTSET = ROOT / "data" / "testset" / "test_set.json"
# Grade the run at the pipeline default (TOP_K in rag.py), not whichever filename
# happens to sort last. k=5 sorts after k=3, so "newest by name" silently picked a
# non-default configuration - the grading has to describe the system as configured.
from rag import DEFAULT_CONFIG, TOP_K
RESULTS_GLOB = f"b0_vs_rag__{DEFAULT_CONFIG}_k{TOP_K}.json"
OUT = ROOT / "results" / "annotation_sheet.xlsx"

OLLAMA = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"

HDR_FILL = PatternFill("solid", fgColor="1F5C8B")
HDR_FONT = Font(color="FFFFFF", bold=True, size=10)
JUDGE_FILL = PatternFill("solid", fgColor="FFF2CC")   # columns the annotator fills
RETRIEVED_FILL = PatternFill("solid", fgColor="E2EBDC")  # chunk was in top-3
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def embed(text: str) -> np.ndarray:
    r = requests.post(OLLAMA, json={"model": EMBED_MODEL, "prompt": text}, timeout=60)
    r.raise_for_status()
    return np.array(r.json()["embedding"])


def style_header(ws, widths: list[int], freeze: str) -> None:
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w
    for cell in ws[1]:
        cell.fill = HDR_FILL
        cell.font = HDR_FONT
        cell.alignment = Alignment(vertical="center", wrap_text=True)
    ws.row_dimensions[1].height = 30
    ws.freeze_panes = freeze


def main() -> None:
    store_path = sorted(SNAPSHOTS.glob(STORE_GLOB))[-1]
    payload = json.loads(store_path.read_text())
    store = payload["chunks"]
    print(f"Judging against snapshot {payload['snapshot_id']} / {payload['config_id']} "
          f"({payload['n_chunks']} chunks)")
    testset = {t["id"]: t for t in json.loads(TESTSET.read_text())}

    runs = sorted((ROOT / "results").glob(RESULTS_GLOB))
    if not runs:
        raise SystemExit(
            f"No run matching {RESULTS_GLOB}.\n"
            f"Generate it: python eval.py --config {DEFAULT_CONFIG} --k {TOP_K}")
    run_payload = json.loads(runs[-1].read_text())
    rows = run_payload["rows"]
    rm = run_payload["run_meta"]
    print(f"Grading run {runs[-1].name}: {rm['config_id']} · k={rm['top_k']}")

    wb = Workbook()

    # ---------------- Sheet 1: how to use ----------------
    ws0 = wb.active
    ws0.title = "How to use"
    guide = [
        ["Group 97 - hand annotation workbook", ""],
        ["", ""],
        ["Two tasks. Yellow columns are yours to fill; everything else is evidence.", ""],
        ["", ""],
        ["SHEET 'Answers'", "13 rows. Grade what the two systems answered."],
        ["", "b0_correct / rag_correct: yes / no / partial - is the answer factually right?"],
        ["", "answerability: what SHOULD have happened vs what did (splits dimension 1)."],
        ["", "wrong_subclass_used: did the answer draw on the wrong visa subclass?"],
        ["", "citation_grounded: was the cited source actually in the retrieved list?"],
        ["", ""],
        ["SHEET 'Chunks'", "377 rows (13 questions x 29 chunks). Judge chunk relevance."],
        ["", "This produces gold_chunk_ids - the qrels. Without it no retrieval"],
        ["", "measure (NDCG, Recall@k, MRR) can be computed at all."],
        ["", ""],
        ["", "Rows are sorted by cosine score within each question, so the likely-"],
        ["", "relevant chunks are at the top of each block. Green rows were actually"],
        ["", "retrieved into the top-3."],
        ["", ""],
        ["RELEVANCE SCALE", "3 = fully answers the question, correct subclass"],
        ["", "2 = correct subclass, partial - needs another chunk to complete"],
        ["", "1 = correct subclass, related context, does not answer"],
        ["", "0 = irrelevant, OR belongs to a different subclass"],
        ["", ""],
        ["", "Graded rather than binary because graded collapses to binary later"],
        ["", "(anything >= 1 is relevant) but binary cannot be expanded without"],
        ["", "re-judging. Judging graded keeps decision D1 open."],
        ["", ""],
        ["wrong_subclass", "Flag chunks that are 0 BECAUSE they belong to another subclass,"],
        ["", "as opposed to merely being off-topic. Keeps open the option of"],
        ["", "scoring them negatively rather than at zero."],
        ["", ""],
        ["SUGGESTED ORDER", "1. Do 'Answers' first - 13 rows, gives you the headline comparison."],
        ["", "2. On 'Chunks', judge the top ~10 rows per question first (130 rows)."],
        ["", "   That covers everything retrieved plus margin, and is enough to"],
        ["", "   compute every measure at k=1,3,5."],
        ["", "3. Extend to the full 377 later if time allows - exhaustive judging"],
        ["", "   avoids pooling bias and is worth claiming in the report."],
        ["", ""],
        ["IMPORTANT", "Put your name in the annotator column. If two people judge the"],
        ["", "same rows independently, the agreement rate is a real methodological"],
        ["", "result worth reporting."],
    ]
    for r in guide:
        ws0.append(r)
    ws0.column_dimensions["A"].width = 22
    ws0.column_dimensions["B"].width = 82
    ws0["A1"].font = Font(bold=True, size=14, color="1F5C8B")
    for row in ws0.iter_rows(min_row=2, max_row=ws0.max_row):
        if row[0].value:
            row[0].font = Font(bold=True, size=10)
        row[1].alignment = Alignment(wrap_text=True, vertical="top")

    # ---------------- Sheet 2: Answers ----------------
    ws1 = wb.create_sheet("Answers")
    headers = [
        "id", "category", "gold_subclass", "question", "gold_answer",
        "b0_answer", "rag_answer", "rag_retrieved_sources",
        "b0_correct", "rag_correct", "answerability",
        "wrong_subclass_used", "citation_grounded", "annotator", "notes",
    ]
    ws1.append(headers)
    for r in rows:
        item = testset.get(r["id"], {})
        ws1.append([
            r["id"], r["category"], str(item.get("subclass", "")), r["question"],
            r["gold_answer"], r["b0_answer"], r["rag_answer"],
            r.get("rag_retrieved_sources", ""),
            "", "", "", "", "", "", "",
        ])
    style_header(ws1, [7, 20, 12, 44, 44, 52, 52, 40, 12, 12, 20, 17, 17, 14, 30], "E2")

    dv_yn = DataValidation(type="list", formula1='"yes,no,partial"', allow_blank=True)
    dv_ans = DataValidation(
        type="list",
        formula1='"answered_correctly,answered_but_wrong,correct_refusal,missed_answerable,over_refused"',
        allow_blank=True,
    )
    dv_sc = DataValidation(type="list", formula1='"yes,no,n/a"', allow_blank=True)
    dv_cite = DataValidation(type="list", formula1='"yes,no,none_given"', allow_blank=True)
    for dv in (dv_yn, dv_ans, dv_sc, dv_cite):
        ws1.add_data_validation(dv)
    last = ws1.max_row
    dv_yn.add(f"I2:J{last}")
    dv_ans.add(f"K2:K{last}")
    dv_sc.add(f"L2:L{last}")
    dv_cite.add(f"M2:M{last}")

    for row in ws1.iter_rows(min_row=2, max_row=last):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
            c.border = BORDER
        for idx in range(8, 15):          # I..O  (judgement + annotator + notes)
            row[idx].fill = JUDGE_FILL

    # ---------------- Sheet 3: Chunks (qrels) ----------------
    ws2 = wb.create_sheet("Chunks")
    ws2.append([
        "question_id", "question", "chunk_id", "source", "chunk_text",
        "cosine_score", "retrieved_top3", "relevance", "wrong_subclass", "annotator",
    ])

    mat = np.array([r["embedding"] for r in store])
    mat_norm = mat / np.linalg.norm(mat, axis=1, keepdims=True)
    retrieved_by_q = {r["id"]: r.get("rag_retrieved_sources", "") for r in rows}

    for qid, item in testset.items():
        qv = embed(item["question"])
        qv = qv / np.linalg.norm(qv)
        sims = mat_norm @ qv
        order = np.argsort(-sims)
        top3_ids = {store[i]["id"] for i in order[:3]}
        for rank, i in enumerate(order, start=1):
            rec = store[i]
            in_top3 = rec["id"] in top3_ids
            text = rec["text"].replace("\n", " ")
            ws2.append([
                qid, item["question"], rec["id"], rec["source"],
                text[:400] + ("..." if len(text) > 400 else ""),
                round(float(sims[i]), 4), "yes" if in_top3 else "",
                "", "", "",
            ])

    style_header(ws2, [12, 44, 34, 32, 78, 12, 13, 12, 15, 14], "C2")

    dv_rel = DataValidation(type="list", formula1='"0,1,2,3"', allow_blank=True)
    dv_ws = DataValidation(type="list", formula1='"yes,no"', allow_blank=True)
    ws2.add_data_validation(dv_rel)
    ws2.add_data_validation(dv_ws)
    last2 = ws2.max_row
    dv_rel.add(f"H2:H{last2}")
    dv_ws.add(f"I2:I{last2}")

    for row in ws2.iter_rows(min_row=2, max_row=last2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
            c.border = BORDER
        if row[6].value == "yes":
            for c in row[:7]:
                c.fill = RETRIEVED_FILL
        for idx in (7, 8, 9):
            row[idx].fill = JUDGE_FILL

    ws2.auto_filter.ref = f"A1:J{last2}"
    ws1.auto_filter.ref = f"A1:O{last}"

    wb.save(OUT)
    print(f"Wrote {OUT}")
    print(f"  Answers sheet: {last - 1} rows to grade")
    print(f"  Chunks sheet : {last2 - 1} rows to judge ({len(testset)} questions x {len(store)} chunks)")


if __name__ == "__main__":
    main()
