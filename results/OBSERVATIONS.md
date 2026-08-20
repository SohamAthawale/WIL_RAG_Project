# B0 vs RAG — Run Observations (seed test set, 13 questions)

**Run date:** 2026-08-20 · **Snapshot:** `data/vector_store.json`, 29 chunks / 5 documents
**Config:** paragraph chunking, `nomic-embed-text`, cosine, top-k=3, `qwen2.5:7b-instruct`, temp 0.1

> **Scope of this file.** It records *verifiable, traceable facts* about what the two
> configurations did — quoted output, retrieved sources, and the corresponding source-document
> lines. It contains **no correctness grades and no accuracy figures**, because those depend on
> the hand annotation in `annotation_worksheet.csv`, which is not yet done. Interpretation and
> argument for the report are the team's to write.

---

## Case 1 — q01: right answer, wrong provenance

**Question:** How many hours a fortnight can a Subclass 500 student visa holder work while their
course is in session?

**RAG answered:** *"A Subclass 500 Student visa holder can work up to 48 hours per fortnight while
their course is in session. [Source: subclass_500_student.txt]"* — which matches the gold answer.

**But the retrieved context was:**

| Rank | Source | Score |
|---|---|---|
| 1 | `subclass_417_working_holiday.txt` | 0.827 |
| 2 | `subclass_485_temporary_graduate.txt` | 0.815 |
| 3 | `subclass_482_skills_in_demand.txt` | 0.781 |

`subclass_500_student.txt` — the document containing the literal answer at line 19 — **was not
retrieved at all**, despite the question naming "Subclass 500 student visa" explicitly.

Two traceable consequences:

1. **The correct figure was available only via cross-reference.** All three retrieved documents
   carry a "Note:" paragraph that mentions the 500 cap in passing (417 line 32, 482 line 33,
   485 line 42). The right number was reachable by accident of corpus wording, not by retrieval
   working.
2. **The citation names a document that was never in context.** `subclass_500_student.txt` did
   not appear in the prompt, so the source attribution was not grounded in retrieved material.

A single accuracy score records this question as a success.

---

## Case 2 — q02: cross-subclass contamination with a traceable cause

**Question:** What is the maximum stay on a Subclass 485 Temporary Graduate visa in the
Post-Higher Education Work stream?

**RAG answered:** *"…can be up to four years."* — with no source citation.

**Source documents say:**

- `subclass_485_temporary_graduate.txt` line 31: `Stay: Usually between 2 and 3 years, depending
  on qualification. Hong Kong and British National Overseas passport holders may stay for 5 years.`
- `subclass_482_skills_in_demand.txt` lines 15/20: `Stay: Up to 4 years (Hong Kong passport
  holders up to 5 years).`

**Retrieved context:** 485 (0.804), **482 (0.795)**, 485 (0.789).

The figure "4 years" does not occur anywhere in the 485 document. It is the **482** figure, and
the 482 chunk was retrieved at rank 2 for a question that names Subclass 485 explicitly.

This is the cross-subclass confusion failure mode (dimension 5) with a complete causal chain from
retrieval to output.

---

## Case 3 — B0 stated superseded figures

Where B0 (bare LLM, no retrieval) gave a work-hours figure, it did not match the current corpus:

| Q | B0 figure | Corpus figure |
|---|---|---|
| q01 | "up to 20 hours per week while classes are in session" | 48 hours per fortnight (`subclass_500_student.txt:19`) |
| q07 | "40 hours" | 48 hours per fortnight |

Both correspond to caps that predate the current rule. RAG returned 48 hours for both questions.
This is the temporal-correctness dimension (8), and it is the clearest argument for retrieval in
this domain.

---

## Mechanical tallies (heuristic — not grades)

Refusal-marker detection on the three `refusal_required` questions (q10, q11, q12):

| Config | Refusal markers detected |
|---|---|
| B0 | 1 of 3 (q10 only) |
| RAG | 3 of 3 |

**Caveat, and why a human must still grade these:** the same detector fires on q07 and q09, where
refusal was *not* expected. On q07 the RAG output both hedged *and* supplied "48 hours" — a hedged
partial answer, not a refusal. The detector cannot tell those apart. Treat every `check_*` column
in the worksheet as a pointer, not a verdict.

---

## Next step

Hand-annotate `results/annotation_worksheet.csv` — fill `b0_correct_manual` and
`rag_correct_manual` (yes / no / partial) and record the annotator. Two members annotating
independently, then reconciling, gives an inter-annotator agreement figure worth reporting.

Until that is done, **no accuracy number for this run exists** and none should be quoted.
