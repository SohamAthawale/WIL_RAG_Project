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

---

# Re-run after the pipeline fix (2026-08-20)

**Config:** snapshot `2d8e935b8999`, `structure_aware_meta`, 26 chunks, k=3.
The earlier run used paragraph chunking with no subclass prefix; the generation path
had not been migrated to the snapshot stores, so those results and the retrieval
results described different systems.

## Case 1 — q01 is fixed, including the attribution failure

`subclass_500_student.txt` now retrieves at **rank 1 (0.829)** and rank 3, where it
previously did not appear in the top-3 at all. The answer is correct and the citation
names a document that was genuinely in context.

Both failures recorded earlier for q01 — the retrieval miss and the ungrounded citation
— are resolved by D6 + D9 together.

## Case 2 — q02: cross-subclass confusion fixed, a second failure exposed underneath

All three retrieved chunks are now from `subclass_485_temporary_graduate.txt`, so
**cross-subclass confusion for this question is zero.** The answer is still wrong.

| | Answer | Correct? |
|---|---|---|
| Before | "up to four years" — the **Subclass 482** figure | Wrong, cross-subclass |
| After | "between 1 and 2 years" — the **Second Post-Higher Education Work** stream | Wrong, cross-stream |
| Gold | "Usually between 2 and 3 years" — Post-Higher Education Work stream | — |

Cause, traceable in the snapshot:

- `subclass_485_temporary_graduate::3` — Post-Higher Education Work stream, `Stay: Usually
  between 2 and 3 years`. The correct chunk.
- `subclass_485_temporary_graduate::4` — **Second** Post-Higher Education Work stream,
  `Stay: Between 1 and 2 years`. Its heading contains the literal string "Post-Higher
  Education Work", so it matches the query strongly.

The model took the figure from `::4`.

## Why this matters for the framework

The headline metric would score q02 as a **success** after the fix, because no wrong
subclass was involved. The answer would still send a graduate a figure that is wrong by
a year or more.

Three consequences for the evaluation:

1. **Cross-subclass confusion alone is insufficient** as a correctness measure. It is a
   necessary condition, not a sufficient one. This is direct evidence for the D13
   decision to implement all four definitions rather than commit to one early.
2. **Dimensions 2 (faithfulness) and 3 (attribution correctness) are load-bearing**, not
   optional extras. Both are currently unimplemented, and both would catch this.
3. **The confusion hierarchy has at least two levels** — between subclasses, and between
   streams within a subclass. Whether the chunk prefix should carry the stream as well as
   the subclass is a design decision the team has not yet taken, and it should be run as
   a controlled experiment like D9 rather than applied silently.

No accuracy figures are stated here. The manual grading for this run has not been done.
