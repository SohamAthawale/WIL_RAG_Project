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

---

# T09 — Chunking strategy sweep (2026-08-21)

34 configurations (4 strategies × sizes/overlaps × with and without the subclass prefix),
each scored with three retrievers. 102 rows in `results/chunking_sweep.csv`.

**Scored on 8 questions** — those with a single gold document. Document-level, not chunk-level;
chunk-level NDCG needs the qrels. Differences below ~0.06 in MRR are under one question and
should be read as noise.

## Finding 1 — the condition 8547 hazard is near-universal

**30 of 34 configurations separate the 8547 rule from its exemption list.** Only
`sentence_window` keeps them together, in all 4 of its configurations.

| Strategy | Configs splitting rule from exemptions |
|---|---|
| fixed_size | 18 of 18 |
| paragraph | 6 of 6 |
| structure_aware | 6 of 6 |
| sentence_window | 0 of 4 |

This is a structural property, not a statistical one — it does not depend on the 8-question
sample and is not noise.

Why `sentence_window` survives: the exemption list is bullets with no sentence-ending
punctuation, so sentence splitting treats the whole list as a single unit, and the rule sentence
that follows falls inside the same 3-sentence window. It preserves the grouping by accident of
formatting rather than by design.

**Raising the size cap does not fix it.** At `max_chars=1400` the exemption chunk is already
1311 characters; adding the 134-character rule sentence would total 1445 and breach the cap, so
the packer splits it. The chunk that results — `whm_6_month_work_limitation::1`, 134 characters —
reads:

> *"If work does not fall within an exemption, you can only work for the same employer for a
> maximum of 6 months while holding a WHM visa."*

A rule with no exemptions attached. Retrieved alone, it would tell a fruit picker they must stop
after six months when in fact plant and animal cultivation is exempt.

**The fix is not a bigger cap.** `chunking.py` already binds a paragraph ending in `:` *forward*
to the list it introduces. It needs the mirror rule: a paragraph that refers back to a preceding
list should bind *backward* to it. That is a targeted change, and it should be run as a
controlled before/after like D9 rather than applied silently.

## Finding 2 — the strategies cannot be told apart at this sample size

This supersedes any reading of the strategy ranking as a result.

| | MRR |
|---|---|
| Spread between best and worst **strategy** | 0.053 |
| Movement from **one question** going rank 2 → rank 1 | 0.063 |
| Movement from **one question** going rank 3 → rank 1 | 0.083 |

The entire difference between the best and worst chunking strategy is smaller than a single
question improving by one rank position.

And the within-strategy spread is far larger than the between-strategy spread:

| Strategy | Mean MRR | Range across its own settings | Spread |
|---|---|---|---|
| structure_aware | 0.851 | 0.771 – 0.938 | 0.167 |
| paragraph | 0.832 | 0.719 – 0.917 | 0.198 |
| fixed_size | 0.823 | 0.677 – 0.906 | 0.229 |
| sentence_window | 0.798 | 0.646 – 0.875 | 0.229 |

Within-strategy variation is three to four times between-strategy variation. Whatever moves
these numbers, it is not the choice of strategy.

**The honest conclusion: at n=8, this evaluation cannot distinguish the four chunking
strategies.** `structure_aware` has the highest point estimate and is a reasonable working
default, but calling it "best" is not supported.

This is itself a useful methodological result. It is a concrete, quantified argument for
expanding the test set — not a general appeal to "more data would be better", but a specific
statement that the current instrument lacks the resolution to answer the question being asked
of it. Report it that way.

## Finding 3 — a safety/performance trade-off

The only strategy that preserves the rule/exemption grouping is also the worst performer.

| Strategy | Mean MRR | 8547 grouping |
|---|---|---|
| structure_aware | 0.851 | split |
| paragraph | 0.832 | split |
| fixed_size | 0.823 | split |
| **sentence_window** | **0.798** | **preserved** |

Given Finding 2, the ordering itself is not meaningful. What survives is the structural point:
**the only strategy that preserves the rule/exemption grouping is the one that does so by
accident of formatting**, and no strategy preserves it by design.

The fix is therefore a targeted binding rule in `chunking.py`, not a choice between strategies —
which is fortunate, because the numbers cannot support choosing between them anyway.

## Finding 4 — the subclass prefix helps the two retrievers differently

Averaged across all 34 configurations:

| Retriever | metadata | R@1 | MRR | wrong-subclass@1 |
|---|---|---|---|---|
| dense | off | 0.728 | 0.834 | 0.306 |
| dense | **on** | 0.728 | 0.818 | **0.235** |
| bm25 | off | 0.596 | 0.761 | 0.200 |
| bm25 | **on** | **0.801** | **0.870** | 0.200 |
| hybrid_rrf | off | 0.706 | 0.820 | 0.306 |
| hybrid_rrf | **on** | 0.765 | 0.855 | **0.176** |

The pattern is mechanistically sensible. The prefix adds a **literal token** — "Subclass 482" —
which BM25 matches directly, giving it the largest ranking gain (MRR 0.761 → 0.870). Dense
retrieval already captured much of that meaning, so its ranking does not improve (and dips
slightly, within noise), but its **subclass confusion falls by a quarter**.

Hybrid gets both effects and reaches the lowest confusion rate of any configuration.

## Best configuration measured

`structure_aware`, `max_chars=1400`, subclass prefix on, hybrid RRF retrieval:
**R@1 0.875 · MRR 0.938 · wrong-subclass@1 0.000** — the only entry in the top ten with no
wrong-subclass rank-1 hits.

Caveat: n=8, document-level, and per Finding 2 not distinguishable from several others. Carry it
forward as the working default, not as a proven optimum.

## Reproducing any row

Sweep snapshots are gitignored — they are large and rebuildable. Every row of
`chunking_sweep.csv` records its exact parameters, so any configuration regenerates with one
ingest command, for example:

```
python ingest.py --strategy structure_aware --max-chars 1400 --prepend-metadata
```

---

# T09b — Fixing the condition 8547 hazard (2026-08-21)

A controlled before/after on the chunking defect the sweep exposed. Same 34 configurations,
same 8 questions, same retrievers — the only change is one binding rule in `chunking.py`.

## The defect

`chunking.py` already bound a paragraph ending in `:` **forward** to the list it introduces. It
had no mirror rule, so the paragraph that *follows* a list could be split away from it.

On the WHM document that produced a 134-character chunk reading:

> *"If work does not fall within an exemption, you can only work for the same employer for a
> maximum of 6 months while holding a WHM visa."*

A rule with its exemptions detached. Retrieved alone it would tell someone doing fruit picking to
stop after six months, when plant and animal cultivation is explicitly exempt.

Every retrieval metric scores that chunk as fine, because it *is* topically relevant. Relevance
and safety come apart here, which is the point.

## The fix

One rule: **a block containing bullets binds to the paragraph that follows it.** A bound unit is
never split, even if it exceeds `max_chars` — correctness beats size. Applied to both
`structure_aware` and `paragraph`, which are the two paragraph-aware strategies.

## Result

| Strategy | Before | After |
|---|---|---|
| `structure_aware` | 6 split / 0 safe | **0 split / 6 safe** |
| `paragraph` | 6 split / 0 safe | **0 split / 6 safe** |
| `sentence_window` | 0 split / 4 safe | 0 split / 4 safe |
| `fixed_size` | 18 split / 0 safe | 18 split / 0 safe |
| **Total** | **30 of 34 split** | **18 of 34 split** |

## Cost: none measurable

Across the two affected strategies (36 rows):

| | MRR | R@1 | wrong-subclass@1 |
|---|---|---|---|
| Before | 0.842 | 0.736 | 0.233 |
| After | 0.841 | 0.736 | 0.228 |

Retrieval quality is unchanged. Given the resolution limits recorded in T09 Finding 2, the right
statement is that **the fix costs nothing measurable at this sample size** — not that it is
exactly free.

## Why `fixed_size` cannot be fixed

All 18 remaining failures are `fixed_size`, and this is inherent rather than a gap in the
implementation. Fixed-size chunking cuts on character counts and has no representation of
paragraphs, lists, or the relationship between them. There is no unit for a binding rule to
operate on.

This is a real finding about the strategy class, and it is worth stating in the report:
**on structured regulatory text, character-based chunking cannot preserve
rule-and-exemption groupings by construction.** Not because our implementation is naive —
because the strategy has no access to the structure it would need.

That is a stronger argument against fixed-size chunking than any of the retrieval numbers, all of
which sit inside the noise floor.

## Best configuration measured

`structure_aware`, `max_chars=600`, subclass prefix on, BM25: R@1 0.875 · MRR 0.938 ·
wrong-subclass@1 0.200. Per T09 Finding 2, not distinguishable from several others — carry it as
the working default, not a proven optimum.

---

# T04 — Text-similarity measures, and where they disagree with correctness (2026-08-21)

ROUGE-1 and a labelled embedding-cosine proxy, over all runs. `results/generation_metrics.csv`.

The proxy is **not BERTScore** and is labelled `cosine_proxy (NOT BERTScore)` in the CSV header.
BERTScore does greedy token-level matching over contextual embeddings and reports P/R/F1; this is
a single cosine over whole-text embeddings from the retrieval model already installed. It is a
defensible stand-in for semantic similarity and an indefensible substitute for the named metric.

## Retrieval helps, on both measures

| System | Mean ROUGE-1 F1 | Mean cosine proxy |
|---|---|---|
| B0 (no retrieval) | 0.117 | 0.668 |
| RAG | 0.300 | 0.765 |

n=13 questions.

## The measures do not track correctness

This is the point of the task, and there is a clean demonstration in the k=3 run.

| Question | Correct? | ROUGE-1 F1 |
|---|---|---|
| **q03** | **Yes** — "a Subclass 482 visa requires employer sponsorship", matching the gold answer | **0.154** |
| **q02** | **No** — gives the Second Post-Higher Education Work stream figure for a question about the Post-Higher Education Work stream | **0.185** |

**The wrong answer scores higher than the correct one.**

q03 is correct but phrased more directly than the gold answer, so it shares few words with it.
q02 is wrong by more than a year on the only figure that matters, but its sentence structure
closely mirrors the reference.

This is not a flaw in the implementation. It is what these measures are: they compare an answer
to a reference **as text**, and text similarity is not truth. Reporting them without saying so
would let a reader infer a quality ranking the numbers cannot support.

It is also the concrete argument for the project's domain-specific measures — cross-subclass
confusion, the answerability split, attribution correctness. Those catch what ROUGE cannot.

## A second observation: q13 hedges and answers

q13 opens with the configured refusal string and then answers anyway, correctly, from the
exemption list. A string-matching refusal detector will classify it as a refusal; it is a hedged
answer. Any refusal metric has to distinguish those, or the answerability split will be wrong in
both directions at once.

Recorded here because it is a known fixture for that work, not a defect found in passing.
