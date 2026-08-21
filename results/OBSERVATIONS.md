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

---

# T08 — top-k sweep: the published finding did not replicate (2026-08-21)

Full test set generated at k = 1, 3 and 5 on `structure_aware_meta`, scored with ROUGE-1 and the
labelled cosine proxy. 13 questions per configuration.

## Result

| k | ROUGE-1 precision | recall | F1 | cosine proxy | mean answer length |
|---|---|---|---|---|---|
| 1 | 0.246 | 0.423 | 0.290 | 0.766 | 54.3 words |
| 3 | 0.273 | 0.451 | 0.310 | 0.764 | 53.6 words |
| 5 | 0.295 | 0.522 | **0.346** | **0.783** | 53.8 words |

**Quality increases monotonically with k on both measures.** The published result this reproduces
reported the opposite — higher answer quality at *lower* k, attributed to additional passages
distracting the model. On this corpus that does not hold.

## The obvious confound, ruled out

ROUGE recall rises mechanically when answers get longer, because a longer answer has more chances
to contain reference words. If that were the explanation, recall would climb while precision fell.

It is not the explanation:

- **Answer length is flat** across k — 54.3, 53.6, 53.8 words.
- **Precision and recall both rise** — 0.246 → 0.295 and 0.423 → 0.522.

Answers at higher k contain more of the right words *without getting longer*. That is a real
effect, not an artefact of verbosity.

## Honest limits

The ROUGE-1 spread across k is 0.056. One question changing its F1 by 0.5 moves the mean by
0.038, so the whole spread is roughly 1.5 questions' worth of movement at n=13. Directional, not
significant.

Two things make it more than noise-shaped, though: the trend is **monotonic across all three
points** on ROUGE, and precision and recall move **together**, which noise would not reliably do.

## Why it may differ

Worth stating in the report rather than leaving as a puzzle. The published study used a curated
FAQ where one entry answered one question, so additional passages were pure distraction. This
corpus is five overlapping government pages where answers are frequently split across sections —
a stay figure in one chunk, the condition qualifying it in another. More context plausibly helps
here for the same reason it hurt there.

That is a hypothesis the corpus structure supports, not a finding. It would be tested by rerunning
the sweep once the test set is large enough to separate the categories: if extra context helps
`cross_subclass_probe` questions more than `answerable_direct` ones, the explanation holds.

---

# Re-run after the corpus correction (2026-08-21)

The condition 8547 exemption list was corrected — `food processing` had been omitted and the list
was presented as closed where the source says "including". Corpus snapshot id changed from
`2d8e935b8999` to `84636c1d6741`, so every prior result had to be re-run before being quoted.

## Defect found while re-running

`eval_configs.py` recorded `config_id` but **not** `snapshot_id`. Two snapshots sharing a config
id — which is exactly what a corpus correction produces — gave indistinguishable rows. A result
from a superseded corpus could sit in the same table as a current one with nothing to separate
them.

This is the same class of failure as the earlier divergence between the generation and retrieval
paths: provenance recorded in the filename but not in the data. Fixed; `snapshot_id` is now the
first field in every row and the first column in the printed table.

## Every conclusion held

| Finding | Old corpus | Corrected corpus |
|---|---|---|
| 8547 rule split from exemptions | 18 of 34 configs | **18 of 34** |
| Which strategies split it | all and only `fixed_size` | **all and only `fixed_size`** |
| Configs with zero wrong-subclass@1 | structure_aware + meta + hybrid | **same, 3 of them** |
| Best measured config | structure_aware + meta | **same** |
| hybrid RRF + metadata MRR | 0.906 | 0.938 |
| bm25 + metadata MRR | 0.917 | 0.938 |

Nothing reversed. The findings were not artefacts of the corpus error, which is worth stating in
the report — a correction of this size is exactly the kind of thing that could have invalidated
them, and it did not.

**Caveat on the MRR movement:** the old snapshot predates the rule/exemption binding fix, so the
old-versus-new comparison mixes two changes. The improvement should not be attributed to the
corpus correction alone.

## Why the fix mattered anyway

The corrected chunk now carries the rule, the full exemption list and `food processing` together
in `whm_6_month_work_limitation::1`. Before the correction, a question about food processing was
unanswerable from our corpus in a way that produced a confident wrong answer rather than a
refusal — the worst combination available.

That the retrieval numbers barely moved is the point: **a corpus error of real consequence was
invisible to every metric in the framework.** Only a human comparison against the live source
found it. Worth saying plainly in the report, next to the metrics.

---

# top-k re-run on the corrected corpus (2026-08-21)

All three k values re-generated on snapshot `84636c1d6741`, then re-scored.

| k | ROUGE-1 precision | recall | F1 | cosine proxy |
|---|---|---|---|---|
| 1 | 0.258 | 0.427 | 0.305 | 0.771 |
| 3 | 0.262 | 0.547 | 0.318 | 0.780 |
| 5 | 0.278 | 0.588 | **0.342** | **0.791** |

## The direction held; the magnitude shrank

| | Old corpus | Corrected corpus |
|---|---|---|
| Monotonic increase with k | yes | **yes** |
| Both precision and recall rise | yes | **yes** |
| Proxy also increases | mixed | **yes, monotonic** |
| ROUGE-1 F1 spread across k | 0.056 | **0.037** |

**The spread is now smaller than one question.** At n=13, a single question changing its F1 by 0.5
moves the mean by 0.038. The entire best-to-worst difference across k is 0.037.

So the honest statement is narrower than before:

> We did not observe the published finding that quality is higher at lower k. Across two corpus
> versions the direction was consistently the opposite, monotonic on both measures, with precision
> and recall rising together. **But the effect size is below what a 13-question set can resolve**,
> so this is a failure to replicate rather than a refutation.

Two things still argue the direction is real rather than noise: it is monotonic across all three
points on both independent measures, and it survived a corpus change. Neither is decisive at this
sample size.

**This is the third finding in this project to land below the resolution limit**, after the
chunking strategy comparison and parts of the retrieval sweep. That pattern is itself a result:
the framework is producing directionally consistent signals it cannot yet confirm, and the binding
constraint is the test set, not the pipeline or the metrics.

Report it as a quantified argument for expansion — the instrument resolves differences of roughly
0.04 in mean ROUGE-1 F1, and three separate findings sit under that threshold.

---

## T06 — Cross-subclass confusion, five counters

`src/confusion.py`, `results/confusion_counters.json`, controls in `tests/test_confusion.py`.

Five defensible ways to count "answered using the wrong visa's rules", computed side by side so
the headline definition is chosen after seeing the numbers.

| | definition | pre-fix, k=3 | k=1 | k=3 | k=5 |
|---|---|---|---|---|---|
| (a) | cites a wrong-subclass document | 0 | 0 | 0 | 0 |
| (b) | states a fact found only in a wrong-subclass chunk | n/a | 0 | 0 | 0 |
| (c) | rank-1 chunk is wrong-subclass | **2** | 1 | 1 | 1 |
| (d) | cites a document never in its context | **1** | 0 | 0 | 0 |
| (e) | fact from the wrong stream of the right subclass | n/a | 0 | 0 | 0 |

Counts are questions out of 13.

**The direction matches what the observations record.** The subclass-identifier fix removed q01
from both (c) and (d). The one remaining (c) is q09, a `cross_subclass_probe` — a question written
to be adversarial, where retrieval puts a Subclass 500 chunk at rank 1 for a 485 question. The
counter firing there is the counter working.

**Confusion does not change with k.** All five counters are flat across k=1, 3 and 5. The
plausible worry that a larger context window admits more cross-subclass contamination is not
supported: the extra chunks are retrieved but not drawn from. Nobody had looked before.

**(b) and (e) cannot be computed on the pre-fix run.** That run's corpus was the superseded
snapshot, which was deleted when the corpus was corrected. Chunk ids are recovered by replaying
retrieval and accepting it only when the source-and-score sequence reproduces exactly; against a
corpus that no longer exists it never does, so those 13 questions are recorded as unresolvable
rather than scored against the wrong chunks. (a), (c) and (d) need no chunk ids and are unaffected.

### Every counter reads zero, and that took work to make meaningful

A detector that cannot fire produces exactly the same table as a system that never errs. Each
counter therefore has a positive control built from the real corpus that must trip it, and the
real answer that must not. Writing them found three defects that had all been reporting clean:

1. **(e) could not fire at all.** The target stream was read from `gold_chunk_ids`, which is every
   chunk graded 1 or above. For q02 the annotator graded all five 485 chunks — including the
   Second Post-Higher Education Work stream, the very chunk (e) exists to catch. Reading the
   target from the *top-graded* chunk instead fixes it. **This is the concrete case for graded
   judgements over binary ones**: the binary gold set destroys the distinction, the graded one
   preserves it.
2. **The subclass number counted as a fact.** `[Subclass 485 …]` in the metadata prefix meant every
   485 chunk looked like the source of every 485 answer.
3. **Ranges lost one endpoint and money lost its thousands.** "between 2 and 3 years" yielded only
   the 2 — and the 3 is exactly what separates the correct stream from the wrong one. `AUD5,750.00`
   parsed as 750.

### Unrelated finding: the citation format is not obeyed

Counting (a) and (d) meant parsing citations, which exposed that the model largely ignores the
instructed `[Source: filename]` format:

| form | k=1 | k=3 | k=5 |
|---|---|---|---|
| `[Source: f]` as instructed | 1 | 2 | 1 |
| `Source: [f]` | 1 | 2 | 2 |
| freeform or absent | 11 | 9 | 10 |

Observed freeform variants include `sourced from [f]`, `in the source document f`, and
`refer to the source document "f"`. **At most 4 of 13 answers cite in a machine-readable form.**
Any attribution measure built on the documented format alone would silently score most of the set
as uncited — the first version of this counter did exactly that. Citations here are instead matched
against the closed set of corpus filenames, which is format-independent.

This belongs to attribution correctness (T07) and should be picked up there.

### What this does to T10

T10 exists to fix q02, which took the Second-stream figure while retrieving only correct-subclass
chunks. **On the corrected corpus q02 answers correctly** — "between 2 and 3 years … up to 5 years
for Hong Kong and British National Overseas passport holders" — so the motivating failure no longer
reproduces. The distractor chunk is still retrieved at rank 2; the system now orders past it.
T10 needs re-scoping as a robustness test or parking, and (e) is retained as the regression guard
either way.

---

# Retrieval comparison re-run at n=61 — a finding reverses (2026-08-21)

T13 expanded the test set from 13 to 100 questions. `src/eval_configs.py` scores at the
**document** level against `gold_source`, which every new question already carries, so the
comparison went from **n=8 to n=61 without a single new relevance judgement**.

Same corpus, same code, same configurations. Only the sample size changed.

## `structure_aware_meta`, before and after

| retriever | R@1 n=8 | R@1 n=61 | MRR n=8 | MRR n=61 | wrong-sc@1 n=8 | wrong-sc@1 n=61 |
|---|---|---|---|---|---|---|
| dense | 0.750 | **0.967** | 0.838 | **0.979** | 0.200 | **0.019** |
| bm25 | 0.875 | 0.934 | 0.938 | 0.967 | 0.200 | 0.037 |
| hybrid_rrf | 0.875 | **0.984** | 0.938 | **0.992** | 0.000 | **0.000** |

## What reversed

**At n=8, BM25 out-ranked dense retrieval — R@1 0.875 against 0.750, MRR 0.938 against 0.838.
At n=61 the ordering is the other way: dense 0.967 against 0.934, MRR 0.979 against 0.967.**

The n=8 result was reported with a sample-size caveat attached, and the caveat was right. Eight
questions is one question per 0.125 of R@1; the gap between the two retrievers was one question
wide. It was noise presented as a ranking.

This is the second finding in this project to fail on more data, after the top-k result. Both
failed in the direction the caveats predicted, which is the only reassuring thing about it.

## What held, and strengthened

**Hybrid RRF fusion leads on every measure** — R@1 0.984, MRR 0.992, and 0.000 wrong-subclass@1,
the only configuration with no rank-1 subclass errors at any sample size tested. At n=8 it merely
tied the better of its two inputs; at n=61 it beats both. Fusion earning its place is a stronger
claim now than it was.

**The subclass prefix still matters.** Within `structure_aware`, metadata on versus off:
wrong-subclass@1 0.019 against 0.111 for dense, 0.037 against 0.093 for BM25. The effect is
roughly five times larger than the resolution limit at this n.

**`structure_aware` still beats `paragraph`**, and by more than before — dense R@1 0.967 against
0.754.

## What this does to the rest of the report

**Three findings still rest on n=8 and have not been re-run.** T09's chunking sweep — including
Finding 4, that the prefix helps BM25 more than dense — is 34 configurations scored at n=8. Given
that the dense-versus-BM25 ordering just reversed at the same sample size, **Finding 4 should be
treated as unverified until the sweep is re-run at n=61.** It is not contradicted; it is
untested.

The `chunking_sweep.csv` re-run is cheap — retrieval only, no generation — and should happen
before T12 quotes any of it.

## The point worth making in the report

A framework that reports a caveat and then, when the sample grows, **catches its own published
finding**, is doing the job. The original result was not sloppy: it was measured correctly,
reported with its limitation stated, and superseded when better evidence arrived. That is the
difference between an evaluation and a demo, and it is a better section than the finding it
replaces.

It also quantifies why T13 mattered. The test-set expansion did not merely make future numbers
more trustworthy — it invalidated an existing one, at no judging cost, on the day it landed.

---

# T08 re-run at n=85 — the top-k effect resolves, and changes shape (2026-08-22)

The top-k sweep was re-run on the expanded test set: 100 questions, of which **85 carry a gold
answer** (the 15 `refusal_required` questions have none to score against). Three runs at k=1, 3
and 5, same corpus `84636c1d6741`, same config `structure_aware_meta`. B0 was generated once and
reused, since it does not depend on k.

## Means, against the n=13 result

| k | n=13 | **n=85** | median | sd | SE |
|---|---|---|---|---|---|
| 1 | 0.290 | **0.377** | 0.361 | 0.156 | 0.017 |
| 3 | 0.310 | **0.424** | 0.419 | 0.146 | 0.016 |
| 5 | 0.346 | **0.424** | 0.420 | 0.137 | 0.015 |

**The resolution limit fell from ~0.038 to ~0.016.** That is what makes the rest of this section
possible: at n=13 the entire spread across k was 0.037, smaller than one question's influence, and
the finding had to be reported as unresolvable.

## Paired comparison, 85 questions

Every run scores the same questions, so the comparison is paired rather than between-groups —
substantially more powerful, because per-question difficulty cancels.

| comparison | mean difference | SE | t | better | worse | tied |
|---|---|---|---|---|---|---|
| k=1 → k=3 | **+0.0473** | 0.0154 | **3.08** | 45 | 31 | 9 |
| k=3 → k=5 | −0.0007 | 0.0087 | −0.09 | 41 | 33 | 11 |
| k=1 → k=5 | **+0.0466** | 0.0167 | **2.79** | 49 | 29 | 7 |

## What this establishes

**Quality rises from k=1 to k=3, then stops.** The k=1→k=3 gain is three standard errors and sits
well clear of the resolution limit. The k=3→k=5 difference is −0.0007 against an SE of 0.0087 —
indistinguishable from zero, and the win/loss split (41/33/11) is what a coin flip looks like.

**The published finding is now contradicted, not merely unreplicated.** Earlier this was stated as
a failure to replicate, because the effect was smaller than the instrument could resolve. It is
now resolvable and points the other way: more context helps up to k=3.

**But the earlier description was also wrong.** At n=13 the pattern read as monotonic — 0.290,
0.310, 0.346 — and was reported that way. It is not monotonic. It rises and then plateaus, and the
n=13 ordering put k=5 highest when k=3 and k=5 are in fact tied. **Both the published claim and
our own characterisation of our own data failed at higher power.**

## Retrieval's benefit is larger than the small sample showed

At k=3: B0 mean 0.139, RAG mean 0.424 — a gap of 0.285, roughly eighteen standard errors. At n=13
the same gap was 0.193. Whatever else moved, this did not weaken.

## What follows

**k=3 is confirmed as the default**, now on evidence rather than convention. k=5 retrieves 67%
more context, costs proportionally more generation time, and returns nothing measurable. That is a
concrete, defensible configuration decision.

**Formal significance testing is T16 and has not been done.** The t values above are effect sizes
divided by their standard errors, reported to show what the instrument can now resolve. They carry
no multiple-comparison correction, and three of the comparisons here are not independent. T16 owns
that analysis; do not quote these as p-values.

## The pattern across today

Three findings were re-examined at larger n. **Two changed.** The dense-versus-BM25 ordering
reversed, and the top-k relationship changed shape. The subclass-prefix effect, the RRF fusion
result, and retrieval-beats-no-retrieval all held and strengthened.

Every one of the findings that moved had been published with an explicit sample-size caveat
attached. The caveats were not defensive boilerplate — they marked exactly the claims that could
not survive, and all of them did fail. That is the evaluation framework working as designed, and
it is a more valuable thing to report than any individual number in it.
