# T13 — Expand the test set · working brief

**Heet Harshad Chanchad (s4218211)** · RMIT WIL Group 97 · Test-Driven RAG for Australian Visa
Conditions & Work Rights

Self-contained. Everything you need is here or in two files named below. Read top to bottom
before writing anything — several things changed after the original task brief was written, and
two of them will cost you a day if you find them the hard way.

---

## 1. What the project is

A retrieval-augmented generation system answers questions about Australian temporary visa work
rights from five Department of Home Affairs documents.

**The evaluation framework is the deliverable, not the chatbot.** A working pipeline is necessary
but not sufficient. What earns marks is measuring the system rigorously and reporting honestly.

The headline metric is **cross-subclass confusion** — how often the system answers using the wrong
visa's rules. Telling a Subclass 485 holder they are capped at 48 hours a fortnight, or telling a
500 holder they are not, are both visa breaches.

## 2. What T13 is

Write approximately 100 questions with verified correct answers, in three categories:
**60 `answerable_direct` · 25 `cross_subclass_probe` · 15 `refusal_required`**.

There are 13 questions now. You are adding roughly 87.

## 3. Why it matters, and the honest version of why

At 13 questions a single question moves any score by about 8%. No statistical test separates a
real improvement from noise at that size. Walert, the paper this project reproduces, used 106.

**Three findings in this project currently sit below what 13 questions can resolve** — the
chunking strategy comparison, the top-k sweep, and parts of the retrieval sweep. The instrument
resolves differences of roughly 0.04 in mean ROUGE-1 F1; those effects are smaller. They are
directionally consistent and cannot be confirmed.

**But be clear about what your task does and does not unblock.** T13 blocks T14 and T16. It does
*not* block T12 (the comparison table) or T22 (the report). The report can be written and
submitted without any of your work.

What your work decides is whether the numbers in it survive scrutiny. That is a different kind of
urgent, and it is easy to deprioritise precisely because nothing appears to be waiting on it.

## 4. Your setup — already done

| | |
|---|---|
| Folder | `~/Documents/GitHub/wil-rag-heet` |
| Branch | `heet/t13`, checked out |
| Git identity | Heet Harshad Chanchad · `190972763+heetchanchad15@users.noreply.github.com` |
| Remote | `github.com/SohamAthawale/WIL_RAG_Project` |

**Work only in that folder.** `~/Documents/GitHub/wil-test-driven-rag` is Soham's and carries his
git identity. This laptop is shared; separate folders is what stops your commits being credited
to him and his to you.

You do not need a virtual environment. Everything below runs on plain `python3`.

Verify your identity is right before your first commit:

```
cd ~/Documents/GitHub/wil-rag-heet && git config user.email
```

Must print your noreply address. If you ever see `user.email has multiple values`, run
`git config --local --unset-all user.email` and set it once, cleanly.

## 5. The dependency checks — run, and passing

The task brief tells you to verify what you are building on before starting. That has been done:

| Check | Result |
|---|---|
| `python3 src/manifest.py --verify` | **No drift** — all 5 documents match the manifest |
| Manifest snapshot | `84636c1d6741` |
| What the pipeline reads | `84636c1d6741`, 24 chunks, `corpus_current: True` |
| Has T02 run? | **Yes** — 13 questions judged, 12 with gold ids, 312 rows |

Re-run the first one yourself in your own folder. It is the check that catches the corpus moving
underneath you.

**Two things the original task brief gets wrong, because it predates the corpus correction:**

- It names snapshot `2d8e935b8999` at 26 chunks. That snapshot has been **deleted**. The corpus is
  `84636c1d6741` at **24 chunks** — it changed when a missing food-processing exemption was found.
- It says "check whether T02 has run". It has. So its instruction to add an empty `gold_chunk_ids`
  list applies **to your new questions only**. Never blank an existing one: those 12 populated
  lists are 312 rows of human judgement and cost roughly a full day to recreate.

## 6. The corpus is smaller than the target assumes

This is the finding that should shape how you work.

| | |
|---|---|
| Body text, all five documents | **1,760 words** |
| Locatable assertions | **83** |
| Already used by the existing 13 questions | 22 |
| **Available for new direct questions** | **61** |

Your target is 60 `answerable_direct`. **You have 61 unasked facts.** That is a margin of one.

Writing 60 direct questions means using essentially every remaining assertion in the corpus,
including the thin ones, with nothing in reserve if you reject a weak candidate.

The other two categories are **not** constrained this way. `cross_subclass_probe` is combinatorial
— five documents give ten pairs, times hours / stay / cost / family / sponsorship. `refusal_required`
questions are by definition not answerable from the corpus at all.

**So if the ratio has to move, move it away from `answerable_direct`.** Raise it with the team;
do not quietly pick. Section 7 of your task brief already asks you to record exactly this as
evidence about corpus coverage — it is a finding, not an obstacle.

## 7. Your two working files

**`results/corpus_inventory.md`** — every assertion in the corpus with its file, line number, and
chunk id. Each row is a candidate question whose gold source is already identified. Rows marked
`ASKED` are used by the existing 13; prefer the others, because re-asking a covered fact grows the
sample without adding information.

**`results/question_patterns.md`** — how the existing 13 questions are constructed, category by
category, and the four traps specific to this corpus.

Work down the inventory. The chunk column gives you `gold_source` for free.

## 8. Writing the questions

### The schema

```json
{
  "id": "q14",
  "category": "answerable_direct",
  "subclass": "500",
  "question": "...",
  "gold_answer": "...",
  "gold_source": "subclass_500_student.txt",
  "gold_chunk_ids": []
}
```

`subclass` is the number, or `"ambiguous"`, or `"none"`. `gold_source` is one filename, or
`"multiple"`, or `"none"`. `gold_chunk_ids` is an empty list on new questions.

### Gold answers must carry their qualifier

Not "48 hours a fortnight" but "48 hours per fortnight **while the course of study or training is
in session**". Every figure in this corpus has a condition attached, and an answer that drops it
is wrong in the way that gets a real person in trouble.

Where the corpus states an exception, include it. The existing q02 gives "usually between 2 and 3
years" *and* "5 years for Hong Kong / BNO passport holders". A gold answer missing the exception
will mark a correct system as wrong.

### Write the hard version of each category

- **Direct** — naming the subclass in the question is quick, but describing a situation that pins
  it is more realistic and tests more.
- **Cross-subclass** — the strongest kind describes a *person*, not a visa, and requires working
  out which visa now applies. The existing q09 is the model.
- **Refusal** — the strongest kind names a real corpus concept, so relevant chunks *are* retrieved
  and the system must decline anyway. The existing q11 does this. A question about Melbourne
  weather only tests the trivial case.

### Four traps in this corpus

1. **The 417 document repeats itself verbatim.** "You can do any kind of work while you are here"
   appears at line 19 (first Working Holiday visa) and line 27 (second). A question touching these
   must say which visa, or it has two defensible answers.
2. **The 485 streams are near-twins.** "Post-Higher Education Work stream" is 2–3 years; "Second
   Post-Higher Education Work stream" is 1–2 years, and its heading contains the other's in full.
   Name the stream explicitly.
3. **Five chunks contain no facts** — `482::0`, `482::1`, `485::0`, `500::4`, `whm::0`. They are
   intros and headers; no direct question can be sourced from them.
4. **Figures repeat across streams.** `AUD5,750.00` appears in two different 485 streams. A
   question whose answer is only a repeated number cannot distinguish which passage was used.

## 9. Verify before you commit

```
python3 -c "import json,collections; d=json.load(open('data/testset/test_set.json')); print(len(d),'questions'); print(collections.Counter(q['category'] for q in d)); print('populated gold lists:',sum(1 for q in d if q.get('gold_chunk_ids')),'(must stay 12)')"
```

The last number **must still be 12**. If it dropped, you have overwritten someone's judgements —
stop and restore before doing anything else.

Also required by the task:

- Every gold answer traceable to a document and line. The inventory gives you this if you work
  from it.
- **Never write a visa rule from memory.** These rules change; that volatility is the entire
  premise of the project. An invented fact permanently poisons the ground truth.
- A second person spot-checks ten of your questions against the source documents.

## 10. Commit and push

```
git add -A
git commit -m "T13: expanded test set from 13 to N questions (60/25/15 split)"
git push -u origin heet/t13
```

Write the **result** into the commit message, not just the change.

Then open the commit on GitHub and confirm **your avatar appears**. A bare name with no avatar
means the commit is unlinked and does not count toward your contribution.

When the work is complete, merge:

```
git checkout main && git pull
git merge --no-ff heet/t13
git push origin main
```

`--no-ff` keeps your work as one identifiable merge commit, so it can be reverted as a unit with
`git revert -m 1 <sha>`. Never force-push `main`.

## 11. Two decisions that are not yours alone

**Raise both with the team before you write question 14.**

**Who judges your questions.** Every question you add creates judging work. The existing 13 were
judged exhaustively — 13 × 24 chunks = 312 rows, one person, one sitting. Scaling that rule to 100
questions means roughly **2,088 new rows**, about seven times the largest task anyone has done on
this project, and **nobody is assigned to it**. An unjudged question contributes to no measure but
looks like coverage, so it is worse than not existing.

Decision D2 in the decision sheet already anticipated this: exhaustive judging now, switch to
**pooling** once the set grows. Your task is what makes it grow. Pooling means judging only the
chunks some system actually retrieved in its top-k — roughly 800 rows instead of 2,088. Its known
cost is that recall is then measured against the pool rather than the corpus, and that belongs in
the report rather than being discovered by a marker.

Three things need deciding: pooled or exhaustive, who judges, and whether the 30% double-annotation
overlap from decision D3 happens this time. It did not happen for T02, which is why the project
currently has no inter-annotator agreement figure.

**The target ratio**, given 61 available facts against 60 direct questions.

**A smaller, fully judged set beats a larger, unjudged one.** If judging for 100 cannot be
resourced, land 40 that are completely judged. Bring the arithmetic to the conversation.

## 12. The result to produce

- `data/testset/test_set.json` — the expanded set
- `results/testset_summary.md` — the final distribution, and any document too thin to support many
  questions. That is useful evidence about corpus coverage, not a failure.
- A short block appended to `results/HANDOFF.md`:

```
### T13 — Heet Harshad Chanchad — <date>
Result: <the number of questions, and the final split>
Config:  snapshot 84636c1d6741, 24 chunks
Watch out for: <anything that surprised you>
```

## 13. What comes next

**T14 — Paraphrases and vocabulary fairness**, also yours, and it needs this finished first.

**T14 must come before Hitesh's T19.** Measure the vocabulary gap before anyone builds a synonym
map to close it, or dimension 7 has no result — you cannot report an improvement against a
baseline that was never taken.

## 14. Rules that apply to every task

1. **Never fabricate a number.** If something is not measured, write "not yet measured". A made-up
   result in an academic submission is misconduct, not a shortcut.
2. **Never state a visa rule from memory.** Read it from the corpus and cite the file and line.
3. **Report sample size next to every number.** Say "directional, not significant" until the set
   grows — which is what you are fixing.
4. **Commit under your own identity.** Being logged in with `gh auth login` does not do this:
   `gh` controls what you may push, `user.email` controls who the commit is credited to.
5. **Definition of done:** the task produced a result, it is saved under `results/`, and it is
   committed. Not when it runs — when the result exists.
6. **Work on your branch**, merge when the result exists.
7. If you hit a decision the task does not settle, **ask the team** — do not quietly pick. Record
   the answer in `docs/decision-sheet.md`.
8. **Commits carry the author's name only.** No co-author trailers, and no tool names in commit
   messages or in source comments. The work is authored and owned by the person committing it.
9. **This course runs a bounded-process AI policy.** AI must not produce final assessable content,
   and all AI assistance must be declared with prompt logs. The questions and gold answers are
   assessable content and must be written by you.
