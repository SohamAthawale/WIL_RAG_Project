# T13 — how the existing 13 questions are built

Derived from the current test set. Use it as the pattern to write against; the questions
themselves are yours to write.

---

## `answerable_direct` — 7 existing, target 60

**Shape A — name the subclass in the question** (q01, q02, q03, q06)

> "How many hours a fortnight can a **Subclass 500 student visa holder** work while their
> course is in session?"

The subclass is stated, so there is exactly one right answer and retrieval has an unambiguous
target. This is the bulk of the category and the fastest to write from the inventory.

**Shape B — describe a person, let the subclass be inferable** (q13)

> "I'm **on a Working Holiday visa doing fruit picking on a farm** — do I need permission to keep
> working for the same farm after 8 months?"

Harder and more realistic. Still `answerable_direct` because the situation pins one visa.

### The gold answer must carry its qualifier

Not "48 hours a fortnight" but "48 hours per fortnight **while the course of study or training is
in session**". Every figure in this corpus has a condition attached, and an answer that drops it
is wrong in the way that gets someone in trouble.

Where the corpus states an exception, include it — q02 gives "between 2 and 3 years" *and*
"5 years for Hong Kong / BNO passport holders". A gold answer missing the exception will mark a
correct system as wrong.

---

## `cross_subclass_probe` — 3 existing, target 25

**Shape A — the bare question with no subclass** (q07, q08)

> "How many hours a fortnight am I allowed to work on my visa?"

There is no single answer. The correct behaviour is to say it depends and enumerate.
`gold_source: "multiple"`, `subclass: "ambiguous"`.

**Shape B — the transition** (q09) — the strongest type

> "I'm an international student who just graduated with a bachelor's degree from an Australian
> university. How long can I stay, and what is my work-hours limit?"

Describes a *person*, not a visa. Tests whether the system recognises this is now a 485 question
and not a continuation of the 500. `gold_source` is the specific document.

**This category is not limited by the corpus.** Comparisons are combinatorial — 5 documents give
10 pairs, times hours / stay / cost / family / sponsorship. Getting to 25 is easy; getting to 60
direct questions is not. If the ratio has to move, move it here.

---

## `refusal_required` — 3 existing, target 15

Three distinct sub-types, and you want all three represented:

| type | example | why it is hard |
|---|---|---|
| Outcome prediction | "Will my Subclass 482 application be approved?" (q10) | invites a guess |
| Personal advice | "Should I take the job?" (q11) | the corpus *has* related content |
| Out of scope | "What's the weather in Melbourne?" (q12) | trivially outside |

**q11 is the type worth copying.** It names a real corpus concept — the Specialist Skills Income
Threshold — so relevant chunks *are* retrieved, and the system must still decline. A refusal
question with no retrievable context only tests the easy case.

All three use `gold_source: "none"`.

---

## Traps specific to this corpus

**1. The 417 document repeats itself.** "You can do any kind of work while you are here" appears at
line 19 (first Working Holiday visa) and line 27 (second). Same for "For people who currently
hold, or have held…". A question touching these must say *which* visa, or it has two defensible
answers.

**2. The 485 streams are near-twins.** "Post-Higher Education Work stream" (2–3 years) versus
"Second Post-Higher Education Work stream" (1–2 years). The second heading contains the first
verbatim. This is the exact failure the confusion counter exists to catch — write questions that
name the stream explicitly.

**3. Five chunks contain no facts at all** — `482::0`, `482::1`, `485::0`, `500::4`, `whm::0`.
They are intros and headers. No direct question can be sourced from them.

**4. Figures repeat across streams.** `AUD5,750.00` appears in two different 485 streams, and the
5-year HK/BNO allowance in several. A question whose answer is only a repeated number cannot
distinguish which passage the system used.

---

## The schema

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

- `subclass` — the number, or `"ambiguous"`, or `"none"`
- `gold_source` — one filename, or `"multiple"`, or `"none"`
- `gold_chunk_ids` — empty list on new questions. **Never touch the 12 populated ones.**

## Before committing

```
python3 -c "import json,collections; d=json.load(open('data/testset/test_set.json')); print(len(d),'questions'); print(collections.Counter(q['category'] for q in d)); print('populated gold lists:',sum(1 for q in d if q.get('gold_chunk_ids')),'(must stay 12)')"
```
