# Corpus drift report

**Task:** T01 — pin the corpus · **Checked:** 2026-08-21 · **Annotator:** Glory Suresh Vanjare

Every document compared against its live Home Affairs source. The site returns 403 to scripted
requests, so this was done in a browser.

**Corpus snapshot id:** `2d8e935b8999` — recorded in `data/documents/MANIFEST.json` and computed
the same way as `corpus_snapshot_id()` in `ingest.py`, so it matches the directory name under
`data/snapshots/`. If they ever diverge, a vector store was built from a different corpus version.

---

## Summary

| Document | Live match | Finding |
|---|---|---|
| `subclass_500_student.txt` | ✅ | No drift. Every material fact verified |
| `subclass_482_skills_in_demand.txt` | ✅ | No drift. All four streams and costs match |
| `subclass_417_working_holiday.txt` | ⚠️ | First-visa figures match; second and third visa costs not independently confirmed |
| `subclass_485_temporary_graduate.txt` | ⚠️ | **Missing a stream** — live page has four, ours has three |
| `whm_6_month_work_limitation.txt` | ❌ | **Missing an exemption category** — material |

---

## Finding 1 — WHM document omits an exemption category (material)

**File:** `whm_6_month_work_limitation.txt`, line 11.

| | Critical sectors listed |
|---|---|
| **Ours** | agriculture, health, aged and disability care, childcare, tourism and hospitality |
| **Live** | agriculture, **food processing**, health, aged and disability care and childcare, tourism and hospitality |

**"food processing" is absent from our corpus.**

Why this matters more than a typo: these are the categories that *exempt* a Working Holiday Maker
from the six-month same-employer limit. A visa holder working in food processing is exempt. Asked
about it, a system reading our corpus would say they are not, and advise them to leave a job they
are entitled to keep.

A second, subtler difference: the live page says "critical sectors, **including** …", which is an
open list. Ours presents a closed one. Answering "is X a critical sector?" for anything outside
the five we list would be wrong in the same direction.

**Recommendation:** correct this before anyone judges relevance. No judgements exist yet (0 of
338), so this is the cheapest moment it will ever be to fix. Once T02 is under way, changing the
document changes the hash and invalidates every judgement made against it.

**This is a team decision, not mine to make** — it alters the corpus and therefore the snapshot id.

## Finding 2 — 485 document is missing a stream

**File:** `subclass_485_temporary_graduate.txt`

Live page has four streams; ours has three. Missing: **Replacement stream**.

The three we do carry are all verbatim correct:

| Stream | Stay — live | Stay — ours |
|---|---|---|
| Post-Vocational Education Work | Up to 18 months | ✅ same |
| Post-Higher Education Work | Usually between 2 and 3 years, depending on your qualification | ✅ same |
| Second Post-Higher Education Work | Between 1 and 2 years depending on… | ✅ same |

Lower severity than Finding 1: a missing stream means questions about it are unanswerable, which
the system should refuse. A missing *exemption* means the system answers, wrongly.

**Side note that validates other work:** this confirms the gold answer for test question q02
("usually between 2 and 3 years") against the live source. The system's incorrect answer took the
Second stream's figure, so that failure is a genuine system error and not a corpus error.

## Finding 3 — 417 partially verified

Stay (12 months) and first-visa cost (AUD840.00) match the live page. The second and third
Working Holiday visa costs recorded in our document (AUD1,000.00 each) were not independently
confirmed — they sit on linked pages I did not open.

Flagged rather than asserted. Someone should confirm before those figures are used in a gold
answer.

## Findings 4 and 5 — no drift

`subclass_500_student.txt` verified in full, including the project's headline fact: *"work up to
48 hours a fortnight when your course of study or training is in session"*, confirmed verbatim.
Stay ("Up to 6 years and in line with your enrolment") and cost (AUD2,500.00) also match.

`subclass_482_skills_in_demand.txt` verified across all four streams — Core Skills, Specialist
Skills, Labour agreement and Subsequent entrant — with stay and cost figures matching.

---

## Is five documents enough?

**Recommendation only. Corpus growth is a team decision, and adding documents after judging
begins invalidates the judgements.**

Five is defensible and I would not expand it lightly. Two candidates, if the team wants a stronger
cross-subclass axis:

- **Subclass 462 (Work and Holiday).** The strongest case. Condition 8547 applies to 462 *and*
  417, and our WHM document says so, but 462 has no page in the corpus. A question about 462 is
  currently answerable only through a passing mention.
- **The student visa work-conditions page.** Would deepen coverage of the 48-hour cap and its
  exceptions rather than adding a new subclass.

Against expanding: the brief asks for a handful of documents, not thousands, and every added
document multiplies the judgement grid — 26 chunks per question already means 338 rows.

**If anything is added, do it before T02 starts.** Afterwards it means re-judging.

---

## What was verified, and how

| Check | Result |
|---|---|
| Manifest is deterministic — same hashes on repeated runs | ✅ |
| A single changed character changes the hash | ✅ verified on a copy |
| `--verify` reports no drift on the unchanged corpus | ✅ |
| `--verify` *detects* a changed document | ✅ tested by modifying and restoring |
| Exit codes: 0 clean, 1 on drift | ✅ suitable for an automated check |
| All five source URLs opened and compared in a browser | ✅ |

The fourth check is not in the task brief. The brief only asks that `--verify` reports no drift
when there is none, which a function that always prints "ok" would pass. Confirming it catches
real drift is the check that matters.
