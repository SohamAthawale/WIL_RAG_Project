# Test-Driven RAG for Australian Visa Conditions & Work Rights

**RMIT Work Integrated Learning Project — Group ID 97**
Course: Case Studies in Data Science (COSC2669 / COSC2816)

---

## Project Aim

> As an **international student or temporary visa holder in Australia**, I want to **ask
> plain-English questions about my visa's work rights and get an answer tied to my own
> subclass with its source cited — or an honest refusal**, so that **I do not breach a visa
> condition by acting on a rule that belongs to a different subclass or has since changed.**

## Why this project is an evaluation, not a chatbot

Having a workable RAG pipeline is necessary but not sufficient. The deliverable of this
project is the **evaluation framework**: eight measured dimensions across four baseline
configurations, benchmarked against the Walert methodology (CHIIR '24).

The headline metric is **cross-subclass confusion rate** — the percentage of answers drawn
from the wrong visa subclass. Telling a Subclass 485 holder they are capped at 48 hours per
fortnight, or telling a Subclass 500 holder they are not, are both visa breaches.

## Team — Group 97

| Student ID | Full name | Workstream (proposed — confirm at Sprint 1 standup) | % contribution |
|---|---|---|---|
| s4233801 | Soham Chaitanya Athawale | Pipeline & retrieval | TBC |
| s4216323 | Hitesh Subhash Chaudhari | Test set & annotation | TBC |
| s4183976 | Manoj Mahadev Bhosale | Metrics & statistics | TBC |
| s4178063 | Vaishnavi Vijayanand Joshi | Corpus & data governance | TBC |
| s4218211 | Heet Harshad Chanchad | Ethics, standards & report | TBC |
| s4203473 | Glory Suresh Vanjare | UI/demo & process stewardship | TBC |

Contact: `<student-id>@student.rmit.edu.au`

> **Workstreams above are a proposal, not a decision.** Confirm ownership at the Sprint 1
> standup and update this table. The `% contribution` column is a Milestone 1 rubric
> requirement and must be agreed by all six members before submission.

## Domain & knowledge base

Australian temporary visa conditions and work rights. Five official Department of Home
Affairs pages, each stored with its source URL and retrieval date:

| Document | Subclass |
|---|---|
| `subclass_500_student.txt` | 500 — Student |
| `subclass_485_temporary_graduate.txt` | 485 — Temporary Graduate |
| `subclass_482_skills_in_demand.txt` | 482 — Skills in Demand |
| `subclass_417_working_holiday.txt` | 417 — Working Holiday |
| `whm_6_month_work_limitation.txt` | Condition 8547 (WHM work limitation) |

The knowledge base is deliberately small — a handful of documents, not thousands.

## Evaluation dimensions

| # | Dimension | Metric |
|---|---|---|
| 1 | Answerability | % unanswered, **split** into should-have-answered vs correctly refused |
| 2 | Faithfulness | Every claim traceable to a retrieved chunk |
| 3 | Attribution correctness | Cites the right document *and* the right subclass |
| 4 | Retrieval quality | NDCG / Recall@k / MRR vs annotated gold chunks |
| 5 | **Cross-subclass confusion** | % answered from the wrong subclass — headline |
| 6 | Refusal compliance | % of case-specific questions correctly declined |
| 7 | Vocabulary fairness | Accuracy across formal / plain / simplified-English registers |
| 8 | Temporal correctness | % reflecting current vs superseded rules |

## Baseline configurations

| ID | Configuration | Status |
|---|---|---|
| B0 | Bare LLM, no retrieval | Implemented |
| B1 | BM25 + LLM (Walert sparse) | Not started |
| B2 | Dense retrieval + LLM (Walert DPR) | Implemented |
| B3 | Hybrid + reranking | Not started |

## Setup

Requires [Ollama](https://ollama.com) running locally. Zero-cost: no paid API is used.

```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:7b-instruct

python3 -m venv .venv
source .venv/bin/activate
pip install numpy requests
```

Build the vector store (embeds the corpus — run once, or after changing chunking):

```bash
cd src && python ingest.py
```

Run the B0-vs-RAG comparison over the test set:

```bash
cd src && python eval.py
```

Outputs land in `results/` as both CSV (for hand annotation) and JSON.

## Repository layout

```
data/documents/    5 source documents, each with source URL + retrieval date
data/testset/      test_set.json — labelled questions with gold answers
data/vector_store.json   embedded chunks (flat JSON — no vector DB at this scale)
src/ingest.py      chunking + embedding
src/rag.py         retrieval, generation, and the B0 no-retrieval baseline
src/eval.py        runs the test set through B0 and RAG → comparison table
docs/              planning, sprint tasks, ethics research
results/           evaluation outputs
```

## Method reference

Pathiyan Cherumanal, Tian, et al., *"Walert: Putting Conversational Information Seeking
Knowledge into Action by Building and Evaluating a Large Language Model-Powered Chatbot"*,
CHIIR '24. Paper: https://arxiv.org/abs/2401.07216 · Code: https://github.com/rmit-ir/walert

We reproduce Walert's **method**, not its stack: their pipeline was cloud-hosted and paid
(Falcon-7B on SageMaker), ours is local and free (`qwen2.5:7b-instruct` via Ollama). Falcon-7B
and Qwen2.5-7B-Instruct are comparable open 7B instruction-tuned models, so the substitution
is methodologically defensible.

## Academic integrity

This course gates AI use per assessment. AI assistance is used for process purposes only —
scaffolding, checks, and critique. All analysis, design decisions, correctness judgements and
written argument are authored and owned by the team members. Every submission carries the
completed **Condition 3: Bounded Use of AI** declaration with prompt logs.
