# T13 test-set summary

## Result

The test set now contains **100 questions**:

| Category | Existing | Added | Final |
|---|---:|---:|---:|
| `answerable_direct` | 7 | 53 | 60 |
| `cross_subclass_probe` | 3 | 22 | 25 |
| `refusal_required` | 3 | 12 | 15 |
| **Total** | **13** | **87** | **100** |

All 87 new questions have `gold_chunk_ids: []`. The 12 previously populated gold lists were preserved.

## Corpus coverage

Questions were written against corpus snapshot `84636c1d6741` (24 chunks) using the five source documents and the line-by-line inventory in `results/corpus_inventory.md`.

The corpus is only 1,760 words and contains 83 locatable assertions, of which 22 were already represented by the original questions. Reaching 60 direct questions therefore uses most of the remaining factual surface area. The Subclass 500 document is especially thin: it provides a short overview, a single work-hours rule, and a compact eligibility list. The Subclass 482 document also repeats the same starting cost and similar stay limits across streams, which limits distinct direct questions. The Working Holiday documents provide more scenario variation because condition 8547 has several sector and location exemptions.

The final mix keeps the requested 60/25/15 ratio. Cross-subclass probes focus on the highest-risk confusions: the Student visa hours cap versus WHM same-employer limits, Subclass 485 versus 500 after graduation, employer-sponsored 482 versus unsponsored graduate work, and the similarly named Subclass 485 streams.

## Judging note

The new questions are intentionally unjudged. Populating their gold chunk lists will require a separate relevance-judging pass; at 87 new questions and 24 chunks, exhaustive judging would add 2,088 question-chunk decisions. Pooling retrieved chunks would reduce that workload but changes recall to pool-relative recall and should be recorded as an evaluation choice.
