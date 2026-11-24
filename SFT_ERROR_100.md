# Error classes on 100-question eval logs

## eval100_kqa_maskobj_ckpt100_b8.out.0 (n=100)

| class | count | share |
|---|---:|---:|
| correct | 43 | 43% |
| missing_gold | 43 | 43% |
| count_no_tool | 9 | 9% |
| retrieved_wrong | 5 | 5% |

KQA Pro missing_gold breakdown (from gold program):
- entity_retrieved=True, relation_used=False: 11
- entity_retrieved=False, relation_used=False: 8
- entity_retrieved=False, relation_used=True: 7
- entity_retrieved=True, relation_used=True: 4

## eval100_2wiki_combined_b12.out.0 (n=100)

| class | count | share |
|---|---:|---:|
| correct | 49 | 49% |
| missing_gold | 29 | 29% |
| retrieved_wrong | 20 | 20% |
| no_answer | 2 | 2% |

## eval100_webqsp_clean200_b40.out.0 (n=100)

| class | count | share |
|---|---:|---:|
| no_answer | 52 | 52% |
| missing_gold | 30 | 30% |
| correct | 16 | 16% |
| retrieved_wrong | 1 | 1% |
| count_no_tool | 1 | 1% |

## eval100_cwq_combined_b12.out.0 (n=100)

| class | count | share |
|---|---:|---:|
| missing_gold | 72 | 72% |
| correct | 16 | 16% |
| retrieved_wrong | 7 | 7% |
| no_answer | 5 | 5% |

## Findings (from the classifier and the raw records)

- **WebQSP: 52% of answers are empty because generation never finishes.** All 52
  hit the 1,024-token limit after a median of 17 fact calls, while the model keeps
  writing "let me try to find…" prose. The decoder budget (40 calls in this config)
  does not stop this, because the loop is mostly narration between calls. Budget
  and token limit are being tested in `configs/eval100_webqsp_b12.json` and
  `configs/eval100_webqsp_b40_tok2048.json`.
- **2Wiki: 20% are "retrieved_wrong"**: the gold string is in the retrieved triples,
  but the answer differs. Spot checks show alias and format mismatches ("New York
  City" vs "New York") and real comparison errors ("neither film has the director
  who died later"). Strict scoring mixes these together.
- **KQA Pro: 43% of missing_gold** have the entity retrieved but never use the gold
  relation (11 of 30 missing-gold cases). Relation choice is the main gap, not
  entity lookup.
- **CWQ: 72% missing_gold** with few fact calls per question. Retrieval does not
  reach the answer at all; it needs entity and relation selection, not longer
  chains.
- Count without `<count>` is rare (≤9%).

The classifier is heuristic (substring match of the gold answer in the triples).
Treat the shares as approximate.
