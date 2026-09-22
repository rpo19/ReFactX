# Error categorization report

This report applies the taxonomy in `ERROR_CATEGORIZATION_GUIDE.md` to the following evaluation logs:

- `logs/mintaka_test_eval_qwen35_4b_grpo_checkpoint720.out.0`
- `logs/twowikimultihop_test_eval_qwen35_4b_grpo_checkpoint720.out.1`

The logs were inspected using the guide's JSONL workflow and the table-generation utility:

```bash
python utils/make_latex_tables.py \
  logs/twowikimultihop_test_eval_qwen35_4b_grpo_checkpoint720.out.1 \
  logs/mintaka_test_eval_qwen35_4b_grpo_checkpoint720.out.0 \
  --output tables.md
```

Each log contains 200 evaluated examples. Mechanical truncation was classified first and was not double-counted as a parser or semantic error.

## Summary

| Primary category | Mintaka | 2Wiki validation | Total |
|---|---:|---:|---:|
| Generation truncation / non-termination | 119 | 30 | 149 |
| Answer-format / parser failure | 0 | 1 | 1 |
| Abstention / unsupported-answer behavior | 3 | 0 | 3 |
| Retrieval failure / missing supporting triple | 6 | 9 | 15 |
| Evidence selection / grounding failure | 0 | 1 | 1 |
| Multi-hop / compositional reasoning failure | 2 | 1 | 3 |
| Entity resolution / ambiguity failure | 1 | 2 | 3 |
| Answer extraction / normalization mismatch | 16 | 31 | 47 |
| Question/dataset or temporal inconsistency | 4 | 0 | 4 |
| **Total failed examples** | **151** | **75** | **226** |
| Correct according to evaluator | 49 | 125 | 174 |

## Generation truncation / non-termination

This is the dominant failure mode:

- **Mintaka:** 119/200 examples
- **2Wiki validation:** 30/200 examples
- **Total:** 149/400 examples

All 149 cases had:

- `reached_max_tokens: true`
- `answer_complete: false`
- `prediction: null`
- generation at the configured token limit

These cases were treated as generation failures rather than parser or reasoning failures.

## Mintaka errors

### Answer extraction / normalization mismatches

Representative examples:

- **Example 9:** `July 9, 1981` vs reference `1981`.
- **Example 154:** `August 2, 1970` vs `2-Aug-70`.
- **Example 164:** `170 cm` vs `5'7"`.
- **Example 172:** `United States` vs `USA`.

These answers are often semantically equivalent but were rejected by the evaluator's normalization rules.

### Abstention / unsupported-answer behavior

Examples 22, 38, and 180 returned “I don't know” despite apparently sufficient evidence. Example 181 was classified as a retrieval failure because the retrieved evidence did not contain the required child-count fact.

### Retrieval failure / missing supporting triple

- **Example 47:** Retrieved evidence exposes four Michael Bay-directed *Transformers* films while the reference expects five.
- **Example 57:** No retrieved cast link connects Robert De Niro to a *Godfather* film.
- **Example 71:** Retrieved award facts are incomplete or inconsistent with the reference count of 11.
- **Example 156:** Only two Uma Thurman children are retrieved while the reference expects three.
- **Example 179:** Retrieved evidence identifies Michael Biehn instead of Arnold Schwarzenegger as the relevant *Terminator* actor.

### Multi-hop / compositional reasoning failure

- **Example 30:** The dates are correctly retrieved, but the model reverses the requested “before/after” relationship.
- **Example 176:** The model retrieves Liam Neeson's birth date but returns the date instead of calculating the requested age.

### Entity resolution / ambiguity failure

- **Example 155:** “Lead actress of *Emma*” is resolved to Kate Beckinsale. The retrieved evidence supports one child, while the reference expects two, making the title/lead-actor interpretation ambiguous.

### Question, dataset, or temporal inconsistency

- **Example 23:** Retrieved heights support a tie, while the reference selects Brad Pitt.
- **Example 53:** Retrieved evidence suggests four *John Wick* films involving Ian McShane, while the reference expects three.
- **Example 141:** Both *The Aviator* and *The Wolf of Wall Street* plausibly satisfy the wording, but only the latter is retrieved.
- **Example 193:** The model computes age using a 2026 date and returns 79, while the reference says 74; this appears time-dependent.

## 2Wiki validation errors

### Answer extraction / normalization mismatches

Representative examples:

- **Example 3:** `June 12, 1516` vs `12 June 1516`.
- **Example 22:** `American` vs `United States`.
- **Example 60:** `Winnipeg` vs `Winnipeg, Manitoba`.
- **Example 181:** `Aarhus` vs `Århus`.
- **Example 200:** `Immacolata` vs `Immaculata`.

Many of these outputs are arguably correct but differ from the reference in alias, date format, geographic granularity, punctuation, or expected answer scope.

### Answer-format / parser failure

- **Example 113:** The model returns both `Southfield High School` and `New York University Tisch School of the Arts`, while the reference expects only the latter institution. This is an over-answer or multi-answer extraction failure.

### Retrieval failure / missing supporting triple

- **Example 10:** Retrieved evidence conflicts with the supporting evidence about Cleopatra IV's father.
- **Example 42:** Retrieved evidence gives `Kingdom of Württemberg`, while the supporting evidence says `German`.
- **Example 81:** Retrieved evidence gives Hisham Zaman's birthplace as `Iraq`, while the supporting evidence/reference says `Kurdish`.
- **Example 104:** The correct Viktoriya Zeynep Güneş entity is not retrieved; the model sees unrelated Viktoriya entries and abstains.
- **Examples 114, 121, 148, and 150:** Supplied triples conflict with the supporting evidence or reference, including conflicts such as India vs. Geumgwan Gaya and China vs. Qatar.

### Entity resolution / ambiguity failure

- **Example 56:** `Otrag` is resolved as `Otago`, yielding New Zealand instead of the gold entity OTRAG in Germany.
- **Example 75:** `ZTR Control Systems` is resolved as the unrelated Czech company `Ztronic` instead of the Canada-linked gold entity.

### Evidence selection / grounding failure

- **Example 106:** The supplied triple explicitly states that Anthony Weiner's employer is Chuck Schumer, but the model answers with the U.S. House of Representatives based on his position.

### Multi-hop / compositional reasoning failure

- **Example 91:** The model compares death years rather than comparing the subjects' lifetimes; the supporting evidence shows that Edward Dearle lived longer.

## Interpretation

The most important issue is generation length: 149 of the 226 failed examples never reached a complete answer. Any analysis of model reasoning should therefore separate those cases from the 77 non-truncated failures.

Among non-truncated failures, the largest group is answer normalization or extraction mismatch, especially in the 2Wiki validation log. Retrieval conflicts and missing supporting triples are also common, particularly when the supplied triples disagree with the dataset's supporting evidence or reference answer.

The normalization category is evaluator-sensitive. Several model outputs are semantically reasonable but differ from the reference in formatting, aliases, granularity, or answer cardinality. Similarly, retrieval and dataset-issue classifications should be interpreted cautiously when the retrieved triples contradict the ground-truth evidence.
