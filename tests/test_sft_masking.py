"""Tests for SFT tool-call masking spans."""
from utils.sft_train import locate_fact_ranges


def test_masks_angle_bracket_facts():
    text = (
        'Reasoning.\n<fact> <Paris> <capital of> <France> . </fact>\n'
        '<count> <France> <border>\n= 8\n</count>\nThe answer.'
    )
    spans = locate_fact_ranges(text)
    masked = "".join(text[start:end] for start, end in spans)
    assert "<fact> <Paris> <capital of> <France> . </fact>" in masked
    assert "<count> <France> <border>\n= 8\n</count>" in masked
    assert "Reasoning." not in masked


def test_masks_only_object_of_angle_bracket_fact():
    text = "<fact> <Paris> <capital of> <France> . </fact>"
    spans = locate_fact_ranges(text, mask_objects=True)
    assert len(spans) == 1
    start, end = spans[0]
    assert text[start:end] == "France"


def test_masks_legacy_fact_lines():
    text = "Reasoning.\nFact: <Paris> <capital of> <France> .\nAnswer: Paris"
    spans = locate_fact_ranges(text)
    masked = "".join(text[start:end] for start, end in spans)
    assert "Fact: <Paris> <capital of> <France> ." in masked
    assert "Answer: Paris" not in masked
