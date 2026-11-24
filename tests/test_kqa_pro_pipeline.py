from utils.sample_kqa_pro import answer_type, primary_type, program_flags
from utils.mix_kqa_sft import is_mintaka_count


def test_program_flags():
    program = [
        {"function": "Find", "inputs": ["France"]},
        {"function": "Count", "inputs": []},
    ]
    flags = program_flags(program)
    assert flags["has_count"] is True
    assert flags["num_relate"] == 0
    assert primary_type(flags) == "count"


def test_qualifier_wins_over_count():
    flags = program_flags([
        {"function": "Count"}, {"function": "QueryRelationQualifier"},
    ])
    assert primary_type(flags) == "qualifier"


def test_verify_and_multihop():
    assert primary_type(program_flags([{"function": "VerifyStr"}])) == "verify"
    assert primary_type(program_flags([
        {"function": "Relate"}, {"function": "Relate"},
    ])) == "multihop"


def test_answer_types():
    assert answer_type(1979) == "integer"
    assert answer_type("yes") == "yesno"
    assert answer_type("2008-01-01") == "date"
    assert answer_type("France") == "string"


def test_mintaka_count_detection():
    assert is_mintaka_count({"dataset": "mintaka", "complexityType": "count"})
    assert is_mintaka_count({"dataset": "mintaka", "question_type": "count"})
    assert not is_mintaka_count({"dataset": "kqa_pro", "question_type": "count"})
    assert not is_mintaka_count({"dataset": "mintaka", "complexityType": "yesno"})
