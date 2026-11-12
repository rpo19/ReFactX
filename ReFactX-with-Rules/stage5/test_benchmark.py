"""Regression coverage for candidate scoring and runtime rule registration."""
import pytest

from eval_rules import Eval, TASKS, generate
from nemo_reasoner import Reasoner
from rule_harness import RuleBench

import pandas as pd
import torch


@pytest.fixture
def triples():
    return pd.DataFrame(
        [("A", "OWN", "B", "40"), ("A", "OWN", "C", "40"),
         ("D", "OWN", "B", "40"), ("A", "ROLE", "B", "director"),
         ("A", "REACHABLE", "B", "")],
        columns=["subject", "predicate", "object", "value"],
    )


def test_scoring_distinguishes_bad_rules_and_results(triples):
    ev = Eval.__new__(Eval)
    ev.bench = RuleBench(triples)
    ev.golds = {"total_owned": {("A", 80), ("D", 40)}}
    task = next(t for t in TASKS if t["id"] == "total_owned")
    gold = task["gold_rule"]
    assert ev.score(gold, task)["outcome"] == "correct"
    assert ev.score(gold.replace("#sum(?p, ?y)", "#sum(?p)"), task)["outcome"] == "valid_wrong"
    assert ev.score(gold + " other(?x) :- OWN(?x, ?y, ?p).", task)["failure_stage"] == "extraction"
    assert ev.score(gold[:-1], task)["failure_stage"] == "extraction"
    assert ev.score(gold.replace("total_owned", "other"), task)["failure_stage"] == "validation"
    assert ev.score(gold.replace("#sum(?p, ?y)", "?p, ?y"), task)["failure_stage"] == "validation"
    assert ev.score(gold.replace("OWN(", "shareholder("), task)["failure_stage"] == "validation"
    assert ev.score(
        "total_owned(x, s) :- OWN(x, y, p), #sum(p) = s, #count(y) > 0.",
        task,
    )["error"] == "aggregates must appear in the head"

    with_literal = 'total_owned(?x, #sum(?p, ?y)) :- OWN(?x, "A, Inc. (#)", ?p), OWN(?x, ?y, ?p), ?p >= 0.01.'
    assert ev.score(with_literal, task)["outcome"] == "valid_wrong"


def test_engine_failure_is_not_a_wrong_result(triples):
    bench = RuleBench(triples)
    rule = "bad(?x, ?y) :- OWN(?x, ?y, ?p), ~bad(?x, ?y) ."
    result = bench.check(rule, "bad", gold=set())
    assert result["outcome"] == "engine_error"
    assert result["failure_stage"] == "execution"


def test_registration_retains_previous_rules_and_rolls_back(triples):
    reasoner = Reasoner(triples, unroll_margin=0)
    reasoner.add_rule("first(?x, ?y) :- OWN(?x, ?y, ?p) .")
    reasoner.add_rule("second(?x, ?y) :- first(?x, ?y) .")
    assert {"first", "second"} <= reasoner.properties("A")
    assert reasoner.get("A", "first") == ["B", "C"]
    assert reasoner.get("A", "second") == ["B", "C"]
    old_engine = reasoner.engine
    with pytest.raises(Exception, match="stratified"):
        reasoner.add_rule("broken(?x, ?y) :- OWN(?x, ?y, ?p), ~broken(?x, ?y) .")
    assert reasoner.engine is old_engine
    assert reasoner.count("A", "first") == 2
    reasoner.add_rule("third(?x, ?y) :- second(?x, ?y) .")
    assert reasoner.ask("A", "third", "B")


def test_incomplete_constrained_generation_is_reported():
    class Tokenizer:
        chat_template = None
        eos_token_id = pad_token_id = 0

        def encode(self, prompt, return_tensors):
            return torch.tensor([[2, 3]])

        def decode(self, ids, clean_up_tokenization_spaces):
            return "total_owned"

    class Model:
        device = "cpu"

        def generate(self, ids, **kwargs):
            return torch.cat((ids, torch.tensor([[1]])), dim=1)

    from rule_constraint import RuleConstraint
    constraint = RuleConstraint("total_owned", 2, {"OWN": 3})
    result = generate(Model(), Tokenizer(), "prompt", constraint, token_index=object(),
                      max_new_tokens=1)
    assert result["termination"] == "token_limit"
    assert result["generation_error"] == "incomplete constrained rule (token_limit)"
