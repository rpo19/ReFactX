"""Stage 4 test suite for the rule constraint.

  1. gold rules      -- accepted char-by-char, can_complete at every prefix
  2. invalid rules   -- die (or get stuck) at the offending position
  3. SOUNDNESS       -- every rule the constraint enumerates is accepted by
                        Nemo's validator (constraint => engine-valid)
  4. tokenizer test  -- with a real BPE tokenizer + logits mask: the gold
                        rule is reconstructible token-by-token through the
                        mask, and bad continuations are masked.

Usage:  python3 stage4/test_constraint.py
"""
import os
import sys
import time

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "stage3"))

from rule_constraint import RuleConstraint          # noqa: E402
from logits_processor import (RuleDecodingError, RuleLogitsProcessor,  # noqa: E402
                              RuleTokenIndex)
from tasks import TASKS                             # noqa: E402

SCHEMA = {"OWN": 3, "ROLE": 3, "REACHABLE": 2, "QUALIFIED_HOLDINGS": 2,
          "INFLUENCE": 2, "CONNECTED": 2, "CONTROL": 2,
          "ULTIMATE_CONTROL": 2, "INDIRECT_ROLE": 2}

TASK_SCHEMA = {}     # task id -> schema (context rules included)
for t in TASKS:
    ctx = {}
    for rule in t["context"]:
        pred = rule.split("(")[0].strip()
        ctx[pred] = rule.split(":-")[0].count("?")   # vars in the head
    TASK_SCHEMA[t["id"]] = dict(SCHEMA, **ctx)

INVALID = [
    "owner_of_owner(?x, ?z) :- bar(?x, ?y), OWN(?y, ?z, ?q) .",
    "owner_of_owner(?x, ?z) :- OWN(?x, ?y) .",
    "owner_of_owner(?x, ?z) :- OWN(?x, ?y, ?p) .",
    "owner_of_owner(?x, ?z) :- 5 > 3 .",
    "owner_count(?y, #count(?x), #sum(?p)) :- OWN(?x, ?y, ?p) .",
    "top_stake(?y, #max(?p)) :- OWN(?x, ?y, ?p), ~OWN(?z, ?y, ?q) .",
    "owner_of_owner(?x, ?z) :- OWN(?x, ?y, ?p), ?q > 50 .",  # ?q unbound
]


def walk(rule, head=None, arity=2, schema=SCHEMA, label=""):
    """Char-by-char walk; returns (status, at_pos)."""
    c = RuleConstraint(head=head or rule.split("(")[0],
                       head_arity=arity, schema=schema,
                       entities=["Abatantuono Group", "Foo SPA"])
    s = c.initial
    for i, ch in enumerate(rule):
        s = c.step(s, ch)
        if not s:
            return "DEAD", i
        if not c.can_complete(s):
            return "STUCK", i
    return ("ACCEPTING" if c.accepting(s) else "INCOMPLETE"), len(rule)


def test_gold():
    print("=" * 70 + "\n1. gold rules must be accepted\n" + "=" * 70)
    for t in TASKS:
        status, pos = walk(t["gold_rule"], head=t["head"],
                           arity=2, schema=TASK_SCHEMA[t["id"]])
        ok = status == "ACCEPTING"
        print("   %-16s %s %s" % (t["id"], status,
                                  "(at %d)" % pos if not ok else ""))
        assert ok, t["id"]


def test_invalid():
    print("=" * 70 + "\n2. invalid rules must die or get stuck\n" + "=" * 70)
    for rule in INVALID:
        head = rule.split("(")[0]
        arity = 2 if head != "owner_count" else 2
        status, pos = walk(rule, head=head, arity=arity)
        print("   %-12s %-8s at %d   %s" % (status, head, pos, rule[:52]))
        assert status in ("DEAD", "STUCK"), rule


def test_soundness(limit=300):
    """Enumerate rules the constraint allows; Nemo must accept every one."""
    print("=" * 70 + "\n3. soundness vs Nemo's validator\n" + "=" * 70)
    small_schema = {"OWN": 3, "ROLE": 3, "REACHABLE": 2}
    c = RuleConstraint("owner_of_owner", 2, small_schema,
                       entities=["Abatantuono Group", "Foo SPA"],
                       max_body_atoms=2, max_number_digits=1,
                       alphabet=("?x", "?y", "?p"))
    rules = c.generate(limit=limit, max_len=110)
    assert rules
    print("   enumerated %d rules from the constraint" % len(rules))

    from nmo_python import load_string
    from nemo_reasoner import build_program, control_rounds, DATA_DIR
    triples = pd.read_csv(f"{DATA_DIR}/triples.csv", keep_default_na=False,
                          low_memory=False)
    base = build_program(control_rounds(triples) + 3, DATA_DIR)

    t0, bad = time.time(), []
    for rule in rules:
        try:
            load_string(base + rule + "\n@output owner_of_owner .\n")
        except Exception as e:
            bad.append((rule, str(e)[:120]))
    print("   Nemo accepted %d/%d in %.1fs" %
          (len(rules) - len(bad), len(rules), time.time() - t0))
    for rule, err in bad[:5]:
        print("   REJECTED: %s\n      %s" % (rule, err))
    assert not bad, "soundness violated: constraint allowed a Nemo-invalid rule"


def test_tokenizer():
    print("=" * 70 + "\n4. tokenizer end-to-end (mask never blocks the gold)\n"
          + "=" * 70)
    from transformers import AutoTokenizer
    tok_path = os.environ.get("REFACTX_TEST_TOKENIZER", "gpt2")
    tok = AutoTokenizer.from_pretrained(tok_path)
    print("   tokenizer:", tok_path, "vocab", tok.vocab_size)

    t = TASKS[0]
    c = RuleConstraint(t["head"], 2, TASK_SCHEMA[t["id"]],
                       entities=pd.read_csv(
                           f"{os.path.join(os.path.dirname(os.path.dirname(
                               os.path.abspath(__file__))), 'data')}/entities.csv",
                           keep_default_na=False).entity.tolist())
    token_index = RuleTokenIndex(tok)
    prompt = tok.encode("User: Write the rule. <rule>\nAssistant:")
    proc = RuleLogitsProcessor(token_index, c, len(prompt))
    ids = prompt.copy()
    scores = torch.zeros(len(tok))
    gold = tok.encode(t["gold_rule"], add_special_tokens=False)
    for token_id in gold:
        masked = proc(torch.tensor([ids]), scores)
        assert torch.isfinite(masked[token_id])
        assert not torch.isfinite(masked).all()
        ids.append(token_id)
    masked = proc(torch.tensor([ids]), scores)
    assert torch.isfinite(masked).nonzero(as_tuple=True)[0].tolist() == [tok.eos_token_id]

    prefix = tok.encode("owner_of_owner(?x, ?z) :- ", add_special_tokens=False)
    masked = proc(torch.tensor([prompt + prefix]), scores)
    allowed = torch.isfinite(masked).nonzero(as_tuple=True)[0].tolist()
    assert allowed
    assert all(not token_index.strings[i].startswith("b") for i in allowed)

    try:
        proc(torch.tensor([prompt + tok.encode("bad")]), scores)
    except RuleDecodingError:
        pass
    else:
        raise AssertionError("a dead generated prefix must stop constrained decoding")

    numeric = "owner_of_owner(?x, ?z) :- OWN(?x, ?z, ?p), ?p >= 0.01"
    prefix = tok.encode(numeric, add_special_tokens=False)
    masked = proc(torch.tensor([prompt + prefix]), scores)
    dot = tok.encode(" .", add_special_tokens=False)
    assert len(dot) == 1 and torch.isfinite(masked[dot[0]])
    assert not c.accepting(c.consume(c.initial, numeric[:-2] + " ."))


if __name__ == "__main__":
    test_gold()
    test_invalid()
    test_soundness()
    test_tokenizer()
    print("\nALL STAGE-4 TESTS PASSED")
