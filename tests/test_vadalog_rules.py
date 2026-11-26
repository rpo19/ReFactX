"""Tests for the Vadalog <rule> mode and the full mode (refactx.rules)."""
import math
import os

import pytest

from refactx.rules.vadalog_constraint import VadalogRuleConstraint

SCHEMA = {"own": ("entity", "entity", "number"), "role": ("entity", "entity", "role"),
          "control": ("entity", "entity")}
CONSTS = {"entity": ["Acme", "Beta SPA", 'Q "x"'], "role": ["ceo", "director"]}


@pytest.fixture(scope="module")
def c():
    return VadalogRuleConstraint(SCHEMA, CONSTS)


GOOD = [
    'owner_count(Y, N) :- own(X, Y, P), N = count(X).',
    ' top(Y, M) :- own(X, Y, P), M = max(P).',
    'owner_of_owner(X, Z) :- own(X, Y, P), own(Y, Z, Q).',
    'maj(X, Y) :- own(X, Y, P), P > 50.',
    'q(X) :- control(X, "Acme").',
    'q(X) :- own(X, "Q \\"x\\"", _).',
    'ind(X, Y) :- control(X, Y), not own(X, Y, _).',
    'ceo_of(X, Y) :- role(X, Y, "ceo").',
    'tot(X, S) :- own(X, Y, P), S = sum(P).',
    'n(N) :- control("Acme", Y), N = count(Y).',
]

BAD = {  # rule -> the prefix at which it is blocked
    'q(X) :- owns(X, Y, P).': 'q(X) :- owns',                      # unknown predicate
    'q(X) :- own(X, Y).': 'q(X) :- own(X, Y)',                      # arity
    'q(X, Z) :- own(X, Y, P).': 'q(X, Z) :- own(X, Y, P).',         # unbound head var
    'q(X) :- own(X, Y, P), Y > 3.': 'q(X) :- own(X, Y, P), Y >',    # order on entities
    'q(X) :- own(X, Y, P), control(P, Y).': 'q(X) :- own(X, Y, P), control(P',  # type clash
    'q(X) :- control(X, "Nope").': 'q(X) :- control(X, "N',         # unknown entity
    'q(X) :- role(X, Y, "cfo").': 'q(X) :- role(X, Y, "cf',         # unknown role
    'own(X, Y) :- control(X, Y).': 'own(',                          # existing predicate
    'is_x(X) :- control(X, Y).': 'is_x(',                           # inverse prefix
    'q(X) :- control(X, Y), not own(Z, Y, _).': 'q(X) :- control(X, Y), not own(Z',  # unsafe
    'q(X, N) :- control(X, Y), N = sum(Y).': 'q(X, N) :- control(X, Y), N = sum(Y',  # sum entity
    'q(X) :- control(X, Y), N = count(Y).': 'q(X) :- control(X, Y), N',  # agg not in head
}


@pytest.mark.parametrize("rule", GOOD)
def test_accepts_valid_rules(c, rule):
    assert c.accepts(rule)
    # every prefix of a valid rule must be completable
    subset = c.initial
    for ch in rule[:-1]:
        subset = c.step(subset, ch)
        assert c.can_complete(subset)


@pytest.mark.parametrize("rule,blocked_at", BAD.items())
def test_rejects_invalid_rules_at_the_right_char(c, rule, blocked_at):
    assert not c.accepts(rule)
    assert rule[:c.first_error(rule) + 1] == blocked_at


def test_body_offers_only_schema_and_bound_vars(c):
    s = c.consume(c.initial, "q(X, Y) :- own(X, Y, P), ")
    chars = c.next_chars(s)
    assert {"o", "r", "c", "n", "X", "Y", "P"} <= chars
    assert "Z" not in chars and "Q" not in chars


def test_fixed_head():
    c = VadalogRuleConstraint(SCHEMA, CONSTS, head="owner_count", head_arity=2,
                              variables="XYNP", max_body_literals=2)
    assert c.accepts("owner_count(Y, N) :- own(X, Y, P), N = count(X).")
    assert not c.accepts("other(Y, N) :- own(X, Y, P), N = count(X).")
    assert not c.accepts("owner_count(Y) :- own(X, Y, P).")
    assert c.generate(limit=3)


# ------------------------------------------------------------------ VadalogKG
vadalog = pytest.importorskip("vadalog")


@pytest.fixture()
def kg():
    from refactx.rules.vadalog_kg import VadalogKG
    program = """
    ctl(X, X) :- own(X, Y, P).
    ctl(X, Y) :- ctl(X, Z), own(Z, Y, P), S = msum(P, <Z>), S > 50.
    control(X, Y) :- ctl(X, Y), X != Y.
    """
    base = {"own": [("a", "b", 60.0), ("b", "c", 30.0), ("a", "c", 25.0), ("d", "c", 45.0)],
            "role": [("p", "a", "ceo")]}
    return VadalogKG(program, base, SCHEMA, ["control"], ["a", "b", "c", "d", "p"],
                     constants={"role": ["ceo"]}, verbose=False)


def test_kg_derives_recursive_control(kg):
    assert sorted(kg.rows("control")) == [("a", "b"), ("a", "c")]   # 25 + 30 > 50
    assert kg.properties("a") == ["control", "is_role", "own"]
    assert kg.properties("c") == ["is_control", "is_own"]
    assert kg.get("c", "is_control") == ["a"]


def test_add_rule_registers_a_new_relation(kg):
    res = kg.add_rule("n_owners(Y, N) :- own(X, Y, P), N = count(X).")
    assert res.ok and res.registered and res.types == ("entity", "number")
    assert "n_owners" in kg.properties("c") and kg.get("c", "n_owners") == ["3"]
    assert "n_owners" in kg.schema()
    # a later rule may use it
    assert kg.add_rule("crowded(Y) :- n_owners(Y, N), N > 2.").rows == [("c",)]
    kg.reset_rules()
    assert "n_owners" not in kg.schema() and "n_owners" not in kg.properties("c")


@pytest.mark.parametrize("rule,error", [
    ("q(X) :- nope(X).", "unknown predicate"),
    ("q(X, Z) :- own(X, Y, P).", "not bound"),
    ("own(X, Y) :- control(X, Y).", "already a predicate"),
])
def test_add_rule_errors(kg, rule, error):
    res = kg.add_rule(rule)
    assert not res.ok and error in res.error and not res.registered


# ------------------------------------------------------- token-level <rule>
def _tokenizer():
    from transformers import AutoTokenizer
    name = os.environ.get("REFACTX_TEST_TOKENIZER", "gpt2")
    try:
        return AutoTokenizer.from_pretrained(name)
    except Exception as exc:  # pragma: no cover - offline
        pytest.skip("tokenizer unavailable: %s" % exc)


def test_token_mask_allows_gold_tokenization(kg):
    from refactx.rules.token_mask import SortedVocab
    tok = _tokenizer()
    vocab = SortedVocab(tok)
    c = kg.rule_constraint()
    rule = ' n(N) :- control("a", Y), N = count(Y).'
    subset, cc = c.initial, {}
    for tid in tok.encode(rule):
        allowed = set(vocab.allowed(c, subset, cc))
        assert tid in allowed, (tok.decode([tid]), rule)
        for other in list(allowed)[:50]:           # soundness on a sample
            assert c.can_complete(c.consume(subset, vocab.spelling[other]))
        subset = c.consume(subset, vocab.spelling[tid])
    assert c.accepting(subset)


def test_rule_generation_runs_vadalog_and_forces_result(kg):
    torch = pytest.importorskip("torch")
    from refactx.generate import PatternConstrainedState
    from refactx.index import DictIndex
    from refactx.rules.generation import RuleGeneration
    tok = _tokenizer()
    state = PatternConstrainedState(tok, DictIndex(), DictIndex())
    gen = RuleGeneration(state, tok, 0, kg=kg)
    seq = tok.encode(' big(X) :- own(X, "c", P), P > 40.')
    vocab = len(tok)
    for i in range(len(seq)):                       # every gold token is allowed
        mask = torch.full((1, vocab), -math.inf)
        gen.constrain(seq[:i], mask, 0)
        assert mask[0, seq[i]] == 0
    forced = []
    while not gen.done:                             # then the tail is forced
        mask = torch.full((1, vocab), -math.inf)
        gen.constrain(seq + forced, mask, 0)
        allowed = (mask[0] == 0).nonzero().flatten().tolist()
        assert len(allowed) == 1
        forced.append(allowed[0])
    tail = tok.decode(forced)
    assert tail.startswith(" </rule>") and 'big("d")' in tail and "<result>" in tail
    assert gen.result.ok and "big" in kg.schema()


def test_segments():
    from refactx.rules.full_mode import segments
    text = ("I check. <kg> <a> <own> <b> . </kg>\nthen <rule> q(X) :- own(X, \"b\", _). </rule>\n"
            "<result> 1 fact: q(\"a\") </result>\n<answer>\n[\"a\"]\n</answer>")
    kinds = [k for k, _ in segments(text)]
    assert kinds == ["free", "kg", "free", "rule", "free", "result", "free", "answer"]
