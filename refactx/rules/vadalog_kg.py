"""A knowledge graph backed by the open-source Vadalog engine (vadalog-portable).

The KG is a Vadalog program plus its stored (base) relations.  Derived
relations are never stored in the fact index: Vadalog computes them by
reasoning (the chase) when the KG is built, and every rule registered later
is evaluated by Vadalog on demand.

The object offers the three interaction levels used by ReFactX:

1. ``entities()``         the entity index (what the prefix tree constrains);
2. ``properties(e)``      the property strings that have answers for ``e``:
                          stored relations, derived relations and session
                          rules alike (``own`` binds ``e`` as subject,
                          ``is_own`` as object);
3. ``get/count/ask``      the answers.

plus the hooks the ``<rule>`` mode needs:

* ``schema()``            {predicate: (type, ...)} for the rule constraint;
* ``rule_constraint()``   a :class:`VadalogRuleConstraint` over the current schema;
* ``add_rule(text)``      validate + evaluate a rule with Vadalog and register
                          its head as a new relation (visible to ``<kg>`` too).

A relation is exposed as a property through its first two columns.
"""
from __future__ import annotations

import hashlib
import os
import pickle
import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import RLock

from refactx.rules.vadalog_constraint import (ANY, NUMBER, VadalogRuleConstraint,
                                              build_trie, escape)

try:  # vadalog-portable: pip install git+https://github.com/joint-kg-labs/vadalog-portable
    from vadalog import Options, Runner, load_text
    from vadalog.errors import VadalogError
except ImportError:  # pragma: no cover - checked at construction time
    Options = Runner = load_text = None
    VadalogError = Exception


def _require_vadalog():
    if load_text is None:
        raise ImportError(
            "vadalog-portable is not installed: "
            "pip install git+https://github.com/joint-kg-labs/vadalog-portable")


def fmt_value(value):
    """How a value is shown to the model: 79.0 -> '79', strings unchanged."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def fmt_fact(pred, row):
    def term(v):
        return '"%s"' % escape(v) if isinstance(v, str) else fmt_value(v)
    return "%s(%s)" % (pred, ", ".join(term(v) for v in row))


class _Runner(Runner if Runner is not None else object):
    """A vadalog Runner whose input facts come from Python, not from files."""

    def __init__(self, program, facts, **kw):
        super().__init__(program, **kw)
        self._extra = facts

    def _load_facts(self):
        super()._load_facts()
        load = self.chase.load_fact
        for pred, rows in self._extra.items():
            for row in rows:
                load(pred, tuple(row))


@dataclass
class RuleResult:
    """Outcome of one rule sent to Vadalog."""
    rule: str
    head: str | None = None
    arity: int = 0
    types: tuple = ()
    rows: list = field(default_factory=list)
    ok: bool = False
    error: str | None = None
    registered: bool = False
    seconds: float = 0.0

    def summary(self, k=5):
        if not self.ok:
            return "error: %s" % self.error
        n = len(self.rows)
        shown = "; ".join(fmt_fact(self.head, r) for r in self.rows[:k])
        more = " ... and %d more not shown" % (n - k) if n > k else ""
        return "%d fact%s%s%s" % (n, "" if n == 1 else "s",
                                  ": " + shown if n else "", more)


class VadalogKG:
    """Reason over ``base`` relations with the Vadalog ``program``.

    Parameters
    ----------
    program:   Vadalog rules (no facts, no ``@input``: base relations are
               passed in ``base``; ``@output`` is added for ``derived``).
    base:      {predicate: rows} -- the stored relations.
    schema:    {predicate: (type, ...)} for every base and derived predicate.
    derived:   predicates computed by the program and exposed to the model.
    entities:  the entity index (names of type ``entity``).
    constants: {type: values} for other quoted-constant types (e.g. roles).
    symmetric: predicates whose inverse equals themselves (no ``is_`` form).
    views:     {property: relation} -- e.g. expose ``reachable`` through its
               symmetric closure ``reach`` while rules still see the stored one.
    cache:     optional path to a pickle of the reasoning result.
    """

    def __init__(self, program, base, schema, derived, entities, constants=None,
                 symmetric=(), views=None, cache=None, options=None, verbose=True):
        _require_vadalog()
        self.program = program.strip() + "\n"
        self.base = {p: [tuple(r) for r in rows] for p, rows in base.items()}
        self._schema = {p: tuple(t) for p, t in schema.items()}
        self.derived = list(derived)
        self._entities = sorted(set(entities))
        self.constants = {t: sorted(set(v)) for t, v in (constants or {}).items()}
        self.symmetric = set(symmetric)
        self.views = dict(views or {})
        self.options = options or Options()
        self.verbose = verbose
        self.session_rules = []           # [(head, rule text)]
        self._lock = RLock()
        self._tries = None
        t0 = time.perf_counter()
        self.relations = self._load_cache(cache)
        self.from_cache = self.relations is not None
        if self.relations is None:
            self.relations = self._reason_program()
            self._save_cache(cache)
        self.reason_seconds = time.perf_counter() - t0
        self.relations.update({p: rows for p, rows in self.base.items()})
        self._out, self._inv = {}, {}
        for p in list(self.relations):
            self._index(p)
        if verbose:
            n_der = sum(len(self.relations[p]) for p in self.derived)
            n_base = sum(len(r) for r in self.base.values())
            print("VadalogKG: %d stored facts, %d derived facts (%s in %.1fs)"
                  % (n_base, n_der, "cache" if self.from_cache else "Vadalog chase",
                     self.reason_seconds))

    # ------------------------------------------------------------- reasoning
    def full_program(self, outputs=True):
        out = "".join('@output("%s").\n' % p for p in self.derived) if outputs else ""
        return self.program + out

    def _reason_program(self):
        prog = load_text(self.full_program())
        runner = _Runner(prog, self.base, options=self.options)
        result = runner.run()
        self.stats = dict(result.stats)
        return {p: [tuple(r) for r in rows] for p, rows in result.relations.items()}

    def _cache_key(self):
        h = hashlib.sha1(self.full_program().encode())
        for p in sorted(self.base):
            h.update(p.encode())
            h.update(str(len(self.base[p])).encode())
            h.update(repr(self.base[p][:50]).encode())
        return h.hexdigest()

    def _load_cache(self, path):
        if not path or not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            key, rels = pickle.load(f)
        return rels if key == self._cache_key() else None

    def _save_cache(self, path):
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "wb") as f:
                derived = {p: r for p, r in self.relations.items() if p not in self.base}
                pickle.dump((self._cache_key(), derived), f)

    def _index(self, pred):
        out, inv = defaultdict(list), defaultdict(list)
        for row in self.relations[pred]:
            if len(row) >= 2:
                s, o = fmt_value(row[0]), fmt_value(row[1])
                out[s].append(o)
                inv[o].append(s)
            elif len(row) == 1:
                out[fmt_value(row[0])].append("true")
        self._out[pred] = {k: sorted(set(v)) for k, v in out.items()}
        self._inv[pred] = {k: sorted(set(v)) for k, v in inv.items()}

    # ----------------------------------------------------------------- schema
    def schema(self):
        return dict(self._schema)

    def entities(self):
        return list(self._entities)

    def constant_tries(self):
        """Tries for quoted constants (built once; ~100k entities)."""
        if self._tries is None:
            self._tries = {"entity": build_trie(self._entities)}
            for t, values in self.constants.items():
                self._tries[t] = build_trie(values)
        return self._tries

    def rule_constraint(self, **kw):
        return VadalogRuleConstraint(self.schema(), self.constant_tries(), **kw)

    # ------------------------------------------------------------ level 2 / 3
    def _view(self, pred):
        return self.views.get(pred, pred)

    def _object_is_entity(self, pred):
        types = self._schema.get(pred, ())
        return len(types) >= 2 and types[1] in ("entity", ANY)

    def properties(self, entity):
        """Property strings with at least one answer for ``entity``."""
        entity = str(entity)
        names = set()
        for pred in self._schema:
            rel = self._view(pred)
            if rel not in self._out:
                continue
            if self._out[rel].get(entity):
                names.add(pred)
            if (pred not in self.symmetric and self._object_is_entity(pred)
                    and self._inv[rel].get(entity)):
                names.add("is_" + pred)
        return sorted(names)

    def _resolve(self, prop):
        prop = str(prop).lower()
        inverse = prop.startswith("is_") and prop[3:] in self._schema
        pred = prop[3:] if inverse else prop
        if pred not in self._schema:
            raise KeyError("unknown property %r" % prop)
        return self._view(pred), inverse

    def get(self, entity, prop):
        rel, inverse = self._resolve(prop)
        return list((self._inv if inverse else self._out)[rel].get(str(entity), ()))

    def count(self, entity, prop):
        return len(self.get(entity, prop))

    def ask(self, subject, prop, obj):
        return str(obj) in self.get(subject, prop)

    def rows(self, pred):
        return list(self.relations.get(pred, ()))

    # ------------------------------------------------------------ rule mode
    def _infer_types(self, rule):
        """Head types from the body positions binding each head variable."""
        types = {}
        for lit in rule.body:
            if lit.negated:
                continue
            sig = self._schema.get(lit.atom.name, ())
            for i, term in enumerate(lit.atom.terms):
                name = getattr(term, "name", None)
                if name and i < len(sig) and name not in types:
                    types[name] = sig[i]
        for cond in rule.conditions:          # aggregates: N = count(X) ...
            expr = cond.expr
            if cond.op == "=" and cond.var not in types and hasattr(expr, "operand"):
                operand = getattr(expr.operand, "name", None)
                types[cond.var] = (types.get(operand, NUMBER)
                                   if expr.op in ("min", "max") else NUMBER)
        head = rule.head[0]
        return tuple(types.get(getattr(t, "name", None), ANY) for t in head.terms)

    def evaluate(self, text, register=False):
        """Run one rule with Vadalog over the relations it uses.

        Checks the closed-world semantics Vadalog does not (known predicates,
        no existential head), then evaluates only the rule: the body
        relations are injected as facts, so this takes ~1s instead of
        re-running the whole program."""
        res = RuleResult(rule=text.strip())
        t0 = time.perf_counter()
        try:
            prog = load_text(res.rule)
            if len(prog.rules) != 1 or prog.facts:
                raise ValueError("expected exactly one rule")
            rule = prog.rules[0]
            if len(rule.head) != 1:
                raise ValueError("expected a single head atom")
            res.head = rule.head[0].name
            res.arity = len(rule.head[0].terms)
            body = {lit.atom.name for lit in rule.body}
            unknown = sorted(body - set(self._schema))
            if unknown:
                raise ValueError("unknown predicate(s): %s" % ", ".join(unknown))
            bound = {t.name for lit in rule.body if not lit.negated
                     for t in lit.atom.terms if hasattr(t, "name")}
            bound |= {c.var for c in rule.conditions if c.op == "="}
            free = sorted({t.name for t in rule.head[0].terms if hasattr(t, "name")} - bound)
            if free:  # Vadalog would invent marked nulls (an existential)
                raise ValueError("head variable(s) not bound in the body: %s" % ", ".join(free))
            if res.head in body:
                raise ValueError("recursive rules are not supported here")
            if register and (res.head in self._schema
                             and res.head not in dict(self.session_rules)):
                raise ValueError("%s is already a predicate of the KG" % res.head)
            res.types = self._infer_types(rule)
            prog = load_text(res.rule + '\n@output("%s").' % res.head)
            # a rule sees a relation the way <kg> shows it (e.g. reachable as
            # its symmetric closure), so both modes agree
            facts = {p: self.relations[self._view(p)] for p in body}
            out = _Runner(prog, facts, options=self.options).run()
            res.rows = sorted((tuple(r) for r in out.relations.get(res.head, [])),
                              key=lambda r: tuple(map(str, r)))
            res.ok = True
        except (VadalogError, ValueError) as err:
            res.error = str(err)
        res.seconds = time.perf_counter() - t0
        if res.ok and register:
            self._register(res)
        return res

    def add_rule(self, text):
        """Evaluate a rule and register its head as a new relation."""
        return self.evaluate(text, register=True)

    def _register(self, res):
        with self._lock:
            self.session_rules = [(h, r) for h, r in self.session_rules if h != res.head]
            self.session_rules.append((res.head, res.rule))
            self._schema[res.head] = res.types
            self.relations[res.head] = res.rows
            self._index(res.head)
            res.registered = True

    def reset_rules(self):
        """Forget every rule registered with ``add_rule``."""
        with self._lock:
            for head, _ in self.session_rules:
                self._schema.pop(head, None)
                self.relations.pop(head, None)
                self._out.pop(head, None)
                self._inv.pop(head, None)
            self.session_rules = []

    # -------------------------------------------------------------- export
    def export(self, directory, name="kg.vada"):
        """Write a standalone program (+ one CSV per stored relation) that the
        vadalog CLI and web UI can run: ``vadalog run DIR/kg.vada``."""
        import csv
        os.makedirs(directory, exist_ok=True)
        lines = []
        for pred, rows in self.base.items():
            with open(os.path.join(directory, pred + ".csv"), "w", newline="") as f:
                csv.writer(f).writerows(rows)
            lines.append('@input("%s").' % pred)
            lines.append('@bind("%s", "csv", ".", "%s.csv").' % (pred, pred))
            for i, t in enumerate(self._schema.get(pred, ())):
                lines.append('@mapping("%s", %d, "c%d", "%s").'
                             % (pred, i, i, "double" if t == NUMBER else "string"))
        body = "\n".join(lines) + "\n\n" + self.program
        body += "".join("%s\n" % r for _, r in self.session_rules)
        body += "".join('@output("%s").\n' % p for p in self.derived)
        path = os.path.join(directory, name)
        with open(path, "w") as f:
            f.write(body)
        return path
