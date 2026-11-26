"""Character-level constraint for ONE Vadalog rule: syntax x semantics.

Port of the Nemo constraint (``ReFactX-with-Rules/stage4/rule_constraint.py``)
to Vadalog syntax, extended with argument *types* and negation.  At every
character it knows which characters keep the rule completable, so a logits
processor can mask every token that would lead to an invalid rule.

What is enforced
----------------
Syntax (the fragment below), and semantics:

* **predicates**: a body atom may only use a predicate of the schema (stored
  relations, derived relations and the rules registered so far), with its
  exact arity.  Vadalog itself is open world (an unknown predicate is just an
  empty relation): this layer is the closed-world check.
* **types**: every argument position has a type (``entity``, ``number``,
  ``role`` ...).  A variable gets the type of the first position that binds
  it and may only be re-used in positions of the same type (no join of a
  stake with a company).  A constant must be an existing value of the
  position's type: entity constants come from the entity index (a trie),
  numbers are digits.  ``>``/``<`` comparisons and ``sum/min/max/avg`` only
  apply to numbers.
* **safety**: every head variable must be bound in the body (otherwise
  Vadalog would silently invent a marked null, i.e. an existential); a
  negated atom, a comparison and an aggregate argument may only use bound
  variables; the aggregate result must be a head variable.
* **head**: a fresh lowercase name (not an existing predicate, not ``is_*``,
  which is the inverse-property prefix), or a fixed head given by the task.
  The head never occurs in the body, so a generated rule is never recursive
  and the program stays stratified.

Fragment (canonical spacing; ``[...]`` optional)::

    rule     := [" "] head " :- " literal (", " literal)* "."
    head     := name "(" VAR (", " VAR)* ")"
    literal  := atom | "not " atom | VAR op term | VAR " = " agg "(" VAR ")"
    atom     := pred "(" arg (", " arg)* ")"
    arg      := VAR | "_" | constant
    op       := " == " | " != " | " > " | " >= " | " < " | " <= "
    agg      := count | sum | min | max | avg
    VAR      := one upper-case letter
    constant := '"' existing value '"' | digits

Implementation: an NFA simulated by subset construction.  A state element is
``(frame, env, spec)``: ``frame`` is the grammar position plus the rule
context, ``env`` the bound variables with their types, ``spec`` the token
being consumed.  ``step`` consumes a character, ``can_complete`` decides (with
memoisation) whether some completion exists.
"""
from __future__ import annotations

from collections import deque

VARS = tuple("XYZWUVPQRSTNMKABCDEFGHIJLO")
AGGS = ("count", "sum", "min", "max", "avg")
NUMERIC_AGGS = ("sum", "min", "max", "avg")
EQ_OPS = (" == ", " != ")
ORD_OPS = (" > ", " >= ", " < ", " <= ")
NUMBER = "number"
ANY = "any"
DIGITS = "0123456789"
NAME_FIRST = "abcdefghijklmnopqrstuvwxyz"
NAME_CHARS = NAME_FIRST + DIGITS + "_"
RESERVED = {"not", "count", "sum", "min", "max", "avg", "msum", "mcount", "mprod"}


# ---------------------------------------------------------------- tries
def escape(value):
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def build_trie(names):
    """dict-trie over the *escaped* spelling: {'next': {ch: node}, 'end': name}."""
    root = {"next": {}, "end": None}
    for name in names:
        node = root
        for ch in escape(name):
            node = node["next"].setdefault(ch, {"next": {}, "end": None})
        node["end"] = name
    return root


def compatible(a, b):
    return a == b or a == ANY or b == ANY


# ---------------------------------------------------------------- specs
class FixedSpec:
    __slots__ = ("token", "pos")

    def __init__(self, token, pos=0):
        self.token, self.pos = token, pos

    def __eq__(self, other):
        return (type(other) is FixedSpec and self.token == other.token
                and self.pos == other.pos)

    def __hash__(self):
        return hash((self.token, self.pos))

    def __repr__(self):
        return "Fixed(%r@%d)" % (self.token, self.pos)


class NumSpec:
    """An integer constant (no decimals: '.' always ends the rule)."""
    __slots__ = ("digits",)

    def __init__(self, digits=""):
        self.digits = digits

    def __eq__(self, other):
        return type(other) is NumSpec and self.digits == other.digits

    def __hash__(self):
        return hash(("num", self.digits))

    def can_end(self):
        return bool(self.digits)

    def __repr__(self):
        return "Num(%s)" % self.digits


class ConstSpec:
    """A quoted constant of type ``ctype`` walked on its trie (node None: before '"')."""
    __slots__ = ("ctype", "node")

    def __init__(self, ctype, node=None):
        self.ctype, self.node = ctype, node

    def __eq__(self, other):
        return (type(other) is ConstSpec and self.ctype == other.ctype
                and self.node is other.node)

    def __hash__(self):
        return hash((self.ctype, id(self.node)))

    def __repr__(self):
        return "Const(%s)" % self.ctype


class NameSpec:
    """The fresh head name, completed by the '(' that follows it."""
    __slots__ = ("text",)

    def __init__(self, text=""):
        self.text = text

    def __eq__(self, other):
        return type(other) is NameSpec and self.text == other.text

    def __hash__(self):
        return hash(("name", self.text))

    def __repr__(self):
        return "Name(%s)" % self.text


_NUM_MARK = NumSpec("0")


# ---------------------------------------------------------------- frames
# frame = (kind, ctx, extra)
#   ctx   = (head, head_vars, n_lits, n_atoms, agg_done)
#   extra = kind-specific tuple
def _ctx(head="", head_vars=(), n_lits=0, n_atoms=0, agg_done=False):
    return (head, head_vars, n_lits, n_atoms, agg_done)


class VadalogRuleConstraint:
    """Constraint for one rule over ``schema`` = {predicate: (type, ...)}.

    ``constants`` maps a type to the values allowed as quoted constants of
    that type (or to a prebuilt trie, see :func:`build_trie`).  Positions of a
    type with no entry accept variables only (``number`` accepts digits).
    """

    def __init__(self, schema, constants=None, head=None, head_arity=None,
                 max_body_literals=4, max_head_arity=3, max_number_digits=4,
                 max_name_len=32, variables=VARS, allow_negation=True,
                 allow_aggregates=True, leading_space=True):
        self.schema = {p: tuple(t) for p, t in schema.items()}
        self.tries = {}
        for ctype, values in (constants or {}).items():
            if isinstance(values, dict) and "next" in values:
                self.tries[ctype] = values
            else:
                self.tries[ctype] = build_trie(values)
        # an empty trie would make the can_complete shortcut unsound
        self.tries = {t: n for t, n in self.tries.items() if n["next"]}
        self.head = head
        self.head_arity = head_arity
        self.max_body_literals = max_body_literals
        self.max_head_arity = head_arity or max_head_arity
        self.max_number_digits = max_number_digits
        self.max_name_len = max_name_len
        self.variables = tuple(variables)
        self.allow_negation = allow_negation
        self.allow_aggregates = allow_aggregates
        self.max_arity = max([len(t) for t in self.schema.values()] + [1])
        self.reserved = set(RESERVED) | {p.lower() for p in self.schema}
        self._fixed_cache = {}
        start = [((("hname", _ctx(), ())), frozenset(), None)]
        if leading_space:
            start.append((("lead", _ctx(), ()), frozenset(), None))
        self.initial = frozenset(start)

    # ---------------------------------------------------------- helpers
    def _fx(self, token):
        spec = self._fixed_cache.get(token)
        if spec is None:
            spec = self._fixed_cache[token] = FixedSpec(token, 0)
        return spec

    @staticmethod
    def _types(env):
        return dict(env)

    def _const_specs(self, ctype):
        specs = []
        if ctype in (NUMBER, ANY):
            specs.append(NumSpec())
        if ctype in self.tries:
            specs.append(ConstSpec(ctype))
        elif ctype == ANY:
            specs.extend(ConstSpec(t) for t in self.tries)
        return specs

    def valid_name(self, name):
        return (bool(name) and name[0] in NAME_FIRST and name not in self.reserved
                and not name.startswith("is_"))

    # ------------------------------------------------- token options
    def next_tokens(self, frame, env):
        """Token specs that may start at this frame."""
        kind, ctx, extra = frame
        head, head_vars, n_lits, n_atoms, agg_done = ctx
        if kind == "lead":
            return [self._fx(" ")]
        if kind == "hname":
            if self.head is not None:
                return [self._fx(self.head + "(")]
            return [NameSpec("")]
        if kind == "hterm":
            return [self._fx(v) for v in self.variables]
        if kind == "hsep":
            (pos,) = extra
            opts = []
            if pos + 1 < self.max_head_arity:
                opts.append(self._fx(", "))
            if self.head_arity is None or pos + 1 == self.head_arity:
                opts.append(self._fx(")"))
            return opts
        if kind == "harrow":
            return [self._fx(" :- ")]
        if kind == "body":
            types = self._types(env)
            opts = [self._fx(p + "(") for p in self.schema]
            if self.allow_negation and n_atoms >= 1:
                opts.append(self._fx("not "))
            for v in types:                                   # comparison
                opts.append(self._fx(v))
            if self.allow_aggregates and not agg_done and n_atoms >= 1:
                for v in head_vars:                           # aggregate result
                    if v not in types:
                        opts.append(self._fx(v))
            return opts
        if kind == "negpred":
            return [self._fx(p + "(") for p in self.schema]
        if kind == "bterm":
            pred, pos, neg = extra
            ctype = self.schema[pred][pos]
            types = self._types(env)
            opts = []
            for v in self.variables:
                if v in types:
                    if compatible(types[v], ctype):
                        opts.append(self._fx(v))
                elif not neg:
                    opts.append(self._fx(v))
            opts.append(self._fx("_"))
            opts.extend(self._const_specs(ctype))
            return opts
        if kind == "bsep":
            pred, pos, neg = extra
            if pos + 1 < len(self.schema[pred]):
                return [self._fx(", ")]
            return [self._fx(")")]
        if kind == "litsep":
            opts = []
            if n_lits < self.max_body_literals:
                opts.append(self._fx(", "))
            if n_atoms >= 1 and all(v in self._types(env) for v in head_vars):
                opts.append(self._fx("."))
            return opts
        if kind == "gop":
            var, ctype = extra
            ops = list(EQ_OPS)
            if ctype in (NUMBER, ANY):
                ops += ORD_OPS
            return [self._fx(op) for op in ops]
        if kind == "grhs":
            var, ctype = extra
            types = self._types(env)
            opts = [self._fx(v) for v, t in types.items()
                    if v != var and compatible(t, ctype)]
            return opts + self._const_specs(ctype)
        if kind == "assign":
            return [self._fx(" = ")]
        if kind == "aggn":
            return [self._fx(a + "(") for a in AGGS]
        if kind == "aggarg":
            var, agg = extra
            types = self._types(env)
            return [self._fx(v) for v, t in types.items()
                    if agg == "count" or compatible(t, NUMBER)]
        if kind == "aggclose":
            return [self._fx(")")]
        if kind == "end":
            return []
        raise AssertionError(kind)

    # ------------------------------------------------- token completion
    def _token(self, frame, env, spec):
        """Advance past the completed token ``spec``: list of (frame', env')."""
        kind, ctx, extra = frame
        head, head_vars, n_lits, n_atoms, agg_done = ctx
        tok = spec.token if isinstance(spec, FixedSpec) else None

        if kind == "lead":
            return [(("hname", ctx, ()), env)]
        if kind == "hname":
            name = spec.text if isinstance(spec, NameSpec) else self.head
            return [(("hterm", _ctx(name), (0,)), env)]
        if kind == "hterm":
            (pos,) = extra
            ctx = _ctx(head, head_vars + (tok,))
            return [(("hsep", ctx, (pos,)), env)]
        if kind == "hsep":
            (pos,) = extra
            if tok == ", ":
                return [(("hterm", ctx, (pos + 1,)), env)]
            return [(("harrow", ctx, ()), env)]
        if kind == "harrow":
            return [(("body", ctx, ()), env)]
        if kind == "body":
            if tok == "not ":
                return [(("negpred", ctx, ()), env)]
            if tok.endswith("("):
                return [(("bterm", ctx, (tok[:-1], 0, False)), env)]
            types = self._types(env)
            if tok in types:
                return [(("gop", ctx, (tok, types[tok])), env)]
            return [(("assign", ctx, (tok,)), env)]
        if kind == "negpred":
            return [(("bterm", ctx, (tok[:-1], 0, True)), env)]
        if kind == "bterm":
            pred, pos, neg = extra
            if tok is not None and tok != "_" and tok not in self._types(env):
                env = env | {(tok, self.schema[pred][pos])}
            return [(("bsep", ctx, extra), env)]
        if kind == "bsep":
            pred, pos, neg = extra
            if tok == ", ":
                return [(("bterm", ctx, (pred, pos + 1, neg)), env)]
            ctx = (head, head_vars, n_lits + 1, n_atoms + (0 if neg else 1), agg_done)
            return [(("litsep", ctx, ()), env)]
        if kind == "litsep":
            if tok == ", ":
                return [(("body", ctx, ()), env)]
            return [(("end", ctx, ()), env)]
        if kind == "gop":
            var, ctype = extra
            return [(("grhs", ctx, (var, ctype)), env)]
        if kind == "grhs":
            ctx = (head, head_vars, n_lits + 1, n_atoms, agg_done)
            return [(("litsep", ctx, ()), env)]
        if kind == "assign":
            return [(("aggn", ctx, extra), env)]
        if kind == "aggn":
            (var,) = extra
            return [(("aggarg", ctx, (var, tok[:-1])), env)]
        if kind == "aggarg":
            var, agg = extra
            arg_type = self._types(env)[tok]
            rtype = arg_type if agg in ("min", "max") else NUMBER
            return [(("aggclose", ctx, (var, rtype)), env)]
        if kind == "aggclose":
            var, rtype = extra
            ctx = (head, head_vars, n_lits + 1, n_atoms, True)
            return [(("litsep", ctx, ()), env | {(var, rtype)})]
        raise AssertionError(kind)

    # ------------------------------------------------------ pruning
    def _alive(self, frame, env):
        """Cheap necessary condition: enough body literals left to bind every
        head variable (an atom binds at most ``max_arity`` of them)."""
        kind, ctx, extra = frame
        if kind in ("lead", "hname", "hterm", "hsep", "harrow", "end"):
            return True
        head, head_vars, n_lits, n_atoms, agg_done = ctx
        bound = {v for v, _ in env}
        unbound = sum(1 for v in head_vars if v not in bound)
        if not unbound:
            return True
        if kind in ("body", "litsep"):
            left = self.max_body_literals - n_lits
            return unbound <= left * self.max_arity
        left = self.max_body_literals - n_lits - 1        # after the current literal
        if kind in ("bterm", "bsep"):
            pred, pos, _neg = extra
            here = len(self.schema[pred]) - pos - (1 if kind == "bsep" else 0)
        else:
            here = 1
        return unbound <= left * self.max_arity + here

    # -------------------------------------------------- transitions
    def _first_char(self, spec, ch):
        """Consume ``ch`` as first char of a fresh spec: spec', 'done' or None."""
        if isinstance(spec, FixedSpec):
            if spec.token[0] != ch:
                return None
            return "done" if len(spec.token) == 1 else FixedSpec(spec.token, 1)
        if isinstance(spec, NumSpec):
            return NumSpec(ch) if ch in DIGITS else None
        if isinstance(spec, ConstSpec):
            if ch == '"' and spec.ctype in self.tries:
                return ConstSpec(spec.ctype, self.tries[spec.ctype])
            return None
        if isinstance(spec, NameSpec):
            return NameSpec(ch) if ch in NAME_FIRST else None
        raise AssertionError(spec)

    def _start(self, frame, env, ch):
        out = []
        for s in self.next_tokens(frame, env):
            r = self._first_char(s, ch)
            if r is None:
                continue
            if r == "done":
                out.extend((f, b, None) for f, b in self._token(frame, env, s))
            else:
                out.append((frame, env, r))
        return out

    def _step_elt(self, frame, env, spec, ch):
        if spec is None:
            return self._start(frame, env, ch)
        if isinstance(spec, FixedSpec):
            if spec.token[spec.pos] != ch:
                return []
            if spec.pos + 1 == len(spec.token):
                return [(f, b, None) for f, b in self._token(frame, env, spec)]
            return [(frame, env, FixedSpec(spec.token, spec.pos + 1))]
        if isinstance(spec, ConstSpec):
            node = spec.node
            if ch == '"' and node["end"] is not None:
                return [(f, b, None) for f, b in self._token(frame, env, spec)]
            child = node["next"].get(ch)
            return [(frame, env, ConstSpec(spec.ctype, child))] if child is not None else []
        if isinstance(spec, NumSpec):
            if ch in DIGITS:
                if len(spec.digits) < self.max_number_digits:
                    return [(frame, env, NumSpec(spec.digits + ch))]
                return []
            out = []
            for f, b in self._token(frame, env, spec):      # the number ends here
                out.extend(self._start(f, b, ch))
            return out
        if isinstance(spec, NameSpec):
            if ch == "(":
                if self.valid_name(spec.text):
                    return [(f, b, None) for f, b in self._token(frame, env, spec)]
                return []
            if ch in NAME_CHARS and len(spec.text) < self.max_name_len:
                return [(frame, env, NameSpec(spec.text + ch))]
            return []
        raise AssertionError(spec)

    def step(self, subset, ch):
        out = set()
        for frame, env, spec in subset:
            for el in self._step_elt(frame, env, spec, ch):
                if self._alive(el[0], el[1]):
                    out.add(el)
        return frozenset(out)

    def consume(self, subset, text):
        for ch in text:
            subset = self.step(subset, ch)
            if not subset:
                return frozenset()
        return subset

    # ---------------------------------------------------- predicates
    def accepting(self, subset):
        return any(frame[0] == "end" and spec is None for frame, _env, spec in subset)

    def _eps(self, subset):
        """A number may end without consuming its terminating character."""
        out = set(subset)
        for frame, env, spec in subset:
            if isinstance(spec, NumSpec) and spec.can_end():
                out.update((f, b, None) for f, b in self._token(frame, env, spec))
        return frozenset(out)

    def next_chars(self, subset):
        chars = set()
        for frame, env, spec in self._eps(subset):
            if spec is None:
                for s in self.next_tokens(frame, env):
                    if isinstance(s, FixedSpec):
                        chars.add(s.token[0])
                    elif isinstance(s, NumSpec):
                        chars.update(DIGITS)
                    elif isinstance(s, ConstSpec):
                        if s.ctype in self.tries:
                            chars.add('"')
                    elif isinstance(s, NameSpec):
                        chars.update(NAME_FIRST)
            elif isinstance(spec, FixedSpec):
                chars.add(spec.token[spec.pos])
            elif isinstance(spec, ConstSpec):
                chars.update(spec.node["next"])
                if spec.node["end"] is not None:
                    chars.add('"')
            elif isinstance(spec, NumSpec):
                if len(spec.digits) < self.max_number_digits:
                    chars.update(DIGITS)
            elif isinstance(spec, NameSpec):
                if len(spec.text) < self.max_name_len:
                    chars.update(NAME_CHARS)
                if self.valid_name(spec.text):
                    chars.add("(")
        return chars

    def _canon(self, subset):
        """Memo key: forget what cannot change completability -- the value of
        a number, the spelling of the head name (any unfinished name can still
        become a valid one)."""
        out = set()
        for (kind, ctx, extra), env, spec in subset:
            if ctx[0]:
                ctx = ("",) + ctx[1:]
            if isinstance(spec, NumSpec) and spec.can_end():
                spec = _NUM_MARK
            elif isinstance(spec, NameSpec):
                ok = len(spec.text) < self.max_name_len or self.valid_name(spec.text)
                spec = ("name", ok)
            out.add(((kind, ctx, extra), env, spec))
        return frozenset(out)

    def can_complete(self, subset, done=None, visiting=None):
        """Does any completion of the rule exist from ``subset``?

        ``done`` caches definitive answers (share it across the candidate
        tokens of one step); ``visiting`` breaks cycles."""
        if not subset:
            return False
        done = {} if done is None else done
        visiting = set() if visiting is None else visiting
        # An unfinished head name can always be extended into a valid fresh
        # name, and any head can be completed by a body over a non-empty
        # schema.  (Searching instead would loop: every extension of a
        # reserved prefix such as `own` shares one canonical memo key.)
        if self.schema and any(
                isinstance(spec, NameSpec)
                and (len(spec.text) < self.max_name_len or self.valid_name(spec.text))
                for _f, _e, spec in subset):
            return True
        # A non-empty trie node always has a complete name below it: jump to
        # the state after the closing quote (existence check only).
        jumped = set()
        for frame, env, spec in subset:
            if isinstance(spec, ConstSpec) and spec.node is not None:
                jumped.update((f, b, None) for f, b in self._token(frame, env, spec))
            else:
                jumped.add((frame, env, spec))
        raw = self._eps(frozenset(jumped))
        if self.accepting(raw):
            return True
        key = self._canon(raw)
        if key in done:
            return done[key]
        if key in visiting:
            return False
        visiting.add(key)
        result = False
        for ch in sorted(self.next_chars(raw)):
            nxt = self.step(raw, ch)
            if nxt and self.can_complete(nxt, done, visiting):
                result = True
                break
        visiting.discard(key)
        done[key] = result
        return result

    # ------------------------------------------------- introspection
    def accepts(self, text):
        """True when ``text`` is a complete, valid rule."""
        return self.accepting(self.consume(self.initial, text))

    def first_error(self, text):
        """Index of the first character that leaves the constraint (None if
        every prefix is fine) -- handy to explain *why* a rule is rejected."""
        subset = self.initial
        for i, ch in enumerate(text):
            subset = self.step(subset, ch)
            if not subset:
                return i
        return None

    def describe(self, subset):
        """Human-readable summary of what may come next."""
        out = []
        for frame, env, spec in sorted(self._eps(subset), key=repr):
            kind = frame[0]
            bound = ", ".join("%s:%s" % ve for ve in sorted(env)) or "-"
            if spec is None:
                toks = []
                for s in self.next_tokens(frame, env):
                    if isinstance(s, FixedSpec):
                        toks.append(repr(s.token))
                    elif isinstance(s, NumSpec):
                        toks.append("<integer>")
                    elif isinstance(s, ConstSpec):
                        toks.append('<"%s" constant>' % s.ctype)
                    elif isinstance(s, NameSpec):
                        toks.append("<new relation name>")
                out.append("%-8s bound={%s} next: %s" % (kind, bound, " | ".join(toks) or "(end)"))
            else:
                out.append("%-8s bound={%s} inside %r" % (kind, bound, spec))
        return out

    def head_signature(self, subset):
        """(head, head_vars, {var: type}) of an accepting subset."""
        for frame, env, spec in subset:
            if frame[0] == "end" and spec is None:
                head, head_vars = frame[1][0], frame[1][1]
                return head, head_vars, dict(env)
        return None

    def generate(self, limit=50, max_len=120):
        """BFS-enumerate allowed rules (soundness tests; use a tiny schema)."""
        results, queue, seen = [], deque([("", self.initial)]), set()
        while queue and len(results) < limit:
            text, subset = queue.popleft()
            if subset in seen:
                continue
            seen.add(subset)
            if self.accepting(subset):
                results.append(text)
                continue
            if len(text) >= max_len:
                continue
            for ch in sorted(self.next_chars(subset)):
                nxt = self.step(subset, ch)
                if nxt:
                    queue.append((text + ch, nxt))
        return results
