"""Stage 4: on-the-fly rule constraint (syntax x semantics).

A character-level automaton over a SAFE FRAGMENT of the Nemo rule language.
At every position it knows the set of characters that keep the rule
completable.  Constraint = grammar (structure) x semantic providers:

  - schema:   body predicates with arities (existing relations only)
  - entities: constants that actually exist (trie over the entity index)
  - head:     the task-given relation name + arity
  - variables: fixed alphabet; body atoms bind them, guards may only use
               bound ones, every head variable must be bound in the body
               (Nemo safety), at least one positive atom, <= one aggregate
               (in the head only), <= max_body_atoms literals.

Fragment grammar (tokens: '(' ')' ', ' ':- ' ' .' ' ' '#' op digits vars):

    rule    := head ":- " body " ."
    head    := pred "(" hterm, ... ")"        # arity(head) terms
    hterm   := var | number | entity | agg
    agg     := "#" (count|min|max|sum) "(" var [, var] ")"
    body    := literal (, literal)*           # >=1 atom, <= max_body_atoms
    literal := atom | guard
    atom    := pred "(" bterm, ... ")"        # pred in schema
    guard   := term op term                   # terms: bound var | number
    op      := > | >= | < | <= | = | !=
    var     := ?x ?y ?z ?u ?v ?w ?p ?q ?r ?s ?m ?n
    number  := [0-9]+ ('.' [0-9]+)?
    entity  := '"' <name in entity index> '"'

Excluded (extension path): negation '~', existential '!v' heads, arithmetic
expressions, global '$' variables, attributes, nested aggregates.

The automaton is an NFA simulated by subset construction.  A state element is
(frame, bound_vars, spec, progress): `spec` tracks the token currently being
consumed (fixed string / trie node / number).  `step` consumes one character;
`can_complete` decides (with memoization) whether any completion exists --
this is the predicate a logits mask asks about each candidate token.
"""
from collections import deque

ALPHABET = ("?x", "?y", "?z", "?u", "?v", "?w",
            "?p", "?q", "?r", "?s", "?m", "?n")
AGG_NAMES = ("count", "min", "max", "sum")
OPS = (">", ">=", "<", "<=", "=", "!=")
DIGITS = "0123456789"


def build_trie(names):
    """dict-trie: {'next': {ch: node}, 'end': name_or_None}."""
    root = {"next": {}, "end": None}
    for name in names:
        node = root
        for ch in name.replace("\\", "\\\\").replace('"', '\\"'):
            node = node["next"].setdefault(ch, {"next": {}, "end": None})
        node["end"] = name
    return root


class TrieSpec:
    __slots__ = ("node",)

    def __init__(self, node):
        self.node = node

    def __eq__(self, other):
        return isinstance(other, TrieSpec) and self.node is other.node

    def __hash__(self):
        return id(self.node)


class FixedSpec:
    __slots__ = ("token", "pos")

    def __init__(self, token, pos):
        self.token, self.pos = token, pos

    def __eq__(self, other):
        return (isinstance(other, FixedSpec)
                and self.token == other.token and self.pos == other.pos)

    def __hash__(self):
        return hash((self.token, self.pos))


class NumSpec:
    __slots__ = ("digits", "has_dot")

    def __init__(self, digits, has_dot):
        self.digits, self.has_dot = digits, has_dot

    def __eq__(self, other):
        return (isinstance(other, NumSpec)
                and self.digits == other.digits and self.has_dot == other.has_dot)

    def __hash__(self):
        return hash(("num", self.digits, self.has_dot))

    def can_end(self):
        return bool(self.digits) and not self.digits.endswith(".")


class EntitySpec:
    __slots__ = ("node",)     # None = expect opening quote

    def __init__(self, node):
        self.node = node

    def __eq__(self, other):
        return isinstance(other, EntitySpec) and self.node is other.node

    def __hash__(self):
        return id(self.node)


class RuleConstraint:
    def __init__(self, head, head_arity, schema, entities=(),
                 allow_agg=True, max_body_atoms=4, alphabet=ALPHABET,
                 max_number_digits=3, *, entity_trie=None):
        self.head, self.head_arity = head, head_arity
        self.schema = dict(schema)
        self.allow_agg = allow_agg
        self.max_body_atoms = max_body_atoms
        self.alphabet = tuple(alphabet)
        self.max_number_digits = max_number_digits
        self.entity_trie = build_trie(entities) if entity_trie is None else entity_trie
        self._ent_marker = EntitySpec(self.entity_trie)
        self._num_marker = NumSpec("0", False)
        # fixed alphabet vars as one-char-per-step specs
        self._spec_cache = {}
        self.initial = self._single(self._frame("hpred", head=head, arity=head_arity),
                                    frozenset(), None)

    # ------------------------------------------------------------- helpers
    def _frame(self, kind, **kw):
        """Frames as (kind, payload-dict) tuples (hashable, immutable)."""
        return (kind, frozenset(kw.items()))

    def _get(self, frame, key, default=None):
        return dict(frame[1]).get(key, default)

    def _with(self, frame, **kw):
        d = dict(frame[1])
        d.update(kw)
        return (frame[0], frozenset(d.items()))

    def _single(self, frame, bound, spec):
        return frozenset([(frame, bound, spec)])

    def _fixed(self, token):
        return self._spec_cache.setdefault(("f", token), FixedSpec(token, 0))

    def _num(self, digits="", has_dot=False):
        return NumSpec(digits, has_dot)

    # ----------------------------------------------------- next token specs
    def next_tokens(self, frame, bound):
        """Token specs that may start at this frame."""
        kind = frame[0]
        if kind == "hpred":
            return [self._fixed(self.head)]
        if kind == "hlp":
            return [self._fixed("(")]
        if kind == "hterm":
            opts = [self._fixed(v) for v in self.alphabet
                    if v not in set(self._get(frame, "agg_vars", ()))]
            opts += [self._num(), EntitySpec(None)]
            if self.allow_agg and not self._get(frame, "agg_used"):
                opts.append(self._fixed("#"))
            return opts
        if kind == "hsep":
            if self._get(frame, "pos") < self._get(frame, "arity") - 1:
                return [self._fixed(", ")]
            return [self._fixed(")")]
        if kind == "harrow":
            return [self._fixed(" :- ")]
        if kind == "body":
            opts = [self._fixed(p) for p in self.schema]
            if bound:
                opts += [self._fixed(v) for v in sorted(bound)]
                opts.append(self._num())
            return opts
        if kind == "bpred":
            return [self._fixed("(")]
        if kind == "blp":
            return ([self._fixed(v) for v in self.alphabet]
                    + [self._num(), EntitySpec(None)])
        if kind == "bterm":
            return ([self._fixed(v) for v in self.alphabet]
                    + [self._num(), EntitySpec(None)])
        if kind == "bsep":
            if self._get(frame, "pos") < self._get(frame, "arity") - 1:
                return [self._fixed(", ")]
            return [self._fixed(")")]
        if kind == "litsep":
            opts = []
            if self._get(frame, "n_lits") < self.max_body_atoms:
                opts.append(self._fixed(", "))
            if self._get(frame, "n_atoms") >= 1:
                opts.append(self._fixed(" ."))
            return opts
        if kind in ("guard1", "guard2"):
            return [self._fixed(v) for v in sorted(bound)] + [self._num()]
        if kind in ("gspace1", "gspace2"):
            return [self._fixed(" ")]
        if kind == "gop":
            return [self._fixed(op) for op in OPS]
        if kind == "aggn":
            return [self._fixed(a) for a in AGG_NAMES]
        if kind == "agglp":
            return [self._fixed("(")]
        if kind == "aggarg":
            return [self._fixed(v) for v in self.alphabet
                    if v not in set(self._get(frame, "agg_vars", ()))
                    and v not in set(self._get(frame, "head_vars", ()))]
        if kind == "aggsep":
            if len(self._get(frame, "agg_vars")) == 1:
                return [self._fixed(", "), self._fixed(")")]
            return [self._fixed(")")]
        if kind == "end":
            return []
        raise AssertionError(kind)

    def _token(self, frame, bound, spec):
        """Complete the token being consumed in `spec` and advance the frame.
        Returns list of (frame', bound') -- one per token interpretation."""
        kind = frame[0]
        if isinstance(spec, FixedSpec):
            tok = spec.token
        elif isinstance(spec, (TrieSpec, NumSpec, EntitySpec)):
            tok = None                       # const-ish tokens, no name check
        else:
            raise AssertionError(spec)

        if kind == "hpred":
            return [(self._frame("hlp", arity=self._get(frame, "arity"),
                                 agg_used=False, head_vars=()), bound)] \
                if tok == self.head else []
        if kind == "hterm":
            d = dict(frame[1])
            if tok == "#":
                return [(self._frame("aggn", **d), bound)]
            if tok in self.alphabet and tok not in set(d.get("agg_vars", ())):
                d["head_vars"] = d["head_vars"] + (tok,)
            return [(self._frame("hsep", **d), bound)]
        if kind == "hsep":
            d = dict(frame[1])
            if tok == ", " and d["pos"] < d["arity"] - 1:
                return [(self._frame("hterm", arity=d["arity"], pos=d["pos"] + 1,
                                     agg_used=d["agg_used"],
                                     head_vars=d["head_vars"],
                                     agg_vars=d.get("agg_vars", ())), bound)]
            if tok == ")":
                return [(self._frame("harrow", arity=d["arity"],
                                     agg_used=d["agg_used"],
                                     head_vars=d["head_vars"]), bound)]
            return []
        if kind == "body":
            d = dict(frame[1])
            if tok in self.schema:
                return [(self._frame("bpred", pred=tok, n_lits=d["n_lits"],
                                     n_atoms=d["n_atoms"],
                                     head_vars=d["head_vars"]), bound)]
            if tok is None or tok in self.alphabet:      # guard first term
                return [(self._frame("gspace1", n_lits=d["n_lits"],
                                     n_atoms=d["n_atoms"],
                                     head_vars=d["head_vars"]), bound)]
            return []
        if kind == "bpred":
            d = dict(frame[1])
            return [(self._frame("bterm", pred=d["pred"],
                                 arity=self.schema[d["pred"]], pos=0,
                                 atom_vars=(), n_lits=d["n_lits"],
                                 n_atoms=d["n_atoms"],
                                 head_vars=d["head_vars"]), bound)]
        if kind == "blp":
            return [(self._frame("bterm", **dict(frame[1]), pos=0), bound)]
        if kind == "bterm":
            d = dict(frame[1])
            if tok in self.alphabet:
                d["atom_vars"] = d["atom_vars"] + (tok,)
            return [(self._frame("bsep", **d), bound)]
        if kind == "bsep":
            d = dict(frame[1])
            if tok == ", " and d["pos"] < d["arity"] - 1:
                return [(self._frame("bterm", pred=d["pred"], arity=d["arity"],
                                     pos=d["pos"] + 1, atom_vars=d["atom_vars"],
                                     n_lits=d["n_lits"], n_atoms=d["n_atoms"],
                                     head_vars=d["head_vars"]), bound)]
            if tok == ")":
                nb = frozenset(bound) | set(d["atom_vars"])
                return [(self._frame("litsep", n_lits=d["n_lits"] + 1,
                                     n_atoms=d["n_atoms"] + 1,
                                     head_vars=d["head_vars"]), nb)]
            return []
        if kind == "litsep":
            d = dict(frame[1])
            if tok == ", " and d["n_lits"] < self.max_body_atoms:
                return [(self._frame("body", n_lits=d["n_lits"],
                                     n_atoms=d["n_atoms"],
                                     head_vars=d["head_vars"]), bound)]
            if tok == " ." and d["n_atoms"] >= 1:
                return [(self._frame("end", head_vars=d["head_vars"]), bound)]
            return []
        if kind == "guard1":
            return [(self._frame("gspace1", **dict(frame[1])), bound)]
        if kind == "gspace1":
            return [(self._frame("gop", **dict(frame[1])), bound)]
        if kind == "gop":
            return [(self._frame("gspace2", **dict(frame[1])), bound)] \
                if tok in OPS else []
        if kind == "gspace2":
            return [(self._frame("guard2", **dict(frame[1])), bound)]
        if kind == "guard2":
            return [(self._frame("litsep", n_lits=self._get(frame, "n_lits") + 1,
                                 n_atoms=self._get(frame, "n_atoms"),
                                 head_vars=self._get(frame, "head_vars")), bound)]
        if kind == "aggn":
            return [(self._frame("agglp", **dict(frame[1]), agg=tok), bound)] \
                if tok in AGG_NAMES else []
        if kind == "agglp":
            d = dict(frame[1])
            d["agg_vars"] = ()
            return [(self._frame("aggarg", **d), bound)]
        if kind == "aggarg":
            d = dict(frame[1])
            if tok in self.alphabet:
                d["agg_vars"] = d["agg_vars"] + (tok,)
                return [(self._frame("aggsep", **d), bound)]
            return []
        if kind == "aggsep":
            d = dict(frame[1])
            if tok == ", " and len(d["agg_vars"]) == 1:
                return [(self._frame("aggarg", **d), bound)]
            if tok == ")":
                return [(self._frame("hsep", arity=d["arity"], pos=d["pos"],
                                     agg_used=True,
                                     head_vars=d["head_vars"] + tuple(d["agg_vars"]),
                                     agg_vars=d["agg_vars"]),   # stays excluded
                         bound)]
            return []
        if kind == "hlp":
            return [(self._frame("hterm", arity=self._get(frame, "arity"),
                                 pos=0, agg_used=self._get(frame, "agg_used"),
                                 head_vars=self._get(frame, "head_vars"),
                                 agg_vars=()), bound)]
        if kind == "harrow":
            return [(self._frame("body", n_lits=0, n_atoms=0,
                                 head_vars=self._get(frame, "head_vars")), bound)]
        if kind == "end":
            return []
        raise AssertionError(kind)

    # ----------------------------------------------------------- transitions
    def _first_char(self, spec, ch):
        """Consume ch as the first character of a fresh spec; None if dead."""
        if isinstance(spec, FixedSpec):
            if not spec.token or spec.token[0] != ch:
                return None
            if len(spec.token) == 1:
                return ("complete", spec.token)
            return FixedSpec(spec.token, 1)
        if isinstance(spec, NumSpec):
            if ch in DIGITS:
                return NumSpec(ch, False)
            return None
        if isinstance(spec, EntitySpec):
            if ch == '"':
                return EntitySpec(self.entity_trie)
            return None
        raise AssertionError(spec)

    def _step_elt(self, frame, bound, spec, ch):
        """All (frame', bound', spec') after consuming ch from one element."""
        if spec is None:                             # between tokens
            out = []
            for s in self.next_tokens(frame, bound):
                r = self._first_char(s, ch)
                if r is None:
                    continue
                if isinstance(r, tuple) and r[0] == "complete":
                    out.extend((f, b, None) for f, b in
                               self._token(frame, bound, s))
                else:
                    out.append((frame, bound, r))
            return out
        if isinstance(spec, FixedSpec):
            if ch != spec.token[spec.pos]:
                return []
            if spec.pos + 1 == len(spec.token):
                return [(f, b, None) for f, b in self._token(frame, bound, spec)]
            return [(frame, bound, FixedSpec(spec.token, spec.pos + 1))]
        if isinstance(spec, TrieSpec):
            node = spec.node["next"].get(ch)
            if node is None:
                return []
            out = []
            if node["end"] is not None:              # name complete: also close
                out.extend((f, b, None) for f, b in
                           self._token(frame, bound, node["end"]))
            out.append((frame, bound, TrieSpec(node)))
            return out
        if isinstance(spec, EntitySpec):
            if spec.node is None:                    # expecting opening quote
                return [(frame, bound, EntitySpec(self.entity_trie))] \
                    if ch == '"' else []
            if ch == '"' and spec.node["end"] is not None:
                # closing quote: the entity token is complete
                return [(f, b, None) for f, b in
                        self._token(frame, bound, EntitySpec(spec.node))]
            node = spec.node["next"].get(ch)
            if node is None:
                return []
            return [(frame, bound, EntitySpec(node))]
        if isinstance(spec, NumSpec):
            room_for_digit = len(spec.digits) - spec.has_dot < self.max_number_digits
            if ch in DIGITS and room_for_digit:
                return [(frame, bound, NumSpec(spec.digits + ch, spec.has_dot))]
            if ch == "." and not spec.has_dot and spec.digits and room_for_digit:
                return [(frame, bound, NumSpec(spec.digits + ".", True))]
            if not spec.can_end():
                return []
            # number ends: the char belongs to the NEXT token
            out = []
            for f, b in self._token(frame, bound, spec):
                for s in self.next_tokens(f, b):
                    r = self._first_char(s, ch)
                    if r is None:
                        continue
                    if isinstance(r, tuple) and r[0] == "complete":
                        out.extend((f2, b2, None) for f2, b2 in
                                   self._token(f, b, s))
                    else:
                        out.append((f, b, r))
            return out
        raise AssertionError(spec)

    def step(self, subset, ch):
        out = set()
        for frame, bound, spec in subset:
            out.update(self._step_elt(frame, bound, spec, ch))
        return frozenset(out)

    # ------------------------------------------------------------ predicates
    def accepting(self, subset):
        """The rule is fully generated ('.' consumed) in some element."""
        for frame, bound, spec in subset:
            if (frame[0] == "end" and spec is None
                    and set(self._get(frame, "head_vars")) <= set(bound)):
                return True
        return False

    def next_chars(self, subset):
        chars = set()
        for frame, bound, spec in self._eps(subset, {}):
            if spec is None:
                for s in self.next_tokens(frame, bound):
                    if isinstance(s, FixedSpec) and s.token:
                        chars.add(s.token[0])
                    elif isinstance(s, NumSpec):
                        chars |= set(DIGITS)
                    elif isinstance(s, EntitySpec):
                        chars.add('"')
            elif isinstance(spec, FixedSpec):
                chars.add(spec.token[spec.pos])
            elif isinstance(spec, TrieSpec):
                chars |= set(spec.node["next"])
            elif isinstance(spec, EntitySpec):
                if spec.node is None:
                    chars.add('"')
                else:
                    if spec.node["end"] is not None:
                        chars.add('"')           # closing quote
                    chars |= set(spec.node["next"])
            elif isinstance(spec, NumSpec):
                if len(spec.digits) - spec.has_dot < self.max_number_digits:
                    chars |= set(DIGITS)
                    if not spec.has_dot and spec.digits:
                        chars.add(".")
        return chars

    def _eps(self, subset, memo):
        """Epsilon-closure: a number may end at any time without consuming
        its terminating character (that char starts the next token)."""
        if subset in memo:
            return memo[subset]
        memo[subset] = subset
        out = set(subset)
        work = list(subset)
        while work:
            frame, bound, spec = work.pop()
            if isinstance(spec, NumSpec) and spec.can_end():
                for f, b in self._token(frame, bound, spec):
                    el = (f, b, None)
                    if el not in out:
                        out.add(el)
                        work.append(el)
        memo[subset] = frozenset(out)
        return memo[subset]

    def _canon(self, subset):
        """Canonicalize a subset for memoization: any entity-trie position is
        equivalent for completability (the continuation after ANY entity is
        the same), and likewise any started number.  This collapses the
        100k-entity / 10-digit branching in the completion search."""
        ent_ok = bool(self.entity_trie["next"])
        out = set()
        for frame, bound, spec in subset:
            if isinstance(spec, EntitySpec) and ent_ok:
                spec = self._ent_marker
            elif isinstance(spec, NumSpec) and spec.can_end():
                spec = self._num_marker
            out.add((frame, bound, spec))
        return frozenset(out)

    def can_complete(self, subset, done=None, visiting=None):
        """Does any completion of the rule exist from this state?

        `done` caches DEFINITIVE answers (safe to share across calls and
        across candidate-token checks); `visiting` handles cycles without
        polluting `done` with optimistically-false entries."""
        if done is None:
            done = {}
        if visiting is None:
            visiting = set()
        raw = self._eps(subset, {})
        key = self._canon(raw)         # memo key (exact: continuation after
        if self.accepting(raw):        #   any entity / number is identical)
            done[key] = True
            return True
        if key in done:
            return done[key]
        if key in visiting:
            return False
        visiting.add(key)
        for ch in sorted(self.next_chars(raw)):
            nxt = self.step(raw, ch)   # step the REAL subset, not the key
            if nxt and self.can_complete(nxt, done, visiting):
                visiting.discard(key)
                done[key] = True
                return True
        visiting.discard(key)
        done[key] = False
        return False

    def consume(self, subset, text):
        for ch in text:
            subset = self.step(subset, ch)
            if not subset:
                return frozenset()
        return subset

    # ------------------------------------------------------------ generation
    def generate(self, limit=200, max_len=120):
        """BFS-enumerate rules the constraint allows (for soundness tests).
        Use a small entity set for tractability."""
        results, queue = [], deque([("", self.initial)])
        seen_states = set()
        while queue and len(results) < limit:
            text, subset = queue.popleft()
            if subset in seen_states:
                continue
            seen_states.add(subset)
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
