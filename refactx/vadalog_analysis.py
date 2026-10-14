"""Symbol table / safety analysis for generated Vadalog (Phase 3).

Parses a ``<vadalog>`` block into rules and reports the properties users of
constrained generation care about:

* the **symbol table** — variables introduced in the head vs used in the body;
* **range restriction / safety** — every head variable must also appear in the
  body (so the head is ground); facts must have no variables;
* **arity consistency** — a predicate is never used with two different arities.

Parsing is best-effort: a partial or malformed block yields whatever could be
parsed plus ``violations``, never an exception.
"""
from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"""
    (?P<ws>\s+)
  | (?P<impl>:-)
  | (?P<punct>[().,~])
  | (?P<number>-?\d+(?:\.\d+)?)
  | (?P<string>"(?:[^"\\]|\\.)*")
  | (?P<var>[A-Z][A-Za-z0-9_]*)
  | (?P<ident>[a-z][a-zA-Z0-9_]*)
""", re.VERBOSE)


def tokenize(text):
    """Best-effort lexer → list of ``(kind, value)`` (whitespace dropped)."""
    tokens, i = [], 0
    while i < len(text):
        match = _TOKEN_RE.match(text, i)
        if not match:
            # Unknown character: skip it (robust to partial output).
            i += 1
            continue
        i = match.end()
        kind = match.lastgroup
        if kind != "ws":
            tokens.append((kind, match.group()))
    return tokens


class _Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.i = 0

    def peek(self, *kinds):
        return self.i < len(self.tokens) and self.tokens[self.i][0] in kinds

    def value(self):
        return self.tokens[self.i][1]

    def take(self):
        tok = self.tokens[self.i]
        self.i += 1
        return tok

    def parse_atom(self):
        if not self.peek("ident"):
            return None
        pred = self.take()[1]
        if not self.peek("punct") or self.value() != "(":
            return None
        self.take()  # (
        args = []
        if not (self.peek("punct") and self.value() == ")"):
            while True:
                if not self.peek("var", "string", "number"):
                    return None
                args.append(self.take())
                if self.peek("punct") and self.value() == ",":
                    self.take()
                    continue
                break
        if not (self.peek("punct") and self.value() == ")"):
            return None
        self.take()  # )
        return {"pred": pred, "args": args}

    def parse_rule(self):
        head = self.parse_atom()
        if head is None:
            return None
        body = []
        has_body = False
        if self.peek("impl"):
            self.take()
            has_body = True
            while True:
                negated = self.peek("punct") and self.value() == "~"
                if negated:
                    self.take()
                lit = self.parse_atom()
                if lit is None:
                    return None
                lit["negated"] = negated
                body.append(lit)
                if self.peek("punct") and self.value() == ",":
                    self.take()
                    continue
                break
        if self.peek("punct") and self.value() == ".":
            self.take()
        return {"head": head, "body": body, "has_body": has_body}


def _vars(atom):
    return {value for kind, value in atom["args"] if kind == "var"}


def parse_program(text):
    """Parse (best-effort) into a list of rules."""
    parser = _Parser(tokenize(text))
    rules = []
    while parser.i < len(parser.tokens):
        before = parser.i
        rule = parser.parse_rule()
        if rule is None or parser.i == before:
            parser.i = before + 1  # skip a token and try to resync
            continue
        rules.append(rule)
    return rules


def analyze(text):
    """Symbol table + safety report for a ``<vadalog>`` block."""
    rules = parse_program(text)
    arity = {}
    violations = []
    rule_reports = []

    def record_arity(atom, where):
        name, n = atom["pred"], len(atom["args"])
        if name in arity and arity[name] != n:
            violations.append(
                f"arity: {name}/{n} but {name}/{arity[name]} elsewhere ({where})")
        arity.setdefault(name, n)

    for r in rules:
        head_vars = _vars(r["head"])
        body_vars = set()
        for lit in r["body"]:
            body_vars |= _vars(lit)
            record_arity(lit, "body")
        record_arity(r["head"], "head")

        if r["has_body"]:
            unsafe = head_vars - body_vars
            if unsafe:
                violations.append(
                    f"unsafe rule (head variable(s) {sorted(unsafe)} not in body): "
                    f"{r['head']['pred']}/... ")
        elif head_vars:
            violations.append(
                f"fact with variable(s) {sorted(head_vars)}: {r['head']['pred']}/...")

        rule_reports.append({
            "head": r["head"]["pred"],
            "head_vars": sorted(head_vars),
            "body_vars": sorted(body_vars),
            "body_only_vars": sorted(body_vars - head_vars),
            "safe": not (head_vars - body_vars) if r["has_body"] else not head_vars,
        })

    return {
        "n_rules": len(rules),
        "rules": rule_reports,
        "arity": arity,
        "violations": violations,
        "safe": not violations,
    }
