"""Check the shape of one bare Nemo rule before handing it to the engine."""
import re


ATOM = re.compile(r"([A-Za-z_][\w]*)\s*\((.*)\)", re.DOTALL)
TOKENS = re.compile(
    r'"(?:[^"\\]|\\.)*"|<[A-Za-z][A-Za-z0-9+.-]*:[^<>\s]+>|'
    r'(?<![\w?])\d+\.\d+|:-|!=|[<>]=?|[()\[\]{},.=#"%]'
)


COMPARISONS = ("=", "!=", "<", "<=", ">", ">=")


def split_top_level(text, delimiters):
    """Split outside quoted literals and nested terms, ignoring decimal dots."""
    if isinstance(delimiters, str):
        delimiters = (delimiters,)
    parts, stack = [], []
    start = 0
    for match in TOKENS.finditer(text):
        token = match.group()
        if token in ("(", "[", "{"):
            stack.append({"(": ")", "[": "]", "{": "}"}[token])
        elif token in (")", "]", "}"):
            if not stack or stack.pop() != token:
                raise ValueError("unbalanced delimiters")
        elif token in ('"', '%'):
            raise ValueError("expected a closed string and no comments")
        elif token in delimiters and not stack:
            parts.append(text[start:match.start()].strip())
            start = match.end()
    if stack:
        raise ValueError("unbalanced delimiters")
    parts.append(text[start:].strip())
    return parts


def atom_signature(text):
    match = ATOM.fullmatch(text)
    if match is None:
        raise ValueError("expected a predicate atom")
    terms = split_top_level(match[2], ",")
    if any(not term for term in terms):
        raise ValueError("empty predicate argument")
    return match[1], len(terms)


def rule_parts(text):
    """Return the head signature and body literals of a single complete rule."""
    statements = split_top_level(text, ".")
    if len(statements) != 2 or statements[1]:
        raise ValueError("expected exactly one rule terminated by a dot")
    sides = split_top_level(statements[0], ":-")
    if len(sides) != 2:
        raise ValueError("expected one ':-' between the head and body")
    head, arity = atom_signature(sides[0])
    body = split_top_level(sides[1], ",")
    if any(not literal for literal in body):
        raise ValueError("empty body literal")
    return head, arity, body


def body_has_aggregate(literal):
    return any(match.group() == "#" for match in TOKENS.finditer(literal))
