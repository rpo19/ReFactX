"""Opt-in semantic answer verification for QA evaluation.

The existing evaluator (``utils/eval.py``) intentionally keeps strict string/set
matching.  This module provides a separate, auditable verification used to
analyse formatting, date, quantity, and Mintaka entity-alias mismatches.

The verifier is deliberately conservative:

* dates are compared by parsed precision, not substring inclusion;
* a year reference may match a full date with the same year, but a full-date
  reference is not satisfied by a year-only prediction;
* two-digit years are compared modulo 100 because Mintaka does not disambiguate
  the century (``4-Oct-23`` is 1923, while ``70`` in ``2-Aug-70`` is 1970);
* quantities are converted only within an explicit unit table;
* Mintaka entity labels are used only for non-count answers;
* no external alias lookup or fuzzy/partial string matching is performed.

Provenance of the alias/transliteration tables is documented inline above
``_ENTITY_ALIASES``: they are hand-curated from observed log failures rather
than derived from the dataset, and the ``answerEntity`` Q-ID labels remain the
principled source of entity aliases.  Measured on the September 2026 logs, the
``alias`` rule fired zero times because the ``answerEntity`` rule subsumes it.

Public entry points
-------------------
``verify_mintaka(prediction, reference, ...)``
    Full verification result with canonicalised forms and the rule that fired.
``compare_answers(...)``
    List-level comparison preserving cardinality.
``parse_date`` / ``dates_match`` / ``parse_quantity`` / ``quantities_match``
    The individual typed rules, usable on their own.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable, Optional


_MONTHS = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sept": 9, "sep": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

_MONTH_ABBREV = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}


# ---------------------------------------------------------------------------
# Text and entity normalisation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Hand-curated alias and transliteration tables
# ---------------------------------------------------------------------------
#
# PROVENANCE (read this before editing)
#
# These tables are *hand-written*, not derived from the data and not extracted
# from Wikidata. They were seeded from failures observed in the Mintaka logs
# (for example ``gt="USA."`` vs ``prediction="United States"``, and
# ``Aarhus`` vs ``Århus``) plus general knowledge of common surface variants.
# There is no automatic generation step and no exhaustive validation behind
# them, so treat them as a small curated list rather than a gazetteer.
#
# PRINCIPLED ALTERNATIVE
#
# Mintaka records carry stable Wikidata Q-IDs in ``input_sample.answerEntity``.
# The robust way to build aliases is to resolve those Q-IDs to their label/alias
# sets offline and cache them as a lookup keyed by Q-ID, then compare entity IDs
# rather than surface strings. That is a data preparation task, deliberately not
# done at verification time (no network access during evaluation).
#
# EMPIRICAL STATUS (measured on the September 2026 logs, ~4,400 scored answers)
#
#   rule='alias' fired 0 times across all five logs.  Reason: whenever a real
#   record supplies ``answerEntity``, the ``mintaka_answer_entity_label`` rule is
#   checked first and subsumes these cases.  The table only takes effect when no
#   ``answerEntity`` is present, which is why it still fires in unit tests but not
#   on the evaluated data.  Keep it small and explicit; it is a fallback, not the
#   main mechanism.
#
# RISK NOTE
#
# ``"us"`` is also an ordinary English word.  It is safe here only because the
# map is applied to the *entire* normalized answer string, never to individual
# tokens, so phrases such as "the us army" are unaffected.  Do not switch this to
# token-level substitution.
_ENTITY_ALIASES = {
    "usa": "united states",
    "us": "united states",
    "u.s.": "united states",
    "u.s.a.": "united states",
    "united states of america": "united states",
    "uk": "united kingdom",
    "u.k.": "united kingdom",
}

#: Nordic/German characters whose single-letter and digraph spellings are both
#: common in the data (e.g. "Århus" vs "Aarhus", "Galați"). Same provenance and
#: caveats as ``_ENTITY_ALIASES``: hand-written, minimal, applied to whole
#: strings. The digraph mappings are needed because plain accent folding turns
#: "å" into "a", which would not match the two-letter spelling "aa".
_TRANSLIT = {
    "\u00e5": "aa",  # å
    "\u00e6": "ae",  # æ
    "\u00f8": "o",   # ø
    "\u00df": "ss",  # ß
}

_ALIAS_TARGETS = set(_ENTITY_ALIASES.values())


def normalize_text(value: Any) -> str:
    """Normalize harmless textual variation without applying fuzzy matching."""
    value = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    value = value.replace("\u2018", "'").replace("\u2019", "'")
    value = re.sub(r"\s+", " ", value)
    return value.rstrip(" .")


def fold_entity(value: Any) -> str:
    """Alias-resolved, accent-folded, punctuation-free form used for entities.

    Examples: ``USA`` -> ``united states``; ``Århus`` -> ``aarhus``;
    ``Monsters, Inc.`` -> ``monsters inc``.
    """
    text = normalize_text(value)
    text = _ENTITY_ALIASES.get(text, text)
    for source, target in _TRANSLIT.items():
        text = text.replace(source, target)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^\w\s]", "", text).strip()


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DateValue:
    year: int
    month: Optional[int] = None
    day: Optional[int] = None
    year_ambiguous: bool = False

    @property
    def precision(self) -> str:
        if self.month is None:
            return "year"
        if self.day is None:
            return "month"
        return "day"


def parse_date(value: Any) -> Optional[DateValue]:
    """Parse the date formats observed in Mintaka and 2Wiki references.

    Two-digit years are stored as-is with ``year_ambiguous=True`` rather than
    being expanded, because the century genuinely varies in the data.
    """
    text = normalize_text(value)
    if re.fullmatch(r"\d{4}", text):
        return DateValue(int(text))

    match = re.fullmatch(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", text)
    if match:
        return DateValue(int(match.group(1)), int(match.group(2)), int(match.group(3)))

    match = re.fullmatch(r"(\d{1,2})[- ]([a-z]+)[- ](\d{2,4})", text)
    if match and match.group(2) in _MONTHS:
        raw_year = match.group(3)
        return DateValue(
            int(raw_year),
            _MONTHS[match.group(2)],
            int(match.group(1)),
            year_ambiguous=len(raw_year) == 2,
        )

    match = re.fullmatch(r"([a-z]+) (\d{1,2}),? (\d{4})", text)
    if match and match.group(1) in _MONTHS:
        return DateValue(int(match.group(3)), _MONTHS[match.group(1)], int(match.group(2)))

    match = re.fullmatch(r"(\d{1,2}) ([a-z]+) (\d{4})", text)
    if match and match.group(2) in _MONTHS:
        return DateValue(int(match.group(3)), _MONTHS[match.group(2)], int(match.group(1)))

    return None


def _years_match(reference: DateValue, prediction: DateValue) -> bool:
    if reference.year_ambiguous or prediction.year_ambiguous:
        return reference.year % 100 == prediction.year % 100
    return reference.year == prediction.year


def dates_match(reference: Any, prediction: Any) -> tuple[bool, bool]:
    """Return ``(normalized_match, partial_match)`` for two date-like values."""
    ref = parse_date(reference)
    pred = parse_date(prediction)
    if ref is None or pred is None:
        return False, False

    if ref.precision == "year":
        # A year-only reference is satisfied by any value in that year.
        return _years_match(ref, pred), False

    if ref.precision == "month" and pred.precision == "month":
        return (ref.month == pred.month and _years_match(ref, pred)), False

    if ref.precision != pred.precision:
        # Lower precision than the reference: same year is partial credit only.
        return False, _years_match(ref, pred)

    same = ref.month == pred.month and ref.day == pred.day and _years_match(ref, pred)
    return same, False


def canonical_date(value: Any) -> Optional[str]:
    """Render a parsed date in Mintaka's reference convention.

    ``1981`` stays a year; a full date becomes ``D-Mon-YY`` when the source year
    was two digits, otherwise ``D-Mon-YYYY``.
    """
    parsed = parse_date(value)
    if parsed is None:
        return None
    if parsed.precision == "year":
        return f"{parsed.year:02d}" if parsed.year_ambiguous else f"{parsed.year:04d}"
    if parsed.precision == "month":
        year = f"{parsed.year:02d}" if parsed.year_ambiguous else f"{parsed.year:04d}"
        return f"{_MONTH_ABBREV[parsed.month]} {year}"
    year = f"{parsed.year:02d}" if parsed.year_ambiguous else f"{parsed.year:04d}"
    return f"{parsed.day}-{_MONTH_ABBREV[parsed.month]}-{year}"


# ---------------------------------------------------------------------------
# Quantities
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QuantityValue:
    value: Decimal
    unit: Optional[str]


_UNIT_ALIASES = {
    "mm": "mm", "millimeter": "mm", "millimeters": "mm",
    "millimetre": "mm", "millimetres": "mm",
    "cm": "cm", "centimeter": "cm", "centimeters": "cm",
    "centimetre": "cm", "centimetres": "cm",
    "m": "m", "meter": "m", "meters": "m", "metre": "m", "metres": "m",
    "km": "km", "kilometer": "km", "kilometers": "km",
    "kilometre": "km", "kilometres": "km",
    "in": "in", "inch": "in", "inches": "in",
    "ft": "ft", "foot": "ft", "feet": "ft",
    "mi": "mi", "mile": "mi", "miles": "mi",
    "mg": "mg", "milligram": "mg", "milligrams": "mg",
    "g": "g", "gram": "g", "grams": "g",
    "kg": "kg", "kilogram": "kg", "kilograms": "kg",
    "lb": "lb", "lbs": "lb", "pound": "lb", "pounds": "lb",
    "oz": "oz", "ounce": "oz", "ounces": "oz",
    "s": "second", "sec": "second", "second": "second", "seconds": "second",
    "min": "minute", "minute": "minute", "minutes": "minute",
    "h": "hour", "hr": "hour", "hour": "hour", "hours": "hour",
    "day": "day", "days": "day",
    "week": "week", "weeks": "week",
    "year": "year", "years": "year",
}

#: Unit -> factor to that dimension's canonical unit (cm, g, second).
_DIMENSIONS = {
    "length": {
        "mm": Decimal("0.1"), "cm": Decimal("1"), "m": Decimal("100"),
        "km": Decimal("100000"), "in": Decimal("2.54"), "ft": Decimal("30.48"),
        "mi": Decimal("160934.4"),
    },
    "mass": {
        "mg": Decimal("0.001"), "g": Decimal("1"), "kg": Decimal("1000"),
        "lb": Decimal("453.59237"), "oz": Decimal("28.349523125"),
    },
    "time": {
        "second": Decimal("1"), "minute": Decimal("60"), "hour": Decimal("3600"),
        "day": Decimal("86400"), "week": Decimal("604800"),
        "year": Decimal("31557600"),
    },
}

#: Absolute tolerance, expressed in the dimension's canonical unit.
_TOLERANCE = {
    "length": Decimal("1"),      # 1 cm
    "mass": Decimal("1"),        # 1 g
    "time": Decimal("1"),        # 1 s
    "count": Decimal("0"),
}

_DIMENSION_UNITS = {unit: dim for dim, table in _DIMENSIONS.items() for unit in table}


def parse_quantity(value: Any) -> Optional[QuantityValue]:
    """Parse ``<number> <unit>`` and feet/inches forms such as ``5'7"``."""
    text = normalize_text(value)

    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*'\s*(\d+(?:\.\d+)?)?\s*\"?", text)
    if match and "'" in text:
        feet = Decimal(match.group(1))
        inches = Decimal(match.group(2)) if match.group(2) else Decimal("0")
        return QuantityValue(feet * 12 + inches, "in")

    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([a-z]+)", text)
    if match:
        try:
            number = Decimal(match.group(1))
        except InvalidOperation:
            return None
        unit = _UNIT_ALIASES.get(match.group(2))
        if unit:
            return QuantityValue(number, unit)

    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        try:
            return QuantityValue(Decimal(text), None)
        except InvalidOperation:
            return None
    return None


def quantities_match(reference: Any, prediction: Any, tolerance: Optional[Decimal] = None) -> bool:
    """Compare two quantities, converting only within a known dimension."""
    ref = parse_quantity(reference)
    pred = parse_quantity(prediction)
    if ref is None or pred is None:
        return False

    if ref.unit is None or pred.unit is None:
        return ref.unit == pred.unit and ref.value == pred.value

    ref_dim = _DIMENSION_UNITS.get(ref.unit)
    pred_dim = _DIMENSION_UNITS.get(pred.unit)
    if ref_dim is None or pred_dim is None or ref_dim != pred_dim:
        return False

    table = _DIMENSIONS[ref_dim]
    ref_base = ref.value * table[ref.unit]
    pred_base = pred.value * table[pred.unit]
    limit = tolerance if tolerance is not None else _TOLERANCE[ref_dim]
    return abs(ref_base - pred_base) <= limit


def canonical_quantity(value: Any) -> Optional[str]:
    """Render a quantity in its dimension's canonical unit (cm, g, second)."""
    parsed = parse_quantity(value)
    if parsed is None:
        return None
    if parsed.unit is None:
        return str(parsed.value)
    dimension = _DIMENSION_UNITS.get(parsed.unit)
    if dimension is None:
        return f"{parsed.value} {parsed.unit}"
    base = parsed.value * _DIMENSIONS[dimension][parsed.unit]
    canonical_unit = {"length": "cm", "mass": "g", "time": "s"}[dimension]
    return f"{base:.2f} {canonical_unit}"


# ---------------------------------------------------------------------------
# Scalar and list comparison
# ---------------------------------------------------------------------------

def _is_count_question(question: Optional[str]) -> bool:
    return bool(question and re.search(
        r"\bhow many\b|\bhow much\b|\bnumber of\b|\bcount\b", question, re.IGNORECASE))


def _mintaka_labels(input_sample: Optional[dict[str, Any]]) -> list[str]:
    if not input_sample:
        return []
    labels = []
    for entity in input_sample.get("answerEntity", []) or []:
        if isinstance(entity, dict) and entity.get("label"):
            labels.append(str(entity["label"]))
    return labels


def scalar_matches(
    reference: Any,
    prediction: Any,
    *,
    dataset: str = "",
    question: Optional[str] = None,
    input_sample: Optional[dict[str, Any]] = None,
) -> tuple[bool, bool, str]:
    """Return ``(normalized_match, partial_match, rule)`` for one answer value.

    Rule precedence is deliberate: the Mintaka ``answerEntity`` check runs before
    generic entity folding so that a curated label match is reported as such.
    """
    if normalize_text(reference) == normalize_text(prediction):
        return True, False, "text"

    date_match, date_partial = dates_match(reference, prediction)
    if date_match:
        return True, False, "date"
    if date_partial:
        return False, True, "date_precision"

    if quantities_match(reference, prediction):
        return True, False, "quantity"

    if dataset.lower() == "mintaka" and not _is_count_question(question):
        prediction_text = fold_entity(prediction)
        if prediction_text and any(
            prediction_text == fold_entity(label) for label in _mintaka_labels(input_sample)
        ):
            reference_text = fold_entity(reference)
            if reference_text.startswith(prediction_text + " "):
                # The prediction drops a qualifier the reference keeps, e.g.
                # "Chester" for "Chester, United Kingdom".  This is a
                # granularity difference, not a surface variant.
                return False, True, "granularity"
            return True, False, "mintaka_answer_entity_label"

    ref_folded = fold_entity(reference)
    if ref_folded and ref_folded == fold_entity(prediction):
        alias_used = (
            normalize_text(reference) in _ENTITY_ALIASES
            or normalize_text(prediction) in _ENTITY_ALIASES
            or ref_folded in _ALIAS_TARGETS
        )
        return True, False, "alias" if alias_used else "entity_fold"

    return False, False, "strict"


@dataclass(frozen=True)
class NormalizedComparison:
    strict_match: bool
    normalized_match: bool
    partial_match: bool = False
    rule: str = "strict"


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def compare_answers(
    prediction: Iterable[Any],
    reference: Iterable[Any],
    *,
    dataset: str = "",
    question: Optional[str] = None,
    input_sample: Optional[dict[str, Any]] = None,
) -> NormalizedComparison:
    """Compare unordered answer lists while preserving cardinality."""
    prediction = _as_list(prediction)
    reference = _as_list(reference)
    # Mirror the official evaluator exactly: plain lowercased set equality, with
    # no punctuation or whitespace normalisation.  ``strict_match`` must agree
    # with ``evaluation.correct`` so that the two metrics can be compared.
    if {str(x).lower() for x in prediction} == {str(x).lower() for x in reference}:
        return NormalizedComparison(True, True, False, "strict")

    if len(prediction) != len(reference):
        return NormalizedComparison(False, False, False, "cardinality")

    unmatched = list(reference)
    partial = False
    partial_rule = ""
    rules = []
    for pred in prediction:
        found = False
        for index, ref in enumerate(unmatched):
            matched, is_partial, rule = scalar_matches(
                ref, pred, dataset=dataset, question=question, input_sample=input_sample
            )
            if matched:
                unmatched.pop(index)
                rules.append(rule)
                found = True
                break
            if is_partial:
                partial = True
                partial_rule = rule
        if not found and not partial:
            return NormalizedComparison(False, False, False, "strict")
    if unmatched:
        return NormalizedComparison(False, False, partial, partial_rule or "partial")
    return NormalizedComparison(False, True, partial, "+".join(sorted(set(rules))) or "normalized")


# ---------------------------------------------------------------------------
# Mintaka verification entry point
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VerificationResult:
    strict_match: bool
    normalized_match: bool
    partial_match: bool
    rule: str
    reference_normalized: list[str]
    prediction_normalized: list[str]

    def as_dict(self) -> dict:
        return {
            "strict_match": self.strict_match,
            "normalized_match": self.normalized_match,
            "partial_match": self.partial_match,
            "rule": self.rule,
            "reference_normalized": self.reference_normalized,
            "prediction_normalized": self.prediction_normalized,
        }


def canonical_value(value: Any) -> str:
    """Canonical, human-readable form of a scalar answer for reporting."""
    date = canonical_date(value)
    if date is not None:
        return date
    quantity = canonical_quantity(value)
    if quantity is not None:
        return quantity
    return fold_entity(value) or normalize_text(value)


def verify_mintaka(
    prediction: Any,
    reference: Any,
    *,
    question: Optional[str] = None,
    input_sample: Optional[dict[str, Any]] = None,
) -> VerificationResult:
    """Verify a Mintaka prediction against the gold answer.

    ``prediction`` and ``reference`` may be scalars or lists. Returns whether the
    answer is strictly correct, semantically equivalent, or only partially
    correct, plus the rule that fired and canonicalised forms for auditing.
    """
    comparison = compare_answers(
        _as_list(prediction),
        _as_list(reference),
        dataset="mintaka",
        question=question,
        input_sample=input_sample,
    )
    return VerificationResult(
        strict_match=comparison.strict_match,
        normalized_match=comparison.normalized_match,
        partial_match=comparison.partial_match,
        rule=comparison.rule,
        reference_normalized=[canonical_value(v) for v in _as_list(reference)],
        prediction_normalized=[canonical_value(v) for v in _as_list(prediction)],
    )
