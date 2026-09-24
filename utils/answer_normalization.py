"""Opt-in semantic answer normalization for QA evaluation.

The existing evaluator intentionally keeps strict string/set matching.  This
module provides a separate, auditable comparison for analyses of formatting,
date, quantity, and Mintaka entity-alias mismatches.

The normalizer is deliberately conservative:
* dates are compared by parsed precision, not substring inclusion;
* a year reference may match a full date with the same year, but a full-date
  reference is not satisfied by a year-only prediction;
* quantities are converted only for explicitly supported units;
* Mintaka entity labels are used only for non-count answers;
* no automatic alias lookup or fuzzy matching is performed.
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


@dataclass(frozen=True)
class DateValue:
    year: int
    month: Optional[int] = None
    day: Optional[int] = None

    @property
    def precision(self) -> str:
        if self.month is None:
            return "year"
        if self.day is None:
            return "month"
        return "day"


@dataclass(frozen=True)
class QuantityValue:
    value: Decimal
    unit: Optional[str]


@dataclass(frozen=True)
class NormalizedComparison:
    strict_match: bool
    normalized_match: bool
    partial_match: bool = False
    rule: str = "strict"


def normalize_text(value: Any) -> str:
    """Normalize harmless textual variation without applying fuzzy matching."""
    value = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    value = value.replace("\u2018", "'").replace("\u2019", "'")
    value = re.sub(r"\s+", " ", value)
    return value.rstrip(" .")


def _year(two_digit_year: str) -> int:
    """Interpret Mintaka's two-digit dates using the usual 00-68/69-99 pivot."""
    value = int(two_digit_year)
    return 2000 + value if value <= 68 else 1900 + value


def parse_date(value: Any) -> Optional[DateValue]:
    """Parse the date formats observed in Mintaka and 2Wiki references."""
    text = normalize_text(value)
    if re.fullmatch(r"\d{4}", text):
        return DateValue(int(text))

    match = re.fullmatch(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", text)
    if match:
        return DateValue(int(match.group(1)), int(match.group(2)), int(match.group(3)))

    match = re.fullmatch(r"(\d{1,2})[- ]([a-z]+)[- ](\d{2,4})", text)
    if match and match.group(2) in _MONTHS:
        year = match.group(3)
        return DateValue(
            _year(year) if len(year) == 2 else int(year),
            _MONTHS[match.group(2)],
            int(match.group(1)),
        )

    match = re.fullmatch(r"([a-z]+) (\d{1,2}),? (\d{4})", text)
    if match and match.group(1) in _MONTHS:
        return DateValue(int(match.group(3)), _MONTHS[match.group(1)], int(match.group(2)))

    match = re.fullmatch(r"(\d{1,2}) ([a-z]+) (\d{4})", text)
    if match and match.group(2) in _MONTHS:
        return DateValue(int(match.group(3)), _MONTHS[match.group(2)], int(match.group(1)))

    return None


def dates_match(reference: Any, prediction: Any) -> tuple[bool, bool]:
    """Return (full/normalized match, partial match)."""
    ref = parse_date(reference)
    pred = parse_date(prediction)
    if ref is None or pred is None:
        return False, False
    if ref.precision == "year":
        return ref.year == pred.year, False
    if ref.precision != pred.precision:
        return False, ref.year == pred.year
    return ref == pred, False


_UNIT_ALIASES = {
    "cm": "cm", "centimeter": "cm", "centimeters": "cm",
    "m": "m", "meter": "m", "meters": "m",
    "km": "km", "kilometer": "km", "kilometers": "km",
    "in": "in", "inch": "in", "inches": "in",
    "lb": "lb", "lbs": "lb", "pound": "lb", "pounds": "lb",
    "kg": "kg", "kilogram": "kg", "kilograms": "kg",
    "week": "week", "weeks": "week",
    "day": "day", "days": "day",
    "year": "year", "years": "year",
    "second": "second", "seconds": "second",
}


def parse_quantity(value: Any) -> Optional[QuantityValue]:
    text = normalize_text(value)
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*'\s*(\d+(?:\.\d+)?)\s*\"", text)
    if match:
        feet = Decimal(match.group(1))
        inches = Decimal(match.group(2))
        return QuantityValue(feet * 12 + inches, "in")

    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([a-z]+)", text)
    if match:
        try:
            value_decimal = Decimal(match.group(1))
        except InvalidOperation:
            return None
        unit = _UNIT_ALIASES.get(match.group(2))
        if unit:
            return QuantityValue(value_decimal, unit)

    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        try:
            return QuantityValue(Decimal(text), None)
        except InvalidOperation:
            return None
    return None



def quantities_match(reference: Any, prediction: Any, tolerance: Decimal = Decimal("0.1")) -> bool:
    ref = parse_quantity(reference)
    pred = parse_quantity(prediction)
    if ref is None or pred is None:
        return False
    if ref.unit != pred.unit:
        # Explicitly support common metric length conversions and height units.
        if {ref.unit, pred.unit} <= {"cm", "in"}:
            ref = QuantityValue(ref.value * Decimal("0.3937007874"), "in") if ref.unit == "cm" else ref
            pred = QuantityValue(pred.value * Decimal("0.3937007874"), "in") if pred.unit == "cm" else pred
        else:
            return ref.unit is None and pred.unit is None and ref.value == pred.value
    return abs(ref.value - pred.value) <= tolerance


def _is_count_question(question: Optional[str]) -> bool:
    return bool(question and re.search(r"\bhow many\b|\bhow much\b|\bnumber of\b|\bcount\b", question, re.IGNORECASE))


def _normalize_entity_text(value: Any) -> str:
    text = normalize_text(value)
    return re.sub(r"[^\w\s]", "", text)


def _mintaka_labels(input_sample: Optional[dict[str, Any]]) -> list[str]:
    if not input_sample:
        return []
    labels = []
    for entity in input_sample.get("answerEntity", []) or []:
        if isinstance(entity, dict) and entity.get("label"):
            labels.append(str(entity["label"]))
    return labels


def scalar_matches(reference: Any, prediction: Any, *, dataset: str = "", question: Optional[str] = None,
                   input_sample: Optional[dict[str, Any]] = None) -> tuple[bool, bool, str]:
    """Return normalized match, partial match, and the applied rule."""
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
        prediction_text = _normalize_entity_text(prediction)
        if any(prediction_text == _normalize_entity_text(label) for label in _mintaka_labels(input_sample)):
            return True, False, "mintaka_answer_entity_label"

    return False, False, "strict"


def compare_answers(prediction: Iterable[Any], reference: Iterable[Any], *, dataset: str = "",
                    question: Optional[str] = None, input_sample: Optional[dict[str, Any]] = None) -> NormalizedComparison:
    """Compare unordered answer lists while preserving cardinality."""
    prediction = list(prediction or [])
    reference = list(reference or [])
    strict = {normalize_text(x) for x in prediction} == {normalize_text(x) for x in reference}
    if strict:
        return NormalizedComparison(True, True, False, "strict")

    if len(prediction) != len(reference):
        return NormalizedComparison(False, False, False, "cardinality")

    unmatched = list(reference)
    partial = False
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
            partial = partial or is_partial
        if not found and not partial:
            return NormalizedComparison(False, False, False, "strict")
    if unmatched:
        return NormalizedComparison(False, False, partial, "date_precision" if partial else "strict")
    return NormalizedComparison(False, True, partial, "+".join(sorted(set(rules))) or "normalized")
