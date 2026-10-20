"""Shared definitions for the answer-behaviour pipeline.

The four behaviours (the same labels are used by the Jev classifier, the SFT
dataset builder and the reports):

1. ``best``       — correct, reached by starting from the question's entities
                    and chaining retrieved external facts.
2. ``acceptable`` — correct, but the model already knew / proposed the answer
                    and then verified it with retrieved facts.
3. ``bad``        — correct, but the retrieved facts do not support it: right
                    answer from guessing or parametric knowledge.
4. ``error``      — wrong/missing answer, and the retrieved facts do not provide
                    it.

This module has no heavy dependencies (no torch/transformers/refactx) so it can
be imported by the log tools and by the API clients.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from utils.answer_normalization import fold_entity, parse_date, verify_answer

BEHAVIOR_LABELS = ["best", "acceptable", "bad", "error"]

BEHAVIOR_TITLES = {
    "best": "Best (grounded traversal)",
    "acceptable": "Acceptable (parametric + verified)",
    "bad": "Bad (correct but unsupported)",
    "error": "Error (wrong / answer not in facts)",
}

BEHAVIOR_DEFINITIONS = {
    "best": (
        "The answer is correct. The model starts from the entities or constraints in the "
        "question and reaches the answer by retrieving and chaining external facts; the "
        "final answer is supported by the retrieved facts and the trace actually uses them."
    ),
    "acceptable": (
        "The answer is correct. The model effectively knows or proposes the answer from "
        "prior/parametric knowledge and then checks or supports it with the retrieved "
        "facts. The facts confirm the answer, but discovering it did not require them."
    ),
    "bad": (
        "The answer is correct, but the retrieved facts do not support it. The model used "
        "wrong, irrelevant or insufficient facts and still produced the right answer by "
        "guessing or parametric knowledge (right answer, wrong or absent evidence)."
    ),
    "error": (
        "The answer is wrong or missing, and the retrieved facts do not provide the answer. "
        "The model failed to find or use the external facts needed to answer."
    ),
}

JUDGE_INSTRUCTIONS = (
    "Classify how the model produced its final answer. Use `answer_is_correct` as the "
    "correctness definition (it is the verified label, not the model's opinion). Read the "
    "question, the retrieved facts, the model trace and the final answer. Choose exactly "
    "one label. Then answer the two companion yes/no questions."
)

ANSWER_BLOCK_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
FACT_RE = re.compile(r"<fact>\s*(.*?)\s*</fact>", re.DOTALL | re.IGNORECASE)
COUNT_RE = re.compile(r"<count>\s*(.*?)\s*</count>", re.DOTALL | re.IGNORECASE)
NO_FURTHER_RE = re.compile(r"no\s+further\s+(?:tokens|records)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Dataset / type helpers (shared by the classifier and the builder)
# ---------------------------------------------------------------------------

def dataset_name(meta: dict) -> str:
    raw = str(meta.get("dataset", "") or meta.get("experiment_name", "")).lower()
    for name in ("mintaka", "webqsp", "cwq"):
        if name in raw:
            return name
    if "2wiki" in raw or "twowiki" in raw:
        return "2wiki"
    return raw or "unknown"


def question_type(dataset: str, input_sample: dict) -> str:
    if dataset == "mintaka":
        return str(input_sample.get("complexityType", "unknown"))
    if dataset == "2wiki":
        return str(input_sample.get("type", "unknown"))
    if dataset == "webqsp":
        return "generic"
    entities = input_sample.get("q_entity") or []
    return "multi-entity" if len(entities) > 1 else "single-entity"


def answer_shape(reference: Any) -> str:
    refs = _as_list(reference)
    lowered = [str(x).strip().lower() for x in refs]
    if refs and all(x in {"yes", "no", "true", "false"} for x in lowered):
        return "yes/no"
    if any(parse_date(x) is not None for x in refs):
        return "date"
    if refs and all(re.fullmatch(r"[+-]?\d+(?:\.\d+)?", str(x).replace(",", "").strip()) for x in refs):
        return "numeric"
    if len(refs) > 1:
        return "list"
    return "entity"


def _as_list(value: Any) -> list:
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


def is_positive(record: dict, verified: bool = False, dataset: Optional[str] = None) -> bool:
    """The repo's positive definition: correct (or verified) and tool-using."""
    if not (record.get("triples") or []):
        return False
    if record.get("reached_max_tokens"):
        return False
    if verified:
        verification = verify_record(record, dataset)
        return bool(verification.normalized_match)
    return (record.get("evaluation") or {}).get("correct") == 1


def verify_record(record: dict, dataset: Optional[str] = None):
    dataset = dataset or dataset_name(record.get("_meta") or {})
    return verify_answer(
        _as_list(record.get("prediction")),
        _as_list(record.get("gt_answer")),
        dataset=dataset,
        question=record.get("question"),
        input_sample=record.get("input_sample"),
    )


def is_verified_correct(record: dict, dataset: Optional[str] = None) -> bool:
    return bool(verify_record(record, dataset).normalized_match)


# ---------------------------------------------------------------------------
# Answer parsing / metrics (mirrors utils/eval.py)
# ---------------------------------------------------------------------------

def parse_answer(full_text: str) -> Optional[list]:
    """Parse the ``<answer>`` block the same way ``refactx.get_answer`` does."""
    text = full_text or ""
    match = ANSWER_BLOCK_RE.search(text)
    if match:
        raw = match.group(1).strip()
        try:
            value = json.loads(raw)
            return value if isinstance(value, list) else [value]
        except json.JSONDecodeError:
            return None
    # Fallback to the evaluator's line-based convention.
    marker = "<answer>\n"
    if marker in text:
        raw = text.split(marker, 1)[1].split("\n", 1)[0].strip()
        try:
            value = json.loads(raw)
            return value if isinstance(value, list) else [value]
        except json.JSONDecodeError:
            return None
    return None


def calculate_metrics(prediction: Any, reference: Any, lowercase: bool = True) -> dict:
    """Precision / recall / F1 / correct / dont_know, matching utils/eval.py."""
    prediction = _as_list(prediction)
    reference = _as_list(reference)
    if not prediction:
        return {"precision": 0, "recall": 0, "f1": 0, "correct": 0, "dont_know": 0}
    if lowercase:
        prediction = [str(p).lower() for p in prediction]
        reference = [str(r).lower() for r in reference]
    prediction_set = set(prediction)
    reference_set = set(reference)
    precision = len(prediction_set & reference_set) / len(prediction_set) if prediction_set else 0
    recall = len(prediction_set & reference_set) / len(reference_set) if reference_set else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    correct = int(prediction_set == reference_set)
    dont_know = int("i don't know" in prediction[0].lower())
    return {"precision": precision, "recall": recall, "f1": f1,
            "correct": correct, "dont_know": dont_know}


# ---------------------------------------------------------------------------
# Jev request / response
# ---------------------------------------------------------------------------

def clip_trace(text: str, max_chars: int = 8000) -> str:
    """Keep the head and the tail of a long trace."""
    text = text or ""
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.6)
    tail = max_chars - head
    return text[:head] + "\n...[truncated]...\n" + text[-tail:]


def build_state(record: dict, dataset: str, *, max_trace_chars: int = 8000,
                max_facts: int = 40, max_fact_chars: int = 300) -> dict:
    """The ``state`` object sent to Jev."""
    triples = [str(t)[:max_fact_chars] for t in (record.get("triples") or [])][:max_facts]
    verification = verify_record(record, dataset)
    return {
        "dataset": dataset,
        "question_type": question_type(dataset, record.get("input_sample") or {}),
        "question": record.get("question"),
        "answer_is_correct": bool(verification.normalized_match),
        "evaluator_correct": (record.get("evaluation") or {}).get("correct") == 1,
        "reference_answer": record.get("gt_answer"),
        "final_answer": record.get("prediction"),
        "retrieved_facts": triples,
        "model_trace": clip_trace(record.get("full_prediction") or "", max_trace_chars),
    }


def build_questions() -> dict:
    """The named questions sent to Jev (one Choice + two Nouls)."""
    return {
        "behavior": {
            "type": "choice",
            "instructions": JUDGE_INSTRUCTIONS,
            "criteria": dict(BEHAVIOR_DEFINITIONS),
        },
        "answer_supported_by_facts": {
            "type": "noul",
            "instructions": (
                "Do the retrieved facts contain enough information to support the final "
                "answer, regardless of how the model produced it?"
            ),
            "criteria": {
                "true": "The retrieved facts entail or directly support the final answer.",
                "false": "The retrieved facts do not support the final answer.",
            },
        },
        "derived_from_facts": {
            "type": "noul",
            "instructions": (
                "Does the model trace derive the final answer from the retrieved facts "
                "(reasoning from the question entities through the facts), rather than "
                "asserting an answer it already knew?"
            ),
            "criteria": {
                "true": "The trace uses the retrieved facts to find or compute the answer.",
                "false": "The trace asserts the answer without deriving it from the facts.",
            },
        },
    }


def parse_jev_response(response: dict) -> dict:
    """Normalise a Jev response into the fields the pipeline stores."""
    answers = (response or {}).get("answers") or {}
    behavior = answers.get("behavior") or {}
    label = behavior.get("choice")
    if label not in BEHAVIOR_LABELS:
        label = None
    supported = (answers.get("answer_supported_by_facts") or {}).get("noul")
    derived = (answers.get("derived_from_facts") or {}).get("noul")
    diagnostic = None
    if label == "error" and supported is not None and supported >= 0.5:
        diagnostic = "wrong_with_facts"
    elif label in {"best", "acceptable", "bad"}:
        diagnostic = label
    return {
        "label": label,
        "confidence": behavior.get("confidence"),
        "probabilities": behavior.get("probabilities") or {},
        "supported_probability": supported,
        "derived_probability": derived,
        "diagnostic": diagnostic,
        "model": (response or {}).get("model"),
        "usage": (response or {}).get("usage") or {},
    }


# ---------------------------------------------------------------------------
# Offline heuristic (mock mode / fallback)
# ---------------------------------------------------------------------------

def _contains(haystack: str, needle: Any) -> bool:
    text = fold_entity(needle)
    if not text:
        return False
    if len(text) < 3 or text.replace(".", "").isdigit():
        return re.search(r"(?<!\w)" + re.escape(text) + r"(?!\w)", haystack) is not None
    return text in haystack


def heuristic_behavior(record: dict, dataset: str) -> dict:
    """A deterministic, deliberately conservative stand-in for Jev (offline mode)."""
    verification = verify_record(record, dataset)
    facts = " ".join(str(t) for t in (record.get("triples") or []))
    facts_folded = fold_entity(facts)
    reference = _as_list(record.get("gt_answer"))
    prediction = [str(p) for p in _as_list(record.get("prediction"))]
    supported = any(_contains(facts_folded, v) for v in reference) or (
        bool(prediction) and any(_contains(facts_folded, v) for v in prediction)
    )
    used = bool(FACT_RE.search(record.get("full_prediction") or ""))
    if not verification.normalized_match:
        label = "error"
    elif supported and used:
        label = "best"
    elif supported:
        label = "acceptable"
    else:
        label = "bad"
    return {
        "label": label,
        "confidence": 1.0,
        "probabilities": {label: 1.0},
        "supported_probability": 1.0 if supported else 0.0,
        "derived_probability": 1.0 if used else 0.0,
        "diagnostic": label,
        "model": "heuristic",
        "usage": {},
    }


def apply_behavior(record: dict, result: dict) -> dict:
    enriched = dict(record)
    enriched["behavior"] = result
    return enriched


def behavior_counts(records: list[dict]) -> dict:
    counts = {label: 0 for label in BEHAVIOR_LABELS}
    by_dataset: dict[str, dict] = {}
    for record in records:
        label = ((record.get("behavior") or {}).get("label")) or "unlabeled"
        counts[label] = counts.get(label, 0) + 1
        dataset = (record.get("_meta") or {}).get("dataset") or record.get("_dataset") or "unknown"
        by_dataset.setdefault(dataset, {})
        by_dataset[dataset][label] = by_dataset[dataset].get(label, 0) + 1
    return {"total": len(records), "labels": counts, "by_dataset": by_dataset}
