"""Shared definitions for the answer-support pipeline.

The classifier answers one question about each record:

    Consider the question and these facts. Is the answer supported by these
    facts? Do not use additional knowledge to decide whether the answer is true
    or not. Limit to simply decide whether the answer is supported by the given
    facts or not.

The facts are the **`triples` field of the evaluation record** (the decoded
``state.generated_triples`` written by ``utils/eval.py``).  There is no separate
``generated_triples`` key in the logs, and no regex extraction is used.

Output labels: ``supported`` / ``unsupported`` (plus the support probability).
``unsupported`` does not mean the answer is false — only that the retrieved
facts do not support it.

This module has no heavy dependencies (no torch/transformers/refactx) so it can
be imported by the log tools and by the API clients.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from utils.answer_normalization import fold_entity, parse_date, verify_answer

SUPPORT_LABELS = ["supported", "unsupported"]
BEHAVIOR_LABELS = SUPPORT_LABELS  # kept as the public name used by the tools

SUPPORT_INSTRUCTIONS = (
    "Consider the question and these facts. Is the answer supported by these facts? "
    "Do not use additional knowledge to decide whether the answer is true or not. "
    "Limit to simply decide whether the answer is supported by the given facts or not."
)

SUPPORT_CRITERIA = {
    "true": "The provided facts contain enough information to support the answer.",
    "false": "The provided facts do not support the answer, even if the answer may be true in reality.",
}

ANSWER_BLOCK_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
FACT_RE = re.compile(r"<fact>", re.IGNORECASE)


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

def build_state(record: dict, dataset: str, *, max_facts: int = 200,
                max_fact_chars: int = 500) -> dict:
    """The ``state`` sent to Jev: the question, the answer, and the log's triples.

    The facts come verbatim from ``record["triples"]`` (the generated triples
    written by ``utils/eval.py``); no regex or trace parsing is applied.
    """
    facts = [str(t)[:max_fact_chars] for t in (record.get("triples") or [])][:max_facts]
    return {
        "question": record.get("question"),
        "answer": record.get("prediction"),
        "facts": facts,
    }


def build_questions() -> dict:
    """The single yes/no support question sent to Jev."""
    return {
        "answer_supported": {
            "type": "noul",
            "instructions": SUPPORT_INSTRUCTIONS,
            "criteria": dict(SUPPORT_CRITERIA),
        }
    }


def parse_jev_response(response: dict, threshold: float = 0.5) -> dict:
    """Normalise a Jev response into the fields the pipeline stores."""
    answer = ((response or {}).get("answers") or {}).get("answer_supported") or {}
    probability = answer.get("noul")
    if probability is not None:
        probability = float(probability)
        label = "supported" if probability >= threshold else "unsupported"
        confidence = max(probability, 1.0 - probability)
        probabilities = {"supported": probability, "unsupported": 1.0 - probability}
    else:
        label = None
        confidence = None
        probabilities = {}
    return {
        "label": label,
        "probability": probability,
        "confidence": confidence,
        "probabilities": probabilities,
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
    """Deterministic stand-in for Jev: does the answer appear in the facts?"""
    facts = fold_entity(" ".join(str(t) for t in (record.get("triples") or [])))
    prediction = _as_list(record.get("prediction"))
    supported = bool(prediction) and any(_contains(facts, value) for value in prediction)
    return {
        "label": "supported" if supported else "unsupported",
        "probability": 1.0 if supported else 0.0,
        "confidence": 1.0,
        "probabilities": {"supported": 1.0 if supported else 0.0,
                          "unsupported": 0.0 if supported else 1.0},
        "model": "heuristic",
        "usage": {},
    }


def apply_behavior(record: dict, result: dict) -> dict:
    enriched = dict(record)
    enriched["behavior"] = result
    return enriched


def behavior_counts(records: list[dict]) -> dict:
    counts = {label: 0 for label in SUPPORT_LABELS}
    by_dataset: dict[str, dict] = {}
    for record in records:
        label = ((record.get("behavior") or {}).get("label")) or "unlabeled"
        counts[label] = counts.get(label, 0) + 1
        dataset = (record.get("_meta") or {}).get("dataset") or record.get("_dataset") or "unknown"
        by_dataset.setdefault(dataset, {})
        by_dataset[dataset][label] = by_dataset[dataset].get(label, 0) + 1
    return {"total": len(records), "labels": counts, "by_dataset": by_dataset}
