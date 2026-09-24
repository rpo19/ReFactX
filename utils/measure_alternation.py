"""Measure alternation between constrained tool calls and free reasoning.

The constrained generation pipeline interleaves two kinds of spans:

* **constrained tool spans**: ``<fact> ... </fact>`` and
  ``<count> ... </count>``, plus their exhausted-retrieval sentinel text
  ``<no further records>``;
* **free reasoning spans**: everything else the model writes, including the
  sentences before the first call, the sentences between two calls, and the
  final reasoning before ``<answer>``.

The hypothesis in the error-categorization report is that models answer better
when they alternate: retrieve, reason in free text, retrieve again. This module
turns that hypothesis into numbers.

Because the JSONL logs store decoded text rather than token ids, "free tokens"
is approximated by whitespace-delimited words. That is a proxy, not the exact
tokenizer count; the proxy is stable across models and is sufficient for ranking
examples and comparing groups.

Usage::

    python utils/measure_alternation.py logs/22_sept_2026/*.out.0 --min-free 4
    python utils/measure_alternation.py logs/22_sept_2026/*.out.0 --json out.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

TOOL_PATTERN = re.compile(
    r"<fact>.*?</fact>|<count>.*?</count>|<no further records>",
    re.DOTALL | re.IGNORECASE,
)
ANSWER_PATTERN = re.compile(r"<answer>.*?</answer>", re.DOTALL | re.IGNORECASE)
THINK_PATTERN = re.compile(r"</?think>", re.IGNORECASE)
WHITESPACE = re.compile(r"\s+")

SEQ_END = "<|"


@dataclass
class ExampleMeasurement:
    idx: int
    question: Optional[str]
    correct: Optional[int]
    answer_complete: Optional[bool]
    reached_max_tokens: Optional[bool]
    question_type: Optional[str]
    tool_blocks: int
    free_spans: int
    free_words: int
    free_word_counts: list[int] = field(default_factory=list)
    min_free_words: int = 0
    median_free_words: float = 0.0
    prefix_words: int = 0
    suffix_words: int = 0
    fact_calls: int = 0
    count_calls: int = 0
    sentinel_calls: int = 0
    constrained_words: int = 0

    @property
    def constrained_calls(self) -> int:
        return self.fact_calls + self.count_calls

    @property
    def free_per_fact(self) -> float:
        """Between-call free reasoning words per constrained fact call."""
        return self.free_words / self.fact_calls if self.fact_calls else 0.0


def _strip_answer(text: str) -> str:
    """Remove the final answer block so trailing prose does not count as reasoning."""
    match = list(ANSWER_PATTERN.finditer(text))
    if match:
        return text[: match[0].start()]
    return text


def _split_tool_spans(text: str) -> list[str]:
    """Return the free-reasoning spans between consecutive tool calls."""
    spans: list[str] = []
    cursor = 0
    for match in TOOL_PATTERN.finditer(text):
        spans.append(text[cursor:match.start()])
        cursor = match.end()
    spans.append(text[cursor:])
    return spans


def measure_completion(text: str) -> dict:
    """Measure free-reasoning spans around constrained tool calls."""
    text = text or ""
    body = _strip_answer(text)
    body = THINK_PATTERN.sub(" ", body)
    body = body.split(SEQ_END, 1)[0]

    blocks = TOOL_PATTERN.findall(body)
    spans = _split_tool_spans(body)
    # The first span is the lead-in before the first call and the last span is
    # the lead-in to <answer>; neither is reasoning *between* two tool calls.
    between = spans[1:-1] if len(spans) > 2 else []

    def words(span: str) -> int:
        return len([w for w in WHITESPACE.split(span.strip()) if w])

    between_counts = [words(span) for span in between]
    prefix = words(spans[0]) if spans else 0
    suffix = words(spans[-1]) if spans else 0

    fact_calls = sum(1 for block in blocks if block.lower().startswith("<fact>"))
    count_calls = sum(1 for block in blocks if block.lower().startswith("<count>"))
    sentinel_calls = len(blocks) - fact_calls - count_calls
    constrained_words = sum(words(block) for block in blocks)

    return {
        "tool_blocks": len(blocks),
        "free_spans": len(between_counts),
        "free_words": sum(between_counts),
        "free_word_counts": between_counts,
        "min_free_words": min(between_counts, default=0),
        "median_free_words": statistics.median(between_counts) if between_counts else 0.0,
        "prefix_words": prefix,
        "suffix_words": suffix,
        "fact_calls": fact_calls,
        "count_calls": count_calls,
        "sentinel_calls": sentinel_calls,
        "constrained_words": constrained_words,
    }


def iter_examples(rows: list[dict]) -> Iterable[tuple[int, dict]]:
    for idx, row in enumerate(rows[1:-1], 1):
        yield idx, row


def measure_log(path: Path, type_field: str = "complexityType") -> list[ExampleMeasurement]:
    with path.open() as handle:
        rows = [json.loads(line) for line in handle]

    measurements = []
    for idx, row in iter_examples(rows):
        text = str(row.get("full_prediction") or "")
        data = measure_completion(text)
        evaluation = row.get("evaluation", {}) or {}
        input_sample = row.get("input_sample", {}) or {}
        measurements.append(
            ExampleMeasurement(
                idx=idx,
                question=row.get("question"),
                correct=evaluation.get("correct"),
                answer_complete=row.get("answer_complete"),
                reached_max_tokens=row.get("reached_max_tokens"),
                question_type=input_sample.get(type_field) or input_sample.get("type"),
                **data,
            )
        )
    return measurements


def summarize(measurements: list[ExampleMeasurement], min_free: int) -> dict:
    """Aggregate alternation statistics and the reasoning/no-reasoning split."""
    total = len(measurements)
    with_tools = [m for m in measurements if m.tool_blocks > 0]
    alternating = [m for m in with_tools if m.min_free_words >= min_free]
    no_reasoning = [m for m in with_tools if m.min_free_words < min_free]

    def accuracy(group: list[ExampleMeasurement]) -> Optional[float]:
        scored = [m for m in group if m.correct is not None]
        if not scored:
            return None
        return sum(1 for m in scored if m.correct == 1) / len(scored)

    def completeness(group: list[ExampleMeasurement]) -> Optional[float]:
        if not group:
            return None
        return sum(1 for m in group if m.answer_complete) / len(group)

    return {
        "examples": total,
        "examples_with_tools": len(with_tools),
        "alternating_examples": len(alternating),
        "alternating_ratio_of_tool_examples": (len(alternating) / len(with_tools)) if with_tools else None,
        "alternating_ratio_of_all_examples": len(alternating) / total if total else None,
        "no_reasoning_examples": len(no_reasoning),
        "accuracy_alternating": accuracy(alternating),
        "accuracy_no_reasoning": accuracy(no_reasoning),
        "accuracy_overall": accuracy(measurements),
        "complete_alternating": completeness(alternating),
        "complete_no_reasoning": completeness(no_reasoning),
        "mean_tool_blocks": statistics.mean([m.tool_blocks for m in with_tools]) if with_tools else 0.0,
        "mean_min_free_words_alternating": (
            statistics.mean([m.min_free_words for m in alternating]) if alternating else 0.0
        ),
        "min_free_threshold": min_free,
    }


def by_question_type(measurements: list[ExampleMeasurement], min_free: int) -> dict:
    groups: dict[str, list[ExampleMeasurement]] = {}
    for measurement in measurements:
        groups.setdefault(str(measurement.question_type or "unknown"), []).append(measurement)
    return {name: summarize(group, min_free) for name, group in sorted(groups.items())}


def format_report(path: Path, summary: dict, type_summary: dict) -> str:
    ratio = summary["alternating_ratio_of_tool_examples"]
    ratio_text = "n/a" if ratio is None else f"{ratio:.3f}"
    lines = [f"\n### {path.name}"]
    lines.append(
        f"examples={summary['examples']} with_tools={summary['examples_with_tools']} "
        f"alternating={summary['alternating_examples']} "
        f"(ratio of tool examples={ratio_text})"
    )
    acc_alt = summary["accuracy_alternating"]
    acc_no = summary["accuracy_no_reasoning"]
    acc_all = summary["accuracy_overall"]
    lines.append(
        "accuracy: alternating={} no_reasoning={} overall={}".format(
            "n/a" if acc_alt is None else f"{acc_alt:.4f}",
            "n/a" if acc_no is None else f"{acc_no:.4f}",
            "n/a" if acc_all is None else f"{acc_all:.4f}",
        )
    )
    lines.append(f"min_free_threshold={summary['min_free_threshold']}")
    for name, sub in type_summary.items():
        lines.append(
            "  type={name:<14} n={examples:<4} with_tools={examples_with_tools:<4} "
            "alt_ratio={ratio} acc_alt={acc_alt} acc_no_reasoning={acc_no}".format(
                name=name,
                ratio=(
                    f"{sub['alternating_ratio_of_tool_examples']:.3f}"
                    if sub["alternating_ratio_of_tool_examples"] is not None else "n/a"
                ),
                acc_alt="n/a" if sub["accuracy_alternating"] is None else f"{sub['accuracy_alternating']:.3f}",
                acc_no="n/a" if sub["accuracy_no_reasoning"] is None else f"{sub['accuracy_no_reasoning']:.3f}",
                **sub,
            )
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", help="Evaluation JSONL log files")
    parser.add_argument(
        "--min-free",
        type=int,
        default=4,
        help="Minimum free words between two tool calls for an example to count as alternating",
    )
    parser.add_argument(
        "--type-field",
        default=None,
        help="Per-dataset type field (default: complexityType for Mintaka, type for 2Wiki)",
    )
    parser.add_argument("--json", default=None, help="Optional path to write machine-readable output")
    args = parser.parse_args()

    results = {}
    for log in args.logs:
        path = Path(log)
        type_field = args.type_field
        if type_field is None:
            type_field = "type" if "twowiki" in path.name.lower() else "complexityType"
        measurements = measure_log(path, type_field=type_field)
        summary = summarize(measurements, args.min_free)
        type_summary = by_question_type(measurements, args.min_free)
        print(format_report(path, summary, type_summary))
        results[path.name] = {
            "summary": summary,
            "by_type": type_summary,
            "examples": [m.__dict__ for m in measurements],
        }

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
