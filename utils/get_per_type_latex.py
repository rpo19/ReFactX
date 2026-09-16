"""Print per-type accuracy and correct-answer counts from an eval log.

Examples:
    python utils/get_per_type_latex.py logs/mintaka_test_sft.out.0
    python utils/get_per_type_latex.py logs/twowikimultihop_eval.out.0.gz \
        --type-field type --caption "2Wiki validation"
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, TextIO


def open_log(path: Path) -> TextIO:
    """Open a JSONL eval log, transparently handling gzip-compressed logs."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def load_log(path: Path) -> tuple[dict, list[dict]]:
    """Return the metadata header and sample records from an eval log."""
    with open_log(path) as log:
        try:
            header = json.loads(next(log))
        except StopIteration as exc:
            raise ValueError(f"{path} is empty") from exc

        samples = []
        for line_number, line in enumerate(log, start=2):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path} at line {line_number}") from exc
            if "input_sample" in record and "evaluation" in record:
                samples.append(record)
    return header, samples


def infer_type_field(header: dict, samples: Iterable[dict]) -> str:
    """Match the type-field convention used by browse_eval_results.ipynb."""
    dataset = str(header.get("dataset", ""))
    if "2wiki" in dataset.lower():
        return "type"
    if "mintaka" in dataset.lower():
        return "complexityType"

    # For other datasets, prefer a field actually present in the samples.
    for candidate in ("complexityType", "type"):
        if any(candidate in sample.get("input_sample", {}) for sample in samples):
            return candidate
    return "complexityType"


def per_type_rows(samples: Iterable[dict], type_field: str) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        input_sample = sample.get("input_sample", {})
        question_type = str(input_sample.get(type_field, "unknown"))
        grouped[question_type].append(sample)

    rows = []
    for question_type in sorted(grouped):
        group = grouped[question_type]
        correct = sum(
            sample.get("evaluation", {}).get("correct") == 1 for sample in group
        )
        count = len(group)
        rows.append(
            {
                "type": question_type,
                "correct": correct,
                "count": count,
                "accuracy": 100 * correct / count,
            }
        )

    total_count = sum(row["count"] for row in rows)
    total_correct = sum(row["correct"] for row in rows)
    if total_count:
        rows.append(
            {
                "type": "Total",
                "correct": total_correct,
                "count": total_count,
                "accuracy": 100 * total_correct / total_count,
            }
        )
    return rows


def latex_escape(value: str) -> str:
    """Escape characters that have special meaning in LaTeX."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def render_latex(rows: list[dict], caption: str, label: str) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Type & Correct & Accuracy \\",
        r"\midrule",
    ]
    lines.extend(
        f"{latex_escape(row['type'])} & {row['correct']} & {row['accuracy']:.1f}" + r"\% \\"
        for row in rows
    )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            f"\\caption{{{latex_escape(caption)}}}",
            f"\\label{{{latex_escape(label)}}}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="JSONL eval log (.out.* or .gz)")
    parser.add_argument(
        "--type-field",
        choices=("auto", "type", "complexityType"),
        default="auto",
        help="Field containing the question type (default: infer from dataset)",
    )
    parser.add_argument("--caption", help="LaTeX table caption")
    parser.add_argument("--label", help="LaTeX table label")
    args = parser.parse_args()

    header, samples = load_log(args.log)
    if not samples:
        parser.error(f"no sample records found in {args.log}")

    type_field = args.type_field
    if type_field == "auto":
        type_field = infer_type_field(header, samples)
    rows = per_type_rows(samples, type_field)

    caption = args.caption or f"Per-type accuracy ({args.log.name})"
    label = args.label or "tab:per-type-accuracy"
    print(render_latex(rows, caption, label))


if __name__ == "__main__":
    main()
