"""Create Markdown or LaTeX tables from one or more ReFactX JSONL evaluation logs.

The evaluator writes a header, one JSON object per question, and (when a run
finishes) a final object containing ``macro_*`` fields.  This utility does not
require that final object, so it is also useful while a log is still growing.

Examples:
    python utils/make_latex_tables.py logs/run1.out logs/run2.out
    python utils/make_latex_tables.py logs/*.out --output paper_tables.md
    python utils/make_latex_tables.py logs/*.out --latex --output paper_tables.tex
    python utils/make_latex_tables.py logs/run.out --strict
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, TextIO, Tuple


TOOL_PATTERNS = {
    "<fact>": re.compile(r"<fact>", re.IGNORECASE),
    "<count>": re.compile(r"<count>", re.IGNORECASE),
    "no further records": re.compile(r"no\s+further\s+(?:tokens|records)", re.IGNORECASE),
    "sentinel": re.compile(r"<no\s+further\s+(?:tokens|records)>|no\s+further\s+(?:tokens|records)", re.IGNORECASE),
}
DATE_PATTERNS = (
    re.compile(r"\b(?:19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2}\b"),
    re.compile(r"\b\d{1,2}[-/]\d{1,2}[-/](?:19|20)?\d{2}\b"),
    re.compile(r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}(?:,\s*\d{4})?\b", re.IGNORECASE)
)


def open_log(path: Path) -> TextIO:
    return gzip.open(str(path), "rt", encoding="utf-8") if path.suffix == ".gz" else path.open(encoding="utf-8")


def load_log(path: Path, strict: bool = False) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Optional[Dict[str, Any]], List[str]]:
    """Load complete sample records and ignore an incomplete final JSON line."""
    warnings: List[str] = []
    with open_log(path) as stream:
        try:
            first = next(stream)
            header = json.loads(first)
        except StopIteration as exc:
            raise ValueError(f"{path} is empty") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON header in {path}: {exc}") from exc

        samples: List[Dict[str, Any]] = []
        macro: Optional[Dict[str, Any]] = None
        for line_number, line in enumerate(stream, 2):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                message = f"{path}:{line_number}: ignoring incomplete/invalid JSON ({exc.msg})"
                if strict:
                    raise ValueError(message) from exc
                warnings.append(message)
                continue
            if "macro_precision" in obj:
                macro = obj
            elif "input_sample" in obj:
                samples.append(obj)
    return header, samples, macro, warnings


def type_field(header: Dict[str, Any], samples: Sequence[Dict[str, Any]], requested: str) -> str:
    if requested != "auto":
        return requested
    dataset = str(header.get("dataset", "")).lower()
    if "2wiki" in dataset:
        return "type"
    if "mintaka" in dataset:
        return "complexityType"
    for candidate in ("complexityType", "type", "question_type"):
        if any(candidate in s.get("input_sample", {}) for s in samples):
            return candidate
    return "complexityType"


def answer_value(sample: Dict[str, Any]) -> Any:
    inp = sample.get("input_sample", {})
    for key in ("answer", "answers", "gt_answer"):
        if key in inp:
            return inp[key]
    return sample.get("gt_answer")


def answer_items(value: Any) -> List[Any]:
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if value is None:
        return []
    return [value]


def answer_kind(value: Any) -> str:
    items = answer_items(value)
    if not items:
        return "empty"
    if any(isinstance(item, str) and any(pattern.search(item) for pattern in DATE_PATTERNS) for item in items):
        return "date"
    return "list" if isinstance(value, (list, tuple, set)) else "scalar"


def prediction_text(sample: Dict[str, Any]) -> str:
    # full_prediction is already prompt-excluded in utils/eval.py.  Do not use
    # full_sample here, since it includes the prompt and would inflate counts.
    return str(sample.get("full_prediction", sample.get("prediction", "")) or "")


def pct(n: int, d: int) -> str:
    return "--" if not d else f"{100.0 * n / d:.1f}%"


def mean(values: Iterable[Any]) -> Optional[float]:
    nums = [float(v) for v in values if isinstance(v, (int, float))]
    return sum(nums) / len(nums) if nums else None


def sample_metrics(group: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(group)
    ev = [s.get("evaluation", {}) for s in group]
    correct = sum(e.get("correct") == 1 for e in ev)
    dont_know = sum(e.get("dont_know") == 1 for e in ev)
    return {
        "n": n,
        "correct": correct,
        "accuracy": pct(correct, n),
        "f1": mean(e.get("f1") for e in ev),
        "precision": mean(e.get("precision") for e in ev),
        "recall": mean(e.get("recall") for e in ev),
        "dont_know": dont_know,
        "dont_know_pct": pct(dont_know, n),
    }


def run_name(path: Path, header: Dict[str, Any]) -> str:
    return str(header.get("experiment_name") or path.name)


def flatten_label(path: Path, header: Dict[str, Any]) -> str:
    model = header.get("model_name") or header.get("model")
    dataset = header.get("dataset")
    bits = [run_name(path, header)]
    if model:
        bits.append(str(model).split("/")[-1])
    if dataset:
        bits.append(str(dataset).split("/")[-1])
    return " / ".join(bits)


def run_rows(runs: Sequence[Tuple[Path, Dict[str, Any], List[Dict[str, Any]], Optional[Dict[str, Any]]]]) -> List[List[Any]]:
    rows = []
    for path, header, samples, macro in runs:
        m = sample_metrics(samples)
        macro_accuracy = f"{100.0 * float(macro['macro_correct_accuracy']):.1f}%" if macro and isinstance(macro.get("macro_correct_accuracy"), (int, float)) else m["accuracy"]
        macro_dk = f"{100.0 * float(macro['macro_dont_know']):.1f}%" if macro and isinstance(macro.get("macro_dont_know"), (int, float)) else m["dont_know_pct"]
        rows.append([flatten_label(path, header), m["n"], m["correct"], m["accuracy"], macro_accuracy, f"{m['f1']:.3f}" if m["f1"] is not None else "--", m["dont_know"], macro_dk])
    return rows


def grouped_rows(runs: Sequence[Tuple[Path, Dict[str, Any], List[Dict[str, Any]], Optional[Dict[str, Any]]]], field: str) -> List[List[Any]]:
    rows = []
    for path, header, samples, _ in runs:
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for sample in samples:
            groups[str(sample.get("input_sample", {}).get(field, "unknown"))].append(sample)
        for label, group in sorted(groups.items()):
            m = sample_metrics(group)
            rows.append([flatten_label(path, header), label, m["n"], m["accuracy"], f"{m['f1']:.3f}" if m["f1"] is not None else "--", m["dont_know"], m["dont_know_pct"]])
    return rows


def answer_rows(runs: Sequence[Tuple[Path, Dict[str, Any], List[Dict[str, Any]], Optional[Dict[str, Any]]]]) -> List[List[Any]]:
    rows = []
    for path, header, samples, _ in runs:
        groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        for sample in samples:
            value = answer_value(sample)
            items = answer_items(value)
            length = str(len(items)) if len(items) <= 5 else "6+"
            groups[(answer_kind(value), length)].append(sample)
        for (kind, length), group in sorted(groups.items()):
            m = sample_metrics(group)
            rows.append([flatten_label(path, header), kind, length, m["n"], m["accuracy"], m["dont_know_pct"]])
    return rows


def tool_rows(runs: Sequence[Tuple[Path, Dict[str, Any], List[Dict[str, Any]], Optional[Dict[str, Any]]]]) -> List[List[Any]]:
    rows = []
    for path, header, samples, _ in runs:
        n = len(samples)
        texts = [prediction_text(s) for s in samples]
        triple_counts = [s.get("triples", []) if isinstance(s.get("triples", []), list) else [] for s in samples]
        row: List[Any] = [flatten_label(path, header), n]
        for name, pattern in TOOL_PATTERNS.items():
            used = sum(bool(pattern.search(text)) for text in texts)
            row.extend([used, pct(used, n)])
        row.extend([f"{mean(len(t) for t in triple_counts):.2f}" if n else "--", sum(bool(t) for t in triple_counts), pct(sum(bool(t) for t in triple_counts), n)])
        rows.append(row)
    return rows


def latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(c, c) for c in text)


def latex_table(title: str, label: str, headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    alignment = "l" + "r" * (len(headers) - 1)
    lines = [r"\begin{table}[t]", r"\centering", "\\caption{" + latex_escape(title) + "}", "\\label{" + latex_escape(label) + "}", "\\begin{tabular}{" + alignment + "}", r"\toprule", " & ".join(latex_escape(h) for h in headers) + r" \\", r"\midrule"]
    lines.extend(" & ".join(latex_escape(v) for v in row) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def markdown_table(title: str, headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [f"### {title}", "", "| " + " | ".join(markdown_escape(h) for h in headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(markdown_escape(v) for v in row) + " |" for row in rows)
    return "\n".join(lines)


def render(runs: Sequence[Tuple[Path, Dict[str, Any], List[Dict[str, Any]], Optional[Dict[str, Any]]]], fields: Sequence[str], latex: bool = False) -> str:
    renderer = (lambda title, label, headers, rows: latex_table(title, label, headers, rows)) if latex else (lambda title, label, headers, rows: markdown_table(title, headers, rows))
    blocks = [renderer("Evaluation summary", "tab:eval-summary", ["Run", "N", "Correct", "Accuracy", "Macro accuracy", "F1", "Don't know", "Macro DK rate"], run_rows(runs))]
    for field in fields:
        blocks.append(renderer(f"Performance by question type ({field})", f"tab:eval-by-{field}", ["Run", "Type", "N", "Accuracy", "F1", "Don't know", "DK rate"], grouped_rows(runs, field)))
    blocks.append(renderer("Performance by answer shape and cardinality", "tab:eval-answer-shape", ["Run", "Shape", "Items", "N", "Accuracy", "DK rate"], answer_rows(runs)))
    blocks.append(renderer("Tool and retrieved-triple usage", "tab:eval-tools", ["Run", "N", "Fact N", "Fact rate", "Count N", "Count rate", "No-further N", "No-further rate", "Sentinel N", "Sentinel rate", "Avg triples", "Has triples N", "Has triples rate"], tool_rows(runs)))
    return "\n\n".join(blocks) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs", nargs="+", type=Path, help="JSONL logs, optionally .gz compressed")
    parser.add_argument("--output", type=Path, help="write the selected format here instead of stdout")
    parser.add_argument("--latex", action="store_true", help="output LaTeX tables (default: Markdown)")
    parser.add_argument("--type-field", choices=("auto", "type", "complexityType", "question_type"), default="auto")
    parser.add_argument("--strict", action="store_true", help="fail on malformed trailing JSON instead of ignoring it")
    args = parser.parse_args(argv)

    runs = []
    all_fields: List[str] = []
    for path in args.logs:
        try:
            header, samples, macro, warnings = load_log(path, args.strict)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        for warning in warnings:
            print(f"warning: {warning}", file=sys.stderr)
        if not samples:
            print(f"warning: {path}: no sample records", file=sys.stderr)
        field = type_field(header, samples, args.type_field)
        if field not in all_fields:
            all_fields.append(field)
        runs.append((path, header, samples, macro))

    # A mixed set of datasets can use different conventions; use all discovered
    # fields rather than silently assigning one dataset's field to every run.
    output = render(runs, all_fields, latex=args.latex)
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
