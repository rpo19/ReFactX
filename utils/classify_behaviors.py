"""Classify every answer in an evaluation log with Jev (TypeSafe System One).

The input is the JSONL log written by ``utils/eval.py`` (header line, one record
per question, optional ``macro_*`` summary). The output is the **same log** with
a ``behavior`` object added to every record, so the file remains readable by
``utils/eval.py`` consumers (``utils/make_latex_tables.py``, the Streamlit viewer,
the SFT builders, ...) and can be used unchanged on test sets.

Behaviour labels (see ``utils/behaviors.py``):

* ``best``       — correct, derived from the question entities through retrieved facts;
* ``acceptable`` — correct, known/parametric but verified with retrieved facts;
* ``bad``        — correct but unsupported by the retrieved facts;
* ``error``      — wrong, and the answer is not in the retrieved facts.

Examples::

    # pilot on 500 records, offline heuristic (no API key needed)
    python utils/classify_behaviors.py logs/.../mintaka.out.3.gz --limit 500 --mock

    # full run with Jev, 8 concurrent requests
    python utils/classify_behaviors.py logs/.../*.out* --workers 8 \
        --output-dir logs/behaviors --report behavior_report.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator, Optional

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils.answer_normalization import fold_entity  # noqa: E402
from utils.behaviors import (  # noqa: E402
    BEHAVIOR_LABELS,
    apply_behavior,
    build_questions,
    build_state,
    dataset_name,
    parse_jev_response,
)
from utils.jev_client import JevClient, JevError, MockJevClient  # noqa: E402
from utils.log_io import open_log  # noqa: E402

BEHAVIOR_METADATA_VERSION = "1"


def _as_list(value) -> list:
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _contains(haystack: str, needle) -> bool:
    text = fold_entity(needle)
    if not text:
        return False
    import re
    if len(text) < 3 or text.replace(".", "").isdigit():
        return re.search(r"(?<!\w)" + re.escape(text) + r"(?!\w)", haystack) is not None
    return text in haystack


def _mock_response(state: dict, questions: Optional[dict] = None) -> dict:
    """Deterministic offline labels, for pilots and tests."""
    correct = bool(state.get("answer_is_correct"))
    facts = fold_entity(" ".join(state.get("retrieved_facts") or []))
    trace = state.get("model_trace") or ""
    reference = _as_list(state.get("reference_answer"))
    prediction = _as_list(state.get("final_answer"))
    supported = any(_contains(facts, v) for v in reference) or any(_contains(facts, v) for v in prediction)
    derived = "<fact>" in trace.lower()
    if not correct:
        label = "error"
    elif supported and derived:
        label = "best"
    elif supported:
        label = "acceptable"
    else:
        label = "bad"
    return {
        "model": "mock",
        "answers": {
            "behavior": {
                "type": "choice",
                "choice": label,
                "confidence": 1.0,
                "probabilities": {name: float(name == label) for name in BEHAVIOR_LABELS},
            },
            "answer_supported_by_facts": {"type": "noul", "noul": 1.0 if supported else 0.0},
            "derived_from_facts": {"type": "noul", "noul": 1.0 if derived else 0.0},
        },
        "usage": {"input_tokens": max(1, len(json.dumps(state)) // 4), "output_tokens": 0},
    }


def _iter_records(fd) -> Iterator[dict]:
    """Yield parsed records, skipping blanks and the header/summary lines."""
    for line in fd:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        yield obj


def classify_log(
    input_path: Path,
    output_path: Path,
    client,
    *,
    dataset_override: Optional[str] = None,
    start: int = 0,
    limit: Optional[int] = None,
    workers: int = 8,
    chunk_size: int = 64,
    max_trace_chars: int = 8000,
    max_facts: int = 40,
    fail_fast: bool = False,
    quiet: bool = False,
    append: bool = False,
) -> dict:
    questions = build_questions()
    counts = {label: 0 for label in BEHAVIOR_LABELS}
    errors = 0
    seen = 0
    written = 0
    scheduled = 0
    summary: Optional[dict] = None

    with open_log(input_path) as fd:
        header = None
        for obj in _iter_records(fd):
            header = obj
            break
        if header is None:
            raise ValueError(f"{input_path} is empty")
        dataset = dataset_override or dataset_name(header)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        def work(record: dict):
            try:
                state = build_state(
                    record, dataset,
                    max_trace_chars=max_trace_chars, max_facts=max_facts,
                )
                response = client.classify(state, questions)
                return record, parse_jev_response(response), None
            except Exception as exc:  # noqa: BLE001 - recorded per record
                if fail_fast:
                    raise
                return record, None, f"{type(exc).__name__}: {exc}"

        write_header = not (append and output_path.exists())
        with output_path.open("a" if not write_header else "w", encoding="utf-8") as out, \
                ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            def flush(chunk: list[dict]) -> None:
                nonlocal errors, written
                if not chunk:
                    return
                for record, result, error in pool.map(work, chunk):
                    if error is not None:
                        errors += 1
                        enriched = apply_behavior(record, {"label": None, "error": error})
                    else:
                        label = result.get("label")
                        if label in counts:
                            counts[label] += 1
                        else:
                            result["label"] = None
                        enriched = apply_behavior(record, result)
                    out.write(json.dumps(enriched) + "\n")
                    written += 1
                out.flush()

            enriched_header = dict(header)
            enriched_header["behavior_metadata"] = {
                "version": BEHAVIOR_METADATA_VERSION,
                "model": getattr(client, "model", "mock"),
                "labels": BEHAVIOR_LABELS,
                "created": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            }
            if write_header:
                out.write(json.dumps(enriched_header) + "\n")

            chunk: list[dict] = []
            for obj in _iter_records(fd):
                if "macro_precision" in obj:
                    summary = obj
                    continue
                if "evaluation" not in obj:
                    continue
                seen += 1
                if seen <= start:
                    continue
                if limit is not None and scheduled >= limit:
                    break
                chunk.append(obj)
                scheduled += 1
                if len(chunk) >= chunk_size:
                    flush(chunk)
                    chunk = []
            if chunk:
                flush(chunk)

            if summary is not None and not append:
                summary = dict(summary)
                summary["behavior_counts"] = counts
                summary["behavior_errors"] = errors
                summary["behavior_usage"] = client.usage_summary()
                out.write(json.dumps(summary) + "\n")

    if not quiet:
        print(f"{input_path.name}: wrote {written} records to {output_path} "
              f"({', '.join(f'{k}={v}' for k, v in counts.items())}, errors={errors})")
    return {
        "input": str(input_path),
        "output": str(output_path),
        "dataset": dataset,
        "written": written,
        "counts": counts,
        "errors": errors,
        "usage": client.usage_summary(),
    }


def build_client(args):
    if args.mock:
        return MockJevClient(_mock_response)
    try:
        return JevClient(model=args.model, max_retries=args.retries)
    except JevError as exc:
        raise SystemExit(str(exc))


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=Path, help="evaluation logs (JSONL, optionally .gz)")
    parser.add_argument("--output-dir", type=Path, help="write enriched logs here (default: next to the input)")
    parser.add_argument("--output", type=Path, help="explicit output path (single input only)")
    parser.add_argument("--report", type=Path, help="write a JSON report here")
    parser.add_argument("--dataset", help="override the dataset label (mintaka/2wiki/webqsp/cwq)")
    parser.add_argument("--model", default="jev-1.14.0", help="Jev model id or alias")
    parser.add_argument("--start", type=int, default=0, help="skip the first N records (resume)")
    parser.add_argument("--limit", type=int, default=None, help="classify at most N records per input")
    parser.add_argument("--workers", type=int, default=8, help="concurrent Jev requests")
    parser.add_argument("--retries", type=int, default=5, help="retry attempts on 429/5xx")
    parser.add_argument("--max-trace-chars", type=int, default=8000)
    parser.add_argument("--max-facts", type=int, default=40)
    parser.add_argument("--mock", action="store_true", help="offline heuristic labels (no API key)")
    parser.add_argument("--fail-fast", action="store_true", help="abort on the first API error")
    parser.add_argument("--append", action="store_true",
                        help="append to an existing output (use with --start/--limit for chunks)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if args.output and len(args.inputs) != 1:
        parser.error("--output can only be used with a single input; use --output-dir otherwise")

    client = build_client(args)
    results = []
    for input_path in args.inputs:
        if args.output:
            output_path = args.output
        else:
            name = input_path.name
            if name.endswith(".gz"):
                name = name[:-3]
            output_path = (args.output_dir or input_path.parent) / f"{name}.behaviors"
        results.append(classify_log(
            input_path, output_path, client,
            dataset_override=args.dataset,
            start=args.start, limit=args.limit, workers=args.workers,
            max_trace_chars=args.max_trace_chars, max_facts=args.max_facts,
            fail_fast=args.fail_fast, quiet=args.quiet, append=args.append,
        ))

    report = {
        "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model": getattr(client, "model", "mock"),
        "results": results,
        "usage": client.usage_summary(),
    }
    if args.report:
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.report}")
    elif not args.quiet:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
