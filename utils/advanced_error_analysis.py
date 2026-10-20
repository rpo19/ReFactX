"""Advanced error analysis for ReFactX dataset-evaluation logs.

Reads the JSONL logs written by :mod:`utils.eval` -- a metadata header, one
record per question, and (when the run finished) a final ``macro_*`` summary
line; the files live under ``ReFactX/logs/*.out.*`` -- classifies every
evaluated question into a compact error taxonomy, and renders a Markdown
report.

The tool deliberately reuses two existing modules instead of re-deriving their
logic:

* :mod:`utils.log_analysis` provides the constrained-generation primitives: the
  ``<s> <r> <o> .`` extraction, the retrieved-triple bookkeeping, and the
  hallucinated-triple check that discounts re-quotes and truncated lookups.
* :mod:`utils.answer_normalization` provides the conservative semantic
  comparison used to separate genuine mistakes from capitalization, alias,
  date and quantity formatting mismatches.

Beyond the mechanical failures (truncation, parser/format), the taxonomy asks
whether the evidence needed to answer was *present in the retrieved triples*.
A strict-wrong answer whose reference is absent from every retrieved triple is
labelled ``retrieval_gap`` (the harness never surfaced the fact); a strict-wrong
answer whose reference *is* present is labelled ``grounding_error`` (the fact
was available but the model selected, composed or resolved it incorrectly).

Usage::

    python utils/advanced_error_analysis.py logs/*.out.* \
        --output error_categorization/advanced_error_analysis_report.md
    python utils/advanced_error_analysis.py logs/mintaka_train_eval_qwen38_27b.out.3 \
        --json analysis.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils.answer_normalization import compare_answers  # noqa: E402
from utils.log_analysis import (  # noqa: E402
    SENTINEL,
    hallucinated_triples,
    open_log,
    record_triples,
)

# Primary categories, in report order.  Each failed question gets exactly one.
CATEGORY_ORDER = [
    ("correct", "Correct"),
    ("truncation", "Generation truncation / non-termination"),
    ("abstention", "Abstention / unsupported answer"),
    ("no_answer", "Answer-format / parser failure"),
    ("normalization", "Answer-extraction / normalization mismatch"),
    ("wrong_reference_present", "Wrong - reference present in retrieved triples"),
    ("wrong_reference_absent", "Wrong - reference absent from retrieved triples"),
]
CATEGORY_LABELS = dict(CATEGORY_ORDER)

# Retrieved-triple-count buckets (inclusive upper bounds).
TRIPLE_BUCKETS = [(0, 0), (1, 2), (3, 5), (6, 10), (11, 20), (21, 10 ** 9)]
TRIPLE_BUCKET_LABELS = ["0", "1-2", "3-5", "6-10", "11-20", "21+"]

ANSWER_BLOCK_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
FACT_RE = re.compile(r"<fact>", re.IGNORECASE)
COUNT_RE = re.compile(r"<count>", re.IGNORECASE)
ABSTAIN_RE = re.compile(r"i\s+don'?t\s+know|do not know|cannot (?:be )?(?:answer|determine)|insufficient", re.IGNORECASE)

MAX_EXAMPLES = 4
EXAMPLE_TEXT_LIMIT = 260


def dataset_name(meta: dict) -> str:
    raw = str(meta.get("dataset", "") or meta.get("experiment_name", "")).lower()
    if "mintaka" in raw:
        return "mintaka"
    if "2wiki" in raw or "twowiki" in raw:
        return "2wiki"
    return raw or "unknown"


def type_of(record: dict) -> str:
    inp = record.get("input_sample") or {}
    for key in ("complexityType", "type", "question_type"):
        value = inp.get(key)
        if value:
            return str(value)
    return "unknown"


def answer_block(full_prediction: str) -> Optional[str]:
    matches = ANSWER_BLOCK_RE.findall(full_prediction or "")
    return matches[-1].strip() if matches else None


def _normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().lower().rstrip(" .")


BOOLEAN_ANSWERS = {"yes", "no", "true", "false"}


def is_boolean(reference: list) -> bool:
    return bool(reference) and all(_normalize(r) in BOOLEAN_ANSWERS for r in reference)


def _contains(haystack: str, needle: Any) -> bool:
    """Heuristic containment test with a token boundary for short/numeric values."""
    text = _normalize(needle)
    if not text or not haystack:
        return False
    if len(text) < 3 or text.isdigit():
        return re.search(r"(?<!\w)" + re.escape(text) + r"(?!\w)", haystack) is not None
    return text in haystack


def classify(record: dict, dataset: str) -> dict:
    """Return a compact feature dict with the primary category and secondary flags."""
    ev = record.get("evaluation") or {}
    full_prediction = record.get("full_prediction") or ""
    prediction = record.get("prediction") or []
    if not isinstance(prediction, list):
        prediction = [prediction]
    gt = record.get("gt_answer")
    reference = gt if isinstance(gt, list) else ([] if gt is None else [gt])

    correct = int(ev.get("correct") == 1)
    truncated = int(bool(record.get("reached_max_tokens")))
    answered = int(bool(prediction))
    dont_know = int(bool(ev.get("dont_know")) or bool(ABSTAIN_RE.search(str(prediction[0]) if prediction else "")))

    raw_triples = record.get("triples") or []
    key_triples = record_triples(record)
    n_triples = len(key_triples)
    haystack = " ".join(str(t) for t in raw_triples).lower()

    gt_present = any(_contains(haystack, g) for g in reference)
    pred_grounded = any(_contains(haystack, p) for p in prediction)

    comparison = compare_answers(
        prediction, reference, dataset=dataset,
        question=record.get("question"), input_sample=record.get("input_sample"),
    )
    norm_match = bool(comparison.normalized_match)
    rule = comparison.rule

    n_fact_calls = len(FACT_RE.findall(full_prediction))
    n_count_calls = len(COUNT_RE.findall(full_prediction))
    # ``<fact>`` without a matching ``</fact>``: the last lookup was cut off.
    incomplete_calls = max(0, n_fact_calls - len(re.findall(r"</fact>", full_prediction, re.IGNORECASE)))
    hallucinated = hallucinated_triples(record, incomplete=incomplete_calls)

    boolean = is_boolean(reference)

    if correct:
        category = "correct"
    elif norm_match:
        category = "normalization"
    elif dont_know:
        category = "abstention"
    elif truncated:
        category = "truncation"
    elif not answered:
        category = "no_answer"
    elif not gt_present:
        category = "wrong_reference_absent"
    else:
        category = "wrong_reference_present"

    flags = {
        "truncated": truncated,
        "zero_triples": int(n_triples == 0),
        "incomplete_calls": int(incomplete_calls > 0),
        "sentinel": int(SENTINEL.lower() in full_prediction.lower() or "no further records" in full_prediction.lower()),
        "hallucination": int(bool(hallucinated)),
        "partial_date": int(bool(comparison.partial_match)),
        "cardinality": int(comparison.rule == "cardinality"),
        "boolean": int(boolean),
        "gt_present": int(gt_present),
        "pred_grounded": int(pred_grounded),
        "dont_know": dont_know,
        "norm_match": int(bool(norm_match)),
    }

    return {
        "type": type_of(record),
        "category": category,
        "correct": correct,
        "f1": float(ev.get("f1", 0.0) or 0.0),
        "answered": answered,
        "answer_complete": int(bool(record.get("answer_complete"))),
        "dont_know": dont_know,
        "truncated": truncated,
        "norm_match": int(bool(norm_match)),
        "rule": rule,
        "boolean": boolean,
        "n_triples": n_triples,
        "n_fact_calls": n_fact_calls,
        "n_count_calls": n_count_calls,
        "incomplete_calls": incomplete_calls,
        "n_hallucinated": len(hallucinated),
        "sentinel": flags["sentinel"],
        "n_new_tokens": int(record.get("new_tokens_generated", 0) or 0),
        "flags": flags,
        "question": record.get("question", ""),
        "gt_answer": gt,
        "prediction": prediction,
        "answer_block": answer_block(full_prediction),
    }


def scan(path: Path, dataset_override: Optional[str] = None) -> dict:
    """Stream one log and return its metadata, summary and per-record features."""
    meta: dict = {}
    summary: Optional[dict] = None
    features: list[dict] = []
    dataset = dataset_override or ""

    with open_log(path) as fd:
        for lineno, line in enumerate(fd):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if lineno == 0 or (not meta and "evaluation" not in obj):
                meta = obj
                continue
            if "macro_precision" in obj:
                summary = obj
                continue
            if "evaluation" not in obj:
                continue
            features.append(classify(obj, dataset or dataset_name(meta)))

    return {"path": path, "meta": meta, "summary": summary, "features": features,
            "dataset": dataset or dataset_name(meta)}


def _percent(n: int, d: int) -> str:
    return "--" if not d else f"{100.0 * n / d:.1f}%"


def _mean(values: Iterable[Any]) -> Optional[float]:
    nums = [float(v) for v in values if isinstance(v, (int, float))]
    return sum(nums) / len(nums) if nums else None


def aggregate(scan_result: dict) -> dict:
    features = scan_result["features"]
    n = len(features)
    cats = Counter(f["category"] for f in features)
    types: dict[str, list[dict]] = defaultdict(list)
    buckets: Counter = Counter()
    bucket_correct: Counter = Counter()
    for f in features:
        types[f["type"]].append(f)
        for (lo, hi), label in zip(TRIPLE_BUCKETS, TRIPLE_BUCKET_LABELS):
            if lo <= f["n_triples"] <= hi:
                buckets[label] += 1
                bucket_correct[label] += f["correct"]
                break

    macro = scan_result.get("summary") or {}
    headline = {
        "n": n,
        "accuracy": _percent(sum(f["correct"] for f in features), n),
        "macro_accuracy": _percent(round(macro.get("macro_correct_accuracy", 0) * n), n) if macro else "--",
        "f1": _mean(f["f1"] for f in features),
        "answered": _percent(sum(f["answered"] for f in features), n),
        "dont_know": _percent(sum(f["dont_know"] for f in features), n),
        "truncated": _percent(sum(f["truncated"] for f in features), n),
        "no_answer": _percent(sum(1 for f in features if f["category"] == "no_answer"), n),
        "normalization_recovered": _percent(sum(1 for f in features if f["norm_match"] and not f["correct"]), n),
        "zero_triples": _percent(sum(f["n_triples"] == 0 for f in features), n),
        "sentinel": _percent(sum(f["sentinel"] for f in features), n),
        "count_tool": _percent(sum(f["n_count_calls"] > 0 for f in features), n),
        "incomplete_calls": sum(f["incomplete_calls"] for f in features),
        "hallucination_records": sum(f["n_hallucinated"] > 0 for f in features),
        "hallucinated_triples": sum(f["n_hallucinated"] for f in features),
        "avg_triples": _mean(f["n_triples"] for f in features),
        "avg_tokens": _mean(f["n_new_tokens"] for f in features),
        "median_tokens": statistics.median([f["n_new_tokens"] for f in features]) if n else None,
        "wrong_absent_boolean": sum(1 for f in features if f["category"] == "wrong_reference_absent" and f["boolean"]),
        "wrong_absent_derived": sum(1 for f in features if f["category"] == "wrong_reference_absent" and not f["boolean"]),
    }
    return {"headline": headline, "categories": cats, "types": types,
            "buckets": buckets, "bucket_correct": bucket_correct, "n": n,
            "norm_rules": Counter(f["rule"] for f in features if f["category"] == "normalization")}


def _example(f: dict) -> str:
    def short(value: Any, limit: int = EXAMPLE_TEXT_LIMIT) -> str:
        text = "" if value is None else str(value).replace("\n", " ")
        return text if len(text) <= limit else text[: limit - 3] + "..."
    flags = [name for name, on in f["flags"].items() if on and name not in {"gt_present", "pred_grounded", "norm_match"}]
    return (f"- **[{f['type']}]** {short(f['question'])}\n"
            f"  - reference: `{short(f['gt_answer'], 120)}`\n"
            f"  - prediction: `{short(f['prediction'], 120)}`  "
            f"(triples: {f['n_triples']}, tokens: {f['n_new_tokens']}"
            f"{', flags: ' + ', '.join(flags) if flags else ''})\n"
            f"  - final answer block: `{short(f['answer_block'], 160)}`")


def render(results: list[dict], aggregates: list[dict], generated: str) -> str:
    out: list[str] = []
    add = out.append

    add("# Advanced error analysis of the ReFactX evaluation logs")
    add("")
    add(f"_Generated by `utils/advanced_error_analysis.py` on {generated}._")
    add("")
    add("This report is a machine-assisted, rule-based categorisation of every "
        "question in the logs below. It complements "
        "`ERROR_CATEGORIZATION_GUIDE.md` (which explains the evidence-based, "
        "manual procedure) and `error_categorization_report.md` (which interprets "
        "the 22 September test runs). The primary category assigned to each failed "
        "question is derived here from the record fields, the retrieved triples and "
        "a conservative semantic answer comparison, not from a human judgement.")
    add("")

    add("## Files analysed")
    add("")
    add("| File | Dataset | Split | Model | Records | Max new tokens |")
    add("|---|---|---|---|---:|---:|")
    for res, agg in zip(results, aggregates):
        meta = res["meta"]
        gen = meta.get("generation_config") or {}
        add("| `{name}` | {ds} | {split} | {model} | {n} | {budget} |".format(
            name=res["path"].name,
            ds=meta.get("dataset", "?"),
            split=meta.get("dataset_split", "?"),
            model=meta.get("model_name", "?"),
            n=agg["n"],
            budget=gen.get("max_new_tokens", meta.get("max_new_tokens", "?")),
        ))
    add("")
    add("> These are teacher-distillation evaluations on **training** splits "
        "(`dataset_split: train`), used to build the SFT mixture; they are not the "
        "held-out test runs of `error_categorization_report.md`. Accuracy figures "
        "here therefore describe the teacher's behaviour on the data it supervises, "
        "not benchmark performance.")
    add("")

    if results and all("27B" in str(r["meta"].get("model_name", "")) for r in results) \
            and all(r["meta"].get("dataset_split") == "train" for r in results):
        add("## Relation to the paper")
        add("")
        add("These are Qwen3.8-27B teacher runs on the training splits, i.e. the "
            "setting the paper's *Training* section uses to distil the SFT "
            "supervision. The per-type accuracies below share the same shape as the "
            "preliminary per-type table in the draft (comparative strongest, "
            "superlative/ordinal/difference weakest), and the truncation share "
            "quantifies how much of the training split never yields a usable trace "
            "at the current per-question token budget. Because these numbers come "
            "from a different execution than the table in the draft, treat them as a "
            "fresh measurement of the same setting rather than the draft's exact "
            "values.")
        add("")

    add("## Method")
    add("")
    add("For each record the tool extracts: strict correctness and F1 from "
        "`evaluation`; whether generation reached `max_new_tokens`; whether an "
        "answer was parsed; the retrieved triples; and the tool calls in "
        "`full_prediction`. Two answer comparisons are used:")
    add("")
    add("- the **strict** evaluator comparison already stored in `evaluation.correct`;")
    add("- a **normalized** comparison from `utils/answer_normalization.py` "
        "(case/punctuation, date precision, quantities and Mintaka answer-entity "
        "labels), used to flag formatting/alias mismatches the strict evaluator "
        "counts as errors, plus a century-ambiguity date rule for two-digit years "
        "such as `13-Jun-46` against `13 June 1946`.")
    add("")
    add("Primary category rules, applied in order:")
    add("")
    add("1. **Correct** - `evaluation.correct == 1`.")
    add("2. **Normalization mismatch** - strict-wrong but the normalized comparison matches (or a partial date match, flagged separately).")
    add("3. **Abstention** - `dont_know` or an explicit \"I don't know\".")
    add("4. **Truncation** - `reached_max_tokens`.")
    add("5. **Answer-format / parser failure** - no answer parsed (`prediction` is null/empty) without truncation.")
    add("6. **Wrong - reference present in retrieved triples** - answered and strict-wrong, yet the reference string occurs in a retrieved triple (a selection or composition error).")
    add("7. **Wrong - reference absent from retrieved triples** - answered and strict-wrong, and the reference does not occur in any retrieved triple. This covers both genuine retrieval gaps and *derived* answers (counts, ages, differences, and yes/no comparisons) that are not expected to appear verbatim in the graph, so it is an upper bound on missing evidence.")
    add("")
    add("The reference-in-triples test is a heuristic substring/word-boundary check, "
        "so it can be wrong for very short or numeric answers; it is reported as a "
        "signal, not a ground truth. Boolean (yes/no) references are flagged, since "
        "neither token can appear as a triple. Hallucinated triples are detected with the "
        "same check used by the `utils/logs_viewer.py` viewer: a triple written in "
        "the text that the constrained generator never produced (re-quotes are "
        "discounted; trailing cut-off lookups are excluded via `incomplete_triples` "
        "when available).")
    add("")

    # --- Headline ---
    add("## Headline metrics")
    add("")
    add("| File | N | Accuracy | Macro acc. | F1 | Answered | Don't know | Truncated | No answer | Norm-recovered | Avg triples | Avg tokens |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for res, agg in zip(results, aggregates):
        h = agg["headline"]
        add("| `{name}` | {n} | {acc} | {macc} | {f1} | {ans} | {dk} | {tr} | {na} | {nr} | {tri} | {tok} |".format(
            name=res["path"].name, n=h["n"], acc=h["accuracy"], macc=h["macro_accuracy"],
            f1=f"{h['f1']:.3f}" if h["f1"] is not None else "--", ans=h["answered"], dk=h["dont_know"],
            tr=h["truncated"], na=h["no_answer"], nr=h["normalization_recovered"],
            tri=f"{h['avg_triples']:.1f}" if h["avg_triples"] is not None else "--",
            tok=f"{h['avg_tokens']:.0f}" if h["avg_tokens"] is not None else "--",
        ))
    add("")

    # --- Category distribution ---
    add("## Error categories")
    add("")
    for res, agg in zip(results, aggregates):
        n = agg["n"]
        add(f"### `{res['path'].name}`")
        add("")
        add("| Category | Count | Share |")
        add("|---|---:|---:|")
        for key, label in CATEGORY_ORDER:
            count = agg["categories"].get(key, 0)
            add(f"| {label} | {count} | {_percent(count, n)} |")
        absent = agg["categories"].get("wrong_reference_absent", 0)
        if absent:
            boolean_absent = agg["headline"]["wrong_absent_boolean"]
            add("")
            add(f"> Last-row breakdown: {boolean_absent} yes/no answers (never a "
                f"triple) out of {absent}; the rest mix derived answers with "
                "retrieval gaps.")
        norm_rules = agg["norm_rules"]
        if norm_rules:
            add("")
            add("> Normalization rules matched: "
                + ", ".join(f"`{rule}` ({count})" for rule, count in norm_rules.most_common()) + ".")
        add("")

    # --- Per type ---
    add("## Breakdown by question type")
    add("")
    add("Accuracy is strict correctness; `trunc`/`no ans`/`norm` are shares of that type.")
    add("")
    for res, agg in zip(results, aggregates):
        field = "complexityType" if res["dataset"] == "mintaka" else "type"
        add(f"### `{res['path'].name}` (grouped by `input_sample.{field}`)")
        add("")
        add("| Type | N | Accuracy | Trunc | No ans | Norm | Avg triples |")
        add("|---|---:|---:|---:|---:|---:|---:|")
        for label in sorted(agg["types"]):
            group = agg["types"][label]
            g = len(group)
            add("| {t} | {n} | {acc} | {tr} | {na} | {nm} | {tri} |".format(
                t=label, n=g,
                acc=_percent(sum(x["correct"] for x in group), g),
                tr=_percent(sum(x["truncated"] for x in group), g),
                na=_percent(sum(1 for x in group if x["category"] == "no_answer"), g),
                nm=_percent(sum(1 for x in group if x["norm_match"] and not x["correct"]), g),
                tri=f"{_mean(x['n_triples'] for x in group):.1f}",
            ))
        add("")

    # --- Retrieval volume ---
    add("## Retrieval volume vs. accuracy")
    add("")
    for res, agg in zip(results, aggregates):
        add(f"### `{res['path'].name}`")
        add("")
        add("| Retrieved triples | N | Accuracy |")
        add("|---|---:|---:|")
        for label in TRIPLE_BUCKET_LABELS:
            count = agg["buckets"].get(label, 0)
            if count == 0:
                continue
            add(f"| {label} | {count} | {_percent(agg['bucket_correct'].get(label, 0), count)} |")
        add("")

    # --- Tool usage ---
    add("## Tool usage and generation")
    add("")
    add("| File | Fact tool | Count tool | Sentinel | Incomplete calls | Records w/ hallucination | Hallucinated triples | Median tokens |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|")
    for res, agg in zip(results, aggregates):
        h = agg["headline"]
        add("| `{name}` | {fact} | {count} | {sent} | {inc} | {hr} | {ht} | {med} |".format(
            name=res["path"].name,
            fact=_percent(sum(f["n_fact_calls"] > 0 for f in res["features"]), agg["n"]),
            count=h["count_tool"], sent=h["sentinel"], inc=h["incomplete_calls"],
            hr=f"{h['hallucination_records']} ({_percent(h['hallucination_records'], agg['n'])})",
            ht=h["hallucinated_triples"],
            med=f"{h['median_tokens']:.0f}" if h["median_tokens"] is not None else "--",
        ))
    add("")

    # --- Truncation ---
    add("## Truncation depth")
    add("")
    for res, agg in zip(results, aggregates):
        trunc = [f for f in res["features"] if f["truncated"]]
        if not trunc:
            continue
        answered = sum(f["answered"] for f in trunc)
        add(f"### `{res['path'].name}`")
        add("")
        add(f"- {len(trunc)} records reached the token limit "
            f"({_percent(len(trunc), agg['n'])} of the run); "
            f"{answered} of them still produced a parsed answer.")
        add(f"- Among truncated records, mean retrieved triples is "
            f"{_mean(f['n_triples'] for f in trunc):.1f} and mean tokens "
            f"{_mean(f['n_new_tokens'] for f in trunc):.0f}.")
        top = Counter(f["type"] for f in trunc).most_common(4)
        add(f"- Most affected types: " + ", ".join(f"{t} ({c})" for t, c in top) + ".")
        add("")

    # --- Hallucinations ---
    hallucinated = [f for res in results for f in res["features"] if f["n_hallucinated"] > 0]
    if hallucinated:
        add("## Hallucinated triples")
        add("")
        add(f"{len(hallucinated)} records write at least one triple that the "
            "constrained generator never produced. Examples:")
        add("")
        for res in results:
            shown = 0
            for f in res["features"]:
                if f["n_hallucinated"] > 0 and shown < MAX_EXAMPLES:
                    add(f"- `{res['path'].name}` [{f['type']}] {str(f['question'])[:110]} "
                        f"- {f['n_hallucinated']} hallucinated triple(s)")
                    shown += 1
            if shown:
                add("")

    # --- Worked examples ---
    add("## Worked examples per category")
    add("")
    for key, label in CATEGORY_ORDER:
        if key == "correct":
            continue
        shown_any = False
        for res in results:
            examples = [f for f in res["features"] if f["category"] == key][:MAX_EXAMPLES]
            if not examples:
                continue
            if not shown_any:
                add(f"### {label}")
                add("")
                shown_any = True
            add(f"_From `{res['path'].name}`:_")
            add("")
            for f in examples:
                add(_example(f))
            add("")

    add("## Interpretation")
    add("")
    add("The categories above split the teacher's errors into the mechanically "
        "detectable part (truncation and unparsed answers) and the semantic part "
        "(retrieval gaps, grounding/composition errors and normalization "
        "mismatches). Because these runs are on training splits, the split matters "
        "for the paper's data pipeline:")
    add("")
    add("- **Truncation** removes otherwise-supervisable traces from the SFT pool; "
        "if it concentrates in count/ordinal/superlative questions it biases the "
        "mixture exactly where the new behaviours are weakest.")
    add("- **Wrong answers with the reference absent from the triples** mix genuine "
        "retrieval gaps with derived answers (counts, ages, differences, boolean "
        "comparisons); the per-type breakdown separates the two, and traces whose "
        "supporting facts were never retrieved should be filtered rather than "
        "imitated.")
    add("- **Wrong answers whose reference is present in the triples** are the most "
        "valuable negatives: the fact was retrieved and still mis-selected or "
        "mis-composed, so they are informative for DPO/RL data selection and for "
        "the multi-hop analysis.")
    add("- **Normalization mismatches** are pipeline bugs rather than model errors; "
        "they inflate the error rate and should be fixed in evaluation before the "
        "numbers are reported.")
    add("")
    add("See `ERROR_CATEGORIZATION_GUIDE.md` for the manual, evidence-based "
        "procedure to confirm individual cases and `error_categorization_report.md` "
        "for the interpretation of the held-out test runs.")
    add("")
    return "\n".join(out) + "\n"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs", nargs="+", type=Path, help="JSONL evaluation logs (optionally .gz)")
    parser.add_argument("--output", type=Path, help="write the Markdown report here")
    parser.add_argument("--json", type=Path, help="also write a machine-readable summary here")
    parser.add_argument("--dataset", help="override the dataset label for every log")
    args = parser.parse_args(argv)

    results = [scan(path, args.dataset) for path in args.logs]
    aggregates = [aggregate(res) for res in results]

    generated = datetime.datetime.now().strftime("%Y-%m-%d")
    report = render(results, aggregates, generated)
    if args.output:
        args.output.write_text(report, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(report, end="")

    if args.json:
        payload = {res["path"].name: {"dataset": res["dataset"],
                                      "headline": agg["headline"],
                                      "categories": dict(agg["categories"]),
                                      "types": {t: len(v) for t, v in agg["types"].items()}}
                   for res, agg in zip(results, aggregates)}
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
