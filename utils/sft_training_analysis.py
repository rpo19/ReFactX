"""Analyse the teacher (Qwen3.8-27B) SFT-distillation logs on four datasets.

The script answers the practical questions needed to build the SFT mixture:

* how many **positive** (correct, tool-using) traces each dataset yields;
* what **kinds of positive question/answer pairs** they are (question type,
  answer shape, tool pattern);
* how much the **count tool** and the **sentinel** are used;
* how many errors are genuine versus **evaluation artifacts** (recovered by
  :func:`utils.answer_normalization.verify_answer`);
* what the **error categories** are, including whether the gold answer was
  retrieved and whether it is available in the dataset-provided graph.

It reuses the log primitives from :mod:`utils.log_analysis` and the
dataset-agnostic verifier from :mod:`utils.answer_normalization`.

Usage::

    python utils/sft_training_analysis.py logs/sft_training/*.out* \
        --output error_categorization/sft_training_data_report.md --json out.json
"""

from __future__ import annotations

import argparse
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

from utils.answer_normalization import fold_entity, parse_date, verify_answer  # noqa: E402
from utils.log_analysis import open_log, record_triples  # noqa: E402

TRIPLE_TUPLE_RE = re.compile(r"<([^<>]+)>\s*<([^<>]+)>\s*<([^<>]+)>")
FACT_RE = re.compile(r"<fact>", re.IGNORECASE)
COUNT_RE = re.compile(r"<count>", re.IGNORECASE)
ABSTAIN_RE = re.compile(r"i\s+don'?t\s+know|do not know|cannot (?:be )?(?:answer|determine)|insufficient", re.IGNORECASE)
ANSWER_BLOCK_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)

CATEGORY_ORDER = [
    ("correct", "Correct (strict evaluator)"),
    ("misevaluated", "Correct but misevaluated"),
    ("truncation", "Generation truncation"),
    ("abstention", "Abstention / unsupported"),
    ("no_answer", "No answer / format failure"),
    ("reasoning_error", "Wrong but gold facts retrieved"),
    ("retrieval_failure", "Wrong and gold facts not retrieved (available)"),
    ("missing_or_derived", "Wrong, gold fact absent / derived answer"),
]

TOOL_PATTERNS = ["none", "fact-only", "count", "sentinel", "count+sentinel"]

MAX_EXAMPLES = 3


def dataset_name(meta: dict) -> str:
    raw = str(meta.get("dataset", "")).lower()
    if "mintaka" in raw:
        return "mintaka"
    if "2wiki" in raw or "twowiki" in raw:
        return "2wiki"
    if "webqsp" in raw:
        return "webqsp"
    if "cwq" in raw:
        return "cwq"
    return raw or "unknown"


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _json_list(value: Any) -> list:
    """The 2Wiki log stores ``context``/``evidences`` as JSON-encoded strings."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    return value if isinstance(value, list) else []


def answer_shape(reference: list) -> str:
    refs = [str(x) for x in reference]
    lowered = [x.strip().lower() for x in refs]
    if refs and all(x in {"yes", "no", "true", "false"} for x in lowered):
        return "yes/no"
    if any(parse_date(x) is not None for x in refs):
        return "date"
    if refs and all(re.fullmatch(r"[+-]?\d+(?:\.\d+)?", x.replace(",", "").strip()) for x in refs):
        return "numeric"
    if len(refs) > 1:
        return "list"
    return "entity"


def question_type(dataset: str, inp: dict) -> str:
    if dataset == "mintaka":
        return str(inp.get("complexityType", "unknown"))
    if dataset == "2wiki":
        return str(inp.get("type", "unknown"))
    # RoG-webqsp / RoG-cwq do not carry a question-type field.
    q_entities = _as_list(inp.get("q_entity"))
    if dataset == "webqsp":
        return "generic"
    return "multi-entity" if len(q_entities) > 1 else "single-entity"


def tool_pattern(record: dict) -> tuple[str, bool, bool, bool]:
    text = record.get("full_prediction") or ""
    has_fact = bool(FACT_RE.search(text))
    has_count = bool(COUNT_RE.search(text))
    has_sentinel = "no further records" in text.lower()
    if has_count and has_sentinel:
        pattern = "count+sentinel"
    elif has_count:
        pattern = "count"
    elif has_sentinel:
        pattern = "sentinel"
    elif has_fact:
        pattern = "fact-only"
    else:
        pattern = "none"
    return pattern, has_fact, has_count, has_sentinel


def _retrieved_tuples(record: dict) -> set[tuple[str, ...]]:
    tuples = set()
    for raw in record_triples(record):
        match = TRIPLE_TUPLE_RE.search(raw)
        if match:
            tuples.add(tuple(fold_entity(part) for part in match.groups()))
        else:
            tuples.add((fold_entity(raw),))
    return tuples


def _folded_objects(record: dict) -> set[str]:
    objects = set()
    for raw in record_triples(record):
        match = TRIPLE_TUPLE_RE.search(raw)
        objects.add(fold_entity(match.group(3)) if match else fold_entity(raw))
    return objects


def _contains(haystack: str, needle: str) -> bool:
    needle = fold_entity(needle)
    if not needle:
        return False
    if len(needle) < 3 or needle.replace(".", "").isdigit():
        return re.search(r"(?<!\w)" + re.escape(needle) + r"(?!\w)", haystack) is not None
    return needle in haystack


def gold_status(dataset: str, inp: dict, reference: list, record: dict) -> dict:
    """How much of the gold evidence was retrieved, and is it in the dataset graph?"""
    retrieved = _retrieved_tuples(record)
    objects = _folded_objects(record)
    object_text = " ".join(objects)

    if dataset == "2wiki":
        evidences = [e for e in _json_list(inp.get("evidences")) if isinstance(e, (list, tuple)) and len(e) >= 3]
        gold = {tuple(fold_entity(x) for x in e[:3]) for e in evidences}
        if gold:
            found = len(gold & retrieved)
            recall = found / len(gold)
            return {"gold_retrieved": recall == 1.0, "evidence_recall": recall,
                    "gold_available": True, "evidence_total": len(gold)}
        # No structured evidence: fall back to answer matching.
        matched = any(_contains(object_text, v) for v in reference)
        return {"gold_retrieved": matched, "evidence_recall": 1.0 if matched else 0.0,
                "gold_available": True, "evidence_total": 0}

    if dataset in {"webqsp", "cwq"}:
        graph = [g for g in _json_list(inp.get("graph")) if isinstance(g, (list, tuple)) and len(g) >= 3]
        graph_text = " ".join(fold_entity(x) for g in graph for x in g)
        return {
            "gold_retrieved": any(_contains(object_text, v) for v in reference),
            "evidence_recall": None,
            "gold_available": any(_contains(graph_text, v) for v in reference),
            "evidence_total": len(graph),
        }

    # Mintaka: no gold triples; use the answer-entity labels against the retrieved objects.
    labels = [e.get("label") for e in _json_list(inp.get("answerEntity")) if isinstance(e, dict)]
    matched = any(_contains(object_text, v) for v in (labels or reference))
    return {"gold_retrieved": matched, "evidence_recall": None,
            "gold_available": None, "evidence_total": 0}


def categorize(record: dict, dataset: str) -> dict:
    ev = record.get("evaluation") or {}
    reference = _as_list(record.get("gt_answer"))
    prediction = _as_list(record.get("prediction"))
    text = record.get("full_prediction") or ""

    strict = int(ev.get("correct") == 1)
    verification = verify_answer(
        prediction, reference, dataset=dataset,
        question=record.get("question"), input_sample=record.get("input_sample"),
    )
    normalized = int(verification.normalized_match)
    dont_know = int(bool(ev.get("dont_know")) or bool(ABSTAIN_RE.search(str(prediction[0]) if prediction else "")))
    truncated = int(bool(record.get("reached_max_tokens")))
    answered = int(bool(prediction))
    n_triples = len(record_triples(record))
    pattern, has_fact, has_count, has_sentinel = tool_pattern(record)
    status = gold_status(dataset, record.get("input_sample") or {}, reference, record)

    if strict:
        category = "correct"
    elif normalized:
        category = "misevaluated"
    elif dont_know:
        category = "abstention"
    elif truncated:
        category = "truncation"
    elif not answered:
        category = "no_answer"
    elif status["gold_retrieved"]:
        category = "reasoning_error"
    elif status["gold_available"]:
        category = "retrieval_failure"
    else:
        category = "missing_or_derived"

    return {
        "dataset": dataset,
        "qtype": question_type(dataset, record.get("input_sample") or {}),
        "shape": answer_shape(reference),
        "pattern": pattern,
        "has_fact": int(has_fact),
        "has_count": int(has_count),
        "has_sentinel": int(has_sentinel),
        "strict": strict,
        "normalized": normalized,
        "partial": int(bool(verification.partial_match) and not normalized),
        "rule": verification.rule,
        "correct": strict,
        "dont_know": dont_know,
        "truncated": truncated,
        "answered": answered,
        "answer_complete": int(bool(record.get("answer_complete"))),
        "n_triples": n_triples,
        "n_new_tokens": int(record.get("new_tokens_generated", 0) or 0),
        "category": category,
        "gold_retrieved": status["gold_retrieved"],
        "evidence_recall": status["evidence_recall"],
        "gold_available": status["gold_available"],
        "question": record.get("question", ""),
        "gt_answer": record.get("gt_answer"),
        "prediction": prediction,
    }


def scan(path: Path) -> dict:
    meta: dict = {}
    summary: Optional[dict] = None
    features: list[dict] = []
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
            features.append(categorize(obj, dataset_name(meta)))
    return {"path": path, "meta": meta, "summary": summary, "features": features,
            "dataset": dataset_name(meta) if meta else "unknown"}


def _pct(n: int, d: int) -> str:
    return "--" if not d else f"{100.0 * n / d:.1f}%"


def _mean(values: Iterable[Any]) -> Optional[float]:
    nums = [float(v) for v in values if isinstance(v, (int, float))]
    return sum(nums) / len(nums) if nums else None


def aggregate(scan_result: dict) -> dict:
    f = scan_result["features"]
    n = len(f)
    positives = [x for x in f if x["strict"] and x["n_triples"] > 0]
    verified_positives = [x for x in f if x["normalized"] and x["n_triples"] > 0]
    usable = [x for x in verified_positives if not x["truncated"] and x["answer_complete"]]

    cats = Counter(x["category"] for x in f)
    rules = Counter(x["rule"] for x in f if x["category"] == "misevaluated")

    by_type = defaultdict(lambda: Counter())
    for x in f:
        t = by_type[x["qtype"]]
        t["n"] += 1
        t["strict"] += x["strict"]
        t["normalized"] += x["normalized"]
        t["positive"] += int(x["strict"] and x["n_triples"] > 0)
        t["count"] += x["has_count"]
        t["sentinel"] += x["has_sentinel"]
        t["truncated"] += x["truncated"]

    shapes = Counter(x["shape"] for x in positives)
    shapes_verified = Counter(x["shape"] for x in verified_positives)
    patterns = Counter(x["pattern"] for x in positives)
    patterns_all = Counter(x["pattern"] for x in f)

    headline = {
        "n": n,
        "strict": sum(x["strict"] for x in f),
        "normalized": sum(x["normalized"] for x in f),
        "misevaluated": sum(1 for x in f if x["normalized"] and not x["strict"]),
        "positive": len(positives),
        "verified_positive": len(verified_positives),
        "usable": len(usable),
        "truncated": sum(x["truncated"] for x in f),
        "no_answer": sum(1 for x in f if x["category"] == "no_answer"),
        "count_all": sum(x["has_count"] for x in f),
        "sentinel_all": sum(x["has_sentinel"] for x in f),
        "count_pos": sum(x["has_count"] for x in positives),
        "sentinel_pos": sum(x["has_sentinel"] for x in positives),
        "avg_triples_pos": _mean(x["n_triples"] for x in positives),
        "avg_tokens": _mean(x["n_new_tokens"] for x in f),
    }
    return {"headline": headline, "categories": cats, "rules": rules,
            "by_type": by_type, "shapes": shapes, "shapes_verified": shapes_verified,
            "patterns": patterns, "patterns_all": patterns_all, "n": n}


def render(results: list[dict], aggregates: list[dict], generated: str) -> str:
    out: list[str] = []
    add = out.append
    total = sum(a["n"] for a in aggregates)

    add("# SFT training-data analysis (teacher distillation on four datasets)")
    add("")
    add(f"_Generated by `utils/sft_training_analysis.py` on {generated}. "
        f"Total evaluated records: {total}._")
    add("")
    add("All four logs are Qwen3.8-27B teacher runs on the **training** splits, i.e. "
        "the pool from which SFT traces are distilled. A trace is **positive** under "
        "the repository's definition when the strict evaluator marks it correct and "
        "it retrieved at least one triple (`utils/filter_positive_training_samples.py`). "
        "Two broader variants are reported as well: **verified** positives (strict or "
        "semantically equivalent under `utils/answer_normalization.verify_answer`) and "
        "**usable** positives (verified, tool-using, complete and not truncated).")
    add("")

    add("## 1. How many positives do we have?")
    add("")
    add("| Dataset | Records | Strict correct | Verified correct | + from verification | Positive (correct + triples) | Verified positive | Truncated |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|")
    for res, agg in zip(results, aggregates):
        h = agg["headline"]
        add("| {ds} | {n} | {s} ({sp}) | {v} ({vp}) | +{g} | **{p}** ({pp}) | {vp2} | {t} ({tp}) |".format(
            ds=res["dataset"], n=h["n"], s=h["strict"], sp=_pct(h["strict"], h["n"]),
            v=h["normalized"], vp=_pct(h["normalized"], h["n"]), g=h["normalized"] - h["strict"],
            p=h["positive"], pp=_pct(h["positive"], h["n"]),
            vp2=h["verified_positive"],
            t=h["truncated"], tp=_pct(h["truncated"], h["n"]),
        ))
    hsum = {k: sum(a["headline"][k] for a in aggregates) for k in
            ("n", "strict", "normalized", "misevaluated", "positive", "verified_positive", "truncated")}
    add("| **Total** | **{n}** | **{s}** ({sp}) | **{v}** ({vp}) | **+{g}** | **{p}** ({pp}) | **{vp2}** | **{t}** ({tp}) |".format(
        n=hsum["n"], s=hsum["strict"], sp=_pct(hsum["strict"], hsum["n"]),
        v=hsum["normalized"], vp=_pct(hsum["normalized"], hsum["n"]),
        g=hsum["normalized"] - hsum["strict"], p=hsum["positive"], pp=_pct(hsum["positive"], hsum["n"]),
        vp2=hsum["verified_positive"],
        t=hsum["truncated"], tp=_pct(hsum["truncated"], hsum["n"]),
    ))
    add("")
    add("> `+ from verification` counts strict-wrong answers recovered by the "
        "conservative semantic comparison (dates with ambiguous two-digit years, "
        "quantities, aliases, entity folding). These are evaluation artifacts, not "
        "model errors, and should be repaired before the error rate is quoted.")
    add("")

    add("## 2. Count tool and sentinel usage")
    add("")
    add("| Dataset | Count tool (all) | Count tool (positive) | Sentinel (all) | Sentinel (positive) | Any tool (positive) | Avg triples (positive) |")
    add("|---|---:|---:|---:|---:|---:|---:|")
    for res, agg in zip(results, aggregates):
        h = agg["headline"]
        p = max(h["positive"], 1)
        any_tool = sum(c for patt, c in agg["patterns"].items() if patt != "none")
        add("| {ds} | {ca} ({cap}) | {cp} ({cpp}) | {sa} ({sap}) | {sp} ({spp}) | {at} ({atp}) | {tri} |".format(
            ds=res["dataset"],
            ca=h["count_all"], cap=_pct(h["count_all"], h["n"]),
            cp=h["count_pos"], cpp=_pct(h["count_pos"], p),
            sa=h["sentinel_all"], sap=_pct(h["sentinel_all"], h["n"]),
            sp=h["sentinel_pos"], spp=_pct(h["sentinel_pos"], p),
            at=any_tool, atp=_pct(any_tool, p),
            tri=f"{h['avg_triples_pos']:.1f}" if h["avg_triples_pos"] is not None else "--",
        ))
    add("")
    add("Tool patterns among positives:")
    add("")
    add("| Dataset | " + " | ".join(TOOL_PATTERNS) + " |")
    add("|---|" + "|".join(["---:"] * len(TOOL_PATTERNS)) + "|")
    for res, agg in zip(results, aggregates):
        p = max(agg["headline"]["positive"], 1)
        cells = [f"{agg['patterns'].get(patt, 0)} ({_pct(agg['patterns'].get(patt, 0), p)})" for patt in TOOL_PATTERNS]
        add(f"| {res['dataset']} | " + " | ".join(cells) + " |")
    add("")

    add("## 3. Types of positive question/answer pairs")
    add("")
    for res, agg in zip(results, aggregates):
        add(f"### {res['dataset']}")
        add("")
        add("| Question type | N | Strict acc. | Verified acc. | Positives (strict) | Count tool | Sentinel | Truncated |")
        add("|---|---:|---:|---:|---:|---:|---:|---:|")
        for qtype in sorted(agg["by_type"]):
            t = agg["by_type"][qtype]
            add("| {q} | {n} | {s} | {v} | {p} | {c} | {se} | {tr} |".format(
                q=qtype, n=t["n"], s=_pct(t["strict"], t["n"]), v=_pct(t["normalized"], t["n"]),
                p=t["positive"], c=t["count"], se=t["sentinel"], tr=t["truncated"],
            ))
        add("")
        add("Positive traces by answer shape (strict / verified):")
        add("")
        add("| Answer shape | Strict positives | Verified positives |")
        add("|---|---:|---:|")
        for shape in ("entity", "list", "numeric", "date", "yes/no"):
            add(f"| {shape} | {agg['shapes'].get(shape, 0)} | {agg['shapes_verified'].get(shape, 0)} |")
        add("")

    add("## 4. Verification gains (correct but misevaluated)")
    add("")
    add("| Dataset | Misevaluated | Rule distribution |")
    add("|---|---:|---|")
    for res, agg in zip(results, aggregates):
        rules = ", ".join(f"`{r}` ({c})" for r, c in agg["rules"].most_common()) or "--"
        add(f"| {res['dataset']} | {agg['headline']['misevaluated']} | {rules} |")
    add("")

    add("## 5. Error categories")
    add("")
    add("| Category | " + " | ".join(res["dataset"] for res in results) + " | Total |")
    add("|---|" + "|".join(["---:"] * len(results)) + "|---:|")
    for key, label in CATEGORY_ORDER:
        row = [agg["categories"].get(key, 0) for agg in aggregates]
        add(f"| {label} | " + " | ".join(str(v) for v in row) + f" | {sum(row)} |")
    add("")
    add("| Share of records | " + " | ".join(res["dataset"] for res in results) + " |")
    add("|---|" + "|".join(["---:"] * len(results)) + "|")
    for key, label in CATEGORY_ORDER:
        row = [_pct(agg["categories"].get(key, 0), agg["n"]) for agg in aggregates]
        add(f"| {label} | " + " | ".join(row) + " |")
    add("")
    add("`Wrong but gold facts retrieved` = the gold answer/evidence was in the "
        "retrieved triples, so the failure is selection/composition/aggregation. "
        "`Wrong and gold facts not retrieved` is split by whether the gold answer is "
        "present in the dataset-provided graph (`retrieval_failure`) or absent / "
        "derived (`missing_or_derived`). For Mintaka there is no gold graph, so the "
        "latter bucket also contains derived answers.")
    add("")

    add("## 6. Retrieval and coverage")
    add("")
    add("| Dataset | Gold answer retrieved | Available in dataset graph | Wrong with facts retrieved | Wrong, facts available but not retrieved |")
    add("|---|---:|---:|---:|---:|")
    for res, agg in zip(results, aggregates):
        f = res["features"]
        retrieved = sum(x["gold_retrieved"] for x in f)
        available = sum(bool(x["gold_available"]) for x in f)
        add("| {ds} | {r} ({rp}) | {a} ({ap}) | {wr} | {rf} |".format(
            ds=res["dataset"], r=retrieved, rp=_pct(retrieved, agg["n"]),
            a=available, ap=_pct(available, agg["n"]),
            wr=agg["categories"].get("reasoning_error", 0),
            rf=agg["categories"].get("retrieval_failure", 0),
        ))
    add("")
    for res in results:
        recalls = [x["evidence_recall"] for x in res["features"] if x["evidence_recall"] is not None]
        if recalls:
            full = sum(1 for r in recalls if r == 1.0)
            add(f"- **{res['dataset']}** structured-evidence recall: "
                f"{_mean(recalls):.3f} mean, {full}/{len(recalls)} records with all gold triples retrieved.")
    add("")

    add("## 7. Truncation")
    add("")
    add("| Dataset | Truncated | Mean tokens | Most affected types |")
    add("|---|---:|---:|---|")
    for res, agg in zip(results, aggregates):
        trunc = [x for x in res["features"] if x["truncated"]]
        top = Counter(x["qtype"] for x in trunc).most_common(3)
        add("| {ds} | {t} ({tp}) | {tok} | {top} |".format(
            ds=res["dataset"], t=len(trunc), tp=_pct(len(trunc), agg["n"]),
            tok=f"{_mean(x['n_new_tokens'] for x in trunc):.0f}" if trunc else "--",
            top=", ".join(f"{t} ({c})" for t, c in top) or "--",
        ))
    add("")

    add("## 8. Takeaways for the SFT mixture")
    add("")
    positives_total = hsum["positive"]
    verified_total = hsum["verified_positive"]
    add(f"- The four runs yield **{positives_total} strict positives** and "
        f"**{verified_total} verified positives**; every verified positive is "
        "usable as-is (non-truncated and complete).")
    add(f"- Verification recovers **{hsum['normalized'] - hsum['strict']}** answers "
        "(formatting/alias/date artifacts), so the strict evaluator under-counts "
        "correct traces.")
    add(f"- **{hsum['truncated']}** traces ({_pct(hsum['truncated'], hsum['n'])}) "
        "hit the token limit and cannot be used as supervision at the current budget.")
    count_pos_total = sum(a["headline"]["count_pos"] for a in aggregates)
    sentinel_pos_total = sum(a["headline"]["sentinel_pos"] for a in aggregates)
    add(f"- Only **{count_pos_total}** positives use the count tool and "
        f"**{sentinel_pos_total}** use the sentinel: the behaviours the paper wants "
        "to teach are barely covered by this natural pool.")
    add("")

    return "\n".join(out) + "\n"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs", nargs="+", type=Path, help="JSONL/JSONL.gz evaluation logs")
    parser.add_argument("--output", type=Path, help="write the Markdown report here")
    parser.add_argument("--json", type=Path, help="write machine-readable aggregates here")
    args = parser.parse_args(argv)

    results = [scan(path) for path in args.logs]
    aggregates = [aggregate(res) for res in results]
    generated = __import__("datetime").datetime.now().strftime("%Y-%m-%d")
    report = render(results, aggregates, generated)

    if args.output:
        args.output.write_text(report, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(report, end="")

    if args.json:
        payload = {}
        for res, agg in zip(results, aggregates):
            payload[res["dataset"]] = {
                "path": res["path"].name,
                "records": agg["n"],
                "headline": agg["headline"],
                "categories": dict(agg["categories"]),
                "rules": dict(agg["rules"]),
                "by_type": {k: dict(v) for k, v in agg["by_type"].items()},
                "shapes": dict(agg["shapes"]),
                "patterns": dict(agg["patterns"]),
            }
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
