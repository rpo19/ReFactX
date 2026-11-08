"""Build the SFT training set from natural positives + synthetic count traces.

Selection rules:

* use every natural positive from Mintaka, WebQSP and CWQ;
* keep at most ``--two-wiki-cap`` (default 2000) 2Wiki positives;
* add the synthetic count traces;
* drop ``unsupported`` traces (correct but not supported by the retrieved
  facts) unless ``--include-bad``; this is the only use of the support labels;
* legacy ``best``/``acceptable`` logs still honour ``--best-ratio`` /
  ``--min-acceptable`` when a cap applies; support-labelled logs have no
  ``acceptable`` class, so a cap keeps all ``supported`` traces;
* deduplicate by question (synthetic count traces win over a duplicate natural
  trace);
* stratify across dataset, question type and answer shape.

The output is a JSONL file consumable by ``utils/sft_train.py`` (it has
``prompt`` and ``full_prediction``), preceded by a ``_report`` line.

Example::

    python utils/build_sft_dataset.py \
        --natural logs/behaviors/*.behaviors \
        --synthetic logs/synthetic/count_traces.jsonl \
        --output logs/sft_training_mix.jsonl \
        --two-wiki-cap 2000
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils.behaviors import (  # noqa: E402
    answer_shape,
    dataset_name,
    is_positive,
    question_type,
)
from utils.log_io import open_log  # noqa: E402

BEHAVIOR_PRIORITY = {
    "best": 3, "supported": 3, "acceptable": 2,
    "bad": 1, "unsupported": 0, "error": 0,
}


def normalize_question(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def iter_log(path: Path, *, min_confidence: float = 0.0) -> tuple[dict, list[dict]]:
    """Yield (header, records) from an eval or behaviour log."""
    with open_log(path) as fd:
        header = None
        records = []
        for lineno, line in enumerate(fd):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if lineno == 0 or (header is None and "evaluation" not in obj):
                header = obj
                continue
            if "macro_precision" in obj or "evaluation" not in obj:
                continue
            behavior = obj.get("behavior") or {}
            label = behavior.get("label")
            confidence = behavior.get("confidence")
            if label not in BEHAVIOR_PRIORITY:
                label = None
            if label and min_confidence > 0 and confidence is not None and confidence < min_confidence:
                label = None
            obj["_behavior"] = label
            obj["_confidence"] = confidence
            records.append(obj)
    return header or {}, records


def candidate(record: dict, dataset: str, source: str, provenance: str) -> dict:
    behavior = record.get("_behavior") or record.get("behavior", {}).get("label")
    confidence = record.get("_confidence")
    if confidence is None:
        confidence = (record.get("behavior") or {}).get("confidence")
    return {
        "record": record,
        "dataset": dataset,
        "qtype": question_type(dataset, record.get("input_sample") or {}),
        "shape": answer_shape(record.get("gt_answer")),
        "behavior": behavior,
        "confidence": confidence,
        "source": source,
        "provenance": provenance,
        "key": normalize_question(record.get("question")),
    }


def group_by_dataset(cands: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for c in cands:
        groups[c["dataset"]].append(c)
    return groups


def group_by_type(cands: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for c in cands:
        groups[c["qtype"]].append(c)
    return groups


def allocate(groups: dict[str, list[dict]], total: int) -> dict[str, int]:
    """Allocate ``total`` slots across groups proportional to their size (largest remainder)."""
    sizes = {k: len(v) for k, v in groups.items()}
    available = sum(sizes.values())
    if total >= available:
        return dict(sizes)
    raw = {k: sizes[k] * total / available for k in sizes}
    quotas = {k: int(math.floor(raw[k])) for k in sizes}
    remaining = total - sum(quotas.values())
    for key in sorted(sizes, key=lambda k: raw[k] - quotas[k], reverse=True):
        if remaining <= 0:
            break
        if quotas[key] < sizes[key]:
            quotas[key] += 1
            remaining -= 1
    # Clamp and redistribute any excess (rare, when a group is tiny).
    for key in sizes:
        quotas[key] = min(quotas[key], sizes[key])
    return quotas


def select_group(group: list[dict], n_target: Optional[int], *, best_ratio: float,
                 min_acceptable: float, include_bad: bool, keep_error: bool,
                 rng: random.Random) -> list[dict]:
    """Select ``n_target`` items, prioritising ``best`` over ``acceptable``."""
    if not group:
        return []
    eligible = [c for c in group if c["behavior"] not in {"bad", "unsupported"} or include_bad]
    eligible = [c for c in eligible if c["behavior"] != "error" or keep_error]
    if not eligible:
        return []
    if n_target is None or n_target >= len(eligible):
        return list(eligible)

    best = [c for c in eligible if c["behavior"] in {"best", "supported"}]
    acceptable = [c for c in eligible if c["behavior"] == "acceptable"]
    other = [c for c in eligible if c["behavior"] not in {"best", "supported", "acceptable"}]

    if not best and not acceptable:
        return rng.sample(other, min(n_target, len(other)))

    total = min(n_target, len(eligible))
    floor = int(round(min_acceptable * total))
    n_acc = min(len(acceptable), max(int(round((1 - best_ratio) * total)), floor))
    n_best = min(len(best), total - n_acc)
    n_acc = min(len(acceptable), total - n_best)

    chosen = rng.sample(best, n_best) + rng.sample(acceptable, n_acc)
    leftover = total - len(chosen)
    if leftover > 0 and other:
        chosen += rng.sample(other, min(leftover, len(other)))
    rng.shuffle(chosen)
    return chosen


def cap_candidates(cands: list[dict], cap: Optional[int], **kwargs) -> list[dict]:
    if cap is None or cap >= len(cands):
        rng = kwargs["rng"]
        return select_group(cands, None, **kwargs)
    groups = group_by_type(cands)
    quotas = allocate(groups, cap)
    rng = kwargs["rng"]
    selected: list[dict] = []
    for qtype, group in groups.items():
        selected.extend(select_group(group, quotas.get(qtype, 0), **kwargs))
    return selected


def enforce_ratio(cands: list[dict], best_ratio: float, min_acceptable: float,
                  rng: random.Random) -> list[dict]:
    """Downsample ``acceptable`` so best:acceptable approaches ``best_ratio``."""
    best = [c for c in cands if c["behavior"] in {"best", "supported"}]
    acceptable = [c for c in cands if c["behavior"] == "acceptable"]
    others = [c for c in cands if c["behavior"] not in {"best", "supported", "acceptable"}]
    if not best or not acceptable:
        return cands
    target_acc = int(round(len(best) * (1 - best_ratio) / max(best_ratio, 1e-9)))
    floor = int(round(min_acceptable * (len(best) + target_acc)))
    keep = min(len(acceptable), max(target_acc, floor))
    if keep >= len(acceptable):
        return cands
    return best + rng.sample(acceptable, keep) + others


def to_output(c: dict) -> dict:
    record = c["record"]
    behavior = c.get("behavior")
    output = {
        "question": record.get("question"),
        "gt_answer": record.get("gt_answer"),
        "prediction": record.get("prediction"),
        "full_prediction": record.get("full_prediction"),
        "prompt": record.get("prompt"),
        "triples": record.get("triples") or [],
        "dataset": c["dataset"],
        "dataset_split": "train",
        "evaluation": record.get("evaluation") or {},
        "complexityType": (record.get("input_sample") or {}).get("complexityType"),
        "question_type": c["qtype"],
        "answer_shape": c["shape"],
        "behavior": behavior,
        "behavior_confidence": c.get("confidence"),
        "provenance": c["provenance"],
        "source_log": c["source"],
        "synthetic": bool(record.get("synthetic")),
    }
    # Behaviour-specific synthetic metadata (e.g. backtracking): keep it so the
    # mixture can be audited and the DPO pairs reused downstream.
    for key in ("behaviour_tags", "verification", "generator", "backtracking", "dpo_pair"):
        if record.get(key) is not None:
            output[key] = record[key]
    return output


def summarize(cands: list[dict]) -> dict:
    def tally(key):
        return dict(Counter(c[key] for c in cands))

    labels = Counter(c["behavior"] for c in cands)
    best = labels.get("best", 0) + labels.get("supported", 0)
    acceptable = labels.get("acceptable", 0)
    ratio = (best / (best + acceptable)) if (best + acceptable) else None
    return {
        "total": len(cands),
        "by_dataset": tally("dataset"),
        "by_type": tally("qtype"),
        "by_shape": tally("shape"),
        "by_behavior": {k: v for k, v in labels.items()},
        "by_provenance": tally("provenance"),
        "best_share_of_best_plus_acceptable": ratio,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--natural", nargs="+", type=Path, required=True,
                        help="behaviour-enriched eval logs")
    parser.add_argument("--synthetic", nargs="*", type=Path, default=[],
                        help="synthetic count-trace logs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, help="default: <output>.manifest.json")
    parser.add_argument("--two-wiki-cap", type=int, default=2000)
    parser.add_argument("--max-per-dataset", type=int, default=None)
    parser.add_argument("--max-per-type", type=int, default=None)
    parser.add_argument("--best-ratio", type=float, default=0.75)
    parser.add_argument("--min-acceptable", type=float, default=0.15,
                        help="floor for behaviour 2 among selected best+acceptable")
    parser.add_argument("--include-bad", action="store_true",
                        help="include correct-but-unsupported traces")
    parser.add_argument("--keep-error-labeled", action="store_true",
                        help="include positives the judge labelled error (label inconsistency)")
    parser.add_argument("--require-behavior", action="store_true",
                        help="drop positives without a behaviour label")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--verified", action="store_true",
                        help="use verified (normalized) correctness instead of the strict evaluator")
    parser.add_argument("--enforce-best-ratio", action="store_true",
                        help="downsample acceptable traces globally to hit --best-ratio")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    selection_kwargs = dict(best_ratio=args.best_ratio, min_acceptable=args.min_acceptable,
                            include_bad=args.include_bad, keep_error=args.keep_error_labeled,
                            rng=rng)

    natural: list[dict] = []
    skipped = Counter()
    for path in args.natural:
        header, records = iter_log(path, min_confidence=args.min_confidence)
        dataset = dataset_name(header)
        for record in records:
            if not is_positive(record, verified=args.verified, dataset=dataset):
                skipped["not_positive"] += 1
                continue
            behavior = record.get("_behavior")
            if behavior is None and args.require_behavior:
                skipped["missing_behavior"] += 1
                continue
            if behavior == "bad" and not args.include_bad:
                skipped["bad"] += 1
                continue
            if behavior == "error" and not args.keep_error_labeled:
                skipped["error_labeled"] += 1
                continue
            natural.append(candidate(record, dataset, path.name, "natural"))

    # Per-dataset caps (2Wiki special case).
    by_dataset = group_by_dataset(natural)
    natural = []
    for dataset, group in by_dataset.items():
        cap = args.two_wiki_cap if dataset == "2wiki" else args.max_per_dataset
        natural.extend(cap_candidates(group, cap, **selection_kwargs))

    # Optional per-type cap.
    if args.max_per_type:
        capped: list[dict] = []
        for dataset, group in group_by_dataset(natural).items():
            for qtype, type_group in group_by_type(group).items():
                capped.extend(select_group(type_group, args.max_per_type, **selection_kwargs))
        natural = capped

    if args.enforce_best_ratio:
        natural = enforce_ratio(natural, args.best_ratio, args.min_acceptable, rng)

    # Synthetic count traces are all accepted and are behaviour "best".
    synthetic: list[dict] = []
    for path in args.synthetic:
        header, records = iter_log(path, min_confidence=0.0)
        dataset = dataset_name(header) or "mintaka"
        for record in records:
            if not record.get("synthetic"):
                continue
            record["_behavior"] = "best"
            synthetic.append(candidate(record, dataset, path.name, "synthetic"))

    # Deduplicate by question; synthetic traces win.
    seen: set[str] = set()
    deduped: list[dict] = []
    duplicates = Counter()
    for cand in synthetic + natural:
        if not cand["key"]:
            skipped["empty_question"] += 1
            continue
        if cand["key"] in seen:
            duplicates[cand["provenance"]] += 1
            continue
        seen.add(cand["key"])
        deduped.append(cand)

    rng.shuffle(deduped)
    output = [to_output(c) for c in deduped]

    summary = summarize(deduped)
    report = {
        "_report": True,
        "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "params": {
            "two_wiki_cap": args.two_wiki_cap,
            "max_per_dataset": args.max_per_dataset,
            "max_per_type": args.max_per_type,
            "best_ratio": args.best_ratio,
            "min_acceptable": args.min_acceptable,
            "include_bad": args.include_bad,
            "verified": args.verified,
            "enforce_best_ratio": args.enforce_best_ratio,
            "seed": args.seed,
        },
        "skipped": dict(skipped),
        "duplicates": dict(duplicates),
        "counts": summary,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        out.write(json.dumps(report) + "\n")
        for sample in output:
            out.write(json.dumps(sample) + "\n")

    manifest_path = args.manifest or args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if not args.quiet:
        print(f"wrote {len(output)} samples to {args.output}")
        print(json.dumps(summary, indent=2))
        print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
