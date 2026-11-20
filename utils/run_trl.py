"""Train a causal LM with TRL's GRPO trainer.

The default reward is deterministic and uses the answer column.  Pass
``--judge-model MODEL`` to replace it with an LLM-as-a-judge reward.

Examples (run from the repository root)::

    python utils/run_trl.py --config configs/trl_qwen35_08b_smoke.json
    python utils/run_trl.py --config configs/trl_qwen35_08b_smoke.json --max-steps 1
    python utils/run_trl.py --config configs/trl_qwen35_08b_smoke.json --judge-model Qwen/Qwen2.5-3B-Instruct
    python utils/run_trl.py --config configs/trl_qwen35_08b_smoke.json --adapter ./grpo-qwen35-08b

Metrics are reported to Weights & Biases by default.  Set ``report_to`` in the
config (or pass ``--report-to none``) to disable reporting.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import warnings
from collections import Counter
from typing import Any, Iterable

from dotenv import load_dotenv


# Config values that name filesystem locations. They may use ${WS_PATH} and
# friends so one config works on both cluster workspaces (capella and alpha)
# and locally, where WS_PATH is simply not set.
PATH_FIELDS = (
    "adapter",
    "dataset",
    "eval_dataset",
    "prompt",
    "generation_output",
    "output_dir",
    "resume_from_checkpoint",
)


def _expand_path(value: Any) -> Any:
    """Expand ~ and $VARS in a path-like config value."""
    if not isinstance(value, str):
        return value
    return os.path.expanduser(os.path.expandvars(value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="JSON config file; CLI options override config values")
    parser.add_argument("--model", default=None, help="Base model name or path")
    parser.add_argument("--adapter", default=None, help="Optional LoRA adapter to load on top of --model")
    parser.add_argument("--judge-model", default=None, help="Optional model used as an LLM judge")
    parser.add_argument("--gspo", action="store_true", default=None, help="Use GSPO sequence-level importance sampling")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--prompt", default=None, help="Prompt template JSON/YAML file")
    parser.add_argument("--question-key", default=None)
    parser.add_argument("--answer-key", default=None)
    parser.add_argument("--train-split", default=None)
    parser.add_argument("--eval-split", default=None)
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Maximum number of eval examples; omit or use null in config for all examples",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Number of eval examples generated together",
    )
    parser.add_argument(
        "--custom-eval-steps",
        type=int,
        default=None,
        help="Run and log custom validation metrics every N optimizer steps; 0 disables periodic evaluation",
    )
    parser.add_argument("--index", default=None, help="Optional ReFactX prefix-tree index URL (defaults to INDEX in .env)")
    parser.add_argument("--tablename", default=None, help="PostgreSQL index table name")
    parser.add_argument("--index-tables", nargs="+", default=None,
                        help="KB tables to preload for per-example routing (e.g. qwen36 qwen36fbwikipops)")
    parser.add_argument("--default-index-table", default=None,
                        help="Fallback KB table when a sample has no usable index_table")
    parser.add_argument("--index-table-column", default=None,
                        help="Dataset column naming the KB table (default: index_table)")
    parser.add_argument("--dataset-column", default=None,
                        help="Dataset column carrying the dataset tag (default: dataset)")
    parser.add_argument("--split-column", default=None,
                        help="Dataset column carrying train/validation (default: split)")
    parser.add_argument("--no-cuda", action="store_true", help="Allow running without CUDA (default: require CUDA)")
    parser.add_argument(
        "--attn-implementation",
        default=None,
        help="Transformers attention backend (e.g. sdpa, flash_attention_2); default: auto",
    )
    parser.add_argument(
        "--dataset-num-proc",
        type=int,
        default=None,
        help="Processes used while formatting the training/eval datasets (>1 speeds up setup)",
    )
    parser.add_argument(
        "--use-generation-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable the KV cache during rollouts/eval (default: true unless configured otherwise)",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=None,
        help="Threads for per-row constraint processing on thread-safe in-memory indexes (0/1 disables)",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--generation-output", default=None, help="Append validation generations as JSONL")
    parser.add_argument("--save-steps", type=int, default=None, help="Save a training checkpoint every N steps")
    parser.add_argument("--save-total-limit", type=int, default=None, help="Maximum number of checkpoints to keep")
    parser.add_argument("--resume-from-checkpoint", default=None, help="Checkpoint path to resume training from")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="Stop after this many optimizer steps")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--num-generations", type=int, default=None)
    parser.add_argument("--max-completion-length", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--min-p", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--fact-pattern", default=None)
    parser.add_argument("--answer-pattern", default=None)
    parser.add_argument(
        "--count-pattern",
        default=None,
        help="Marker that triggers the constrained count tool (default: <count>)",
    )
    parser.add_argument(
        "--sentinel",
        action="store_true",
        default=None,
        help="Enable the exhausted-retrieval '<no further records>' sentinel during generation",
    )
    parser.add_argument(
        "--reward-alpha", type=float, default=None,
        help="Gated multiplier paid when a count question uses <count> and is correct",
    )
    parser.add_argument(
        "--reward-beta", type=float, default=None,
        help="Gated multiplier paid when an enumeration question uses the sentinel and is correct",
    )
    parser.add_argument(
        "--fact-soft-cap", type=int, default=None,
        help="Fact calls allowed before the retrieval-loop penalty applies",
    )
    parser.add_argument(
        "--fact-penalty", type=float, default=None,
        help="Penalty per fact call beyond the soft cap",
    )
    parser.add_argument(
        "--max-fact-penalty", type=float, default=None,
        help="Upper bound on the fact-count penalty",
    )
    parser.add_argument(
        "--count-type", default=None,
        help="Question-type value that marks a counting question (default: count)",
    )
    parser.add_argument(
        "--enumeration-datasets", nargs="+", default=None,
        help="List-answer datasets whose multi-item answers gate the sentinel bonus (default: webqsp)",
    )
    parser.add_argument(
        "--no-normalized-reward", action="store_true", default=None,
        help="Disable semantically-normalized answer credit in the reward",
    )
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--report-to", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    load_dotenv()

    config = {}
    if args.config:
        with open(args.config, encoding="utf-8") as config_file:
            config = json.load(config_file)

    # Accept either flat generation keys or the nested format used by the
    # Mintaka generation config.
    generation = config.get("generation_config", {})

    # Config files use the same names as the CLI, except that model_name is
    # also accepted for consistency with the other repository configs.
    defaults = {
        "experiment_name": config.get("experiment_name", "trl"),
        "wandb_project": config.get("wandb_project", "refactx_grpo"),
        "wandb_run_name": config.get("wandb_run_name", config.get("experiment_name")),
        "model": config.get("model", config.get("model_name", "Qwen/Qwen2.5-0.5B-Instruct")),
        "adapter": config.get("adapter"),
        "judge_model": config.get("judge_model"),
        "gspo": config.get("gspo", False),
        # TRL's GRPO loss formulation. With gspo=True the sequence-level
        # importance ratios are used; "grpo" makes the aggregation pure GSPO
        # instead of TRL's default length-weighted "dapo".
        "loss_type": config.get("loss_type", "dapo"),
        "dataset": config.get("dataset", "rmanluo/RoG-cwq"),
        "eval_dataset": config.get("eval_dataset"),
        "prompt": config.get("prompt", "prompts/prompt_qwen36_angular2_nothink.yaml"),
        "question_key": config.get("question_key", "question"),
        "answer_key": config.get("answer_key", "answer"),
        "train_split": config.get("train_split", "train"),
        "eval_split": config.get("eval_split", "validation"),
        "max_eval_samples": config.get("max_eval_samples", 100),
        "eval_batch_size": config.get("eval_batch_size", 1),
        "generation_output": config.get("generation_output"),
        "custom_eval_steps": config.get("custom_eval_steps", 100),
        "index": config.get("index") or os.getenv("INDEX") or os.getenv("BASE_INDEX_PATH"),
        "tablename": config.get("tablename"),
        "index_tables": config.get("index_tables"),
        "default_index_table": config.get("default_index_table"),
        "index_table_column": config.get("index_table_column", "index_table"),
        "dataset_column": config.get("dataset_column", "dataset"),
        "split_column": config.get("split_column", "split"),
        "attn_implementation": config.get("attn_implementation"),
        "dataset_num_proc": config.get("dataset_num_proc"),
        "use_generation_cache": config.get("use_generation_cache", True),
        "parallel_workers": config.get("parallel_workers", 0),
        "output_dir": config.get("output_dir", "./grpo-output"),
        "save_steps": config.get("save_steps", 100),
        "save_total_limit": config.get("save_total_limit", 2),
        "resume_from_checkpoint": config.get("resume_from_checkpoint"),
        "epochs": config.get("epochs", 1),
        "max_steps": config.get("max_steps"),
        "batch_size": config.get("batch_size", 1),
        "gradient_accumulation_steps": config.get("gradient_accumulation_steps", 4),
        "num_generations": config.get("num_generations", config.get("n", 4)),
        "max_completion_length": config.get(
            "max_completion_length", generation.get("max_new_tokens", 256)
        ),
        "temperature": config.get("temperature", generation.get("temperature", 0.7)),
        "top_p": config.get("top_p", generation.get("top_p", 0.8)),
        "top_k": config.get("top_k", generation.get("top_k", 20)),
        "min_p": config.get("min_p", generation.get("min_p", 0.0)),
        "repetition_penalty": config.get(
            "repetition_penalty", generation.get("repetition_penalty", 1.0)
        ),
        "fact_pattern": config.get("fact_pattern", "<fact>"),
        "answer_pattern": config.get("answer_pattern", "<answer>"),
        "sentinel": config.get("sentinel", True),
        "count_pattern": config.get("count_pattern", "<count>"),
        "reward_alpha": config.get("reward_alpha", 0.1),
        "reward_beta": config.get("reward_beta", 0.05),
        "fact_soft_cap": config.get("fact_soft_cap", 20),
        "fact_penalty": config.get("fact_penalty", 0.01),
        "max_fact_penalty": config.get("max_fact_penalty", 0.2),
        "orphan_tool_penalty": config.get("orphan_tool_penalty", 0.1),
        "max_orphan_penalty": config.get("max_orphan_penalty", 0.3),
        "count_type": config.get("count_type", "count"),
        "enumeration_datasets": config.get("enumeration_datasets", ["webqsp"]),
        "use_normalized_reward": config.get("use_normalized_reward", True),
        "learning_rate": config.get("learning_rate", 5e-6),
        # Default to wandb; an explicit "report_to" still wins, and an
        # explicit "wandb": false opts out.
        "report_to": config.get(
            "report_to", "none" if config.get("wandb", None) is False else "wandb"
        ),
        "seed": config.get("seed", 42),
    }
    for name, default in defaults.items():
        if getattr(args, name, None) is None:
            setattr(args, name, default)
    for name in PATH_FIELDS:
        value = _expand_path(getattr(args, name, None))
        if isinstance(value, str) and "${" in value:
            warnings.warn(
                f"{name} still contains an unexpanded variable: {value} "
                f"(is the environment variable set for this job?)"
            )
        setattr(args, name, value)
    if args.no_normalized_reward:
        args.use_normalized_reward = False
    return args


def _text(completion: Any) -> str:
    """Accept both old TRL string completions and newer chat completions."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(completion)


def _marker_count(text: str, pattern: str) -> int:
    if pattern.startswith("<") and pattern.endswith(">"):
        name = pattern[1:-1].split()[0]
        return len(re.findall(rf"<{re.escape(name)}\b[^>]*>", text, flags=re.IGNORECASE))
    return len(re.findall(re.escape(pattern), text, flags=re.IGNORECASE))


def _tagged_blocks(text: str, pattern: str) -> list[str]:
    """Return the contents of well-formed ``<tag> ... </tag>`` sections.

    A bare mention of a marker (e.g. "I will use `<count>`") is not a tool call
    and must not be treated as one. Only complete sections count.
    """
    if not (pattern.startswith("<") and pattern.endswith(">")):
        return []
    name = pattern[1:-1].split()[0]
    return re.findall(
        rf"<{re.escape(name)}\b[^>]*>(.*?)</{re.escape(name)}\s*>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )


def _answer(text: str, answer_pattern: str = "<answer>") -> str:
    text = re.split(r"<\|im_end\|>|<\|end_of_text\|>", text, maxsplit=1)[0]
    if answer_pattern.startswith("<") and answer_pattern.endswith(">"):
        name = answer_pattern[1:-1].split()[0]
        tagged = re.search(
            rf"<{re.escape(name)}\b[^>]*>\s*(.*?)\s*</{re.escape(name)}\s*>",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if tagged:
            return tagged.group(1).strip()
    match = re.search(re.escape(answer_pattern) + r"\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def _as_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(v).strip().lower() for v in value]
    return [str(value).strip().lower()]


def _coerce_answer_value(value: Any) -> Any:
    """Parse the string-list answers used by the RoG Freebase datasets.

    Mintaka/2Wiki answers are plain strings and are returned unchanged.  A
    value such as ``"['A', 'B']"`` becomes ``['A', 'B']`` so enumeration and
    multi-answer scoring behave the same as for a native list column.
    """
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return value
    if isinstance(parsed, (list, tuple)):
        return [str(v) for v in parsed]
    return value


def _parsed_answer(text: str, answer_pattern: str = "<answer>") -> list[str] | None:
    try:
        return _as_list(json.loads(_answer(text, answer_pattern)))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


COUNT_MARKER = "<count>"
SENTINEL_MARKER = "no further records"
# Warn when the text-based usage counter drifts from the exact decoder counter
# by more than this many calls (prevents one-off noise from spamming logs).
TOOL_COUNTER_GAP_THRESHOLD = 5


def _tool_usage_stats(completions: Iterable[Any]) -> dict[str, int]:
    """Count constrained-tool usage in generated completions.

    Only the generated text is inspected (never the prompt), so a marker that
    appears in the system prompt cannot inflate the counters.

    Returns total occurrences and the number of completions containing each
    marker, for both the ``<count>`` tool and the exhausted-retrieval sentinel.
    """
    stats = {
        "count_uses": 0,
        "count_examples": 0,
        "sentinel_uses": 0,
        "sentinel_examples": 0,
    }
    for completion in completions:
        text = _text(completion)
        n_count = text.count(COUNT_MARKER)
        n_sentinel = text.lower().count(SENTINEL_MARKER)
        stats["count_uses"] += n_count
        stats["sentinel_uses"] += n_sentinel
        if n_count:
            stats["count_examples"] += 1
        if n_sentinel:
            stats["sentinel_examples"] += 1
    return stats


def _accumulate_tool_usage(counter: dict[str, int], completions: Iterable[Any]) -> None:
    for key, value in _tool_usage_stats(completions).items():
        counter[key] = counter.get(key, 0) + value


def _at(values: Any, index: int) -> Any:
    """Index a per-example column that TRL passes as a list (or a scalar)."""
    if values is None:
        return None
    if isinstance(values, (list, tuple)):
        return values[index] if index < len(values) else None
    return values


def _as_list_raw(value: Any) -> list[str]:
    """Like :func:`_as_list` but preserves case for the normalizer."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value]
    return [str(value).strip()]


def _dataset_tag(name: Any) -> str:
    """Map a dataset identifier onto the tags used by the normalizer."""
    lowered = str(name or "").lower()
    if "mintaka" in lowered:
        return "mintaka"
    if "2wiki" in lowered or "twowiki" in lowered:
        return "2wiki"
    if "webqsp" in lowered:
        return "webqsp"
    if "cwq" in lowered:
        return "cwq"
    return lowered


def _is_count_question_text(question: Any) -> bool:
    """Fallback count-question detector when no type column is available."""
    if not question:
        return False
    return bool(re.search(
        r"\bhow many\b|\bhow much\b|\bnumber of\b|\bcount\b",
        str(question), flags=re.IGNORECASE,
    ))


def _processor_kwargs(fact_pattern: str, count_pattern: str, sentinel: bool,
                      parallel_workers: int = 0) -> dict:
    """Static kwargs forwarded to the constrained logits processor.

    Centralised so the training and validation construction sites forward
    ``sentinel`` and ``count_pattern`` identically; tests assert the forwarding
    (a previously silent bug was that ``sentinel`` was never forwarded).
    """
    return {
        "num_beams": 1,
        "fact_pattern": fact_pattern,
        "count_pattern": count_pattern,
        "sentinel": bool(sentinel),
        "return_list": True,
        "avoid_duplicates": True,
        "reinit_states": True,
        "parallel_workers": parallel_workers,
    }


def _tool_counter_gaps(text_counter: dict, decoder_counter: dict) -> dict:
    """Text-detected usage minus exact decoder-side calls, per tool.

    The reward has to gate per example on the generated text, but the decoder
    counter is ground truth for aggregate usage. A persistent non-zero gap means
    the text heuristic is drifting (e.g. echoed or hallucinated markers).
    """
    return {
        "count": int(text_counter.get("count_uses", 0)) - int(decoder_counter.get("count_calls", 0)),
        "sentinel": int(text_counter.get("sentinel_uses", 0)) - int(decoder_counter.get("sentinel_calls", 0)),
    }


def _derive_question_type(example: dict) -> Any:
    """Resolve the question-type column across dataset schemas.

    Mintaka exposes ``complexityType`` (top level; nested under ``input_sample``
    in tooling output), 2Wiki uses ``type``, and some datasets nest either one.
    """
    for key in ("complexityType", "type", "question_type"):
        value = example.get(key)
        if value is not None:
            return value
    nested = example.get("input_sample")
    if isinstance(nested, dict):
        for key in ("complexityType", "type", "question_type"):
            value = nested.get(key)
            if value is not None:
                return value
    return None


def _derive_requires_enumeration(answer: Any, dataset: str, enumeration_datasets: Iterable[Any]) -> bool:
    """Flag multi-item answers on configured list-answer datasets.

    Conservative proxy for the report's "single-relation fan-out equals answer
    cardinality" heuristic: it only applies to the configured list-answer
    datasets (default WebQSP) and only for answers with two or more items, so it
    never turns a generic multi-answer Mintaka question into a sentinel target.
    """
    allowed = {_dataset_tag(name) for name in (enumeration_datasets or [])}
    if _dataset_tag(dataset) not in allowed:
        return False
    values = answer if isinstance(answer, (list, tuple)) else [answer]
    return sum(1 for v in values if v not in (None, "")) >= 2


def _select_stratified_eval(dataset, max_samples, dataset_column, count_type, seed=42):
    """Select an eval subset balanced over (dataset, question type).

    Every group present in the split keeps at least one example (budget
    permitting), and count questions are oversampled (4x weight) so the eval
    actually measures the count tool instead of a couple of lucky samples.
    """
    total = len(dataset)
    if max_samples is None or max_samples <= 0 or max_samples >= total:
        return dataset

    import random

    groups: dict[tuple[str, str], list[int]] = {}
    for index in range(total):
        example = dataset[index]
        dataset_tag = _dataset_tag(example.get(dataset_column))
        qtype = _derive_question_type(example)
        qtype = str(qtype).lower() if qtype is not None else "unknown"
        groups.setdefault((dataset_tag, qtype), []).append(index)

    count_label = str(count_type).lower()
    weights = {
        key: len(indices) * (4.0 if key[1] == count_label else 1.0)
        for key, indices in groups.items()
    }
    quota = {key: 0 for key in groups}
    used = 0

    # First give every group a slot (largest groups first) so rare types and
    # counts are represented even with a small budget.
    for key in sorted(groups, key=lambda k: -weights[k]):
        if used >= max_samples:
            break
        quota[key] = 1
        used += 1

    # Then fill the remainder proportionally, with the count boost.
    while used < max_samples:
        best_key = None
        best_score = -1.0
        for key, indices in groups.items():
            if quota[key] >= len(indices):
                continue
            score = weights[key] / (quota[key] + 1)
            if score > best_score:
                best_score = score
                best_key = key
        if best_key is None:
            break
        quota[best_key] += 1
        used += 1

    rng = random.Random(seed)
    selected: list[int] = []
    for key, indices in groups.items():
        take = min(quota[key], len(indices))
        if take <= 0:
            continue
        if take >= len(indices):
            selected.extend(indices)
        else:
            selected.extend(rng.sample(sorted(indices), take))
    selected.sort()
    return dataset.select(selected)


# Cached import probe for the conservative answer normalizer. ``None`` means
# "not yet attempted"; ``False`` records that the module is unavailable.
_COMPARE_ANSWERS = None


def _get_compare_answers():
    global _COMPARE_ANSWERS
    if _COMPARE_ANSWERS is None:
        try:
            from utils.answer_normalization import compare_answers
        except ImportError:
            try:
                from answer_normalization import compare_answers
            except ImportError:
                compare_answers = None
        _COMPARE_ANSWERS = compare_answers if compare_answers is not None else False
    return _COMPARE_ANSWERS or None


def _answers_normalized_match(prediction, reference, dataset="", question=None, input_sample=None):
    """True when the predicted answer is semantically equivalent to the gold."""
    compare_answers = _get_compare_answers()
    if compare_answers is None:
        return False
    try:
        result = compare_answers(
            _as_list_raw(prediction),
            _as_list_raw(reference),
            dataset=dataset,
            question=question if isinstance(question, str) else None,
            input_sample=input_sample if isinstance(input_sample, dict) else None,
        )
    except (TypeError, ValueError, KeyError, AttributeError, ArithmeticError):
        # Reward computation must never crash a training batch.
        return False
    return bool(result.normalized_match)


def exact_reward(
    completions: Iterable[Any],
    answer: Iterable[Any],
    fact_pattern: str = "<fact>",
    answer_pattern: str = "<answer>",
    question: Any = None,
    question_type: Any = None,
    requires_enumeration: Any = None,
    dataset: str = "",
    count_pattern: str = "<count>",
    count_type: str = "count",
    alpha: float = 0.1,
    beta: float = 0.05,
    fact_soft_cap: int = 20,
    fact_penalty: float = 0.01,
    max_fact_penalty: float = 0.2,
    orphan_tool_penalty: float = 0.1,
    max_orphan_penalty: float = 0.3,
    use_normalized: bool = True,
    stats: dict | None = None,
    details: list | None = None,
    **kwargs: Any,
) -> list[float]:
    """Score structure, correctness, retrieval efficiency and gated tool use.

    The reward is deliberately made up of small, interpretable terms:

    * ``0.3`` for a complete ``<fact>`` section;
    * ``0.3`` for exactly one complete ``<answer>`` section;
    * ``0.5`` for producing a JSON answer;
    * ``0..1`` for answer credit: full credit for a semantically equivalent
      answer (via ``utils/answer_normalization.py``), otherwise the strict
      answer-set intersection-over-union (IoU);
    * an optional penalty for fact calls beyond ``fact_soft_cap`` (disabled by
      setting ``fact_penalty=0``);
    * a penalty for marker mentions that never form a tool section, e.g.
      ``<count>``/``<fact>`` written in prose (``orphan_tool_penalty``);
    * a correct-answer-gated multiplier: ``alpha`` for count questions that
      emitted a complete ``<count>`` section, ``beta`` for enumeration
      questions whose ``<fact>`` section contains the exhausted-retrieval
      sentinel.

    A malformed answer still receives the formatting points, but no answer
    credit. When ``stats`` is given it is updated in place with component sums
    so each term can be logged separately and reward hacking is visible.
    """
    datasets = dataset if isinstance(dataset, (list, tuple)) else None
    input_samples = kwargs.get("input_sample")
    rewards = []
    for index, completion in enumerate(completions):
        text = _text(completion)
        normalized = text.lower()
        reference = _at(answer, index)
        qtype = _at(question_type, index)
        dataset_tag = _dataset_tag(_at(datasets, index) if datasets is not None else dataset)

        # Count only well-formed tool sections. A bare marker mention such as
        # "I will use `<count>`" is not a tool call and must not be rewarded.
        fact_blocks = _tagged_blocks(text, fact_pattern)
        count_blocks = _tagged_blocks(text, count_pattern)
        answer_blocks = _tagged_blocks(text, answer_pattern)
        fact_mentions = _marker_count(normalized, fact_pattern)
        count_mentions = _marker_count(normalized, count_pattern)
        orphan_fact = max(0, fact_mentions - len(fact_blocks))
        orphan_count = max(0, count_mentions - len(count_blocks))

        # Encourage the configured output sections. These are soft incentives
        # and do not require a complete proof.
        structure = 0.0
        if fact_blocks:
            structure += 0.3
        if len(answer_blocks) == 1:
            structure += 0.3

        # Optional retrieval-loop breaker. Set fact_penalty=0 to optimize
        # answers instead of brevity (the default for this round of RL).
        fact_calls = len(fact_blocks)
        excess = max(0, fact_calls - fact_soft_cap) if fact_soft_cap > 0 else 0
        penalty = min(max_fact_penalty, fact_penalty * excess) if fact_penalty > 0 else 0.0

        # Penalize marker mentions that never became a tool call so the policy
        # does not learn to write `<count>`/`<fact>` in prose.
        if orphan_tool_penalty > 0:
            orphan_penalty = min(
                max_orphan_penalty,
                orphan_tool_penalty * (orphan_fact + orphan_count),
            )
        else:
            orphan_penalty = 0.0

        json_score = 0.0
        answer_credit = 0.0
        normalized_match = False
        is_correct = False
        predicted_values = []
        try:
            # Only the text inside the configured answer section is evaluated.
            # This prevents facts or reasoning from counting as answers.
            predicted = json.loads(_answer(text, answer_pattern))
            predicted_values = _as_list(predicted)
            reference_values = _as_list(reference)

            # IoU gives partial credit for multi-answer questions: an answer
            # with some correct entities is better than an entirely wrong one.
            intersection = set(predicted_values) & set(reference_values)
            union = set(predicted_values) | set(reference_values)
            strict_iou = len(intersection) / len(union) if union else 0.0
            is_correct = bool(predicted_values) and set(predicted_values) == set(reference_values)

            json_score = 0.5
            # A semantically equivalent answer (same date, quantity, alias or
            # entity under folding) should not be punished for formatting.
            if use_normalized:
                normalized_match = _answers_normalized_match(
                    predicted,
                    reference,
                    dataset=dataset_tag,
                    question=_at(question, index),
                    input_sample=_at(input_samples, index),
                )
            answer_credit = 1.0 if normalized_match else strict_iou
            is_correct = is_correct or normalized_match
        except (TypeError, ValueError, json.JSONDecodeError):
            # Invalid JSON is common early in RL training. Do not make the
            # whole batch fail; keep the structural reward accumulated above.
            pass

        used_count = len(count_blocks) > 0
        used_sentinel = any(SENTINEL_MARKER in block.lower() for block in fact_blocks)
        if qtype:
            is_count_question = str(qtype).strip().lower() == count_type.lower()
        else:
            is_count_question = _is_count_question_text(_at(question, index))

        score = structure + json_score + answer_credit - penalty - orphan_penalty

        # Tool use is rewarded only when it coincides with a correct answer and
        # a question type where the tool is designed to help. A multiplier (not
        # an additive bonus) keeps the incentive modest.
        tool_bonus = 0.0
        if is_correct and score > 0:
            if is_count_question and used_count:
                tool_bonus += score * alpha
            if bool(_at(requires_enumeration, index)) and used_sentinel:
                tool_bonus += score * beta
        score += tool_bonus

        rewards.append(score)

        if stats is not None:
            stats["n"] = stats.get("n", 0) + 1
            stats["total"] = stats.get("total", 0.0) + score
            stats["structure"] = stats.get("structure", 0.0) + structure
            stats["json"] = stats.get("json", 0.0) + json_score
            stats["answer"] = stats.get("answer", 0.0) + answer_credit
            stats["fact_penalty"] = stats.get("fact_penalty", 0.0) + penalty
            stats["orphan_penalty"] = stats.get("orphan_penalty", 0.0) + orphan_penalty
            stats["orphan_fact"] = stats.get("orphan_fact", 0) + orphan_fact
            stats["orphan_count"] = stats.get("orphan_count", 0) + orphan_count
            stats["fact_blocks"] = stats.get("fact_blocks", 0) + len(fact_blocks)
            stats["count_blocks"] = stats.get("count_blocks", 0) + len(count_blocks)
            stats["tool_bonus"] = stats.get("tool_bonus", 0.0) + tool_bonus
            stats["fact_calls"] = stats.get("fact_calls", 0) + fact_calls
            stats["correct"] = stats.get("correct", 0) + int(is_correct)
            if normalized_match:
                stats["normalized_matches"] = stats.get("normalized_matches", 0) + 1

        if details is not None:
            details.append({
                "score": score,
                "correct": is_correct,
                "normalized_match": normalized_match,
                "fact_calls": fact_calls,
                "fact_blocks": len(fact_blocks),
                "count_blocks": len(count_blocks),
                "orphan_fact": orphan_fact,
                "orphan_count": orphan_count,
                "orphan_penalty": orphan_penalty,
                "used_count": used_count,
                "used_sentinel": used_sentinel,
                "question_type": None if qtype is None else str(qtype),
                "is_count_question": is_count_question,
                "answered": bool(predicted_values),
                "dont_know": bool(predicted_values and "i don't know" in predicted_values[0]),
                "answer_credit": answer_credit,
                "structure": structure,
                "json": json_score,
            })
    return rewards


class _GenerationKVCache:
    """Temporarily enable the KV cache for a no-grad generation pass.

    ``model.config.use_cache`` is disabled for gradient checkpointing during
    training, but generation does not need that and is dramatically slower
    without the cache.  The original values are restored afterwards.
    """

    def __init__(self, model: Any, enabled: bool = True):
        self.model = model
        self.enabled = enabled
        self.saved = []

    def __enter__(self):
        if self.enabled:
            for obj in (
                getattr(self.model, "config", None),
                getattr(self.model, "generation_config", None),
            ):
                if obj is not None and hasattr(obj, "use_cache"):
                    self.saved.append((obj, obj.use_cache))
                    obj.use_cache = True
        return self

    def __exit__(self, exc_type, exc, tb):
        for obj, value in self.saved:
            obj.use_cache = value
        return False


def _generation_kv_cache(model: Any, enabled: bool = True):
    return _GenerationKVCache(model, enabled)


def evaluate_policy(
    model: Any,
    tokenizer: Any,
    dataset: Any,
    max_completion_length: int,
    fact_pattern: str,
    answer_pattern: str,
    eval_batch_size: int = 1,
    generation_output: str | None = None,
    evaluation_label: str = "evaluation",
    decoder_counter: dict[str, int] | None = None,
    reward_options: dict | None = None,
    reward_stats: dict | None = None,
) -> dict[str, float]:
    """Evaluate deterministic completions against the dataset answer column.

    Accuracy is exact set match, while the reward remains useful for tracking
    partial answer overlap and formatting progress. Validation deliberately
    uses greedy decoding so the metric is not affected by sampling noise.
    """
    if dataset is None or len(dataset) == 0:
        return {}
    if eval_batch_size < 1:
        raise ValueError("eval_batch_size must be at least 1")

    import refactx
    import torch

    # Snapshot the exact decoder counters so this validation pass can report its
    # own tool-call totals rather than only the cumulative training totals.
    decoder_before = dict(decoder_counter) if decoder_counter is not None else None

    model.eval()
    rewards = []
    exact_matches = 0
    valid_answers = 0
    formatted = 0
    total = 0
    count_tool_uses = 0
    count_tool_examples = 0
    sentinel_uses = 0
    sentinel_examples = 0
    normalized_matches = 0
    dont_know_total = 0
    confident_wrong = 0
    answered = 0
    answered_without_triples = 0
    fact_calls_values = []
    completion_lengths = []
    truncated = 0
    type_total = {}
    type_correct = {}
    dataset_total = {}
    dataset_correct = {}
    count_used = 0
    count_used_correct = 0
    count_unused = 0
    count_unused_correct = 0
    count_type_total = 0
    reward_config = reward_options or {}
    fact_soft_cap = int(reward_config.get("fact_soft_cap", 20))

    if generation_output:
        output_parent = os.path.dirname(generation_output)
        if output_parent:
            os.makedirs(output_parent, exist_ok=True)

    for start in range(0, len(dataset), eval_batch_size):
        examples = [
            dataset[index]
            for index in range(start, min(start + eval_batch_size, len(dataset)))
        ]
        prompts = [example["prompt"] for example in examples]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad(), _generation_kv_cache(model, True):
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_completion_length,
                do_sample=False,
                num_beams=1,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
            )

        # ReFactX stores the constrained facts on generation state. Preserve
        # the same per-sample triple information as utils/eval.py when the
        # active model has a constrained logits processor.
        constrained_states = None
        try:
            constrained_states = refactx.get_constrained_states()
            constrained_states.beam_permutation()
        except (AttributeError, IndexError, RuntimeError):
            constrained_states = None

        input_length = inputs["input_ids"].shape[-1]
        for index, example in enumerate(examples):
            output_ids = outputs[index]
            completion_ids = output_ids[input_length:]
            new_tokens_generated = 0
            end_offset = len(completion_ids)
            for token in completion_ids:
                token_id = int(token)
                if token_id == tokenizer.pad_token_id:
                    end_offset = new_tokens_generated
                    break
                if token_id == tokenizer.eos_token_id:
                    end_offset = new_tokens_generated
                    break
                new_tokens_generated += 1
            completion_ids = completion_ids[:end_offset]
            completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
            reference = example["answer"]
            parsed = _parsed_answer(completion, answer_pattern)
            prediction = parsed if parsed is not None else []
            reference_values = set(_as_list(reference))
            precision = (
                len(set(prediction) & reference_values) / len(set(prediction))
                if prediction else 0.0
            )
            recall = (
                len(set(prediction) & reference_values) / len(reference_values)
                if reference_values else 0.0
            )
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            correct = int(set(prediction) == reference_values) if prediction else 0
            dont_know = int(bool(prediction) and "i don't know" in prediction[0])
            if parsed is not None:
                valid_answers += 1
                exact_matches += correct
            normalized_completion = completion.lower()
            is_formatted = (
                _marker_count(normalized_completion, fact_pattern) > 0
                and _marker_count(normalized_completion, answer_pattern) == 1
            )
            if is_formatted:
                formatted += 1
            usage = _tool_usage_stats([completion])
            count_tool_uses += usage["count_uses"]
            count_tool_examples += usage["count_examples"]
            sentinel_uses += usage["sentinel_uses"]
            sentinel_examples += usage["sentinel_examples"]
            detail = []
            sample_reward = exact_reward(
                [completion],
                [reference],
                fact_pattern=fact_pattern,
                answer_pattern=answer_pattern,
                question=[example.get("question")],
                question_type=[
                    example.get("question_type")
                    or example.get("complexityType")
                    or example.get("type")
                ],
                requires_enumeration=[example.get("requires_full_enumeration")],
                dataset=[example.get("dataset")],
                input_sample=[dict(example)],
                stats=reward_stats,
                details=detail,
                **(reward_options or {}),
            )[0]
            info = detail[0] if detail else {}
            rewards.append(sample_reward)

            triples = []
            if constrained_states is not None:
                try:
                    state = constrained_states[index, 0]
                    triples = [
                        tokenizer.decode(triple)
                        for triple in getattr(state, "generated_triples", [])
                    ]
                except (AttributeError, IndexError, TypeError, RuntimeError):
                    triples = []

            normalized_matches += int(bool(info.get("normalized_match")))
            answered += int(bool(info.get("answered")))
            dont_know_total += int(bool(info.get("dont_know")))
            confident_wrong += int(
                bool(info.get("answered")) and not info.get("correct") and not info.get("dont_know")
            )
            fact_calls_values.append(int(info.get("fact_calls", 0)))
            completion_lengths.append(new_tokens_generated)
            if len(completion_ids) >= max_completion_length:
                truncated += 1
            if info.get("answered") and not triples:
                answered_without_triples += 1
            qtype = info.get("question_type")
            if qtype:
                type_total[qtype] = type_total.get(qtype, 0) + 1
                type_correct[qtype] = type_correct.get(qtype, 0) + int(bool(info.get("correct")))
            dataset_tag = _dataset_tag(example.get("dataset") or "unknown")
            dataset_total[dataset_tag] = dataset_total.get(dataset_tag, 0) + 1
            dataset_correct[dataset_tag] = dataset_correct.get(dataset_tag, 0) + int(bool(info.get("correct")))
            if info.get("is_count_question"):
                count_type_total += 1
                if info.get("used_count"):
                    count_used += 1
                    count_used_correct += int(bool(info.get("correct")))
                else:
                    count_unused += 1
                    count_unused_correct += int(bool(info.get("correct")))

            if generation_output:
                sample = {
                    "input_sample": dict(example),
                    "gt_answer": reference,
                    "question": example.get("question"),
                    "answer_complete": bool(prediction),
                    "prediction": prediction,
                    "full_prediction": completion,
                    "prompt": tokenizer.decode(inputs["input_ids"][index]),
                    "full_sample": tokenizer.decode(output_ids),
                    "triples": triples,
                    "new_tokens_generated": new_tokens_generated,
                    "reached_max_tokens": len(completion_ids) >= max_completion_length,
                    "evaluation": {
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "correct": correct,
                        "dont_know": dont_know,
                    },
                    # Keep the existing fields for consumers of the previous
                    # TRL JSONL format.
                    "evaluation_label": evaluation_label,
                    "reference_answer": reference,
                    "completion": completion,
                    "parsed_answer": parsed,
                    "reward": sample_reward,
                    "formatted": is_formatted,
                }
                with open(generation_output, "a", encoding="utf-8") as output_file:
                    output_file.write(json.dumps(sample, ensure_ascii=False) + "\n")
            total += 1

    metrics = {
        "eval/answer_accuracy": exact_matches / total,
        "eval/normalized_accuracy": normalized_matches / total,
        "eval/valid_answer_rate": valid_answers / total,
        "eval/format_accuracy": formatted / total,
        "eval/mean_reward": sum(rewards) / total,
        "eval/answered_rate": answered / total,
        "eval/abstention_rate": dont_know_total / total,
        "eval/confident_wrong_rate": confident_wrong / total,
        "eval/answered_without_triples_rate": answered_without_triples / total,
        "eval/fact_calls_mean": sum(fact_calls_values) / total,
        "eval/fact_calls_p50": _percentile(fact_calls_values, 0.50),
        "eval/fact_calls_p90": _percentile(fact_calls_values, 0.90),
        "eval/fact_calls_p99": _percentile(fact_calls_values, 0.99),
        "eval/fact_calls_over_cap_frac": (
            sum(1 for value in fact_calls_values if value > fact_soft_cap) / total
        ),
        "eval/completion_len_mean": sum(completion_lengths) / total,
        "eval/completion_len_p90": _percentile(completion_lengths, 0.90),
        "eval/truncation_rate": truncated / total,
        "eval/count_tool_uses": count_tool_uses,
        "eval/count_tool_rate": count_tool_examples / total,
        "eval/count_tool_mean": count_tool_uses / total,
        "eval/sentinel_uses": sentinel_uses,
        "eval/sentinel_rate": sentinel_examples / total,
        "eval/sentinel_mean": sentinel_uses / total,
    }
    if count_type_total:
        metrics["eval/tool/count_usage_on_count_type"] = count_used / count_type_total
    if count_used:
        metrics["eval/tool/count_accuracy_used"] = count_used_correct / count_used
    if count_unused:
        metrics["eval/tool/count_accuracy_unused"] = count_unused_correct / count_unused
    for qtype, count in type_total.items():
        safe = str(qtype).replace(" ", "_").replace("/", "_")
        metrics[f"eval/per_type/{safe}/accuracy"] = type_correct.get(qtype, 0) / count
        metrics[f"eval/per_type/{safe}/n"] = count
    for dataset_tag, count in dataset_total.items():
        safe = str(dataset_tag).replace(" ", "_").replace("/", "_")
        metrics[f"eval/dataset/{safe}/accuracy"] = dataset_correct.get(dataset_tag, 0) / count
        metrics[f"eval/dataset/{safe}/n"] = count
    metrics.update(_reward_component_metrics(reward_stats, prefix="eval/"))
    if decoder_before is not None and decoder_counter is not None:
        metrics["eval/count_tool_calls"] = decoder_counter["count_calls"] - decoder_before["count_calls"]
        metrics["eval/sentinel_calls"] = decoder_counter["sentinel_calls"] - decoder_before["sentinel_calls"]
    return metrics


def _reward_component_metrics(stats: dict | None, prefix: str = "") -> dict[str, float]:
    """Convert accumulated reward component sums into per-example means."""
    if not stats:
        return {}
    n = stats.get("n", 0)
    if not n:
        return {}
    return {
        f"{prefix}reward_mean": stats.get("total", 0.0) / n,
        f"{prefix}reward_structure": stats.get("structure", 0.0) / n,
        f"{prefix}reward_json": stats.get("json", 0.0) / n,
        f"{prefix}reward_answer": stats.get("answer", 0.0) / n,
        f"{prefix}reward_fact_penalty": stats.get("fact_penalty", 0.0) / n,
        f"{prefix}reward_orphan_penalty": stats.get("orphan_penalty", 0.0) / n,
        f"{prefix}reward_orphan_fact": stats.get("orphan_fact", 0) / n,
        f"{prefix}reward_orphan_count": stats.get("orphan_count", 0) / n,
        f"{prefix}reward_fact_blocks": stats.get("fact_blocks", 0) / n,
        f"{prefix}reward_count_blocks": stats.get("count_blocks", 0) / n,
        f"{prefix}reward_tool_bonus": stats.get("tool_bonus", 0.0) / n,
        f"{prefix}reward_fact_calls": stats.get("fact_calls", 0) / n,
        f"{prefix}reward_correct_rate": stats.get("correct", 0) / n,
        f"{prefix}reward_normalized_matches": stats.get("normalized_matches", 0) / n,
    }


def _stats_delta(current: dict, previous: dict) -> dict:
    """Per-key difference between two running accumulators (window = now - then)."""
    keys = set(current) | set(previous)
    return {key: current.get(key, 0) - previous.get(key, 0) for key in keys}


def _training_log_keys(metrics: dict) -> dict:
    """Strip a leading ``train/`` because Trainer adds the split prefix itself.

    Without this, ``train/window/reward_answer`` is reported as
    ``train/train/window/reward_answer`` (and the W&B dashboards miss it).
    """
    return {
        key[len("train/"):] if key.startswith("train/") else key: value
        for key, value in metrics.items()
    }


def _percentile(values, q: float) -> float:
    """Linear-interpolated percentile without pulling in numpy."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def _tool_counter_metrics(tool_counter: dict, decoder_counter: dict, prefix: str = "train/tool/") -> dict:
    """Aggregate tool usage from the exact decoder counters plus text gaps."""
    gaps = _tool_counter_gaps(tool_counter, decoder_counter)
    return {
        f"{prefix}count_uses": tool_counter.get("count_uses", 0),
        f"{prefix}count_examples": tool_counter.get("count_examples", 0),
        f"{prefix}sentinel_uses": tool_counter.get("sentinel_uses", 0),
        f"{prefix}sentinel_examples": tool_counter.get("sentinel_examples", 0),
        f"{prefix}count_calls": decoder_counter.get("count_calls", 0),
        f"{prefix}sentinel_calls": decoder_counter.get("sentinel_calls", 0),
        f"{prefix}count_text_decoder_gap": gaps["count"],
        f"{prefix}sentinel_text_decoder_gap": gaps["sentinel"],
    }


def _accumulate_group_metrics(rewards, details, num_generations, stats: dict) -> None:
    """Accumulate GRPO/GSPO group signal and per-example behavior counts.

    Groups are consecutive chunks of ``num_generations`` completions (TRL emits
    each prompt's generations contiguously). If the batch does not divide evenly
    the group portion is skipped and counted instead of silently mis-grouped.
    """
    total = len(rewards)
    if num_generations and num_generations > 1 and total and total % num_generations == 0:
        for start in range(0, total, num_generations):
            group = rewards[start:start + num_generations]
            group_details = details[start:start + num_generations]
            mean = sum(group) / len(group)
            std = (sum((value - mean) ** 2 for value in group) / len(group)) ** 0.5
            stats["groups"] = stats.get("groups", 0) + 1
            stats["group_std_sum"] = stats.get("group_std_sum", 0.0) + std
            if max(group) == min(group):
                stats["zero_adv_groups"] = stats.get("zero_adv_groups", 0) + 1
            corrects = [bool(d.get("correct")) for d in group_details]
            if corrects and all(corrects):
                stats["all_correct_groups"] = stats.get("all_correct_groups", 0) + 1
            if not any(corrects):
                stats["all_wrong_groups"] = stats.get("all_wrong_groups", 0) + 1
            if any(corrects):
                stats["pass_groups"] = stats.get("pass_groups", 0) + 1
            tools = [bool(d.get("used_count") or d.get("used_sentinel")) for d in group_details]
            if len(set(tools)) > 1:
                stats["tool_diverse_groups"] = stats.get("tool_diverse_groups", 0) + 1
    else:
        stats["groups_skipped"] = stats.get("groups_skipped", 0) + 1

    for detail in details:
        stats["examples"] = stats.get("examples", 0) + 1
        stats["answered"] = stats.get("answered", 0) + int(bool(detail.get("answered")))
        stats["dont_know"] = stats.get("dont_know", 0) + int(bool(detail.get("dont_know")))
        stats["correct"] = stats.get("correct", 0) + int(bool(detail.get("correct")))
        stats["confident_wrong"] = stats.get("confident_wrong", 0) + int(
            bool(detail.get("answered")) and not detail.get("correct") and not detail.get("dont_know")
        )
        stats["normalized_correct"] = stats.get("normalized_correct", 0) + int(bool(detail.get("normalized_match")))
        stats["sum_answer_credit"] = stats.get("sum_answer_credit", 0.0) + float(detail.get("answer_credit", 0.0))
        stats["sum_structure"] = stats.get("sum_structure", 0.0) + float(detail.get("structure", 0.0))
        stats["sum_json"] = stats.get("sum_json", 0.0) + float(detail.get("json", 0.0))
        stats["used_sentinel"] = stats.get("used_sentinel", 0) + int(bool(detail.get("used_sentinel")))
        is_count = bool(detail.get("is_count_question"))
        used_count = bool(detail.get("used_count"))
        stats["count_type"] = stats.get("count_type", 0) + int(is_count)
        stats["count_used"] = stats.get("count_used", 0) + int(is_count and used_count)
        stats["count_used_correct"] = stats.get("count_used_correct", 0) + int(
            is_count and used_count and detail.get("correct")
        )
        stats["count_unused"] = stats.get("count_unused", 0) + int(is_count and not used_count)
        stats["count_unused_correct"] = stats.get("count_unused_correct", 0) + int(
            is_count and not used_count and detail.get("correct")
        )


def _group_metrics(stats: dict) -> dict:
    """Group-relative diagnostics (does the rollout group carry a gradient?)."""
    groups = stats.get("groups", 0)
    metrics = {
        "train/group/groups": groups,
        "train/group/groups_skipped": stats.get("groups_skipped", 0),
    }
    if groups:
        metrics.update({
            "train/group/zero_advantage_frac": stats.get("zero_adv_groups", 0) / groups,
            "train/group/reward_std": stats.get("group_std_sum", 0.0) / groups,
            "train/group/all_correct_frac": stats.get("all_correct_groups", 0) / groups,
            "train/group/all_wrong_frac": stats.get("all_wrong_groups", 0) / groups,
            "train/group/pass_at_k": stats.get("pass_groups", 0) / groups,
            "train/group/tool_diverse_frac": stats.get("tool_diverse_groups", 0) / groups,
        })
    return metrics


def _behavior_metrics(stats: dict) -> dict:
    """Per-window behavior and alignment metrics from per-example details."""
    examples = stats.get("examples", 0)
    if not examples:
        return {}
    count_type = stats.get("count_type", 0)
    count_used = stats.get("count_used", 0)
    count_unused = stats.get("count_unused", 0)
    metrics = {
        "train/window/abstention_rate": stats.get("dont_know", 0) / examples,
        "train/window/confident_wrong_rate": stats.get("confident_wrong", 0) / examples,
        "train/window/normalized_match_frac": stats.get("normalized_correct", 0) / examples,
        "train/window/answer_vs_correct_gap": (
            stats.get("sum_answer_credit", 0.0) - stats.get("correct", 0)
        ) / examples,
        "train/window/format_saturation": (stats.get("sum_structure", 0.0) / examples) / 0.6,
        "train/tool/count_type_rate": count_type / examples,
        "train/tool/sentinel_usage_rate": stats.get("used_sentinel", 0) / examples,
    }
    if count_type:
        metrics["train/tool/count_usage_on_count_type"] = count_used / count_type
    if count_used:
        metrics["train/tool/count_accuracy_used"] = stats.get("count_used_correct", 0) / count_used
    if count_unused:
        metrics["train/tool/count_accuracy_unused"] = stats.get("count_unused_correct", 0) / count_unused
    return metrics


class CustomMetricsCallback:
    """Log deterministic validation metrics during long GRPO runs."""

    def __init__(self, tokenizer, dataset, max_completion_length, fact_pattern,
                 answer_pattern, eval_batch_size, every_steps, triple_counter,
                 generation_output, model, tool_counter=None, decoder_counter=None,
                 reward_options=None, train_reward_stats=None, train_group_stats=None):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_completion_length = max_completion_length
        self.fact_pattern = fact_pattern
        self.answer_pattern = answer_pattern
        self.eval_batch_size = eval_batch_size
        self.every_steps = every_steps
        self.triple_counter = triple_counter
        self.tool_counter = tool_counter or {}
        self.decoder_counter = decoder_counter or {}
        self.generation_output = generation_output
        self.model = model
        self.reward_options = reward_options or {}
        self.train_reward_stats = train_reward_stats or {}
        self.train_group_stats = train_group_stats or {}
        self.trainer = None
        self.pending_metrics = None
        self._tool_gap_warned = {}
        self._reward_snapshot = {}
        self._group_snapshot = {}

    def __getattr__(self, name):
        # TrainerCallbackHandler invokes every lifecycle event directly.
        # This callback only needs on_step_end and on_log; all other events
        # leave the trainer control object unchanged.
        if name.startswith("on_"):
            return lambda args, state, control, **kwargs: control
        raise AttributeError(name)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        model = self.model if self.model is not None else model
        if state.global_step == 0:
            return control

        logging_steps = int(getattr(args, "logging_steps", 0) or 0)
        log_due = logging_steps > 0 and state.global_step % logging_steps == 0
        should_eval = (
            self.dataset is not None
            and len(self.dataset) > 0
            and self.every_steps > 0
            and state.global_step % self.every_steps == 0
            and model is not None
        )
        if not (log_due or should_eval):
            return control

        metrics = {}
        if should_eval:
            was_training = model.training
            eval_reward_stats = {}
            metrics.update(evaluate_policy(
                model, self.tokenizer, self.dataset, self.max_completion_length,
                self.fact_pattern, self.answer_pattern, self.eval_batch_size,
                generation_output=self.generation_output,
                evaluation_label=f"step_{state.global_step}",
                decoder_counter=self.decoder_counter,
                reward_options=self.reward_options,
                reward_stats=eval_reward_stats,
            ))
            if was_training:
                model.train()

        # Window deltas are taken only when logging, so each point covers the
        # whole logging interval rather than a single optimizer step.
        reward_window = _stats_delta(self.train_reward_stats, self._reward_snapshot)
        group_window = _stats_delta(self.train_group_stats, self._group_snapshot)
        self._reward_snapshot = dict(self.train_reward_stats)
        self._group_snapshot = dict(self.train_group_stats)

        metrics["constrained_triples_generated"] = self.triple_counter["count"]
        metrics.update(_tool_counter_metrics(self.tool_counter, self.decoder_counter))
        metrics.update(_reward_component_metrics(
            self.train_reward_stats, prefix="train/cumulative/"
        ))
        metrics.update(_reward_component_metrics(
            reward_window, prefix="train/window/"
        ))
        metrics.update(_group_metrics(group_window))
        metrics.update(_behavior_metrics(group_window))

        self._warn_on_tool_gap(_tool_counter_gaps(self.tool_counter, self.decoder_counter))
        if self.trainer is not None and metrics:
            self.trainer.log(_training_log_keys(metrics))
        return control

    def _warn_on_tool_gap(self, gaps):
        """Warn when text-detected usage drifts from the decoder counter.

        Only warns when a gap grows past the threshold, so a persistent small
        offset does not spam every logging step.
        """
        for name, gap in gaps.items():
            if abs(gap) <= TOOL_COUNTER_GAP_THRESHOLD:
                continue
            previous = self._tool_gap_warned.get(name, 0)
            if abs(gap) <= abs(previous):
                continue
            self._tool_gap_warned[name] = gap
            warnings.warn(
                f"{name} tool text/decoder mismatch is {gap} "
                f"(text counter {name}_uses vs decoder {name}_calls); "
                f"check for echoed or hallucinated tool markers.",
                RuntimeWarning,
                stacklevel=2,
            )

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Reward/window/group metrics are logged from on_step_end. Mutating
        # `logs` here is too late: the W&B callback runs before this one, which
        # is why train/window/* used to be missing from W&B.
        return control


class JudgeReward:
    """Callable reward function backed by a frozen causal language model."""

    def __init__(self, model_name: str, torch_module: Any, transformers_module: Any, use_cuda: bool = True):
        torch = torch_module
        AutoModelForCausalLM = transformers_module.AutoModelForCausalLM
        AutoTokenizer = transformers_module.AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16 if use_cuda else torch.float32,
            device_map="auto",
        )
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def prompt(question: str, answer: str, facts: str) -> str:
        return f"""You are a strict evaluator for question answering.

Question:
{question}

Model answer:
{answer}

Supporting facts:
{facts}

Return ONLY valid JSON with integer keys: supported, complete, correct.
Use 1 when true and 0 when false. Do not include markdown.
"""

    def _score_one(self, question: str, completion: Any, facts: str) -> float:
        prompt = self.prompt(question, _text(completion), facts)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.model.device)
        with self.torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = self.tokenizer.decode(
            output[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        )
        match = re.search(r"\{.*?\}", generated, flags=re.DOTALL)
        if not match:
            return 0.0
        try:
            scores = json.loads(match.group(0))
            return float(scores.get("correct", 0)) + 0.5 * float(scores.get("supported", 0)) + 0.5 * float(scores.get("complete", 0))
        except (TypeError, ValueError, json.JSONDecodeError):
            return 0.0

    def __call__(self, completions, question=None, **kwargs) -> list[float]:
        questions = question or kwargs.get("questions") or [""] * len(completions)
        facts = kwargs.get("facts") or [""] * len(completions)
        return [self._score_one(q, c, f) for q, c, f in zip(completions, questions, facts)]


def main() -> None:
    args = parse_args()

    # W&B picks its project from WANDB_PROJECT; without it runs land in the
    # generic "huggingface" project. Set it before the Trainer builds its
    # integration callbacks. An explicit environment value still wins.
    if args.report_to != "none":
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    import torch
    if not args.no_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required by default but is not available. "
            "Use --no-cuda only when a CPU run is intentional."
        )
    use_cuda = torch.cuda.is_available() and not args.no_cuda

    import refactx
    from datasets import load_dataset
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    prompt_template = refactx.load_prompt(args.prompt)
    print(f"Loaded prompt from {args.prompt}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # T4/Turing does not support native bfloat16; use fp16 for local smoke runs
    # while keeping bf16 on Ampere+ hardware.
    supports_bf16 = bool(use_cuda and torch.cuda.get_device_capability()[0] >= 8)
    model_dtype = (
        torch.bfloat16 if supports_bf16
        else (torch.float16 if use_cuda else torch.float32)
    )
    model_kwargs = {
        "dtype": model_dtype,
        "device_map": "auto",
    }
    attn_implementation = args.attn_implementation or os.getenv("REFACTX_ATTN_IMPLEMENTATION")
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    # Rollouts and validation still use the KV cache; the log-probability pass
    # in GRPO explicitly disables it.  Gradient checkpointing only requires
    # ``config.use_cache=False`` for the training forward pass.
    model.config.use_cache = False
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = bool(args.use_generation_cache)
    if args.adapter:
        # Load the existing adapter as the trainable policy. The base model
        # must match the one recorded in adapter_config.json.
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=8, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ))
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = bool(args.use_generation_cache)
    model.print_trainable_parameters()

    def _load_training_dataset(path, split_column, train_split, eval_split):
        """Load a HF dataset or a local JSONL with a split column."""
        if os.path.isfile(path) and path.lower().endswith((".jsonl", ".json", ".parquet")):
            data_format = "parquet" if path.lower().endswith(".parquet") else "json"
            loaded = load_dataset(data_format, data_files=path, split="train")
            if split_column and split_column in loaded.column_names:
                train_split_ds = loaded.filter(
                    lambda example: str(example[split_column]) == str(train_split)
                )
                eval_split_ds = loaded.filter(
                    lambda example: str(example[split_column]) == str(eval_split)
                )
            else:
                train_split_ds, eval_split_ds = loaded, None
        else:
            raw = load_dataset(path)
            train_split_ds = raw[train_split]
            eval_split_ds = raw.get(eval_split, None)
        return train_split_ds, eval_split_ds

    train, evaluation = _load_training_dataset(
        args.dataset, args.split_column, args.train_split, args.eval_split
    )
    if args.eval_dataset:
        if os.path.isfile(args.eval_dataset):
            evaluation = load_dataset(
                "json", data_files=args.eval_dataset, split="train"
            )
        else:
            warnings.warn(
                f"eval_dataset not found at {args.eval_dataset}; "
                f"using the '{args.eval_split}' split",
                RuntimeWarning,
                stacklevel=2,
            )
    if evaluation is not None and len(evaluation) == 0:
        evaluation = None
    if evaluation is not None and args.max_eval_samples is not None:
        evaluation = _select_stratified_eval(
            evaluation, args.max_eval_samples, args.dataset_column,
            args.count_type, args.seed,
        )

    def format_example(example):
        question = example[args.question_key]
        answer = _coerce_answer_value(example[args.answer_key])
        dataset_name = example.get(args.dataset_column) or args.dataset or ""
        index_table = (
            example.get(args.index_table_column)
            or args.default_index_table
            or args.tablename
        )
        question_type = example.get("question_type")
        if question_type is None:
            question_type = _derive_question_type(example)
        requires_enumeration = example.get("requires_full_enumeration")
        if requires_enumeration is None:
            requires_enumeration = _derive_requires_enumeration(
                answer, dataset_name, args.enumeration_datasets
            )
        prompt = refactx.apply_prompt_template(
            tokenizer, prompt_template=prompt_template, question=question
        )
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        return {
            "prompt": prompt,
            # Stable key used by the multi-index generation wrapper to recover
            # this example's KB table from the padded input ids.
            "prompt_key": tokenizer.decode(prompt_ids, skip_special_tokens=True).strip(),
            "question": question,
            "answer": answer,
            # Per-example metadata used for reward normalization and KB routing.
            "dataset": dataset_name,
            "index_table": index_table,
            # Normalized across datasets so the reward does not have to guess
            # whether the type lives in complexityType, type, or input_sample.
            "question_type": question_type,
            "requires_full_enumeration": bool(requires_enumeration),
        }

    map_kwargs = {}
    if args.dataset_num_proc and args.dataset_num_proc > 1:
        map_kwargs["num_proc"] = args.dataset_num_proc
    train = train.map(format_example, **map_kwargs)
    if evaluation is not None:
        evaluation = evaluation.map(format_example, **map_kwargs)

    judge = JudgeReward(args.judge_model, torch, type("Transformers", (), {
        "AutoTokenizer": AutoTokenizer, "AutoModelForCausalLM": AutoModelForCausalLM,
    }), use_cuda=use_cuda) if args.judge_model else None
    # Reward configuration shared by the training reward and the validation pass.
    reward_options = {
        "count_pattern": args.count_pattern,
        "count_type": args.count_type,
        "alpha": args.reward_alpha,
        "beta": args.reward_beta,
        "fact_soft_cap": args.fact_soft_cap,
        "fact_penalty": args.fact_penalty,
        "max_fact_penalty": args.max_fact_penalty,
        "orphan_tool_penalty": args.orphan_tool_penalty,
        "max_orphan_penalty": args.max_orphan_penalty,
        "use_normalized": args.use_normalized_reward,
    }
    # Cumulative component sums so on_log can report each reward term separately.
    train_reward_stats = {}
    # Cumulative group/behavior sums; on_log derives per-window deltas from these.
    train_group_stats = {}
    # Keep a named function here: TRL 1.0 records reward names from
    # ``__name__`` and does not handle callable objects or partials.
    reward_debug = {"done": False}
    triple_counter = {"count": 0}
    tool_counter = {
        "count_uses": 0,
        "count_examples": 0,
        "sentinel_uses": 0,
        "sentinel_examples": 0,
    }
    # Exact decoder-side counts: incremented by the constrained processor when it
    # computes a count or emits the exhausted-retrieval sentinel. Unlike the
    # text-based counters these reflect real tool invocations, not marker text.
    decoder_counter = {
        "count_calls": 0,
        "sentinel_calls": 0,
    }

    def on_count_generated(_prefix_text, _count):
        decoder_counter["count_calls"] += 1

    def on_sentinel_generated():
        decoder_counter["sentinel_calls"] += 1

    def reward(completions, answer, question=None, **kwargs):
        # Track which constrained tools the policy actually used on this rollout
        # batch. This runs on generated text only, never on the prompt.
        _accumulate_tool_usage(tool_counter, completions)
        if os.getenv("REFACTX_DEBUG_REWARD_COLUMNS") and not reward_debug["done"]:
            reward_debug["done"] = True
            print("reward kwargs:", sorted(kwargs))
            print("reward datasets:", kwargs.get("dataset"))
        if judge is not None:
            return judge(completions, question=question, **kwargs)
        details = []
        rewards = exact_reward(
            completions,
            answer,
            fact_pattern=args.fact_pattern,
            answer_pattern=args.answer_pattern,
            question=question,
            question_type=kwargs.get("question_type") or kwargs.get("complexityType") or kwargs.get("type"),
            requires_enumeration=kwargs.get("requires_full_enumeration"),
            dataset=kwargs.get("dataset") or args.dataset or "",
            stats=train_reward_stats,
            details=details,
            **reward_options,
        )
        # GRPO/GSPO group diagnostics and conditional tool/behavior metrics.
        _accumulate_group_metrics(rewards, details, args.num_generations, train_group_stats)
        return rewards

    def on_triple_generated(_sequence):
        triple_counter["count"] += 1

    if args.index:
        table_names = args.index_tables
        table_urls = {}
        if isinstance(table_names, dict):
            table_urls = {str(name): url for name, url in table_names.items()}
            table_names = list(table_urls)
        elif table_names:
            table_names = [str(name) for name in table_names]
            table_urls = {name: args.index for name in table_names}
        elif args.tablename:
            table_names = [args.tablename]
            table_urls = {args.tablename: args.index}
        else:
            # A mixed JSONL can declare the tables itself; preload exactly the
            # tables it references.
            table_names = sorted({
                example.get(args.index_table_column)
                for example in list(train) + ([] if evaluation is None else list(evaluation))
                if example.get(args.index_table_column)
            })
            table_urls = {name: args.index for name in table_names}

        if not table_names:
            raise ValueError(
                "--index was given but no table names were configured; pass "
                "--index-tables/--tablename or include an index_table column"
            )
        registry = {
            name: refactx.load_index(
                table_urls.get(name) or args.index,
                tokenizer=tokenizer,
                tablename=name,
            )
            for name in table_names
        }
        default_table = args.default_index_table or table_names[0]
        index = refactx.RoutedIndex(registry, default=default_table)
        print(f"Loaded indexes: {sorted(registry)} (default: {default_table})")

        # Map the prompt (without special tokens) to its KB table so the wrapper
        # can assign the right index to every row of a mixed generation batch.
        prompt_tables = {}
        ambiguous_prompts = set()
        for example in list(train) + ([] if evaluation is None else list(evaluation)):
            key = example.get("prompt_key")
            table = example.get(args.index_table_column) or default_table
            if not key:
                continue
            previous = prompt_tables.get(key)
            if previous is not None and previous != table:
                ambiguous_prompts.add(key)
            else:
                prompt_tables[key] = table
        if ambiguous_prompts:
            warnings.warn(
                f"{len(ambiguous_prompts)} prompt(s) map to multiple KB tables; "
                f"using {default_table!r} for those rows",
                RuntimeWarning,
                stacklevel=2,
            )

        original_generate = model.generate
        routing_stats = {"misses": 0, "calls": 0, "tables": Counter(), "logged": False}

        def _row_table(input_ids, row):
            key = tokenizer.decode(input_ids[row].tolist(), skip_special_tokens=True).strip()
            table = prompt_tables.get(key)
            if table is None:
                routing_stats["misses"] += 1
                table = default_table
            if key in ambiguous_prompts:
                table = default_table
            return table

        def constrained_generate(*call_args, **call_kwargs):
            # GRPO generation uses the expanded training batch, while the
            # validation pass generates its own batch. Build a processor for
            # the actual batch and attach one KB table per row.
            input_ids = call_kwargs.get("input_ids")
            if input_ids is None and call_args:
                input_ids = call_args[0]
            if input_ids is None:
                return original_generate(*call_args, **call_kwargs)
            batch_size = input_ids.shape[0]
            num_return_sequences = int(call_kwargs.get("num_return_sequences") or 1)
            expanded_batch = batch_size * num_return_sequences
            if not routing_stats["logged"]:
                generation_config = call_kwargs.get("generation_config")
                print(
                    "routing debug: input_ids shape=",
                    tuple(input_ids.shape) if input_ids is not None else None,
                    "num_return_sequences=",
                    num_return_sequences,
                    "config_num_return_sequences=",
                    getattr(generation_config, "num_return_sequences", None),
                )
            active_processor = refactx.get_constrained_logits_processor(
                tokenizer, index,
                num_batches=expanded_batch,
                on_triple_generated=on_triple_generated,
                on_count_generated=on_count_generated,
                on_sentinel_generated=on_sentinel_generated,
                **_processor_kwargs(
                    args.fact_pattern, args.count_pattern, args.sentinel,
                    args.parallel_workers,
                ),
            )
            if batch_size is not None:
                row_tables = []
                for row in range(batch_size):
                    table = _row_table(input_ids, row)
                    row_tables.append(table)
                    # transformers expands prompts with repeat_interleave for
                    # num_return_sequences; mirror that order when attaching
                    # the KB table to the constrained states.
                    for replica in range(num_return_sequences):
                        state_index = row * num_return_sequences + replica
                        try:
                            active_processor.states.states[state_index][0].index_table = table
                        except (AttributeError, IndexError):
                            break
                routing_stats["calls"] += 1
                routing_stats["tables"].update(table for table in row_tables for _ in range(num_return_sequences))
                if not routing_stats["logged"]:
                    routing_stats["logged"] = True
                    print(
                        "index routing (first generation batch): "
                        f"{dict(Counter(table for table in row_tables for _ in range(num_return_sequences)))} "
                        f"misses={routing_stats['misses']}"
                    )
            call_kwargs.setdefault("logits_processor", active_processor)
            return original_generate(*call_args, **call_kwargs)

        model.generate = constrained_generate

    config = GRPOConfig(
        output_dir=args.output_dir,
        # Keep dataset/index_table/dataset columns so the reward and the
        # multi-index generation wrapper receive per-example metadata.
        remove_unused_columns=False,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps is not None else -1,
        learning_rate=args.learning_rate,
        num_generations=args.num_generations,
        # TRL 1.0 defaults to one generation batch, which is invalid when
        # more than one completion is requested per prompt. Deriving this
        # from the generation count also scales with the global batch size
        # on multi-process runs.
        steps_per_generation=args.num_generations,
        # Custom validation below controls the detailed evaluation; keep the
        # trainer's own eval batch valid for small smoke-test batches.
        num_generations_eval=1,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
        beta=0.1,
        # GSPO is exposed by recent TRL through GRPOConfig rather than a
        # separate trainer: sequence-level importance ratios are the key change.
        **({"importance_sampling_level": "sequence"} if args.gspo else {}),
        loss_type=args.loss_type,
        logging_steps=10,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        gradient_checkpointing=True,
        bf16=supports_bf16,
        fp16=bool(use_cuda and not supports_bf16),
        report_to=args.report_to,
        run_name=args.wandb_run_name,
        seed=args.seed,
    )
    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=train,
        eval_dataset=evaluation,
        processing_class=tokenizer,
        reward_funcs=reward,
    )

    trainer.log({
        "constrained_triples_generated": triple_counter["count"],
        **_tool_counter_metrics(tool_counter, decoder_counter),
        "custom_metrics_ready": 0,
    })

    # Registered unconditionally: the callback also reports the cumulative tool
    # counters from on_log, which must work even without a validation split.
    metrics_callback = CustomMetricsCallback(
        tokenizer, evaluation, args.max_completion_length,
        args.fact_pattern, args.answer_pattern, args.eval_batch_size,
        args.custom_eval_steps, triple_counter, args.generation_output, model,
        tool_counter=tool_counter,
        decoder_counter=decoder_counter,
        reward_options=reward_options,
        train_reward_stats=train_reward_stats,
        train_group_stats=train_group_stats,
    )
    trainer.add_callback(metrics_callback)
    metrics_callback.trainer = trainer

    if evaluation is not None:
        initial_reward_stats = {}
        initial_metrics = evaluate_policy(
            model, tokenizer, evaluation, args.max_completion_length,
            args.fact_pattern, args.answer_pattern, args.eval_batch_size,
            generation_output=args.generation_output,
            evaluation_label="initial",
            decoder_counter=decoder_counter,
            reward_options=reward_options,
            reward_stats=initial_reward_stats,
        )
        initial_metrics["constrained_triples_generated"] = triple_counter["count"]
        initial_metrics.update(_tool_counter_metrics(tool_counter, decoder_counter))
        initial_metrics["custom_metrics_ready"] = 1
        trainer.log(initial_metrics)
        print("Initial validation metrics:", json.dumps(initial_metrics, sort_keys=True))

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    if evaluation is not None:
        final_reward_stats = {}
        eval_metrics = evaluate_policy(
            model, tokenizer, evaluation, args.max_completion_length,
            args.fact_pattern, args.answer_pattern, args.eval_batch_size,
            generation_output=args.generation_output,
            evaluation_label="final",
            decoder_counter=decoder_counter,
            reward_options=reward_options,
            reward_stats=final_reward_stats,
        )
        eval_metrics["constrained_triples_generated"] = triple_counter["count"]
        eval_metrics.update(_tool_counter_metrics(tool_counter, decoder_counter))
        trainer.log(eval_metrics)
        print("Validation metrics:", json.dumps(eval_metrics, sort_keys=True))

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    if args.index:
        print(
            "index routing totals: "
            f"calls={routing_stats['calls']} tables={dict(routing_stats['tables'])} "
            f"misses={routing_stats['misses']}"
        )


if __name__ == "__main__":
    main()
