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
import json
import os
import re
from typing import Any, Iterable

from dotenv import load_dotenv


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
    parser.add_argument("--no-cuda", action="store_true", help="Allow running without CUDA (default: require CUDA)")
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
        "--sentinel",
        action="store_true",
        default=None,
        help="Enable the exhausted-retrieval '<no further records>' sentinel during generation",
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
        "model": config.get("model", config.get("model_name", "Qwen/Qwen2.5-0.5B-Instruct")),
        "adapter": config.get("adapter"),
        "judge_model": config.get("judge_model"),
        "gspo": config.get("gspo", False),
        "dataset": config.get("dataset", "rmanluo/RoG-cwq"),
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
        "learning_rate": config.get("learning_rate", 5e-6),
        # Default to wandb; an explicit "report_to" still wins, and an
        # explicit "wandb": false opts out.
        "report_to": config.get(
            "report_to", "none" if config.get("wandb", None) is False else "wandb"
        ),
        "seed": config.get("seed", 42),
    }
    for name, default in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, default)
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


def _parsed_answer(text: str, answer_pattern: str = "<answer>") -> list[str] | None:
    try:
        return _as_list(json.loads(_answer(text, answer_pattern)))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


COUNT_MARKER = "<count>"
SENTINEL_MARKER = "no further records"


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


def exact_reward(
    completions: Iterable[Any],
    answer: Iterable[Any],
    fact_pattern: str = "<fact>",
    answer_pattern: str = "<answer>",
    **_: Any,
) -> list[float]:
    """Score structure and answer overlap without loading a second model.

    The reward is deliberately made up of small, interpretable terms:

    * ``0.3`` for using at least one configured fact section;
    * ``0.3`` for having exactly one configured answer section;
    * ``0.5`` for producing a JSON answer;
    * ``0..1`` for answer-set intersection-over-union (IoU).

    A malformed answer still receives the formatting points, but no answer
    correctness points.  This gives the policy a learning signal before it
    consistently emits valid JSON.
    """
    rewards = []
    for completion, reference in zip(completions, answer):
        text = _text(completion)
        normalized = text.lower()
        score = 0.0

        # Encourage the configured output markers. These are soft incentives
        # and do not require a complete proof.
        if _marker_count(normalized, fact_pattern) > 0:
            score += 0.3
        if _marker_count(normalized, answer_pattern) == 1:
            score += 0.3

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
            iou = len(intersection) / len(union) if union else 0.0
            score += 0.5 + iou
        except (TypeError, ValueError, json.JSONDecodeError):
            # Invalid JSON is common early in RL training. Do not make the
            # whole batch fail; keep the structural reward accumulated above.
            pass

        rewards.append(score)
    return rewards


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
        with torch.no_grad():
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
            sample_reward = exact_reward(
                [completion], [reference],
                fact_pattern=fact_pattern,
                answer_pattern=answer_pattern,
            )[0]
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

    return {
        "eval_answer_accuracy": exact_matches / total,
        "eval_valid_answer_rate": valid_answers / total,
        "eval_format_accuracy": formatted / total,
        "eval_mean_reward": sum(rewards) / total,
        "eval_count_tool_uses": count_tool_uses,
        "eval_count_tool_rate": count_tool_examples / total,
        "eval_count_tool_mean": count_tool_uses / total,
        "eval_sentinel_uses": sentinel_uses,
        "eval_sentinel_rate": sentinel_examples / total,
        "eval_sentinel_mean": sentinel_uses / total,
        **({
            "eval_count_tool_calls": decoder_counter["count_calls"] - decoder_before["count_calls"],
            "eval_sentinel_calls": decoder_counter["sentinel_calls"] - decoder_before["sentinel_calls"],
        } if decoder_before is not None and decoder_counter is not None else {}),
    }


class CustomMetricsCallback:
    """Log deterministic validation metrics during long GRPO runs."""

    def __init__(self, tokenizer, dataset, max_completion_length, fact_pattern,
                 answer_pattern, eval_batch_size, every_steps, triple_counter,
                 generation_output, model, tool_counter=None, decoder_counter=None):
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
        self.trainer = None
        self.pending_metrics = None

    def __getattr__(self, name):
        # TrainerCallbackHandler invokes every lifecycle event directly.
        # This callback only needs on_step_end and on_log; all other events
        # leave the trainer control object unchanged.
        if name.startswith("on_"):
            return lambda args, state, control, **kwargs: control
        raise AttributeError(name)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        model = self.model if self.model is not None else model
        # Without a validation dataset there is nothing to evaluate; the
        # cumulative tool counters are still reported from on_log.
        if (
            self.dataset is None
            or len(self.dataset) == 0
            or self.every_steps <= 0
            or state.global_step == 0
            or state.global_step % self.every_steps != 0
            or model is None
        ):
            return control
        was_training = model.training
        self.pending_metrics = evaluate_policy(
            model, self.tokenizer, self.dataset, self.max_completion_length,
            self.fact_pattern, self.answer_pattern, self.eval_batch_size,
            generation_output=self.generation_output,
            evaluation_label=f"step_{state.global_step}",
            decoder_counter=self.decoder_counter,
        )
        self.pending_metrics["constrained_triples_generated"] = self.triple_counter["count"]
        self._add_tool_counters(self.pending_metrics)
        if self.trainer is not None:
            self.trainer.log(self.pending_metrics)
            self.pending_metrics = None
        if was_training:
            model.train()
        return control

    def _add_tool_counters(self, metrics):
        """Attach cumulative constrained-tool usage counters to a metrics dict."""
        metrics["count_tool_uses"] = self.tool_counter.get("count_uses", 0)
        metrics["count_tool_examples"] = self.tool_counter.get("count_examples", 0)
        metrics["sentinel_uses"] = self.tool_counter.get("sentinel_uses", 0)
        metrics["sentinel_examples"] = self.tool_counter.get("sentinel_examples", 0)
        metrics["count_tool_calls"] = self.decoder_counter.get("count_calls", 0)
        metrics["sentinel_calls"] = self.decoder_counter.get("sentinel_calls", 0)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            logs["constrained_triples_generated"] = self.triple_counter["count"]
            self._add_tool_counters(logs)
        if logs is not None and self.pending_metrics:
            logs.update(self.pending_metrics)
            self.pending_metrics = None
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

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16 if use_cuda else torch.float32,
        device_map="auto",
    )
    model.config.use_cache = False
    if args.adapter:
        # Load the existing adapter as the trainable policy. The base model
        # must match the one recorded in adapter_config.json.
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=8, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ))
    model.print_trainable_parameters()

    raw = load_dataset(args.dataset)
    train = raw[args.train_split]
    evaluation = raw.get(args.eval_split, None)
    if evaluation is not None and args.max_eval_samples is not None:
        evaluation = evaluation.select(
            range(min(args.max_eval_samples, len(evaluation)))
        )

    def format_example(example):
        question = example[args.question_key]
        return {
            "prompt": refactx.apply_prompt_template(
                tokenizer, prompt_template=prompt_template, question=question
            ),
            "question": question,
            "answer": example[args.answer_key],
        }

    train = train.map(format_example)
    if evaluation is not None:
        evaluation = evaluation.map(format_example)

    judge = JudgeReward(args.judge_model, torch, type("Transformers", (), {
        "AutoTokenizer": AutoTokenizer, "AutoModelForCausalLM": AutoModelForCausalLM,
    }), use_cuda=use_cuda) if args.judge_model else None
    # Keep a named function here: TRL 1.0 records reward names from
    # ``__name__`` and does not handle callable objects or partials.
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
        if judge is not None:
            return judge(completions, question=question, **kwargs)
        return exact_reward(
            completions,
            answer,
            fact_pattern=args.fact_pattern,
            answer_pattern=args.answer_pattern,
            **kwargs,
        )

    def on_triple_generated(_sequence):
        triple_counter["count"] += 1

    if args.index:
        index = refactx.load_index(
            args.index, tokenizer=tokenizer, tablename=args.tablename
        )
        processor = refactx.get_constrained_logits_processor(
            tokenizer, index, num_beams=1,
            num_batches=args.batch_size * args.num_generations,
            fact_pattern=args.fact_pattern,
            count_pattern=args.count_pattern,
            sentinel=args.sentinel,
            return_list=True,
            avoid_duplicates=True,
            reinit_states=True,
            on_triple_generated=on_triple_generated,
            on_count_generated=on_count_generated,
            on_sentinel_generated=on_sentinel_generated,
        )
        original_generate = model.generate

        def constrained_generate(*call_args, **call_kwargs):
            # GRPO generation uses the expanded training batch, while the
            # validation pass generates one prompt at a time. The constrained
            # processor must be initialized for the actual generation batch.
            input_ids = call_kwargs.get("input_ids")
            if input_ids is None and call_args:
                input_ids = call_args[0]
            batch_size = input_ids.shape[0] if input_ids is not None else None
            active_processor = processor
            if batch_size is not None and batch_size != args.batch_size * args.num_generations:
                active_processor = refactx.get_constrained_logits_processor(
                    tokenizer, index, num_beams=1, num_batches=batch_size,
                    fact_pattern=args.fact_pattern,
                    count_pattern=args.count_pattern,
                    sentinel=args.sentinel,
                    return_list=True,
                    avoid_duplicates=True,
                    reinit_states=True,
                    on_triple_generated=on_triple_generated,
                    on_count_generated=on_count_generated,
                    on_sentinel_generated=on_sentinel_generated,
                )
            call_kwargs.setdefault("logits_processor", active_processor)
            return original_generate(*call_args, **call_kwargs)

        model.generate = constrained_generate

    config = GRPOConfig(
        output_dir=args.output_dir,
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
        logging_steps=10,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        gradient_checkpointing=True,
        bf16=use_cuda,
        report_to=args.report_to,
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
        "count_tool_uses": tool_counter["count_uses"],
        "sentinel_uses": tool_counter["sentinel_uses"],
        "count_tool_calls": decoder_counter["count_calls"],
        "sentinel_calls": decoder_counter["sentinel_calls"],
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
    )
    trainer.add_callback(metrics_callback)
    metrics_callback.trainer = trainer

    if evaluation is not None:
        initial_metrics = evaluate_policy(
            model, tokenizer, evaluation, args.max_completion_length,
            args.fact_pattern, args.answer_pattern, args.eval_batch_size,
            generation_output=args.generation_output,
            evaluation_label="initial",
            decoder_counter=decoder_counter,
        )
        initial_metrics["constrained_triples_generated"] = triple_counter["count"]
        initial_metrics["count_tool_uses"] = tool_counter["count_uses"]
        initial_metrics["sentinel_uses"] = tool_counter["sentinel_uses"]
        initial_metrics["count_tool_calls"] = decoder_counter["count_calls"]
        initial_metrics["sentinel_calls"] = decoder_counter["sentinel_calls"]
        initial_metrics["custom_metrics_ready"] = 1
        trainer.log(initial_metrics)
        print("Initial validation metrics:", json.dumps(initial_metrics, sort_keys=True))

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    if evaluation is not None:
        eval_metrics = evaluate_policy(
            model, tokenizer, evaluation, args.max_completion_length,
            args.fact_pattern, args.answer_pattern, args.eval_batch_size,
            generation_output=args.generation_output,
            evaluation_label="final",
            decoder_counter=decoder_counter,
        )
        eval_metrics["constrained_triples_generated"] = triple_counter["count"]
        eval_metrics["count_tool_uses"] = tool_counter["count_uses"]
        eval_metrics["sentinel_uses"] = tool_counter["sentinel_uses"]
        eval_metrics["count_tool_calls"] = decoder_counter["count_calls"]
        eval_metrics["sentinel_calls"] = decoder_counter["sentinel_calls"]
        trainer.log(eval_metrics)
        print("Validation metrics:", json.dumps(eval_metrics, sort_keys=True))

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
