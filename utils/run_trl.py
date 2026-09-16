"""Train a causal LM with TRL's GRPO trainer.

The default reward is deterministic and uses the answer column.  Pass
``--judge-model MODEL`` to replace it with an LLM-as-a-judge reward.

Examples (run from the repository root)::

    python utils/run_trl.py --config configs/trl_qwen35_08b_smoke.json
    python utils/run_trl.py --config configs/trl_qwen35_08b_smoke.json --max-steps 1
    python utils/run_trl.py --config configs/trl_qwen35_08b_smoke.json --judge-model Qwen/Qwen2.5-3B-Instruct
    python utils/run_trl.py --config configs/trl_qwen35_08b_smoke.json --adapter ./grpo-qwen35-08b

The script intentionally does not initialise wandb.  Select it with
``--report-to wandb`` when desired.
"""

from __future__ import annotations

import argparse
import functools
import json
import re
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="JSON config file; CLI options override config values")
    parser.add_argument("--model", default=None, help="Base model name or path")
    parser.add_argument("--adapter", default=None, help="Optional LoRA adapter to load on top of --model")
    parser.add_argument("--judge-model", default=None, help="Optional model used as an LLM judge")
    parser.add_argument("--gspo", action="store_true", default=None, help="Use GSPO sequence-level importance sampling")
    parser.add_argument("--dataset", default=None)
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
    parser.add_argument("--index", default=None, help="Optional ReFactX prefix-tree index")
    parser.add_argument("--output-dir", default=None)
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
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--report-to", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

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
        "question_key": config.get("question_key", "question"),
        "answer_key": config.get("answer_key", "answer"),
        "train_split": config.get("train_split", "train"),
        "eval_split": config.get("eval_split", "validation"),
        "max_eval_samples": config.get("max_eval_samples", 100),
        "index": config.get("index"),
        "output_dir": config.get("output_dir", "./grpo-output"),
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
        "learning_rate": config.get("learning_rate", 5e-6),
        "report_to": config.get(
            "report_to", "wandb" if config.get("wandb", False) else "none"
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
) -> dict[str, float]:
    """Evaluate deterministic completions against the dataset answer column.

    Accuracy is exact set match, while the reward remains useful for tracking
    partial answer overlap and formatting progress. Validation deliberately
    uses greedy decoding so the metric is not affected by sampling noise.
    """
    if dataset is None or len(dataset) == 0:
        return {}

    import torch

    model.eval()
    rewards = []
    exact_matches = 0
    valid_answers = 0
    formatted = 0
    total = 0

    for example in dataset:
        prompt = example["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_completion_length,
                do_sample=False,
                num_beams=1,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(
            output[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        )
        reference = example["answer"]
        parsed = _parsed_answer(completion, answer_pattern)
        reference_values = set(_as_list(reference))
        if parsed is not None:
            valid_answers += 1
            if set(parsed) == reference_values:
                exact_matches += 1
        normalized_completion = completion.lower()
        if (
            _marker_count(normalized_completion, fact_pattern) > 0
            and _marker_count(normalized_completion, answer_pattern) == 1
        ):
            formatted += 1
        rewards.extend(
            exact_reward(
                [completion], [reference],
                fact_pattern=fact_pattern,
                answer_pattern=answer_pattern,
            )
        )
        total += 1

    return {
        "eval_answer_accuracy": exact_matches / total,
        "eval_valid_answer_rate": valid_answers / total,
        "eval_format_accuracy": formatted / total,
        "eval_mean_reward": sum(rewards) / total,
    }


class JudgeReward:
    """Callable reward function backed by a frozen causal language model."""

    def __init__(self, model_name: str, torch_module: Any, transformers_module: Any):
        torch = torch_module
        AutoModelForCausalLM = transformers_module.AutoModelForCausalLM
        AutoTokenizer = transformers_module.AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
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
    from datasets import load_dataset
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer
    import refactx

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
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
            "prompt": refactx.apply_prompt_template(tokenizer, question=question),
            "question": question,
            "answer": example[args.answer_key],
        }

    train = train.map(format_example)
    if evaluation is not None:
        evaluation = evaluation.map(format_example)

    judge = JudgeReward(args.judge_model, torch, type("Transformers", (), {
        "AutoTokenizer": AutoTokenizer, "AutoModelForCausalLM": AutoModelForCausalLM,
    })) if args.judge_model else None
    reward = judge if judge is not None else functools.partial(
        exact_reward,
        fact_pattern=args.fact_pattern,
        answer_pattern=args.answer_pattern,
    )

    if args.index:
        index = refactx.load_index(args.index, tokenizer=tokenizer)
        processor = refactx.get_constrained_logits_processor(
            tokenizer, index, num_beams=1,
            num_batches=args.batch_size * args.num_generations,
            fact_pattern=args.fact_pattern,
            return_list=True,
            avoid_duplicates=True,
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
            if batch_size == 1:
                active_processor = refactx.get_constrained_logits_processor(
                    tokenizer, index, num_beams=1, num_batches=1,
                    fact_pattern=args.fact_pattern,
                    return_list=True,
                    avoid_duplicates=True,
                )
            call_kwargs.setdefault("logits_processor", active_processor)
            return original_generate(*call_args, **call_kwargs)

        model.generate = constrained_generate

    config = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps is not None else -1,
        learning_rate=args.learning_rate,
        num_generations=args.num_generations,
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
        save_total_limit=2,
        gradient_checkpointing=True,
        bf16=torch.cuda.is_available(),
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
    trainer.train()

    if evaluation is not None:
        eval_metrics = evaluate_policy(
            model, tokenizer, evaluation, args.max_completion_length,
            args.fact_pattern, args.answer_pattern,
        )
        trainer.log(eval_metrics)
        print("Validation metrics:", json.dumps(eval_metrics, sort_keys=True))

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
