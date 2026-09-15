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
    parser.add_argument("--index", default=None, help="Optional ReFactX prefix-tree index")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="Stop after this many optimizer steps")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--num-generations", type=int, default=None)
    parser.add_argument("--max-completion-length", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--report-to", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = {}
    if args.config:
        with open(args.config, encoding="utf-8") as config_file:
            config = json.load(config_file)

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
        "index": config.get("index"),
        "output_dir": config.get("output_dir", "./grpo-output"),
        "epochs": config.get("epochs", 1),
        "max_steps": config.get("max_steps"),
        "batch_size": config.get("batch_size", 1),
        "gradient_accumulation_steps": config.get("gradient_accumulation_steps", 4),
        "num_generations": config.get("num_generations", 4),
        "max_completion_length": config.get("max_completion_length", 256),
        "learning_rate": config.get("learning_rate", 5e-6),
        "report_to": config.get("report_to", "none"),
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


def _answer(text: str) -> str:
    text = re.split(r"<\|im_end\|>|<\|end_of_text\|>", text, maxsplit=1)[0]
    match = re.search(r"answer:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def _as_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(v).strip().lower() for v in value]
    return [str(value).strip().lower()]


def exact_reward(completions: Iterable[Any], answer: Iterable[Any], **_: Any) -> list[float]:
    """Score structure and answer overlap without loading a second model.

    The reward is deliberately made up of small, interpretable terms:

    * ``0.3`` for using at least one ``Fact:`` section;
    * ``0.3`` for having exactly one ``Answer:`` section;
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

        # Encourage the ReFactX answer format. These are soft incentives:
        # they do not require the model to produce a complete proof yet.
        if "fact:" in normalized:
            score += 0.3
        if normalized.count("answer:") == 1:
            score += 0.3

        try:
            # Only the text after Answer: is evaluated. This prevents facts
            # or reasoning from accidentally counting as predicted answers.
            predicted = json.loads(_answer(text))
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
    reward = judge if judge is not None else exact_reward

    if args.index:
        index = refactx.load_index(args.index, tokenizer=tokenizer)
        processor = refactx.get_constrained_logits_processor(
            tokenizer, index, num_beams=1, num_batches=args.batch_size * args.num_generations,
            return_list=True, avoid_duplicates=True,
        )
        original_generate = model.generate

        def constrained_generate(*call_args, **call_kwargs):
            call_kwargs.setdefault("logits_processor", processor)
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
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
