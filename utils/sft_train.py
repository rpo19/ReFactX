import json
import argparse
from pathlib import Path
from collections import defaultdict

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType


def load_jsonl(path):
    samples = []
    with open(path) as f:
        first = json.loads(f.readline())
        if first.get("_report"):
            pass
        else:
            samples.append(first)
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def build_text(sample):
    return sample["prompt"] + sample["full_prediction"]


def locate_fact_ranges(text):
    ranges = []
    start = 0
    while True:
        idx = text.find("Fact:", start)
        if idx == -1:
            break
        end = text.find("\n", idx)
        if end == -1:
            end = len(text)
        else:
            end += 1
        ranges.append((idx, end))
        start = end
    return ranges


def calculate_metrics(prediction, input_sample, answer_key="answer", lowercase=True):
    prediction_set = set()
    for p in prediction:
        p = str(p).strip()
        if lowercase:
            p = p.lower()
        if p:
            prediction_set.add(p)

    ground_truth = input_sample.get(answer_key, input_sample.get("gt_answer", ""))
    if isinstance(ground_truth, str):
        reference_set = {ground_truth.strip().lower() if lowercase else ground_truth.strip()}
    elif isinstance(ground_truth, list):
        reference_set = set()
        for gt in ground_truth:
            gt = str(gt).strip().lower() if lowercase else str(gt).strip()
            if gt:
                reference_set.add(gt)
    else:
        reference_set = {str(ground_truth).strip().lower() if lowercase else str(ground_truth).strip()}

    dont_know = not prediction_set or "i don't know" in prediction_set
    correct = 1 if prediction_set == reference_set else 0

    precision = len(prediction_set & reference_set) / len(prediction_set) if prediction_set else 0
    recall = len(prediction_set & reference_set) / len(reference_set) if reference_set else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    return precision, recall, f1, correct, dont_know


def stratified_train_val_split(samples, val_ratio, stratify_keys):
    groups = defaultdict(list)
    for i, s in enumerate(samples):
        key = tuple(str(s.get(k, "unknown")) for k in stratify_keys)
        groups[key].append(i)

    train_idx = []
    val_idx = []
    for key, indices in groups.items():
        n = len(indices)
        n_val = max(1, round(n * val_ratio))
        val_idx.extend(indices[:n_val])
        train_idx.extend(indices[n_val:])

    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    return train_samples, val_samples


def generate_answer(model, tokenizer, prompt, max_new_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def evaluate(model, tokenizer, val_samples, args):
    model.eval()
    metrics_by_type = defaultdict(lambda: {"precision": [], "recall": [], "f1": [], "correct": [], "dont_know": []})
    overall = {"precision": [], "recall": [], "f1": [], "correct": [], "dont_know": []}

    for s in val_samples:
        pred_text = generate_answer(model, tokenizer, s["prompt"])
        prediction = [pred_text]
        precision, recall, f1, correct, dont_know = calculate_metrics(prediction, s)

        overall["precision"].append(precision)
        overall["recall"].append(recall)
        overall["f1"].append(f1)
        overall["correct"].append(correct)
        overall["dont_know"].append(dont_know)

        qtype = s.get("complexityType", "unknown")
        origin = s.get("dataset", "unknown")
        for key in [qtype, origin]:
            metrics_by_type[key]["precision"].append(precision)
            metrics_by_type[key]["recall"].append(recall)
            metrics_by_type[key]["f1"].append(f1)
            metrics_by_type[key]["correct"].append(correct)
            metrics_by_type[key]["dont_know"].append(dont_know)

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    results = {
        "val_precision": avg(overall["precision"]),
        "val_recall": avg(overall["recall"]),
        "val_f1": avg(overall["f1"]),
        "val_accuracy": avg(overall["correct"]),
        "val_dont_know_rate": avg(overall["dont_know"]),
    }

    for group_name, group_metrics in sorted(metrics_by_type.items()):
        results[f"val_{group_name}_f1"] = avg(group_metrics["f1"])
        results[f"val_{group_name}_accuracy"] = avg(group_metrics["correct"])
        results[f"val_{group_name}_count"] = len(group_metrics["f1"])

    return results


def main():
    parser = argparse.ArgumentParser(description="SFT on positive training samples")
    parser.add_argument("--config", default=None, help="JSON config file (overrides CLI defaults)")
    parser.add_argument("--data", default="logs/positive_training_samples.jsonl", help="Input JSONL (filtered positive samples)")
    parser.add_argument("--output-dir", default="./sft_output", help="Output directory")
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-0.8B", help="Base model")
    parser.add_argument("--mask-triples", action="store_true", help="Mask loss on Fact: lines")
    parser.add_argument("--max-length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--epochs", type=float, default=1.0, help="Number of epochs")
    parser.add_argument("--max-steps", type=int, default=None, help="Max training steps (overrides epochs)")
    parser.add_argument("--save-steps", type=int, default=50, help="Save checkpoint every N steps")
    parser.add_argument("--logging-steps", type=int, default=10, help="Log every N steps")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay for optimizer")
    parser.add_argument("--optim", default="adamw_torch", help="Optimizer type")
    parser.add_argument("--betas", type=float, nargs=2, default=[0.9, 0.999], metavar=("BETA1", "BETA2"), help="Adam betas")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--report-to", default="wandb", help="Reporting destination (wandb, none)")
    parser.add_argument("--wandb-project", default="sft_training", help="Wandb project name")
    parser.add_argument("--wandb-run-name", default=None, help="Wandb run name")
    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
        for k, v in cfg.items():
            if hasattr(args, k) and v is not None:
                setattr(args, k, v)

    print(f"Loading data: {args.data}")
    samples = load_jsonl(args.data)
    print(f"  {len(samples)} samples loaded")

    if args.val_split > 0:
        train_samples, val_samples = stratified_train_val_split(
            samples, args.val_split, stratify_keys=["dataset", "complexityType"]
        )
        print(f"  Train: {len(train_samples)}, Val: {len(val_samples)} (stratified by dataset, complexityType)")
    else:
        train_samples = samples
        val_samples = []

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_dora=True,
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print("Tokenizing...")
    texts = [build_text(s) for s in train_samples]
    prompt_texts = [s["prompt"] for s in train_samples]
    enc = tokenizer(texts, truncation=True, max_length=args.max_length, padding=False, add_special_tokens=True)

    input_ids_list = enc["input_ids"]
    labels_list = []
    masked_triple_count = 0

    for i in range(len(train_samples)):
        input_ids = input_ids_list[i]
        labels = input_ids.copy()
        full_text = texts[i]
        prompt_text = prompt_texts[i]

        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_len = len(prompt_ids)
        for j in range(min(prompt_len, len(labels))):
            labels[j] = -100

        if args.mask_triples:
            fact_ranges = locate_fact_ranges(train_samples[i]["full_prediction"])
            if fact_ranges:
                masked_triple_count += len(fact_ranges)
            for fr_start, fr_end in fact_ranges:
                fact_text = train_samples[i]["full_prediction"][fr_start:fr_end]
                fp_offset = len(train_samples[i]["prompt"])
                global_start = fp_offset + fr_start
                global_end = fp_offset + fr_end
                prefix_ids = tokenizer.encode(full_text[:global_start], add_special_tokens=False)
                tok_start = len(prefix_ids)
                fact_ids = tokenizer.encode(fact_text, add_special_tokens=False)
                tok_end = tok_start + len(fact_ids)
                for j in range(tok_start, min(tok_end, len(labels))):
                    if labels[j] != -100:
                        labels[j] = -100

        labels_list.append(labels)

    if args.mask_triples:
        print(f"  Masked {masked_triple_count} Fact: lines across {len(train_samples)} samples")

    dataset = Dataset.from_dict({
        "input_ids": input_ids_list,
        "attention_mask": enc["attention_mask"],
        "labels": labels_list,
    })

    report_to = args.report_to if args.report_to != "none" else "none"

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs if args.max_steps is None else 1e10,
        max_steps=args.max_steps if args.max_steps is not None else 0,
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        bf16=True,
        gradient_checkpointing=True,
        report_to=report_to,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        weight_decay=args.weight_decay,
        optim=args.optim,
        adam_beta1=args.betas[0],
        adam_beta2=args.betas[1],
        run_name=args.wandb_run_name,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    print("\nStarting training...")
    train_result = trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nModel saved to {args.output_dir}")

    if val_samples:
        print(f"\nEvaluating on {len(val_samples)} validation samples...")
        val_metrics = evaluate(model, tokenizer, val_samples, args)

        print("\nValidation Results:")
        for k, v in sorted(val_metrics.items()):
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

        if args.report_to == "wandb":
            import wandb
            wandb.log(val_metrics)


if __name__ == "__main__":
    main()
