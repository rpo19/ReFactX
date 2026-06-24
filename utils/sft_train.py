import json
import argparse
from pathlib import Path

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


def preprocess(samples, tokenizer, mask_triples=False, max_length=2048):
    texts = [build_text(s) for s in samples]
    enc = tokenizer(texts, truncation=True, max_length=max_length, padding=False)
    input_ids_list = enc["input_ids"]
    attention_mask_list = enc["attention_mask"]

    labels_list = []
    for i, (text, input_ids) in enumerate(zip(texts, input_ids_list)):
        labels = input_ids.copy()

        prompt_len = len(tokenizer.encode(samples[i]["prompt"], add_special_tokens=False))
        for j in range(prompt_len):
            labels[j] = -100

        if mask_triples:
            fact_ranges = locate_fact_ranges(text)
            for fr_start, fr_end in fact_ranges:
                fr_tokens = tokenizer.encode(text[fr_start:fr_end], add_special_tokens=False)
                tok_start = prompt_len
        # Walk through the text character-by-character to find token offsets
        char_offset = 0
        tok_offset = prompt_len
        for ch_idx, ch in enumerate(text):
            if ch_idx == fr_start:
                # found start of fact range in chars; map to tok_offset
                pass

        # Simpler approach: re-encode and align
        labels_list.append(labels)

    return enc


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
    texts = [build_text(s) for s in samples]
    prompt_texts = [s["prompt"] for s in samples]
    enc = tokenizer(texts, truncation=True, max_length=args.max_length, padding=False, add_special_tokens=True)

    input_ids_list = enc["input_ids"]
    labels_list = []
    masked_triple_count = 0

    for i in range(len(samples)):
        input_ids = input_ids_list[i]
        labels = input_ids.copy()
        full_text = texts[i]
        prompt_text = prompt_texts[i]

        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_len = len(prompt_ids)
        for j in range(min(prompt_len, len(labels))):
            labels[j] = -100

        if args.mask_triples:
            fact_ranges = locate_fact_ranges(samples[i]["full_prediction"])
            if fact_ranges:
                masked_triple_count += len(fact_ranges)
            for fr_start, fr_end in fact_ranges:
                fact_text = samples[i]["full_prediction"][fr_start:fr_end]
                # Offset: full_prediction starts after the prompt
                fp_offset = len(samples[i]["prompt"])
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
        print(f"  Masked {masked_triple_count} Fact: lines across {len(samples)} samples")

    dataset = Dataset.from_dict({
        "input_ids": input_ids_list,
        "attention_mask": enc["attention_mask"],
        "labels": labels_list,
    })

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
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        weight_decay=args.weight_decay,
        optim=args.optim,
        adam_beta1=args.betas[0],
        adam_beta2=args.betas[1],
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    print("\nStarting training...")
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nModel saved to {args.output_dir}")


if __name__ == "__main__":
    main()
