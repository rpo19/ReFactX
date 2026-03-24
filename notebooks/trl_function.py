import sys
sys.settrace(None)

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer

import refactx

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_NAME = "rmanluo/RoG-cwq"

TEXT_COLUMN = "prompt"
LABEL_COLUMN = "answer"

MAX_LENGTH = 1024

dataset = load_dataset(DATASET_NAME)

train_dataset = dataset["train"]
eval_dataset = dataset.get("validation", None)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True,
    padding_side="right",
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def build_prompt(ds, tokenizer=tokenizer, prompt_template=None):
    if prompt_template is not None:
        prompt_str = refactx.apply_prompt_template(tokenizer, prompt_template, question=ds["question"])
    else:
        prompt_str = refactx.apply_prompt_template(tokenizer, question=ds["question"])
    return {"prompt": prompt_str}


train_dataset = train_dataset.map(build_prompt)
if eval_dataset is not None:
    eval_dataset = eval_dataset.map(build_prompt)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    quantization_config=bnb_config,
    device_map="cuda:0",
)

model.config.use_cache = False

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


def reward_fn(completions, prompts, **kwargs):
    rewards = []
    for completion, prompt in zip(completions, prompts):
        completion_lower = completion.strip().lower()
        reward = 0.0
        if len(completion_lower) > 10:
            reward += 0.1
        if "answer:" in completion_lower:
            reward += 0.5
        if "reasoning:" in completion_lower:
            reward += 0.3
        rewards.append(reward)
    return rewards


grpo_config = GRPOConfig(
    output_dir="./grpo-lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=1,
    num_generations=4,
    max_completion_length=512,
    beta=0.1,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    bf16=True,
    report_to="none",
)


trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    train_dataset=train_dataset,
    reward_funcs=reward_fn,
)

trainer.generation_kwargs

POSTGRES_URL = 'postgres://secondment:ofa3eebohgh6chioqu9Aep9maev6eejothith5bot4iuqu3oge7doo8uoCe0ooda@10.0.0.118:5432/postgres?tablename=trienewgpt'

index = refactx.load_index(POSTGRES_URL)

constrained_processor = refactx.get_constrained_logits_processor(
    tokenizer, index, num_beams=1, num_batches=1, return_list=True, avoid_duplicates=True
)

trainer.generation_kwargs = {
    "max_new_tokens": grpo_config.max_completion_length,
    "do_sample": True,
    "temperature": 1.0,
    "top_p": 0.95,
    "logits_processor": constrained_processor,
}

trainer.train()

trainer.model.save_pretrained("./grpo-lora-adapters")
tokenizer.save_pretrained("./grpo-lora-adapters")
