import sys

from flask import json
sys.settrace(None)

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
import wandb
import re

torch.autograd.set_detect_anomaly(True)

import refactx

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_NAME = "rmanluo/RoG-cwq"

PROMPT_TEMPLATE = [{'role': 'system', 'content': 'You are a helpful question-answering assistant that bases its answers on facts from a knowledge base.\n\n    You receive an input question.\n\n    You determine the reasoning path needed to answer.\n\n    You MUST get relevant facts with the "Fact:" command (e.g., "Fact: <Smith> <date of birth> <2000-10-01>"). You MUST rely on these facts and use them a proof for your answer.\n    While getting facts you continue the reasoning explaining it step by step.\n\n    You conclude with a concise answer that MUST be based on the proofs you found with "Fact:".\n\nExample answer format: `Answer: ["Italy", "United States", "France"]`.\n\nIf you didn\'t find proofs with "Fact:" that support an answer you stop and you reply: "I don\'t know.".\n\n'}]

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


def build_prompt(ds, tokenizer=tokenizer, prompt_template=PROMPT_TEMPLATE):
    if prompt_template is not None:
        prompt_str = refactx.apply_prompt_template(tokenizer, prompt_template, question=ds["question"])
    else:
        prompt_str = refactx.apply_prompt_template(tokenizer, question=ds["question"])
    return {"prompt": prompt_str}


train_dataset = train_dataset.map(build_prompt)
if eval_dataset is not None:
    eval_dataset = eval_dataset.map(build_prompt)

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
# )

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    # device_map="cuda:0",
    device_map="auto",
)

model.config.use_cache = False

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    use_dora=True,
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

#@click.option('--split-pattern', required=False, default=r'(<\|im_end\|>|<\|end_of_text\|>)', help='Pattern to split the full prediction. Use with --fix-predictions.')
answer_pattern = 'answer: '
split_pattern = tokenizer.eos_token
def get_answer(full_prediction, split_pattern, answer_pattern=answer_pattern):
    prediction = ''

    full_prediction = re.split(split_pattern, full_prediction, 1)[0]
    idx = full_prediction.index(answer_pattern)
    if idx != -1:
        prediction = full_prediction[idx + len(answer_pattern):].strip()

    return prediction

def reward_fn(completions, question, answer, **kwargs):
    ESTIMATED_NUM_WORDS=50

    rewards = []
    for i, (completion, question_i, answer_i) in enumerate(zip(completions, question, answer)):
        assert question_i in completion, f"Question not found in completion. Question: {question_i}, Completion: {completion}"
        completion_lower = completion.strip().lower()
        reward = 0.0
        if len(completion_lower) > 10:
            reward += 0.1
        if "fact:" in completion_lower:
            reward += 0.5
        answer_count = completion_lower.count("answer:")
        if answer_count == 1:
            reward += 0.3
        elif answer_count > 1:
            reward -= 0.2
        
        extracted_answer = get_answer(completion_lower, remove_dot=True)
        try:
            # answer should be a list of answers
            answer_json = json.loads(extracted_answer)
            reward += 0.5

            if answer_json:
                answer_json_lower = [ans.strip().lower() for ans in answer_json]
                ref = [r.strip().lower() for r in answer_i]

                # intersection over union
                intersection = set(answer_json_lower).intersection(set(ref))
                union = set(answer_json_lower).union(set(ref))
                iou = len(intersection) / len(union) if union else 0
                reward += iou

                # incentivate concise answers (if correct)
                if iou > 0.5:
                    word_count = len(completion_lower.split())
                    reward += ESTIMATED_NUM_WORDS / word_count

        except Exception as e:
            pass

        rewards.append(reward)
    return rewards


class GRPOTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        train_dataset,
        reward_funcs,
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
        bf16=True,
        generation_kwargs=None,
        mask_token_ids=None,
        mask_constrained_generation=True,
        logits_processor_list=[],
        eval_dataset=None,
        eval_steps=100,
        num_beams=1,
        use_wandb=False,
        wandb_project=None,
        wandb_run_name=None,
        eval_max_batches=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.reward_funcs = reward_funcs if isinstance(reward_funcs, list) else [reward_funcs]
        self.output_dir = output_dir
        self.per_device_train_batch_size = per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.num_generations = num_generations
        self.max_completion_length = max_completion_length
        self.beta = beta
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        self.bf16 = bf16
        self.eval_max_batches = eval_max_batches
        
        if self.use_wandb:
            wandb.init(
                project=self.wandb_project or "refactx_grpo",
                name=self.wandb_run_name or None,
                resume="allow"
            )
        self.generation_kwargs = generation_kwargs or {}
        self.mask_token_ids = mask_token_ids or []
        if tokenizer.pad_token_id is not None:
            self.mask_token_ids = list(set(self.mask_token_ids + [tokenizer.pad_token_id]))
        if tokenizer.eos_token_id is not None:
            self.mask_token_ids = list(set(self.mask_token_ids + [tokenizer.eos_token_id]))
        if tokenizer.bos_token_id is not None:
            self.mask_token_ids = list(set(self.mask_token_ids + [tokenizer.bos_token_id]))
        self.mask_constrained_generation = mask_constrained_generation
        self.global_step = 0
        self.generation_history = []
        self.logits_processor_list = logits_processor_list
        
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate
        )

    def generate(self, prompts):
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.model.device)

        gen_kwargs = {
            "max_new_tokens": self.max_completion_length,
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 0.95,
            "pad_token_id": self.tokenizer.pad_token_id,
            "logits_processor": self.logits_processor_list
        }
        gen_kwargs.update(self.generation_kwargs)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            prompts_len = inputs["input_ids"].shape[1]
            token_indices = [
                outputs[i, prompts_len:].cpu().tolist()
                for i in range(outputs.shape[0])
            ]

        prompts_len = inputs["input_ids"].shape[1]
        completions = [
            self.tokenizer.decode(output[prompts_len:], skip_special_tokens=True)
            for output in outputs
        ]
        return completions, token_indices


    def compute_log_probs(self, input_ids, attention_mask, completion_ids):
        logits = self.model(input_ids, attention_mask=attention_mask).logits
        log_probs = F.log_softmax(logits, dim=-1)
        
        batch_size = input_ids.shape[0]
        rewards = torch.zeros(batch_size, device=input_ids.device)
        
        for i in range(batch_size):
            prompt_len = (input_ids[i] == self.tokenizer.pad_token_id).sum().item()
            prompt_len = input_ids.shape[1] - prompt_len
            
            completion_len = completion_ids[i].shape[0]
            
            for j in range(prompt_len, input_ids.shape[1] - 1):
                token_id = input_ids[i, j + 1]
                rewards[i] += log_probs[i, j, token_id].item() / completion_len
        
        return rewards

    def train_step(self, prompts, completions, token_indices=None, references=None, mask_token_ids=None, mask_constrained_generation=True, refactx_generated_idx=None):
        # why tokenize again?
        # TODO generate should return tokens in this case
        prompts_tok = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.model.device)
        
        completions_tok = self.tokenizer(
            completions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_completion_length,
        ).to(self.model.device)
        
        prompts_len = (prompts_tok["input_ids"] != self.tokenizer.pad_token_id).sum(dim=1)
        
        input_ids = torch.cat([prompts_tok["input_ids"], completions_tok["input_ids"]], dim=1)
        attention_mask = torch.cat([prompts_tok["attention_mask"], completions_tok["attention_mask"]], dim=1)
        
        token_mask = torch.ones_like(input_ids, dtype=torch.bool)
        if mask_token_ids is not None:
            for mask_id in mask_token_ids:
                token_mask &= (input_ids != mask_id)

        if mask_constrained_generation:
            for i in range(len(refactx_generated_idx)):
                for idx in refactx_generated_idx[i]:
                    token_mask[i, idx] = False

        with torch.set_grad_enabled(True):
            # why calling the model again?
            # doing this only for last token but why?
            logits = self.model(input_ids, attention_mask=attention_mask).logits
            log_probs = F.log_softmax(logits, dim=-1)
            
            batch_size = input_ids.shape[0]
            per_token_logps = torch.zeros(batch_size, device=input_ids.device)
            valid_token_counts = torch.zeros(batch_size, device=input_ids.device)
            
            # calculating avg log per token
            for i in range(batch_size):
                start_idx = prompts_len[i].item()
                end_idx = input_ids.shape[1] - 1
                
                logp = 0.0
                count = 0
                
                for j in range(start_idx, end_idx):
                    if token_mask[i, j + 1]:
                        token_id = input_ids[i, j + 1]
                        logp = logp + log_probs[i, j, token_id]
                        count += 1
                
                if count > 0:
                    logp = logp / count
                
                per_token_logps[i] = logp
                valid_token_counts[i] = count
        
        rewards = reward_fn(completions, prompts, references=references)
        rewards_tensor = torch.tensor(rewards, device=per_token_logps.device)
        
        mean_reward = rewards_tensor.mean()
        std_reward = rewards_tensor.std()
        if std_reward > 0:
            normalized_rewards = (rewards_tensor - mean_reward) / std_reward
        else:
            normalized_rewards = rewards_tensor - mean_reward
        
        advantages = normalized_rewards
        
        loss = -(advantages * per_token_logps).mean()
        
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return loss.item(), mean_reward.item()

    def evaluate(self):
        if self.eval_dataset is None:
            return {}
        
        self.model.eval()
        eval_dataloader = DataLoader(self.eval_dataset, batch_size=self.per_device_train_batch_size, shuffle=False)
        
        all_rewards = []
        all_prompts = []
        all_completions = []
        all_references = []
        all_token_indices = []
        
        with torch.no_grad():
            batch_count = 0
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                if self.eval_max_batches is not None and batch_count >= self.eval_max_batches:
                    break
                prompts = batch["prompt"]
                references = batch["answer"]
                
                completions, token_idxs = self.generate(prompts)
                # TODO eventually save generated triples for assessing factuality in the reward fn
                self.logits_processor_list[0].states.reset()
                
                rewards = reward_fn(completions, prompts, references=references)
                all_rewards.extend(rewards)
                all_prompts.extend(prompts)
                all_completions.extend(completions)
                all_references.extend(references)
                all_token_indices.extend(token_idxs)
                batch_count += 1
        
        self.model.train()
        
        metrics = {
            "eval_reward_mean": sum(all_rewards) / len(all_rewards) if all_rewards else 0,
            "eval_samples": len(all_rewards),
        }
        
        return metrics

    def train(self):
        self.model.train()
        dataloader = DataLoader(self.train_dataset, batch_size=self.per_device_train_batch_size, shuffle=True)
        
        if self.eval_steps > 0 and self.eval_dataset is not None:
            print("Evaluating initial model...")
            eval_metrics = self.evaluate()
            for key, value in eval_metrics.items():
                print(f"  {key}: {value:.4f}")
            if self.use_wandb:
                print('logging to wandb')
                wandb.log({
                    "eval/reward_mean": eval_metrics.get("eval_reward_mean", 0),
                    "eval/samples": eval_metrics.get("eval_samples", 0),
                    "train/step": 0,
                })
        
        for epoch in range(self.num_train_epochs):
            epoch_pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
            for batch in epoch_pbar:
                prompts = batch["prompt"]
                answers = batch["answer"]
                
                all_completions = []
                all_token_indices = []
                refactx_generated_idx = []
                for i in range(self.num_generations):
                    completions, token_idxs = self.generate(prompts)
                    all_completions.extend(completions)
                    all_token_indices.extend(token_idxs)
                    refactx_generated_idx.append([])
                    # for each batch
                    for _, states_batch in enumerate(self.logits_processor_list[0].states):
                        assert len(states_batch) == 1, "Expected batch size of 1 (No beam search)"
                        refactx_generated_idx[i].append(copy.deepcopy(states_batch[0].generated_triples_idx))

                    self.logits_processor_list[0].states.reset()

                grouped_prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]
                grouped_answers = [answer for answer in answers for _ in range(self.num_generations)]
                
                loss, mean_reward = self.train_step(
                    grouped_prompts, all_completions, 
                    token_indices=all_token_indices,
                    references=grouped_answers, 
                    mask_token_ids=self.mask_token_ids,
                    mask_constrained_generation=self.mask_constrained_generation,
                    refactx_generated_idx=refactx_generated_idx,
                )
                
                for prompt, completion, token_idx, answer in zip(grouped_prompts, all_completions, all_token_indices, grouped_answers):
                    self.generation_history.append({
                        "prompt": prompt,
                        "completion": completion,
                        "token_indices": token_idx,
                        "reference": answer,
                    })
                
                self.global_step += 1
                
                if self.global_step % self.logging_steps == 0:
                    epoch_pbar.set_postfix({"loss": f"{loss:.4f}", "reward": f"{mean_reward:.4f}"})
                    if self.use_wandb:
                        print('logging to wandb')
                        wandb.log({
                            "train/loss": loss,
                            "train/reward": mean_reward,
                            "train/step": self.global_step,
                        })
                
                if self.global_step % self.save_steps == 0:
                    self.model.save_pretrained(f"{self.output_dir}/checkpoint-{self.global_step}")
                    self.tokenizer.save_pretrained(f"{self.output_dir}/checkpoint-{self.global_step}")
                    torch.save(
                        {"history": self.generation_history},
                        f"{self.output_dir}/generation_history-{self.global_step}.pt"
                    )
                
                if self.eval_steps > 0 and self.global_step % self.eval_steps == 0:
                    eval_metrics = self.evaluate()
                    print(f"Step {self.global_step} evaluation:")
                    for key, value in eval_metrics.items():
                        print(f"  {key}: {value:.4f}")
                    epoch_pbar.set_postfix(eval_metrics)
                    if self.use_wandb:
                        wandb.log({
                            "eval/reward_mean": eval_metrics.get("eval_reward_mean", 0),
                            "eval/samples": eval_metrics.get("eval_samples", 0),
                            "train/step": self.global_step,
                        })


NUM_BATCHES = 1
NUM_BEAMS = 1

grpo_config = {
    "output_dir": "./grpo-lora",
    "per_device_train_batch_size": NUM_BATCHES,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "num_train_epochs": 1,
    "num_generations": 4,
    "max_completion_length": 512,
    "beta": 0.1,
    "logging_steps": 1,
    "save_steps": 50,
    "eval_steps": 50,
    "use_wandb": True,
    "wandb_project": None,
    "wandb_run_name": None,
    "eval_max_batches": 10,
}



POSTGRES_URL = 'postgres://secondment:ofa3eebohgh6chioqu9Aep9maev6eejothith5bot4iuqu3oge7doo8uoCe0ooda@10.0.0.118:5432/postgres?tablename=trienewqwen'

index = refactx.load_index(POSTGRES_URL)

constrained_processor = refactx.get_constrained_logits_processor(
    tokenizer, index, num_beams=NUM_BEAMS, num_batches=NUM_BATCHES, return_list=True, avoid_duplicates=True
)

trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    reward_funcs=reward_fn,
    eval_dataset=eval_dataset,
    mask_token_ids=[],
    mask_constrained_generation=True,
    logits_processor_list = constrained_processor,
    num_beams=1,
    **grpo_config
)

print(f"Wandb enabled: {trainer.use_wandb}, project: {trainer.wandb_project or 'refactx_grpo'}")

trainer.train()

trainer.model.save_pretrained("./grpo-lora-adapters")
tokenizer.save_pretrained("./grpo-lora-adapters")
torch.save(
    {"history": trainer.generation_history},
    "./grpo-lora-adapters/generation_history.pt"
)
