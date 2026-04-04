import json
import os
from shutil import copy
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig, get_peft_model
import copy
import re

import wandb

wandb.init(project="trl", name="trl_test2_qwen_refactx_masking")

import refactx

from dotenv import load_dotenv
load_dotenv()




MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_NAME = "rmanluo/RoG-cwq"

PROMPT_TEMPLATE = [{'role': 'system', 'content': 'You are a helpful question-answering assistant that bases its answers on facts from a knowledge base.\n\n    You receive an input question.\n\n    You determine the reasoning path needed to answer.\n\n    You MUST get relevant facts with the "Fact:" command (e.g., "Fact: <Smith> <date of birth> <2000-10-01>"). You MUST rely on these facts and use them a proof for your answer.\n    While getting facts you continue the reasoning explaining it step by step.\n\n    You conclude with a concise answer that MUST be based on the proofs you found with "Fact:".\n\nExample answer format: `Answer: ["Italy", "United States", "France"]`.\n\nIf you didn\'t find proofs with "Fact:" that support an answer you stop and you reply: Answer: ["I don\'t know."].\n\n'}]


TEXT_COLUMN = "prompt"
LABEL_COLUMN = "answer"

MAX_LENGTH = 1024

# -----------------------
# Load model + tokenizer
# -----------------------

def patch_generate(model, processor):

    original_generate = model.generate

    def new_generate(*args, **kwargs):
        kwargs["logits_processor"] = processor

        return original_generate(*args, **kwargs)

    model.generate = new_generate
    return model


model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto"
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
# -----------------------
# Reward function
# -----------------------
#@click.option('--split-pattern', required=False, default=r'(<\|im_end\|>|<\|end_of_text\|>)', help='Pattern to split the full prediction. Use with --fix-predictions.')
answer_pattern = re.compile(r'answer: (.*)\.?')
split_pattern = tokenizer.eos_token
def get_answer(full_prediction, remove_dot=True, answer_pattern=answer_pattern, split_pattern=split_pattern):
    prediction = ''

    full_prediction = re.split(split_pattern, full_prediction, maxsplit=1)[0]
    if remove_dot and full_prediction.endswith('.'):
        full_prediction = full_prediction[:-len('.')]
    match = answer_pattern.search(full_prediction)
    if match:
        prediction = match.group(1)

    return prediction

def reward_fn(completions, question, answer, references=None, log_extra=None, log_metric=None, **kwargs):
    ESTIMATED_NUM_WORDS=50

    rewards = []
    tp = 0
    for i, (completion, question_i, answer_i) in enumerate(zip(completions, question, answer)):
        assert question_i in completion, f"Question not found in completion. Question: {question_i}, Completion: {completion}"

        completion_lower = completion.strip().lower()
        reward = 0.0
        if len(completion_lower) > 10:
            reward += 0.1
        # TODO fact should be before answer
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
                if iou >= 0.9:
                    tp += 1
                    word_count = len(completion_lower.split())
                    reward += ESTIMATED_NUM_WORDS / word_count

        except Exception as e:
            print('Warning: Exception in reward_fn: {}'.format(e))
            pass

        rewards.append(reward)

        accuracy = tp / len(completions) if completions else 0

    if log_extra:
        for i, item in enumerate(completions):
            if 'fact:' in item.lower():
                to_log = 'completion', [item, references[i] if references is not None else None]
                break
        log_extra(*to_log)
        print('log_extra', *to_log)
    if log_metric:
        log_metric('accuracy', accuracy)
        log_metric('fact_calls', completion_lower.count("fact:"))
    return rewards


# refactx

POSTGRES_URL = os.environ.get("POSTGRES_URL")
index = refactx.load_index(POSTGRES_URL)


constrained_processor = refactx.get_constrained_logits_processor(
    tokenizer, index, num_beams=1, num_batches=4, return_list=True, avoid_duplicates=True, reinit_states=True
)

# -----------------------
# GRPO config
# -----------------------
config = GRPOConfig(
    output_dir="./grpo-qwen",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=1,

    learning_rate=5e-6,

    # GRPO-specific
    num_generations=4,   # samples per prompt batch_size?? TODO make dynamic in refactx?
    max_completion_length=256,

    logging_steps=10,
    save_steps=200,

    # generation_kwargs = {
    #     'logits_processor': constrained_processor
    # } # not working here

    report_to='wandb',
)

patch_generate(model, constrained_processor)


class ReFactXGRPOTrainer(GRPOTrainer):

    def __init__(self, mask_constrained_generation=True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.get_state_idx = 0

        self.mask_constrained_generation = mask_constrained_generation

    def _get_next_state(self):
        states = refactx.get_constrained_states().states
        if self.get_state_idx >= len(states):
            self.get_state_idx = 0
        state = states[self.get_state_idx]
        self.get_state_idx += 1
        return state

    def get_mask(self, completion_ids):
        """
        (Pdb) p completion_mask
tensor([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], device='cuda:0')
(Pdb) p completion_mask.shape
torch.Size([2, 64])

(Pdb) p inputs["tool_mask"]
[[[], [], [], [], [], [], [], []], [[], [], [], [], [], [], [], []], [[], [], [], [], [], [], [], []], [[], [], [], [], [], [], [], []]]
4 x 8 empty??? maybe no token has been generated with refactx 

--- second test with current config
completion_mask
torch.Size([4, 247]) # 247 could be the maximum length of the generation?


inputs['tool_mask'] shape # TODO make tool mask a 1/0 matrix as completion_mask
(4, 32)




        """
        refactx_generated_idx = []
        for i in range(self.num_generations):
            refactx_generated_idx.append([])
            state = self._get_next_state()
            refactx_generated_idx[i].append(copy.deepcopy(state[0].generated_triples_idx))
        # TODO need to change mask and also to keep the masking somewhere until it is used
        # TODO probably need to use a global variable to fill and empty
        # refactx_generated_idx[i] contains the indexes of the tokens generated with constrained gen
        mask = torch.ones(completion_ids.shape, dtype=completion_ids.dtype, device=completion_ids.device)
        assert mask.shape[0] == len(refactx_generated_idx), f"Mask batch size {mask.shape[0]} does not match number of generations {len(refactx_generated_idx)}"
        for i, indexes in enumerate(refactx_generated_idx):
            mask[i, indexes] = 0
        return mask

    def _compute_loss(self, model, inputs):
        if self.mask_constrained_generation:
            mask = self.get_mask(inputs['completion_ids'])
            assert 'tool_mask' not in inputs, 'Error: tool_mask already exists.'
            inputs['tool_mask'] = mask
        # call the original method
        return super()._compute_loss(model, inputs)

    def compute_liger_loss(self, unwrapped_model, inputs):
        if self.mask_constrained_generation:
            mask = self.get_mask()
            assert 'tool_mask' not in inputs, 'Error: tool_mask already exists.'
            inputs['tool_mask'] = mask
        # call the original method
        return super().compute_liger_loss(unwrapped_model, inputs)

# -----------------------
# Trainer
# -----------------------
trainer = ReFactXGRPOTrainer(
    model=model,
    # tokenizer=tokenizer,
    args=config,
    train_dataset=train_dataset,

    reward_funcs=reward_fn,
)

# -----------------------
# Train
# -----------------------
trainer.train()


# TODO refactx states should reinit automatically now
# TODO getting triples generated with constrained gen and masking - cannot be automated - probably requires modifying trl. we can use loss_mask in liger_grpo_loss or mask in _compute_loss 
# https://github.com/huggingface/trl/blob/1ad25f94e5556b2deb4869e11f62d3f91aa7bbac/trl/trainer/grpo_trainer.py#L2265 # i can alter _copute loss and inject tool_mask in the input. same for liger loss