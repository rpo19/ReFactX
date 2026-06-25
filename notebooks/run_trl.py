# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: Python (trl rpozzi)
#     language: python
#     name: trl
# ---

# %%
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig
)
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer

import refactx

# %%
#MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_NAME = "rmanluo/RoG-cwq"

TEXT_COLUMN = "prompt"
LABEL_COLUMN = "answer"

MAX_LENGTH = 1024


# %%
'''base_prompt = [
    {
        "role": "system",
        "content": "You are a helpful question-answering assistant that answers strictly using facts from a knowledge base and always follows this prompt exactly.\n\nIMPORTANT CONSTRAINTS:\n- Do NOT call any tools or functions.\n- Do NOT use JSON or any structured data format in your answers.\n- Output must be plain text only.\n- Do NOT invent new formats.\n\nThe process to answer questions:\n\n1. Read the input question.\n2. Determine the reasoning path needed to answer the question.\n3. Determine the expected answer type: yes/no, single entity, or list of entities. If a list is required, include all valid entities.\n4. Retrieve relevant facts ONLY by explicitly writing them as plain text lines starting with `Fact:`.\nExplain your reasoning step by step while presenting the facts.\n6. Conclude with a concise final answer written on a single line starting with `Answer:`.\n\nOUTPUT FORMAT (MANDATORY AND UNCHANGING):\n\nReasoning: <free text explanation>\nFact: <subject> <property> <value> .\nFact: <subject> <property> <value> .\n...\nAnswer: <final answer>\n\nRules:\n- The final answer MUST be supported by facts written with `Fact:`.\n NEVER look in your memory to answer.\n- If no supporting facts are found, reply exactly: `Answer: I don't know.`\n- If the question asks whether an event happened and no proof is found, assume it did not happen.\n- If reasoning becomes unproductive, stop and answer using the facts collected so far.\n- NEVER change the output structure.\n- NEVER introduce JSON, YAML, XML, or tool calls."
    },
    {
        "role": "user",
        "content": "When was the director of Slumdog Millionaire born?"
    },
    {
        "role": "assistant",
        "content": "Reasoning: To answer this question, I need to identify the director of Slumdog Millionaire and then find that person's date of birth.\n\nFact: <Slumdog Millionaire> <description> <2008 film directed by Danny Boyle> .\nFact: <Danny Boyle> <date of birth> <1956-10-20T00:00:00Z> .\n\nAnswer: October 20, 1956.\n\n"
    },
    {
        "role": "user",
        "content": "During which years did Napoleon hold power in France?"
    },
    {
        "role": "assistant",
        "content": "Reasoning: To answer this question, I need to identify the period during which Napoleon held power in France.\n\nFact: <Napoleon Bonaparte> <description> <French military leader, French Emperor 1804-1814 and again in 1815> .\n\nAnswer: From 1804 to 1814, and again in 1815.\n\n"
    }
]'''

# %%
dataset = load_dataset(DATASET_NAME)

train_dataset = dataset["train"]
eval_dataset = dataset.get("validation", None)


# %%
dataset

# %%
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True,
    padding_side="right",
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# %%
def build_prompt(ds, tokenizer=tokenizer,prompt_template=None):
    if prompt_template is not None:
        prompt_str = refactx.apply_prompt_template(tokenizer, prompt_template, question=ds["question"])
    else:
        prompt_str = refactx.apply_prompt_template(tokenizer, question=ds["question"])
    return {"prompt": prompt_str}

train_dataset = train_dataset.map(build_prompt)
if eval_dataset is not None:
    eval_dataset = eval_dataset.map(build_prompt)

# %%
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
# )

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    # quantization_config=bnb_config,
    device_map="cuda:0",
)

# for training
model.config.use_cache = False


# %%
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


# %%
# def reward_fn(prompts, generations, references, **kwargs):
#     rewards = []

#     for gen, ref in zip(generations, references):
#         gen = gen.strip().lower()
#         ref = ref.strip().lower()

#         reward = 0.0

#         if ref in gen:
#             reward += 1.0

#         # discourage empty / extremely short answers
#         reward -= 0.001 * abs(len(gen) - len(ref))

#         rewards.append(reward)

#     return rewards


# %%
# trainer = GRPOTrainer(
#     model=model,
#     tokenizer=tokenizer,
#     args=grpo_config,
#     train_dataset=train_dataset,
#     eval_dataset=eval_dataset,
#     reward_fn=reward_fn,
#     prompt_column=TEXT_COLUMN,
#     response_column=LABEL_COLUMN,
# )


# %% [markdown]
# # Judge

# %%
JUDGE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


# %%
from transformers import AutoTokenizer, AutoModelForCausalLM

judge_tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL)

# judge_bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
# )

judge_model = AutoModelForCausalLM.from_pretrained(
    JUDGE_MODEL,
    dtype=torch.bfloat16,
    # quantization_config=judge_bnb_config,
    device_map="cuda:0",
)

judge_model.eval()
for p in judge_model.parameters():
    p.requires_grad = False


# %%
def build_judge_prompt(question, answer, facts):
    return f"""
You are a strict evaluator for question answering.

Question:
{question}

Model answer:
{answer}

Supporting facts:
{facts}

Evaluate the answer and return ONLY a JSON object with the following keys:
- supported: 1 if the answer is fully supported by the facts, else 0
- complete: 1 if the answer addresses all parts of the question, else 0
- correct: 1 if the answer is factually correct, else 0

Return only valid JSON.
"""



# %%
import json

@torch.no_grad()
def run_judge(prompt, max_new_tokens=128):
    inputs = judge_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
    ).to(judge_model.device)

    outputs = judge_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        pad_token_id=judge_tokenizer.eos_token_id,
    )

    text = judge_tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )

    try:
        return json.loads(text)
    except Exception:
        return {"supported": 0, "complete": 0, "correct": 0}



# %%
# dict_keys(['prompts', 'completions', 'completion_ids', 'id',
# 'question', 'answer', 'q_entity', 'a_entity', 'graph', 'choices', 'trainer_state'])
# 8 prompts
def llm_judge_reward(
    **kwargs,
):

    completions = kwargs['completions'],
    prompt = kwargs['prompts'],
    facts=None,
    rewards = []

    for i in range(len(completions)):
        judge_prompt = build_judge_prompt(
            question=prompt[i],
            answer=completions[i],
            facts=facts[i] if facts is not None else "",
        )

        scores = run_judge(judge_prompt)

        reward = (
            1.0 * scores["correct"]
            + 0.5 * scores["supported"]
            + 0.5 * scores["complete"]
        )

        rewards.append(float(reward))

    return rewards

"""
def llm_judge_reward(
    prompts,
    generations,
    references=None,
    facts=None,
    **kwargs,
):
    rewards = []

    for i in range(len(generations)):
        judge_prompt = build_judge_prompt(
            question=prompts[i],
            answer=generations[i],
            facts=facts[i],
        )

        scores = run_judge(judge_prompt)

        # weighted sum (tune freely)
        reward = (
            1.0 * scores["correct"]
            + 0.5 * scores["supported"]
            + 0.5 * scores["complete"]
        )

        rewards.append(float(reward))

    return rewards
"""

# %%
grpo_config = GRPOConfig(
    output_dir="./grpo-lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4, # was 8
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


# %%
trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    train_dataset=train_dataset,
    reward_funcs=llm_judge_reward,
)


# %%
'''trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    args=grpo_config,
    train_dataset=train_dataset,
    reward_fns=[
        llm_judge_reward,
        length_penalty_reward,
    ],
    reward_weights=[1.0, 0.1],
)
'''

# %% [markdown]
# ## Inject refactx

# %%
# trainer.generation_kwargs

# %%

# %%
INDEX_PATH = '../indexes/simple_index.txt.gz'
# index = refactx.load_index(
#     POSTGRES_URL, 
#     #tokenizer,
#     #configkey=-200,
#     #cache='simple'
# )
index = refactx.load_index(INDEX_PATH, tokenizer=tokenizer)

constrained_processor = refactx.get_constrained_logits_processor(
    tokenizer, index, num_beams=1, num_batches=1, return_list=True, avoid_duplicates=True)

# %%
constrained_processor

# %%

trainer.generation_config = GenerationConfig(
    max_new_tokens=grpo_config.max_completion_length,
    do_sample=True,
    temperature=1.0,
    top_p=0.95,
    # logits_processor=constrained_processor,
)

original_generate = model.generate

def patched_generate(*args, **kwargs):

    if "logits_processor" not in kwargs:
        kwargs["logits_processor"] = constrained_processor

    return original_generate(*args, **kwargs)

model.generate = patched_generate

# %%
trainer.train()


# %%
# dict_keys(['prompts', 'completions', 'completion_ids', 'id',
# 'question', 'answer', 'q_entity', 'a_entity', 'graph', 'choices', 'trainer_state'])
# 8 prompts

# %%
trainer.model.save_pretrained("./grpo-lora-adapters")
tokenizer.save_pretrained("./grpo-lora-adapters")

