import torch
import transformers
import bz2
import pickle
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer #, CodeGenTokenizer
from transformers.generation.logits_process import LogitsProcessorList
import psycopg
from ctrie import ConstrainedLogitsProcessor

# TODO tokenize and load batch
# TODO stopping criteria and get answer



### dataset
import sys
import json
import importlib

from torch.utils.data import DataLoader
from tqdm import tqdm

index_config_path = sys.argv[1]
model_config_path = sys.argv[2] # .py
dataset_config_path = sys.argv[3] # .py
experiment_name = sys.argv[4]

index_module = importlib.import_module(index_config_path)
Index = getattr(index_module, 'Index')
index = Index()

model_module = importlib.import_module(model_config_path)
Model = getattr(model_module, 'Model')
model = Model()

dataset_module = importlib.import_module(dataset_config_path)
QADataset = getattr(dataset_module, 'QADataset')
dataset = QADataset(model.tokenizer, model.prompt_template, model.device)

assert index.rootkey > max(model.tokenizer.vocab.values())

constrained_processor = ConstrainedLogitsProcessor(
    index = index.index,
    switch_pattern = model.switch_pattern,
    end_token = model.newline_token)
logits_processor_list = LogitsProcessorList([
    constrained_processor
])

# getanswer = GetAnswer(
#     answer = model.answer_tokens,
#     newline = model.newline_token,
#     early_stop_token = model.early_stop_token)
# stopping_criteria = StoppingCriteriaList([
#     getanswer
# ])

dataloader = DataLoader(dataset, batch_size=model.batch_size, shuffle=False)

model.model.eval()
with torch.no_grad():
    for batch in tqdm(dataloader):
        # getanswer.set_prompt(inputs)

        output = model.model.generate(
            **batch,
            logits_processor=logits_processor_list,
            # stopping_criteria=stopping_criteria,
            **model.generate_args
        )

        # get answer