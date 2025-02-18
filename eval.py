import torch
import transformers
import bz2
import pickle
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer, DynamicCache, StoppingCriteriaList
from transformers.generation.logits_process import LogitsProcessorList
import psycopg
from ctrie import ConstrainedLogitsProcessor, ConstrainedStateList, ConstrainedState, DictIndex, GetAnswer

# TODO tokenize and load batch
# TODO stopping criteria and get answer



### dataset
import sys
import json
import importlib

from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
import datetime

index_config_path = sys.argv[1]
model_config_path = sys.argv[2] # .py
dataset_config_path = sys.argv[3] # .py
experiment_name = sys.argv[4]
output_file = sys.argv[5]

def get_utc_date_and_time():
    now = datetime.datetime.now(datetime.timezone.utc)
    nowstr = now.strftime("%d/%m/%Y %H:%M:%S UTC")
    return nowstr

with open(output_file, 'w') as output_fd:
    index_module = importlib.import_module(index_config_path)
    Index = getattr(index_module, 'Index')
    index = Index()

    model_module = importlib.import_module(model_config_path)
    Model = getattr(model_module, 'Model')
    model = Model()

    dataset_module = importlib.import_module(dataset_config_path)
    QADataset = getattr(dataset_module, 'QADataset')
    dataset = QADataset()

    metadata_plus = {
        'index_config_path': index_config_path,
        'model_config_path': model_config_path,
        'dataset_config_path': dataset_config_path,
        'experiment_name': experiment_name,
        'date': get_utc_date_and_time(),
        'index_config': dict(index),
        'model_config': dict(model),
        'dataset_config': dict(dataset)
    }
    output_fd.write(json.dumps(metadata_plus))
    output_fd.write('\n')

    assert index.rootkey > max(model.tokenizer.vocab.values())

    num_states = model.batch_size * model.generate_args.get('num_beams', 1)
    states = ConstrainedStateList(
        [ConstrainedState(
                begin_pattern = model.switch_pattern,
                end_pattern = model.newline_token,
                cache_index = DictIndex(end_of_triple=index.end_of_triple),
                subtree_cache = DictIndex(end_of_triple=index.end_of_triple),
                oneleaf_cache = DictIndex(end_of_triple=index.end_of_triple),
            ) for _ in range(num_states)])

    constrained_processor = ConstrainedLogitsProcessor(
        index=index.index,
        end_token=model.newline_token,
        states=states)
    logits_processor_list = LogitsProcessorList([
        constrained_processor
    ])

    getanswer = GetAnswer(model.answer_tokens, model.eofanswer, all)
    stopping_criteria = StoppingCriteriaList([
        getanswer
    ])

    dataloader = DataLoader(dataset, batch_size=model.batch_size, shuffle=False)

    model.model.eval()
    with torch.no_grad():

        prompt_cache = DynamicCache()
        inputs_prompt_begin = model.tokenizer(
            [model.apply_prompt_template()] * model.batch_size * model.generate_args.get('num_beams', 1),
            return_tensors='pt',
            padding=False)
        inputs_prompt_begin.to(model.model.device)

        prompt_cache = model.model(
                **inputs_prompt_begin,
                use_cache=True,
                past_key_values=prompt_cache,
            ).past_key_values

        for batch_number, batch in enumerate(tqdm(dataloader)):
            prompted_batch = list(map(model.apply_prompt_template, batch))

            batch_inputs = model.tokenize_fun(prompted_batch)

            getanswer.set_prompt(batch_inputs.input_ids[0])

            if inputs_prompt_begin.input_ids.shape[0] != batch_inputs.input_ids.shape[0] * model.generate_args.get('num_beams', 1):
                # last batch can mismatch in dimensions wrt the prompt cache
                assert batch_number == len(dataloader) - 1
                # in this case do not use the cache
                past_key_values = None
            else:
                past_key_values = copy.deepcopy(prompt_cache)

            output = model.model.generate(
                **batch_inputs,
                logits_processor=logits_processor_list,
                stopping_criteria=stopping_criteria,
                **model.generate_args,
                use_cache=True,
                past_key_values=past_key_values,
                kwargs = {'constrained_state': states}, # passing state
            )

            for question, prompted_question, output_i in zip(batch, prompted_batch, output):
                answer_complete, answer = getanswer.get_answer(output_i, return_answer=True)
                sample = dict(
                    question=question,
                    answer_complete=answer_complete,
                    prediction=answer,
                    full_prediction=model.tokenizer.decode(output_i[len(batch_inputs.input_ids[0]):]),
                    prompt=model.tokenizer.decode(output_i[:len(batch_inputs.input_ids[0])]),
                    full_sample=model.tokenizer.decode(output_i))
                output_fd.write(json.dumps(sample))
                output_fd.write('\n')
