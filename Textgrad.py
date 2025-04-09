# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] id="fecb2fb5-b87f-4428-8175-e3a46fe77371"
# ## Tutorial: Optimizing a Prompt
#
# ![TextGrad](https://github.com/vinid/data/blob/master/logo_full.png?raw=true)
#
# An autograd engine -- for textual gradients!
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zou-group/TextGrad/blob/main/examples/notebooks/Prompt-Optimization.ipynb)
# [![GitHub license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
# [![Arxiv](https://img.shields.io/badge/arXiv-2406.07496-B31B1B.svg)](https://arxiv.org/abs/2406.07496)
# [![Documentation Status](https://readthedocs.org/projects/textgrad/badge/?version=latest)](https://textgrad.readthedocs.io/en/latest/?badge=latest)
# [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/textgrad)](https://pypi.org/project/textgrad/)
# [![PyPI](https://img.shields.io/pypi/v/textgrad)](https://pypi.org/project/textgrad/)
#
# **Objectives:**
#
# * In this tutorial, we will run prompt optimization.
#
# **Requirements:**
#
# * You need to have an OpenAI API key to run this tutorial. This should be set as an environment variable as OPENAI_API_KEY.
#

# %% id="7add4547-4278-411b-a827-79be521851f1" outputId="b5e1d95c-ebca-4633-db1e-4b6da8206423"
# #!pip install textgrad # you might need to restart the notebook after installing textgrad

import concurrent
from tqdm import tqdm
import textgrad as tg
from textgrad.tasks import load_task
import numpy as np
import random

from textgrad.engine.base import EngineLM, CachedEngine
import platformdirs
import os
import importlib
from transformers import LogitsProcessorList
from ctrie import DictIndex, ConstrainedStateList, ConstrainedState, ConstrainedLogitsProcessor
import torch

from textgrad.autograd.string_based_ops import StringBasedFunction
from openai import OpenAI
from textgrad.engine.local_model_openai_api import ChatExternalClient

import wandb
import json


# %% [markdown] id="9a459a37-7446-4c4a-a7e0-38182b5dbd3e"
# Let's first define some support functions

# %% id="1ccc3b21bf9ddc48"
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


# %% id="649e06aef34d0990"
def eval_sample(item, eval_fn, model):
    """
    This function allows us to evaluate if an answer to a question in the prompt is a good answer.

    """
    x, y = item

    x = tg.Variable(x, requires_grad=False, role_description="query to the language model")
    y = tg.Variable(y, requires_grad=False, role_description="correct answer for the query")
    response = model(x)
    #try:
    eval_output_variable = eval_fn(inputs=dict(prediction=response, ground_truth_answer=y))
    return int(eval_output_variable.value)
    #except:
    #    eval_output_variable = eval_fn([x, y, response])
    #    eval_output_parsed = eval_fn.parse_output(eval_output_variable)
    #    return int(eval_output_parsed)


# %% id="9559a31e07e54d7f"
def eval_dataset(test_set, eval_fn, model, max_samples: int=None):
    if max_samples is None:
        max_samples = len(test_set)
    accuracy_list = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        for _, sample in enumerate(test_set):

            future = executor.submit(eval_sample, sample, eval_fn, model)
            futures.append(future)
            if len(futures) >= max_samples:
                break
        tqdm_loader = tqdm(concurrent.futures.as_completed(futures), total=len(futures), position=0)
        for future in tqdm_loader:
            acc_item = future.result()
            accuracy_list.append(acc_item)
            tqdm_loader.set_description(f"Accuracy: {np.mean(accuracy_list)}")
    return accuracy_list

def eval_dataset_slow(test_set, eval_fn, model, max_samples: int=None):
    if max_samples is None:
        max_samples = len(test_set)
    accuracy_list = []
    tqdm_loader = tqdm(test_set)
    for sample in tqdm_loader:
        acc_item = eval_sample(sample, eval_fn, model)
        accuracy_list.append(acc_item)
        tqdm_loader.set_description(f"Accuracy: {np.mean(accuracy_list)}")
    return accuracy_list


# %% id="4ea732b7edf34eb9"
def run_validation_revert(system_prompt: tg.Variable, results, model, eval_fn, val_set):
    val_performance = np.mean(eval_dataset(val_set, eval_fn, model))
    previous_performance = np.mean(results["validation_acc"][-1])
    print("val_performance: ", val_performance)
    print("previous_performance: ", previous_performance)
    previous_prompt = results["prompt"][-1]

    if val_performance < previous_performance:
        print(f"rejected prompt: {system_prompt.value}")
        system_prompt.set_value(previous_prompt)
        val_performance = previous_performance

    results["validation_acc"].append(val_performance)


# %%
# custom engine
# I need to directly use huggingface locally for constrained generation



class ChatConstrainedHF(EngineLM, CachedEngine):
    def __init__(
        self,
        model_config_path: str,
        index_config_path: str,
        system_prompt: None,
        **kwargs):
        """
        :param model_string:
        :param system_prompt:
        """

        if model_config_path.endswith('.py'):
            model_config_path = model_config_path[:-3]
        model_module = importlib.import_module(model_config_path)
        self.model_config = getattr(model_module, 'model_config')
        self.model_config.model.eval()

        if system_prompt is None:
            system_prompt = self.model_config.apply_prompt_template()

        if index_config_path.endswith('.py'):
            index_config_path = index_config_path[:-3]
        index_module = importlib.import_module(index_config_path)
        index_config = getattr(index_module, 'index_config')

        assert index_config.rootkey > max(self.model_config.tokenizer.vocab.values())

        root = platformdirs.user_cache_dir("textgrad")
        cache_path = os.path.join(root, f"cache_hf_{model_config_path}.db")

        super().__init__(cache_path=cache_path)

        self.system_prompt = system_prompt

        self.model = self.model_config.model

        self.is_multimodal = False

        self.num_states = self.model_config.generate_args.get('num_beams', 1)
        self.states = ConstrainedStateList(
            [ConstrainedState(
                    begin_pattern = self.model_config.switch_pattern,
                    end_pattern = self.model_config.newline_token,
                    cache_index = DictIndex(end_of_triple=index_config.end_of_triple),
                    subtree_cache = DictIndex(end_of_triple=index_config.end_of_triple),
                    oneleaf_cache = DictIndex(end_of_triple=index_config.end_of_triple),
                ) for _ in range(self.num_states)],
            num_beams=1,
            batch_size = self.model_config.batch_size,
            pad_token_id = self.model_config.generate_args['pad_token_id'])

        self.constrained_processor = ConstrainedLogitsProcessor(
            index=index_config.index,
            end_token=self.model_config.newline_token,
            states=self.states,
            tokenizer=self.model_config.tokenizer
            )
        self.logits_processor_list = LogitsProcessorList([
            self.constrained_processor
        ])

    def generate(self, prompt: str, system_prompt: str=None):
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        '''
        messages=[
                {"role": "system", "content": sys_prompt_arg},
                {"role": "user", "content": prompt},
            ]
        '''
        self.model_config.model.eval()
        with torch.no_grad():
            #batch = [messages]
            #prompted_batch = list(map(self.model_config.apply_prompt_template, batch))
            prompted_batch = [system_prompt + prompt]

            self.states.reset() # reset caches

            batch_inputs = self.model_config.tokenize_fun(prompted_batch)

            output = self.model_config.model.generate(
                **batch_inputs,
                #logits_processor=self.logits_processor_list,
                **self.model_config.generate_args,
                kwargs = {'constrained_state': self.states}, # passing stat
            )

            full_prediction = self.model_config.tokenizer.decode(output[0][len(batch_inputs.input_ids[0]):])
            #prediction = self.model_config.get_prediction(full_prediction)
            #prediction_complete = bool(prediction)

            response = full_prediction
            self._save_cache(sys_prompt_arg + prompt, response)

            return response

    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)



# %%
def load_dataset(dataset_config_path):
    if dataset_config_path.endswith('.py'):
        dataset_config_path = dataset_config_path[:-3]
    dataset_module = importlib.import_module(dataset_config_path)
    dataset = getattr(dataset_module, 'dataset')
    dataset_x_y = [(item['question'], item['answer']['mention']) for item in dataset]
    return dataset_x_y

def answer_equality_fn(prediction: tg.Variable, ground_truth_answer: tg.Variable):
    # TODO can consider other stuff like answer in prediction or viceversa
    # can consider if there are triples generated
    global llm_api_test
    full_prediction = str(prediction.value)
    actual_prediction = llm_api_test.model_config.get_prediction(full_prediction).lower()
    gt_str = str(ground_truth_answer.value).lower()
    eq = int(actual_prediction == gt_str)
    inclusion = int(gt_str in actual_prediction) + int(actual_prediction in gt_str)
    inclusion_wide = int(gt_str in full_prediction.lower())

    #result = (eq * 4 + inclusion * 3 + inclusion_wide * 1) / 8
    result = eq
    return result

if __name__ == '__main__':
    wandb.init(
        project='textgrad',
        config={},
        name=f"textgrad",
    )
    train_path = 'mintaka_train_ssample200'
    train_set = load_dataset(train_path)
    val_path = 'mintaka_dev_ssample72'
    val_set = load_dataset(val_path)
    test_path = 'mintaka_test_ssample200'
    test_set = load_dataset(test_path)

    # %%
    llm_api_test = ChatConstrainedHF(
        model_config_path="qwen25_7B_model",
        index_config_path="http_index_qwen",
        system_prompt=None,
        cache_path=os.path.join(platformdirs.user_cache_dir("textgrad"), "cache_hf_llama1.db"),
    )

    # %%



    fn_purpose = "The runtime of string-based function that checks if the prediction is correct."
    eval_fn = StringBasedFunction(answer_equality_fn, function_purpose=fn_purpose)

    # %%

    # %% id="e69f8431-661c-42f8-b7fc-efccea588a03" outputId="88a5a39d-b34c-4a7f-e17d-5e608eecad29"
    set_seed(12)

    client = OpenAI(base_url="http://localhost:8000/v1", api_key="lm-studio")
    llm_api_eval = ChatExternalClient(client=client, model_string=os.environ.get('MODEL_NAME'))
    tg.set_backward_engine(llm_api_eval, override=True)

    STARTING_SYSTEM_PROMPT = llm_api_test.system_prompt

    # %% [markdown] id="f40b576c-4ba0-4e6e-b3ed-81eb44524676"
    # This is the system prompt we are going to start from:

    # %% id="d3ed3261-6f9d-4906-8c4b-a3ad570f5950"
    print(STARTING_SYSTEM_PROMPT)


    # %%
    # # DEBUG
    # test_set = test_set[:4]
    # val_set = val_set[:4]
    # train_set = train_set[:10]

    # %% id="f7544127-38e0-4c74-8632-003efcc645ee" outputId="879c12c1-b24d-47d7-aa8d-90906df4acbe"
    train_loader = tg.tasks.DataLoader(train_set, batch_size=3, shuffle=True)


    # Testing the 0-shot performance of the evaluation engine
    system_prompt = tg.Variable(STARTING_SYSTEM_PROMPT,
                                requires_grad=True,
                                role_description="system prompt to the language model")
    model_evaluation = tg.BlackboxLLM(llm_api_eval, system_prompt)

    system_prompt = tg.Variable(STARTING_SYSTEM_PROMPT,
                                requires_grad=True,
                                role_description="structured system prompt to a somewhat capable language model that specifies the behavior and strategies for the QA task")
    model = tg.BlackboxLLM(llm_api_test, system_prompt)

    optimizer = tg.TextualGradientDescent(engine=llm_api_eval, parameters=[system_prompt])

    results = {"test_acc": [], "prompt": [], "validation_acc": []}
    results["test_acc"].append(eval_dataset(test_set, eval_fn, model))
    results["validation_acc"].append(eval_dataset(val_set, eval_fn, model))
    results["prompt"].append(system_prompt.get_value())


    # %% id="b3807736-1d81-4349-95db-257c20110d1a" outputId="b75a8864-bee3-4dbe-f2d2-b3002765fddf"
    for epoch in range(3):
        for steps, (batch_x, batch_y) in enumerate((pbar := tqdm(train_loader, position=0))):
            pbar.set_description(f"Training step {steps}. Epoch {epoch}")
            optimizer.zero_grad()
            losses = []
            for (x, y) in zip(batch_x, batch_y):
                x = tg.Variable(x, requires_grad=False, role_description="query to the language model")
                y = tg.Variable(y, requires_grad=False, role_description="correct answer for the query")
                response = model(x)
                try:
                    eval_output_variable = eval_fn(inputs=dict(prediction=response, ground_truth_answer=y))
                except:
                    eval_output_variable = eval_fn([x, y, response])
                losses.append(eval_output_variable)
            total_loss = tg.sum(losses)
            total_loss.backward()
            optimizer.step()

            run_validation_revert(system_prompt, results, model, eval_fn, val_set)

            print("sys prompt: ", system_prompt)
            test_acc = eval_dataset(test_set, eval_fn, model)
            results["test_acc"].append(test_acc)
            results["prompt"].append(system_prompt.get_value())
            if steps == 3:
                break

    with open('textgrad_results.json','w') as fd:
        json.dump(results, fd)
