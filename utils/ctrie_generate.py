import click
import torch
import importlib
import time
from transformers import TextStreamer
from transformers.generation.logits_process import LogitsProcessorList
from refactx import ConstrainedLogitsProcessor, ConstrainedStateList, ConstrainedState, DictIndex
import sys
import os
import json

@click.command()
@click.option('--index', 'index_config_path', default='qwen25_index', help='Index module to import.')
@click.option('--model', 'model_config_path', default='qwen25_1B_model', help='Model module to import.')
@click.option('--generation-config', 'generation_config_str', default="num_beams=1,max_new_tokens=512,do_sample=False,temperature=None,top_k=None,top_p=None,min_p=None", help='Generation config (e.g. "max_new_tokens=512,top_k=5").')
@click.option('--prompt-module', 'prompt_module_name', required=False, default="prompt_base", help='Module from which to import PROMPT_TEMPLATE.')
@click.option('--prompt', default=None, help='Prompt (str or json) to use.')
def main(index_config_path, model_config_path, generation_config_str, prompt_module_name, prompt):
    prepare(index_config_path, model_config_path, generation_config_str, prompt_module_name, prompt)
    ask()

def prepare(index_config_path=None,
    model_config_path=None,
    generation_config_str=None,
    prompt_module_name=None,
    prompt=None,
    num_batches=1,
    ):

    global model_config
    global PROMPT_TEMPLATE
    global states
    global logits_processor_list
    global auto_streamer
    global generation_config

    try:
        generation_config = eval(f'dict({generation_config_str})')
    except Exception as e:
        print(generation_config_str)
        print(f"Error parsing generation config: {e}")
        sys.exit(1)

    if index_config_path.endswith('.py'):
        index_config_path = index_config_path[:-3]
    index_module = importlib.import_module(index_config_path)
    index_config = getattr(index_module, 'index_config')

    if model_config_path.endswith('.py'):
        model_config_path = model_config_path[:-3]
    model_module = importlib.import_module(model_config_path)
    model_config = getattr(model_module, 'model_config')

    assert prompt_module_name is not None or prompt is not None, 'Error: either --prompt-module or --prompt must be set.'
    if prompt_module_name:
        if prompt_module_name.endswith('.py'):
            prompt_module_name = prompt_module_name[:-3]
        prompt_module = importlib.import_module(prompt_module_name)
        PROMPT_TEMPLATE = prompt_module.PROMPT_TEMPLATE
    else:
        try:
            PROMPT_TEMPLATE = json.loads(prompt)
        except:
            print('Cannot load the prompt as JSON. Loading it as the system prompt.')
            PROMPT_TEMPLATE = [{
                'role': 'system',
                'content': prompt
            }]

    streamer = TextStreamer(model_config.tokenizer)

    num_beams = generation_config.get('num_beams', 1)
    auto_streamer = streamer if num_beams == 1 else None

    states_lol = [[ConstrainedState(
            begin_pattern = model_config.switch_pattern,
            end_pattern = model_config.newline_token,
            cache_index = DictIndex(end_of_triple=index_config.index.end_of_triple),
            subtree_cache = DictIndex(end_of_triple=index_config.index.end_of_triple),
            oneleaf_cache = DictIndex(end_of_triple=index_config.index.end_of_triple)
        ) for _ in range(num_beams)] 
            for _ in range(num_batches)]

    states = ConstrainedStateList(states_lol,
                num_beams=num_beams,
                num_batches = num_batches,
                pad_token_id = model_config.tokenizer.eos_token_id)

    constrained_processor = ConstrainedLogitsProcessor(
        index=index_config.index,
        end_token=model_config.newline_token, states=states, tokenizer=model_config.tokenizer)
    logits_processor_list = LogitsProcessorList([
        constrained_processor
    ])

    model_config.model.eval()

def ask(question=None, print_out=True, print_triples=True):
    global model_config
    global PROMPT_TEMPLATE
    global states
    global logits_processor_list
    global auto_streamer
    global generation_config
    global out

    interactive = not question

    while True:
        if interactive:
            print('Insert question. (CTRL+C to exit).')
            try:
                question = input('> ')
            except EOFError:
                return

        states.reset()

        prompted_texts = [model_config.apply_prompt_template(PROMPT_TEMPLATE, question)]
        print(prompted_texts[0])

        inputs = model_config.tokenizer(prompted_texts, return_tensors='pt', padding=True, padding_side='right')
        inputs = inputs.to(model_config.model.device)

        start = time.time()

        with torch.no_grad():
            out = model_config.model.generate(
                **inputs,
                logits_processor=logits_processor_list,
                streamer=auto_streamer,
                kwargs={'constrained_state': states},  # passing state
                **generation_config,
            )

        states.beam_permutation() # final permutation to match final beams

        torch.cuda.empty_cache()

        print('Elapsed', time.time() - start)

        if print_out:
            # print beam search results
            _from = len(inputs.input_ids[0])
            for i in range(out.shape[0]):
                print('-' * 30, sum(out[i][_from:]), len(out[i][_from:]))
                print(model_config.tokenizer.decode(out[i][_from:]))

        if print_triples:
            for batch_i in range(states.num_batches):
                for beam_i in range(states.num_beams):
                    for triple in states[batch_i, beam_i].generated_triples:
                        print(batch_i, beam_i, model_config.tokenizer.decode(triple)[:-1], triple, end='\n')

        if not interactive:
            break

if __name__ == '__main__':
    main()

