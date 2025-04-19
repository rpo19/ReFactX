import torch
from tqdm import tqdm
from transformers import DynamicCache, LogitsProcessorList
from ctrie import ConstrainedLogitsProcessor, ConstrainedStateList, ConstrainedState, DictIndex
import json
import importlib
import os
from torch.utils.data import DataLoader
import copy
import datetime
import click
import time

def eq_metadata(metadata1, metadata2):
    if metadata1['index_config_path'] != metadata2['index_config_path']:
        return False
    if metadata1['model_config_path'] != metadata2['model_config_path']:
        return False
    if metadata1['dataset_config_path'] != metadata2['dataset_config_path']:
        return False
    # if metadata1['experiment_name'] != metadata2['experiment_name']: # ignore exp name
    #     return False
    # if metadata1['date'] != metadata2['date']: # ignore date
    #     return False
    if metadata1['index_config'] != metadata2['index_config']:
        return False
    if metadata1['model_config'] != metadata2['model_config']:
        return False
    if metadata1['dataset_config'] != metadata2['dataset_config']:
        return False
    return True

def logrotate(file_name, dataset_length=None, metadata=None):
    idx = 0
    dataset_start_from = 0
    while True:
        if os.path.isfile(f'{file_name}.{idx}'):
            print(f'Found file: {file_name}.{idx}. Checking if it is complete.')
            if dataset_length is not None:
                with open(f'{file_name}.{idx}', 'r') as fd:
                    prev_output = fd.readlines()
                    header = json.loads(prev_output[0])
                    prev_output = prev_output[1:]
                    prev_dataset_length = len(prev_output)
                if prev_dataset_length < dataset_length:
                    if eq_metadata(header, metadata):
                        dataset_start_from = prev_dataset_length
                        print(f'Found incomplete run file: {file_name}.{idx}. Continuing from {dataset_start_from}.')
                        break
                    else:
                        print(f'Found incomplete run file: {file_name}.{idx}, but metadata mismatch. Ignoring it.')
        else:
            break
        idx += 1

    return f'{file_name}.{idx}', dataset_start_from

def get_utc_date_and_time():
    now = datetime.datetime.now(datetime.timezone.utc)
    nowstr = now.strftime("%d/%m/%Y %H:%M:%S UTC")
    return nowstr

@click.command()
@click.option("--name", "experiment_name", default=None, required=False, help="Experiment name.")
@click.option("--output", "output_file", required=False, default=None, type=click.Path(), help="Output file for the results.")
@click.option("--index", "index_config_path", required=True, help="Index configuration module (without .py).")
@click.option("--model", "model_config_path", required=True, help="Model configuration module (without .py).")
@click.option("--dataset", "dataset_config_path", required=True, help="Dataset configuration module (without .py).")
@click.option("--wandb", "wandb", is_flag=True, default=False, help="Log in wandb")
@click.option("--unconstrained-generation", is_flag=True, help="Unconstrained generation")
@click.option("--debug", is_flag=True, help="Print debug information.")
@click.option("--continue", 'continue_from_previous_run', is_flag=True, help="Continue previous run if not concluded (and if config was the same).")
@click.option("--log-dir", default='.', help="Log dir (only use if --output is not specified).")
def main(experiment_name, output_file, index_config_path, model_config_path, dataset_config_path, wandb, unconstrained_generation, debug, continue_from_previous_run, log_dir):
    if index_config_path.endswith('.py'):
        index_config_path = index_config_path[:-3]
    index_module = importlib.import_module(index_config_path)
    index_config = getattr(index_module, 'index_config')

    if model_config_path.endswith('.py'):
        model_config_path = model_config_path[:-3]
    model_module = importlib.import_module(model_config_path)
    model_config = getattr(model_module, 'model_config')

    if dataset_config_path.endswith('.py'):
        dataset_config_path = dataset_config_path[:-3]
    dataset_module = importlib.import_module(dataset_config_path)
    dataset = getattr(dataset_module, 'dataset').questions_dataset()

    if experiment_name is None:
        experiment_name = f'{os.path.basename(dataset_config_path)}.{os.path.basename(model_config_path)}.{os.path.basename(index_config_path)}'
    if output_file is None:
        output_file = f'{experiment_name}.out'
        output_file = os.path.join(log_dir, output_file)

    prompt_length = model_config.tokenizer(model_config.apply_prompt_template(dataset.prompt_template),
                    return_tensors='pt',
                    padding=False)['input_ids'].shape[1]

    metadata_plus = {
        'index_config_path': index_config_path,
        'model_config_path': model_config_path,
        'dataset_config_path': dataset_config_path,
        'experiment_name': experiment_name,
        'date': get_utc_date_and_time(),
        'index_config': dict(index_config),
        'model_config': dict(model_config),
        'dataset_config': dict(dataset.dump_config()),
        'prompt_length': prompt_length,
    }

    if continue_from_previous_run:
        output_file, dataset_start_from = logrotate(output_file, len(dataset), metadata_plus)
    else:
        output_file, dataset_start_from = logrotate(output_file)
    print('Output file:', output_file)
    if wandb:
        print('Logging in wandb.')
        time.sleep(5) # let the user time to stop

    assert os.path.isfile(output_file) or dataset_start_from == 0

    if dataset_start_from > 0:
        output_file_mode = 'a'
        dataset = dataset[dataset_start_from:]
    else:
        output_file_mode = 'w'
    with open(output_file, output_file_mode) as output_fd:
        if dataset_start_from == 0:
            output_fd.write(json.dumps(metadata_plus) + '\n')

        if wandb:
            import wandb
            wandb.init(
                project=experiment_name,
                config=metadata_plus,
                name=f"{experiment_name}_{get_utc_date_and_time()}",
            )
        try:
            if index_config.rootkey <= max(model_config.tokenizer.vocab.values()):
                print('WARNING: rootkey could interfere with model tokens (if using postgres index)')
        except:
            print('WARNING: rootkey could interfere with model tokens (if using postgres index)')

        num_states = model_config.batch_size * model_config.generate_args.get('num_beams', 1)
        states = ConstrainedStateList(
            [ConstrainedState(
                    begin_pattern = model_config.switch_pattern,
                    end_pattern = model_config.newline_token,
                    cache_index = DictIndex(end_of_triple=index_config.end_of_triple),
                    subtree_cache = DictIndex(end_of_triple=index_config.end_of_triple),
                    oneleaf_cache = DictIndex(end_of_triple=index_config.end_of_triple),
                ) for _ in range(num_states)],
            num_beams=model_config.generate_args.get('num_beams', 1),
            batch_size = model_config.batch_size,
            pad_token_id = model_config.generate_args['pad_token_id'])

        constrained_processor = ConstrainedLogitsProcessor(
            index=index_config.index,
            end_token=model_config.newline_token,
            states=states,
            tokenizer=model_config.tokenizer
            )
        logits_processor_list = LogitsProcessorList([
            constrained_processor
        ])
        if unconstrained_generation:
            logits_processor_list = LogitsProcessorList([])

        dataloader = DataLoader(dataset, batch_size=model_config.batch_size, shuffle=False)

        # cache the prompt only when batch_size == 1 or the padding is right
        # otherwise if padding left the prompt will change for each batch because of padding
        cache_prompt = model_config.generate_args.get('use_cache', False) and (
                    model_config.batch_size == 1 or model_config.tokenizer_args.get('padding_side')=='right')

        model_config.model.eval()
        with torch.no_grad():

            if cache_prompt:
                # only cache the prompt if padding_size is right
                prompt_cache = DynamicCache()
                inputs_prompt_begin = model_config.tokenizer(
                    [model_config.apply_prompt_template(dataset.prompt_template)] * model_config.batch_size * model_config.generate_args.get('num_beams', 1),
                    return_tensors='pt',
                    padding=False)
                inputs_prompt_begin.to(model_config.model.device)

                prompt_cache = model_config.model(
                        **inputs_prompt_begin,
                        use_cache=True,
                        past_key_values=prompt_cache,
                    ).past_key_values
            else:
                prompt_cache = None

            for batch_number, batch in enumerate(tqdm(dataloader)):
                if debug:
                    print(f'\nBatch {batch_number}:')
                    for question in batch:
                        print(question)
                prompted_batch = [model_config.apply_prompt_template(dataset.prompt_template, question) for question in batch]

                states.reset() # reset caches

                batch_inputs = model_config.tokenize_fun(prompted_batch)

                if cache_prompt:
                    if inputs_prompt_begin.input_ids.shape[0] != batch_inputs.input_ids.shape[0] * model_config.generate_args.get('num_beams', 1):
                        # last batch can mismatch in dimensions wrt the prompt cache
                        assert batch_number == len(dataloader) - 1
                        # in this case do not use the cache
                        past_key_values = None
                    else:
                        past_key_values = copy.deepcopy(prompt_cache)
                else:
                    past_key_values = None

                if len(states) != batch_inputs.input_ids.shape[0] * model_config.generate_args.get('num_beams', 1):
                    assert batch_number == len(dataloader) - 1
                    states = states[:batch_inputs.input_ids.shape[0] * model_config.generate_args.get('num_beams', 1)]
                    states.batch_size = batch_inputs.input_ids.shape[0]
                    constrained_processor.states = states

                output = model_config.model.generate(
                    **batch_inputs,
                    logits_processor=logits_processor_list,
                    **model_config.generate_args,
                    past_key_values=past_key_values,
                    kwargs = {'constrained_state': states}, # passing state
                )

                states.beam_permutation() # final permutation to match final beams

                state_idx_generator = range(0, num_states, model_config.generate_args.get('num_beams', 1))
                for question, prompted_question, output_i, state_idx in zip(batch, prompted_batch, output, state_idx_generator):
                    full_prediction = model_config.tokenizer.decode(output_i[len(batch_inputs.input_ids[0]):])
                    prediction = model_config.get_prediction(full_prediction)
                    prediction_complete = bool(prediction)
                    # TODO also save worse beams

                    state = states[state_idx] # TODO check if states are permuted before or after beam step

                    new_tokens_generated = 0
                    pad_token_id = model_config.generate_args.get('pad_token_id')
                    for token in output_i[len(batch_inputs.input_ids[0]):]:
                        if token == pad_token_id:
                            break
                        new_tokens_generated += 1
                    reached_max_tokens = bool(output_i[len(batch_inputs.input_ids[0]):].shape[0] == model_config.generate_args.get('max_new_tokens') and output_i[-1] != pad_token_id)

                    sample = dict(
                            question=question,
                            answer_complete=prediction_complete,
                            prediction=prediction,
                            full_prediction=full_prediction,
                            prompt=model_config.tokenizer.decode(output_i[:len(batch_inputs.input_ids[0])]),
                            full_sample=model_config.tokenizer.decode(output_i),
                            triples=list(map(model_config.tokenizer.decode, state.generated_triples)),
                            new_tokens_generated=new_tokens_generated,
                            reached_max_tokens=reached_max_tokens,
                        )
                    output_fd.write(json.dumps(sample) + '\n')

                    if wandb:
                        wandb.log(sample)

if __name__ == "__main__":
    main()
