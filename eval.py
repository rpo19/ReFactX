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

def logrotate(file_name):
    idx = 0
    while True:
        if not os.path.isfile(f'{file_name}.{idx}'):
            break
        idx += 1

    return f'{file_name}.{idx}'

def get_utc_date_and_time():
    now = datetime.datetime.now(datetime.timezone.utc)
    nowstr = now.strftime("%d/%m/%Y %H:%M:%S UTC")
    return nowstr

@click.command()
@click.argument("experiment_name")
@click.option("--output", "output_file", required=False, default=None, type=click.Path(), help="Output file for the results.")
@click.option("--index", "index_config_path", required=True, help="Index configuration module (without .py).")
@click.option("--model", "model_config_path", required=True, help="Model configuration module (without .py).")
@click.option("--dataset", "dataset_config_path", required=True, help="Dataset configuration module (without .py).")
@click.option("--wandb", "wandb", is_flag=True, help="Log in wandb")
@click.option("--unconstrained-generation", is_flag=True, help="Unconstrained generation")
def main(experiment_name, output_file, index_config_path, model_config_path, dataset_config_path, wandb, unconstrained_generation):
    if output_file is None:
        output_file = f'{experiment_name}.out'
    output_file = logrotate(output_file)
    print('Output file:', output_file)
    if wandb:
        print('Logging in wandb.')
        time.sleep(5) # let the user time to stop

    with open(output_file, 'w') as output_fd:
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

        metadata_plus = {
            'index_config_path': index_config_path,
            'model_config_path': model_config_path,
            'dataset_config_path': dataset_config_path,
            'experiment_name': experiment_name,
            'date': get_utc_date_and_time(),
            'index_config': dict(index_config),
            'model_config': dict(model_config),
            'dataset_config': dict(dataset.dump_config())
        }
        output_fd.write(json.dumps(metadata_plus))
        output_fd.write('\n')

        if wandb:
            import wandb
            wandb.init(
                project=experiment_name,
                config=metadata_plus,
                name=f"{experiment_name}_{get_utc_date_and_time()}",
            )

        assert index_config.rootkey > max(model_config.tokenizer.vocab.values())

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
                    [model_config.apply_prompt_template()] * model_config.batch_size * model_config.generate_args.get('num_beams', 1),
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
                print(f'\nBatch {batch_number}:')
                for question in batch:
                    print(question)
                prompted_batch = list(map(model_config.apply_prompt_template, batch))

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
                    
                    reached_max_tokens = output_i[len(batch_inputs.input_ids[0]):].shape[0] == model_config.generate_args.get('max_new_tokens') and output_i[-1] != model_config.generate_args.get('pad_token_id')

                    sample = dict(
                            question=question,
                            answer_complete=prediction_complete,
                            prediction=prediction,
                            full_prediction=full_prediction,
                            prompt=model_config.tokenizer.decode(output_i[:len(batch_inputs.input_ids[0])]),
                            full_sample=model_config.tokenizer.decode(output_i),
                            triples=list(map(model_config.tokenizer.decode, state.generated_triples)),
                            reached_max_tokens=reached_max_tokens,
                        )
                    output_fd.write(json.dumps(sample))
                    output_fd.write('\n')

                    if wandb:
                        wandb.log(sample)

if __name__ == "__main__":
    main()
