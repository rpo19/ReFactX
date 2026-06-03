from dotenv import load_dotenv
import torch
from tqdm import tqdm
from transformers import LogitsProcessorList
from refactx import patch_model
import refactx
import json
import importlib
import os
from torch.utils.data import DataLoader
import copy
import datetime
import click
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
from datasets import load_dataset


def eq_metadata(m1, m2):
    keys = ['index_config_path', 'model_config_path', 'dataset_config_path',
            'index_config', 'model_config', 'dataset_config']
    return all(m1.get(k) == m2.get(k) for k in keys)


def logrotate(file_name, dataset_length=None, metadata=None):
    idx = 0
    dataset_start_from = 0
    while True:
        path = f'{file_name}.{idx}'
        if not os.path.isfile(path):
            break
        print(f'Found file: {path}. Checking if it is complete.')
        if dataset_length is not None:
            with open(path) as fd:
                prev_output = fd.readlines()
                header = json.loads(prev_output[0])
                prev_output = prev_output[1:]
                prev_dataset_length = len(prev_output)
            if prev_dataset_length < dataset_length:
                if eq_metadata(header, metadata):
                    dataset_start_from = prev_dataset_length
                    print(f'Found incomplete run file: {path}. Continuing from {dataset_start_from}.')
                    break
                else:
                    print(f'Found incomplete run file: {path}, but metadata mismatch. Ignoring it.')
        idx += 1
    return f'{file_name}.{idx}', dataset_start_from


def get_utc_date_and_time():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%d/%m/%Y %H:%M:%S UTC")


def import_module(path):
    if path.endswith('.py'):
        path = path[:-3]
    return importlib.import_module(path)


@click.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True), help="Path to JSON config file.")
def main(config_path):
    with open(config_path) as f:
        cfg = json.load(f)

    print("Loading tokenizer and model...")
    try:
        processor = None
        tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], padding_side='left')
    except Exception as e:
        print('exc vlm', e)
        processor = AutoProcessor.from_pretrained(cfg["model_name"], padding_side='left')
        tokenizer = processor.tokenizer

    device = cfg.get("device", "auto")
    try:
        if device == "auto":
            model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], device_map="auto")
        else:
            model = AutoModelForCausalLM.from_pretrained(cfg["model_name"]).to(device)
    except Exception as e:
        print('exc loading model, trying VLM path', type(e), e)
        if device == "auto":
            model = AutoModelForImageTextToText.from_pretrained(cfg["model_name"], device_map="auto")
        else:
            model = AutoModelForImageTextToText.from_pretrained(cfg["model_name"]).to(device)

    patch_model(model)
    model.eval()

    if cfg.get("prompt"):
        PROMPT_TEMPLATE = refactx.load_prompt(cfg["prompt"])
    else:
        PROMPT_TEMPLATE = None
        print(20 * '-', 'Using default prompt!')

    experiment_name = cfg.get("experiment_name")
    if experiment_name is None:
        experiment_name = f'{os.path.basename(cfg["dataset"])}.{os.path.basename(cfg.get("model_config", cfg["model_name"]))}.{os.path.basename(cfg["index_config"])}'

    output_file = cfg.get("output")
    if output_file is None:
        output_file = os.path.join(cfg.get("log_dir", "."), f'{experiment_name}.out')

    prompt_length = tokenizer(refactx.apply_prompt_template(tokenizer, PROMPT_TEMPLATE, "question"),
                    return_tensors='pt', padding=False)['input_ids'].shape[1]


    print("Loading index...")
    load_dotenv()
    index_path = os.getenv("INDEX_PATH")
    assert index_path is not None, 'ERROR: index must be provided via --index or INDEX_PATH environment variables.'

    tablename = cfg.get("tablename", None)
    assert tablename or 'tablename' in index_path, 'tablename must be provided in config or as part of index filename'

    if tablename:
        index_path = f'{index_path}?tablename={tablename}'

    index = refactx.load_index(index_path, rootcert=cfg.get("http_rootcert"))
    index.set_tokenizer(tokenizer)

    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id

    generation_config = cfg.get("generation_config", {})
    if not 'pad_token_id' in generation_config:
        generation_config['pad_token_id'] = pad_token_id
        cfg['generation_config'] = generation_config

    metadata = {**cfg, 'date': get_utc_date_and_time(), 'prompt_length': prompt_length}

    dataset = load_dataset(
        cfg["dataset"],
        revision=cfg.get("dataset_revision", None),
        split=cfg.get("dataset_split", "train"),
    )

    if cfg.get("continue", False):
        output_file, dataset_start_from = logrotate(output_file, len(dataset), metadata)
    else:
        output_file, dataset_start_from = logrotate(output_file)
    print('Output file:', output_file)

    if cfg.get("wandb", False):
        print('Logging in wandb.')
        time.sleep(5)

    assert os.path.isfile(output_file) or dataset_start_from == 0

    if dataset_start_from > 0:
        output_file_mode = 'a'
        dataset = dataset.select(range(dataset_start_from, len(dataset)))
    else:
        output_file_mode = 'w'

    with open(output_file, output_file_mode) as output_fd:
        if dataset_start_from == 0:
            output_fd.write(json.dumps(metadata) + '\n')

        if cfg.get("wandb", False):
            import wandb
            wandb.init(
                project=experiment_name,
                config=metadata,
                name=f"{experiment_name}_{get_utc_date_and_time()}",
            )

        if index.rootkey >= 0 and index.rootkey <= max(tokenizer.vocab.values()):
            print(f'WARNING: rootkey ({index.rootkey}) could interfere with model tokens (if using postgres index)')


        if cfg.get("unconstrained_generation", False):
            logits_processor_list = LogitsProcessorList([])
        else:
            logits_processor_list = refactx.get_constrained_logits_processor(
                tokenizer,
                index,
                cfg.get('num_beams', 1),
                cfg.get('batch_size', 1),
                cfg.get('avoid_duplicates', True)
            )

        dataloader = DataLoader(
            dataset,
            batch_size=cfg.get('batch_size', 1),
            sampler=cfg.get('sampler', None)
        )

        patch_model(model)
        model.eval()

        with torch.no_grad():
            first_inputs = None

            for batch_number, batch in enumerate(tqdm(dataloader)):
                if cfg.get("debug", False):
                    print(f'\nBatch {batch_number}:')
                    for q in batch:
                        print(q)

                prompted_batch = [
                    refactx.apply_prompt_template(
                        tokenizer,
                        PROMPT_TEMPLATE,
                        q,
                        enable_thinking=cfg.get("thinking", False)
                    ) for q in batch['question']]

                batch_inputs = tokenizer(prompted_batch, return_tensors="pt", padding=True).to(model.device)

                refactx.get_constrained_states().reset()
                logits_processor_list[0]._reinit_states_to_input_ids(batch_inputs['input_ids'])

                if first_inputs is None:
                    first_inputs = batch_inputs

                output = model.generate(
                    **batch_inputs,
                    logits_processor=logits_processor_list,
                    **generation_config,
                )

                refactx.get_constrained_states().beam_permutation()

                for i, (question, _, output_i) in enumerate(zip(batch['question'], prompted_batch, output)):
                    state = refactx.get_constrained_states()[i, 0]

                    start_idx = len(batch_inputs.input_ids[0])

                    new_tokens_generated = 0
                    end_idx = start_idx
                    for token in output_i[start_idx:]:
                        if token == pad_token_id:
                            break
                        if token == eos_token_id and end_idx == start_idx:
                            # for removing imend, eos and padding tokens
                            end_idx = start_idx + new_tokens_generated
                        new_tokens_generated += 1
                    reached_max_tokens = bool(
                        output_i[start_idx:].shape[0] == generation_config.get('max_new_tokens')
                        and output_i[-1] != pad_token_id
                    )

                    full_prediction = tokenizer.decode(output_i[start_idx:end_idx])
                    prediction = refactx.get_answer(full_prediction)
                    prediction_complete = bool(prediction)

                    sample = dict(
                        question=question,
                        answer_complete=prediction_complete,
                        prediction=prediction,
                        full_prediction=full_prediction,
                        prompt=tokenizer.decode(output_i[:start_idx]),
                        full_sample=tokenizer.decode(output_i),
                        triples=list(map(tokenizer.decode, state.generated_triples)),
                        new_tokens_generated=new_tokens_generated,
                        reached_max_tokens=reached_max_tokens,
                    )
                    output_fd.write(json.dumps(sample) + '\n')

                    if cfg.get("wandb", False):
                        wandb.log(sample)


if __name__ == "__main__":
    main()
