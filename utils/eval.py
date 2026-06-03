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
        tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    except Exception as e:
        print('exc vlm', e)
        processor = AutoProcessor.from_pretrained(cfg["model_name"])
        tokenizer = processor.tokenizer

    device = cfg.get("device", "auto")
    try:
        if device == "auto":
            model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], device_map="auto")
        else:
            model = AutoModelForCausalLM.from_pretrained(cfg["model_name"]).to(device)
    except Exception as e:
        print('exc loading model, trying VLM path', e)
        if device == "auto":
            model = AutoModelForImageTextToText.from_pretrained(cfg["model_name"], device_map="auto")
        else:
            model = AutoModelForImageTextToText.from_pretrained(cfg["model_name"]).to(device)

    patch_model(model)
    model.eval()

    if cfg.get("prompt"):
        with open(cfg["prompt"]) as fd:
            PROMPT_TEMPLATE = json.load(fd)
    else:
        PROMPT_TEMPLATE = None
        print(20 * '-', 'Using default prompt!')

    experiment_name = cfg.get("experiment_name")
    if experiment_name is None:
        experiment_name = f'{os.path.basename(cfg["dataset"])}.{os.path.basename(cfg.get("model_config", cfg["model_name"]))}.{os.path.basename(cfg["index_config"])}'

    output_file = cfg.get("output")
    if output_file is None:
        output_file = os.path.join(cfg.get("log_dir", "."), f'{experiment_name}.out')

    prompt_length = tokenizer(refactx.apply_prompt_template(PROMPT_TEMPLATE),
                    return_tensors='pt', padding=False)['input_ids'].shape[1]

    index = refactx.load_index(cfg["index_data"], rootcert=cfg.get("http_rootcert"))
    index.set_tokenizer(tokenizer)

    metadata = {**cfg, 'date': get_utc_date_and_time(), 'prompt_length': prompt_length}

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
        dataset = dataset[dataset_start_from:]
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

        if index.rootkey <= max(tokenizer.vocab.values()):
            print('WARNING: rootkey could interfere with model tokens (if using postgres index)')


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

        dataset = load_dataset(
            cfg["dataset"],
            revision=cfg.get("dataset_revision", None),
            split=cfg.get("dataset_split", "train"),
        )

        dataloader = DataLoader(
            dataset,
            batch_size=cfg.get('batch_size', 1),
            sampler=cfg.get('sampler', None)
        )

        pad_token_id = tokenizer.pad_token_id

        patch_model(model)
        model.eval()

        with torch.no_grad():
            prompt_cache = None
            first_inputs = None

            for batch_number, batch in enumerate(tqdm(dataloader)):
                if cfg.get("debug", False):
                    print(f'\nBatch {batch_number}:')
                    for q in batch:
                        print(q)

                prompted_batch = [refactx.apply_prompt_template(PROMPT_TEMPLATE, q, enable_thinking=cfg.get("thinking", False)) for q in batch]

                refactx.CONSTRAINED_STATES.reset()

                batch_inputs = tokenizer(prompted_batch, return_tensors="pt").to(model.device)

                if first_inputs is None:
                    first_inputs = batch_inputs

                output = model.generate(
                    **batch_inputs,
                    logits_processor=logits_processor_list,
                    **cfg.get("generation_config", {}),
                )

                refactx.CONSTRAINED_STATES.beam_permutation()

                for i, (question, _, output_i) in enumerate(zip(batch, prompted_batch, output)):
                    full_prediction = tokenizer.decode(output_i[len(batch_inputs.input_ids[0]):])
                    prediction = refactx.get_answer(full_prediction)
                    prediction_complete = bool(prediction)

                    state = refactx.CONSTRAINED_STATES[i, 0]

                    new_tokens_generated = 0
                    for token in output_i[len(batch_inputs.input_ids[0]):]:
                        if token == pad_token_id:
                            break
                        new_tokens_generated += 1
                    reached_max_tokens = bool(
                        output_i[len(batch_inputs.input_ids[0]):].shape[0] == cfg.get('generation_config', {}).get('max_new_tokens')
                        and output_i[-1] != pad_token_id
                    )

                    sample = dict(
                        question=question,
                        answer_complete=prediction_complete,
                        prediction=prediction,
                        full_prediction=full_prediction,
                        prompt=tokenizer.decode(output_i[:len(batch_inputs.input_ids[0])]),
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
