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
    ignore_keys = set(["date"])
    all_keys = set(m1.keys()).intersection(set(m2.keys())) - ignore_keys
    return all(m1.get(k) == m2.get(k) for k in all_keys)

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


def calculate_metrics(prediction, input_sample, answer_key='answer', lowercase=True):
    # Calculate precision, recall, F1 score, boolean accuracy (1 correct or 0 error), i don't know
    if not bool(prediction):
        return 0, 0, 0, 0, 0
    reference = input_sample.get(answer_key, [])
    if isinstance(reference, str):
        reference = [reference]

    assert isinstance(prediction, list)
    if lowercase:
        prediction = [p.lower() for p in prediction]
        reference = [r.lower() for r in reference]

    reference_set = set(reference)
    prediction_set = set(prediction)
    precision = len(prediction_set & reference_set) / len(prediction_set) if prediction_set else 0
    recall = len(prediction_set & reference_set) / len(reference_set) if reference_set else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    correct = int(prediction_set == reference_set)
    dont_know = int("i don't know" in prediction[0].lower())
    return precision, recall, f1, correct, dont_know


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

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    torch_dtype = dtype_map.get(
        cfg.get("model_dtype", "bfloat16"),
        torch.bfloat16,
    )

    try:
        if device == "auto":
            model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], device_map="auto", dtype=torch_dtype)
        else:
            model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], dtype=torch_dtype).to(device)
    except Exception as e:
        print('exc loading model, trying VLM path', type(e), e)
        if device == "auto":
            model = AutoModelForImageTextToText.from_pretrained(cfg["model_name"], device_map="auto", dtype=torch_dtype)
        else:
            model = AutoModelForImageTextToText.from_pretrained(cfg["model_name"], dtype=torch_dtype).to(device)

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
        os.makedirs(cfg.get("log_dir", "."), exist_ok=True)
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

    n = cfg.get("n", None)
    if n is not None:
        dataset = dataset.select(range(min(n, len(dataset))))
        print(f'Limited dataset to first {n} samples.')

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

        macro_evaluation = {
            "precision": [],
            "recall": [],
            "f1": [],
            "correct": [],
            "dont_know": []
        }

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

                for i, (question, output_i) in enumerate(zip(batch['question'], output)):
                    input_sample = {k: batch[k][i] for k in batch}
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

                    # calculate metrics p r f1
                    precision, recall, f1, correct, dont_know = calculate_metrics(prediction, input_sample, answer_key=cfg.get('answer_key', 'answer'))
                    macro_evaluation["precision"].append(precision)
                    macro_evaluation["recall"].append(recall)
                    macro_evaluation["f1"].append(f1)
                    macro_evaluation["correct"].append(correct)
                    macro_evaluation["dont_know"].append(dont_know)

                    sample = dict(
                        input_sample=input_sample,
                        gt_answer=input_sample.get(cfg.get('answer_key', 'answer'), None),
                        question=question,
                        answer_complete=prediction_complete,
                        prediction=prediction,
                        full_prediction=full_prediction,
                        prompt=tokenizer.decode(output_i[:start_idx]),
                        full_sample=tokenizer.decode(output_i),
                        triples=list(map(tokenizer.decode, state.generated_triples)),
                        new_tokens_generated=new_tokens_generated,
                        reached_max_tokens=reached_max_tokens,
                        evaluation={
                            "precision": precision,
                            "recall": recall,
                            "f1": f1,
                            "correct": correct,
                            "dont_know": dont_know
                        }
                    )

                    output_fd.write(json.dumps(sample) + '\n')

                    if cfg.get("wandb", False):
                        wandb.log(sample)

                macro_precision = sum(macro_evaluation["precision"]) / len(macro_evaluation["precision"]) if macro_evaluation["precision"] else 0
                macro_recall = sum(macro_evaluation["recall"]) / len(macro_evaluation["recall"]) if macro_evaluation["recall"] else 0
                macro_f1 = sum(macro_evaluation["f1"]) / len(macro_evaluation["f1"]) if macro_evaluation["f1"] else 0
                macro_correct_accuracy = sum(macro_evaluation["correct"]) / len(macro_evaluation["correct"]) if macro_evaluation["correct"] else 0
                macro_dont_know = sum(macro_evaluation["dont_know"]) / len(macro_evaluation["dont_know"]) if macro_evaluation["dont_know"] else 0
                
                
                # metrics on the answered questions only (excluding "i don't know" answers)
                macro_answered_precision = 0
                macro_answered_recall = 0
                macro_answered_f1 = 0
                macro_answered_accuracy = 0
                
                answered_question_num = 0
                for p, r, f1, correct, dont_know in zip(macro_evaluation["precision"], macro_evaluation["recall"], macro_evaluation["f1"], macro_evaluation["correct"], macro_evaluation["dont_know"]):
                    if not dont_know:
                        answered_question_num += 1
                        macro_answered_precision += p
                        macro_answered_recall += r
                        macro_answered_f1 += f1
                        macro_answered_accuracy += correct

                macro_answered_precision /= answered_question_num if answered_question_num > 0 else 0
                macro_answered_recall /= answered_question_num if answered_question_num > 0 else 0
                macro_answered_f1 /= answered_question_num if answered_question_num > 0 else 0
                macro_answered_accuracy /= answered_question_num if answered_question_num > 0 else 0

                macro_metrics = {
                    "macro_precision": macro_precision,
                    "macro_recall": macro_recall,
                    "macro_f1": macro_f1,
                    "macro_correct_accuracy": macro_correct_accuracy,
                    "macro_dont_know": macro_dont_know,
                    "macro_answered_precision": macro_answered_precision,
                    "macro_answered_recall": macro_answered_recall,
                    "macro_answered_f1": macro_answered_f1,
                    "macro_answered_accuracy": macro_answered_accuracy
                }

                output_fd.write(json.dumps(macro_metrics) + '\n')

                if cfg.get("wandb", False):
                    wandb.log(macro_metrics)

if __name__ == "__main__":
    main()
