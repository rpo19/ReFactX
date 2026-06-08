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


def calculate_metrics(prediction, input_sample, answer_key='answer'):
    # Calculate precision, recall, and F1 score
    if not bool(prediction):
        return 0, 0, 0
    reference = input_sample.get(answer_key, [])
    if isinstance(reference, str):
        reference = list(reference)
    reference_set = set(reference)
    assert isinstance(prediction, list)
    prediction_set = set(prediction)
    precision = len(prediction_set & reference_set) / len(prediction_set) if prediction_set else 0
    recall = len(prediction_set & reference_set) / len(reference_set) if reference_set else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return precision, recall, f1


@click.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True), help="Path to JSON config file.")
def main(config_path):
    with open(config_path) as f:
        cfg = json.load(f)

    experiment_name = cfg.get("experiment_name")
    if experiment_name is None:
        experiment_name = f'{os.path.basename(cfg["dataset"])}.{os.path.basename(cfg.get("model_config", cfg["model_name"]))}.{os.path.basename(cfg["index_config"])}'

    input_file = cfg.get("output")
    assert os.path.isfile(input_file)

    output_file = input_file + '.reeval'

    dataset = load_dataset(
        cfg["dataset"],
        revision=cfg.get("dataset_revision", None),
        split=cfg.get("dataset_split", "train"),
    )

    output_file_mode = 'w'

    with open(input_file) as input_fd:  

        with open(output_file, output_file_mode) as output_fd:
            metadata = json.loads(input_fd.readline())
            metadata['reeval'] = datetime.datetime.now(datetime.timezone.utc).strftime("%d/%m/%Y %H:%M:%S UTC")
            output_fd.write(json.dumps(metadata) + '\n')

            dataloader = DataLoader(
                dataset,
                batch_size=cfg.get('batch_size', 1),
                sampler=cfg.get('sampler', None)
            )

            macro_evaluation = {
                "precision": [],
                "recall": [],
                "f1": []
            }

            for batch_number, batch in enumerate(tqdm(dataloader)):
                if cfg.get("debug", False):
                    print(f'\nBatch {batch_number}:')
                    for q in batch:
                        print(q)



                # prediction from output
                # gt answer from batch

                for i in range(len(batch['question'])):
                    input_sample = {k: batch[k][i] for k in batch}

                    output = input_fd.readline()
                    if not output:
                        print('end at batch', batch_number)
                        break
                    output = json.loads(output)

                    
                    

                
                

                    full_prediction = output['full_prediction']
                    prediction = output['prediction']
                    prediction_complete = output['prediction_complete']

                    # calculate metrics p r f1
                    precision, recall, f1 = calculate_metrics(prediction, input_sample, answer_key=cfg.get('answer_key', 'answer'))
                    macro_evaluation["precision"].append(precision)
                    macro_evaluation["recall"].append(recall)
                    macro_evaluation["f1"].append(f1)

                    sample = output

                    sample['evaluation'] = {
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                    }

                    output_fd.write(json.dumps(sample) + '\n')


                macro_precision = sum(macro_evaluation["precision"]) / len(macro_evaluation["precision"]) if macro_evaluation["precision"] else 0,
                macro_recall = sum(macro_evaluation["recall"]) / len(macro_evaluation["recall"]) if macro_evaluation["recall"] else 0,
                macro_f1 = sum(macro_evaluation["f1"]) / len(macro_evaluation["f1"]) if macro_evaluation["f1"] else 0,

                macro_metrics = {
                    "macro_precision": macro_precision,
                    "macro_recall": macro_recall,
                    "macro_f1": macro_f1,
                }

                output_fd.write(json.dumps(macro_metrics) + '\n')

if __name__ == "__main__":
    main()
