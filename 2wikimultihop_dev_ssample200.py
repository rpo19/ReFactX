import json
import os
from base_dataset_config import QADataset

path = QADataset.get_dataset_path('2wikimultihop_dev_ssample200.json')

config = {
    'path': path,
    'random_seed': 42,
    'sample_size': 200,
    'stratify': True,
    'columns': ['type'],
}

print(f'Loading {config["path"]}')
with open(config['path']) as fd:
    raw_dataset = json.load(fd)

DATASET_START_FROM = int(os.environ.get('DATASET_START_FROM', 0))
if DATASET_START_FROM:
    raw_dataset = raw_dataset[DATASET_START_FROM:]
    config['start_from'] = DATASET_START_FROM

dataset = QADataset(raw_dataset, config)
