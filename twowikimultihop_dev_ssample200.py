import json
import os
from twowikimultihop_base import TwoWikiMultiHopDataset

path = TwoWikiMultiHopDataset.get_dataset_path('2wikimultihop_dev_ssample200.json')

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

dataset = TwoWikiMultiHopDataset(raw_dataset, config)
