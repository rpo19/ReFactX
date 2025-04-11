import json
import os
from cwq_base import CWQDataset

path = CWQDataset.get_dataset_path('ComplexWebQuestions_dev.json')

config = {
    'path': path,
}

print(f'Loading {config["path"]}')
with open(config['path']) as fd:
    raw_dataset = json.load(fd)

dataset = CWQDataset(raw_dataset, config)
