import json
from base_dataset_config import QADataset

config = {
    'path': '/workspace/data/mintaka_test.json'
}

print(f'Loading {config["path"]}')
with open(config['path']) as fd:
    raw_dataset = json.load(fd)

dataset = QADataset(raw_dataset, config)
