import json
from WebQSP_base import WebQSPDataset

path = WebQSPDataset.get_dataset_path('WebQSP_test_sample200.json')

config = {
    'path': path,
    'random_seed': 42,
    'sample_size': 200,
    'stratify': False,
}

print(f'Loading {config["path"]}')
with open(config['path']) as fd:
    raw_dataset = json.load(fd)

dataset = WebQSPDataset(raw_dataset, config)
