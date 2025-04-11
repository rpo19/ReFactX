import random
import json
from base_dataset_config import QADataset

path = QADataset.get_dataset_path('2wikimultihop/data/dev.json')

config = {
    'path': path,
    'random_seed': 1234,
    'num_samples': 50,
}

print(f'Loading {config["path"]}')
with open(config['path']) as fd:
    raw_dataset = json.load(fd)

# sample
random.seed(config['random_seed'])
raw_dataset = random.sample(raw_dataset, config['num_samples'])

dataset = QADataset(raw_dataset, config)
