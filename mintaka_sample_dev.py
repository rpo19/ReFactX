import random
import json
from base_dataset_config import QADataset

config = {
    'path': '/workspace/data/mintaka_dev.json',
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
