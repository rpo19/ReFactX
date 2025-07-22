import random
import json
from mintaka_base import MintakaDataset

config = {
    'path': '/workspace/data/mintaka_test.json',
    'random_seed': 1234,
    'num_samples': 50,
}

print(f'Loading {config["path"]}')
with open(config['path']) as fd:
    raw_dataset = json.load(fd)

# sample
random.seed(config['random_seed'])
raw_dataset = random.sample(raw_dataset, config['num_samples'])

dataset = MintakaDataset(raw_dataset, config)

