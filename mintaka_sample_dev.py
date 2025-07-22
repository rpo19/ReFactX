import random
import json
from mintaka_base import MintakaDataset

config = {
    'path': './data/mintaka_dev.json',
    'random_seed': 1234,
    'num_samples': 100,
}

print(f'Loading {config["path"]}')
with open(config['path']) as fd:
    raw_dataset = json.load(fd)

# sample
random.seed(config['random_seed'])
raw_dataset = random.sample(raw_dataset, config['num_samples'])

dataset = MintakaDataset(raw_dataset, config)

