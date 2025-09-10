import json
from mintaka_base import MintakaDataset

path = MintakaDataset.get_dataset_path('mintaka_dev.json')

config = {
    'path': path
}

print(f'Loading {config["path"]}')
with open(config['path']) as fd:
    raw_dataset = json.load(fd)

dataset = MintakaDataset(raw_dataset, config)
