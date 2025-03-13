import json
from mintaka_base import MintakaDataset

config = {
    'path': '/workspace/data/mintaka_dev.json'
}

print(f'Loading {config["path"]}')
with open(config['path']) as fd:
    raw_dataset = json.load(fd)

dataset = MintakaDataset(raw_dataset, config)
