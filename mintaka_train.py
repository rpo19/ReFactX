import json
import os
from mintaka_base import MintakaDataset

path = MintakaDataset.get_dataset_path('mintaka_train.json')

config = {
    'path': path
}

print(f'Loading {config["path"]}')
with open(config['path']) as fd:
    raw_dataset = json.load(fd)

dataset = MintakaDataset(raw_dataset, config)

if __name__ == '__main__':
    print('Length', len(dataset))
    print('First', dataset[0])

