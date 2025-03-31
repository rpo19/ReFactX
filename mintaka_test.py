import json
import os
from mintaka_base import MintakaDataset

path = MintakaDataset.get_dataset_path('mintaka_test.json')

config = {
    'path': path
}

print(f'Loading {config["path"]}')
with open(config['path']) as fd:
    raw_dataset = json.load(fd)

DATASET_START_FROM = int(os.environ.get('DATASET_START_FROM', 0))
if DATASET_START_FROM:
    raw_dataset = raw_dataset[DATASET_START_FROM:]

dataset = MintakaDataset(raw_dataset, config)

if __name__ == '__main__':
    print('Length', len(dataset))
    print('First', dataset[0])
 
