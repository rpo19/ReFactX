import json
from mintaka_base import MintakaDataset

path = MintakaDataset.get_dataset_path('mintaka_train_ssample200.json')

config = {
    'path': path,
    'random_seed': 42,
    'sample_size': 200,
    'stratify': True,
    'columns': ['complexityType', 'category'],
}

print(f'Loading {config["path"]}')
with open(config['path']) as fd:
    raw_dataset = json.load(fd)

dataset = MintakaDataset(raw_dataset, config)

if __name__ == '__main__':
    print('Length', len(dataset))
    print('First', dataset[0])

