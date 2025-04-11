import json
from twowikimultihop_base import TwoWikiMultiHopDataset

path = TwoWikiMultiHopDataset.get_dataset_path('2wikimultihop_dev.json')

config = {
    'path': path,
}

print(f'Loading {config["path"]}')
with open(config['path']) as fd:
    raw_dataset = json.load(fd)

dataset = TwoWikiMultiHopDataset(raw_dataset, config)
