import json
from cwq_base import CWQDataset

path = CWQDataset.get_dataset_path('ComplexWebQuestions_dev_ssample200.json')

config = {
    'path': path,
    'random_seed': 42,
    'sample_size': 200,
    'stratify': True,
    'columns': ['compositionality_type'],
    'fix_encoding': False, # already fixed during sampling
}

print(f'Loading {config["path"]}')
with open(config['path']) as fd:
    raw_dataset = json.load(fd)

dataset = CWQDataset(raw_dataset, config)
