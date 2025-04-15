from bi_base import BIDataset

path = BIDataset.get_dataset_path('qa_queries.zip')

config = {
    'path': path,
}

print(f'Loading {config["path"]}')
with open(config['path']) as fd:
    raw_dataset = json.load(fd)

dataset = BIDataset([], config)
dataset.load_from_path(path)