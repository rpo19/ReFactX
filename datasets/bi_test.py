from bi_base import BIDataset

path = BIDataset.get_dataset_path('qa_queries.zip')

config = {
    'path': path,
}

print(f'Loading {config["path"]}')

dataset = BIDataset([], config)
dataset.load_from_path(path)
