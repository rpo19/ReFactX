import json
from base_dataset_config import QADataset

class MintakaDataset(QADataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_answer(self, i):
        return self.dataset[i]['answer']['mention']
