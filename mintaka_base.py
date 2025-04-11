from base_dataset_config import QADataset

class MintakaDataset(QADataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_answer(self, i) -> list:
        return [self.dataset[i]['answer']['mention']]
