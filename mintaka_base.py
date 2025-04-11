from base_dataset_config import QADataset

class MintakaDataset(QADataset):
    def get_answer(self, i) -> list:
        return [self.dataset[i]['answer']['mention']]

    def get_question_type(self, i):
        return self.dataset[i]['complexityType']
