from base_dataset_config import QADataset

class TwoWikiMultiHopDataset(QADataset):
    def get_question_type(self, i):
        # Optionally, extract the type of question (e.g., single-hop or multi-hop)
        return self.dataset[i]['type']