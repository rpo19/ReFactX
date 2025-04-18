from base_dataset_config import QADataset

class MintakaDataset(QADataset):
    def get_answer_from_sample(self, sample):
        return sample['answer']['mention']
    def get_answer_type(self, sample):
        answer_type = sample['answer']['answerType']
        if len(sample['answer']['answer']) > 1:
            answer_type += ',list'
        return answer_type
    def get_question_type(self, sample):
        return sample['complexityType']
