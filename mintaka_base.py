from base_dataset_config import QADataset
import inflect

class MintakaDataset(QADataset):
    def __init__(self, dataset, config):
        super().__init__(dataset, config)
        self.inflect_engine = inflect.engine()

    def _get_numerical_aliases(self, num):
        self.inflect_engine.number_to_words(num)

    def _get_answer(self, answer):
        if answer['answer_type'] == 'entity':
            answer_list = [{'answer': answer['label']['en'], 'aliases': []} for answer in answer['answer']]
        elif answer['answer_type'] == 'string':
            answer_list = [{'answer': answer, 'aliases': []} for answer in answer['answer']]
        elif answer['answer_type'] == 'numerical':
            answer_list = [{'answer': answer, 'aliases': [self.inflect_engine.number_to_words(answer)]} for answer in answer['answer']]
        elif answer['answer_type'] == 'date':
            # "2017-08-25"
            # "2017"
            # TODO add aliases for dates
            answer_list = [{'answer': answer, 'aliases': []} for answer in answer['answer']]
        elif answer['answer_type'] == 'boolean':
            answer_list = [{'answer': answer, 'aliases': []} for answer in answer['answer']]

        return answer_list

    def get_answer(self, i):
        answers = self._get_answer(self.dataset[i]['answer'])
        return answers

    def get_question_type(self, i):
        return self.dataset[i]['complexityType']
