from base_dataset_config import QADataset

class CWQDataset(QADataset):
    def get_question_type(self, i):
        # Optionally, extract the type of question (e.g., single-hop or multi-hop)
        return self.dataset[i]['compositionality_type']
    def get_answer(self, i):
        answers = []
        for answer in self.dataset[i]['answers']:
            aliases = answer.get('aliases', [])
            current_answer = {'answer': answer['answer'], 'aliases': aliases}
            answers.append(current_answer)
        return answers