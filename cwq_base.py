from base_dataset_config import QADataset

class CWQDataset(QADataset):
    def get_question_type(self, i):
        # Optionally, extract the type of question (e.g., single-hop or multi-hop)
        return self.dataset[i]['compositionality_type']
    def get_answer(self, i):
        textual_answer = ''
        for answer in self.dataset[i]['answers']:
            aliases = answer.get('aliases', [])
            textual_answer += answer['answer'] + ' '
            if aliases:
                textual_answer += '(' + ', '.join(aliases) + ') '
            textual_answer += ', '
        # Remove the trailing comma and space
        if textual_answer.endswith(', '):
            textual_answer = textual_answer[:-2]
        return textual_answer.strip()