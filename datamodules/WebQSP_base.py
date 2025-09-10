from base_dataset_config import QADataset

class WebQSPDataset(QADataset):
    def preprocess(self, dataset):
        self.Version = dataset['Version']
        self.FreebaseVersion = dataset['FreebaseVersion']
        return dataset['Questions']

    def dump(self):
        raw_data = {
            'Version': self.Version,
            'FreebaseVersion': self.FreebaseVersion,
            'Questions': self.dataset
        }
        return raw_data

    def get_answer_type(self, sample):
        answers_types = set()
        answer_count = 0
        for parse in sample['Parses']:
            for answer in parse['Answers']:
                answers_types.add(answer['AnswerType'].lower())
                answer_count += 1
        if answer_count > 1:
            answers_types.add('list')
        answer_type = ', '.join(answers_types)
        return answer_type

    def get_answer_from_sample(self, sample):
        # we assume that in case of duplication the llm-as-a-judge will judge correctly
        answers = []
        for parse in sample['Parses']:
            for answer in parse['Answers']:
                if answer['AnswerType'] == "Entity":
                    answers.append(answer['EntityName'])
                elif answer['AnswerType'] == "Value":
                    answers.append(answer['AnswerArgument'])
        textual_answer = ', '.join(answers)
        return textual_answer.strip()

    def get_question_from_sample(self, sample):
        return sample['RawQuestion']
