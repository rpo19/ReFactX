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

    def get_answer(self, i) -> list:
        # we assume that in case of duplication the llm-as-a-judge will judge correctly
        answers = []
        for parse in self.dataset[i]['Parses']:
            for answer in parse['Answers']:
                if answer['AnswerType'] == "Entity":
                    answers.append(answer['EntityName'])
                elif answer['AnswerType'] == "Value":
                    answers.append(answer['AnswerArgument'])
        textual_answer = ', '.join(answers)
        return textual_answer.strip()

    def get_question_from_sample(self, sample):
        return sample['RawQuestion']
