from base_dataset_config import QADataset
from zipfile import ZipFile
import os
import pandas as pd
import json
from prompt_base import get_prompt_template, get_few_shot

PROMPT_TEMPLATE = get_prompt_template(
                        description='',
                        few_shot_examples=get_few_shot('no_descr.mh_slumdog,no_descr.asnot_clinton,no_descr.count_spain'))

# TODO this dataset assumes it is complete: if there is no evidence that proves something, then it is false.

# different prompt
# there are no description and short description
# TODO add example for multiple answers. e.g. which entities (more than one) own a company?

class BIDataset(QADataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_template = PROMPT_TEMPLATE

    def load_from_path(self, dataset_path):
        bi_data_path2 = os.environ.get('BI_DATA_PATH2')
        if bi_data_path2 is None:
            raise Exception('Missing env var BI_DATA_PATH2.')
        with ZipFile(dataset_path) as myzip:
            with myzip.open('anony_queries_v0.csv', pwd=bi_data_path2.encode()) as myfile:
                df = pd.read_csv(myfile)
        self.dataset = df.to_dict(orient='records')
    def get_question_type(self, sample):
        return sample['query_type:']
    def get_question_from_sample(self, sample) -> str:
        return sample['question:']
    def get_answer_type(self, sample) -> str:
        answer_type = 'simple'
        answer = sample['answer:']
        if answer.startswith('['):
            try:
                answer = json.loads(answer)
            except:
                pass
        if isinstance(answer, list):
            answer_type = 'list'
        return answer_type
    def get_answer_from_sample(self, sample) -> str:
        answer = sample['answer:']
        if answer.startswith('['):
            try:
                answer = json.loads(answer)
            except:
                pass
        if isinstance(answer, list):
            answer = ' ,'.join(answer)
        assert isinstance(answer, str)
        return answer
