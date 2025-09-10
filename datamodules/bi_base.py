from base_dataset_config import QADataset
from zipfile import ZipFile
import os
import pandas as pd
import json
from refactx.prompt_base import get_prompt_template, get_few_shot

RELATION_DESCRIPTION = """
The facts use these relations:

- 'has role in' / 'role held by': Indicates that one entity has a role in another (e.g., a person is CEO of a company). This does not imply ownership or control.

- 'owns minority' / 'owned by (minority)': Indicates that the subject owns less than 50% of the other entity.

- 'owns majority' / 'owned by (majority)': Indicates that the subject owns 50% or more of the other entity.

- 'controls' / 'controlled by': Indicates that the subject has control over the other entity, usually implying significant ownership or influence.

- 'ultimately controls' / 'ultimately controlled by':
  Indicates that the subject is the ultimate or final controller of the other entity, possibly through intermediate layers.

- 'has holdings in' / 'held by': Indicates that the subject holds a stake or interest (e.g., voting rights) in the other entity, not necessarily through share ownership.
"""

PROMPT_TEMPLATE = get_prompt_template(
                        description='',
                        few_shot_examples=get_few_shot('no_descr.mh_slumdog,no_descr.asnot_clinton,no_descr.count_spain'),
                        custom_prompt=RELATION_DESCRIPTION)

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
                answer_list = json.loads(answer.replace("'",'"'))
                assert isinstance(answer_list, list)
                answer = answer_list
            except:
                pass
        if isinstance(answer, list):
            answer_type = 'list'
        return answer_type
    def get_answer_from_sample(self, sample) -> str:
        answer = sample['answer:']
        if answer.startswith('['):
            try:
                answer_list = json.loads(answer.replace("'",'"'))
                assert isinstance(answer_list, list)
                answer = answer_list
            except:
                pass
        if isinstance(answer, list):
            answer = ', '.join(answer)
        return answer
