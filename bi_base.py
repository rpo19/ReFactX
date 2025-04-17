from base_dataset_config import QADataset
from zipfile import ZipFile
import os
import pandas as pd

class BIDataset(QADataset):
    def load_from_path(self, dataset_path):
        bi_data_path2 = os.environ.get('BI_DATA_PATH2')
        if bi_data_path2 is None:
            raise Exception('Missing env var BI_DATA_PATH2.')
        with ZipFile(dataset_path) as myzip:
            with myzip.open('anony_queries_v0.csv', pwd=bi_data_path2.encode()) as myfile:
                df = pd.read_csv(myfile)
        self.dataset = df.to_dict(orient='records')
    def get_question_type(self, i):
        return self.dataset[i]['query_type:']
    def get_question_from_sample(self, sample) -> str:
        return sample['question:']
    def get_answer(self, i) -> str:
        answer = self.dataset[i]['answer:']
        if answer.startswith('['):
            try:
                answer = json.loads(answer)
            except:
                pass
        if isinstance(answer, list):
            answer = ' ,'.join(answer)
        assert isinstance(answer, str)
        return answer