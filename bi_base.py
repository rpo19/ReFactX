from base_dataset_config import QADataset
from zipfile import ZipFile
import os
import pandas as pd

# TODO this dataset assumes it is complete: if there is no evidence that proves something, then it is false.

# different prompt
# there are no description and short description
PROMPT_TEMPLATE = [
    {
        'role': 'system',
        'content': '''You are a helpful question-answering assistant that bases its answers on facts from a knowledge base and always respects the prompt.

Process:

    You receive an input question.

    You determine the reasoning path needed to answer the question based on the information available.

    You explicitly list relevant facts, one per line, starting with "Fact:".

    You explain your reasoning step by step and provide a detailed answer, justifying it based on the facts.

    You conclude with a short and concise answer that must be grounded by the facts you found.

This knowledge base is complete. If a fact is not present we assume that facts is false.

After generating facts, please reason on top of the facts you find and avoid generating lot of useless facts.

You must always follow these instructions precisely and ensure your responses adhere strictly to this prompt.'''
    },
            {
                'role': 'user',
                'content': '''Which mountain is taller between Mont Blanc and Mount Rainier?
'''
            },
            {
                'role': 'assistant',
                'content': '''Reasoning: I need to provide the height of Mont Blanc and the height of Mount Rainier, then I need to compare the two heights and the final answer will be the taller mountain.
Fact: <Mont Blanc> <elevation above sea level> <4,807.02±0.5 meters> .
Now I need the height of Mount Rainier.
Fact: <Mount Rainier> <elevation above sea level> <4,389 meters> .
Long answer: Mont Blanc is 4,807 meters tall, while Mount Rainier is 4,389 meters, so Mont Blanc is taller than Mount Rainier.

Answer: Mont Blanc.
'''
            },
            {
                'role': 'user',
                'content': '''When was the director of Slumdog Millionaire born?
'''
            },
            {
                'role': 'assistant',
                'content': '''Reasoning: To answer this question, I need to find the birth date of the director of Slumdog Millionaire.
Fact: <Slumdog Millionaire> <director> <Danny Boyle>.
Now I need the birth date of Danny Boyle.
Fact: <Danny Boyle> <date of birth> <1956-10-20T00:00:00Z>.
Long answer: The director of Slumdog Millionaire is Danny Boyle, who was born on October 20, 1956, so this is the answer to the question.

Answer: 20-Oct-56.
'''
            },
            {
                'role': 'user',
                'content': '''When did John V, Prince Of Anhalt-Zerbst's father die?
'''
            },
            {
                'role': 'assistant',
                'content': '''Reasoning: To answer this question, I need to find the date of death of John V, Prince of Anhalt-Zerbst's father.
Fact: <John V, Prince of Anhalt-Zerbst> <father> <Ernest I, Prince of Anhalt-Dessau> .
Fact: <Ernest I, Prince of Anhalt-Dessau> <date of death> <1516-06-12T00:00:00Z> .
Long answer: The father of John V, Prince of Anhalt-Zerbst, Ernest I, Prince of Anhalt-Dessau, died on June 12, 1516.

Answer: June 12, 1516.
'''
            },
            {
                'role': 'user',
                'content': '''Is Johnny Depp older than Brad Pitt?
'''
            },
            {
                'role': 'assistant',
                'content': '''Reasoning: To determine if Johnny Depp is older than Brad Pitt, I need to find their respective birth years.
Fact: <Johnny Depp> <date of birth> <1963-06-09T00:00:00Z> .
Fact: <Brad Pitt> <date of birth> <1963-12-18T00:00:00Z> .
Long answer: Johnny Depp was born on June 9, 1963, while Brad Pitt was born on December 18, 1963. So yes, Johnny Depp is older than Brad Pitt.

Answer: Yes.
'''
    },
    {
        'role': 'user',
        'content': '''Has Hillary Clinton ever served as President of the United States?'''
    },
    {
        'role': 'assistant',
        'content': '''Reasoning: To answer this question, I need to find out if Hillary Clinton has ever held the position of President of the United States.
Fact: <Hillary Clinton> <position held> <United States Secretary of State> .
This fact tells me that Hillary Clinton was the United States Secretary of State, but it doesn't answer the question about her presidency.
Fact: <Hillary Clinton> <occupation> <Politician> .
This facts tells she is a Politician but it is not enough.
Fact: <Hillary Clinton> <occupation> <Diplomat> .
This facts tells she is a Diplomat but not that she served as President.
I don't have any facts that directly answer the question about Hillary Clinton's presidency, so I assume Hillary Clinton never served as President of the United States.
Long answer: Since there are no facts confirming that Hillary Clinton served as President of the United States, I assume she never did it, so the answer is no.

Answer: No.'''
    },
]

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