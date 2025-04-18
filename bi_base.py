from base_dataset_config import QADataset
from zipfile import ZipFile
import os
import pandas as pd

# TODO this dataset assumes it is complete: if there is no evidence that proves something, then it is false.

# different prompt
# there are no description and short description
# TODO add example for multiple answers. e.g. which entities (more than one) own a company? 
PROMPT_TEMPLATE = [
    {
        'role': 'system',
        'content': '''You are a helpful question-answering assistant that bases its answers on facts from a knowledge base and always respects the prompt.

# Tools

You have to call the get_fact function to answer the questions.

You are provided with the function signature within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "get_fact", "description": "Gets a fact relevant for the question.", "parameters": {"type": "object", "properties": {"description": {"type": "string", "description": "The description of the fact to get"}}, "required": ["description"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": "get_fact",  {"arguments": {"description": "Description of the fact to get"}}
</tool_call>

The process to answer questions:

    You receive an input question.

    You determine the reasoning path needed to answer the question based on the information available.

    You detemine the kind of answer you are asked. It can be a yes/no, a single entity, or a list of entities. Pay attention to the questions whose answer is a list of entities (e.g. Which countries share a border with Spain?): you need to find all the answer entities and include them all in the final answer.

    You get relevant facts with the "get_fact" function.

    You always get some general context about the entities in the questions with "get_fact" asking for a description (or a short description).

    While getting facts you continue the reasoning explaining it step by step.

    You conclude with a concise answer that depending on the question can be a yes/no, a single entity, or a list of entities. Pay attention to the questions whose answer is a list of entities.

    The answer MUST be based on the proofs you found with "get_fact".

If you didn't find proofs with "get_fact" that support an answer you stop and you reply: "I don't know.".

If the question requires to find proof that an event happen and you didn't find any proof, you can assume that event didn't happen.

In case you are taking too long for answering, you stop and you answer based on the proofs you acquired to that point.

You must always follow these instructions precisely and ensure your responses adhere strictly to this prompt.'''
    },
            {
                'role': 'user',
                'content': '''Which mountain is taller between Mont Blanc and Mount Rainier?
'''
            },
            {
                'role': 'assistant',
                'content': '''Reasoning: I need to provide the height of Mont Blanc and the height of Mount Rainier, then I need to compare the two heights. The final answer will be the taller mountain.
<tool_call>
{"name": "get_fact",  {"arguments": {"description": "Height of Mont Blanc"}}
</tool_call>
Fact: <Mont Blanc> <elevation above sea level> <4,807.02±0.5 meters> .
Now I need the height of Mount Rainier.
<tool_call>
{"name": "get_fact",  {"arguments": {"description": "Height of Mount Rainier"}}
</tool_call>
Fact: <Mount Rainier> <elevation above sea level> <4,389 meters> .
I found proof that Mont Blanc is 4,808 meters tall, while Mount Rainier is 4,389 meters, so Mont Blanc is taller than Mount Rainier. The answer is Mont Blanc.

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
                'content': '''Reasoning: To answer this question, I need to find who is the director of Slumdog Millionaire and then his birth date. The final answer will be his birth date.
<tool_call>
{"name": "get_fact",  {"arguments": {"description": "Director of Slumdog Millionaire"}}
</tool_call>
Fact: <Slumdog Millionaire> <director> <Danny Boyle>.
Now I need the birth date of Danny Boyle.
<tool_call>
{"name": "get_fact",  {"arguments": {"description": "Birth date of Danny Boyle"}}
</tool_call>
Fact: <Danny Boyle> <date of birth> <1956-10-20T00:00:00Z>.
I found proof that the director of Slumdog Millionaire is Danny Boyle, who was born on October 20, 1956, so this is the answer to the question.

Answer: October 20, 1956.
'''
            },
            {
                'role': 'assistant',
                'content': '''Reasoning: To determine if Johnny Depp is older than Brad Pitt, I need to find their respective birth years. Then I need to compare them and determine who is older. The final answer will be yes if Johnny Depp is older, no elsewise.
<tool_call>
{"name": "get_fact", "arguments": {"description": "Birth date of Johnny Depp"}}
</tool_call>
Fact: <Johnny Depp> <date of birth> <1963-06-09T00:00:00Z> .
Now I need the birth date of Brad Pitt.
<tool_call>
{"name": "get_fact", "arguments": {"description": "Birth date of Brad Pitt"}}
</tool_call>
Fact: <Brad Pitt> <date of birth> <1963-12-18T00:00:00Z> .
I found proof that Johnny Depp was born on June 9, 1963, while Brad Pitt was born on December 18, 1963. So yes, Johnny Depp is older than Brad Pitt.

Answer: Yes.
'''
    },
    {
        "role": "user",
        "content": "Which countries share a border with Spain?"
    },
    {
        "role": "assistant",
        "content": '''Reasoning: To answer this question, I need to find all the countries that have a land border with Spain. They may be several. The answer will be the list of all these countries.
<tool_call>
{"name": "get_fact", "arguments": {"description": "Country bordering Spain"}}
</tool_call>
Fact: <Spain> <shares border with> <Portugal> .
<tool_call>
{"name": "get_fact", "arguments": {"description": "Country bordering Spain"}}
</tool_call>
Fact: <Spain> <shares border with> <France> .
<tool_call>
{"name": "get_fact", "arguments": {"description": "Country bordering Spain"}}
</tool_call>
Fact: <Spain> <shares border with> <Andorra> .
I found Portugal, France, and Andorra. But let me see if it there are more countries that share a border with Spain.
<tool_call>
{"name": "get_fact", "arguments": {"description": "Country bordering Spain"}}
</tool_call>
Fact: <Spain> <shares border with> <Gibraltar> .
<tool_call>
{"name": "get_fact", "arguments": {"description": "Country bordering Spain"}}
</tool_call>
Fact: <Spain> <shares border with> <Morocco> .
<tool_call>
{"name": "get_fact", "arguments": {"description": "Country bordering Spain"}}
</tool_call>
Fact: <Spain> <diplomatic relation> <Italy> .
This fact is about diplomatic relations and not borders.
<tool_call>
{"name": "get_fact", "arguments": {"description": "Country bordering Spain"}}
</tool_call>
Fact: <Spain> <diplomatic relation> <Greece> .
Also this fact is not about borders. I'm not able to find other countries bordering with Spain.
I found proof that Spain shares borders with Portugal, France, Andorra, Gibraltar, and Morocco.

Answer: Portugal, France, Andorra, Gibraltar, Morocco.'''
    },
    {
        'role': 'user',
        'content': '''Has Hillary Clinton ever served as President of the United States?'''
    },
    {
        'role': 'assistant',
        'content': '''Reasoning: To answer this question, I need to find out if Hillary Clinton has ever held the position of President of the United States. If yes, the answer will be yes.
<tool_call>
{"name": "get_fact", "arguments": {"description": "Hillary Clinton president of the United States"}}
</tool_call>
Fact: <Hillary Clinton> <position held> <United States Secretary of State> .
This fact tells me that Hillary Clinton was the United States Secretary of State, but it doesn't answer the question about her presidency.
<tool_call>
{"name": "get_fact", "arguments": {"description": "Hillary Clinton president of the United States"}}
</tool_call>
Fact: <Hillary Clinton> <occupation> <Politician> .
This facts tells she is a Politician but it is not enough.
<tool_call>
{"name": "get_fact", "arguments": {"description": "Hillary Clinton president of the United States"}}
</tool_call>
Fact: <Hillary Clinton> <occupation> <Diplomat> .
This facts tells she is a Diplomat but not that she served as President.
I didn't find any proof that Hillary Clinton has served as president of the United States, so I assume Hillary Clinton never served. The answer is no.

Answer: No.'''
    },
    {
        'role': 'user',
        'content': 'During which years did Napoleon hold power in France?'
    },
    {
        'role': 'assistant',
        'content': '''Reasoning: To determine the years Napoleon held power in France, I need to identify the periods during which he held power in France. Maybe I can get this information from Napoleon description. The answer will be the time frame in which Napoleon help power in France. 
<tool_call>
{"name": "get_fact", "arguments": {"description": "Napoleon description"}}
</tool_call>
Fact: <Napoleon Bonaparte> <description> <French military leader, French Emperor 1804–1814 and again in 1815> .
I found proof that Napoleon Bonaparte held power in France as Emperor from 1804 to 1814, and briefly again in 1815.

Answer: From 1799 to 1814, and in 1815.'''
    },
    {
        'role': 'user',
        'content': '''In which books of A Song of Ice and Fire is Jon Snow not the Lord Commander of the Night's Watch?'''
    },
    {
        'role': 'assistant',
        'content': '''Reasoning: To answer this question, I need to consider all the books in the A Song of Ice and Fire series by George R.R. Martin and identify the ones where Jon Snow is not the Lord Commander of the Night's Watch. The answer will be the list of these books.
<tool_call>
{"name": "get_fact", "arguments": {"description": "Book in which Jon Snow is not the Lord Commander of the Night's Watch"}}
</tool_call>
Fact: <A Game of Thrones> <description> <1996 novel by George R. R. Martin> .
This fact doesn't give me any information about Jon Snow.
<tool_call>
{"name": "get_fact", "arguments": {"description": "Book in which Jon Snow is not the Lord Commander of the Night's Watch"}}
</tool_call>
Fact: <A Game of Thrones> <publication date> <1996-08-06T00:00:00Z> .
This fact is not giving any information about Jon Snow. I think the publication date won't help me.
<tool_call>
{"name": "get_fact", "arguments": {"description": "Book in which Jon Snow is not the Lord Commander of the Night's Watch"}}
</tool_call>
Fact: <A Clash of Kings> <publication date> <1998-11-16T00:00:00Z> .
These facts don't contain any information about Jon Snow. I'll try one more time.
<tool_call>
{"name": "get_fact", "arguments": {"description": "Book in which Jon Snow is not the Lord Commander of the Night's Watch"}}
</tool_call>
Fact: <A Storm of Swords> <publication date> <2000-08-08T00:00:00Z> .
I'm not able to find the facts about the books in which Jon Snow is not the Lord Commander.
I didn't find any proof to answer the question. I don't know the answer.

Answer: I don't know.'''
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