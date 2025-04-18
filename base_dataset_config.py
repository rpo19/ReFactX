from torch.utils.data import Dataset
from dotenv import load_dotenv
import os

load_dotenv()

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
        'content': 'Which mountain is taller between Mont Blanc and Mount Rainier?'
    },
    {
        'role': 'assistant',
        'content': '''Reasoning: I need to provide the height of Mont Blanc and the height of Mount Rainier, then I need to compare the two heights. The final answer will be the taller mountain.
Let's start with the description.
<tool_call>
{"name": "get_fact",  {"arguments": {"description": "Description of Mont Blanc"}}
</tool_call>
Fact: <Mont Blanc> <description> <highest mountain in the Alps> .
I found that Mont Blanc is the highest mountain in the Alps, but I didn't find its height.
<tool_call>
{"name": "get_fact",  {"arguments": {"description": "Description of Mount Rainier"}}
</tool_call>
Fact: <Mount Rainier> <description> <stratovolcano in the U.S. state of Washington> .
I found that Mount Rainier is a stratovolcano and its location. But I didn't find its height.
<tool_call>
{"name": "get_fact",  {"arguments": {"description": "Short description of Mont Blanc"}}
</tool_call>
Fact: <Mont Blanc> <short description> <Highest mountain in the Alps (4,808 m)> .
I found the height of Mont Blanc.
<tool_call>
{"name": "get_fact",  {"arguments": {"description": "Short description of Mount Rainier"}}
</tool_call>
Fact: <Mount Rainier> <short description> <Stratovolcano in the U.S. state of Washington> .
I found again that Mount Rainier is a stratovolcano, but I still need to find its height.
<tool_call>
{"name": "get_fact",  {"arguments": {"description": "Height of Mount Rainier"}}
</tool_call>
Fact: <Mount Rainier> <elevation above sea level> <4,389 meters> .
I found proof that Mont Blanc is 4,808 meters tall, while Mount Rainier is 4,389 meters, so Mont Blanc is taller than Mount Rainier. The answer is Mont Blanc.

Answer: Mont Blanc.'''
    },
    {
        'role': 'user',
        'content': 'When was the director of Slumdog Millionaire born?'
    },
    {
        'role': 'assistant',
        'content': '''Reasoning: To answer this question, I need to find who is the director of Slumdog Millionaire and then his birth date. The final answer will be his birth date.
Let's start with Slumdog Millionaire description.
<tool_call>
{"name": "get_fact",  {"arguments": {"description": "Description of Slumdog Millionaire"}}
</tool_call>
Fact: <Slumdog Millionaire> <description> <2008 film directed by Danny Boyle> .
I found that the director of Slumdog Millionaire is Danny Boyle. Now I need the birth date of Danny Boyle.
<tool_call>
{"name": "get_fact",  {"arguments": {"description": "Birth date of Danny Boyle"}}
</tool_call>
Fact: <Danny Boyle> <date of birth> <1956-10-20T00:00:00Z>.
I found proof that the director of Slumdog Millionaire is Danny Boyle, who was born on October 20, 1956, so this is the answer to the question.

Answer: October 20, 1956.'''
    },
    {
        'role': 'user',
        'content': 'Is Johnny Depp older than Brad Pitt?'
    },
    {
        'role': 'assistant',
        'content': '''Reasoning: To determine if Johnny Depp is older than Brad Pitt, I need to find their respective birth years. Then I need to compare them and determine who is older. The final answer will be yes if Johnny Depp is older, no elsewise. 
Let's start with the description.
<tool_call>
{"name": "get_fact",  {"arguments": {"description": "Description of Johnny Depp"}}
</tool_call>
Fact: <Johnny Depp> <description> <American actor (born 1963)> .
I found the year of birth of Johnny Depp. I need Brad Pitt's one.
<tool_call>
{"name": "get_fact",  {"arguments": {"description": "Description of Brad Pitt"}}
</tool_call>
Fact: <Brad Pitt> <description> <American actor and filmmaker> .
I cannot find Brad Pitt date of birth in the description.
<tool_call>
{"name": "get_fact",  {"arguments": {"description": "Short description of Brad Pitt"}}
</tool_call>
Fact: <Brad Pitt> <short description> <American actor (born 1963)> .
I found Brad Pitt's year of birth. But since it is the same as Johnny Depp's, I need more precise information about their date of birth.
<tool_call>
{"name": "get_fact",  {"arguments": {"description": "Johnny Depp birth date"}}
</tool_call>
Fact: <Johnny Depp> <date of birth> <1963-06-09T00:00:00Z> .
I found when Johnny Depp was born, now I need Brad Pitt's date of birth.
<tool_call>
{"name": "get_fact",  {"arguments": {"description": "Brad Pitt birth date"}}
</tool_call>
Fact: <Brad Pitt> <date of birth> <1963-12-18T00:00:00Z> .
I found proof that Johnny Depp was born on June 9, 1963, while Brad Pitt was born on December 18, 1963. So yes, Johnny Depp is older than Brad Pitt.

Answer: Yes.'''
    },
    {
        'role': 'user',
        'content': 'During which years did Napoleon hold power in France?'
    },
    {
        'role': 'assistant',
        'content': '''Reasoning: To determine the years Napoleon held power in France, I need to identify the periods during which he held power in France. Maybe I can get this information from Napoleon description. The answer will be the time frame in which Napoleon help power in France.
Let's start with the description.
<tool_call>
{"name": "get_fact", "arguments": {"description": "Napoleon description"}}
</tool_call>
Fact: <Napoleon Bonaparte> <description> <French military leader, French Emperor 1804–1814 and again in 1815> .
I found proof that Napoleon Bonaparte held power in France as Emperor from 1804 to 1814, and briefly again in 1815.

Answer: From 1799 to 1814, and in 1815.'''
    },
    {
        "role": "user",
        "content": "Which countries share a border with Spain?"
    },
    {
        "role": "assistant",
        "content": '''Reasoning: To answer this question, I need to find all the countries that have a land border with Spain. They may be several. The answer will be the list of all these countries.
Let's start from the description.
<tool_call>
{"name": "get_fact", "arguments": {"description": "Spain description"}}
</tool_call>
Fact: <Spain> <description> <country in southwestern Europe with territories in Africa> .
Now I know that some bordering countries are in Africa.
<tool_call>
{"name": "get_fact", "arguments": {"description": "Spain short description"}}
</tool_call>
Fact: <Spain> <short description> <Country in southwestern Europe> .
I need to find more specific facts.
<tool_call>
{"name": "get_fact", "arguments": {"description": "Country bordering Spain"}}
</tool_call>
Fact: <Spain> <shares border with> <Portugal> .
I found Portugal, now I continue to find all the others.
<tool_call>
{"name": "get_fact", "arguments": {"description": "Country bordering Spain"}}
</tool_call>
Fact: <Spain> <shares border with> <France> .
<tool_call>
{"name": "get_fact", "arguments": {"description": "Country bordering Spain"}}
</tool_call>
Fact: <Spain> <shares border with> <Andorra> .
I found Portugal, France, and Andorra. But let me see if it there are more countries that share a border with Spain. I also know there is at least one country in Africa.
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
        'content': '''In which books of A Song of Ice and Fire is Jon Snow not the Lord Commander of the Night's Watch?'''
    },
    {
        'role': 'assistant',
        'content': '''Reasoning: To answer this question, I need to consider all the books in the A Song of Ice and Fire series by George R.R. Martin and identify the ones where Jon Snow is not the Lord Commander of the Night's Watch. The answer will be the list of these books.
Let's start with the description.
<tool_call>
{"name": "get_fact", "arguments": {"description": "Description of A Song of Ice and Fire"}}
</tool_call>
Fact: <A Song of Ice and Fire> <description> <series of epic fantasy novels by George R. R. Martin> .
<tool_call>
{"name": "get_fact", "arguments": {"description": "Description of Jon Snow"}}
</tool_call>
Fact: <Jon Snow (character)> <description> <character in A Song of Ice and Fire> .
<tool_call>
{"name": "get_fact", "arguments": {"description": "Description of the Night's Watch"}}
</tool_call>
Now I have more context but I need more information. Let's try to find facts about Jon Snow being Lord Commander of the Night's Watch.
<tool_call>
{"name": "get_fact", "arguments": {"description": "Jon Snow lord commander of the Night's Watch"}}
</tool_call>
Fact: <Jon Snow (character)> <member of> <Night's Watch (A Song of Ice and Fire organization)> .
I found he was member of the Night's Watch but it's not enough.
<tool_call>
{"name": "get_fact", "arguments": {"description": "Jon Snow lord commander of the Night's Watch"}}
</tool_call>
Fact: <Jon Snow (character)> <occupation> <Soldier> .
I found he was a soldier but it's not relevant.
<tool_call>
{"name": "get_fact", "arguments": {"description": "Jon Snow lord commander of the Night's Watch"}}
</tool_call>
Fact: <Jon Snow (character)> <occupation> <ruler (person who reigns over a certain region or country)> .
I found he was a ruler but it's not specifically related to the Night's Watch.
I'm not able to find the facts about the books in which Jon Snow is not the Lord Commander.
I didn't find any proof to answer the question. I don't know the answer.

Answer: I don't know.'''
    },
    {
        'role': 'user',
        'content': '''Has Hillary Clinton ever served as President of the United States?'''
    },
    {
        'role': 'assistant',
        'content': '''Reasoning: To answer this question, I need to find proofs that Hillary Clinton has ever held the position of President of the United States. If yes, the answer will be yes.
Let's start from the description.
<tool_call>
{"name": "get_fact", "arguments": {"description": "Hillary Clinton description"}}
</tool_call>
Fact: <Hillary Clinton> <description> <American politician and diplomat (born 1947)> .
The description doesn't tell if she has been President of the United States.
<tool_call>
{"name": "get_fact", "arguments": {"description": "Hillary Clinton short description"}}
</tool_call>
Fact: <Hillary Clinton> <short description> <American politician and diplomat (born 1947)> .
Also the short description doesn't tell if she has been President of the United States. I need more specific facts.
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
]

class QADataset(Dataset):
    def dump(self):
        return self.dataset

    def preprocess(self, dataset):
        return dataset

    def __init__(self, dataset, config, preprocess=True, name='QADataset'):
        self.dataset = dataset
        self.config = config
        if preprocess:
            self.dataset = self.preprocess(self.dataset)
            # now self.dataset should be a list
            dataset_start_from = int(os.environ.get('DATASET_START_FROM', 0))
            if dataset_start_from:
                print(f"Skipping first {dataset_start_from} samples from dataset. Previous length: {len(self.dataset)}, current length: {len(self.dataset) - dataset_start_from}")
                self.dataset = self.dataset[dataset_start_from:]
                config['start_from'] = dataset_start_from

        self.prompt_template = PROMPT_TEMPLATE

        self.skip_serialize = set(['skip_serialize','dataset'])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self.__class__(self.dataset[key], self.config)
        else:
            return self.dataset[key]

    def __iter__(self):
        for sample in self.dataset:
            yield sample

    def dump_config(self):
        keys = set(self.__dict__.keys())
        keys.difference_update(self.skip_serialize)
        for key in keys:
            yield (key, self.__dict__[key])

    def questions_dataset(self):
        # Dynamically create a subclass of the calling class
        class DynamicQuestionsDataset(self.__class__):
            def __init__(dataset_self, dataset, config, preprocess=False):
                super().__init__(dataset, config, preprocess)

            def __getitem__(dataset_self, idx):
                if isinstance(idx, slice):
                    config = self.config.copy()
                    config['start_from'] = idx.start
                    return DynamicQuestionsDataset(dataset_self.dataset[idx], dataset_self.config, preprocess=False)
                else:
                    return dataset_self.get_question(idx)

            def __iter__(dataset_self):
                for sample in super().__iter__():
                    yield dataset_self.get_question_from_sample(sample)

        # Return an instance of the dynamically created class
        return DynamicQuestionsDataset(self.dataset, self.config, preprocess=False)

    def get_answer(self, i) -> str:
        answer = self.dataset[i]['answer']
        return answer

    def get_question_from_sample(self, sample) -> str:
        return sample['question']

    def get_question(self, i) -> str:
        sample = self.dataset[i]
        return self.get_question_from_sample(sample)

    def get_question_type(self, i):
        return 'unknown'

    '''
    Usage:
    ```
        from datasets import Dataset
        ds = Dataset.from_generator(QADataset.to_verl(prompt_fn))
        ds.to_parquet(path)
    ```
    '''
    def to_verl(self, prompt_fn=None, name=None):
        if prompt_fn is None:
            prompt_fn = lambda question: question

        if name is None:
            name = self.name

        for i in range(len(self)):
            question = self.get_question(i)
            prompt = prompt_fn(question)
            # TODO system prompt
            data = {
                "data_source": name,
                "prompt": [{
                    "role": "user",
                    "content": prompt,
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": self.get_answer(i)
                },
                "extra_info": {
                    'index': i,
                    "question": question,
                }
            }
            yield data

    @staticmethod
    def get_dataset_path(dataset_name):
        DATA_PATH = os.getenv('DATA_PATH', './data')
        path = os.path.join(DATA_PATH, dataset_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found at {path}. Please check the path.")
        return path
