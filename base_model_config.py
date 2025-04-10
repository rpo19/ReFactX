from transformers import AutoTokenizer, AutoModelForCausalLM
import re

class ModelConfig():
    def __init__(self, model_name, switch_pattern, newline_token, load_model = True, load_model_args = None, device_map = 'cuda',
                model_class = AutoModelForCausalLM, torch_dtype='bfloat32'):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # self.tokenizer.pad_token = 'Ċ' # use padding right with newline as padding token
        # self.quantization_config = dict(
        #     load_in_4bit=True,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type='nf4',
        #     bnb_4bit_compute_dtype='bfloat16')
        self.model_class = model_class
        self.model_class_name = str(model_class)
        if load_model:
            print(f'Loading {self.model_name}')
            if load_model_args is None:
                load_model_args = {}
            if 'device_map' not in load_model_args and device_map is not None:
                load_model_args['device_map'] = device_map
            self.model = model_class.from_pretrained(self.model_name,
                **load_model_args
                )
        self.switch_pattern = switch_pattern # "Fact:" after newline
        self.newline_token = newline_token
        # self.answer_tokens = answer_tokens # "Answer:" after newline
        self.eofanswer = [self.newline_token, self.tokenizer.eos_token_id]

        self.prompt_template = [
    {
        'role': 'system',
        'content': '''You are a helpful question-answering assistant that bases its answers on facts from a knowledge base and always respects the prompt.

Process:

    You receive an input question.

    You get some context about the entities in the questions using description and short description.

    You determine the reasoning path needed to answer the question based on the information available.

    You explicitly list relevant facts, one per line, starting with "Fact:".

    You explain your reasoning step by step and provide a detailed answer, justifying it based on the facts.

    You conclude with a short and concise answer that must be grounded by the facts you found.

You have to answer only basing on the facts you find in the knowledge base.
If you cannot find relevant facts in the knowledge base to support an answer you stop and you reply: "I don't know.".
After generating facts, please reason on top of the facts you find and avoid generating lot of useless facts.

You must always follow these instructions precisely and ensure your responses adhere strictly to this prompt.'''
    },
    {
        'role': 'user',
        'content': 'Which mountain is taller between Mont Blanc and Mount Rainier?'
    },
    {
        'role': 'assistant',
        'content': '''Reasoning: I need to provide the height of Mont Blanc and the height of Mount Rainier, then I need to compare the two heights and the final answer will be the taller mountain.
Let's try to get a bit of context about these two mountains starting with the descriptions.
Fact: <Mont Blanc> <description> <highest mountain in the Alps> .
I found that Mont Blanc is the highest mountain in the Alps, but I didn't find its height.
Fact: <Mount Rainier> <description> <stratovolcano in the U.S. state of Washington> .
I found that Mount Rainier is a stratovolcano and its location. But I didn't find its height.
Fact: <Mont Blanc> <short description> <Highest mountain in the Alps (4,808 m)> .
I found the height of Mont Blanc.
Fact: <Mount Rainier> <short description> <Stratovolcano in the U.S. state of Washington> .
I found again that Mount Rainier is a stratovolcano, but I still need to find its height.
Fact: <Mount Rainier> <elevation above sea level> <4,389 meters> .
I found Mount Rainier elevation above sea level.
Now that I know the height of both mountains I can give a grounded answer.
Long answer: Mont Blanc is 4,808 meters tall, while Mount Rainier is 4,389 meters, so Mont Blanc is taller than Mount Rainier.

Answer: Mont Blanc.'''
    },
    {
        'role': 'user',
        'content': 'When was the director of Slumdog Millionaire born?'
    },
    {
        'role': 'assistant',
        'content': '''Reasoning: To answer this question, I need to find the birth date of the director of Slumdog Millionaire. Let's start with Slumdog Millionaire description.
Fact: <Slumdog Millionaire> <description> <2008 film directed by Danny Boyle> .
I found that the director of Slumdog Millionaire is Danny Boyle. Now I need the birth date of Danny Boyle.
Fact: <Danny Boyle> <date of birth> <1956-10-20T00:00:00Z>.
Now I have all the information I need for answering.
Long answer: The director of Slumdog Millionaire is Danny Boyle, who was born on October 20, 1956, so this is the answer to the question.

Answer: October 20, 1956.'''
    },
    {
        'role': 'user',
        'content': "When did John V, Prince Of Anhalt-Zerbst's father die?"
    },
    {
        'role': 'assistant',
        'content': '''Reasoning: To answer this question, I need to find who the father of John V, Prince of Anhalt-Zerbst, was and then find the date of his death. Let's start with the description.
Fact: <John V, Prince of Anhalt-Zerbst> <description> <Prince of Anhalt-Dessau and Anhalt-Zerbst> .
Ok, I still need to find his father.
Fact: <John V, Prince of Anhalt-Zerbst> <father> <Ernest I, Prince of Anhalt-Dessau> .
Now I know who is the father. I need to find when he died.
Fact: <Ernest I, Prince of Anhalt-Dessau> <date of death> <1516-06-12T00:00:00Z> .
Now I have all the information I need to give a grounded answer.
Long answer: The father of John V, Prince of Anhalt-Zerbst, Ernest I, Prince of Anhalt-Dessau, died on June 12, 1516.

Answer: June 12, 1516.'''
    },
    {
        'role': 'user',
        'content': 'Is Johnny Depp older than Brad Pitt?'
    },
    {
        'role': 'assistant',
        'content': '''Reasoning: To determine if Johnny Depp is older than Brad Pitt, I need to find their respective birth years. Let's start with the description.
Fact: <Johnny Depp> <description> <American actor (born 1963)> .
I found the year of birth of Johnny Depp. I need Brad Pitt's one.
Fact: <Brad Pitt> <description> <American actor and filmmaker> .
I cannot find Brad Pitt date of birth in the description.
Fact: <Brad Pitt> <short description> <American actor (born 1963)> .
I found Brad Pitt's year of birth. But since it is the same as Johnny Depp's, I need more precise information about their date of birth.
Fact: <Johnny Depp> <date of birth> <1963-06-09T00:00:00Z> .
I found when Johnny Depp was born, now I need Brad Pitt's date of birth.
Fact: <Brad Pitt> <date of birth> <1963-12-18T00:00:00Z> .
Now I have all the information I need to give a grounded answer.
Long answer: Johnny Depp was born on June 9, 1963, while Brad Pitt was born on December 18, 1963. So yes, Johnny Depp is older than Brad Pitt.

Answer: Yes.'''
    },
    {
        'role': 'user',
        'content': 'During which years did Napoleon hold power in France?'
    },
    {
        'role': 'assistant',
        'content': '''Reasoning: To determine the years Napoleon held power in France, I need to identify the periods during which he held official leadership roles such as First Consul or Emperor. Maybe I can get this information from the description.
Fact: <Napoleon Bonaparte> <description> <French military leader, French Emperor 1804–1814 and again in 1815> .
Now I have all the information I need to give a grounded answer.
Long answer: Based on the description, Napoleon Bonaparte held power in France as Emperor from 1804 to 1814, and briefly again in 1815.

Answer: From 1799 to 1814, and in 1815.'''
    },
    {
        'role': 'user',
        'content': '''In which books of A Song of Ice and Fire is Jon Snow not the Lord Commander of the Night's Watch?'''
    },
    {
        'role': 'assistant',
        'content': '''Reasoning: To answer this question, I need to consider the books in the A Song of Ice and Fire series by George R.R. Martin and identify the ones where Jon Snow is not the Lord Commander of the Night's Watch. Let's start with the description.
Fact: <A Game of Thrones> <description> <1996 novel by George R. R. Martin> .
This fact doesn't give me any information about Jon Snow.
Fact: <A Game of Thrones> <publication date> <1996-08-06T00:00:00Z> .
This fact is not giving any information about Jon Snow. I think the publication date won't help me.
Fact: <A Clash of Kings> <publication date> <1998-11-16T00:00:00Z> .
These facts don't contain any information about Jon Snow. I'll try one more time.
Fact: <A Storm of Swords> <publication date> <2000-08-08T00:00:00Z> .
I'm not able to find the facts about the books in which Jon Snow is not the Lord Commander, so I cannot give a grounded answer.
Long answer: I don't know. I was not able to find the relevant facts in the knowledge base to motivate an answer.

Answer: I don't know.'''
    },
]


        self.tokenizer_args = dict(
            padding=True,
            padding_side='left'
        )
        self.generate_args = dict(
            num_beams = 3,
            num_return_sequences = 1,
            do_sample = False,
            temperature = None,
            top_k = None,
            top_p = None,
            max_new_tokens = 800,
            pad_token_id = self.tokenizer.eos_token_id,
            use_cache = True,
            )
        self.batch_size = 8

        self.answer_pattern = re.compile(r'Answer: (.*)\.?')

        self.skip_serialize = set(['skip_serialize','tokenizer', 'model', 'index', 'answer_pattern', 'model_class'])

    def apply_prompt_template(self, question=None):
        if question is None:
            # only prompt for caching
            return self.tokenizer.apply_chat_template(self.prompt_template, tokenize=False, add_generation_prompt=False)
        else:
            question_w_role = {'role':'user', 'content': question}
            return self.tokenizer.apply_chat_template(self.prompt_template + [question_w_role], tokenize=False, add_generation_prompt=True)

    def tokenize_fun(self, questions):
        return self.tokenizer(
            questions,
            return_tensors='pt',
            **self.tokenizer_args,
        ).to(self.model.device)

    def __iter__(self):
        keys = set(self.__dict__.keys())
        keys.difference_update(self.skip_serialize)
        for key in keys:
            yield (key, self.__dict__[key])

    def get_prediction(self, full_prediction, remove_dot=True):
        prediction = ''

        full_prediction = full_prediction.split(self.tokenizer.eos_token, 1)[0]
        if remove_dot and full_prediction.endswith('.'):
            full_prediction = full_prediction[:-len('.')]
        match = self.answer_pattern.search(full_prediction)
        if match:
            prediction = match.group(1)

        return prediction


