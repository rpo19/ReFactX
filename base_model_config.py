from transformers import AutoTokenizer, AutoModelForCausalLM
import re

class ModelConfig():
    def __init__(self, model_name, switch_pattern, newline_token, load_model = True, load_model_args = None, device_map = 'cuda',
                model_class = AutoModelForCausalLM):
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
                load_model_args = dict(device_map = device_map)
            self.model = model_class.from_pretrained(self.model_name,
                **load_model_args
                )
        self.switch_pattern = switch_pattern # "Fact:" after newline
        self.newline_token = newline_token
        # self.answer_tokens = answer_tokens # "Answer:" after newline
        self.eofanswer = [self.newline_token, self.tokenizer.eos_token_id]

        self.prompt_template = [
            {
                'role':'system',
                'content': '''You are a helpful question-answering assistant that bases its answers on facts from a knowledge base and always respects the prompt.

Process:

    You receive an input question.

    You determine the reasoning path needed to answer the question based on the information available.

    You explicitly list relevant facts, one per line, starting with "Fact:".

    You explain your reasoning step by step and provide a detailed answer, justifying it based on the facts.

    You conclude with a short and concise answer.

Handling missing facts:

    If no relevant facts are found in the knowledge base, you:
    a) Provide an answer based on general knowledge, explicitly stating that no supporting facts were found.
    b) If you do not know the answer or are not confident enough, simply reply: "I don't know."

You must always follow these instructions precisely and ensure your responses adhere strictly to this prompt.
'''
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

Answer: October 20, 1956.
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
            }]
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
        self.batch_size = 1

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
        prediction = None

        full_prediction = full_prediction.split(self.tokenizer.eos_token, 1)[0]
        if remove_dot and full_prediction.endswith('.'):
            full_prediction = full_prediction[:-len('.')]
        match = self.answer_pattern.search(full_prediction)
        if match:
            prediction = match.group(1)

        return prediction

