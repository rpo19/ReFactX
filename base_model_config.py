from transformers import AutoTokenizer, AutoModelForCausalLM
from ctrie import ConstrainedLogitsProcessor
import torch

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
                'content': '''You are a helpful question answering assistant that bases its answer on facts from a knowledge base.
1) You receive an input question.
2) You reason on the path you need to follow to reach the answer starting from the information in the question.
3) You explicitly provide relevant facts, one per line starting with "Fact:".
4) You explain your reasoning process and provide a long answer with your motivations based on the facts.
5) You provide a short concise answer.

Sometimes it could happen that there are no relevant facts in the knowledge base.
In these cases you either:
a) provide your own answer clearly stating that there were no supporting facts to your answer;
b) in case you don\'t know the answer or you are not confident enough, you just answer "I don't know".
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
            max_new_tokens = 400,
            pad_token_id = self.tokenizer.eos_token_id,
            use_cache = True,
            )
        self.batch_size = 3

        self.skip_serialize = set(['skip_serialize','tokenizer', 'model', 'index'])

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
