from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from ctrie import ConstrainedLogitsProcessor
import torch

class Model():
    def __init__(self):
        self.model_name =  'microsoft/Phi-3-mini-128k-instruct'
        print(f'Loading {self.model_name}')
        self.device = 'cuda:0'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = '<0x0A>' # use padding right with newline as padding token
        self.quantization_config = dict(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype='bfloat16')
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name,
            trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(**self.quantization_config),
            low_cpu_mem_usage=True,
            # attn_implementation="flash_attention_2"
            ).to(self.device)
        self.switch_pattern = [20738, 29901]
        self.newline_token = 13
        self.answer_tokens = [22550, 29901]
        self.eofanswer = [self.newline_token, self.tokenizer.eos_token_id]
        self.early_stop_token = 29958 # >

        self.prompt_template = ('''<|system|>
You are a helpful question answering assistant that bases its answer on facts from a knowledge base.
1) You receive an input question.
2) You reason on the path you need to follow to reach the answer starting from the information in the question.
3) You provide the relevant facts useful to reach the answer and you reason on top of them.
4) You explain your reasoning process and provide a long answer with your motivations.
5) You provide a short concise answer.
<|end|>

<|user|>
Which mountain is taller between Mont Blanc and Mount Rainier?
<|end|>

<|assistant|>
Reasoning: I need to provide the height of Mont Blanc and the height of Mount Rainier, then I need to compare the two heights and the final answer will be the taller mountain.
Fact: <Mont Blanc> <elevation above sea level> <4,807.02±0.5 meters> .
I found the height of Mont Blanc. I still need the height of Mount Rainier.
Fact: <Mount Rainier> <elevation above sea level> <4,389 meters> .
I also found the height of Mount Rainier. Now I can compare the heights and provide an answer.
Long answer: Mont Blanc is 4,807 meters tall, while Mount Rainier is 4,389 meters, so Mont Blanc is taller than Mount Rainier.
Final answer: Mont Blanc.
<|end|>

<|user|>
''','''
<|end|>

<|assistant|>
''')

        self.generate_args = dict(
            num_beams = 3,
            num_return_sequences = 1,
            # do_sample = True,
            # top_k = 3,
            max_new_tokens = 400,
            )
        self.batch_size = 3

        self.skip_serialize = set(['skip_serialize','tokenizer', 'model', 'index'])

    def __iter__(self):
        keys = set(self.__dict__.keys())
        keys.difference_update(self.skip_serialize)
        for key in keys:
            yield (key, self.__dict__[key])

    def apply_prompt_template(self, question):
        return self.prompt_template[0] + question + self.prompt_template[1]

    def tokenize_fun(self, questions):
        return self.tokenizer(
            questions,
            return_tensors='pt',
            padding=True,
            padding_side='right'
        ).to(self.model.device)

