from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import torch

class ModelConfig():
    def load_model(self, load_model_args = None, device_map = 'cuda', torch_dtype=torch.float32):
        if isinstance(torch_dtype, str):
            torch_dtype = getattr(torch, torch_dtype)
        print(f'Loading {self.model_name}')
        if load_model_args is None:
            load_model_args = {}
        if 'device_map' not in load_model_args and device_map is not None:
            load_model_args['device_map'] = device_map
        if 'torch_dtype' not in load_model_args and torch_dtype is not None:
            load_model_args['torch_dtype'] = torch_dtype
        self.model = self.model_class.from_pretrained(self.model_name,
            **load_model_args
            )
    def __init__(self, model_name, switch_pattern, newline_token, load_model = True, load_model_args = None, device_map = 'cuda',
                model_class = AutoModelForCausalLM, torch_dtype=torch.float32):
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
        self.load_model_args = load_model_args
        self.model_class_name = str(model_class)
        if load_model:
            self.load_model(self.load_model_args, device_map, torch_dtype)
        self.switch_pattern = switch_pattern # "Fact:" after newline
        self.newline_token = newline_token
        # self.answer_tokens = answer_tokens # "Answer:" after newline
        self.eofanswer = [self.newline_token, self.tokenizer.eos_token_id]

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

        self.skip_serialize = set(['skip_serialize','tokenizer', 'model', 'index', 'answer_pattern', 'model_class', 'load_model_args'])

    def apply_prompt_template(self, prompt_template, question=None):
        if question is None:
            # only prompt for caching
            return self.tokenizer.apply_chat_template(prompt_template, tokenize=False, add_generation_prompt=False)
        else:
            question_w_role = {'role':'user', 'content': question}
            return self.tokenizer.apply_chat_template(prompt_template + [question_w_role], tokenize=False, add_generation_prompt=True)

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


