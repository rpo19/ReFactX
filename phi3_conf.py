from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from ctrie import ConstrainedLogitsProcessor
import torch

class Model():
    def __init__(self):
        self.model_name =  "microsoft/Phi-3-mini-128k-instruct"
        print(f'Loading {self.model_name}')
        self.device = "cuda"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
            )
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name,
            trust_remote_code=True,
            quantization_config=self.quantization_config,
            low_cpu_mem_usage=True,
            # attn_implementation="flash_attention_2"
            ).to(self.device)
        self.switch_pattern = [20738, 29901]
        self.newline_token = 13
        self.answer_tokens = [22550, 29901]
        self.eofanswer = self.newline_token
        self.early_stop_token = 29958 # >

        self.prompt_template = '''You are a question-answering system that reasons using structured data in the form of facts.
Given an input question, you generate a concise single answer based on knowledge facts.
Follow this format:

Question: The question to be answered.
Facts for the reasoning process: some facts containing entities, relationships, and values relevant to the question.
Long answer: the reasoning process you followed to reach the answer also based on the facts.
Answer: the concise answer.

Example:
Question: Is Mont Blanc taller than Mount Rainier?
Facts for the reasoning process:
Fact: <Mont Blanc> <elevation above sea level> <4,807.02±0.5 metre> .
Fact: <Mount Rainier> <elevation above sea level> <4,389 metre> .
Long answer: Basing on the evidence that the elevation above sea level of Mont Blanc (4,807.02±0.5 metres) is greater than the elevation above sea level of Mount Rainier (4,389 metres), Mont Blanc is taller than Mount Rainier.
Answer: Yes, Mont Blanc is taller than Mount Rainier.

As you can see in the example, triples generally start with information contained in the question and provide additional information.
Unfortunately, some of the retrieved facts may irrelevant. You should ignore these irrelevant triples.


Now, answer the following question:
Question: {}
Triples for the reasoning process:
Fact:'''

        self.generate_args = dict(
            num_beams = 3,
            num_return_sequences = 1,
            # do_sample = True,
            # top_k = 3,
            max_new_tokens = 300,
            use_cache = True)
        self.batch_size = 4

