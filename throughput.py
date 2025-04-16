import torch
from ctrie import ConstrainedLogitsProcessor, ConstrainedStateList, ConstrainedState, DictIndex
import json
import importlib
import click
import time
from transformers import LogitsProcessor, LogitsProcessorList
from transformers import TextStreamer

class TimingLogitsProcessor(LogitsProcessor):
    def __init__(self): # considering only 1 constrained state
        self.timings = []
        self.last_time = None

    def __call__(self, input_ids, scores):
        now = time.time()
        if self.last_time is not None:
            elapsed = now - self.last_time
            self.timings.append(elapsed)
        self.last_time = now

        return scores

    def report(self):
        _report = {
            'timings': self.timings,
            'max': max(self.timings),
            'min': min(self.timings),
            'avg': sum(self.timings) / len(self.timings)
        }
        return _report

class AlwaysConstrainedLogitsProcessor(ConstrainedLogitsProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        assert input_ids.shape[0] == len(self.states), \
            f'Error: number of states ({len(self.states)}) should match `batch_size * num_beams` ({input_ids.shape[0]})'

        self.states.beam_permutation()

        for i in range(input_ids.shape[0]):
            if not self.states.beam_is_done(i):
                sequence = input_ids[i].tolist()

                if not self.states[i].first_call():
                    last_token = sequence[-1]
                    self.states[i].update(last_token)

                self.states[i].state = 1 # FORCE ALWAYS CONSTRAINED GENERATION

                if self.states[i].is_constrained(): # odd number means constrained generation
                    # constrained generation
                    constrain_generation_sequence = sequence[len(sequence) - self.states[i].get_cursor():]
                    scores[[i],:] = self.constrained_generation(
                        constrain_generation_sequence, scores[[i],:], state=self.states[i])

                # else:
                #     # normal generation
                #     # scores are not altered
                #     pass

        return scores

@click.command()
@click.option('--model', 'model_config_path', required=True, help="HuggingFace model to load.")
@click.option("--index", "index_config_path", required=True, help="Index configuration module (without .py).")
@click.option('--max-tokens', required=True, default=512, help="Tokens to generate.")
@click.option('--device-map', required=False, default='cuda', help="Where to load the model.")
@click.option("--output", "output_file", required=False, default=None, type=click.Path(), help="Output file for the results.")
@click.option("--unconstrained-generation", is_flag=True, help="Unconstrained generation")
@click.option("--debug", is_flag=True, help="Debug with streamer (Slow).")
@click.option('--torch-dtype', required=False, default='bfloat16', help="Torch dtype for loading the model.")
def main(model_config_path, index_config_path, max_tokens, device_map, output_file, unconstrained_generation, debug, torch_dtype):
    batch_size = 1
    nun_beams = 1

    prompt = [{
        'role': 'system',
        'content': 'You are an excellent story teller and your stories are very long.'
    },
    {
        'role': 'user',
        'content': 'Hello, tell me a story, please.'
    }]

    if index_config_path.endswith('.py'):
        index_config_path = index_config_path[:-3]
    index_module = importlib.import_module(index_config_path)
    index_config = getattr(index_module, 'index_config')

    if model_config_path.endswith('.py'):
        model_config_path = model_config_path[:-3]
    model_module = importlib.import_module(model_config_path)
    model_config = getattr(model_module, 'model_config')

    # going single batch single beam for throughput calculation
    model_config.batch_size = batch_size
    model_config.generate_args['num_beams'] = 1

    try:
        if index_config.rootkey <= max(model_config.tokenizer.vocab.values()):
            print('WARNING: rootkey could interfere with model tokens (if using postgres index)')
    except:
        print('WARNING: rootkey could interfere with model tokens (if using postgres index)')

    num_states = model_config.batch_size * model_config.generate_args.get('num_beams', 1)
    states = ConstrainedStateList(
        [ConstrainedState(
                begin_pattern = model_config.switch_pattern,
                end_pattern = model_config.newline_token,
                cache_index = DictIndex(end_of_triple=index_config.end_of_triple),
                subtree_cache = DictIndex(end_of_triple=index_config.end_of_triple),
                oneleaf_cache = DictIndex(end_of_triple=index_config.end_of_triple),
                state=1, # CONSTRAINED by default
            ) for _ in range(num_states)],
        num_beams=model_config.generate_args.get('num_beams', 1),
        batch_size = model_config.batch_size,
        pad_token_id = model_config.generate_args['pad_token_id'])

    constrained_processor = AlwaysConstrainedLogitsProcessor(
        index=index_config.index,
        end_token=model_config.newline_token,
        states=states,
        tokenizer=model_config.tokenizer
        )
    timingprocessor = TimingLogitsProcessor()
    logits_processor_list = LogitsProcessorList([
        constrained_processor,
        timingprocessor,
    ])
    if unconstrained_generation:
        logits_processor_list = LogitsProcessorList([timingprocessor])

    if debug:
        streamer = TextStreamer(model_config.tokenizer)
    else:
        streamer = None

    model_config.model.eval()
    with torch.no_grad():

        tokenized = model_config.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

        batch_inputs = model_config.tokenizer(
            tokenized,
            return_tensors='pt',
        ).to(model_config.model.device)

        output = model_config.model.generate(
            **batch_inputs,
            logits_processor=logits_processor_list,
            do_sample=True,
            temperature=2.,
            streamer=streamer,
            use_cache=True,
            max_new_tokens=max_tokens,
            eos_token_id=None,
            kwargs = {'constrained_state': states}, # passing state
        )

        states.beam_permutation() # final permutation to match final beams

    with open(output_file, 'w') as f:
        dump = {
            'config': dict(model_config_path=model_config_path, index_config_path=index_config_path, max_tokens=max_tokens, device_map=device_map, unconstrained_generation=unconstrained_generation, torch_dtype=torch_dtype),
            'timings': timingprocessor.report()
        }
        json.dump(dump, f)


if __name__ == "__main__":
    main()
