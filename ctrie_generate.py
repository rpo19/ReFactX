import click
import torch
import importlib
import time
from transformers import TextStreamer
from transformers.generation.logits_process import LogitsProcessorList
from ctrie import ConstrainedLogitsProcessor, ConstrainedStateList, ConstrainedState, DictIndex
import sys

@click.command()
@click.option('--index', 'index_config_path', default='qwen25_index', help='Index module to import.')
@click.option('--model', 'model_config_path', default='qwen25_1B_model', help='Model module to import.')
@click.option('--question', default=None, help='Question to process (if any: interactive mode).')
@click.option('--num-beams', default=1, help='Number of beams for beam search.')
@click.option('--generation-config', 'generation_config_str', default="max_new_tokens=512,do_sample=False,temperature=None,top_k=None", help='Generation config (e.g. "max_new_tokens=512,top_k=5").')
def main(index_config_path, model_config_path, question, num_beams, generation_config_str):
    try:
        generation_config = eval(f'dict({generation_config_str})')
    except Exception as e:
        print(f"Error parsing generation config: {e}")
        sys.exit(1)

    if index_config_path.endswith('.py'):
        index_config_path = index_config_path[:-3]
    index_module = importlib.import_module(index_config_path)
    index_config = getattr(index_module, 'index_config')

    if model_config_path.endswith('.py'):
        model_config_path = model_config_path[:-3]
    model_module = importlib.import_module(model_config_path)
    model_config = getattr(model_module, 'model_config')

    streamer = TextStreamer(model_config.tokenizer)

    auto_streamer = streamer if num_beams == 1 else None

    states = ConstrainedStateList(
        [ConstrainedState(
                    begin_pattern=model_config.switch_pattern,
                    end_pattern=model_config.newline_token,
                    cache_index=DictIndex(end_of_triple=index_config.index.end_of_triple),
                    subtree_cache=DictIndex(end_of_triple=index_config.index.end_of_triple),
                    oneleaf_cache=DictIndex(end_of_triple=index_config.index.end_of_triple)
                ) for _ in range(num_beams)],
                num_beams=num_beams,
                batch_size=1,
                pad_token_id=model_config.tokenizer.eos_token_id)

    constrained_processor = ConstrainedLogitsProcessor(
        index=index_config.index,
        end_token=model_config.newline_token, states=states, tokenizer=model_config.tokenizer)
    logits_processor_list = LogitsProcessorList([
        constrained_processor
    ])

    model_config.model.eval()

    interactive_mode = question is None

    while True:
        if interactive_mode:
            print('Insert question. (CTRL+C to exit).')
            question = input('>')

        prompted_texts = [model_config.apply_prompt_template(question)]
        print(prompted_texts[0])

        inputs = model_config.tokenizer(prompted_texts, return_tensors='pt', padding=True, padding_side='right')
        inputs = inputs.to(model_config.model.device)

        start = time.time()

        with torch.no_grad():
            out = model_config.model.generate(
                **inputs,
                logits_processor=logits_processor_list,
                streamer=auto_streamer,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                use_cache=True,
                kwargs={'constrained_state': states},  # passing state
                **generation_config,
            )

        torch.cuda.empty_cache()

        print('Elapsed', time.time() - start)

        # print beam search results
        _from = len(inputs.input_ids[0])
        for i in range(out.shape[0]):
            print('-' * 30, sum(out[i][_from:]), len(out[i][_from:]))
            print(model_config.tokenizer.decode(out[i][_from:]))

        # print triples
        for i, triple in enumerate(states[0].generated_triples):
            print(i, model_config.tokenizer.decode(triple)[:-1], triple, end='\n')

        if not interactive_mode:
            break

if __name__ == '__main__':
    main()

