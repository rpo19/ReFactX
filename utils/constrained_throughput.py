"""Measure token-generation time with and without ReFactX constraints.

This mirrors the throughput comparison used in the paper: the same prompt and
token budget are decoded once with the constrained logits processor active and
once with plain generation, recording the wall time between decoding steps.

Example::

    python utils/constrained_throughput.py \
        --model Qwen/Qwen3.5-0.8B --table qwen36 --max-tokens 128 \
        --output logs/throughput_constrained.json

The index URL is taken from ``--index`` or ``BASE_INDEX_PATH``/``INDEX``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

from dotenv import load_dotenv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

import refactx
from refactx.generate import (
    ConstrainedLogitsProcessor,
    activate_constrained_fact,
    get_constrained_logits_processor,
)


class TimingLogitsProcessor(LogitsProcessor):
    """Record the wall time between successive decoding steps."""

    def __init__(self):
        self.timings = []
        self.last_time = None

    def __call__(self, input_ids, scores):
        now = time.perf_counter()
        if self.last_time is not None:
            self.timings.append(now - self.last_time)
        self.last_time = now
        return scores

    def report(self):
        if not self.timings:
            return {}
        values = sorted(self.timings)
        return {
            'num_steps': len(values),
            'total_s': sum(values),
            'mean_ms': 1000 * sum(values) / len(values),
            'p50_ms': 1000 * values[len(values) // 2],
            'p95_ms': 1000 * values[min(len(values) - 1, int(0.95 * len(values)))],
            'max_ms': 1000 * max(values),
            'min_ms': 1000 * min(values),
        }


class AlwaysConstrainedLogitsProcessor(ConstrainedLogitsProcessor):
    """Keep every row constrained, re-arming after each completed fact."""

    def __init__(self, *args, index=None, sentinel=False, eot=' </fact>\n', **kwargs):
        super().__init__(*args, **kwargs)
        self.fact_index = index
        self.fact_sentinel = sentinel
        self.fact_eot = eot

    def _rearm(self):
        for batch in self.states.states:
            for state in batch:
                if state.active_generation is None and state.input_ids:
                    activate_constrained_fact(
                        state, self.fact_index, sentinel=self.fact_sentinel,
                        eot=self.fact_eot,
                    )

    def __call__(self, input_ids, scores):
        self._rearm()
        return super().__call__(input_ids, scores)


def _resolve_index_url(index, table):
    url = index or os.getenv('BASE_INDEX_PATH') or os.getenv('INDEX')
    if not url:
        raise SystemExit('Set --index or BASE_INDEX_PATH/INDEX')
    if url.startswith(('postgres://', 'postgresql://')) and table and 'tablename=' not in url:
        url = f'{url}?tablename={table}'
    return url


def _load_index(url, tokenizer):
    if os.path.isfile(url):
        return refactx.load_index(url, tokenizer=tokenizer)
    return refactx.load_index(url)


def _load_model(model_name, dtype, attn_implementation):
    supports_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    if dtype == 'auto':
        torch_dtype = torch.bfloat16 if supports_bf16 else (
            torch.float16 if torch.cuda.is_available() else torch.float32)
    else:
        torch_dtype = getattr(torch, dtype)
    kwargs = {'dtype': torch_dtype, 'device_map': 'auto'}
    if attn_implementation:
        kwargs['attn_implementation'] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    model.config.use_cache = True
    if getattr(model, 'generation_config', None) is not None:
        model.generation_config.use_cache = True
    return model


def _build_prompt(args, tokenizer):
    if args.prompt_text:
        return args.prompt_text
    if args.prompt_template:
        template = refactx.load_prompt(args.prompt_template)
        return refactx.apply_prompt_template(
            tokenizer, prompt_template=template,
            question=args.question, enable_thinking=False,
        )
    return args.question


def _prepare_inputs(tokenizer, prompt, batch_size, device):
    encoded = tokenizer([prompt] * batch_size, return_tensors='pt', padding=True)
    return {key: value.to(device) for key, value in encoded.items()}


def _decode(model, inputs, processors, max_tokens):
    timing = TimingLogitsProcessor()
    logits_processors = LogitsProcessorList([*processors, timing])
    with torch.no_grad():
        output = model.generate(
            **inputs,
            logits_processor=logits_processors,
            max_new_tokens=max_tokens,
            do_sample=False,
            num_beams=1,
            num_return_sequences=1,
            use_cache=True,
            eos_token_id=None,
            pad_token_id=(
                model.generation_config.pad_token_id
                if getattr(model, 'generation_config', None) is not None
                else None
            ) or (getattr(model.config, 'eos_token_id', None) or 0),
        )
    new_tokens = output.shape[1] - inputs['input_ids'].shape[1]
    return timing.report(), output, new_tokens


def _summary(report, new_tokens, batch_size):
    total = report.get('total_s', 0.0)
    steps = report.get('num_steps', 0)
    tokens = new_tokens * batch_size
    return {
        **report,
        'generated_tokens': tokens,
        'tokens_per_s': (tokens / total) if total else 0.0,
        'steps': steps,
    }


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='Qwen/Qwen3.5-0.8B')
    parser.add_argument('--index', default=None)
    parser.add_argument('--table', default='qwen36')
    parser.add_argument('--question', default='Tell me a fact about Danny Boyle.')
    parser.add_argument('--prompt-template', default=None,
                        help='Optional ReFactX prompt YAML/JSON (uses --question)')
    parser.add_argument('--prompt-text', default=None,
                        help='Literal prompt; overrides --prompt-template')
    parser.add_argument('--max-tokens', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--dtype', default='auto', choices=('auto', 'float16', 'bfloat16', 'float32'))
    parser.add_argument('--attn-implementation', default=None)
    parser.add_argument('--sentinel', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--eot', default=' </fact>\n')
    parser.add_argument('--fact-pattern', default='<fact>')
    parser.add_argument('--count-pattern', default='<count>')
    parser.add_argument('--warmup-tokens', type=int, default=8)
    parser.add_argument('--output', default='logs/throughput_constrained.json')
    parser.add_argument('--modes', nargs='+', default=['unconstrained', 'constrained'],
                        choices=('unconstrained', 'constrained'))
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = _load_model(args.model, args.dtype, args.attn_implementation)
    index = _load_index(_resolve_index_url(args.index, args.table), tokenizer)
    prompt = _build_prompt(args, tokenizer)
    inputs = _prepare_inputs(tokenizer, prompt, args.batch_size, model.device)

    if args.warmup_tokens > 0:
        _decode(model, inputs, [], args.warmup_tokens)

    results = {'config': vars(args), 'prompt': prompt}
    for mode in args.modes:
        processors = []
        if mode == 'constrained':
            processor = get_constrained_logits_processor(
                tokenizer, index,
                num_beams=1, num_batches=args.batch_size,
                sentinel=args.sentinel,
                fact_pattern=args.fact_pattern,
                count_pattern=args.count_pattern,
                eot=args.eot,
                return_list=False,
            )
            for batch in processor.states.states:
                for state in batch:
                    activate_constrained_fact(
                        state, index, sentinel=args.sentinel, eot=args.eot,
                    )
            always = AlwaysConstrainedLogitsProcessor(
                processor.states, tokenizer, index=index,
                sentinel=args.sentinel, eot=args.eot,
            )
            processors = [always]
        report, output, new_tokens = _decode(
            model, inputs, processors, args.max_tokens)
        results[mode] = _summary(report, new_tokens, args.batch_size)
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        results[mode]['sample'] = text[-600:]
        print(f"{mode}: {results[mode]['tokens_per_s']:.2f} tok/s "
              f"({results[mode]['mean_ms']:.2f} ms/step)")

    if 'unconstrained' in results and 'constrained' in results:
        base = results['unconstrained']['tokens_per_s']
        constrained = results['constrained']['tokens_per_s']
        results['overhead'] = {
            'slowdown_x': (base / constrained) if constrained else math.inf,
            'throughput_ratio': (constrained / base) if base else 0.0,
        }
        print(f"constrained/unconstrained throughput: "
              f"{results['overhead']['throughput_ratio']:.3f}x "
              f"({results['overhead']['slowdown_x']:.2f}x slower)")

    parent = os.path.dirname(args.output)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(results, handle, indent=2)
    print(f"Wrote {args.output}")


if __name__ == '__main__':
    main()
