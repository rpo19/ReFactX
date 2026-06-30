"""
End-to-end generation script using vLLM + XgrammarConstrainedProcessor.

Generates text with constrained fact extraction, saves a report.
"""
import json
import math
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Union

import torch
from transformers import AutoTokenizer

import xgrammar as xgr
from refactx.index import DictIndex, EmptyIndexException, TripleNotFoundException


class VLLMXgrammarAdapter:
    """
    Adapter wrapping XgrammarConstrainedProcessor logic for vLLM's
    logits processor interface: (past_tokens_ids, logits_row) -> logits_row.
    """

    MODE_FREE = "free"
    MODE_TRIPLE = "triple"
    MODE_JSON = "json"

    def __init__(
        self,
        tokenizer,
        kb_index: DictIndex,
        json_schema: Optional[Union[str, Dict]] = None,
        fact_pattern: str = "Fact:",
        answer_pattern: str = "Answer:",
        eot: str = "\n",
        avoid_duplicates: bool = True,
        regex_window: int = 20,
    ):
        self.tokenizer = tokenizer
        self.kb_index = kb_index
        self.fact_pattern = fact_pattern.lower()
        self.answer_pattern = answer_pattern.lower()
        self.eot = eot
        self.avoid_duplicates = avoid_duplicates
        self.regex_window = regex_window
        self.vocab_size = len(tokenizer)

        self._make_eot_token()

        self.bitmask = xgr.allocate_token_bitmask(1, self.vocab_size)

        self._json_matcher = None
        if json_schema is not None:
            tok_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
            compiler = xgr.GrammarCompiler(tok_info)
            compiled = compiler.compile_json_schema(json_schema)
            self._json_matcher = xgr.GrammarMatcher(
                compiled, terminate_without_stop_token=True
            )

        self.reset()

    def _make_eot_token(self):
        self.eot_token = None
        if self.eot is not None:
            enc = self.tokenizer.encode(self.eot, add_special_tokens=False)
            self.eot_token = enc[0] if enc else None

    def reset(self):
        self._reset_count = getattr(self, '_reset_count', 0) + 1
        self.mode = self.MODE_FREE
        self.generated_tokens: List[int] = []
        self.triple_seq: List[int] = []
        self.generated_triples: List[List[int]] = []
        self._triple_start_idx = -1
        self._visited_index = DictIndex()
        if self._json_matcher is not None:
            self._json_matcher.reset()

    def _get_text_window(self) -> str:
        window = self.generated_tokens[-self.regex_window:]
        text = self.tokenizer.decode(window)
        return text

    def _check_pattern(self):
        text = self._get_text_window()
        if self.mode == self.MODE_FREE:
            normalized = "".join(text.split())
            if normalized.lower().endswith(self.fact_pattern):
                return self.fact_pattern
            if normalized.lower().endswith(self.answer_pattern):
                return self.answer_pattern
        return None

    def _allow_tokens(self, logits_row, token_ids):
        logits_row[:] = -math.inf
        for tid in token_ids:
            if 0 <= tid < self.vocab_size:
                logits_row[tid] = 0

    def _mask_for_triple(self, logits_row):
        if self._triple_start_idx < 0:
            return

        raw_ids = self.generated_tokens[self._triple_start_idx:]
        if not raw_ids:
            return
        triple_prefix = list(raw_ids)
        self.triple_seq = triple_prefix

        try:
            possible, _ = self.kb_index.next_tokens(triple_prefix)
        except (EmptyIndexException, TripleNotFoundException):
            # Prefix not in KB — no valid triple. Return to free mode.
            self.mode = self.MODE_FREE
            self.triple_seq = []
            self._triple_start_idx = -1
            if self.eot_token is not None:
                self._allow_tokens(logits_row, [self.eot_token])
            return

        natural_end = (len(possible) == 0)

        if self.avoid_duplicates:
            self.subtract_visited_tokens(triple_prefix, possible)

        if isinstance(possible, dict):
            allowed = list(possible.keys())
        else:
            allowed = list(possible) if possible else []

        if len(allowed) == 0:
            if natural_end and triple_prefix:
                self.generated_triples.append(triple_prefix[:])
                if self.avoid_duplicates:
                    # Add both space-prefixed and non-prefixed forms
                    # to cover different tokenizations of the same triple.
                    triple_text = self.tokenizer.decode(triple_prefix).lstrip()
                    for prefix in ("", " "):
                        alt_ids = self.tokenizer.encode(
                            prefix + triple_text, add_special_tokens=False
                        )
                        self._visited_index.add(alt_ids)
            self.mode = self.MODE_FREE
            self.triple_seq = []
            self._triple_start_idx = -1
            if self.eot_token is not None:
                allowed = [self.eot_token]

        self._allow_tokens(logits_row, allowed)

    def subtract_visited_tokens(self, triple_prefix, possible):
        if not hasattr(self, '_visited_index'):
            self._visited_index = DictIndex()
        try:
            visited, _ = self._visited_index.next_tokens(triple_prefix)
        except (EmptyIndexException, TripleNotFoundException):
            visited = {}
        for tid in list(possible.keys()):
            if tid in visited:
                diff = possible[tid] - visited[tid]
                if diff <= 0:
                    del possible[tid]


    def _mask_for_json(self, logits_row):
        if self._json_matcher is None:
            return
        if len(self.generated_tokens) == 0:
            return
        self._json_matcher.accept_token(self.generated_tokens[-1])
        self._json_matcher.fill_next_token_bitmask(self.bitmask, 0)
        logits_f32 = logits_row.unsqueeze(0).float()
        xgr.apply_token_bitmask_inplace(logits_f32, self.bitmask)
        logits_row[:] = logits_f32.squeeze(0).to(logits_row.dtype)
        if self._json_matcher.is_terminated():
            self.mode = self.MODE_FREE

    def __call__(self, past_tokens_ids: List[int], logits_row: torch.Tensor) -> torch.Tensor:
        # Sync state with vLLM's token list
        if len(past_tokens_ids) > len(self.generated_tokens):
            new_tokens = past_tokens_ids[len(self.generated_tokens):]
            self.generated_tokens.extend(new_tokens)
        elif len(past_tokens_ids) < len(self.generated_tokens):
            self.reset()
            self.generated_tokens = list(past_tokens_ids)

        if self.mode == self.MODE_FREE:
            detected = self._check_pattern()
            if detected == self.fact_pattern:
                self.mode = self.MODE_TRIPLE
                self.triple_seq = []
                self._triple_start_idx = len(self.generated_tokens)
            elif detected == self.answer_pattern:
                self.mode = self.MODE_JSON
                if self._json_matcher is not None:
                    self._json_matcher.reset()

        if self.mode == self.MODE_TRIPLE:
            self._mask_for_triple(logits_row)
        elif self.mode == self.MODE_JSON:
            self._mask_for_json(logits_row)

        return logits_row


def build_kb_index(tokenizer) -> DictIndex:
    """Build sample KB with geography facts."""
    kb_index = DictIndex()
    triples = [
        "<Paris> <capital of> <France>",
        "<France> <continent> <Europe>",
        "<Mont Blanc> <elevation> <4808 meters>",
        "<Europe> <contains> <France>",
        "<France> <contains> <Paris>",
    ]
    triple_ids = tokenizer(triples, add_special_tokens=False, padding=False)["input_ids"]
    for ids in triple_ids:
        kb_index.add(ids)

    # Also add space-prefixed variants for tokenizers that produce
    # leading space tokens (e.g., token 366 = ' <' in Qwen).
    space_triples = [" " + t for t in triples]
    space_ids = tokenizer(space_triples, add_special_tokens=False, padding=False)["input_ids"]
    for ids in space_ids:
        kb_index.add(ids)

    return kb_index


def get_json_schema():
    return {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "facts": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["answer", "facts"],
    }


PROMPT = """Context: Paris is the capital of France. France is a country located in Europe. The highest mountain in Europe is Mont Blanc, which has an elevation of 4808 meters.

Task: Extract factual knowledge from the context as triples.
Format: Fact: <subject> <relation> <object>
Then answer: Answer: {"answer": "summary of key facts", "facts": ["triple1", "triple2"]}

Fact: <Paris> <capital of> <France>
Fact: <France> <continent> <Europe>"""


def main():
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    report_path = "/workspace/notebooks/refactx_clean/report_xgrammar_vllm.md"

    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print("Building KB index...")
    kb_index = build_kb_index(tokenizer)
    print(f"KB triples loaded: {len(kb_index.trie.keys() if hasattr(kb_index, 'trie') else 'OK')}")

    json_schema = get_json_schema()

    print("Creating vLLM logits processor adapter...")
    adapter = VLLMXgrammarAdapter(
        tokenizer=tokenizer,
        kb_index=kb_index,
        json_schema=json_schema,
        fact_pattern="Fact:",
        answer_pattern="Answer:",
        eot="\n",
    )

    from vllm import LLM, SamplingParams

    print(f"Loading vLLM model {model_name} (this may take a while)...")
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        dtype="half",
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=512,
        stop=["\n\n"],
        logits_processors=[adapter],
    )

    print(f"\nPrompt:\n{PROMPT}\n")
    print("Generating...")
    t0 = time.time()
    outputs = llm.generate([PROMPT], sampling_params)
    elapsed = time.time() - t0

    generated_text = outputs[0].outputs[0].text
    all_tokens = outputs[0].outputs[0].token_ids

        # Decode generated triples (strip leading whitespace)
    triple_strings = []
    for triple_ids in adapter.generated_triples:
        triple_text = tokenizer.decode(triple_ids).lstrip()
        triple_strings.append(triple_text)

    xgr_version = getattr(xgr, "__version__", None)
    if xgr_version is None:
        try:
            import importlib.metadata
            xgr_version = importlib.metadata.version("xgrammar")
        except Exception:
            xgr_version = "unknown"

    print(f"\n=== Generated Text ===")
    print(generated_text)

    # Deduplicate triples for the report
    unique_triples = list(dict.fromkeys(triple_strings))
    all_triples_block = "\n".join(triple_strings)
    unique_triples_block = "\n".join(unique_triples)

    report = f"""# Xgrammar + vLLM End-to-End Report

**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Model:** {model_name}
**Library versions:** vLLM=0.8.3, xgrammar={xgr_version}

## Configuration

- **KB Triples (base):** 5 geography facts (space-prefixed variants auto-added)
- **JSON Schema:** `answer` (string) + `facts` (array of strings)
- **Patterns:** `Fact:` -> KB triple constraint, `Answer:` -> JSON schema constraint
- **Sampling:** temperature=0.7, top_p=0.9, max_tokens=256, stop=`\\n\\n`

## Generation Summary

| Metric | Value |
|--------|-------|
| Generation time | {elapsed:.2f}s |
| Output tokens | {len(all_tokens)} |
| Throughput | {len(all_tokens)/elapsed:.1f} tok/s |
| Triples extracted | {len(adapter.generated_triples)} |
| Unique triples | {len(unique_triples)} |
| Final FSM mode | `{adapter.mode}` |

### Prompt

```
{PROMPT}
```

### Generated Text

```
{generated_text}
```

### Extracted Triples (all {len(adapter.generated_triples)})

```
{all_triples_block}
```

### Unique Triples ({len(unique_triples)})

```
{unique_triples_block}
```

## FSM Trace

- Pattern detection: `Fact:` -> triple mode, `Answer:` -> JSON mode
- Leading whitespace tokens are included in the KB index
- `avoid_duplicates` uses a secondary `DictIndex` tracking visited triples
- Each triple is recorded in `generated_triples` when the KB path is exhausted

## Implementation Notes

- Triple constraint uses `DictIndex.next_tokens()` for KB lookup
- JSON constraint uses xgrammar `GrammarMatcher.fill_next_token_bitmask()`
- Pattern detection normalizes whitespace: `"".join(text.split())`
- vLLM adapter uses interface `(past_tokens_ids, logits_row) -> logits_row`
- xgrammar bitmask requires float32 logits; adapter converts float16->f32->back
"""

    with open(report_path, "w") as f:
        f.write(report)

    print(f"\nReport saved to {report_path}")
    print(f"Generated text ({len(all_tokens)} tokens, {elapsed:.2f}s):")
    print(generated_text)
    print(f"\nFSM generated triples: {triple_strings}")
    print(f"Final mode: {adapter.mode}")


if __name__ == "__main__":
    main()
