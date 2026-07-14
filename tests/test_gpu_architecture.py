"""GPU smoke test for the architecture refactor — Qwen3.5-4B VLM."""
import gc
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, LogitsProcessorList
from transformers import ProcessorMixin

from refactx.index import load_index
from refactx.generate import (
    patch_model, CONSTRAINED_STATES,
    FactGeneration, CountBranchesGeneration,
    ConstrainedLogitsProcessor,
)

MODEL_ID = "Qwen/Qwen3.5-4B"
KB_PATH = "/workspace/notebooks/refactx_clean/indexes/simple_index.txt.gz"

SYSTEM_PROMPT = (
    "You are a helpful question-answering assistant that bases its answers on facts from a knowledge base.\n"
    "\n"
    "## Fact retrieval\n"
    "To obtain facts, use the Fact: command (e.g. Fact: <Danny Boyle> <date of birth> <1956-10-20T00:00:00Z> .).\n"
    "Facts are always reliable. You must support all your claims with facts.\n"
    "\n"
    "## Counting tool\n"
    "When you need to count how many objects exist for a subject-relation prefix, use the count_branches tool.\n"
    "Write exactly: count_branches: <Subject> <relation>\n"
    "The system will reply with = <number>. Use that number in your answer.\n"
    "Example: count_branches: <Spain> <shares border with> -> the system replies = 5\n"
    "\n"
    "## Reasoning\n"
    "You may think step by step before answering. Keep your reasoning concise.\n"
    "After reasoning, give a concise answer prefixed with Answer:."
)

_model = None
_tokenizer = None


def load_kb(tokenizer):
    return load_index(KB_PATH, tokenizer=tokenizer)


def make_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, device_map="auto")
    _tokenizer = processor
    patch_model(model, verbose=False)
    _model = model
    return _model, _tokenizer


def _tokenize(tokenizer, text):
    if isinstance(tokenizer, ProcessorMixin):
        return tokenizer.tokenizer(text, return_tensors="pt")
    return tokenizer(text, return_tensors="pt")


def _decode(tokenizer, ids):
    if isinstance(tokenizer, ProcessorMixin):
        return tokenizer.tokenizer.decode(ids, skip_special_tokens=True)
    return tokenizer.decode(ids, skip_special_tokens=True)


def test_fact_generation():
    """Test FactGeneration with sentinel via add_pattern."""
    model, tokenizer = make_model()
    idx = load_kb(tokenizer)

    CONSTRAINED_STATES.__init__(
        "auto", num_beams=1, num_batches=1, debug_tokenizer=tokenizer
    )
    processor = ConstrainedLogitsProcessor(
        states=CONSTRAINED_STATES, tokenizer=tokenizer
    )
    processor.add_pattern(
        "Fact:", FactGeneration,
        index=idx, sentinel=True, eot=None,
    )
    logits_processor = LogitsProcessorList([processor])

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "What is the population of France?"},
    ]
    full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _tokenize(tokenizer, full_prompt).to(model.device)

    print("=== Test 1: FactGeneration with sentinel ===")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            logits_processor=logits_processor,
            max_new_tokens=128,
            do_sample=False,
            num_beams=1,
            num_return_sequences=1,
            use_cache=True,
        )
    result = _decode(tokenizer, output[0][inputs.input_ids.shape[1]:])
    print(result)

    state = CONSTRAINED_STATES.states[0][0]
    print(f"Generated triples: {len(state.generated_triples)}")
    triples_str = [_decode(tokenizer, t) for t in state.generated_triples]
    for i, t in enumerate(triples_str):
        print(f"  {i}: {t}")
    assert len(state.generated_triples) > 0, "Expected at least 1 generated triple"
    print("PASS: FactGeneration with sentinel\n")


def test_count_branches_via_add_pattern():
    """Test CountBranchesGeneration via add_pattern API."""
    model, tokenizer = make_model()
    idx = load_kb(tokenizer)

    CONSTRAINED_STATES.__init__(
        "auto", num_beams=1, num_batches=1, debug_tokenizer=tokenizer
    )
    processor = ConstrainedLogitsProcessor(
        states=CONSTRAINED_STATES, tokenizer=tokenizer
    )
    processor.add_pattern(
        "count_branches:",
        CountBranchesGeneration,
        kb_index=idx,
    )
    logits_processor = LogitsProcessorList([processor])

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "How many countries share a border with France? Use the count_branches tool."},
    ]
    full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _tokenize(tokenizer, full_prompt).to(model.device)

    print("=== Test 2: CountBranchesGeneration via add_pattern ===")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            logits_processor=logits_processor,
            max_new_tokens=128,
            do_sample=False,
            num_beams=1,
            num_return_sequences=1,
            use_cache=True,
        )
    result = _decode(tokenizer, output[0][inputs.input_ids.shape[1]:])
    print(result)

    state = CONSTRAINED_STATES.states[0][0]
    all_calls = []
    gen = state.active_generation
    if gen and hasattr(gen, "calls"):
        all_calls.extend(gen.calls)
    for g in state.generation_history:
        if hasattr(g, "calls"):
            all_calls.extend(g.calls)
    print(f"Generated triples: {len(state.generated_triples)}")
    print(f"Generation history: {len(state.generation_history)}")
    print(f"Count branches calls: {all_calls}")
    has_cb = any(type(g).__name__ == "CountBranchesGeneration"
                 for g in state.generation_history) or \
             (gen is not None and type(gen).__name__ == "CountBranchesGeneration")
    assert has_cb, "Expected CountBranchesGeneration to be activated"
    print("PASS: CountBranchesGeneration via add_pattern\n")


def test_combined_patterns():
    """Test both FactGeneration and CountBranchesGeneration registered."""
    model, tokenizer = make_model()
    idx = load_kb(tokenizer)

    CONSTRAINED_STATES.__init__(
        "auto", num_beams=1, num_batches=1, debug_tokenizer=tokenizer
    )
    processor = ConstrainedLogitsProcessor(
        states=CONSTRAINED_STATES, tokenizer=tokenizer
    )
    processor.add_pattern(
        "Fact:", FactGeneration,
        index=idx, sentinel=True, eot=None,
    )
    processor.add_pattern(
        "count_branches:",
        CountBranchesGeneration,
        kb_index=idx,
    )
    logits_processor = LogitsProcessorList([processor])

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "What is the population of France? Use Fact: commands to look up facts."},
    ]
    full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _tokenize(tokenizer, full_prompt).to(model.device)

    print("=== Test 3: Combined patterns (Fact + CountBranches) ===")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            logits_processor=logits_processor,
            max_new_tokens=128,
            do_sample=False,
            num_beams=1,
            num_return_sequences=1,
            use_cache=True,
        )
    result = _decode(tokenizer, output[0][inputs.input_ids.shape[1]:])
    print(result)

    state = CONSTRAINED_STATES.states[0][0]
    print(f"Generated triples: {len(state.generated_triples)}")
    print(f"Generation history count: {len(state.generation_history)}")
    for i, g in enumerate(state.generation_history):
        cls_name = type(g).__name__
        print(f"  history[{i}]: {cls_name}")
    assert len(state.generated_triples) > 0 or len(state.generation_history) > 0, \
        "Expected at least some constrained generation activity"
    print("PASS: Combined patterns\n")


if __name__ == "__main__":
    test_fact_generation()
    test_count_branches_via_add_pattern()
    test_combined_patterns()
    print("\n=== ALL GPU TESTS PASSED ===")
