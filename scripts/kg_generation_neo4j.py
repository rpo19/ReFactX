"""Headless end-to-end run: KnowledgeGraphGeneration + Neo4j (Qwen model).

Env: MODEL (default Qwen/Qwen3.5-0.8B), LONG_CHAINS (0/1), AVOID_DUPLICATES (0/1), QUESTION.
"""
import os
import sys
import time
from pathlib import Path

REPO = Path("/workspace/notebooks/ReFactX_dev")
sys.path.insert(0, str(REPO))

import torch
from transformers import (AutoProcessor, AutoTokenizer, AutoModelForImageTextToText,
                          ProcessorMixin)

import refactx
from refactx.index import DictIndex
from refactx.generate import ConstrainedLogitsProcessor, ConstrainedStateList, KnowledgeGraphGeneration
from refactx.neo4j_kg import Neo4jKnowledgeGraph

MODEL = os.environ.get("MODEL", "Qwen/Qwen3.5-0.8B")
LONG_CHAINS = os.environ.get("LONG_CHAINS", "0").lower() in ("1", "true", "yes")
AVOID_DUPLICATES = os.environ.get("AVOID_DUPLICATES", "1").lower() in ("1", "true", "yes")
NUM_BEAMS = 1
NUM_BATCHES = 1

try:
    processor = AutoProcessor.from_pretrained(MODEL)
except Exception as e:
    print("AutoProcessor failed, falling back to AutoTokenizer:", type(e).__name__)
    processor = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForImageTextToText.from_pretrained(MODEL, device_map="auto", dtype="bfloat16")
tokenizer = processor
tok = tokenizer.tokenizer if isinstance(tokenizer, ProcessorMixin) else tokenizer
print("loaded", MODEL, "device:", next(model.parameters()).device)

prompt_messages = refactx.load_prompt(str(REPO / "prompts" / "prompt_qwen36_angular2_kg_nothink.yaml"))

kg = Neo4jKnowledgeGraph()
SEEDS = ["Barack Obama", "Paris", "Albert Einstein"]

def entity_index():
    return kg.entity_index(limit=5000, seeds=SEEDS, hops=1)

property_strings = kg.property_strings
get = kg.get

index = DictIndex()
index.set_tokenizer(tokenizer)
entities = entity_index()
for entity in entities:
    index.add(tok.encode(f" <{entity}>", add_special_tokens=False))
print(f"Loaded {len(entities)} entities into the KG index.")

states = ConstrainedStateList("auto", num_beams=NUM_BEAMS, num_batches=NUM_BATCHES, debug_tokenizer=tokenizer)
logits_processor = ConstrainedLogitsProcessor(states=states, tokenizer=tokenizer)
logits_processor.add_pattern(
    "<kg>", KnowledgeGraphGeneration,
    index=index, get_relations=property_strings, get_objects=get,
    long_chains=LONG_CHAINS, eot=" </kg>\n",
    avoid_duplicates=AVOID_DUPLICATES,
)
print("long_chains =", LONG_CHAINS, "| avoid_duplicates =", AVOID_DUPLICATES)

def ask(question, max_new_tokens=200):
    logits_processor.reset_states()
    prompt = refactx.apply_prompt_template(
        tokenizer, question=question, enable_thinking=False, prompt_template=prompt_messages)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            logits_processor=[logits_processor],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=NUM_BEAMS,
            num_return_sequences=1,
            use_cache=True,
        )
    state = logits_processor.states[0, 0]
    answer = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    paths = []
    for gen in state.generation_history:
        paths.extend(getattr(gen, "generated_path_metadata", []))
    print("\n--- GENERATED ---")
    print(answer)
    print(f"--- KG paths ({len(paths)}) ---")
    for p in paths:
        print(" ", p)
    print(f"--- elapsed {time.time() - t0:.1f}s ---")
    return answer, paths

if __name__ == "__main__":
    ask(os.environ.get("QUESTION", "Which books did Barack Obama write?"))
    kg.close()
    print("DONE")
