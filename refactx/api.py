"""
FastAPI server for ReFactX.

Provides OpenAI-compatible APIs for:
- Chat completions (/v1/chat/completions)
- Completion (/v1/completions, not implemented)

Key features:
- Integration with ReFactX for constrained generation.
- Support for human oversight by returning generated facts.
- Configurable via environment variables: MODEL, INDEX, DEVICE, NUM_BEAMS, QUANTIZE.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os

import torch
from transformers.generation.logits_process import LogitsProcessorList
from transformers import AutoModelForCausalLM, AutoTokenizer
from datamodules.base_dataset_config import PROMPT_TEMPLATE
from refactx import ConstrainedLogitsProcessor, ConstrainedStateList, \
                    PatternConstrainedState, DictIndex, patch_model
import refactx

# -----------------------------
# CONFIGURATION FROM ENV VARS
# -----------------------------

MODEL = os.getenv("MODEL")
INDEX = os.getenv("INDEX")
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# Optional generation parameters
NUM_BEAMS = int(os.getenv("NUM_BEAMS", 1))  # default to 1 (no beam search)

# Simple quantization flag (example: use 4-bit if set)
QUANTIZE = os.getenv("QUANTIZE", "false").lower() == "true"

# -----------------------------
# MODEL LOADING
# -----------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL)

if QUANTIZE:
    # Example simple quantization (4-bit)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        device_map="auto",
        load_in_4bit=True
    )
else:
    model = AutoModelForCausalLM.from_pretrained(MODEL)
    model.to(DEVICE)

print(f"Model: {MODEL}")
print(f"Index: {INDEX}")
print(f"Device: {DEVICE}")
print(f"Num beams: {NUM_BEAMS}")
print(f"Quantized: {QUANTIZE}")

tokenizer = AutoTokenizer.from_pretrained(MODEL)

# no need for num_beams=1
patch_model(model)

# %%
index = refactx.load_index(INDEX, tokenizer)

app = FastAPI()


# -------------------------------------------------------------
# REQUEST / RESPONSE MODELS (mirroring OpenAI fields)
# -------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str     # "user", "assistant", "system"
    content: str


class ChatCompletionBody(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class ChatCompletionResponse(BaseModel):
    response: str


class CompletionBody(BaseModel):
    model: str
    prompt: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class CompletionResponse(BaseModel):
    response: str


# -------------------------------------------------------------
# WRAPPER FUNCTIONS
# -------------------------------------------------------------

def do_chat_completion(req: ChatCompletionBody) -> str:
    """
    Generates a chat response using ReFactX.

    Args:
        req (ChatCompletionBody): Object containing chat messages and generation options.

    Returns:
        dict: Contains the generated text and the list of extracted triples (facts).
    """
    try:
        chat = [{"role": m.role, "content": m.content} for m in req.messages]

        # Apply chat template to the tokenizer input
        inputs = tokenizer.apply_chat_template(
            chat,
            return_tensors="pt",
        )

        inputs = inputs.to(model.device)

        num_beams = NUM_BEAMS

        # Configure constrained state for ReFactX generation
        states = [[PatternConstrainedState(
                        pattern = 'Fact:',
                        tokenizer = tokenizer,
                        cache_index = DictIndex(end_of_triple=index.end_of_triple, tokenizer=tokenizer),
                        subtree_cache = DictIndex(end_of_triple=index.end_of_triple, tokenizer=tokenizer),
                    )]]

        refactx.CONSTRAINED_STATES = ConstrainedStateList(states,
                    num_beams=num_beams,
                    num_batches = 1,
            )

        constrained_processor = ConstrainedLogitsProcessor(
            index=index,
            states=refactx.CONSTRAINED_STATES, tokenizer=tokenizer)
        logits_processor_list = LogitsProcessorList([
            constrained_processor
        ])

        model.eval()

        with torch.no_grad():
            out = model.generate(
                **inputs,
                logits_processor=logits_processor_list,
                max_new_tokens=800,
                do_sample = False,
                temperature = None,
                top_k=None,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                use_cache=True,
                top_p=None,
                min_p=None,
            )

        # index where new tokens start
        _from = inputs.input_ids.shape[-1]

        # ---- BEST BEAM (index 0) ----
        best_ids = out[0][_from:]
        best_text = tokenizer.decode(best_ids, skip_special_tokens=True)

        # ---- ReFactX triples ----
        generated_triples = []
        for triple_ids in refactx.CONSTRAINED_STATES[0][0].generated_triples:
            triple_text = tokenizer.decode(triple_ids, skip_special_tokens=True)
            generated_triples.append(triple_text)

        # final result (best beam only)
        result = {
            "text": best_text,
            "triples": generated_triples,
        }

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def do_completion(req: CompletionBody) -> str:
    """
    Endpoint for generating chat completions.

    Accepts requests similar to the OpenAI API and returns:
    - response: text generated by ReFactX
    """
    raise NotImplementedError('Completion is not implemented. Use chat completion.')
    try:
        result = None
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------------------------
# ENDPOINTS (exact same structure as OpenAI)
# -------------------------------------------------------------

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
def chat_completion(request: ChatCompletionBody):
    reply = do_chat_completion(request)
    return ChatCompletionResponse(response=reply)


@app.post("/v1/completions", response_model=CompletionResponse)
def completion(request: CompletionBody):
    reply = do_completion(request)
    return CompletionResponse(response=reply)
