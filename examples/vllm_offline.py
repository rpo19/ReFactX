"""Offline ReFactX constrained generation through vLLM.

Prerequisites::

    pip install "refactx[vllm]"
    export REFACTX_INDEX=/path/to/triples.txt      # or postgres://...?tablename=...
    # optional: export REFACTX_INDEX_TABLES=qwen36,qwen36fbwikipops
    # optional: export REFACTX_SENTINEL=1
    # optional: export REFACTX_FORCE_CONSTRAINED=1   # skip trigger detection

Run::

    python examples/vllm_offline.py --model Qwen/Qwen2.5-0.5B-Instruct \
        --question "Who directed Slumdog Millionaire?"

On Turing/T4 GPUs (no FlashAttention-2) the script defaults to vLLM's Triton
attention backend.  If importing vLLM fails on ``libtorchcodec``/``libnvrtc``,
add the NVIDIA/CUDA and PyNvVideoCodec library directories to
``LD_LIBRARY_PATH``.  vLLM 0.25 also expects a newer ``ALLOWED_LAYER_TYPES``
symbol; ``refactx.vllm_proc`` patches it before importing vLLM, so import the
adapter before ``vllm`` (this script does).
"""
import argparse
import os

# Import the adapter first: it applies the transformers/vLLM compatibility shim.
from refactx.vllm_proc import RefactxLogitsProcessor
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

import refactx


def _default_attention_backend():
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 8:
            return "TRITON_ATTN"
    except Exception:
        pass
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--question", default="Who directed Slumdog Millionaire?")
    parser.add_argument("--prompt-template", default="prompts/prompt_qwen36_angular2_nothink.yaml")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--attention-backend", default=_default_attention_backend())
    parser.add_argument("--no-thinking", dest="thinking", action="store_false")
    parser.set_defaults(thinking=False)
    args = parser.parse_args()

    if not os.environ.get("REFACTX_INDEX"):
        raise SystemExit("Set REFACTX_INDEX first")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    template = refactx.load_prompt(args.prompt_template)
    prompt = refactx.apply_prompt_template(
        tokenizer, prompt_template=template, question=args.question,
        enable_thinking=args.thinking,
    )

    kwargs = dict(
        model=args.model,
        logits_processors=[RefactxLogitsProcessor],
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    if args.max_model_len is not None:
        kwargs["max_model_len"] = args.max_model_len
    if args.attention_backend:
        kwargs["attention_backend"] = args.attention_backend
    llm = LLM(**kwargs)

    params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        extra_args={"refactx": True, "refactx_id": "q0"},
    )
    output = llm.generate([prompt], params)
    text = output[0].outputs[0].text
    print("=== completion ===")
    print(text)
    print("\n=== fact lines ===")
    for line in text.splitlines():
        if line.strip().startswith("<fact>"):
            print(line.strip())


if __name__ == "__main__":
    main()
