"""Merge a LoRA adapter (for example an SFT checkpoint) into its base model.

The eval path does not need this: ``utils/eval.py`` can load an adapter via the
``adapter_path`` config key. Use this only when a standalone merged checkpoint is
required (e.g. for an external tool that cannot load PEFT adapters).

Defaults target the supported-mix SFT run and deliberately contain no reference
to the legacy ``sft_output`` / ``sft_merged`` paths.
"""

import argparse
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    ws_path = os.environ.get("WS_PATH")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen3.5-4B",
        help="Base model that the adapter was trained on",
    )
    parser.add_argument(
        "--adapter",
        default=str(Path(ws_path) / "sft_output_supported_mix") if ws_path else None,
        help="LoRA adapter directory (default: $WS_PATH/sft_output_supported_mix)",
    )
    parser.add_argument(
        "--output",
        default=str(Path(ws_path) / "sft_merged_supported_mix") if ws_path else None,
        help="Output directory for the merged model (default: $WS_PATH/sft_merged_supported_mix)",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype while merging",
    )
    args = parser.parse_args()
    if not args.adapter or not args.output:
        parser.error(
            "WS_PATH is not set; pass --adapter and --output explicitly "
            "(and optionally --base-model)."
        )
    return args


def main() -> None:
    args = parse_args()
    dtype = getattr(torch, args.dtype)

    print(f"Loading base model {args.base_model}...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading adapter {args.adapter}...")
    model = PeftModel.from_pretrained(base, args.adapter)

    print("Merging...")
    model = model.merge_and_unload(safe_merge=True)
    model = model.to(dtype)

    print(f"Saving merged model to {args.output}...")
    Path(args.output).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)

    print("Saving tokenizer...")
    try:
        tok = AutoTokenizer.from_pretrained(args.adapter)
    except OSError:
        tok = AutoTokenizer.from_pretrained(args.base_model)
    tok.save_pretrained(args.output)

    print("Done!")


if __name__ == "__main__":
    main()
