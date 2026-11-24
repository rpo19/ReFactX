"""Build a compact, clean SFT mixture from teacher traces.

The teacher evaluation logs used to build ``sft_training_mix_supported`` contain
three artifacts that hurt small-model SFT:

* assistant traces that never receive the chat end token, so the student never
  learns to stop;
* a very long few-shot prompt (~1.5k tokens), leaving little room to generate;
* malformed exhausted-retrieval calls such as ``<relation <no further records>``.

This script rebuilds each trace with a compact prompt, repairs malformed
sentinel calls, appends ``<|im_end|>``, optionally keeps only traces whose
benchmark answer was strictly correct, and drops sequences above a token
budget.

Example::

    python -m utils.build_clean_sft \
        --input logs/sft_training_mix_supported.jsonl.gz \
        --output logs/sft_training_mix_v3.jsonl \
        --prompt prompts/prompt_compact.yaml \
        --max-tokens 1280 \
        --require-correct
"""
import argparse
import gzip
import json
import re

from transformers import AutoTokenizer

import refactx


def _open(path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path)


def repair_trace(text):
    text = text.rstrip()
    # <Subject> <relation <no further records> ... -> <Subject> <relation> <no further records> ...
    text = re.sub(
        r"(<[^<>\n]+>)\s*<([^<>\n]+?)\s*<no further records>",
        r"\1 <\2> <no further records>",
        text,
    )
    # Drop fact blocks that are still malformed.
    text = re.sub(
        r"<fact>.*?</fact>",
        lambda m: "" if re.search(r"<[^<>\n]*<no further records>", m.group(0)) else m.group(0),
        text,
        flags=re.S,
    )
    text = re.sub(r"(?m)^\s*</fact>\s*$\n?", "", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", default="prompts/prompt_compact.yaml")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--max-tokens", type=int, default=1280)
    parser.add_argument("--require-correct", action="store_true",
                        help="Keep only traces whose evaluation.correct == 1")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    template = refactx.load_prompt(args.prompt)

    kept = 0
    skipped = {"not_correct": 0, "no_answer": 0, "too_long": 0}
    with _open(args.input) as src, open(args.output, "w") as dst:
        for line in src:
            record = json.loads(line)
            if "full_prediction" not in record:
                continue
            if args.require_correct and (record.get("evaluation") or {}).get("correct") != 1:
                skipped["not_correct"] += 1
                continue

            trace = repair_trace(record["full_prediction"])
            if "<answer>" not in trace:
                skipped["no_answer"] += 1
                continue
            if not trace.endswith("<|im_end|>"):
                trace += "<|im_end|>"

            prompt = refactx.apply_prompt_template(
                tokenizer, template, record["question"], enable_thinking=False)
            length = (
                len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
                + len(tokenizer(trace, add_special_tokens=False)["input_ids"])
            )
            if length > args.max_tokens:
                skipped["too_long"] += 1
                continue

            record = dict(record)
            record["original_prompt"] = record["prompt"]
            record["prompt"] = prompt
            record["full_prediction"] = trace
            dst.write(json.dumps(record) + "\n")
            kept += 1

    print(f"wrote {kept} samples to {args.output}")
    print(f"skipped: {skipped}")


if __name__ == "__main__":
    main()
