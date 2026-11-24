"""Build reasoning traces with minimal, correct calls and natural templates.

Count questions keep only prerequisite facts that lead to the counted subject
(object matches the count subject), then the count call. Other facts are dropped.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

from transformers import AutoTokenizer

import refactx
from utils.enrich_reasoning_traces import (
    PROMPTS, add_count_calls, load_raw, norm, parse_facts,
)


def subj_base(value):
    return re.sub(r"\s*\([^)]*\)\s*$", "", str(value)).strip().lower()


def select_calls(calls, is_count):
    if not is_count:
        return calls
    counts = [c for c in calls if c["type"] == "count"]
    if not counts:
        return []
    fact_calls = [c for c in calls if c["type"] == "fact"]
    count = counts[0]
    base = subj_base(count["subject"])
    # Keep only facts that lead to the counted subject.
    prereq = [f for f in fact_calls if subj_base(f.get("object", "")) == base]
    return prereq + [count]


def build(mode, question, answer, calls):
    answer_block = "<answer>\n" + json.dumps(answer, ensure_ascii=False) + "\n</answer>"
    if not calls:
        return None
    if calls[0]["type"] == "count":
        lead = (f"To answer \"{question}\", I need to count the {calls[0]['relation']} "
                f"of {calls[0]['subject']}.")
    else:
        lead = (f"To answer \"{question}\", I need the {calls[0]['relation']} "
                f"of {calls[0]['subject']} first.")
    between = []
    for i, call in enumerate(calls):
        nxt = calls[i + 1] if i + 1 < len(calls) else None
        if call["type"] == "fact":
            sentence = f"The retrieved object is {call['object']}."
            if nxt and nxt["type"] == "count":
                sentence += f" Now count the {nxt['relation']} of {nxt['subject']}."
            elif nxt and nxt["type"] == "fact":
                sentence += f" Now I need the {nxt['relation']} of {nxt['subject']}."
            between.append(sentence)
        else:
            between.append("")

    if mode == "nothink_reasoning":
        parts = [lead]
        for call, sentence in zip(calls, between):
            parts.append(call["text"])
            if sentence:
                parts.append(sentence)
        return "\n".join(parts) + "\n\n" + answer_block
    if mode == "plain":
        # Control: identical tool calls and answer, no narration.
        calls_block = "\n".join(c["text"] for c in calls)
        return f"{calls_block}\n{answer_block}"
    if mode == "think_tools_outside":
        plan_bits = []
        for i, call in enumerate(calls):
            if call["type"] == "count":
                step = f"count the {call['relation']} of {call['subject']}"
            else:
                step = f"retrieve the {call['relation']} of {call['subject']}"
            plan_bits.append(step)
        # Only the first step is introduced with "first"; later steps are joined
        # with ", then " (no second "then").
        plan = f"To answer \"{question}\", I'll first " + ", then ".join(plan_bits) + "."
        calls_block = "\n".join(c["text"] for c in calls)
        return f"<think>\n{plan}\n</think>\n\n{calls_block}\n{answer_block}"
    if mode == "think_tools_in_side":
        parts = [lead]
        for call, sentence in zip(calls, between):
            parts.append(call["text"])
            if sentence:
                parts.append(sentence)
        return "<think>\n" + "\n".join(parts) + "\n</think>\n\n" + answer_block
    raise ValueError(mode)


FILTER_WORDS = re.compile(
    r"\b(or|and|not|less|more|greater|fewer|than|before|after|between|named|"
    r"whose|that|which|with|without|ever|first|last|most|least)\b",
    re.IGNORECASE,
)


def _norm_text(s):
    return re.sub(r"[^a-z0-9 ]", "", str(s).lower()).strip()


def answer_supported(trace, answer):
    """Keep a trace only if every answer string appears in the facts or counts it retrieves."""
    if not answer:
        return False
    evidence = " ".join(
        re.findall(r"<fact>(.*?)</fact>", trace, re.S)
        + re.findall(r"<count>(.*?)</count>", trace, re.S)
    )
    evidence = _norm_text(evidence)
    return all(_norm_text(a) in evidence for a in answer if _norm_text(a))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("logs/kg_trace_sft.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("logs/reasoning_templates_v3"))
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--max-tokens", type=int, default=1280)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    templates = {key: refactx.load_prompt(path) for key, path in PROMPTS.items()}
    by_id, by_question = load_raw()
    modes = ["nothink_reasoning", "think_tools_outside", "think_tools_in_side", "plain"]
    outputs = {m: (args.output_dir / f"{m}.jsonl").open("w", encoding="utf-8") for m in modes}
    counts = collections.Counter()
    with args.input.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            raw = by_id.get(str(record.get("id"))) or by_question.get(norm(record.get("question")))
            calls = parse_facts(record.get("full_prediction") or "")
            is_count = record.get("question_type") == "count"
            if is_count and FILTER_WORDS.search(record["question"]):
                # A bare <count> cannot apply extra constraints; the trace would be wrong.
                counts["count_with_filter_dropped"] += 1
                continue
            if is_count:
                calls = add_count_calls(calls, raw, record.get("gt_answer"))
                if not any(c["type"] == "count" for c in calls):
                    counts["count_without_tool"] += 1
                    continue
            calls = select_calls(calls, is_count)
            if not calls:
                counts["no_calls"] += 1
                continue
            answer = record.get("gt_answer")
            if isinstance(answer, str):
                answer = [answer]
            dataset = str(record.get("dataset", "unknown"))
            template = templates.get(dataset, templates["default"])
            for mode in modes:
                trace = build(mode, record["question"], answer, calls)
                if trace is None:
                    continue
                trace = trace.rstrip()
                if not answer_supported(trace, answer):
                    counts[mode + "_unsupported_answer"] += 1
                    continue
                if not trace.endswith("<|im_end|>"):
                    trace += "<|im_end|>"
                prompt = refactx.apply_prompt_template(
                    tokenizer, template, record["question"],
                    enable_thinking=mode.startswith("think"))
                length = (
                    len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
                    + len(tokenizer(trace, add_special_tokens=False)["input_ids"])
                )
                if length > args.max_tokens:
                    counts[mode + "_too_long"] += 1
                    continue
                row = dict(record)
                row["prompt"] = prompt
                row["full_prediction"] = trace
                row["reasoning_mode"] = mode
                row["mask_mode"] = "objects"
                outputs[mode].write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[mode] += 1
    for fd in outputs.values():
        fd.close()
    print(dict(counts))
    if counts.get("nothink_reasoning"):
        # print one count example for inspection
        for line in (args.output_dir / "nothink_reasoning.jsonl").open(encoding="utf-8"):
            row = json.loads(line)
            if "<count>" in row["full_prediction"]:
                print(row["question"])
                print(row["full_prediction"])
                break


if __name__ == "__main__":
    main()
