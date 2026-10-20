"""Generate count-tool traces with DeepSeek for the SFT mixture.

DeepSeek is a remote chat model, so it cannot run the local constrained decoder.
This tool therefore *simulates* the ``<count>`` tool: the model emits a count
request, the tool looks up the verified count (gold answer or an external KG
count endpoint), injects ``= N``/``</count>`` back into the trace, and asks the
model to continue. A trace is kept only when it contains a ``<count>`` call with
the verified value and its ``<answer>`` equals that value.

The output is a JSONL log in the same shape as ``utils/eval.py`` (header,
records, ``macro_*`` summary) with extra synthetic fields, so it can be fed to
``utils/build_sft_dataset.py`` or inspected with the existing tools.

Examples::

    # offline dry run (no API key) on 20 Mintaka count questions
    python utils/generate_count_traces.py \
        --input logs/sft_training/mintaka_train_eval_qwen38_27b.out.3.gz \
        --select teacher-wrong-or-no-count --n 20 --mock --output /tmp/counts.jsonl

    # real run: 1000 accepted traces, gold-verified counts
    DEEPSEEK_API_KEY=... python utils/generate_count_traces.py \
        --input logs/sft_training/mintaka_train_eval_qwen38_27b.out.3.gz \
        --select teacher-wrong-or-no-count --n 1000 \
        --output logs/synthetic/count_traces.jsonl
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

import requests
import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils.behaviors import (  # noqa: E402
    calculate_metrics,
    dataset_name,
    parse_answer,
    question_type,
)
from utils.log_analysis import open_log  # noqa: E402

DEFAULT_PROMPT = "prompts/prompt_qwen36_angular2_nothink.yaml"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"

COUNT_CALL_RE = re.compile(r"<count\b", re.IGNORECASE)
FACT_RE = re.compile(r"<fact>\s*(.*?)\s*</fact>", re.DOTALL | re.IGNORECASE)


# ---------------------------------------------------------------------------
# Prompt and trace parsing (pure functions, tested)
# ---------------------------------------------------------------------------

def load_system_prompt(path: str | Path = DEFAULT_PROMPT) -> str:
    content = Path(path).read_text(encoding="utf-8")
    if str(path).lower().endswith((".yaml", ".yml")):
        messages = yaml.safe_load(content)
        if isinstance(messages, list):
            return "\n\n".join(
                str(m.get("content", "")) for m in messages if str(m.get("role")) == "system"
            )
    return content


def build_count_user_message(question: str) -> str:
    return (
        f"Question: {question}\n\n"
        "This is a counting question. Use the <count> tool: emit exactly one count "
        "request and do not invent the number.\n"
        "<count> <Subject> <relation>\n"
        "</count>\n"
        "The harness will return `= N` and close the tag. Then reason with the "
        "returned count and answer with <answer>[\"N\"]</answer>."
    )


def parse_count_calls(text: str) -> list[dict]:
    """Find ``<count> <subject> <relation> [= N] </count>`` calls in a trace."""
    text = text or ""
    calls: list[dict] = []
    for match in COUNT_CALL_RE.finditer(text):
        start = match.start()
        open_end = text.find(">", start)
        if open_end == -1:
            continue
        open_end += 1
        close = re.search(r"</count\s*>", text[open_end:], re.IGNORECASE)
        if close:
            close_start = open_end + close.start()
            close_end = open_end + close.end()
            body = text[open_end:close_start]
            closed = True
        else:
            close_start = close_end = None
            body = text[open_end:open_end + 500]
            closed = False
        first_line = body.split("\n", 1)[0]
        prefix = re.match(r"\s*<([^<>]+)>\s*<([^<>]+)>", first_line)
        head = "\n".join(body.split("\n")[:3])
        value_match = re.search(r"=\s*([+-]?\d+(?:\.\d+)?)", head)
        value = None
        if value_match:
            try:
                value = float(value_match.group(1))
            except ValueError:
                value = None
        calls.append({
            "start": start,
            "open_end": open_end,
            "prefix_end": (open_end + prefix.end()) if prefix else open_end,
            "close_start": close_start,
            "close_end": close_end,
            "closed": closed,
            "subject": prefix.group(1).strip() if prefix else None,
            "relation": prefix.group(2).strip() if prefix else None,
            "value": value,
        })
    return calls


def inject_count_result(text: str, call: dict, value: float) -> str:
    """Insert ``= value`` and a closing tag into the (last) count call."""
    rendered = f"{value:g}"
    if call["closed"] and call["close_start"] is not None:
        return text[:call["close_start"]] + f"= {rendered}\n" + text[call["close_start"]:]
    return text[:call["prefix_end"]] + f"\n= {rendered}\n</count>\n" + text[call["prefix_end"]:]


def validate_trace(trace: str, expected: Optional[float]) -> tuple[bool, list[str]]:
    reasons = []
    calls = parse_count_calls(trace)
    if not calls:
        reasons.append("no <count> call")
        return False, reasons
    last = calls[-1]
    if last["value"] is None:
        reasons.append("count call has no value")
    elif expected is not None and abs(last["value"] - expected) > 1e-9:
        reasons.append(f"count {last['value']} != expected {expected}")
    answer = parse_answer(trace)
    if answer is None:
        reasons.append("no parseable <answer>")
    elif expected is not None:
        expected_answer = f"{expected:g}"
        if [str(a).strip() for a in answer] != [expected_answer]:
            reasons.append(f"answer {answer} != [{expected_answer}]")
    return not reasons, reasons


def extract_facts(trace: str) -> list[str]:
    return [f" {m.group(1).strip()}" for m in FACT_RE.finditer(trace or "")]


# ---------------------------------------------------------------------------
# DeepSeek client
# ---------------------------------------------------------------------------

class DeepSeekError(RuntimeError):
    pass


class DeepSeekClient:
    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL,
                 base_url: str = DEFAULT_BASE_URL, timeout: float = 120.0,
                 max_retries: int = 5, thinking: Optional[str] = "disabled",
                 session: Optional[requests.Session] = None):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise DeepSeekError("DEEPSEEK_API_KEY is not set (or pass api_key=...).")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.thinking = thinking
        self.session = session or requests.Session()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0

    def chat(self, messages: list[dict], *, temperature: float = 0.7,
             max_tokens: int = 2048) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if self.thinking:
            payload["thinking"] = {"type": self.thinking}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload, headers=headers, timeout=self.timeout,
                )
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(min(2 ** attempt + random.random(), 30))
                continue
            if response.status_code == 200:
                data = response.json()
                usage = data.get("usage") or {}
                self.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
                self.completion_tokens += int(usage.get("completion_tokens", 0) or 0)
                self.calls += 1
                return data
            if response.status_code in (429, 500, 502, 503, 504):
                last_error = DeepSeekError(f"HTTP {response.status_code}: {response.text[:300]}")
                time.sleep(min(2 ** attempt + random.random(), 30))
                continue
            if response.status_code == 400 and "thinking" in response.text.lower() and "thinking" in payload:
                # Older deployments may not accept the thinking field; drop it.
                payload.pop("thinking", None)
                continue
            raise DeepSeekError(f"HTTP {response.status_code}: {response.text[:500]}")
        raise DeepSeekError(f"DeepSeek request failed after {self.max_retries} retries: {last_error}")

    def content(self, response: dict) -> tuple[str, str, dict]:
        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        return message.get("content") or "", choice.get("finish_reason"), response.get("usage") or {}

    def usage_summary(self) -> dict:
        return {"calls": self.calls, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens}


# ---------------------------------------------------------------------------
# Count sources
# ---------------------------------------------------------------------------

class GoldCountSource:
    """Return the question's gold count (safe for Mintaka count questions)."""

    def __call__(self, record: dict, subject: Optional[str], relation: Optional[str]) -> Optional[float]:
        reference = record.get("gt_answer")
        values = reference if isinstance(reference, (list, tuple)) else [reference]
        if not values:
            return None
        try:
            return float(str(values[0]).replace(",", "").strip())
        except ValueError:
            return None


class KGExtCountSource:
    """Query an external count endpoint (``REFACTX_COUNT_URL``) for ``{count}``."""

    def __init__(self, url: Optional[str] = None, token: Optional[str] = None, timeout: float = 30.0):
        self.url = url or os.getenv("REFACTX_COUNT_URL")
        self.token = token or os.getenv("REFACTX_TOKEN")
        self.timeout = timeout

    def __call__(self, record: dict, subject: Optional[str], relation: Optional[str]) -> Optional[float]:
        if not self.url or not subject or not relation:
            return None
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        try:
            response = requests.post(self.url, json={"subject": subject, "relation": relation},
                                     headers=headers, timeout=self.timeout)
            response.raise_for_status()
            value = response.json().get("count")
            return float(value) if value is not None else None
        except (requests.RequestException, ValueError, TypeError):
            return None


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_one(record: dict, client, count_source, system_prompt: str, *,
                 max_turns: int = 3, temperature: float = 0.7,
                 max_tokens: int = 2048) -> dict:
    question = record.get("question") or ""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_count_user_message(question)},
    ]
    trace_parts: list[str] = []
    finish_reasons: list[str] = []
    usages: list[dict] = []
    expected: Optional[float] = None

    for turn in range(max_turns):
        response = client.chat(messages, temperature=temperature, max_tokens=max_tokens)
        content, finish_reason, usage = client.content(response)
        finish_reasons.append(finish_reason or "")
        usages.append(usage)
        messages.append({"role": "assistant", "content": content})

        calls = parse_count_calls(content)
        unclosed = [c for c in calls if c["value"] is None]
        if unclosed:
            call = unclosed[-1]
            value = count_source(record, call["subject"], call["relation"])
            if value is None:
                return {"ok": False, "reason": "count source returned no value",
                        "trace": "".join(trace_parts) + content}
            expected = value
            trace_parts.append(inject_count_result(content, call, value))
            messages.append({"role": "user",
                             "content": f"= {value:g}\n</count>\nThe count tool returned {value:g}. "
                                        "Continue your reasoning and give the final answer."})
            continue

        trace_parts.append(content)
        if "<answer>" in content.lower():
            break
        # No count call and no answer: nudge once.
        messages.append({"role": "user",
                         "content": "Use the <count> tool before answering, then give the final <answer>."})

    trace = "".join(trace_parts)
    if expected is None:
        calls = parse_count_calls(trace)
        if calls and calls[-1]["value"] is not None:
            expected = calls[-1]["value"]
    ok, reasons = validate_trace(trace, expected)
    return {"ok": ok, "reason": "; ".join(reasons), "trace": trace, "expected": expected,
            "finish_reasons": finish_reasons, "usages": usages}


def mock_generate(record: dict, count_source) -> dict:
    expected = count_source(record, None, None)
    if expected is None:
        return {"ok": False, "reason": "no gold count", "trace": "", "expected": None}
    value = f"{expected:g}"
    trace = (
        "I need to count the objects for this relation, so I will use the count tool "
        "instead of retrieving each object.\n"
        "<count> <Subject> <relation>\n"
        f"= {value}\n"
        "</count>\n"
        f"The count tool returned {value}, which answers the question.\n\n"
        f"<answer>\n[\"{value}\"]\n</answer>"
    )
    ok, reasons = validate_trace(trace, expected)
    return {"ok": ok, "reason": "; ".join(reasons), "trace": trace, "expected": expected,
            "finish_reasons": ["stop"], "usages": [{"total_tokens": 0}]}


def build_record(record: dict, question: str, result: dict, system_prompt: str,
                 generator_meta: dict) -> dict:
    trace = result["trace"]
    prediction = parse_answer(trace)
    expected = result.get("expected")
    reference = [f"{expected:g}"] if expected is not None else []
    metrics = calculate_metrics(prediction, reference)
    facts = extract_facts(trace)
    return {
        "input_sample": record.get("input_sample") or {},
        "gt_answer": record.get("gt_answer"),
        "question": question,
        "answer_complete": prediction is not None,
        "prediction": prediction,
        "full_prediction": trace,
        "prompt": f"{system_prompt}\n\n{build_count_user_message(question)}",
        "full_sample": f"{system_prompt}\n\n{build_count_user_message(question)}{trace}",
        "triples": facts,
        "new_tokens_generated": sum(int(u.get("completion_tokens", 0) or 0) for u in result.get("usages", [])),
        "reached_max_tokens": any(r == "length" for r in result.get("finish_reasons", [])),
        "evaluation": metrics,
        "synthetic": True,
        "generator": generator_meta,
        "verification": {
            "count": expected,
            "answer_ok": metrics["correct"] == 1,
            "ok": bool(result.get("ok")),
            "reasons": result.get("reason", ""),
        },
        "source_dataset": record.get("_dataset"),
        "source_question": record.get("question"),
    }


# ---------------------------------------------------------------------------
# Dataset scanning / CLI
# ---------------------------------------------------------------------------

def read_candidates(path: Path, *, select: str = "all") -> list[dict]:
    meta: dict = {}
    candidates: list[dict] = []
    seen_questions = set()
    with open_log(path) as fd:
        for lineno, line in enumerate(fd):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if lineno == 0 or (not meta and "evaluation" not in obj):
                meta = obj
                continue
            if "macro_precision" in obj or "evaluation" not in obj:
                continue
            input_sample = obj.get("input_sample") or {}
            if question_type(dataset_name(meta), input_sample) != "count":
                continue
            correct = (obj.get("evaluation") or {}).get("correct") == 1
            used_count = "<count>" in (obj.get("full_prediction") or "").lower()
            if select == "teacher-wrong" and correct:
                continue
            if select == "teacher-no-count" and used_count:
                continue
            if select == "teacher-wrong-or-no-count" and (correct and used_count):
                continue
            question = obj.get("question")
            if question in seen_questions:
                continue
            seen_questions.add(question)
            obj["_dataset"] = dataset_name(meta)
            candidates.append(obj)
    return candidates


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path, help="eval log with count questions")
    parser.add_argument("--output", required=True, type=Path, help="output JSONL (eval-log format)")
    parser.add_argument("--n", type=int, default=1000, help="accepted traces to produce")
    parser.add_argument("--max-attempts", type=int, default=None, help="default: 3 * n")
    parser.add_argument("--select", choices=("all", "teacher-wrong", "teacher-no-count",
                                             "teacher-wrong-or-no-count"),
                        default="teacher-wrong-or-no-count")
    parser.add_argument("--count-source", choices=("gold", "kgext"), default="gold")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--thinking", choices=("disabled", "enabled", "omit"), default="disabled")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mock", action="store_true", help="offline generation (no API key)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    candidates = read_candidates(args.input, select=args.select)
    if not candidates:
        raise SystemExit("No matching count questions found.")
    random.Random(args.seed).shuffle(candidates)

    system_prompt = load_system_prompt(args.prompt)
    client = None
    if not args.mock:
        client = DeepSeekClient(model=args.model, thinking=None if args.thinking == "omit" else args.thinking)
    count_source = GoldCountSource() if args.count_source == "gold" else KGExtCountSource()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "experiment_name": f"synthetic_count_{args.input.stem}",
        "dataset": (candidates[0].get("_dataset") or "unknown"),
        "dataset_split": "train",
        "answer_key": "answerText",
        "model_name": args.model,
        "prompt": args.prompt,
        "synthetic": True,
        "generator": {"type": "deepseek", "model": args.model, "count_source": args.count_source,
                      "temperature": args.temperature, "max_tokens": args.max_tokens},
        "date": datetime.datetime.now(datetime.timezone.utc).strftime("%d/%m/%Y %H:%M:%S UTC"),
    }

    accepted = 0
    attempted = 0
    max_attempts = args.max_attempts or (3 * args.n)
    with args.output.open("w", encoding="utf-8") as out:
        out.write(json.dumps(header) + "\n")
        for record in candidates:
            if accepted >= args.n or attempted >= max_attempts:
                break
            attempted += 1
            if args.mock:
                result = mock_generate(record, count_source)
            else:
                result = generate_one(record, client, count_source, system_prompt,
                                      max_turns=args.max_turns, temperature=args.temperature,
                                      max_tokens=args.max_tokens)
            if not result.get("ok"):
                if not args.quiet:
                    print(f"reject [{attempted}]: {result.get('reason')}")
                continue
            sample = build_record(record, record.get("question") or "", result, system_prompt,
                                  {"type": "deepseek" if not args.mock else "mock",
                                   "model": args.model, "count_source": args.count_source})
            out.write(json.dumps(sample) + "\n")
            accepted += 1
            if not args.quiet and accepted % 50 == 0:
                print(f"accepted {accepted}/{args.n} ({attempted} attempts)")
        out.write(json.dumps({"macro_correct_accuracy": 1.0 if attempted else 0.0,
                              "synthetic_count": True, "accepted": accepted,
                              "attempted": attempted}) + "\n")

    print(f"wrote {accepted} accepted count traces to {args.output} "
          f"({attempted} attempts, acceptance {accepted / max(attempted, 1):.1%})")
    if client is not None:
        print(json.dumps(client.usage_summary()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
