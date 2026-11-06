"""Generate count-tool traces with a chat model for the SFT mixture.

Deliberately simple. For every candidate question the tool:

1. sends the real task prompt plus the question **and its gold answer** to the
   model (gold-conditioned generation);
2. takes the single response and checks, with a regex, that the ``<count>``
   tool was used and that the ``<answer>`` block parses;
3. saves the trace in the usual eval-log format.

There is no tool simulation and no external count lookup: the model writes the
count call itself and the data is fully synthetic. The gold answer only steers
the trace; every sample is still scored with ``utils/behaviors.calculate_metrics``
and samples whose answer does not match the reference are dropped unless
``--keep-incorrect`` is given.

Generation uses a meta prompt (``prompts/prompt_qwen36_synthetic_generator.yaml``)
that wraps the real task prompt (``--task-prompt``). The task prompt is also
what is stored in the output records, so the SFT context matches deployment.

Examples::

    DEEPSEEK_API_KEY=... python utils/generate_count_traces.py \
        --input logs/sft_training/mintaka_train_eval_qwen38_27b.out.3.gz \
        --n 1000 --output logs/synthetic/count_traces.jsonl

    # resume an existing run: skip questions already in the output
    DEEPSEEK_API_KEY=... python utils/generate_count_traces.py \
        --input logs/sft_training/mintaka_train_eval_qwen38_27b.out.3.gz \
        --n 1000 --continue --output logs/synthetic/count_traces.jsonl
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
from typing import Any, Optional

import requests
import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils.behaviors import calculate_metrics, dataset_name, parse_answer, question_type  # noqa: E402
from utils.log_io import open_log  # noqa: E402

DEFAULT_PROMPT = "prompts/prompt_qwen36_synthetic_generator.yaml"
TASK_PROMPT = "prompts/prompt_qwen36_angular2_nothink.yaml"
TASK_PROMPT_PLACEHOLDER = "{{TASK_PROMPT}}"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"

COUNT_CALL_RE = re.compile(r"<count\b", re.IGNORECASE)
FACT_RE = re.compile(r"<fact>\s*(.*?)\s*</fact>", re.DOTALL | re.IGNORECASE)


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def load_system_prompt(path: str | Path = DEFAULT_PROMPT) -> str:
    """Load the system prompt from a YAML message list or a plain-text file."""
    content = Path(path).read_text(encoding="utf-8")
    if str(path).lower().endswith((".yaml", ".yml")):
        messages = yaml.safe_load(content)
        if isinstance(messages, list):
            return "\n\n".join(
                str(message.get("content", "")) for message in messages
                if str(message.get("role")) == "system"
            )
    return content


def load_generator_prompt(meta_path: str | Path = DEFAULT_PROMPT,
                          task_path: str | Path = TASK_PROMPT) -> str:
    """Wrap the real task prompt inside the synthetic-example generator prompt."""
    meta = load_system_prompt(meta_path)
    task = load_system_prompt(task_path)
    return meta.replace(TASK_PROMPT_PLACEHOLDER, task)


def _format_reference(value: Any) -> str:
    if value in (None, "", []):
        return "unknown"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return str(value)


def reference_answers(record: dict) -> list[str]:
    raw = record.get("gt_answer")
    if raw in (None, "", []):
        raw = (record.get("input_sample") or {}).get("answerText")
    if isinstance(raw, dict):
        raw = raw.get("answer") or raw.get("value")
    if raw is None:
        return []
    values = raw if isinstance(raw, (list, tuple)) else [raw]
    return [str(value).strip() for value in values if str(value).strip()]


def build_generation_message(question: str, answer: Any) -> str:
    return (
        f"Question: {question}\n"
        f"Reference answer: {_format_reference(answer)}\n\n"
        "Generate the assistant response for this counting question, following the "
        "task prompt. Use the <count> tool, include its returned value in the trace, "
        "and finish with <answer> containing exactly the reference answer. Derive the "
        "value from the count call instead of copying it from this message."
    )


def build_sft_prompt(task_prompt: str, question: str) -> str:
    return f"{task_prompt}\n\nQuestion: {question}"


# ---------------------------------------------------------------------------
# Checks and record building
# ---------------------------------------------------------------------------

def count_used(trace: str) -> bool:
    return bool(COUNT_CALL_RE.search(trace or ""))


def extract_facts(trace: str) -> list[str]:
    return [f" {match.group(1).strip()}" for match in FACT_RE.finditer(trace or "")]


def check_sample(trace: str) -> tuple[bool, list[str]]:
    """The whole acceptance rule: a <count> call and a parseable <answer>."""
    reasons = []
    if not count_used(trace):
        reasons.append("no <count> call")
    if parse_answer(trace) is None:
        reasons.append("no parseable <answer>")
    return not reasons, reasons


def build_record(record: dict, question: str, result: dict, *, task_prompt: str,
                 generator_meta: dict) -> dict:
    trace = result["trace"]
    prediction = parse_answer(trace)
    reference = reference_answers(record)
    metrics = calculate_metrics(prediction, reference)
    return {
        "input_sample": record.get("input_sample") or {},
        "gt_answer": record.get("gt_answer") if record.get("gt_answer") is not None else reference,
        "question": question,
        "answer_complete": prediction is not None,
        "prediction": prediction,
        "full_prediction": trace,
        "prompt": build_sft_prompt(task_prompt, question),
        "full_sample": build_sft_prompt(task_prompt, question) + "\n\n" + trace,
        "triples": extract_facts(trace),
        "new_tokens_generated": sum(
            int(usage.get("completion_tokens", 0) or 0) for usage in result.get("usages", [])
        ),
        "reached_max_tokens": any(r == "length" for r in result.get("finish_reasons", [])),
        "evaluation": metrics,
        "synthetic": True,
        "generator": generator_meta,
        "verification": {
            "count_called": count_used(trace),
            "answer_ok": metrics["correct"] == 1,
            "ok": bool(result.get("ok")),
            "reasons": result.get("reason", ""),
        },
        "source_dataset": record.get("_dataset"),
        "source_question": record.get("question"),
    }


# ---------------------------------------------------------------------------
# Chat client
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
# Generation
# ---------------------------------------------------------------------------

def generate_one(record: dict, client, system_prompt: str, *,
                 temperature: float = 0.7, max_tokens: int = 2048) -> dict:
    """One chat completion; the response is the whole trace."""
    question = record.get("question") or ""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_generation_message(question, record.get("gt_answer"))},
    ]
    response = client.chat(messages, temperature=temperature, max_tokens=max_tokens)
    content, finish_reason, usage = client.content(response)
    return {"trace": content, "finish_reasons": [finish_reason or ""], "usages": [usage]}


def load_existing(output: Path) -> tuple[dict, list[dict], set[str], int, int]:
    """Read an existing output file for ``--continue``.

    Returns ``(header, records, seen questions, correct count, previous
    attempts)``. Only objects that carry an ``evaluation`` block count as
    records, so a previous summary line is ignored (except for its cumulative
    ``attempted`` counter). Missing or empty files return empties.
    """
    if not output.exists():
        return {}, [], set(), 0, 0
    parsed: list[dict] = []
    with output.open(encoding="utf-8") as fd:
        for line in fd:
            line = line.strip()
            if not line:
                continue
            try:
                parsed.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not parsed:
        return {}, [], set(), 0, 0
    first = parsed[0]
    is_header = ("evaluation" not in first
                 and "synthetic_count" not in first
                 and "macro_correct_accuracy" not in first)
    header = first if is_header else {}
    records = [obj for obj in parsed if "evaluation" in obj]
    seen = {str(obj.get("question", "")).strip() for obj in records}
    correct = sum((obj.get("evaluation") or {}).get("correct") == 1 for obj in records)
    previous_attempted = 0
    for obj in reversed(parsed):
        if "evaluation" in obj:
            continue
        previous_attempted = int(obj.get("attempted", 0) or 0)
        break
    # Lower bound when the previous summary is missing (e.g. an interrupted run):
    # every accepted record needed at least one attempt.
    previous_attempted = max(previous_attempted, len(records))
    return header, records, seen, correct, previous_attempted


def read_candidates(path: Path, *, select: str = "teacher-wrong-or-no-count") -> list[dict]:
    """Count questions from an eval log, filtered by the teacher's outcome."""
    meta: dict = {}
    candidates: list[dict] = []
    seen_questions: set[str] = set()
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
            dataset = dataset_name(meta)
            if question_type(dataset, obj.get("input_sample") or {}) != "count":
                continue
            correct = (obj.get("evaluation") or {}).get("correct") == 1
            used_count = count_used(obj.get("full_prediction") or "")
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
            obj["_dataset"] = dataset
            candidates.append(obj)
    return candidates


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path, help="eval log with count questions")
    parser.add_argument("--output", required=True, type=Path, help="output JSONL (eval-log format)")
    parser.add_argument("--n", type=int, default=1000, help="accepted traces to produce")
    parser.add_argument("--max-attempts", type=int, default=None, help="default: 3 * n")
    parser.add_argument("--select", choices=("all", "teacher-wrong", "teacher-no-count",
                                             "teacher-wrong-or-no-count"),
                        default="teacher-wrong-or-no-count")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT,
                        help="meta prompt that wraps the task prompt and asks for an example")
    parser.add_argument("--task-prompt", default=TASK_PROMPT,
                        help="the real task prompt (embedded in the meta prompt and stored in SFT)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--thinking", choices=("disabled", "enabled", "omit"), default="disabled")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-incorrect", action="store_true",
                        help="keep samples whose answer does not match the reference")
    parser.add_argument("--continue", dest="resume", action="store_true",
                        help="resume: skip questions already present in --output and append")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    candidates = read_candidates(args.input, select=args.select)
    if not candidates:
        raise SystemExit("No matching count questions found.")
    random.Random(args.seed).shuffle(candidates)

    header, existing_records, seen, correct, attempted_before = {}, [], set(), 0, 0
    if args.resume:
        header, existing_records, seen, correct, attempted_before = load_existing(args.output)
        candidates = [c for c in candidates
                      if str(c.get("question", "")).strip() not in seen]

    system_prompt = load_generator_prompt(args.prompt, args.task_prompt)
    task_prompt = load_system_prompt(args.task_prompt)
    client = DeepSeekClient(model=args.model,
                            thinking=None if args.thinking == "omit" else args.thinking)

    generator_meta = {
        "type": "deepseek",
        "model": args.model,
        "prompt": args.prompt,
        "task_prompt": args.task_prompt,
        "gold_conditioned": True,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not header:
        dataset = (candidates[0].get("_dataset") if candidates else None) or "unknown"
        header = {
            "experiment_name": f"synthetic_count_{args.input.stem}",
            "dataset": dataset,
            "dataset_split": "train",
            "answer_key": "answerText",
            "model_name": args.model,
            "prompt": args.prompt,
            "task_prompt": args.task_prompt,
            "synthetic": True,
            "generator": generator_meta,
            "date": datetime.datetime.now(datetime.timezone.utc).strftime("%d/%m/%Y %H:%M:%S UTC"),
        }

    accepted = len(existing_records)
    attempted = 0
    remaining = max(args.n - accepted, 0)
    max_attempts = args.max_attempts if args.max_attempts is not None else (3 * remaining)

    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover - tqdm is a declared dependency
        tqdm = None
    progress = (tqdm(candidates, total=min(len(candidates), max_attempts),
                     desc="count traces", unit="q", disable=args.quiet)
                if tqdm is not None else candidates)

    with args.output.open("w", encoding="utf-8") as out:
        out.write(json.dumps(header) + "\n")
        for record in existing_records:
            out.write(json.dumps(record) + "\n")
        for record in progress:
            if accepted >= args.n or attempted >= max_attempts:
                break
            attempted += 1
            result = generate_one(record, client, system_prompt,
                                  temperature=args.temperature, max_tokens=args.max_tokens)
            trace = result["trace"]
            ok, reasons = check_sample(trace)
            if ok and not args.keep_incorrect:
                prediction = parse_answer(trace)
                if calculate_metrics(prediction, reference_answers(record))["correct"] != 1:
                    ok = False
                    reasons.append("answer does not match the reference")
            if not ok:
                if not args.quiet:
                    print(f"reject [{attempted}]: {'; '.join(reasons)}")
                continue

            result["ok"] = True
            result["reason"] = ""
            sample = build_record(record, record.get("question") or "", result,
                                  task_prompt=task_prompt, generator_meta=generator_meta)
            out.write(json.dumps(sample) + "\n")
            accepted += 1
            correct += sample["evaluation"]["correct"] == 1
            if tqdm is not None and not args.quiet:
                progress.set_postfix(accepted=accepted,
                                     new=accepted - len(existing_records),
                                     tried=attempted)
        out.write(json.dumps({
            "macro_correct_accuracy": (correct / accepted) if accepted else 0.0,
            "synthetic_count": True,
            "accepted": accepted,
            "attempted": attempted_before + attempted,
            "attempted_this_run": attempted,
            "resumed_from": len(existing_records),
        }) + "\n")

    new_accepted = accepted - len(existing_records)
    print(f"wrote {accepted} accepted count traces to {args.output} "
          f"(+{new_accepted} this run, {attempted} attempts, "
          f"acceptance {new_accepted / max(attempted, 1):.1%})")
    print(json.dumps(client.usage_summary()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
