"""Generate backtracking (explore → detect error → correct) SFT traces.

This implements ``sft_synthetic_data_plan.md`` §3.4. A kept trace must show the
model exploring a *plausible but wrong* branch grounded in real retrieved facts,
noticing that it is wrong, **closing** that branch, and then recovering to the
correct answer. Each trace also carries the wrong continuation as a ``dpo_pair``
(``chosen`` = corrected trace, ``rejected`` = trace that answers the wrong
candidate).

Two generation modes:

``inject`` (default, offline, no API key)
    Start from a *correct* teacher trace and splice in a detour built only from
    that record's retrieved triples: a real distractor fact becomes the wrong
    branch, an explicit rejection closes it, and the real answer path resumes.
    Deterministic, free, and grounded by construction. This is the recommended
    source because every fact in the output comes from the record's own
    ``triples``.

``generate`` (DeepSeek)
    Prompt the model to produce the self-correcting trace, simulating the
    ``<fact>`` tool from the record's retrieved triples (the model emits
    ``<fact> <Subject> <relation>``; the harness injects the real object, or
    ``<no further records>`` when the relation is exhausted). Every candidate
    then goes through the same strict validator as ``inject``, so ungrounded
    facts, missing correction spans and open wrong branches are rejected.
    Use ``--mock`` for a deterministic offline dry run.

The output is a JSONL log in the same shape as ``utils/eval.py`` (header,
records, ``macro_*`` summary) with the synthetic fields used by
``utils/build_sft_dataset.py``.

``--mode generate`` uses a *meta* prompt
(``prompts/prompt_qwen36_synthetic_generator.yaml``) that wraps the real task
prompt (``--task-prompt``) and asks the model to produce one training example
from a question and its correct answer. The correct answer only steers the
trace; the validator still enforces grounded facts and closes the wrong branch.
Every record is tagged ``generator.gold_conditioned = true``. The task prompt is
also what is stored in the output records, so the SFT context matches
deployment.

Examples::

    # offline: build backtracking traces from correct Mintaka traces
    python utils/generate_backtracking_traces.py \
        --input logs/sft_training/mintaka_train_eval_qwen38_27b.out.3.gz \
        --select positive --n 400 --mode inject \
        --output logs/synthetic/backtracking_mintaka.jsonl

    # offline dry run of the prompted generator (no API key)
    python utils/generate_backtracking_traces.py \
        --input logs/sft_training/mintaka_train_eval_qwen38_27b.out.3.gz \
        --n 20 --mode generate --mock --output /tmp/backtracking.jsonl

    # real prompted generation
    DEEPSEEK_API_KEY=... python utils/generate_backtracking_traces.py \
        --input logs/behaviors/mintaka.behaviors --select wrong-with-facts \
        --n 400 --mode generate --output logs/synthetic/backtracking_prompted.jsonl
"""

from __future__ import annotations

import argparse
import datetime
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils.answer_normalization import fold_entity, scalar_matches, verify_answer  # noqa: E402
from utils.behaviors import (  # noqa: E402
    ANSWER_BLOCK_RE,
    calculate_metrics,
    dataset_name,
    parse_answer,
)
from utils.log_io import open_log  # noqa: E402

DEFAULT_PROMPT = "prompts/prompt_qwen36_synthetic_generator.yaml"
TASK_PROMPT = "prompts/prompt_qwen36_angular2_nothink.yaml"
TASK_PROMPT_PLACEHOLDER = "{{TASK_PROMPT}}"
SENTINEL = "no further records"
BEHAVIOUR_TAGS = ["backtracking", "self-correction", "grounded"]

FACT_BLOCK_RE = re.compile(r"<fact>\s*(.*?)\s*</fact>", re.DOTALL | re.IGNORECASE)
FACT_OPEN_RE = re.compile(r"<fact\s*>", re.IGNORECASE)
TRIPLE_PARTS_RE = re.compile(r"<([^<>\n]+)>\s*<([^<>\n]+)>\s*<([^<>\n]+)>")
OPEN_FACT_PREFIX_RE = re.compile(r"\s*<([^<>]+)>\s*<([^<>]+)>")

# "explore → detect error" markers (plan §3.4.3).
CORRECTION_RE = re.compile(
    r"(?:\bwrong\b|\bincorrect\b|\binstead\b|\bactually\b|\brather\b|"
    r"that'?s not|that is not|not the right|\bno,|"
    r"\bmistake\b|\berror\b)",
    re.IGNORECASE,
)
# "close the wrong branch" markers (plan §3.4.3).
CLOSURE_RE = re.compile(
    r"(?:\breject\w*|\bdiscard\w*|\bignor\w*|\brule[sd]?\s+out|\babandon\w*|"
    r"\bset\s+aside|\bdrop\w*|\bclos\w*|\bwrong\s+branch|"
    r"\bdead\s+end|\bback\s+out|\bbacktrack\w*)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Triple / trace parsing (pure functions, tested)
# ---------------------------------------------------------------------------

def normalize_triple(text: str) -> str:
    """Collapse whitespace and drop a trailing period (same as log_analysis)."""
    return " ".join(str(text or "").split()).rstrip(".").strip()


def load_system_prompt(path: str | Path = DEFAULT_PROMPT) -> str:
    """Load the system prompt from a YAML message list or a plain-text file."""
    content = Path(path).read_text(encoding="utf-8")
    if str(path).lower().endswith((".yaml", ".yml")):
        import yaml
        messages = yaml.safe_load(content)
        if isinstance(messages, list):
            return "\n\n".join(
                str(message.get("content", "")) for message in messages
                if str(message.get("role")) == "system"
            )
    return content


def parse_triple(text: str) -> Optional[dict]:
    """Parse one ``<s> <r> <o>`` string (optionally with a trailing period)."""
    match = TRIPLE_PARTS_RE.search(text or "")
    if not match:
        return None
    subject, relation, obj = (group.strip() for group in match.groups())
    return {
        "subject": subject,
        "relation": relation,
        "object": obj,
        "triple": f"<{subject}> <{relation}> <{obj}>",
        "sentinel": SENTINEL.lower() in obj.lower(),
    }


def record_triples(record: dict) -> list[dict]:
    """The record's retrieved triples, parsed and deduplicated (order kept)."""
    parsed: list[dict] = []
    seen: set[str] = set()
    for raw in record.get("triples") or []:
        triple = parse_triple(normalize_triple(str(raw)))
        if triple is None or triple["triple"] in seen:
            continue
        seen.add(triple["triple"])
        parsed.append(triple)
    return parsed


def parse_fact_calls(trace: str) -> list[dict]:
    """All ``<fact> ... </fact>`` blocks with offsets and the triple they carry."""
    calls: list[dict] = []
    for match in FACT_BLOCK_RE.finditer(trace or ""):
        triple = parse_triple(match.group(1))
        if triple is None:
            continue
        calls.append({
            "start": match.start(),
            "end": match.end(),
            "subject": triple["subject"],
            "relation": triple["relation"],
            "object": triple["object"],
            "triple": triple["triple"],
            "sentinel": triple["sentinel"],
        })
    return calls


def parse_open_fact_calls(trace: str) -> list[dict]:
    """``<fact> <Subject> <relation>`` calls the harness still has to answer."""
    calls: list[dict] = []
    text = trace or ""
    for match in FACT_OPEN_RE.finditer(text):
        rest = text[match.end():]
        if re.search(r"</fact\s*>", rest, re.IGNORECASE):
            continue  # already complete
        prefix = OPEN_FACT_PREFIX_RE.match(rest)
        if not prefix:
            continue
        calls.append({
            "start": match.start(),
            "prefix_end": match.end() + prefix.end(),
            "subject": prefix.group(1).strip(),
            "relation": prefix.group(2).strip(),
        })
    return calls


def inject_fact_result(text: str, call: dict, obj: Optional[str]) -> str:
    """Complete a ``<fact> <S> <R>`` call with the harness-returned object."""
    value = obj if obj is not None else SENTINEL
    insertion = f" <{value}> . </fact>\n"
    return text[:call["prefix_end"]] + insertion + text[call["prefix_end"]:]


def extract_facts(trace: str) -> list[str]:
    """Facts in the trace, in the shape ``utils/log_analysis.record_triples`` expects."""
    return [f" {m.group(1).strip()}" for m in FACT_BLOCK_RE.finditer(trace or "")]


def first_correction(trace: str, calls: list[dict]) -> Optional[dict]:
    """The first correction marker that follows at least one fact call."""
    for match in CORRECTION_RE.finditer(trace or ""):
        if any(call["end"] <= match.start() for call in calls):
            return {"start": match.start(), "end": match.end(), "text": match.group(0)}
    return None


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


def _contains(haystack: str, needle: str) -> bool:
    needle = fold_entity(needle)
    if not needle:
        return False
    if len(needle) < 3 or needle.replace(".", "").isdigit():
        return re.search(r"(?<!\w)" + re.escape(needle) + r"(?!\w)", haystack) is not None
    return needle in haystack


def matches_answer(value: str, answers: list[str]) -> bool:
    """Whether a triple object bears one of the answers.

    Exact/alias containment covers entity answers; ``scalar_matches`` adds the
    dataset-aware date and quantity rules (e.g. the ISO object
    ``1956-10-20T00:00:00Z`` bearing the answer ``20-Oct-56``).
    """
    folded = fold_entity(value)
    for answer in answers:
        if _contains(folded, answer) or _contains(fold_entity(answer), value):
            return True
        try:
            matched, _, _ = scalar_matches(answer, value)
        except (TypeError, ValueError):
            matched = False
        if matched:
            return True
    return False


# ---------------------------------------------------------------------------
# Validation (shared by both modes)
# ---------------------------------------------------------------------------

def validate_backtracking_trace(trace: str, record: Optional[dict] = None, *,
                                dataset: str = "", max_chars: int = 6000) -> tuple[bool, list[str], dict]:
    """Check structure, correctness, the backtracking contract and grounding.

    Returns ``(ok, reasons, diagnostics)``. All checks are required: the final
    answer must be correct, the trace must contain a correction *after* a wrong
    branch and a recovery *after* the correction, the wrong branch must be
    closed, and every ``<fact>`` must be among the record's retrieved triples.
    """
    reasons: list[str] = []
    diagnostics: dict = {}
    text = trace or ""

    # 1. Structure.
    if len(ANSWER_BLOCK_RE.findall(text)) != 1:
        reasons.append("expected exactly one <answer> block")
    if text.lower().count("<fact>") != text.lower().count("</fact>"):
        reasons.append("incomplete <fact> call")
    if len(text) > max_chars:
        reasons.append(f"trace too long ({len(text)} > {max_chars} chars)")

    prediction = parse_answer(text)
    if prediction is None:
        reasons.append("no parseable <answer>")

    # 2. Correctness (strict evaluator OR dataset-aware normalisation).
    answers = reference_answers(record or {})
    if prediction is not None and answers:
        strict = calculate_metrics(prediction, answers)["correct"] == 1
        verification = verify_answer(
            prediction, answers, dataset=dataset,
            question=(record or {}).get("question"),
            input_sample=(record or {}).get("input_sample"),
        )
        diagnostics["strict_match"] = strict
        diagnostics["normalized_match"] = verification.normalized_match
        diagnostics["verification_rule"] = verification.rule
        if not (strict or verification.normalized_match):
            reasons.append(f"final answer {prediction} is not correct")
    elif prediction is not None and not answers:
        reasons.append("no reference answer to verify against")

    # 3. Backtracking contract.
    calls = parse_fact_calls(text)
    diagnostics["n_fact_calls"] = len(calls)
    if len(calls) < 2:
        reasons.append("need at least two <fact> calls (wrong branch + recovery)")

    correction = first_correction(text, calls)
    if correction is None:
        reasons.append("no correction span after a <fact> call")
    else:
        diagnostics["correction"] = correction["text"]
        diagnostics["correction_index"] = correction["start"]
        before = [call for call in calls if call["end"] <= correction["start"]]
        after = [call for call in calls if call["start"] >= correction["end"]]
        if not before:
            reasons.append("correction precedes every <fact> call (no wrong branch explored)")
        if not after:
            reasons.append("no recovery <fact> after the correction")
        if before:
            wrong_branch = before[-1]
            diagnostics["wrong_candidate"] = wrong_branch["object"]
            diagnostics["wrong_branch_end"] = wrong_branch["end"]
            if matches_answer(wrong_branch["object"], answers):
                reasons.append("wrong branch already contains the answer")
            if after:
                window = text[wrong_branch["end"]:after[0]["start"]]
            else:
                window = text[wrong_branch["end"]:correction["start"] + 200]
            closed = bool(CLOSURE_RE.search(window)) or SENTINEL.lower() in window.lower()
            diagnostics["wrong_branch_closed"] = closed
            if not closed:
                reasons.append("wrong branch is not closed (no rejection/sentinel)")
        else:
            diagnostics["wrong_branch_closed"] = False

    # 4. Grounding: every asserted fact must be a retrieved triple.
    if record is not None:
        known = {triple["triple"] for triple in record_triples(record)}
        ungrounded = [call["triple"] for call in calls
                      if not call["sentinel"] and call["triple"] not in known]
        diagnostics["grounded"] = not ungrounded
        if ungrounded:
            reasons.append("ungrounded fact(s): " + "; ".join(ungrounded[:3]))
    else:
        diagnostics["grounded"] = None

    diagnostics["answer_ok"] = prediction is not None and not any(
        r.startswith("final answer") for r in reasons
    )
    return not reasons, reasons, diagnostics


# ---------------------------------------------------------------------------
# Mode 1: programmatic injection (offline)
# ---------------------------------------------------------------------------

def _answer_bearing(triples: list[dict], answers: list[str], max_fact_chars: int = 200) -> list[dict]:
    return [
        t for t in triples
        if not t["sentinel"]
        and len(t["object"]) <= max_fact_chars
        and matches_answer(t["object"], answers)
    ]


def _pick_distractor(triples: list[dict], path: list[dict], answers: list[str],
                     max_fact_chars: int = 200) -> Optional[dict]:
    """A real fact that is plausible but cannot answer the question.

    Only triples *outside* the reconstructed answer path are eligible: rejecting
    a hop the answer actually needs would make the trace self-contradictory.
    Prefer a triple that shares the subject of the path (the classic "right
    entity, wrong relation" detour).
    """
    path_keys = {t["triple"] for t in path}
    path_subjects = {fold_entity(t["subject"]) for t in path}
    path_relations = {fold_entity(t["relation"]) for t in path}
    candidates = [
        t for t in triples
        if t["triple"] not in path_keys
        and not t["sentinel"]
        and len(t["object"]) <= max_fact_chars
        and not matches_answer(t["object"], answers)
        and not matches_answer(t["subject"], answers)
    ]
    if not candidates:
        return None
    preferred = [
        t for t in candidates
        if fold_entity(t["subject"]) in path_subjects
        and fold_entity(t["relation"]) not in path_relations
    ]
    return (preferred or candidates)[0]


def correct_path(triples: list[dict], bearing: list[dict]) -> list[dict]:
    """The answer path, walking backwards from the answer-bearing triple(s)."""
    chosen = list(bearing)
    changed = True
    while changed:
        changed = False
        chosen_keys = {t["triple"] for t in chosen}
        subjects = {fold_entity(t["subject"]) for t in chosen}
        for triple in triples:
            if triple["triple"] in chosen_keys:
                continue
            if triple["sentinel"] or fold_entity(triple["object"]) not in subjects:
                continue
            chosen.append(triple)
            changed = True
    order = {triple["triple"]: i for i, triple in enumerate(triples)}
    return sorted(chosen, key=lambda t: order.get(t["triple"], 0))


def build_injected_trace(record: dict, *, max_fact_chars: int = 200) -> dict:
    """Splice a wrong branch + correction into a correct trace, offline."""
    answers = reference_answers(record)
    if not answers:
        return {"ok": False, "reason": "no reference answer"}
    triples = record_triples(record)
    if len(triples) < 2:
        return {"ok": False, "reason": "need at least two retrieved triples"}
    bearing = _answer_bearing(triples, answers, max_fact_chars)
    if not bearing:
        return {"ok": False, "reason": "no answer-bearing triple"}
    path = correct_path(triples, bearing)
    wrong = _pick_distractor(triples, path, answers, max_fact_chars)
    if wrong is None:
        return {"ok": False, "reason": "no distractor triple outside the answer path"}

    question = (record.get("question") or "").strip()
    if question and question[-1] not in ".?!":
        question += "."
    right = bearing[-1]
    intro = (
        f"I need to answer: {question} The most direct first step seems to be the relation "
        f"\"{wrong['relation']}\" of \"{wrong['subject']}\", so I will check that first."
    )
    wrong_fact = (
        f"<fact> <{wrong['subject']}> <{wrong['relation']}> <{wrong['object']}> . </fact>"
    )
    rejection = (
        f"This returns \"{wrong['object']}\". Taken on its own it is a plausible candidate, but "
        f"the question asks for the \"{right['relation']}\" of \"{right['subject']}\", not the "
        f"\"{wrong['relation']}\" of \"{wrong['subject']}\", so \"{wrong['object']}\" is not the "
        f"answer. That's not the right path: I reject this candidate and back out of this branch."
    )
    recovery_intro = (
        "I will instead follow the retrieved facts that actually answer the question and drop "
        "this branch."
    )
    facts = [
        f"<fact> <{t['subject']}> <{t['relation']}> <{t['object']}> . </fact>" for t in path
    ]
    conclusion = (
        f"The retrieved facts give {', '.join(answers)}, which supports the answer and matches "
        f"the question, so I have sufficient evidence."
    )
    answer_block = "<answer>\n" + json.dumps(answers, ensure_ascii=False) + "\n</answer>"

    trace = "\n".join([intro, wrong_fact, rejection, recovery_intro, *facts, conclusion, answer_block])

    rejected_trace = "\n".join([
        intro,
        wrong_fact,
        f"This returns \"{wrong['object']}\", which fits the first reading of the question, "
        f"so I will answer with it.",
        "<answer>\n" + json.dumps([wrong["object"]], ensure_ascii=False) + "\n</answer>",
    ])

    return {
        "ok": True,
        "reason": "",
        "trace": trace,
        "rejected_trace": rejected_trace,
        "wrong_candidate": wrong["object"],
        "expected": answers,
        "finish_reasons": ["stop"],
        "usages": [],
    }


def rejected_from_trace(trace: str, diagnostics: dict) -> Optional[str]:
    """A wrong completion for DPO, cut at the end of the wrong branch."""
    wrong = diagnostics.get("wrong_candidate")
    end = diagnostics.get("wrong_branch_end")
    if not wrong or end is None:
        return None
    return (
        trace[:end].rstrip()
        + f"\nThis gives \"{wrong}\", which fits the first reading of the question, so I "
          f"will answer with it.\n"
        + "<answer>\n" + json.dumps([wrong], ensure_ascii=False) + "\n</answer>"
    )


def inject_one(record: dict, *, max_chars: int = 6000, max_fact_chars: int = 200,
               dataset: str = "") -> dict:
    """Build an injected trace and validate it."""
    result = build_injected_trace(record, max_fact_chars=max_fact_chars)
    if not result["ok"]:
        return result
    ok, reasons, diagnostics = validate_backtracking_trace(
        result["trace"], record, dataset=dataset, max_chars=max_chars
    )
    result.update({"ok": ok, "reason": "; ".join(reasons), "diagnostics": diagnostics})
    return result


# ---------------------------------------------------------------------------
# Mode 2: prompted generation (DeepSeek + simulated fact tool)
# ---------------------------------------------------------------------------

def load_generator_prompt(meta_path: str | Path = DEFAULT_PROMPT,
                          task_path: str | Path = TASK_PROMPT) -> str:
    """Wrap the real task prompt inside the synthetic-example generator prompt."""
    meta = load_system_prompt(meta_path)
    task = load_system_prompt(task_path)
    return meta.replace(TASK_PROMPT_PLACEHOLDER, task)


def build_backtracking_user_message(question: str, answer=None) -> str:
    """Gold-conditioned user turn for the example generator."""
    reference = _format_reference(answer)
    return (
        f"Question: {question}\n"
        f"Reference answer: {reference}\n\n"
        "Generate the assistant response for this question, following the task prompt, "
        "and demonstrate a self-correcting (backtracking) trace with exactly this shape:\n"
        "1. Choose a plausible but WRONG first branch and retrieve it with one <fact> "
        "request written as `<fact> <Subject> <relation>` (the harness appends the real "
        "object).\n"
        "2. Say explicitly why that candidate is wrong for this question (wrong relation, "
        "wrong entity type, or contradiction with another retrieved fact).\n"
        "3. CLOSE the wrong branch: reject it explicitly or exhaust it to "
        "`<no further records>`. Never leave a wrong path open.\n"
        "4. Say you will instead do something different, then retrieve the correct path "
        "with further <fact> requests.\n"
        "5. Answer with the reference answer in <answer>[...]</answer>, derived from the "
        "retrieved facts rather than copied from this message.\n"
        "Only use objects the harness returns; never invent facts.\n\n"
        "Example of the required shape (the harness fills in the objects and closes "
        "the tags):\n"
        "Question: Who directed Slumdog Millionaire?\n\n"
        "I will first check a plausible but likely wrong relation, the producer of "
        "Slumdog Millionaire, to rule it out.\n"
        "<fact> <Slumdog Millionaire> <producer>\n"
        "The fact tool returned Christian Colson. That is the producer, not the "
        "director, so it cannot answer the question. That's not the right path: I "
        "reject this candidate and back out of this branch.\n"
        "I will instead retrieve the relation the question actually asks for.\n"
        "<fact> <Slumdog Millionaire> <director>\n"
        "The fact tool returned Danny Boyle, which directly answers the question, so "
        "I have sufficient evidence and need no further facts.\n"
        "<answer>\n[\"Danny Boyle\"]\n</answer>"
    )


def _format_reference(value) -> str:
    if value in (None, "", []):
        return "unknown"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return str(value)


class SourceFactOracle:
    """Answer ``<fact>`` calls from the source record's retrieved triples."""

    def __init__(self, record: dict):
        self.by_prefix: dict[tuple[str, str], list[str]] = {}
        for triple in record_triples(record):
            if triple["sentinel"]:
                continue
            key = (fold_entity(triple["subject"]), fold_entity(triple["relation"]))
            objects = self.by_prefix.setdefault(key, [])
            if triple["object"] not in objects:
                objects.append(triple["object"])
        self.served: Counter = Counter()

    def _lookup(self, subject: str, relation: str) -> Optional[tuple[str, str]]:
        key = (fold_entity(subject), fold_entity(relation))
        if key in self.by_prefix:
            return key
        matches = [
            candidate for candidate in self.by_prefix
            if _contains(candidate[0], subject) and _contains(candidate[1], relation)
        ]
        return matches[0] if len(matches) == 1 else None

    def __call__(self, subject: str, relation: str) -> dict:
        key = self._lookup(subject, relation)
        if key is None:
            return {"object": None, "complete": True, "unknown": True}
        objects = self.by_prefix[key]
        served = self.served[key]
        if served >= len(objects):
            return {"object": None, "complete": True, "unknown": False}
        self.served[key] = served + 1
        return {"object": objects[served], "complete": False, "unknown": False}


def generate_one(record: dict, client, system_prompt: str, *, max_turns: int = 4,
                 temperature: float = 0.7, max_tokens: int = 2048) -> dict:
    """Prompt the model, simulating the fact tool from the record's triples."""
    question = record.get("question") or ""
    user_message = build_backtracking_user_message(question, record.get("gt_answer"))
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    oracle = SourceFactOracle(record)
    parts: list[str] = []
    finish_reasons: list[str] = []
    usages: list[dict] = []

    for _ in range(max_turns):
        response = client.chat(messages, temperature=temperature, max_tokens=max_tokens)
        content, finish_reason, usage = client.content(response)
        finish_reasons.append(finish_reason or "")
        usages.append(usage)

        pending = parse_open_fact_calls(content)
        if pending:
            call = pending[-1]
            result = oracle(call["subject"], call["relation"])
            injected = inject_fact_result(content, call, result["object"])
            parts.append(injected)
            messages.append({"role": "assistant", "content": injected})
            if result["complete"]:
                reply = f"<{SENTINEL}> . </fact>\nThe relation is exhausted; no further records."
            else:
                reply = f"<{result['object']}> . </fact>\nThe fact tool returned that object."
            messages.append({"role": "user", "content": reply})
            continue

        parts.append(content)
        messages.append({"role": "assistant", "content": content})
        if "<answer>" in content.lower():
            break
        messages.append({"role": "user", "content":
                         "Continue: explore a wrong branch, correct it, then answer."})

    trace = "".join(parts)
    return {
        "ok": True,
        "reason": "",
        "trace": trace,
        "rejected_trace": None,
        "wrong_candidate": None,
        "expected": reference_answers(record),
        "finish_reasons": finish_reasons,
        "usages": usages,
        "user_message": user_message,
    }


def mock_generate(record: dict) -> dict:
    """Deterministic offline stand-in for ``generate_one`` (same as injection)."""
    result = build_injected_trace(record)
    result["user_message"] = build_backtracking_user_message(record.get("question") or "")
    return result


# ---------------------------------------------------------------------------
# Records and CLI
# ---------------------------------------------------------------------------

def build_record(record: dict, question: str, result: dict, *, system_prompt: str,
                 generator_meta: dict, dataset: str, sft_prompt: Optional[str] = None) -> dict:
    trace = result["trace"]
    prediction = parse_answer(trace)
    answers = reference_answers(record)
    metrics = calculate_metrics(prediction, answers)
    verification = verify_answer(
        prediction, answers, dataset=dataset, question=question,
        input_sample=record.get("input_sample"),
    )
    sft_system_prompt = sft_prompt if sft_prompt is not None else system_prompt
    diagnostics = result.get("diagnostics") or {}
    return {
        "input_sample": record.get("input_sample") or {},
        "gt_answer": record.get("gt_answer") if record.get("gt_answer") is not None else answers,
        "question": question,
        "answer_complete": prediction is not None,
        "prediction": prediction,
        "full_prediction": trace,
        # Inference-style prompt: the behaviour instruction is *not* part of the
        # prompt, so the trace teaches the behaviour rather than obedience to a
        # synthetic instruction. The generation prompt is kept under `generator`.
        "prompt": f"{sft_system_prompt}\n\nQuestion: {question}",
        "full_sample": f"{sft_system_prompt}\n\nQuestion: {question}\n\n{trace}",
        "triples": extract_facts(trace),
        "new_tokens_generated": sum(
            int(usage.get("completion_tokens", 0) or 0) for usage in result.get("usages", [])
        ),
        "reached_max_tokens": any(r == "length" for r in result.get("finish_reasons", [])),
        "evaluation": metrics,
        "synthetic": True,
        "generator": generator_meta,
        "behaviour_tags": list(BEHAVIOUR_TAGS),
        "verification": {
            "strict": diagnostics.get("strict_match"),
            "normalized": diagnostics.get("normalized_match") or verification.normalized_match,
            "rule": verification.rule,
            "grounded": diagnostics.get("grounded"),
            "closed": diagnostics.get("wrong_branch_closed"),
            "answer_ok": metrics["correct"] == 1,
            "ok": bool(result.get("ok")),
            "reasons": result.get("reason", ""),
        },
        "backtracking": {
            "correction": diagnostics.get("correction"),
            "correction_index": diagnostics.get("correction_index"),
            "wrong_candidate": result.get("wrong_candidate") or diagnostics.get("wrong_candidate"),
            "closed": diagnostics.get("wrong_branch_closed"),
        },
        "dpo_pair": ({"rejected_trace": result["rejected_trace"],
                      "rejected_answer": [result["wrong_candidate"]]}
                     if result.get("rejected_trace") else None),
        "source_dataset": dataset,
        "source_question": record.get("question"),
    }


def read_candidates(path: Path, *, select: str = "positive", verified: bool = False,
                    min_triples: int = 2) -> list[dict]:
    """Read positive (or ``wrong_with_facts``) records usable as seeds."""
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
            if len(record_triples(obj)) < min_triples:
                continue
            if obj.get("reached_max_tokens"):
                continue

            ev = obj.get("evaluation") or {}
            diagnostic = (obj.get("behavior") or {}).get("diagnostic")
            if select == "wrong-with-facts" and diagnostic != "wrong_with_facts":
                continue
            if select == "positive":
                if verified:
                    if not verify_answer(
                        obj.get("prediction"), reference_answers(obj), dataset=dataset,
                        question=obj.get("question"), input_sample=obj.get("input_sample"),
                    ).normalized_match:
                        continue
                elif ev.get("correct") != 1:
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
    parser.add_argument("--input", required=True, type=Path, help="eval or behaviour log")
    parser.add_argument("--output", required=True, type=Path, help="output JSONL (eval-log format)")
    parser.add_argument("--mode", choices=("inject", "generate"), default="inject",
                        help="inject: offline splicing from correct traces; "
                             "generate: prompted DeepSeek generation")
    parser.add_argument("--n", type=int, default=1000, help="accepted traces to produce")
    parser.add_argument("--max-attempts", type=int, default=None,
                        help="default: all seeds in inject mode, 3 * n in generate mode")
    parser.add_argument("--select", choices=("positive", "wrong-with-facts", "all"),
                        default="positive")
    parser.add_argument("--verified", action="store_true",
                        help="select positives with the dataset-aware verifier")
    parser.add_argument("--min-triples", type=int, default=2)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT,
                        help="meta prompt that wraps the task prompt and asks for an example")
    parser.add_argument("--task-prompt", default=TASK_PROMPT,
                        help="the real task prompt (embedded in the meta prompt and stored in SFT)")
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--max-chars", type=int, default=6000,
                        help="reject traces longer than this (plan §8: avoid rambling)")
    parser.add_argument("--max-fact-chars", type=int, default=200,
                        help="ignore facts whose object is longer (e.g. descriptions)")
    parser.add_argument("--thinking", choices=("disabled", "enabled", "omit"), default="disabled")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mock", action="store_true",
                        help="offline dry run of --mode generate")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    candidates = read_candidates(args.input, select=args.select,
                                 verified=args.verified, min_triples=args.min_triples)
    if not candidates:
        raise SystemExit("No matching seed records found.")
    random.Random(args.seed).shuffle(candidates)

    # Imported lazily so the offline (``inject``) path and the tests do not need
    # the API client module (which pulls in ``pandas`` via ``log_analysis``).
    system_prompt = load_generator_prompt(args.prompt, args.task_prompt)
    sft_system_prompt = load_system_prompt(args.task_prompt)
    client = None
    if args.mode == "generate" and not args.mock:
        from utils.generate_count_traces import DeepSeekClient

        client = DeepSeekClient(model=args.model,
                                thinking=None if args.thinking == "omit" else args.thinking)

    dataset = candidates[0].get("_dataset") or "unknown"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "experiment_name": f"synthetic_backtracking_{args.input.stem}",
        "dataset": dataset,
        "dataset_split": "train",
        "answer_key": "answerText",
        "model_name": args.model if args.mode == "generate" else "inject",
        "prompt": args.prompt,
        "task_prompt": args.task_prompt,
        "synthetic": True,
        "behaviour": "backtracking",
        "generator": {
            "type": "inject" if args.mode == "inject" or args.mock else "deepseek",
            "model": None if args.mode == "inject" else args.model,
            "mode": args.mode,
            "prompt": args.prompt,
            "task_prompt": args.task_prompt,
            "gold_conditioned": True,
            "max_chars": args.max_chars,
            "temperature": args.temperature if args.mode == "generate" else None,
        },
        "date": datetime.datetime.now(datetime.timezone.utc).strftime("%d/%m/%Y %H:%M:%S UTC"),
    }

    accepted = 0
    attempted = 0
    rejects: Counter = Counter()
    max_attempts = args.max_attempts
    if max_attempts is None:
        max_attempts = len(candidates) if args.mode == "inject" else 3 * args.n
    with args.output.open("w", encoding="utf-8") as out:
        out.write(json.dumps(header) + "\n")
        for record in candidates:
            if accepted >= args.n or attempted >= max_attempts:
                break
            attempted += 1
            if args.mode == "inject":
                result = inject_one(record, max_chars=args.max_chars,
                                    max_fact_chars=args.max_fact_chars, dataset=dataset)
            elif args.mock:
                result = mock_generate(record)
                if result.get("ok"):
                    ok, reasons, diagnostics = validate_backtracking_trace(
                        result["trace"], record, dataset=dataset, max_chars=args.max_chars
                    )
                    result.update({"ok": ok, "reason": "; ".join(reasons),
                                   "diagnostics": diagnostics})
            else:
                result = generate_one(record, client, system_prompt,
                                      max_turns=args.max_turns, temperature=args.temperature,
                                      max_tokens=args.max_tokens)
                ok, reasons, diagnostics = validate_backtracking_trace(
                    result["trace"], record, dataset=dataset, max_chars=args.max_chars
                )
                result.update({"ok": ok, "reason": "; ".join(reasons), "diagnostics": diagnostics})
                if result.get("rejected_trace") is None:
                    result["rejected_trace"] = rejected_from_trace(result["trace"], diagnostics)
                    result.setdefault("wrong_candidate", diagnostics.get("wrong_candidate"))

            if not result.get("ok"):
                rejects[result.get("reason") or "unknown"] += 1
                if not args.quiet:
                    print(f"reject [{attempted}]: {result.get('reason')}")
                continue

            generator_meta = dict(header["generator"])
            if args.mode == "generate" and not args.mock:
                generator_meta["instruction"] = build_backtracking_user_message(
                    record.get("question") or "", record.get("gt_answer"))
            sample = build_record(record, record.get("question") or "", result,
                                  system_prompt=system_prompt, generator_meta=generator_meta,
                                  dataset=dataset, sft_prompt=sft_system_prompt)
            out.write(json.dumps(sample) + "\n")
            accepted += 1
            if not args.quiet and accepted % 50 == 0:
                print(f"accepted {accepted}/{args.n} ({attempted} attempts)")

        out.write(json.dumps({
            "macro_correct_accuracy": 1.0 if attempted else 0.0,
            "synthetic_backtracking": True,
            "mode": args.mode,
            "accepted": accepted,
            "attempted": attempted,
            "rejects": dict(rejects),
        }) + "\n")

    print(f"wrote {accepted} accepted backtracking traces to {args.output} "
          f"({attempted} attempts, acceptance {accepted / max(attempted, 1):.1%})")
    if args.mode == "inject" and args.select != "positive" and not args.quiet:
        print("note: inject mode rebuilds the correct path, so non-positive seeds are "
              "rejected by the validator; use --select positive.")
    if rejects and not args.quiet:
        print("top rejections: " + ", ".join(f"{reason} ({count})"
                                             for reason, count in rejects.most_common(5)))
    if client is not None:
        print(json.dumps(client.usage_summary()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
