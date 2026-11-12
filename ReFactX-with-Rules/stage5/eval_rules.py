"""Generate one bare Nemo rule and score its results against independent gold."""
import os
import sys
import time
from importlib.metadata import version

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (ROOT, os.path.join(ROOT, "stage3"), os.path.join(ROOT, "stage4")):
    if path not in sys.path:
        sys.path.insert(0, path)

from logits_processor import (RuleDecodingError, RuleLogitsProcessor,  # noqa: E402
                              RuleTokenIndex)
from nemo_rule import (COMPARISONS, atom_signature, body_has_aggregate,  # noqa: E402
                       rule_parts, split_top_level)
from rule_constraint import RuleConstraint, build_trie  # noqa: E402
from rule_harness import RuleBench  # noqa: E402
from tasks import SCHEMA, TASKS, compute_gold  # noqa: E402

ABLATIONS = ("unconstrained", "grammar_distractors", "full")
OUTCOMES = ("generation_error", "invalid", "engine_error", "valid_wrong", "correct")
DISTRACTORS = {"parent": 2, "grandparent": 2, "shareholder": 2,
               "holds": 3, "has_role": 3, "buys": 3}
SIGNATURES = {
    "OWN": "owner, company, stake_percent",
    "ROLE": "person, company, role",
    "REACHABLE": "entity, other_entity",
    "QUALIFIED_HOLDINGS": "owner, company",
    "INFLUENCE": "entity, company",
    "CONNECTED": "entity, other_entity",
    "CONTROL": "controller, company",
    "ULTIMATE_CONTROL": "ultimate_controller, company",
    "INDIRECT_ROLE": "person, company",
    "top_stake": "company, max_stake_percent",
}
DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16,
          "float32": torch.float32, "auto": "auto"}

PROMPT_TMPL = (
    "Write exactly one Nemo Datalog rule for the requested relation. "
    "Output only the rule with a final dot.\n\n"
    "The available company-ownership relations are listed in argument order.\n{schema}\n\n"
    "Variables start with ? (for example, ?x). Put aggregates #count, #min, "
    "#max, and #sum in the head. For #sum(?p, ?y), ?p is the value to sum "
    "and ?y distinguishes equal values from different records.\n"
    "The following example uses a separate salary relation.\n"
    "department_payroll(?x, #sum(?p, ?y)) :- salary(?y, ?x, ?p) .\n\n"
    "{instruction}"
)


def task_schema(task):
    schema = dict(SCHEMA)
    for rule in task["context"]:
        head, arity, _ = rule_parts(rule)
        schema[head] = arity
    return schema


def make_prompt(task):
    schema = "\n".join(f"{pred}({SIGNATURES[pred]})" for pred in sorted(task_schema(task)))
    return PROMPT_TMPL.format(schema=schema, instruction=task["instruction"])


def load_model(model_id, dtype="bfloat16", device=None, trust_remote_code=True,
               device_map=None):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    dtype_arg = "dtype" if int(version("transformers").split(".", 1)[0]) >= 5 else "torch_dtype"
    kwargs = {dtype_arg: DTYPES.get(dtype, dtype), "trust_remote_code": trust_remote_code}
    if device_map:
        kwargs["device_map"] = device_map
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except ValueError:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    if device_map:
        return model.eval(), tok, model.device.type
    if device is None:
        device = ("mps" if torch.backends.mps.is_available()
                  else "cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device).eval(), tok, device


def build_constraint(task, ablation, entity_trie):
    if ablation == "unconstrained":
        return None
    schema = task_schema(task)
    if ablation == "grammar_distractors":
        schema.update(DISTRACTORS)
    elif ablation != "full":
        raise ValueError(f"unknown ablation {ablation!r}")
    return RuleConstraint(task["head"], 2, schema, entity_trie=entity_trie)


def prompt_ids(tok, prompt):
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            enable_thinking=False,
        )["input_ids"]
    return tok.encode(prompt, return_tensors="pt")


def generate(model, tok, prompt, constraint=None, token_index=None,
             max_new_tokens=160, do_sample=True, temperature=0.7,
             top_p=0.8, top_k=20, seed=42):
    torch.manual_seed(seed)
    ids = prompt_ids(tok, prompt).to(model.device)
    generation_start = ids.shape[-1]
    processor = None
    if constraint is not None:
        if token_index is None:
            token_index = RuleTokenIndex(tok)
        processor = RuleLogitsProcessor(token_index, constraint, generation_start)
    sampling = (dict(temperature=temperature, top_p=top_p, top_k=top_k) if do_sample
                else dict(temperature=1.0, top_p=1.0, top_k=50))
    t0 = time.perf_counter()
    error = ""
    try:
        with torch.inference_mode():
            out = model.generate(
                ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                num_beams=1,
                num_return_sequences=1,
                return_dict_in_generate=False,
                logits_processor=[processor] if processor is not None else [],
                pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
                **sampling,
            )
        new = out[0, generation_start:].tolist()
        n_tokens = len(new)
        eos = bool(new) and new[-1] == tok.eos_token_id
        text = tok.decode(new[:-1] if eos else new, clean_up_tokenization_spaces=False)
        termination = "eos" if eos else "token_limit" if n_tokens == max_new_tokens else "stopped"
    except RuleDecodingError as failure:
        text, n_tokens = failure.prefix, failure.n_tokens
        termination, error = "mask_error", str(failure)

    if constraint is not None and not error:
        state = constraint.consume(constraint.initial, text)
        if not constraint.accepting(state):
            error = f"incomplete constrained rule ({termination})"
        elif processor.mask_steps != n_tokens:
            error = f"mask applied at {processor.mask_steps} of {n_tokens} generation steps"
    return {
        "raw_text": text, "n_tokens": n_tokens, "termination": termination,
        "gen_s": time.perf_counter() - t0, "generation_error": error,
        "mask_steps": processor.mask_steps if processor is not None else 0,
        "mask_s": processor.mask_s if processor is not None else 0.0,
    }


def validate_candidate(head, arity, body, task):
    if (head, arity) != (task["head"], 2):
        raise ValueError(f"expected {task['head']}/2, got {head}/{arity}")
    schema = task_schema(task)
    for literal in body:
        if body_has_aggregate(literal):
            raise ValueError("aggregates must appear in the head")
        parts = split_top_level(literal, COMPARISONS)
        if len(parts) == 2 and all(parts):
            continue
        pred, nargs = atom_signature(literal.removeprefix("~").lstrip())
        if pred not in schema:
            raise ValueError(f"unknown body predicate {pred!r}")
        if schema[pred] != nargs:
            raise ValueError(f"body predicate {pred!r} expects {schema[pred]} arguments, got {nargs}")


class Eval:
    def __init__(self):
        triples = pd.read_csv(f"{ROOT}/data/triples.csv", keep_default_na=False,
                              low_memory=False)
        self.bench = RuleBench(triples)
        self.golds = compute_gold(triples, write=False)
        self._entity_trie = None
        self._token_index = None

    def decoding_inputs(self, tok, task, ablation):
        if ablation == "unconstrained":
            return None, None
        if self._entity_trie is None:
            entities = pd.read_csv(f"{ROOT}/data/entities.csv", keep_default_na=False)
            self._entity_trie = build_trie(entities.entity)
        if self._token_index is None or self._token_index.tokenizer is not tok:
            self._token_index = RuleTokenIndex(tok)
        return build_constraint(task, ablation, self._entity_trie), self._token_index

    def score(self, text, task):
        rule = text.strip()
        try:
            head, arity, body = rule_parts(rule)
        except ValueError as error:
            return {"outcome": "invalid", "failure_stage": "extraction", "error": str(error)}
        try:
            validate_candidate(head, arity, body, task)
        except ValueError as error:
            return {"outcome": "invalid", "failure_stage": "validation",
                    "error": str(error), "rule_text": rule}
        result = self.bench.check(rule, head, task["context"], self.golds[task["id"]])
        return dict(result, rule_text=rule)

    def run_one(self, model, tok, task, ablation, cfg, seed):
        t0 = time.perf_counter()
        constraint, token_index = self.decoding_inputs(tok, task, ablation)
        generation = generate(
            model, tok, make_prompt(task), constraint, token_index, seed=seed, **cfg)
        error = generation.pop("generation_error")
        if error:
            result = {"outcome": "generation_error", "failure_stage": "generation",
                      "error": error}
        else:
            result = self.score(generation["raw_text"], task)
        row = {"task": task["id"], "ablation": ablation, "seed": seed,
               **generation, **result, "wall_s": time.perf_counter() - t0}
        return row, constraint

    def preflight(self, tok, tasks, ablations):
        """Verify the chat boundary and every selected gold rule without model weights."""
        for task in tasks:
            gold = task["gold_rule"]
            result = self.score(gold, task)
            if result["outcome"] != "correct":
                raise RuntimeError(f"{task['id']}: gold rule failed: {result}")
            input_ids = prompt_ids(tok, make_prompt(task))
            gold_ids = tok.encode(gold, add_special_tokens=False)
            scores = torch.zeros(len(tok))
            for ablation in ablations:
                constraint, token_index = self.decoding_inputs(tok, task, ablation)
                if constraint is None:
                    continue
                ids = input_ids
                processor = RuleLogitsProcessor(token_index, constraint, ids.shape[-1])
                for token_id in gold_ids:
                    masked = processor(ids, scores)
                    if not torch.isfinite(masked[token_id]) or torch.isfinite(masked).all():
                        raise RuntimeError(f"{task['id']}/{ablation}: gold token is blocked or unmasked")
                    ids = torch.cat((ids, torch.tensor([[token_id]])), dim=-1)
                masked = processor(ids, scores)
                allowed = torch.isfinite(masked).nonzero(as_tuple=True)[0].tolist()
                if allowed != [tok.eos_token_id]:
                    raise RuntimeError(f"{task['id']}/{ablation}: expected EOS after the gold rule")
            print(f"{task['id']}: gold rows={result['rows']}, selected masks verified", flush=True)

    def run(self, model, tok, tasks=None, ablations=("full",), n_samples=4,
            seed=42, cfg=None, verbose=False):
        cfg = cfg or {}
        tasks = TASKS if tasks is None else tasks
        rows = []
        for task in tasks:
            for ablation in ablations:
                for i in range(n_samples):
                    row, _ = self.run_one(model, tok, task, ablation, cfg, seed + i)
                    rows.append(row)
                    if verbose:
                        print(f"[{task['id']}|{ablation}|{i}] {describe(row)}", flush=True)
        return pd.DataFrame(rows)

    @staticmethod
    def summary(df):
        by_arm = df.groupby("ablation")
        summary = by_arm.agg(
            n=("outcome", "size"), mean_tokens=("n_tokens", "mean"),
            mean_gen_s=("gen_s", "mean"), mean_mask_s=("mask_s", "mean"),
        )
        for outcome in OUTCOMES:
            summary[outcome] = by_arm.outcome.apply(lambda s: 100 * (s == outcome).mean())
        return summary[["n", *OUTCOMES, "mean_tokens", "mean_gen_s", "mean_mask_s"]].round(2)


def describe(row):
    error = row.get("error", "")
    detail = " ".join(error.split()) if error else row.get("rule_text", "")
    stage = f"/{row['failure_stage']}" if row.get("failure_stage") else ""
    return f"{row['outcome']}{stage} (mask steps={row['mask_steps']}) {detail[:240]}"
