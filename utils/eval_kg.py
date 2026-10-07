"""Evaluate ReFactX ``<kg>`` KnowledgeGraphGeneration on a whole dataset.

This adapts ``utils/eval.py`` to the in-memory, callback-backed setup used in
``notebooks/ReFactX-with-Rules/try_refactx_knowledge_graph.ipynb``: entities come
from the rule-backed ``api.entity_index()``, relations/objects from
``api.property_strings`` / ``api.get``, and generation is driven by the ``<kg>``
pattern with :class:`~refactx.generate.KnowledgeGraphGeneration` (no
Postgres/HTTP index).

``--dataset`` selects the questions to evaluate:

* ``rules``            -- the ReFactX-with-Rules benchmark (``api.QA``);
* ``path/to/file.csv`` -- any CSV, using ``--question-key`` / ``--answer-key``.

In both cases the knowledge graph is provided by ``--rules-dir``.

All options can also be given in a JSON config passed with ``--config``; explicit
command-line options override the config. See ``configs/config_kg_rules_qwen35_4b.json``.

The rule-backed knowledge graph (``api.py``) lives in the **parent**
``ReFactX-with-Rules`` repo, so ``--rules-dir`` defaults to ``..``.

Examples
--------
    # run from inside the ReFactX submodule
    python utils/eval_kg.py --config configs/config_kg_rules_qwen35_4b.json
    python utils/eval_kg.py --config configs/config_kg_rules_qwen35_4b.json --n 20 --debug
    python utils/eval_kg.py --dataset rules --rules-dir .. --n 10
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

# Allow running as `python utils/eval_kg.py` from the repo root: make the local
# package importable before importing refactx / utils.eval.
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import click
import pandas as pd
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)

import refactx
from refactx.index import DictIndex
from refactx.generate import (
    ConstrainedLogitsProcessor,
    ConstrainedStateList,
    KnowledgeGraphGeneration,
)

from utils.eval import calculate_metrics, get_utc_date_and_time, logrotate

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

METRIC_KEYS = ("precision", "recall", "f1", "correct", "dont_know")


def load_rule_backed(rules_dir: Path):
    """Import the rule-backed ``api`` module (chdir so its relative data paths work)."""
    rules_dir = rules_dir.resolve()
    if not (rules_dir / "api.py").exists():
        raise FileNotFoundError(f"api.py not found in {rules_dir}")
    os.chdir(rules_dir)
    sys.path.insert(0, str(rules_dir))
    return importlib.import_module("api")


def reference_list(answer, answer_type=None):
    """Normalise a ground-truth answer into a list of strings.

    List answers are stored as ``"; ".join(values)``; other answers are scalars
    (``yes``/``no``/count/entity).
    """
    if answer_type == "list" or (isinstance(answer, str) and "; " in answer):
        return [part.strip() for part in str(answer).split(";") if part.strip()]
    return [answer]


def aggregate(rows):
    out = {"n": len(rows)}
    for key in METRIC_KEYS:
        out[key] = sum(r[key] for r in rows) / len(rows) if rows else 0.0
    out["answered"] = sum(r["answered"] for r in rows) / len(rows) if rows else 0.0
    return out


def merge_config(ctx, params, cfg):
    """Return effective options: values explicitly passed on the CLI win over config."""
    merged = {}
    for name, cli_value in params.items():
        source = ctx.get_parameter_source(name)
        if source == click.core.ParameterSource.DEFAULT:
            merged[name] = cfg.get(name, cli_value)
        else:
            merged[name] = cli_value
    return merged


@click.command()
@click.option("--config", "config_path", type=click.Path(exists=True), default=None,
              help="JSON config file; explicit CLI options override it.")
@click.option("--dataset", default="rules", show_default=True,
              help="'rules' for the ReFactX-with-Rules benchmark, or a CSV path.")
@click.option("--rules-dir", default="..", show_default=True,
              help="Directory containing the rule-backed api.py (default: the parent "
                   "ReFactX-with-Rules repo).")
@click.option("--experiment-name", default=None, help="Name used for the default output file.")
@click.option("--model-name", default="Qwen/Qwen3.5-4B", show_default=True)
@click.option("--model-dtype", type=click.Choice(list(DTYPES)), default="bfloat16", show_default=True)
@click.option("--device", default="auto", show_default=True)
@click.option("--prompt", default="prompts/prompt_qwen36_angular2_kg_nothink.yaml",
              show_default=True, help="Prompt file, relative to the repo root.")
@click.option("--question-key", default="question", show_default=True)
@click.option("--answer-key", default="answer", show_default=True)
@click.option("--answer-type-key", default="answer_type", show_default=True)
@click.option("--kind-key", default="kind", show_default=True,
              help="Column used for the per-kind breakdown (empty to disable).")
@click.option("--output", default=None, help="Output JSONL file (default: logs/<experiment>.out).")
@click.option("--log-dir", default="logs", show_default=True)
@click.option("--n", type=int, default=None, help="Evaluate only the first n questions.")
@click.option("--max-new-tokens", type=int, default=512, show_default=True)
@click.option("--thinking", is_flag=True, default=False, help="Enable the model's thinking mode.")
@click.option("--long-chains", is_flag=True, default=False,
              help="Let the object of a hop become the subject of the next.")
@click.option("--avoid-duplicates", is_flag=True, default=True,
              help="Do not let <kg> regenerate an already produced triple.")
@click.option("--do-sample", is_flag=True, default=False, help="Sample instead of greedy decoding.")
@click.option("--temperature", type=float, default=0.7, show_default=True)
@click.option("--top-p", type=float, default=0.80, show_default=True)
@click.option("--top-k", type=int, default=20, show_default=True)
@click.option("--batch-size", type=int, default=1, show_default=True)
@click.option("--seed", type=int, default=None)
@click.option("--continue-run", is_flag=True, default=False, help="Resume an interrupted run.")
@click.option("--no-flush", is_flag=True, default=False, help="Do not flush the output after each batch.")
@click.option("--debug", is_flag=True, default=False, help="Print each prediction as it is produced.")
@click.pass_context
def main(ctx, config_path, **params):
    cfg = {}
    if config_path:
        with open(config_path) as fd:
            cfg = json.load(fd)
        print("Loaded config:", config_path)
    opt = merge_config(ctx, params, cfg)

    dataset = opt["dataset"]
    rules_dir = opt["rules_dir"]
    model_name = opt["model_name"]
    model_dtype = opt["model_dtype"]
    device = opt["device"]
    prompt = opt["prompt"]
    question_key = opt["question_key"]
    answer_key = opt["answer_key"]
    answer_type_key = opt["answer_type_key"]
    kind_key = opt["kind_key"]
    experiment_name = opt["experiment_name"]
    output = opt["output"]
    log_dir = opt["log_dir"]
    n = opt["n"]
    max_new_tokens = opt["max_new_tokens"]
    thinking = opt["thinking"]
    long_chains = opt["long_chains"]
    avoid_duplicates = opt["avoid_duplicates"]
    do_sample = opt["do_sample"]
    temperature = opt["temperature"]
    top_p = opt["top_p"]
    top_k = opt["top_k"]
    batch_size = opt["batch_size"]
    seed = opt["seed"]
    continue_run = opt["continue_run"]
    flush = not opt["no_flush"]
    debug = opt["debug"]

    torch_dtype = DTYPES[model_dtype]
    if seed is not None:
        torch.manual_seed(seed)

    # Resolve repo-relative paths before the rule-backed api chdirs into --rules-dir.
    prompt_path = Path(prompt)
    if not prompt_path.is_absolute():
        prompt_path = REPO / prompt_path
    assert prompt_path.is_file(), f"prompt not found: {prompt_path}"

    dataset_is_file = dataset != "rules"
    dataset_path = Path(dataset)
    if dataset_is_file and not dataset_path.is_absolute():
        dataset_path = (REPO / dataset_path).resolve()

    print("Loading rule-backed knowledge graph...")
    api = load_rule_backed(Path(rules_dir))
    print("  loaded api from", rules_dir)

    if dataset_is_file:
        df = pd.read_csv(dataset_path, keep_default_na=False)
        print(f"  dataset: {dataset_path} ({len(df)} questions)")
    else:
        df = pd.read_csv(api.QA, keep_default_na=False)
        print(f"  dataset: {api.QA} ({len(df)} questions)")
    df = df.reset_index(drop=True)

    if n is not None:
        df = df.head(n).reset_index(drop=True)
        print(f"  limited to first {len(df)} questions")

    experiment = experiment_name or f"{Path(dataset).name}.{Path(model_name).name}.{prompt_path.stem}"
    output = Path(output) if output else (Path(log_dir) / f"{experiment}.out")
    if not output.is_absolute():
        output = REPO / output
    output = str(output)
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading model {model_name} ({model_dtype}, device={device})...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    except Exception:
        tokenizer = AutoProcessor.from_pretrained(model_name, padding_side="left").tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {"dtype": torch_dtype}
    if device == "auto":
        load_kwargs["device_map"] = "auto"
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    except Exception:
        model = AutoModelForImageTextToText.from_pretrained(model_name, **load_kwargs)
    if device != "auto":
        model = model.to(device)
    model.eval()

    PROMPT_TEMPLATE = refactx.load_prompt(str(prompt_path))
    print("  prompt:", prompt_path)

    print("Building the KG index from api.entity_index()...")
    index = DictIndex()
    index.set_tokenizer(tokenizer)
    entities = api.entity_index()
    for entity in entities:
        index.add(tokenizer.encode(f" <{entity}>", add_special_tokens=False))
    print(f"  {len(entities)} entities")

    logits_processor = ConstrainedLogitsProcessor(
        states=ConstrainedStateList("auto", num_beams=1, num_batches=batch_size,
                                    debug_tokenizer=tokenizer),
        tokenizer=tokenizer,
    )
    logits_processor.add_pattern(
        "<kg>", KnowledgeGraphGeneration,
        index=index,
        get_relations=api.property_strings,
        get_objects=api.get,
        long_chains=long_chains,
        avoid_duplicates=avoid_duplicates,
        eot=" </kg>\n",
    )

    generation_config = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        num_beams=1,
        num_return_sequences=1,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        generation_config.update(temperature=temperature, top_p=top_p, top_k=top_k)

    metadata = {
        "experiment_name": experiment,
        "dataset": dataset if not dataset_is_file else str(dataset_path),
        "rules_dir": str(rules_dir),
        "model_name": model_name,
        "prompt": str(prompt_path),
        "long_chains": long_chains,
        "avoid_duplicates": avoid_duplicates,
        "thinking": thinking,
        "do_sample": do_sample,
        "max_new_tokens": max_new_tokens,
        "n_questions": len(df),
        "date": get_utc_date_and_time(),
    }

    if continue_run:
        output, start_from = logrotate(output, len(df), metadata)
    else:
        output, start_from = logrotate(output)
    print("Output:", output, "(resume from %d)" % start_from if start_from else "")

    mode = "a" if start_from else "w"

    with open(output, mode) as out_fd:
        if start_from == 0:
            out_fd.write(json.dumps(metadata) + "\n")

        for batch_start in tqdm(range(0, len(df), batch_size), desc="eval"):
            rows = df.iloc[batch_start:batch_start + batch_size]
            questions = rows[question_key].tolist()
            kinds = (rows[kind_key].tolist() if kind_key else [None] * len(questions))
            references = [
                reference_list(row[answer_key],
                               row.get(answer_type_key) if answer_type_key else None)
                for _, row in rows.iterrows()
            ]

            if batch_start + len(questions) <= start_from:
                continue  # already done

            prompted = [
                refactx.apply_prompt_template(
                    tokenizer, question=q, enable_thinking=thinking,
                    prompt_template=PROMPT_TEMPLATE)
                for q in questions
            ]
            if batch_size > 1:
                tokenizer.padding_side = "left"
            batch_inputs = tokenizer(
                prompted, return_tensors="pt", padding=(batch_size > 1)).to(model.device)

            logits_processor.reset_states(num_beams=1, num_batches=len(questions))
            with torch.no_grad():
                outputs = model.generate(
                    **batch_inputs, logits_processor=[logits_processor], **generation_config)

            start_idx = batch_inputs.input_ids.shape[1]
            for i, (question, reference, kind) in enumerate(zip(questions, references, kinds)):
                if batch_start + i < start_from:
                    continue
                gen_ids = outputs[i][start_idx:].tolist()
                end = len(gen_ids)
                for j, tok in enumerate(gen_ids):
                    if tok in (tokenizer.pad_token_id, tokenizer.eos_token_id):
                        end = j
                        break
                full_prediction = tokenizer.decode(gen_ids[:end])
                prediction = refactx.get_answer(full_prediction)

                input_sample = {answer_key: reference, question_key: question, kind_key: kind}
                precision, recall, f1, correct, dont_know = calculate_metrics(
                    prediction, input_sample, answer_key=answer_key)

                state = logits_processor.states[i, 0]
                triples = list(map(tokenizer.decode, state.generated_triples))

                metrics = dict(zip(METRIC_KEYS, (precision, recall, f1, correct, dont_know)))
                metrics["answered"] = int(bool(prediction))

                record = dict(
                    index=batch_start + i,
                    question=question,
                    gt_answer=reference,
                    prediction=prediction,
                    full_prediction=full_prediction,
                    triples=triples,
                    kind=kind,
                    evaluation={k: metrics[k] for k in METRIC_KEYS},
                    answered=metrics["answered"],
                )
                out_fd.write(json.dumps(record) + "\n")
                if debug:
                    print(f"[{batch_start + i}] {question}\n   -> {prediction} (gt={reference})")

            if flush:
                out_fd.flush()

        # Recompute the summary from the whole file so resumed runs report the
        # totals over every evaluated question, not just this run's samples.
        out_fd.flush()
        overall = []
        per_kind = {}
        with open(output) as in_fd:
            for line in in_fd:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                evaluation = record.get("evaluation")
                if not isinstance(evaluation, dict):
                    continue
                metrics = {k: evaluation.get(k, 0) for k in METRIC_KEYS}
                metrics["answered"] = int(record.get("answered", 0))
                overall.append(metrics)
                per_kind.setdefault(record.get("kind"), []).append(metrics)

        summary = {
            "macro": aggregate(overall),
            "by_kind": {k: aggregate(v) for k, v in sorted(per_kind.items(), key=lambda kv: str(kv[0]))},
            "n": len(overall),
            "date": get_utc_date_and_time(),
        }
        out_fd.write(json.dumps(summary) + "\n")

    print("\n=== Macro ===")
    print(f"  n            {summary['macro']['n']}")
    for k, v in summary["macro"].items():
        if k == "n":
            continue
        print(f"  {k:12s} {v:.4f}")
    print("=== By kind ===")
    for kind, m in summary["by_kind"].items():
        print(f"  {str(kind):10s} n={m['n']:4d} f1={m['f1']:.3f} "
              f"acc={m['correct']:.3f} dk={m['dont_know']:.3f} ans={m['answered']:.3f}")
    print("Summary written to", output)


if __name__ == "__main__":
    main()
