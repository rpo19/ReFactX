"""Run the Stage 5 rule-generation benchmark or its tokenizer-only preflight."""
import argparse
import csv
import os
import time
from importlib.metadata import PackageNotFoundError, version

import pandas as pd
from transformers import AutoTokenizer

from eval_rules import ABLATIONS, Eval, ROOT, TASKS, describe, load_model


FIELDS = (
    "task", "ablation", "seed", "outcome", "failure_stage", "error", "rule_text",
    "raw_text", "termination", "n_tokens", "mask_steps", "mask_s", "gen_s",
    "nemo_s", "wall_s", "rows", "missing", "extra",
)


def nemo_version():
    try:
        return version("nemo-python")
    except PackageNotFoundError:
        try:
            return version("nmo-python")
        except PackageNotFoundError:
            return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    run_mode = ap.add_mutually_exclusive_group()
    run_mode.add_argument("--preflight", action="store_true",
                          help="check gold rules and masks using only the tokenizer")
    run_mode.add_argument("--quick-test", action="store_true",
                          help="run a one-task smoke test with Qwen2.5-0.5B")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=("bfloat16", "float16", "float32", "auto"))
    ap.add_argument("--device", default=None, help="cuda/mps/cpu (default auto)")
    ap.add_argument("--device-map", default=None,
                    help="'auto' to split the model across visible GPUs")
    ap.add_argument("--tasks", default=None, help="comma-separated task ids (default: all)")
    ap.add_argument("--ablations", default=",".join(ABLATIONS),
                    help="comma-separated arms. 'syntax' aliases 'grammar_distractors'")
    ap.add_argument("--n-samples", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--no-sample", action="store_true", help="use greedy decoding")
    ap.add_argument("--check-masks", action="store_true",
                    help="exit with an error if a constrained generation is incomplete")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "stage5", "results"))
    args = ap.parse_args()

    if args.quick_test:
        args.model = "Qwen/Qwen2.5-0.5B-Instruct"
        args.tasks = "owner_of_owner"
        args.n_samples = 1
        args.no_sample = True
    task_ids = ({t.strip() for t in args.tasks.split(",")} if args.tasks is not None
                else {t["id"] for t in TASKS})
    tasks = [t for t in TASKS if t["id"] in task_ids]
    unknown_tasks = task_ids - {t["id"] for t in tasks}
    if unknown_tasks:
        ap.error(f"unknown tasks: {', '.join(sorted(unknown_tasks))}")
    ablations = tuple("grammar_distractors" if a.strip() == "syntax" else a.strip()
                      for a in args.ablations.split(","))
    if set(ablations) - set(ABLATIONS) or len(set(ablations)) != len(ablations):
        ap.error(f"choose distinct ablations from {', '.join(ABLATIONS)}")
    if args.n_samples < 1 or args.max_new_tokens < 1:
        ap.error("--n-samples and --max-new-tokens must be positive")

    print(f"model={args.model} dtype={args.dtype} device={args.device or 'auto'}")
    print(f"tasks={[t['id'] for t in tasks]} ablations={ablations} "
          f"samples={args.n_samples} seed={args.seed} sample={not args.no_sample}")
    print(f"Nemo={nemo_version()} torch={version('torch')} transformers={version('transformers')}",
          flush=True)

    ev = Eval()
    if args.preflight:
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        ev.preflight(tok, tasks, ablations)
        print("preflight passed", flush=True)
        return

    model, tok, device = load_model(args.model, args.dtype, args.device,
                                    device_map=args.device_map)
    print(f"loaded {args.model} on {device} | vocab={len(tok)}", flush=True)
    gen_cfg = dict(max_new_tokens=args.max_new_tokens, do_sample=not args.no_sample,
                   temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, time.strftime("rules_%Y%m%d_%H%M%S") + f"_{os.getpid()}.csv")
    print(f"writing samples to {out}", flush=True)
    rows, t0 = [], time.perf_counter()
    with open(out, "x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        for task in tasks:
            for ablation in ablations:
                for i in range(args.n_samples):
                    row, _ = ev.run_one(model, tok, task, ablation, gen_cfg, args.seed + i)
                    writer.writerow(row)
                    file.flush()
                    rows.append(row)
                    print(f"[{task['id']}|{ablation}|{i}] {describe(row)}", flush=True)

    results = pd.DataFrame(rows)
    print("\n===== summary (% of samples) =====")
    print(ev.summary(results).to_string())
    print("\n===== % correct per task × ablation =====")
    pivot = results.pivot_table(index="task", columns="ablation", values="outcome",
                                aggfunc=lambda s: 100 * (s == "correct").mean()).round(1)
    print(pivot.to_string())
    print(f"\nsaved {out} | total {(time.perf_counter() - t0) / 60:.1f} min")
    if args.check_masks and any(row["outcome"] == "generation_error"
                                and row["ablation"] != "unconstrained" for row in rows):
        raise SystemExit("smoke test failed: at least one constrained rule did not complete")


if __name__ == "__main__":
    main()
