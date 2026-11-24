# Handoff: ReFactX 0.8B retrieval work (from the local 2xT4 box)

Read this first. It records what was done, what the numbers mean, and what to run next.

## Branch and code

- Branch: `feat/sentinel-count-branches` (local merge commit `3bdf011` merges
  `feat/constrained-decoding-efficiency` into it). Not pushed yet at the time
  of writing; check `git log -1` on the HPC.
- Key files (all tracked):
  - `SFT_REASONING_AUDIT.md`: audit of the reasoning-SFT eval and all results below.
  - `utils/build_reasoning_traces_v2.py`: fixed reasoning-template builder (defaults to `logs/reasoning_templates_v3`).
  - `utils/kqa_gold_traces.py`: gold-KoPL trace generator for KQA Pro (simple chains only, about 2.5% of train).
  - `scripts/run_reason3_ablation.sh`, `scripts/run_kqa_gold.sh`: experiment launchers. They hardcode `/workspace/notebooks/refactx_clean` and `venv/bin/activate`; edit both for the HPC.
  - `configs/eval100_{kqa,2wiki,webqsp,cwq}.json`: 100-question eval configs.

## Results (100 questions, greedy, same IDs; 2 GPUs)

| Model | KQA | 2Wiki | WebQSP | CWQ |
|---|---:|---:|---:|---:|
| combined ckpt-100 (parent, no extra SFT) | - | 49 | - | 16 |
| clean ckpt-200 | - | - | 16 | - |
| mask-object ckpt-100 (KQA baseline) | 43 | - | - | - |
| plain control (parent + 100 steps, no narration) | 40 | 43 | 15 | 15 |
| no-think reasoning (fixed templates) | 32 | 22 | 7 | 10 |
| gold-trace KQA adapter (2,332 traces) | 34 | - | - | - |

Takeaways:
- Narration in the target hurts on every dataset. Don't use it.
- Extra SFT from the parent does not clearly help. The plain control is at or below the parent.
- The gold-trace adapter overfits narrow chains (train loss 0.08) and scores below controls.
- At n=100 the standard error is about 5 points. Differences under about 8 points are not reliable.

## Retrieval is the bottleneck

Earlier error analysis (`SFT_ERROR_ANALYSIS.md`): most errors are retrieval_missing
(wrong entity or relation, or a missing next-hop object). Programmatic traces only
help if they cover the hard cases: multi-`Find` programs, `FindAll`, and filters.

## Next steps

1. Copy the adapters and data listed in `SFT_REASONING_AUDIT.md` and in the
   HPC copy list. Pass `-e "ssh -p PORT"` if the login node uses a non-default SSH port.
2. Build the venv with `bash hpc/setup_venv.sh`, then `source hpc/env.sh`.
3. Point the `/workspace/data/kqa_pro/...` paths in the configs at the HPC copy
   (`${WS_PATH}/...`), or symlink the directory.
4. Pick a direction before spending GPU time:
   - (a) extend `utils/kqa_gold_traces.py` to multi-`Find` and `FindAll`/filter programs, then retrain from the parent;
   - (b) a stronger teacher with sampling and filtering (needs API or the big model on HPC), then the same matched controls;
   - (c) evaluate the existing adapters on more questions before any new training.
5. Keep every adapter compared against a parent-matched control with the same steps.

## Notes

- `todo.local.md` is a personal scratch list. It is not part of the project.
- `backup/pre-sentinel-rebase` is a local backup branch and can be deleted.
- `.env` is not in git. Create it on the cluster with that cluster's Postgres/Redis URLs and key.
