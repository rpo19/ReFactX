"""Sample dataset question ids for re-evaluation, weighted by inverse per-type accuracy.

Queries that were harder (lower per-type accuracy) are drawn more often, so
that re-running them yields more chances to produce a correct answer. The
output is a flat JSON list of dataset ids, ready to be consumed by an eval
pipeline. A question may appear multiple times (each occurrence is one re-run).

By default the pool is the whole dataset; with ``--failed-only`` the pool is
restricted to ids whose previous answer was marked incorrect/unsuccessful.

Sampling formula
----------------
Per-type accuracy  acc_t = correct_t / total_t            (computed from the eval log)
Per-type weight    w_t   = 1 / max(acc_t, min_accuracy)   (inverse accuracy)
Per-type budget    n_t   = round(N * w_t / sum_t' w_t')   (largest-remainder: sum n_t == N)

So a type that is twice as accurate gets half the samples of a type that is
half as accurate. Within a type, ids are drawn uniformly with replacement.

Example
-------
If ``comparative`` has acc 0.44 and ``difference`` has acc 0.02, w_comp=2.3 and
w_diff=50, so difference questions get ~20x more samples than comparative ones.
"""

import json
import random
from collections import defaultdict

import click

EPSILON = 1e-6


def get_field(sample, key):
    value = sample
    for part in key.split('.'):
        value = value[part]
    return value


def load_eval_samples(eval_log, type_key, id_key, correct_key):
    samples = []
    with open(eval_log) as fd:
        for line in fd:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if 'input_sample' not in obj or 'evaluation' not in obj:
                continue
            samples.append({
                'id': obj['input_sample'].get(id_key),
                'type': str(obj['input_sample'].get(type_key, 'unknown')),
                'correct': int(get_field(obj, correct_key) == 1),
            })
    return samples


def largest_remainder_allocation(budgets, total_budget):
    total = sum(budgets.values())
    if total <= 0:
        return {t: 0 for t in budgets}
    floor = {t: int(w * total_budget // total) for t, w in budgets.items()}
    remainder = {t: (w * total_budget % total) / total for t, w in budgets.items()}
    remaining = total_budget - sum(floor.values())
    for t in sorted(remainder, key=remainder.get, reverse=True)[:remaining]:
        floor[t] += 1
    return floor


def sample_ids(ids_by_type, total_budget, seed, weights):
    random.seed(seed)
    unknown_types = set(ids_by_type) - set(weights)
    if unknown_types:
        print('WARNING: no accuracy in eval log for types: '
              f'{", ".join(sorted(unknown_types))}. Using fallback weight.')

    budgets = largest_remainder_allocation(
        {t: weights.get(t, 1.0) for t in ids_by_type}, total_budget,
    )
    selected = []
    for t, ids in ids_by_type.items():
        selected.extend(random.choices(ids, k=budgets[t]))
    return selected, budgets


@click.command()
@click.option('--dataset', default=None,
              help='HF dataset path, e.g. rpozzi/mintaka. Not needed with --failed-only.')
@click.option('--split', default='train', help='Dataset split to sample from. Default: train.')
@click.option('--dataset-config', default=None, help='Dataset config/subset name (default: None).')
@click.option('--eval-log', 'eval_log', required=True, type=click.Path(exists=True),
              help='Eval output log (jsonl) used to compute per-type accuracies.')
@click.option('--type-key', 'type_key', required=True,
              help='Key holding the question type in the dataset rows / input_sample, '
                   'e.g. complexityType.')
@click.option('--id-key', 'id_key', default='id', help='Key holding the unique question id. Default: id.')
@click.option('--failed-only', 'failed_only', is_flag=True,
              help='Sample only ids whose previous answer was incorrect (from the eval log).')
@click.option('--correct-key', 'correct_key', default='evaluation.correct',
              help='Dotted path to the correctness flag inside each eval-log sample. '
                   'Default: evaluation.correct.')
@click.option('--n', default=10000, type=int,
              help='Total number of ids to sample. Default: 10000.')
@click.option('--seed', default=42, type=int, help='Random seed for reproducibility. Default: 42.')
@click.option('--min-accuracy', 'min_accuracy', default=EPSILON, type=float,
              help='Floor applied to acc_t before inverting to avoid division by zero.')
@click.option('--output', 'output_path', required=True, type=click.Path(),
              help='Output JSON file containing the flat list of sampled ids.')
def main(dataset, split, dataset_config, eval_log, type_key, id_key,
         failed_only, correct_key, n, seed, min_accuracy, output_path):
    samples = load_eval_samples(eval_log, type_key, id_key, correct_key)
    if not samples:
        raise click.ClickException(f'no samples found in eval log: {eval_log}')

    accuracies = defaultdict(lambda: [0, 0])
    for s in samples:
        accuracies[s['type']][0] += s['correct']
        accuracies[s['type']][1] += 1
    accuracies = {t: correct / total for t, (correct, total) in accuracies.items()}
    weights = {t: 1.0 / max(acc, min_accuracy) for t, acc in accuracies.items()}

    if failed_only:
        ids_by_type = defaultdict(list)
        for s in samples:
            if s['correct'] == 0:
                ids_by_type[s['type']].append(s['id'])
    else:
        if dataset is None:
            raise click.ClickException('--dataset is required unless --failed-only is used.')
        from datasets import load_dataset
        ds = load_dataset(dataset, dataset_config, split=split)
        ids_by_type = defaultdict(list)
        for row in ds:
            ids_by_type[str(row.get(type_key, 'unknown'))].append(row[id_key])

    selected, budgets = sample_ids(ids_by_type, n, seed, weights)

    if len(selected) != n:
        raise click.ClickException(
            f'planned {n} samples but only {len(selected)} ids could be drawn '
            '(fewer ids available than requested budget).')

    with open(output_path, 'w') as fd:
        json.dump(selected, fd, indent=2)

    print(f'Per-type accuracy (from {eval_log}):')
    for t in sorted(accuracies):
        print(f'  {t:<16} acc={accuracies[t]:.4f}  budget={budgets.get(t, 0)}')
    print(f'Wrote {n} ids (duplicates allowed, unique={len(set(selected))}) to {output_path}')


if __name__ == '__main__':
    main()