from copy import deepcopy
import sys
import json
import importlib
import re
import click
import numpy as np
import nltk
from rouge_score import rouge_scorer
import pandas as pd
import os

nltk.download('wordnet')
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def pd_generator(evaluation, dataset, EM, IM, bleu1, bleu4, meteor, rougeL, final_answers, dontknow, group=None):
    for i in range(len(evaluation)):
        prediction = evaluation[i]
        target = dataset[i]

        assert prediction['question'] == target['question']

        if group is not None:
            group_name = dataset[i][group]
        else:
            group_name = None

        yield (target['question'],
            prediction['prediction'],
            dataset.get_answer(i),
            EM[i],
            IM[i],
            bleu1[i],
            bleu4[i],
            meteor[i],
            rougeL[i,2], # F1
            i in final_answers,
            i in dontknow,
            prediction['full_prediction'],
            prediction['full_sample'],
            prediction['triples'],
            target['answer'],
            group_name,
            group_name
            )

def get_evaldf(evaluation, dataset, exact_match, inclusion_match, bleu_1, bleu_4, meteor, rougeL, final_answers, dontknow, group = None):
    columns = ['Question', 'Prediction', 'Answer', 'EM', 'IM', 'BLEU1', 'B4', 'METEOR', 'ROUGEL', 'Answered', 'DontKnow', 'FULL prediction', 'FULL sample', 'Triples', 'AnswerBig']
    if group is not None:
        columns.append('Group')
        columns.append(group)
    data = pd_generator(evaluation, dataset, exact_match, inclusion_match, bleu_1, bleu_4, meteor, rougeL, final_answers, dontknow, group = group)
    evaldf = pd.DataFrame(data, columns = columns)
    return evaldf

answer_pattern = re.compile(r'Answer: (.*)\.?')
def get_prediction(full_prediction, split_pattern, remove_dot=True):
    prediction = ''

    full_prediction = re.split(split_pattern, full_prediction, 1)[0]
    if remove_dot and full_prediction.endswith('.'):
        full_prediction = full_prediction[:-len('.')]
    match = answer_pattern.search(full_prediction)
    if match:
        prediction = match.group(1)

    return prediction

def print_metrics(title, columns, values, do_print=True):
    if do_print:
        print(title)
    metrics_data = {
        'Metric': columns,
        'Value': values
    }
    metrics_df = pd.DataFrame(metrics_data)
    if do_print:
        print(metrics_df.to_markdown(index=False))
    return metrics_df

def get_answered(predictions, evaluation):
    # only the answered
    answered = set([i for i in predictions if evaluation[i].get('prediction', '')])
    dontknow = set([i for i in predictions if evaluation[i].get('prediction', '').startswith("I don't know")])
    final_answers = answered - dontknow

    # Create a metrics table
    metrics_columns = ['Num', 'Percentage Answered', 'Percentage Don\'t Know', 'Final Answers (Answered - Don\'t Know)',  'Num 0 Triples', 'Percentage 0 Triples', 'Percentage 0 Triples (Final Answers)']
    metrics_values = [
        len(evaluation),
        len(answered) / (len(evaluation) + sys.float_info.min),
        len(dontknow) / (len(evaluation) + sys.float_info.min),
        len(final_answers) / (len(evaluation) + sys.float_info.min),
        '{}/{}'.format(sum(map(lambda x: x['triples'] == [], evaluation)), len(evaluation)),
        sum(map(lambda x: x['triples'] == [], evaluation)) / (len(evaluation) + sys.float_info.min),
        sum(1 for i in final_answers if evaluation[i]['triples'] == []) / (len(final_answers) + sys.float_info.min),
    ]
    answered_metrics = print_metrics('Answered Metrics', metrics_columns, metrics_values)

    final_answers_idx = list(final_answers)

    return answered_metrics, answered, dontknow, final_answers, final_answers_idx

def get_exact_match(evaluation, dataset, idx=None):
    if idx is None:
        idx = range(len(evaluation))
    exact_match = np.zeros((len(idx),))
    j = 0
    for i in idx:
        if evaluation[i]['prediction'] == dataset.get_answer(i):
            exact_match[j] = 1
        j += 1
    return exact_match

def get_inclusion_match(evaluation, dataset, idx=None):
    if idx is None:
        idx = range(len(evaluation))
    inclusion_match = np.zeros((len(idx),))
    j = 0
    for i in idx:
        if dataset.get_answer(i) in evaluation[i]['prediction']:
            inclusion_match[j] = 1
        j += 1
    return inclusion_match

# ## BLEU, METEOR, ROUGE

def get_other_metrics(evaluation, dataset, name='other_metrics', idx=None, do_print=True):
    if idx is None:
        idx = range(len(evaluation))
    bleu_1 = np.zeros((len(idx),))
    bleu_4 = np.zeros((len(idx),))
    meteor = np.zeros((len(idx),))
    rougeL = np.zeros((len(idx), 3)) # precision, recall, f1
    j = 0
    for i in idx:
        reference = dataset.get_answer(i)
        reference_list = reference.split()
        hypothesis = evaluation[i]['prediction']
        hypothesis_list = hypothesis.split()
        bleu_1[j] = nltk.translate.bleu_score.sentence_bleu([reference_list], hypothesis_list, weights=(1,))
        bleu_4[j] = nltk.translate.bleu_score.sentence_bleu([reference_list], hypothesis_list, weights=(0.25, 0.25, 0.25, 0.25))
        meteor[j] = nltk.translate.meteor_score.meteor_score([reference_list], hypothesis_list)
        rouge_scores = scorer.score(reference, hypothesis)
        rougeL[j] = np.array(rouge_scores['rougeL'])

        j += 1

    other_df = print_metrics(name, ['BLEU1', 'BLEU4', 'METEOR', 'ROUGEL-P', 'ROUGEL-R', 'ROUGEL-F1'],
        [bleu_1.mean(), bleu_4.mean(), meteor.mean(), rougeL[:, 0].mean(), rougeL[:, 1].mean(), rougeL[:, 2].mean()], do_print=do_print)

    return other_df, bleu_1, bleu_4, meteor, rougeL

def grouped_analysis(evaluation, dataset, group, answered, dontknow, final_answers):
    print('--- Groups ---')
    complexityType = {}
    for i in range(len(evaluation)):
        if not dataset[i][group] in complexityType:
            complexityType[dataset[i][group]] = []
        complexityType[dataset[i][group]].append(i)

    complexity_stats = {k:len(v) / (len(evaluation) + sys.float_info.min) for k,v in complexityType.items()}

    complexity_df = pd.DataFrame(list(complexity_stats.items()), columns=['Complexity Type', 'Percentage'])
    print_metrics('Complexity Type Distribution', complexity_df['Complexity Type'], complexity_df['Percentage'])

    grouped_answered_metrics = pd.DataFrame(columns=['Num', 'Percentage Answered', 'Percentage Don\'t Know',
        'Final Answers (Answered - Don\'t Know)',  'Num 0 Triples',
        'Percentage 0 Triples', 'Percentage 0 Triples (Final Answers)',
        'Exact Match', 'Exact Match (Final Answer)',
        'Inclusion Match', 'Inclusion Match (Final Answer)',
        'BLEU1', 'BLEU4', 'METEOR', 'ROUGEL-P', 'ROUGEL-R', 'ROUGEL-F1',
        'BLEU1-Final', 'BLEU4-Final', 'METEOR-Final', 'ROUGEL-P-Final', 'ROUGEL-R-Final', 'ROUGEL-F1-Final',
        ])

    for g in complexityType.keys():
        g_set = set(complexityType[g])
        g_idx = list(set(g_set))
        answered_g = answered.intersection(g_set)
        dontknow_g = dontknow.intersection(g_set)
        final_answers_g = final_answers.intersection(g_set)
        final_answers_g_idx = list(final_answers_g)

        evaluation_g = []
        for i in g_idx:
            sample = deepcopy(evaluation[i])
            sample['original_idx'] = i
            evaluation_g.append(sample)

        # final_answers_g_idx_adapted = [i for i, sample in enumerate(evaluation_g) if sample['original_idx'] in final_answers_g]

        exact_match_g = get_exact_match(evaluation, dataset, idx=g_idx)
        exact_match_g_final_answers = get_exact_match(evaluation, dataset, idx=final_answers_g_idx)

        inclusion_match_g = get_inclusion_match(evaluation, dataset, idx=g_idx)
        inclusion_match_g_final_answers = get_inclusion_match(evaluation, dataset, idx=final_answers_g_idx)

        _, bleu_1, bleu_4, meteor, rougeL = get_other_metrics(evaluation, dataset, name=f'{g} -- ALL', do_print=False, idx=g_idx)
        _, bleu_1_f, bleu_4_f, meteor_f, rougeL_f = get_other_metrics(evaluation, dataset, name=f'{g} -- FINAL', do_print=False, idx=final_answers_g_idx)

        # Create a metrics table
        row = [
            len(g_idx),
            len(answered_g) / (len(g_idx) + sys.float_info.min),
            len(dontknow_g) / (len(g_idx) + sys.float_info.min),
            len(final_answers_g) / (len(g_idx) + sys.float_info.min),
            '{}/{}'.format(sum(map(lambda x: x['triples'] == [], evaluation_g)), len(g_idx)),
            sum(map(lambda x: x['triples'] == [], evaluation_g)) / (len(g_idx) + sys.float_info.min),
            sum(1 for i in final_answers_g if evaluation[i]['triples'] == []) / (len(final_answers_g) + sys.float_info.min),
            exact_match_g.mean(),
            exact_match_g_final_answers.mean(),
            inclusion_match_g.mean(),
            inclusion_match_g_final_answers.mean(),
            bleu_1.mean(), bleu_4.mean(), meteor.mean(), rougeL[:, 0].mean(), rougeL[:, 1].mean(), rougeL[:, 2].mean(),
            bleu_1_f.mean(), bleu_4_f.mean(), meteor_f.mean(), rougeL_f[:, 0].mean(), rougeL_f[:, 1].mean(), rougeL_f[:, 2].mean()
        ]

        grouped_answered_metrics.loc[g] = row

    return complexity_df, grouped_answered_metrics


@click.command()
@click.option('--dataset', 'dataset_path', required=False, default=None, help='Path to the dataset configuration file.')
@click.option('--infile', required=True, help='Path to the input file.')
@click.option('--outfile', required=False, default=None, help='Path to the output xlsx file (automatic if missing).')
@click.option('--fix-predictions', is_flag=True, default=False, help='Fix (missing) predictions in the evaluation.')
@click.option('--no-fix-none-prediction', is_flag=True, default=True, help='Do not replace None predictions with an empty string.')
@click.option('--split-pattern', required=False, default=r'<\|im_end\|>', help='Pattern to split the full prediction. Use with --fix-predictions.')
@click.option('--force', is_flag=True, default=False, help='Overwrite outfile if existing.')
@click.option('--group', required=False, default=None, help='Calculate grouped metrics (e.g. by question type).')
def main(dataset_path, infile, outfile, fix_predictions, no_fix_none_prediction, split_pattern, force, group):
    evaluation_raw = []
    with open(infile) as fd:
        line = fd.readline()
        while line:
            evaluation_raw.append(json.loads(line))
            line = fd.readline()

    header = evaluation_raw[0]

    if dataset_path is None:
        dataset_path = header.get('dataset_config', {}).get('config', {}).get('path')
        if dataset_path is None:
            raise Exception('No dataset path provided. Nor in the --infile nor as --dataset.')
    if dataset_path.endswith('.py'):
        dataset_path = dataset_path[:-3]
    dataset_name = os.path.basename(dataset_path)
    dataset_module = importlib.import_module(dataset_path)
    dataset = getattr(dataset_module, 'dataset')

    start_from = header.get('dataset_config', {}).get('config', {}).get('start_from', 0)
    if start_from > 0:
        print('Datasets starts from', start_from)
        dataset = dataset[start_from:]
    evaluation = evaluation_raw[1:]

    predictions = set(range(len(evaluation)))

    if fix_predictions:
        print('Split pattern:', split_pattern)

        for i in range(len(evaluation)):
            evaluation[i]['prediction'] = get_prediction(evaluation[i]['full_prediction'], split_pattern=split_pattern)

    if no_fix_none_prediction:
        for i in range(len(evaluation)):
            if evaluation[i]['prediction'] is None:
                evaluation[i]['prediction'] = ''

    answered_metrics, answered, dontknow, final_answers, final_answers_idx = get_answered(predictions, evaluation)

    for i in range(len(evaluation)):
        assert evaluation[i]['question'] == dataset[i]['question']

    # # Metrics

    # ## Exact match

    exact_match = get_exact_match(evaluation, dataset)

    inclusion_match = get_inclusion_match(evaluation, dataset)

    em_im_metrics = print_metrics(
        'Exact and Inclusion Match Metrics',
        [   'Num',
            'Exact Match (All)', 'Exact Match (Final Answers)',
            'Inclusion Match (All)', 'Inclusion Match (Final Answers)'
        ],
        [
            len(evaluation),
            exact_match.mean(), exact_match[final_answers_idx].mean(),
            inclusion_match.mean(), inclusion_match[final_answers_idx].mean()
        ]
    )

    other_metrics_all, bleu_1, bleu_4, meteor, rougeL = get_other_metrics(evaluation, dataset, name='ALL')
    bleu_1_f, bleu_4_f, meteor_f, rougeL_f = bleu_1[final_answers_idx], bleu_4[final_answers_idx], meteor[final_answers_idx], rougeL[final_answers_idx]
    other_metrics_answered = print_metrics('FINAL_ANSWERS', ['BLEU1', 'BLEU4', 'METEOR', 'ROUGEL-P', 'ROUGEL-R', 'ROUGEL-F1'],
        [bleu_1_f.mean(), bleu_4_f.mean(), meteor_f.mean(), rougeL_f[:, 0].mean(), rougeL_f[:, 1].mean(), rougeL_f[:, 2].mean()], do_print=True)

    # ## Group
    if group is not None:
        complexity_df, grouped_answered_metrics = grouped_analysis(evaluation, dataset, group, answered, dontknow, final_answers)

    # ## Excel

    evaldf = get_evaldf(evaluation, dataset, exact_match, inclusion_match, bleu_1, bleu_4, meteor, rougeL, final_answers, dontknow, group = group)

    if outfile:
        xlsx_file = outfile
    else:
        xlsx_file = f'{dataset_name}_{os.path.basename(infile)}.xlsx'
    if not force:
        assert not os.path.isfile(xlsx_file), f'Error: {xlsx_file} already exists'
    print(f'Writing {xlsx_file}')
    with pd.ExcelWriter(xlsx_file) as writer:
        evaldf.to_excel(writer, sheet_name="Per answer")
        answered_metrics.to_excel(writer, sheet_name="Answered metrics")
        em_im_metrics.to_excel(writer, sheet_name="EM IM metrics")
        other_metrics_all.to_excel(writer, sheet_name="Other metrics")
        other_metrics_answered.to_excel(writer, sheet_name="Other metrics (Only answered)")
        if group is not None:
            complexity_df.to_excel(writer, sheet_name="Complexity distribution")
            grouped_answered_metrics.to_excel(writer, sheet_name="Grouped metrics")
        metadata_gen = [('dataset_path', dataset_path), ('input_file', infile), ('fix_predictions', fix_predictions), ('no_fix_none_prediction', no_fix_none_prediction), ('split_pattern', split_pattern)]
        metadata_gen.extend(header.items())
        metadata_df = pd.DataFrame(metadata_gen, columns=['Key', 'Value'])
        metadata_df.to_excel(writer, sheet_name="Metadata", index=False)


if __name__ == '__main__':
    main()
