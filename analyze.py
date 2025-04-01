
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

def pd_generator(evaluation, dataset, EM, IM, bleu1, bleu4, meteor, rougeL, final_answers, dontknow):
    for i in range(len(evaluation)):
        prediction = evaluation[i]
        target = dataset[i]

        assert prediction['question'] == target['question']

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
            target['answer'])

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

def print_metrics(title, columns, values):
    print(title)
    metrics_data = {
        'Metric': columns,
        'Value': values
    }
    metrics_df = pd.DataFrame(metrics_data)
    print(metrics_df.to_markdown(index=False))

@click.command()
@click.option('--dataset', required=True, help='Path to the dataset configuration file.')
@click.option('--infile', required=True, help='Path to the input file.')
@click.option('--outfile', required=False, default=None, help='Path to the output xlsx file (automatic if missing).')
@click.option('--fix-predictions', is_flag=True, default=False, help='Fix (missing) predictions in the evaluation.')
@click.option('--no-fix-none-prediction', is_flag=True, default=True, help='Do not replace None predictions with an empty string.')
@click.option('--split-pattern', required=False, default=r'<\|im_end\|>', help='Pattern to split the full prediction. Use with --fix-predictions.')
@click.option('--force', is_flag=True, default=False, help='Overwrite outfile if existing.')
def main(dataset, infile, outfile, fix_predictions, no_fix_none_prediction, split_pattern, force):
    if dataset.endswith('.py'):
        dataset = dataset[:-3]
    dataset_name = os.path.basename(dataset)
    dataset_module = importlib.import_module(dataset)
    dataset = getattr(dataset_module, 'dataset')

    evaluation_raw = []
    with open(infile) as fd:
        line = fd.readline()
        while line:
            evaluation_raw.append(json.loads(line))
            line = fd.readline()

    header = evaluation_raw[0]
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

    # only the answered
    answered = set([i for i in predictions if evaluation[i].get('prediction', '')])
    dontknow = set([i for i in predictions if evaluation[i].get('prediction', '').startswith("I don't know")])
    final_answers = answered - dontknow

    # Create a metrics table
    metrics_columns = ['Percentage Answered', 'Percentage Don\'t Know', 'Final Answers (Answered - Don\'t Know)',  'Num 0 Triples', 'Percentage 0 Triples', 'Percentage 0 Triples (Final Answers)']
    metrics_values = [
        len(answered) / len(evaluation),
        len(dontknow) / len(evaluation),
        len(final_answers) / len(evaluation),
        '{}/{}'.format(sum(map(lambda x: x['triples'] == [], evaluation)), len(evaluation)),
        sum(map(lambda x: x['triples'] == [], evaluation)) / len(evaluation),
        sum(1 for i in final_answers if evaluation[i]['triples'] == []) / len(final_answers),
    ]
    print_metrics('Answered Metrics', metrics_columns, metrics_values)

    for i in final_answers:
        assert evaluation[i]['question'] == dataset[i]['question']

    # for mintaka
    try:
        complexityType = {}
        for i in final_answers:
            if not dataset[i]['complexityType'] in complexityType:
                complexityType[dataset[i]['complexityType']] = 0
            complexityType[dataset[i]['complexityType']] += 1 / len(final_answers)

        complexity_df = pd.DataFrame(list(complexityType.items()), columns=['Complexity Type', 'Percentage'])
        print_metrics('Complexity Type Distribution', complexity_df['Complexity Type'], complexity_df['Percentage'])
    except:
        print('Complexity type not found.')

    for i in dontknow:
        print(dataset[i]['question'], '--', evaluation[i]['prediction'], '?==', dataset.get_answer(i))
        break

    evaluation[1]

    # # Metrics

    final_answers_idx = list(final_answers)

    # ## Exact match

    exact_match = np.zeros((len(evaluation),))
    for i in final_answers:
        if evaluation[i]['prediction'] == dataset.get_answer(i):
            exact_match[i] = 1

    inclusion_match = np.zeros((len(evaluation),))
    for i in final_answers:
        if dataset.get_answer(i) in evaluation[i]['prediction']:
            inclusion_match[i] = 1

    print_metrics(
        'Exact and Inclusion Match Metrics',
        [
            'Exact Match (All)', 'Exact Match (Final Answers)',
            'Inclusion Match (All)', 'Inclusion Match (Final Answers)'
        ],
        [
            exact_match.mean(), exact_match[final_answers_idx].mean(),
            inclusion_match.mean(), inclusion_match[final_answers_idx].mean()
        ]
    )

    # ## BLEU, METEOR, ROUGE

    bleu_1 = np.zeros((len(evaluation),))
    bleu_4 = np.zeros((len(evaluation),))
    meteor = np.zeros((len(evaluation),))
    rougeL = np.zeros((len(evaluation), 3)) # precision, recall, f1
    for i in final_answers:
        reference = dataset.get_answer(i)
        reference_list = reference.split()
        hypothesis = evaluation[i]['prediction']
        hypothesis_list = hypothesis.split()
        bleu_1[i] = nltk.translate.bleu_score.sentence_bleu([reference_list], hypothesis_list, weights=(1,))
        bleu_4[i] = nltk.translate.bleu_score.sentence_bleu([reference_list], hypothesis_list, weights=(0.25, 0.25, 0.25, 0.25))
        meteor[i] = nltk.translate.meteor_score.meteor_score([reference_list], hypothesis_list)
        rouge_scores = scorer.score(reference, hypothesis)
        rougeL[i] = np.array(rouge_scores['rougeL'])

    print_metrics('ALL', ['BLEU1', 'BLEU4', 'METEOR', 'ROUGEL-P', 'ROUGEL-R', 'ROUGEL-F1'],
            [bleu_1.mean(), bleu_4.mean(), meteor.mean(), rougeL[:, 0].mean(), rougeL[:, 1].mean(), rougeL[:, 2].mean()])
    print_metrics('FINAL ANSWERS', ['BLEU1', 'BLEU4', 'METEOR', 'ROUGEL-P', 'ROUGEL-R', 'ROUGEL-F1'],
            [bleu_1[final_answers_idx].mean(), bleu_4[final_answers_idx].mean(),
            meteor[final_answers_idx].mean(), rougeL[final_answers_idx, 0].mean(),
            rougeL[final_answers_idx, 1].mean(), rougeL[final_answers_idx, 2].mean()])

    # ## Excel

    data = pd_generator(evaluation, dataset, exact_match, inclusion_match, bleu_1, bleu_4, meteor, rougeL, final_answers, dontknow)

    columns = ['Question', 'Prediction', 'Answer', 'EM', 'IM', 'BLEU1', 'B4', 'METEOR', 'ROUGEL', 'Answered', 'DontKnow', 'FULL prediction', 'FULL sample', 'Triples', 'AnswerBig']
    evaldf = pd.DataFrame(data, columns = columns)
    # print(evaldf.shape)
    # evaldf.head()

    if outfile:
        xlsx_file = outfile
    else:
        xlsx_file = f'{dataset_name}_{os.path.basename(infile)}.xlsx'
    if not force:
        assert not os.path.isfile(xlsx_file), f'Error: {xlsx_file} already exists'
    print(f'Writing {xlsx_file}')
    with pd.ExcelWriter(xlsx_file) as writer:
        evaldf.to_excel(writer, sheet_name=f"{dataset_name} X {os.path.basename(infile)}")

if __name__ == '__main__':
    main()
