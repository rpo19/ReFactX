from datasets import Dataset
import click
import importlib

import re

split_pattern=r'(<\|im_end\|>|<\|end_of_text\|>)'
answer_pattern = re.compile(r'Answer: (.*)\.?')
def get_prediction(full_prediction, remove_dot=True):
    prediction = ''

    full_prediction = re.split(split_pattern, full_prediction, 1)[0]
    if remove_dot and full_prediction.endswith('.'):
        full_prediction = full_prediction[:-len('.')]
    match = answer_pattern.search(full_prediction)
    if match:
        prediction = match.group(1)

    return prediction

def extract_subjects(triples):
    # with regex
    return []
def extract_preds(triples):
    # with regex
    return []
def extract_objects(triples):
    # with regex
    return []

# max: at least one triple should do this
# next triples could follow e.g. in a chain
def jaccard_triples(question, triples, extractor_fun, max_score=.1, aggregate=max):
    scores = []
    for sub in extractor_fun(triples):
        scores.append(jaccard(question, sub))
    score = aggregate(scores) * max_score
    return score

def extract_solution(solution_str, method='strict'):
    assert method in ['strict', 'flexible']

    if method == 'strict':
        # this also tests the formatting of the model
        solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
        if solution is None:
            final_answer = None
        else:
            final_answer = solution.group(0)
            final_answer = final_answer.split('#### ')[1].replace(',', '').replace('$', '')
    elif method == 'flexible':
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            invalid_str = ['', '.']
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer

def jaccard(answer, ground_truth):
    # jaccard index on words
    answer_words = set(answer.lower().split())
    gt_words = set(ground_truth.lower().split())
    num = answer_words.intersection(gt_words)
    denom = answer_words.union(gt_words)
    iou = len(num) / len(denom)
    return iou


# def compute_score(solution_str, ground_truth, method='strict', format_score=0., score=1.):
def compute_score(solution_str, ground_truth, question, triples):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = get_prediction(solution_str).lower()
    ground_truth = ground_truth.lower()

    score = 0.
    # check if generates constrained triples: MUST HAVE
    if len(triples) > 0:
        # exact match case insensitive
        if answer == "i don't know":
            score += .1 # better than no answer
        elif answer == ground_truth:
            score += .8 # need also good reasoning for 1
        else:
            # jaccard index for complex comparison
            iou = jaccard(answer, ground_truth)
            score += iou*.8 # max 0.8

        # Reasoning max +.03
        # check if it is using question for triples
        score += jaccard_triples(question, triples, extractor_fun=extract_subjects, max_score=0.1)
        score += jaccard_triples(question, triples, extractor_fun=extract_preds, max_score=0.1)
        # check if it is using triples for answering
        score += jaccard_triples(answer, triples, extractor_fun=extract_objects, max_score=0.1)

    # clip score
    score = max([score, 1.])

    return score


def prompt_fn(question):
    return question

@click.command()
@click.option("--name", default=None, required=False, help="Dataset name.")
@click.option("--output", "output_file", required=True, default=None, type=click.Path(), help="Output file for the results.")
@click.option("--dataset", "dataset_config_path", required=True, help="Dataset configuration module.")
def main(name, output_file, dataset_config_path):
    if dataset_config_path.endswith('.py'):
        dataset_config_path = dataset_config_path[:-3]
    dataset_module = importlib.import_module(dataset_config_path)
    dataset = getattr(dataset_module, 'dataset')

    ds = Dataset.from_generator(dataset.to_verl(prompt_fn=prompt_fn, name=name))
    ds.to_parquet(output_file)

if __name__ == "__main__":
    main()
