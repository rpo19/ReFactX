import click
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
import importlib
from transformers.generation.logits_process import LogitsProcessor,LogitsProcessorList
import re
from tqdm import trange

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

# import torch
# class YesNoLogitsProcessor(LogitsProcessor):
#     # yes and no tokens are lists of token ids
#     def __init__(self, yes_tokens, no_tokens):
#         super().__init__()
#         self.yes_token = yes_tokens
#         self.no_token = no_tokens
#         self.allowed_tokens = yes_tokens + no_tokens

#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
#         allowed_scores = scores[:, self.allowed_tokens]
#         scores[:,:] = -float('inf')
#         scores[:, self.allowed_tokens] = allowed_scores
#         return scores

# allowed_answers = {
#     'yes': ['yes', 1],
# }

@click.command()
@click.option('--model', default="gpt2", help="HuggingFace model to load.")
@click.option('--dataset', 'dataset_path', required=False, help="Path to the dataset file (JSON).")
@click.option('--predictions', required=True, help="Path to the predictions file (JSON).")
@click.option('--fix-predictions', is_flag=True, default=False, help='Fix (missing) predictions in the evaluation.')
@click.option('--no-fix-none-prediction', is_flag=True, default=True, help='Do not replace None predictions with an empty string.')
@click.option('--split-pattern', required=False, default=r'<\|im_end\|>', help='Pattern to split the full prediction. Use with --fix-predictions.')
@click.option('--outfile', required=False, default=None, help="Output file for the results.")
@click.option('--device-map', required=False, default='cuda', help="Where to load the model.")
def judge_predictions(model, dataset_path, predictions, fix_predictions, no_fix_none_prediction, split_pattern, outfile, device_map):
    """
    Use an LLM to judge the correctness of predictions based on a dataset.
    """
    # Load the HuggingFace model and tokenizer
    print(f"Loading model: {model}")
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForCausalLM.from_pretrained(model, device_map=device_map)

    if outfile is None:
        outfile = f"{os.path.basename(predictions)}_llm_as_a_judge_results.out"

    evaluation_raw = []
    with open(predictions) as fd:
        line = fd.readline()
        while line:
            evaluation_raw.append(json.loads(line))
            line = fd.readline()

    header = evaluation_raw[0]

    # Load dataset and predictions
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

    # Save results to a JSONL file
    with open(outfile, "w") as f:
        for i in trange(len(evaluation)):
            assert evaluation[i]['question'] == dataset[i]['question']
            question = dataset[i]['question']
            correct_answer = dataset.get_answer(i)
            predicted_answer = evaluation[i]['prediction']

            if question and correct_answer and predicted_answer:
                # Construct the prompt for the LLM
                prompt = (
                    f"Given the question: '{question}', the correct answer: '{correct_answer}', "
                    f"and the predicted answer: '{predicted_answer}', is the predicted answer correct? (yes/no)"
                )

                # Tokenize the input
                inputs = tokenizer(prompt, return_tensors="pt")
                inputs.to(model.device)

                # Generate a single token
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    num_beams=1,
                    num_return_sequences=1,
                    top_p=None,
                    top_k=None,
                    # logits_processor=None, # TODO constrain the output to 'yes' or 'no'
                )

                # Decode the generated token
                llm_output = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().lower()

                # Extract the decision (yes/no)
                decision = "yes" if "yes" in llm_output else "no"
                current_result = {
                    "index": i,
                    "question": question,
                    "correct_answer": correct_answer,
                    "predicted_answer": predicted_answer,
                    "llm_decision": decision,
                    "llm_full_answer": llm_output,
                }

                json.dump(current_result, f)
                f.write('\n')
            else:
                missing_fields = []
                if not question:
                    missing_fields.append("question")
                if not correct_answer:
                    missing_fields.append("correct_answer")
                if not predicted_answer:
                    missing_fields.append("predicted_answer")
                print(f"Skipping sample {i} due to missing fields: {', '.join(missing_fields)}.")
                current_result = {
                    "index": i,
                    "question": question,
                    "correct_answer": correct_answer,
                    "predicted_answer": predicted_answer,
                    "llm_decision": "skipped",
                    "llm_full_answer": f"Missing fields: {', '.join(missing_fields)}",
                }
                json.dump(current_result, f)
                f.write('\n')

    print(f"Judgment completed. Results saved to '{outfile}'.")

if __name__ == "__main__":
    judge_predictions()
