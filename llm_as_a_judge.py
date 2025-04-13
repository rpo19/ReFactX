import time
import click
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
import importlib
from transformers.generation.logits_process import LogitsProcessor,LogitsProcessorList
import re
from tqdm import tqdm
from eval import get_utc_date_and_time
from torch.utils.data import DataLoader
import torch

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

class YesNoLogitsProcessor(LogitsProcessor):
    # yes and no tokens are lists of token ids
    def __init__(self, yes_tokens, no_tokens):
        super().__init__()
        self.yes_token = yes_tokens
        self.no_token = no_tokens
        self.allowed_tokens = yes_tokens + no_tokens

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        allowed_scores = scores[:, self.allowed_tokens]
        scores[:,:] = -float('inf')
        scores[:, self.allowed_tokens] = allowed_scores
        return scores

allowed_answers = {
    'yes': [' yes', ' Yes'],
    'no': [' no', ' No'],
}

@click.command()
@click.option('--model', 'model_name', required=True, help="HuggingFace model to load.")
@click.option('--dataset', 'dataset_path', required=False, help="Path to the dataset file (JSON).")
@click.option('--predictions', 'input_file', required=True, help="Path to the predictions file (JSON).")
@click.option('--fix-predictions', is_flag=True, default=False, help='Fix (missing) predictions in the evaluation.')
@click.option('--no-fix-none-prediction', is_flag=True, default=True, help='Do not replace None predictions with an empty string.')
@click.option('--split-pattern', required=False, default=r'(<\|im_end\|>|<\|end_of_text\|>)', help='Pattern to split the full prediction. Use with --fix-predictions.')
@click.option('--outfile', required=False, default=None, help="Output file for the results.")
@click.option('--device-map', required=False, default='cuda', help="Where to load the model.")
@click.option("--wandb", "wandb", is_flag=True, help="Log in wandb")
@click.option('--batch-size', default=1, help="Batch size for the dataloader.")
@click.option('--torch-dtype', required=False, default='bfloat16', help="Torch dtype for loading the model.")
def judge_predictions(model_name, dataset_path, input_file, fix_predictions, no_fix_none_prediction, split_pattern, outfile, device_map, wandb, batch_size, torch_dtype):
    """
    Use an LLM to judge the correctness of predictions based on a dataset.
    """
    if wandb:
        print('Logging in wandb.')
        time.sleep(5) # let the user time to stop
    # Load the HuggingFace model and tokenizer
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    yes_tokens = [tokenizer(answer, add_special_tokens=False)['input_ids'][-1] for answer in allowed_answers['yes']]
    no_tokens = [tokenizer(answer, add_special_tokens=False)['input_ids'][-1] for answer in allowed_answers['no']]

    print('Checking yes and no tokens:')
    print("Yes tokens: {} - ".format(yes_tokens, tokenizer.convert_ids_to_tokens(yes_tokens)))
    print([tokenizer.decode(tokenizer("Is the answer correct?", add_special_tokens=False)['input_ids'] + [token]) for token in yes_tokens])
    print("No tokens: {} - ".format(no_tokens, tokenizer.convert_ids_to_tokens(no_tokens)))
    print([tokenizer.decode(tokenizer("Is the answer correct?", add_special_tokens=False)['input_ids'] + [token]) for token in no_tokens])

    yesnoprocessor = YesNoLogitsProcessor(yes_tokens, no_tokens)
    processor = LogitsProcessorList([yesnoprocessor])

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map, torch_dtype=torch_dtype)
    model.pad_token_id = tokenizer.eos_token_id

    if outfile is None:
        outfile = f"{os.path.basename(input_file)}_llm_as_a_judge_results.out"

    evaluation_raw = []
    with open(input_file) as fd:
        line = fd.readline()
        while line:
            evaluation_raw.append(json.loads(line))
            line = fd.readline()

    header = evaluation_raw[0]

    # Load dataset and predictions
    if dataset_path is None:
        dataset_path = header.get('dataset_config_path')
        if dataset_path is None:
            raise Exception('No dataset path provided. Nor in the --infile nor as --dataset.')
    if dataset_path.endswith('.py'):
        dataset_path = dataset_path[:-3]
    dataset_module = importlib.import_module(dataset_path)
    dataset = getattr(dataset_module, 'dataset')

    start_from = header.get('dataset_config', {}).get('config', {}).get('start_from', 0)
    if start_from > 0:
        print('Datasets starts from', start_from)
        dataset = dataset[start_from:]
    evaluation = evaluation_raw[1:]

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
        # writing metadata
        experiment_name = os.path.basename(input_file)
        metadata_plus = {
            "experiment_name": experiment_name,
            "model": model_name,
            "dataset_config_path": dataset_path,
            "prediction_file": input_file,
            "input_file": input_file,
            "fix_predictions": fix_predictions,
            "no_fix_none_prediction": no_fix_none_prediction,
            "split_pattern": split_pattern,
            "outfile": outfile,
            "device_map": device_map,
        }
        f.write(json.dumps(metadata_plus))
        f.write('\n')

        if wandb:
            import wandb

            wandb.init(
                project=experiment_name,
                config=metadata_plus,
                name=f"llm_as_a_judge_{experiment_name}_{get_utc_date_and_time()}",
            )

        # prepare a dataset : list with all the prompts
        output_complete = []
        prompts = []
        prompts_idx = []
        skipped_idx = []
        prompt_template = {
            'role': 'system',
            'content': 'You are a judge. You will be given a question, the correct answer, and a predicted answer. The correct answer is a list that contains the different correct answers that must be mentioned in a correct prediction. Optionally the answer can have aliases. You consider correct when a prediction contains all the answers (or eventually their aliases).'
        }
        # TODO add few shot examples?
        for i in range(len(evaluation)):
            question = dataset.get_question(i)
            assert evaluation[i]['question'] == question, (f"Question mismatch: {evaluation[i]['question']} != {question}")
            correct_answer = dataset.get_answer(i)
            predicted_answer = evaluation[i]['prediction']
            if question and correct_answer and predicted_answer:
                prompt_question = {
                    'role': 'user',
                    'content': f'''f"Given the question: '{question}', the correct answers: '{correct_answer}' and the predicted answer: '{predicted_answer}', is the predicted answer correct? (yes/no)
'''}
                prompt = tokenizer.apply_chat_template(prompt_template + [prompt_question], tokenize=False, add_generation_prompt=True)
                prompts.append(prompt)
                prompts_idx.append(i)
                current_result = {
                    "index": i,
                    "question": question,
                    "correct_answer": correct_answer,
                    "predicted_answer": predicted_answer,
                    "prompt": prompt,
                }
                output_complete.append(current_result)
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
                output_complete.append(current_result)
                skipped_idx.append(i)

        assert len(output_complete) == len(evaluation)

        # TODO solve Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
        dataloader = DataLoader(prompts, batch_size=batch_size, shuffle=False)

        output_complete_cursor = 0

        model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader):
                # Tokenize the input
                inputs = tokenizer(batch, return_tensors="pt",
                                    padding=True,
                                    padding_side='left')
                inputs.to(model.device)

                # Generate a single token
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False,
                    num_beams=1,
                    num_return_sequences=1,
                    top_p=None,
                    top_k=None,
                    temperature=None,
                    logits_processor=processor,
                )

                for prompt, output in zip(batch, outputs):
                    while output_complete_cursor in skipped_idx:
                        json.dump(output_complete[output_complete_cursor], f)
                        f.write('\n')
                        if wandb:
                            wandb.log(output_complete[output_complete_cursor])
                        output_complete_cursor += 1
                        continue
                    assert output_complete_cursor in prompts_idx
                    assert output_complete[output_complete_cursor]['prompt'] == prompt
                    llm_answer = tokenizer.decode(output[inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().lower()
                    decision = llm_answer

                    assert decision == 'yes' or decision == 'no'

                    output_complete[output_complete_cursor]['llm_decision'] = decision
                    output_complete[output_complete_cursor]['llm_full_answer'] = llm_answer
                    json.dump(output_complete[output_complete_cursor], f)
                    f.write('\n')
                    if wandb:
                        wandb.log(output_complete[output_complete_cursor])
                    output_complete_cursor += 1

            while output_complete_cursor in skipped_idx:
                json.dump(output_complete[output_complete_cursor], f)
                f.write('\n')
                if wandb:
                    wandb.log(output_complete[output_complete_cursor])
                output_complete_cursor += 1
                continue
            assert output_complete_cursor == len(evaluation), (output_complete_cursor, len(evaluation))

    print(f"Judgment completed. Results saved to '{outfile}'.")

if __name__ == "__main__":
    judge_predictions()
