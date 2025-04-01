import click
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

@click.command()
@click.option('--model', default="gpt2", help="HuggingFace model to load.")
@click.option('--dataset', required=True, help="Path to the dataset file (JSON).")
@click.option('--predictions', required=True, help="Path to the predictions file (JSON).")
def judge_predictions(model, dataset, predictions):
    """
    Use an LLM to judge the correctness of predictions based on a dataset.
    """
    # Load the HuggingFace model and tokenizer
    print(f"Loading model: {model}")
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForCausalLM.from_pretrained(model)

    # Load dataset and predictions
    with open(dataset, 'r') as f:
        dataset_samples = json.load(f)
    with open(predictions, 'r') as f:
        prediction_samples = json.load(f)

    # Ensure dataset and predictions align
    if len(dataset_samples) != len(prediction_samples):
        raise ValueError("Dataset and predictions must have the same number of samples.")

    # Iterate through samples and judge predictions
    results = []
    for i, (data, prediction) in enumerate(zip(dataset_samples, prediction_samples)):
        question = data.get("question")
        correct_answer = data.get("correct_answer")
        predicted_answer = prediction.get("predicted_answer")

        if not question or not correct_answer or not predicted_answer:
            print(f"Skipping sample {i} due to missing fields.")
            continue

        # Construct the prompt for the LLM
        prompt = (
            f"Given the question: '{question}', the correct answer: '{correct_answer}', "
            f"and the predicted answer: '{predicted_answer}', is the predicted answer correct? (yes/no)"
        )

        # Tokenize the input
        inputs = tokenizer(prompt, return_tensors="pt")

        # Generate a single token
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            num_beams=1,
            num_return_sequences=1,
            logits_processor=None, # TODO constrain the output to 'yes' or 'no'
        )

        # Decode the generated token
        llm_output = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().lower()

        # Extract the decision (yes/no)
        decision = "yes" if "yes" in llm_output else "no"
        results.append({
            "question": question,
            "correct_answer": correct_answer,
            "predicted_answer": predicted_answer,
            "llm_decision": decision
        })

    # Save results to a file
    with open("judgment_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Judgment completed. Results saved to 'judgment_results.json'.")

if __name__ == "__main__":
    judge_predictions()