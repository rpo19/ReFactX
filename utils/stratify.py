import json
import pandas as pd
from sklearn.model_selection import train_test_split
import click
import random
import importlib

def distribution(df, column: str, do_print=True):
    """
    Calculate the distribution of a column in the DataFrame.
    """
    assert isinstance(column, str)
    dist = df[column].value_counts(normalize=True).reset_index()
    dist_df = pd.DataFrame(dist)
    if do_print:
        print(dist_df.to_markdown())
    return dist_df

@click.command()
@click.option('--dataset', 'dataset_config_path', required=True, type=str, help="Path to the dataset config.")
@click.option('--columns', required=False, multiple=True, default=[], help="Columns to stratify by (e.g., --columns complexityType --columns category).")
@click.option('--sample-size', type=int, help="Size of the sample (use this or --sample-fraction).")
@click.option('--sample-fraction', type=float, help="Fraction of the dataset to sample (use this or --sample-size).")
@click.option('--output', required=True, type=click.Path(), help="Path to save the stratified sample.")
@click.option('--random-seed', type=int, default=42, help="Random seed for reproducibility.")
@click.option('--json-questions-path', type=str, default=".", help='Json path for the questions in the json file (e.g. ".dataset.questions").')
@click.option('--input-encoding', type=str, default="utf-8", help='Input dataset encoding.')
@click.option('--output-encoding', type=str, default="utf-8", help='Input dataset encoding.')
def stratify_dataset(dataset_config_path, columns, sample_size, sample_fraction, output, random_seed, json_questions_path, input_encoding, output_encoding):
    """
    Perform stratified sampling on a dataset based on specified columns.
    """
    # Validate that either sample_size or sample_fraction is provided, but not both or neither
    if (sample_size is None and sample_fraction is None) or (sample_size is not None and sample_fraction is not None):
        raise ValueError("You must specify exactly one of --sample-size or --sample-fraction.")

    if dataset_config_path.endswith('.py'):
        dataset_config_path = dataset_config_path[:-3]
    dataset_module = importlib.import_module(dataset_config_path)
    dataset = getattr(dataset_module, 'dataset')
    data = dataset.dataset

    if len(columns) > 0:
        # Do stratified sampling
        print('Stratified sampling.')
        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Print the initial distributions of each specified column
        print('----- Original Distribution -----')
        for column in columns:
            print(f"Distribution for column '{column}':")
            distribution(df, column)

        # Determine the test_size for train_test_split
        if sample_fraction is not None:
            test_size = 1 - sample_fraction
        else:
            test_size = 1 - (sample_size / len(df))

        # Perform stratified sampling
        stratified_sample, _ = train_test_split(
            df,
            test_size=test_size,  # Keep the desired fraction or size
            stratify=df[list(columns)],
            random_state=random_seed  # Use the provided random seed
        )

        # Convert back to JSON
        sampled_data = stratified_sample.to_dict(orient="records")

        # Print the stratified distributions of each specified column
        print('----- Sampled Distribution -----')
        for column in columns:
            print(f"Distribution for column '{column}':")
            distribution(stratified_sample, column)
    else:
        # If no columns are specified, just sample randomly
        print('Normal sampling.')
        if sample_fraction is not None:
            sample_size = int(len(data) * sample_fraction)
        random.seed(random_seed)  # Set the random seed for reproducibility
        sampled_data = random.sample(data, sample_size)

    dataset.dataset = sampled_data
    raw_data = dataset.dump()

    # Save the stratified sample to a new JSON file
    with open(output, "w", encoding=output_encoding) as f:
        json.dump(raw_data, f)

    print(f"Stratified sample of {len(sampled_data)} items saved to '{output}'")

if __name__ == "__main__":
    stratify_dataset()
