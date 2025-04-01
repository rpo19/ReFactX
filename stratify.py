import json
import pandas as pd
from sklearn.model_selection import train_test_split
import click

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
@click.option('--dataset', 'dataset_path', required=True, type=click.Path(exists=True), help="Path to the input JSON dataset.")
@click.option('--columns', required=True, multiple=True, help="Columns to stratify by (e.g., --columns complexityType --columns category).")
@click.option('--sample-size', type=int, help="Size of the sample (use this or --sample-fraction).")
@click.option('--sample-fraction', type=float, help="Fraction of the dataset to sample (use this or --sample-size).")
@click.option('--output', required=True, type=click.Path(), help="Path to save the stratified sample.")
@click.option('--random-seed', type=int, default=42, help="Random seed for reproducibility.")
def stratify_dataset(dataset_path, columns, sample_size, sample_fraction, output, random_seed):
    """
    Perform stratified sampling on a dataset based on specified columns.
    """
    # Validate that either sample_size or sample_fraction is provided, but not both or neither
    if (sample_size is None and sample_fraction is None) or (sample_size is not None and sample_fraction is not None):
        raise ValueError("You must specify exactly one of --sample-size or --sample-fraction.")

    # Load dataset from JSON file
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

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

    # Save the stratified sample to a new JSON file
    with open(output, "w", encoding="utf-8") as f:
        json.dump(sampled_data, f, indent=4, ensure_ascii=False)

    # Print the stratified distributions of each specified column
    print('----- Sampled Distribution -----')
    for column in columns:
        print(f"Distribution for column '{column}':")
        distribution(stratified_sample, column)

    print(f"Stratified sample of {len(sampled_data)} items saved to '{output}'")

if __name__ == "__main__":
    stratify_dataset()
