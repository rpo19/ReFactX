import json
import sys
import pandas as pd
from pathlib import Path
import os

DEBUG = os.environ.get('DEBUG', False)

# Mapping of raw dataset file names to pretty names (None means ignore)
DATASET_RENAME = {
    "twowikimultihop_dev_ssample200": "2Wiki",
    "mintaka_test_ssample200": "Mintaka",
    "WebQSP_test_sample200": "WebQSP",
    # "WebQSP_test_sample200": None,
    "bi_test": "BI",
    # "bi_test": None,
    # skip other datasets if None
}


DATASET_ORDER = [
    "BI",
    "Mintaka",
    "2Wiki",
    "WebQSP",
]

# Mapping of metric names to clean names (None means ignore)
METRIC_RENAME = {
    "Percentage Answered": None,
    "Final Answers (Answered - Don't Know)": "Ans",
    "Percentage Don't Know": "DNK",
    # "Percentage 0 Triples": "0F",
    "Percentage 0 Triples": None,
    "Percentage 0 Triples (Final Answers)": None,
    "Percentage Max Tokens": "MaxT",
    # Add more as needed
    "Num": "NUM" if DEBUG else None,  # filtered out
    "New tokens (min, avg, max)": None,
    "New tokens (Final Answers)": None,
}

# Define your preferred order
METRIC_ORDER = [
    "Ans.",
    "DNK",
    "0F",
    "MaxT",
]

# Sort columns using defined orders
def sort_key(col):
    dataset, metric = col
    dataset_rank = DATASET_ORDER.index(dataset) if dataset in DATASET_ORDER else float('inf')
    metric_rank = METRIC_ORDER.index(metric) if metric in METRIC_ORDER else float('inf')
    return (dataset_rank, metric_rank)


def load_metrics(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    answered_metrics = pd.DataFrame.from_dict(data['answered_metrics'])

    metadata = dict(data["metadata"])
    model_name = metadata["model_config_path"].split("/")[-1].replace(".json", "").replace('_model','').replace('_', ' ')
    raw_dataset_name = metadata["dataset_path"].split("/")[-1]

    # Rename or skip dataset
    dataset_name = DATASET_RENAME.get(raw_dataset_name, raw_dataset_name)
    if dataset_name is None:
        return None  # skip this file

    # Filter and rename metrics
    rows = []
    for _, row in answered_metrics.iterrows():
        raw_metric = row["Metric"]
        new_metric = METRIC_RENAME.get(raw_metric)
        if new_metric is None:
            continue
        rows.append((new_metric, row["Value"]))

    if not rows:
        return None

    # Convert to DataFrame row
    renamed_metrics = pd.Series(dict(rows))
    return pd.DataFrame([renamed_metrics.values], columns=[(dataset_name, m) for m in renamed_metrics.index], index=[model_name])

def main(json_files):
    all_rows = []

    for file_path in json_files:
        row_df = load_metrics(file_path)
        all_rows.append(row_df)

    # Merge all rows (models) into one DataFrame
    merged = pd.concat(all_rows)

     # Group by model and aggregate non-NaN values
    # For example, you can use 'first' or 'max' to collapse values
    merged_grouped = merged.groupby(merged.index).first()  # Collapse by taking the first non-NaN value

    merged_grouped = merged_grouped.reindex(columns=sorted(merged_grouped.columns, key=sort_key))


    # Ensure proper MultiIndex for columns (model, dataset)
    merged_grouped.columns = pd.MultiIndex.from_tuples(merged_grouped.columns)
    merged_grouped.index.name = "Model"

    # Sort the dataframe by model and dataset
    # merged_grouped = merged_grouped.sort_index(axis=1).sort_index(axis=0)

    merged_grouped*=100

    # Output LaTeX
    print(merged_grouped.to_latex(
        index=True,
        multicolumn=True,
        multicolumn_format='c',
        float_format="%.1f",
        caption="Merged Answered Metrics by Model and Dataset",
        label="tab:answered_metrics_grouped"
    ))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python get_latex_tables.py <file1.json> <file2.json> ...")
        sys.exit(1)

    main(sys.argv[1:])

