import json
import pandas as pd
import sys
from collections import defaultdict

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


# Maps raw metric names to (metric_type, eval_setting)
METRIC_MAP = {
    "Exact Match (All)": ("Exact Match", "C"),
    "Exact Match (Final Answers)": ("Exact Match", "P"),
    "Judge Match (All)": ("LLM-Judge", "C"),
    "Judge Match (Final Answers)": ("LLM-Judge", "P"),
}

PC_ORDER = ['P','C']

def extract_model_dataset(meta):
    model = meta["model_config_path"].split("/")[-1].replace(".json", "").replace('_model','').replace("_", " ")
    raw_dataset = meta["dataset_path"].split("/")[-1]
    dataset = DATASET_RENAME.get(raw_dataset, raw_dataset)
    return model, dataset

def load_metrics(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    metadata = dict(data["metadata"])
    model, dataset = extract_model_dataset(metadata)

    records = {}
    for metric, (metric_type, setting) in METRIC_MAP.items():
        try:
            index = list(data["em_im_metrics"]["Metric"].values()).index(metric)
            value = float(data["em_im_metrics"]["Value"][str(index)])
            records[(metric_type, dataset, setting)] = value
        except (KeyError, ValueError):
            continue  # Skip missing metrics

    return model, records

def main(json_files):
    model_records = defaultdict(dict)

    for file in json_files:
        model, metrics = load_metrics(file)
        model_records[model].update(metrics)

    # Collect all columns
    all_columns = sorted(
        {key for metrics in model_records.values() for key in metrics},
        key=lambda x: (x[0], DATASET_ORDER.index(x[1]), PC_ORDER.index(x[2]))  # Sort by metric_type, dataset order, setting
    )

    df = pd.DataFrame(index=model_records.keys(), columns=pd.MultiIndex.from_tuples(all_columns))

    for model, metrics in model_records.items():
        for col, val in metrics.items():
            df.loc[model, col] = val

    df = df.sort_index()

    df *= 100

    print(df.to_latex(
        float_format="%.1f",
        multicolumn=True,
        multirow=True,
        index_names=True,
        na_rep="--",
        caption="Exact Match and LLM-Judge performance per dataset and evaluation setting.",
        label="tab:em_judge_results"
    ))

if __name__ == "__main__":
    main(sys.argv[1:])
