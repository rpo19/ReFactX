import json, os
from datasets import Dataset

cached_dir = "/data/horse/ws/ripo631h-quokka/huggingface/datasets/Rexhaif___mintaka-qa-en/default/0.0.0/aa8060833dacf80e78e83b3666ccf4e39802ec9a"
out_dir = f"{os.environ['WS_PATH']}/eval_data"
os.makedirs(out_dir, exist_ok=True)

ds = Dataset.from_file(f"{cached_dir}/mintaka-qa-en-test.arrow")
print(f"Mintaka test: {len(ds)} rows")

with open(f"{out_dir}/mintaka_test.jsonl", "w") as f:
    for i, s in enumerate(ds):
        f.write(json.dumps({
            "question": s["question"],
            "answerText": s["answer"],
        }, ensure_ascii=False) + "\n")
    print(f"Saved {i+1} samples to {out_dir}/mintaka_test.jsonl")

