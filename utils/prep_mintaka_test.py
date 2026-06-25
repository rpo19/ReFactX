import json, os
from datasets import load_dataset

out_dir = f"{os.environ['WS_PATH']}/eval_data"
os.makedirs(out_dir, exist_ok=True)

print("Downloading mintaka test...")
ds = load_dataset("AmazonScience/mintaka", "en", split="test", trust_remote_code=True)
with open(f"{out_dir}/mintaka_test.jsonl", "w") as f:
    for s in ds:
        f.write(json.dumps({"question": s["question"], "answerText": s["answerText"]}, ensure_ascii=False) + "\n")
print(f"  {len(ds)} samples -> {out_dir}/mintaka_test.jsonl")
