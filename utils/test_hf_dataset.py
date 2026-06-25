import sys
from datasets import load_dataset

name = sys.argv[1]
try:
    ds = load_dataset(name, "en", split="test", trust_remote_code=True)
    print(f"OK: {len(ds)} samples")
    print(f"Keys: {list(ds[0].keys())}")
except Exception as e:
    print(f"FAIL: {e}")
