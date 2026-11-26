import os
import subprocess
from pathlib import Path

# --- equivalent of PS1LOGIN="${PS1LOGIN:-$(hostname -A)}"
ps1login = os.environ.get("PS1LOGIN")
if not ps1login:
    ps1login = subprocess.check_output(["hostname", "-A"], text=True).strip()

print(ps1login)

# --- case matching logic
if "capella" in ps1login:
    ws_path = Path(f"/data/cat/ws/{os.environ.get('USER','unknown')}-horse")
    print("capella")

elif "alpha" in ps1login:
    ws_path = Path(f"/data/horse/ws/{os.environ.get('USER','unknown')}-quokka")
    print("alpha")

else:
    raise RuntimeError("ERROR: unrecognized host.")

# --- export-like behavior (Python process environment)
os.environ["WS_PATH"] = str(ws_path)

hf_home = ws_path / "huggingface"
os.environ["HF_HOME"] = str(hf_home)
os.environ["HF_HUB_CACHE"] = str(hf_home / "hub")
os.environ["TRANSFORMERS_CACHE"] = str(hf_home / "transformers")
os.environ["HF_DATASETS_CACHE"] = str(hf_home / "datasets")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_home / "hub")

# --- mkdir -p equivalents
paths = [
    hf_home,
    hf_home / "hub",
    hf_home / "transformers",
    hf_home / "datasets",
    ws_path / ".cache",
]

for p in paths:
    p.mkdir(parents=True, exist_ok=True)

os.environ["XDG_CACHE_HOME"] = str(ws_path / ".cache")
