import os
import subprocess
from pathlib import Path

# --- equivalent of PS1LOGIN="${PS1LOGIN:-$(hostname -A)}"
ps1login = os.environ.get("PS1LOGIN")
if not ps1login:
    ps1login = subprocess.check_output(["hostname", "-A"], text=True).strip()

print(ps1login)

user = os.environ.get("USER", "unknown")

# --- case matching logic
# $HOME is shared between the two clusters, but each node only mounts/writes
# one of these workspace paths. The hostname is only a hint, so prefer whichever
# path actually exists AND is writable, else fall back to the other one.
if "capella" in ps1login:
    preferred = Path(f"/data/cat/ws/{user}-horse")
    fallback = Path(f"/data/horse/ws/{user}-quokka")
    print("capella")

elif "alpha" in ps1login:
    preferred = Path(f"/data/horse/ws/{user}-quokka")
    fallback = Path(f"/data/cat/ws/{user}-horse")
    print("alpha")

else:
    raise RuntimeError("ERROR: unrecognized host.")


def _usable(path: Path) -> bool:
    return path.is_dir() and os.access(path, os.W_OK)


if _usable(preferred):
    ws_path = preferred
elif _usable(fallback):
    print(f"workspace fallback: {fallback}")
    ws_path = fallback
else:
    raise RuntimeError(
        f"ERROR: no writable workspace found (tried {preferred} and {fallback})."
    )

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
]

for p in paths:
    p.mkdir(parents=True, exist_ok=True)

# opencode (and other XDG-aware tools) create $XDG_CACHE_HOME/opencode at
# startup and abort if that path is not writable. Only redirect the cache into
# the workspace when the mkdir succeeds; otherwise leave XDG_CACHE_HOME unset
# so tools fall back to the default ~/.cache.
cache_home = ws_path / ".cache"
try:
    cache_home.mkdir(parents=True, exist_ok=True)
except OSError as exc:
    print(f"WARNING: cannot create {cache_home}; leaving XDG_CACHE_HOME unset: {exc}")
else:
    os.environ["XDG_CACHE_HOME"] = str(cache_home)
