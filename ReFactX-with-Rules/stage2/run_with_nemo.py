"""Run any repo script with nemo_reasoner injected as `reasoner`.

The repo scripts (api.py, qa_gen.py, add_groundtruth.py, test.py) do
`from reasoner import load` / `import reasoner as rz` and are NOT modified:
we pre-register nemo_reasoner under the name `reasoner` in sys.modules, so
their imports resolve to the Nemo-backed implementation.

Usage (from the repo root):
    python3 stage2/run_with_nemo.py api.py
    python3 stage2/run_with_nemo.py test.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import nemo_reasoner  # noqa: E402
sys.modules["reasoner"] = nemo_reasoner

if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(1)

target = sys.argv[1]
if not os.path.isfile(target):
    print("no such script:", target)
    sys.exit(1)

print(">>> running %s with backend: %s\n" % (target, nemo_reasoner.__file__))
with open(target) as f:
    code = compile(f.read(), target, "exec")
exec(code, {"__name__": "__main__", "__file__": os.path.abspath(target),
            "__builtins__": __builtins__})
