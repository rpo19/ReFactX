#!/usr/bin/env bash
# Download, install and populate vadalog-portable with the synthetic KG.
#
#   bash vadalog_full/setup_vadalog.sh [DATA_DIR]
#
# 1. installs vadalog-portable (pure Python, MIT, no server, no JVM) from GitHub;
# 2. generates the synthetic KG in DATA_DIR if it is missing (kg_gen.py);
# 3. runs the company-ownership program once and caches the derived relations
#    (DATA_DIR/vadalog_cache.pkl, ~25s the first time);
# 4. exports a standalone program + CSVs to DATA_DIR/vadalog/ and checks it
#    with the vadalog CLI.
#
# "Starting" Vadalog: the engine runs in-process (refactx.rules.VadalogKG).
# To browse/edit the exported program in the Vadalog web editor:
#     vadalog ui --root DATA_DIR/vadalog --no-browser      # http://127.0.0.1:8765
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${1:-$HERE/data}"
VADALOG_REF="${VADALOG_REF:-b0e96f5b6c60ba4a507916c60795f515e16b26cf}"
PY="${PYTHON:-python3}"

if ! "$PY" -c "import vadalog" 2>/dev/null; then
  echo "== installing vadalog-portable@${VADALOG_REF:0:8}"
  "$PY" -m pip install -q "git+https://github.com/joint-kg-labs/vadalog-portable@${VADALOG_REF}"
fi
"$PY" -c "import vadalog; print('vadalog-portable', vadalog.__version__)"

mkdir -p "$DATA_DIR"
if [ ! -f "$DATA_DIR/triples.csv" ]; then
  echo "== generating the synthetic KG in $DATA_DIR"
  (cd "$DATA_DIR/.." && "$PY" "$HERE/kg_gen.py")
fi

echo "== reasoning + export"
"$PY" -c "
import sys; sys.path.insert(0, '$HERE')
import vadalog_reasoner as vr
kg = vr.build_kg('$DATA_DIR/triples.csv')
print('exported', kg.export('$DATA_DIR/vadalog', 'company_kg.vada'))
"
if command -v vadalog >/dev/null; then
  (cd "$DATA_DIR/vadalog" && vadalog check company_kg.vada)
else
  (cd "$DATA_DIR/vadalog" && "$PY" -m vadalog check company_kg.vada)
fi
echo "== done. Web editor: vadalog ui --root $DATA_DIR/vadalog --no-browser"
