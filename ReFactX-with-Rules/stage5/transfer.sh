#!/bin/bash
# Package Stage 5 locally, optionally upload it, and print the server commands.
set -euo pipefail

if [[ $# -gt 1 || (${1:-} != "" && ${1:-} != "--upload") ]]; then
    echo "Usage: $0 [--upload]" >&2
    exit 2
fi

root="$(cd "$(dirname "$0")/.." && pwd)"
server="${REFACTX_SERVER:-}"
port="${REFACTX_PORT:-22}"
if [[ ${1:-} == --upload && -z "$server" ]]; then
    echo 'Set REFACTX_SERVER to your SSH destination (user@host) before uploading.' >&2
    exit 2
fi
archive="${REFACTX_ARCHIVE:-/tmp/rules_srv.tgz}"
temp_archive="$(mktemp "${archive}.XXXXXX")"
trap 'rm -f "$temp_archive"' EXIT

files=(
    requirements.txt
    requirements-benchmark.txt
    nemo_rule.py
    nemo_reasoner.py
    stage3/tasks.py
    stage3/rule_harness.py
    stage4/rule_constraint.py
    stage4/logits_processor.py
    stage5/eval_rules.py
    stage5/run_benchmark.py
    stage5/rerun_server.sh
    data/triples.csv
    data/entities.csv
)

COPYFILE_DISABLE=1 tar -czf "$temp_archive" -C "$root" "${files[@]}"
mv -f "$temp_archive" "$archive"
echo "Archive: $archive"

if [[ ${1:-} == --upload ]]; then
    scp -P "$port" "$archive" "${server}:~/rules_srv.tgz"
elif [[ -n "$server" ]]; then
    printf 'Upload from the Mac: scp -P %q %q %q\n' "$port" "$archive" "${server}:~/rules_srv.tgz"
else
    printf 'Upload example: scp -P %q %q user@host:~/rules_srv.tgz\n' "$port" "$archive"
fi

if [[ -n "$server" ]]; then
    printf '\nConnect: ssh -p %q %q\n' "$port" "$server"
fi
cat <<'EOF'
On the server:
  mkdir -p ~/refactx && tar -xzf ~/rules_srv.tgz -C ~/refactx && \
    bash ~/refactx/stage5/rerun_server.sh smoke

For the three-sample run:
  bash ~/refactx/stage5/rerun_server.sh benchmark
EOF
