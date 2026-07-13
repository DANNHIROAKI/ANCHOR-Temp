#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-python3}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$repo_root/python${PYTHONPATH:+:$PYTHONPATH}"

args=("$@")
if (($# == 0)); then
  args=(--all --data-root "$repo_root/data")
fi
exec "$python_bin" -m anchor_exp.cli.prepare_real "${args[@]}"
