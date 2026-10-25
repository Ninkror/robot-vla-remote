#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${root}/src"
exec "${root}/.venv/bin/python" -m common.launch lerobot dataset.curate "$@"
