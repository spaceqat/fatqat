#!/usr/bin/env bash
# Lint gate: verify black formatting and run pylint static analysis.
# Read-only — makes no changes. This is exactly what the Lint CI workflow runs.
#
# Usage: scripts/lint.sh
# To auto-fix formatting failures, run scripts/format.sh.
# Install the tools first with: python -m pip install --group dev --group lint
# Config lives in [tool.black] and [tool.pylint] in pyproject.toml.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> black --check"
python -m black --check --diff src tests conftest.py

echo "==> pylint"
python -m pylint src/fatqat tests conftest.py
