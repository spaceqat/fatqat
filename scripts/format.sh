#!/usr/bin/env bash
# Auto-format the codebase with black (writes changes in place).
#
# Usage: scripts/format.sh
# Install black first with: python -m pip install --group lint
# Config lives in [tool.black] in pyproject.toml. Run scripts/lint.sh to verify
# formatting and static analysis without modifying files (what CI runs).
set -euo pipefail

cd "$(dirname "$0")/.."

python -m black src tests conftest.py
