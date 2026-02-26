#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/create-venv.sh [python_executable]
# Example:
#   scripts/create-venv.sh python3.11

PYTHON_BIN="${1:-python3}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"

cd "$PROJECT_ROOT"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Error: Python executable '$PYTHON_BIN' not found on PATH." >&2
  exit 1
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -r "$PROJECT_ROOT/requirements.txt"

echo "Created virtual environment at: $VENV_DIR"
echo "Activate it with: source .venv/bin/activate"
