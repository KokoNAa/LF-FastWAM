#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -c 'import torch; print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")'
PYTHONPATH=src "${PYTHON_BIN}" -m unittest discover -s tests -v
"${PYTHON_BIN}" scripts/validate_language_intervention_manifest.py \
  configs/eval/language_intervention_eval.example.jsonl
"${PYTHON_BIN}" -m compileall -q src experiments scripts tests

echo "LF-FastWAM static and small-model validation passed."
