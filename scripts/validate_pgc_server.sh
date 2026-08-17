#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -c \
  'import torch; print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")'
PYTHONPATH=src "${PYTHON_BIN}" -m unittest discover -s tests -v
"${PYTHON_BIN}" -m compileall -q src experiments scripts tests
bash -n scripts/train_pgc_libero.sh scripts/train_pgc_libero_suite.sh \
  scripts/train_pgc_v3_libero_suite.sh \
  scripts/train_pgc_v4_libero_suite.sh \
  scripts/train_pgc_v5_libero_suite.sh \
  scripts/train_pgc_v6_libero_suite.sh \
  scripts/train_pgc_v7_libero_suite.sh \
  scripts/eval_pgc_libero.sh \
  scripts/build_pgc_libero_datasets.sh scripts/validate_pgc_server.sh

echo "PGC-FastWAM static, regression, and small-model validation passed."
