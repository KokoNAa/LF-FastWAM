#!/usr/bin/env bash
set -euo pipefail

SUITE="${1:?Usage: bash scripts/build_pgc_v9_libero_suite.sh <suite> <historical_manifest> <historical_cf_dataset> <output_root> [seed]}"
HISTORICAL_MANIFEST="${2:?Missing historical PGC manifest}"
HISTORICAL_CF_DATASET="${3:?Missing historical counterfactual LeRobot dataset}"
OUTPUT_ROOT="${4:?Missing V9 output root}"
BUILD_SEED="${5:-42}"

case "${SUITE}" in
  libero_spatial|libero_object|libero_goal|libero_10) ;;
  *)
    echo "Unsupported LIBERO suite: ${SUITE}." >&2
    exit 1
    ;;
esac
if ! [[ "${BUILD_SEED}" =~ ^[0-9]+$ ]]; then
  echo "Seed must be a non-negative integer." >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-/root/gpufree-data/fastwam/FastWAM/data/libero_mujoco3.3.2}"
LIBERO_DEMO_ROOT="${LIBERO_DEMO_ROOT:-/root/gpufree-data/fastwam/third_party/LIBERO/libero/datasets}"
NATIVE_DATASET="${LIBERO_DATA_ROOT}/${SUITE}_no_noops_lerobot"

for file in "${HISTORICAL_MANIFEST}"; do
  if [[ ! -f "${file}" ]]; then
    echo "Required manifest not found: ${file}" >&2
    exit 1
  fi
done
for directory in "${NATIVE_DATASET}" "${HISTORICAL_CF_DATASET}" "${LIBERO_DEMO_ROOT}"; do
  if [[ ! -d "${directory}" ]]; then
    echo "Required dataset directory not found: ${directory}" >&2
    exit 1
  fi
done

mkdir -p "${OUTPUT_ROOT}/manifests" "${OUTPUT_ROOT}/sidecars"
OUTPUT_ROOT="$(cd -- "${OUTPUT_ROOT}" && pwd -P)"
HISTORICAL_MANIFEST="$(cd -- "$(dirname -- "${HISTORICAL_MANIFEST}")" && pwd -P)/$(basename -- "${HISTORICAL_MANIFEST}")"
HISTORICAL_CF_DATASET="$(cd -- "${HISTORICAL_CF_DATASET}" && pwd -P)"
NATIVE_DATASET="$(cd -- "${NATIVE_DATASET}" && pwd -P)"
LIBERO_DEMO_ROOT="$(cd -- "${LIBERO_DEMO_ROOT}" && pwd -P)"

STRICT_MANIFEST="${OUTPUT_ROOT}/manifests/${SUITE}_strict_conflict.jsonl"
STRICT_REPORT="${OUTPUT_ROOT}/manifests/${SUITE}_strict_conflict.coverage.json"
STRICT_DATASET="${OUTPUT_ROOT}/${SUITE}_pgc_strict_counterfactual_lerobot"
NATIVE_SIDECAR="${OUTPUT_ROOT}/sidecars/${SUITE}_native_eraf"
HISTORICAL_SIDECAR="${OUTPUT_ROOT}/sidecars/${SUITE}_historical_cf_eraf"
STRICT_SIDECAR="${OUTPUT_ROOT}/sidecars/${SUITE}_strict_cf_eraf"

echo "[PGC-FastWAM] Building V9 strict-conflict manifest"
"${PYTHON_BIN}" scripts/prepare_pgc_libero_strict_manifest.py \
  --suite "${SUITE}" \
  --demo-root "${LIBERO_DEMO_ROOT}" \
  --output "${STRICT_MANIFEST}" \
  --report "${STRICT_REPORT}" \
  --demos-per-direction 5 \
  --max-demos-per-candidate 50 \
  --max-candidates-per-task 30 \
  --min-coverage 8 \
  --seed "${BUILD_SEED}"

STRICT_RESUME_ARGS=()
if [[ -d "${STRICT_DATASET}" ]]; then
  STRICT_RESUME_ARGS+=(--resume)
fi
echo "[PGC-FastWAM] Collecting five strict state-aligned demonstrations per pair"
"${PYTHON_BIN}" scripts/build_pgc_libero_data.py \
  --manifest "${STRICT_MANIFEST}" \
  --demo-root "${LIBERO_DEMO_ROOT}" \
  --output "${STRICT_DATASET}" \
  --suite "${SUITE}" \
  --episodes-per-pair 5 \
  --max-demos-per-pair 50 \
  --seed "${BUILD_SEED}" \
  "${STRICT_RESUME_ARGS[@]}"

for sidecar in "${NATIVE_SIDECAR}" "${HISTORICAL_SIDECAR}" "${STRICT_SIDECAR}"; do
  if [[ -e "${sidecar}/index.json" ]]; then
    echo "ERAF sidecar already exists: ${sidecar}." >&2
    echo "Choose a clean output root; sidecar audit files are immutable." >&2
    exit 1
  fi
done

echo "[PGC-FastWAM] Replaying native episodes for ERAF labels"
"${PYTHON_BIN}" scripts/build_pgc_libero_entity_relations.py \
  --dataset "${NATIVE_DATASET}" \
  --output "${NATIVE_SIDECAR}" \
  --manifest "${HISTORICAL_MANIFEST}" \
  --suite "${SUITE}" \
  --hdf5-root "${LIBERO_DEMO_ROOT}" \
  --seed "${BUILD_SEED}"

echo "[PGC-FastWAM] Replaying historical counterfactual episodes for ERAF labels"
"${PYTHON_BIN}" scripts/build_pgc_libero_entity_relations.py \
  --dataset "${HISTORICAL_CF_DATASET}" \
  --output "${HISTORICAL_SIDECAR}" \
  --manifest "${HISTORICAL_MANIFEST}" \
  --suite "${SUITE}" \
  --seed "${BUILD_SEED}"

echo "[PGC-FastWAM] Replaying strict counterfactual episodes for ERAF labels"
"${PYTHON_BIN}" scripts/build_pgc_libero_entity_relations.py \
  --dataset "${STRICT_DATASET}" \
  --output "${STRICT_SIDECAR}" \
  --manifest "${STRICT_MANIFEST}" \
  --suite "${SUITE}" \
  --seed "${BUILD_SEED}"

echo "[PGC-FastWAM] V9 data build complete"
echo "STRICT_MANIFEST=${STRICT_MANIFEST}"
echo "STRICT_REPORT=${STRICT_REPORT}"
echo "STRICT_DATASET=${STRICT_DATASET}"
echo "NATIVE_SIDECAR=${NATIVE_SIDECAR}"
echo "HISTORICAL_SIDECAR=${HISTORICAL_SIDECAR}"
echo "STRICT_SIDECAR=${STRICT_SIDECAR}"
