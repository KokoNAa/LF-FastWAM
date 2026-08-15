#!/usr/bin/env bash
set -euo pipefail

DEMO_ROOT="${1:?Usage: bash scripts/build_pgc_libero_datasets.sh <libero_hdf5_root> <output_root> [episodes_per_pair] [seed]}"
OUTPUT_ROOT="${2:?Missing output root}"
EPISODES_PER_PAIR="${3:-5}"
SEED="${4:-42}"

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
MAX_DEMOS_PER_PAIR="${PGC_MAX_DEMOS_PER_PAIR:-50}"
VIDEO_CODEC="${PGC_VIDEO_CODEC:-h264}"
PLAN_ONLY="${PGC_PLAN_ONLY:-false}"
RESUME="${PGC_RESUME:-false}"
ALLOW_PARTIAL="${PGC_ALLOW_PARTIAL:-false}"
RELAXED_SCENE_MATCH="${PGC_RELAXED_SCENE_MATCH:-false}"
MANIFEST_ROOT="${OUTPUT_ROOT}/manifests"
LOG_ROOT="${OUTPUT_ROOT}/logs"

if [[ ! -d "${DEMO_ROOT}" ]]; then
  echo "LIBERO HDF5 demonstration root not found: ${DEMO_ROOT}" >&2
  exit 1
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

mkdir -p "${MANIFEST_ROOT}" "${LOG_ROOT}"

manifest_args=(--output-dir "${MANIFEST_ROOT}")
if [[ "${ALLOW_PARTIAL}" == "true" ]]; then
  manifest_args+=(--allow-incomplete)
fi
if [[ "${RELAXED_SCENE_MATCH}" == "true" ]]; then
  manifest_args+=(--relaxed-scene-match)
fi
"${PYTHON_BIN}" scripts/prepare_pgc_libero_manifests.py "${manifest_args[@]}"

suites=(libero_spatial libero_object libero_goal libero_10)
pids=()

for index in "${!suites[@]}"; do
  suite="${suites[$index]}"
  manifest="${MANIFEST_ROOT}/${suite}_pgc.jsonl"
  output="${OUTPUT_ROOT}/${suite}_pgc_counterfactual_lerobot"
  log="${LOG_ROOT}/${suite}.log"
  extra_args=()
  if [[ "${PLAN_ONLY}" == "true" ]]; then
    extra_args+=(--plan-only)
  else
    extra_args+=(--output "${output}")
  fi
  if [[ "${RESUME}" == "true" ]]; then
    extra_args+=(--resume)
  fi
  if [[ "${ALLOW_PARTIAL}" == "true" ]]; then
    extra_args+=(--allow-partial)
  fi

  echo "Starting ${suite} PGC collection on visible GPU ${index}; log=${log}"
  env \
    CUDA_VISIBLE_DEVICES="${index}" \
    MUJOCO_GL=egl \
    MUJOCO_EGL_DEVICE_ID=0 \
    PYTHONPATH="${PWD}/src:${PWD}" \
    "${PYTHON_BIN}" scripts/build_pgc_libero_data.py \
      --suite "${suite}" \
      --manifest "${manifest}" \
      --demo-root "${DEMO_ROOT}" \
      --episodes-per-pair "${EPISODES_PER_PAIR}" \
      --max-demos-per-pair "${MAX_DEMOS_PER_PAIR}" \
      --seed "$((SEED + index * 1000))" \
      --video-codec "${VIDEO_CODEC}" \
      "${extra_args[@]}" \
      >"${log}" 2>&1 &
  pids+=("$!")
done

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "Completed ${suites[$index]}"
  else
    echo "FAILED ${suites[$index]} (see ${LOG_ROOT}/${suites[$index]}.log)" >&2
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  exit 1
fi
if [[ "${PLAN_ONLY}" == "true" ]]; then
  echo "PGC four-suite plan validation passed."
  exit 0
fi

DATASET_LIST="${OUTPUT_ROOT}/pgc_counterfactual_datasets.txt"
: >"${DATASET_LIST}"
for suite in "${suites[@]}"; do
  printf '%s\n' "${OUTPUT_ROOT}/${suite}_pgc_counterfactual_lerobot" \
    >>"${DATASET_LIST}"
done

"${PYTHON_BIN}" scripts/validate_pgc_counterfactual_datasets.py \
  --list "${DATASET_LIST}" \
  --require-suite libero_spatial \
  --require-suite libero_object \
  --require-suite libero_goal \
  --require-suite libero_10 \
  --require-complete-task-coverage

echo "PGC four-suite datasets complete: ${DATASET_LIST}"
