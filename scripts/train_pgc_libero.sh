#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${1:?Usage: bash scripts/train_pgc_libero.sh <gpus> <base_checkpoint> <counterfactual_datasets.txt> [seed] [max_steps]}"
BASE_CHECKPOINT="${2:?Missing released FastWAM base checkpoint}"
COUNTERFACTUAL_LIST="${3:?Missing direct-counterfactual dataset list file}"
TRAIN_SEED="${4:-42}"
MAX_STEPS="${5:-4000}"
TRAIN_SUITE="${PGC_TRAIN_SUITE:-all}"
ALLOW_JOINT_TRAINING="${PGC_ALLOW_JOINT_TRAINING:-false}"

PYTHON_BIN="${PYTHON_BIN:-python}"
LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-data/libero_mujoco3.3.2}"
STATS_PATH="${STATS_PATH:-${DIFFSYNTH_MODEL_BASE_PATH:-./checkpoints}/fastwam_release/libero_uncond_2cam224_dataset_stats.json}"
CACHE_DIR="${TEXT_CACHE_DIR:-data/text_embeds_cache/libero}"
OVERSAMPLE_FACTOR="${PGC_COUNTERFACTUAL_OVERSAMPLE_FACTOR:-2}"
LEARNING_RATE="${PGC_LEARNING_RATE:-1.0e-5}"
SAVE_EVERY="${PGC_SAVE_EVERY:-500}"
SAVE_TRAINING_STATE="${PGC_SAVE_TRAINING_STATE:-false}"
LORA_RANK="${PGC_LORA_RANK:-16}"
LORA_ALPHA="${PGC_LORA_ALPHA:-32}"
LORA_DROPOUT="${PGC_LORA_DROPOUT:-0.05}"
RUN_TAG="${RUN_TAG:-${TRAIN_SUITE}-action-lora-r${LORA_RANK}-${MAX_STEPS}-seed${TRAIN_SEED}-v1}"

ALL_SUITES=(libero_spatial libero_object libero_goal libero_10)
case "${TRAIN_SUITE}" in
  libero_spatial|libero_object|libero_goal|libero_10)
    REQUIRED_SUITES=("${TRAIN_SUITE}")
    ;;
  all)
    if [[ "${ALLOW_JOINT_TRAINING}" != "true" ]]; then
      echo "Joint four-suite training is locked." >&2
      echo "Run the four isolated suite experiments first; then set PGC_ALLOW_JOINT_TRAINING=true only after explicit approval." >&2
      exit 1
    fi
    REQUIRED_SUITES=("${ALL_SUITES[@]}")
    ;;
  *)
    echo "PGC_TRAIN_SUITE must be one of: ${ALL_SUITES[*]}, all; got ${TRAIN_SUITE}." >&2
    exit 1
    ;;
esac

for value_name in NPROC_PER_NODE MAX_STEPS OVERSAMPLE_FACTOR SAVE_EVERY LORA_RANK; do
  value="${!value_name}"
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${value_name} must be a positive integer, got ${value}." >&2
    exit 1
  fi
done
"${PYTHON_BIN}" - "${LORA_ALPHA}" "${LORA_DROPOUT}" <<'PY'
import sys

alpha = float(sys.argv[1])
dropout = float(sys.argv[2])
if alpha <= 0:
    raise SystemExit(f"PGC_LORA_ALPHA must be positive, got {alpha}")
if not 0 <= dropout < 1:
    raise SystemExit(f"PGC_LORA_DROPOUT must be in [0, 1), got {dropout}")
PY
if [[ ! -f "${BASE_CHECKPOINT}" ]]; then
  echo "Base checkpoint not found: ${BASE_CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -f "${COUNTERFACTUAL_LIST}" ]]; then
  echo "Counterfactual dataset list not found: ${COUNTERFACTUAL_LIST}" >&2
  exit 1
fi
if [[ ! -f "${STATS_PATH}" ]]; then
  echo "Dataset stats not found: ${STATS_PATH}" >&2
  exit 1
fi

VALIDATOR_ARGS=(--list "${COUNTERFACTUAL_LIST}" --require-complete-task-coverage)
for suite_name in "${REQUIRED_SUITES[@]}"; do
  VALIDATOR_ARGS+=(--require-suite "${suite_name}")
done
"${PYTHON_BIN}" scripts/validate_pgc_counterfactual_datasets.py \
  "${VALIDATOR_ARGS[@]}"

NATIVE_JSON="$(${PYTHON_BIN} - "${LIBERO_DATA_ROOT}" "${TRAIN_SUITE}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
scope = sys.argv[2]
mapping = {
    "libero_spatial": "libero_spatial_no_noops_lerobot",
    "libero_object": "libero_object_no_noops_lerobot",
    "libero_goal": "libero_goal_no_noops_lerobot",
    "libero_10": "libero_10_no_noops_lerobot",
}
names = list(mapping.values()) if scope == "all" else [mapping[scope]]
paths = [root / name for name in names]
missing = [str(path) for path in paths if not (path / "meta/tasks.jsonl").is_file()]
if missing:
    raise SystemExit("Missing native LIBERO datasets: " + ", ".join(missing))
print(json.dumps([str(path) for path in paths], separators=(",", ":")))
PY
)"

CF_JSON="$(${PYTHON_BIN} - "${COUNTERFACTUAL_LIST}" "${NATIVE_JSON}" "${TRAIN_SUITE}" <<'PY'
import json
import sys
from pathlib import Path

list_path = Path(sys.argv[1])
native = {str(Path(path).resolve()) for path in json.loads(sys.argv[2])}
scope = sys.argv[3]
expected_suites = (
    {"libero_spatial", "libero_object", "libero_goal", "libero_10"}
    if scope == "all"
    else {scope}
)
paths = []
covered_suites = set()
for raw in list_path.read_text(encoding="utf-8").splitlines():
    raw = raw.strip()
    if not raw or raw.startswith("#"):
        continue
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (list_path.parent / path).resolve()
    else:
        path = path.resolve()
    if str(path) in native:
        raise SystemExit(f"Counterfactual dataset duplicates native data: {path}")
    for required in ("meta/info.json", "meta/tasks.jsonl", "meta/episodes.jsonl"):
        if not (path / required).is_file():
            raise SystemExit(f"Incomplete counterfactual dataset {path}: missing {required}")
    provenance_path = path / "meta/pgc_provenance.json"
    if not provenance_path.is_file():
        raise SystemExit(f"Missing PGC provenance: {provenance_path}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    suites = {str(item) for item in provenance.get("source_suites", [])}
    if not suites or not suites.issubset(expected_suites):
        raise SystemExit(
            f"Counterfactual dataset {path} leaks suites {sorted(suites)} "
            f"outside training scope {sorted(expected_suites)}"
        )
    covered_suites.update(suites)
    paths.append(str(path))
if not paths:
    raise SystemExit("The counterfactual dataset list contains no usable paths")
if len(set(paths)) != len(paths):
    raise SystemExit("The counterfactual dataset list contains duplicate paths")
if covered_suites != expected_suites:
    raise SystemExit(
        f"Counterfactual scope mismatch: covered={sorted(covered_suites)} "
        f"expected={sorted(expected_suites)}"
    )
print(json.dumps(paths, separators=(",", ":")))
PY
)"

"${PYTHON_BIN}" - "${BASE_CHECKPOINT}" <<'PY'
import sys
import torch

payload = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
if "mot" not in payload:
    raise SystemExit("PGC must initialize from a full released FastWAM checkpoint with `mot` weights")
if payload.get("format") in {"fastwam_lora_adapter_v1", "fastwam_policy_guard_v1"}:
    raise SystemExit("PGC base must be the immutable full FastWAM release, not an adapter")
print(f"Validated protected base: format={payload.get('format', 'legacy_full')} tensors={len(payload['mot'])}")
PY

MISSING_CACHE_COUNT="$(${PYTHON_BIN} - "${NATIVE_JSON}" "${CF_JSON}" "${CACHE_DIR}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

dataset_dirs = json.loads(sys.argv[1]) + json.loads(sys.argv[2])
cache_dir = Path(sys.argv[3])
template = "A video recorded from a robot's point of view executing the following instruction: {task}"
tasks = set()
for dataset_dir in dataset_dirs:
    with (Path(dataset_dir) / "meta/tasks.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                tasks.add(str(json.loads(line)["task"]))
missing = []
for task in sorted(tasks):
    prompt = template.format(task=task)
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    path = cache_dir / f"{digest}.t5_len128.wan22ti2v5b.pt"
    if not path.is_file():
        missing.append(str(path))
if missing:
    print("\n".join(missing), file=sys.stderr)
print(len(missing))
PY
)"
if (( MISSING_CACHE_COUNT > 0 )); then
  echo "Missing ${MISSING_CACHE_COUNT} text embedding caches under ${CACHE_DIR}." >&2
  echo "Precompute caches for every native and direct-counterfactual task first." >&2
  exit 1
fi

VISIBLE_GPU_COUNT="$(${PYTHON_BIN} -c 'import torch; print(torch.cuda.device_count())')"
if (( VISIBLE_GPU_COUNT < NPROC_PER_NODE )); then
  echo "Requested ${NPROC_PER_NODE} GPUs, PyTorch sees ${VISIBLE_GPU_COUNT}." >&2
  exit 1
fi

echo "[PGC-FastWAM] action-only LoRA training"
echo "  training_scope=${TRAIN_SUITE}"
echo "  protected_base=${BASE_CHECKPOINT}"
echo "  native_datasets=${NATIVE_JSON}"
echo "  direct_counterfactual_datasets=${CF_JSON}"
echo "  counterfactual_oversample=${OVERSAMPLE_FACTOR}"
echo "  seed=${TRAIN_SEED} max_steps=${MAX_STEPS} lr=${LEARNING_RATE}"
echo "  lora=action-only rank=${LORA_RANK} alpha=${LORA_ALPHA} dropout=${LORA_DROPOUT}"
echo "  save_training_state=${SAVE_TRAINING_STATE}"

RUN_ID="pgc-${RUN_TAG}" bash scripts/train_zero1.sh "${NPROC_PER_NODE}" \
  task=libero_pgc_2cam224 \
  "resume=${BASE_CHECKPOINT}" \
  "data.train.dataset_dirs=${NATIVE_JSON}" \
  "data.train.pgc_counterfactual_dataset_dirs=${CF_JSON}" \
  "data.train.pgc_counterfactual_oversample_factor=${OVERSAMPLE_FACTOR}" \
  "++data.train.pretrained_norm_stats=${STATS_PATH}" \
  "data.train.text_embedding_cache_dir=${CACHE_DIR}" \
  "seed=${TRAIN_SEED}" \
  "max_steps=${MAX_STEPS}" \
  num_epochs=1 \
  "learning_rate=${LEARNING_RATE}" \
  "save_every=${SAVE_EVERY}" \
  "save_training_state=${SAVE_TRAINING_STATE}" \
  model.action_dit_config.use_latent_action_queries=false \
  model.langforce_mvp.enabled=false \
  model.langforce_mvp.enable_prior=false \
  model.langforce_mvp.enable_posterior_advantage=false \
  model.transition_contract.enabled=false \
  model.policy_guard.enabled=true \
  model.policy_guard.version=1 \
  model.policy_guard.require_direct_counterfactual_actions=true \
  model.lora.enabled=true \
  "model.lora.rank=${LORA_RANK}" \
  "model.lora.alpha=${LORA_ALPHA}" \
  "model.lora.dropout=${LORA_DROPOUT}" \
  "model.lora.experts=[action]" \
  "model.lora.extra_trainable_patterns=[]"

echo "[PGC-FastWAM] training complete."
