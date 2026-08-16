#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${1:?Usage: bash scripts/train_pgc_libero.sh <gpus> <base_checkpoint> <counterfactual_datasets.txt> [seed] [max_steps]}"
BASE_CHECKPOINT="${2:?Missing released FastWAM base checkpoint}"
COUNTERFACTUAL_LIST="${3:?Missing direct-counterfactual dataset list file}"
TRAIN_SEED="${4:-42}"
MAX_STEPS="${5:-4000}"
TRAIN_SUITE="${PGC_TRAIN_SUITE:-all}"
ALLOW_JOINT_TRAINING="${PGC_ALLOW_JOINT_TRAINING:-false}"
INIT_CHECKPOINT="${PGC_INIT_CHECKPOINT:-${BASE_CHECKPOINT}}"
CONTINUE_FROM_STEP="${PGC_CONTINUE_FROM_STEP:-}"
PGC_VERSION="${PGC_VERSION:-2}"

PYTHON_BIN="${PYTHON_BIN:-python}"
LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-data/libero_mujoco3.3.2}"
STATS_PATH="${STATS_PATH:-${DIFFSYNTH_MODEL_BASE_PATH:-./checkpoints}/fastwam_release/libero_uncond_2cam224_dataset_stats.json}"
CACHE_DIR="${TEXT_CACHE_DIR:-data/text_embeds_cache/libero}"
OVERSAMPLE_FACTOR="${PGC_COUNTERFACTUAL_OVERSAMPLE_FACTOR:-1}"
BALANCE_NATIVE_COUNTERFACTUAL="${PGC_BALANCE_NATIVE_COUNTERFACTUAL:-true}"
LEARNING_RATE="${PGC_LEARNING_RATE:-1.0e-5}"
SAVE_EVERY="${PGC_SAVE_EVERY:-500}"
SAVE_TRAINING_STATE="${PGC_SAVE_TRAINING_STATE:-false}"
LORA_RANK="${PGC_LORA_RANK:-16}"
LORA_ALPHA="${PGC_LORA_ALPHA:-32}"
LORA_DROPOUT="${PGC_LORA_DROPOUT:-0.05}"
RESIDUAL_REG_WEIGHT="${PGC_RESIDUAL_REGULARIZATION_WEIGHT:-0.01}"
RESIDUAL_SMOOTHNESS_WEIGHT="${PGC_RESIDUAL_SMOOTHNESS_WEIGHT:-0.01}"
RESIDUAL_MAX_ABS="${PGC_VELOCITY_RESIDUAL_MAX_ABS:-1.0}"
ACTION_CHUNK_RESIDUAL_MAX_ABS="${PGC_ACTION_CHUNK_RESIDUAL_MAX_ABS:-2.0}"
ROLLOUT_INFERENCE_STEPS="${PGC_ROLLOUT_INFERENCE_STEPS:-10}"
ACTION_GRIPPER_WEIGHT="${PGC_ACTION_GRIPPER_WEIGHT:-2.0}"
ADVANTAGE_TEMPERATURE="${PGC_ADVANTAGE_TEMPERATURE:-0.25}"
ADVANTAGE_CLIP="${PGC_ADVANTAGE_CLIP:-4.0}"
CANDIDATE_MAX_SATURATION_FRACTION="${PGC_CANDIDATE_MAX_SATURATION_FRACTION:-0.25}"
CANDIDATE_MAX_DELTA_RMS="${PGC_CANDIDATE_MAX_DELTA_RMS:-2.0}"
TRAIN_GATE_THRESHOLD="${PGC_TRAIN_GATE_THRESHOLD:-0.20}"
VERIFIER_START_STEP="${PGC_VERIFIER_START_STEP:-1000}"
VERIFIER_RAMP_STEPS="${PGC_VERIFIER_RAMP_STEPS:-500}"
EXECUTION_PREFIX_STEPS="${PGC_EXECUTION_PREFIX_STEPS:-10}"
SUFFIX_LOSS_WEIGHT="${PGC_SUFFIX_LOSS_WEIGHT:-0.10}"
SAME_STATE_SOURCE_ZERO_WEIGHT="${PGC_SAME_STATE_SOURCE_ZERO_WEIGHT:-1.0}"
GOAL_SEPARATION_WEIGHT="${PGC_GOAL_SEPARATION_WEIGHT:-0.25}"
GOAL_SEPARATION_MARGIN="${PGC_GOAL_SEPARATION_MARGIN:-0.20}"
RESIDUAL_SEPARATION_WEIGHT="${PGC_RESIDUAL_SEPARATION_WEIGHT:-0.25}"
RESIDUAL_SEPARATION_MARGIN="${PGC_RESIDUAL_SEPARATION_MARGIN:-0.05}"
VERIFIER_WRONG_LANGUAGE_WEIGHT="${PGC_VERIFIER_WRONG_LANGUAGE_WEIGHT:-0.50}"
VERIFIER_BAD_CANDIDATE_WEIGHT="${PGC_VERIFIER_BAD_CANDIDATE_WEIGHT:-0.50}"
RUN_TAG="${RUN_TAG:-${TRAIN_SUITE}-pgc-v${PGC_VERSION}-${MAX_STEPS}-seed${TRAIN_SEED}}"

case "${PGC_VERSION}" in
  2|3|4|5) ;;
  *)
    echo "PGC_VERSION must be 2, 3, 4, or 5; got ${PGC_VERSION}." >&2
    exit 1
    ;;
esac

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

for value_name in NPROC_PER_NODE MAX_STEPS OVERSAMPLE_FACTOR SAVE_EVERY ROLLOUT_INFERENCE_STEPS EXECUTION_PREFIX_STEPS; do
  value="${!value_name}"
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${value_name} must be a positive integer, got ${value}." >&2
    exit 1
  fi
done
if [[ "${PGC_VERSION}" == "2" ]] && ! [[ "${LORA_RANK}" =~ ^[1-9][0-9]*$ ]]; then
  echo "LORA_RANK must be a positive integer for PGC v2, got ${LORA_RANK}." >&2
  exit 1
fi
for value_name in VERIFIER_START_STEP VERIFIER_RAMP_STEPS; do
  value="${!value_name}"
  if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
    echo "${value_name} must be a non-negative integer, got ${value}." >&2
    exit 1
  fi
done
if [[ -n "${CONTINUE_FROM_STEP}" ]]; then
  if ! [[ "${CONTINUE_FROM_STEP}" =~ ^[1-9][0-9]*$ ]]; then
    echo "PGC_CONTINUE_FROM_STEP must be a positive integer when set." >&2
    exit 1
  fi
  if (( CONTINUE_FROM_STEP >= MAX_STEPS )); then
    echo "PGC continuation requires MAX_STEPS > PGC_CONTINUE_FROM_STEP; got ${MAX_STEPS} <= ${CONTINUE_FROM_STEP}." >&2
    exit 1
  fi
fi
case "${BALANCE_NATIVE_COUNTERFACTUAL}" in
  true|false) ;;
  *)
    echo "PGC_BALANCE_NATIVE_COUNTERFACTUAL must be true or false." >&2
    exit 1
    ;;
esac
if [[ "${BALANCE_NATIVE_COUNTERFACTUAL}" == "true" && "${OVERSAMPLE_FACTOR}" != "1" ]]; then
  echo "PGC exact 1:1 balancing requires PGC_COUNTERFACTUAL_OVERSAMPLE_FACTOR=1." >&2
  exit 1
fi
"${PYTHON_BIN}" - \
  "${PGC_VERSION}" \
  "${LORA_ALPHA}" \
  "${LORA_DROPOUT}" \
  "${RESIDUAL_REG_WEIGHT}" \
  "${RESIDUAL_SMOOTHNESS_WEIGHT}" \
  "${RESIDUAL_MAX_ABS}" \
  "${ACTION_CHUNK_RESIDUAL_MAX_ABS}" \
  "${ACTION_GRIPPER_WEIGHT}" \
  "${ADVANTAGE_TEMPERATURE}" \
  "${ADVANTAGE_CLIP}" \
  "${CANDIDATE_MAX_SATURATION_FRACTION}" \
  "${CANDIDATE_MAX_DELTA_RMS}" \
  "${TRAIN_GATE_THRESHOLD}" \
  "${SUFFIX_LOSS_WEIGHT}" \
  "${SAME_STATE_SOURCE_ZERO_WEIGHT}" \
  "${GOAL_SEPARATION_WEIGHT}" \
  "${GOAL_SEPARATION_MARGIN}" \
  "${RESIDUAL_SEPARATION_WEIGHT}" \
  "${RESIDUAL_SEPARATION_MARGIN}" \
  "${VERIFIER_WRONG_LANGUAGE_WEIGHT}" \
  "${VERIFIER_BAD_CANDIDATE_WEIGHT}" <<'PY'
import sys

version = int(sys.argv[1])
alpha = float(sys.argv[2])
dropout = float(sys.argv[3])
regularization = float(sys.argv[4])
smoothness = float(sys.argv[5])
cap = float(sys.argv[6])
action_chunk_cap = float(sys.argv[7])
gripper_weight = float(sys.argv[8])
advantage_temperature = float(sys.argv[9])
advantage_clip = float(sys.argv[10])
candidate_max_saturation = float(sys.argv[11])
candidate_max_delta_rms = float(sys.argv[12])
gate_threshold = float(sys.argv[13])
suffix_weight = float(sys.argv[14])
paired_weights = [float(value) for value in sys.argv[15:22]]
if version == 2:
    if alpha <= 0:
        raise SystemExit(f"PGC_LORA_ALPHA must be positive, got {alpha}")
    if not 0 <= dropout < 1:
        raise SystemExit(f"PGC_LORA_DROPOUT must be in [0, 1), got {dropout}")
if regularization < 0 or smoothness < 0:
    raise SystemExit("PGC residual regularization weights must be non-negative")
if cap <= 0:
    raise SystemExit("PGC v3 velocity residual cap must be positive")
if gate_threshold < 0:
    raise SystemExit("PGC training gate threshold must be non-negative")
if version >= 4:
    if action_chunk_cap <= 0:
        raise SystemExit("PGC v4 final-action residual cap must be positive")
    if gripper_weight <= 0:
        raise SystemExit("PGC v4 action gripper weight must be positive")
    if advantage_temperature <= 0 or advantage_clip <= 0:
        raise SystemExit("PGC v4 advantage temperature/clip must be positive")
    if not 0 <= candidate_max_saturation <= 1:
        raise SystemExit(
            "PGC v4 candidate saturation fraction must be in [0, 1]"
        )
    if candidate_max_delta_rms <= 0:
        raise SystemExit("PGC v4 candidate max delta RMS must be positive")
if version >= 5:
    if not 0 <= suffix_weight <= 1:
        raise SystemExit("PGC v5 suffix loss weight must be in [0, 1]")
    if any(value < 0 for value in paired_weights):
        raise SystemExit("PGC v5 paired-language weights must be non-negative")
PY
if [[ ! -f "${BASE_CHECKPOINT}" ]]; then
  echo "Base checkpoint not found: ${BASE_CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -f "${INIT_CHECKPOINT}" ]]; then
  echo "PGC initialization checkpoint not found: ${INIT_CHECKPOINT}" >&2
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
if payload.get("format") in {
    "fastwam_lora_adapter_v1",
    "fastwam_policy_guard_v1",
    "fastwam_policy_guard_v2",
    "fastwam_policy_guard_v3",
    "fastwam_policy_guard_v4",
    "fastwam_policy_guard_v5",
}:
    raise SystemExit("PGC base must be the immutable full FastWAM release, not an adapter")
print(f"Validated protected base: format={payload.get('format', 'legacy_full')} tensors={len(payload['mot'])}")
PY

WEIGHT_ONLY_START_STEP=null
if [[ -n "${CONTINUE_FROM_STEP}" ]]; then
  "${PYTHON_BIN}" - \
    "${INIT_CHECKPOINT}" \
    "${BASE_CHECKPOINT}" \
    "${CONTINUE_FROM_STEP}" \
    "${PGC_VERSION}" \
    "${LORA_RANK}" \
    "${LORA_ALPHA}" \
    "${LORA_DROPOUT}" \
    "${RESIDUAL_REG_WEIGHT}" \
    "${RESIDUAL_SMOOTHNESS_WEIGHT}" \
    "${RESIDUAL_MAX_ABS}" \
    "${ACTION_CHUNK_RESIDUAL_MAX_ABS}" \
    "${ROLLOUT_INFERENCE_STEPS}" \
    "${ACTION_GRIPPER_WEIGHT}" \
    "${ADVANTAGE_TEMPERATURE}" \
    "${ADVANTAGE_CLIP}" \
    "${CANDIDATE_MAX_SATURATION_FRACTION}" \
    "${CANDIDATE_MAX_DELTA_RMS}" \
    "${TRAIN_GATE_THRESHOLD}" \
    "${VERIFIER_START_STEP}" \
    "${VERIFIER_RAMP_STEPS}" \
    "${EXECUTION_PREFIX_STEPS}" \
    "${SUFFIX_LOSS_WEIGHT}" \
    "${SAME_STATE_SOURCE_ZERO_WEIGHT}" \
    "${GOAL_SEPARATION_WEIGHT}" \
    "${GOAL_SEPARATION_MARGIN}" \
    "${RESIDUAL_SEPARATION_WEIGHT}" \
    "${RESIDUAL_SEPARATION_MARGIN}" \
    "${VERIFIER_WRONG_LANGUAGE_WEIGHT}" \
    "${VERIFIER_BAD_CANDIDATE_WEIGHT}" <<'PY'
import math
import sys
from pathlib import Path

import torch

init_path = Path(sys.argv[1]).expanduser().resolve()
base_path = Path(sys.argv[2]).expanduser().resolve()
expected_step = int(sys.argv[3])
expected_version = int(sys.argv[4])
expected_rank = int(sys.argv[5])
expected_alpha = float(sys.argv[6])
expected_dropout = float(sys.argv[7])
expected_residual_regularization = float(sys.argv[8])
expected_residual_smoothness = float(sys.argv[9])
expected_residual_cap = float(sys.argv[10])
expected_action_chunk_cap = float(sys.argv[11])
expected_rollout_steps = int(sys.argv[12])
expected_gripper_weight = float(sys.argv[13])
expected_advantage_temperature = float(sys.argv[14])
expected_advantage_clip = float(sys.argv[15])
expected_candidate_max_saturation = float(sys.argv[16])
expected_candidate_max_delta_rms = float(sys.argv[17])
expected_gate_threshold = float(sys.argv[18])
expected_verifier_start = int(sys.argv[19])
expected_verifier_ramp = int(sys.argv[20])
expected_execution_prefix = int(sys.argv[21])
expected_v5_scalars = {
    "suffix_loss_weight": float(sys.argv[22]),
    "same_state_source_zero_weight": float(sys.argv[23]),
    "goal_separation_weight": float(sys.argv[24]),
    "goal_separation_margin": float(sys.argv[25]),
    "residual_separation_weight": float(sys.argv[26]),
    "residual_separation_margin": float(sys.argv[27]),
    "verifier_wrong_language_weight": float(sys.argv[28]),
    "verifier_bad_candidate_weight": float(sys.argv[29]),
}
payload = torch.load(init_path, map_location="cpu", weights_only=False)
metadata = payload.get("architecture_metadata") or {}
expected_format = f"fastwam_policy_guard_v{expected_version}"
if payload.get("format") != expected_format:
    raise SystemExit(
        f"PGC weight-only continuation requires a {expected_format} "
        f"checkpoint, got {payload.get('format')!r}"
    )
if metadata.get("architecture") != "pgc_fastwam" or int(
    metadata.get("policy_guard_version", -1)
) != expected_version:
    raise SystemExit("PGC continuation checkpoint has incompatible architecture metadata")
if int(payload.get("step", -1)) != expected_step:
    raise SystemExit(
        "PGC continuation step mismatch: "
        f"checkpoint={payload.get('step')} requested={expected_step}"
    )
recorded_base = Path(str(payload.get("base_checkpoint", ""))).expanduser()
if not recorded_base.is_absolute():
    recorded_base = init_path.parent / recorded_base
recorded_base = recorded_base.resolve()
if recorded_base != base_path:
    raise SystemExit(
        "PGC continuation base mismatch: "
        f"checkpoint={recorded_base} requested={base_path}"
    )
actual_gate_threshold = float(metadata.get("gate_threshold", float("nan")))
if not math.isclose(
    actual_gate_threshold,
    expected_gate_threshold,
    rel_tol=0.0,
    abs_tol=1.0e-12,
):
    raise SystemExit(
        "PGC continuation gate threshold mismatch: "
        f"checkpoint={actual_gate_threshold} requested={expected_gate_threshold}"
    )
if expected_version == 2:
    lora = payload.get("counterfactual_lora_config") or {}
    if int(lora.get("rank", -1)) != expected_rank:
        raise SystemExit(
            f"PGC continuation LoRA rank mismatch: checkpoint={lora.get('rank')} "
            f"requested={expected_rank}"
        )
    for name, expected in (
        ("alpha", expected_alpha),
        ("dropout", expected_dropout),
    ):
        try:
            actual = float(lora[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(f"PGC continuation checkpoint has invalid LoRA {name}") from exc
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12):
            raise SystemExit(
                f"PGC continuation LoRA {name} mismatch: "
                f"checkpoint={actual} requested={expected}"
            )
elif expected_version == 3:
    if metadata.get("counterfactual_tuning") != "bounded_velocity_residual":
        raise SystemExit("PGC v3 continuation lacks bounded residual metadata")
    if payload.get("counterfactual_action_adapter") is not None:
        raise SystemExit("PGC v3 continuation unexpectedly contains Action-Expert LoRA")
    for name, expected in (
        ("residual_regularization_weight", expected_residual_regularization),
        ("residual_smoothness_weight", expected_residual_smoothness),
    ):
        actual = float(metadata.get(name, float("nan")))
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12):
            raise SystemExit(
                f"PGC v3 continuation {name} mismatch: "
                f"checkpoint={actual} requested={expected}"
            )
    caps = [float(value) for value in metadata.get("velocity_residual_max_abs", [])]
    if not caps or any(
        not math.isclose(value, expected_residual_cap, rel_tol=0.0, abs_tol=1.0e-6)
        for value in caps
    ):
        raise SystemExit(
            "PGC v3 continuation residual cap mismatch: "
            f"checkpoint={caps} requested={expected_residual_cap}"
        )
    if int(metadata.get("verifier_start_step", -1)) != expected_verifier_start:
        raise SystemExit("PGC v3 continuation verifier_start_step mismatch")
    if int(metadata.get("verifier_ramp_steps", -1)) != expected_verifier_ramp:
        raise SystemExit("PGC v3 continuation verifier_ramp_steps mismatch")
else:
    expected_tuning = (
        "paired_language_prefix_aligned_action_residual"
        if expected_version >= 5
        else "rollout_aligned_final_action_residual"
    )
    if metadata.get("counterfactual_tuning") != expected_tuning:
        raise SystemExit(
            f"PGC v{expected_version} continuation lacks compatible "
            "rollout/paired-language metadata"
        )
    if any(
        key in payload
        for key in (
            "counterfactual_action_adapter",
            "counterfactual_action_expert",
            "counterfactual_lora_config",
        )
    ):
        raise SystemExit("PGC v4 continuation unexpectedly contains Action-Expert tensors")
    for name, expected in (
        ("residual_regularization_weight", expected_residual_regularization),
        ("residual_smoothness_weight", expected_residual_smoothness),
        ("action_gripper_weight", expected_gripper_weight),
        ("advantage_temperature", expected_advantage_temperature),
        ("advantage_clip", expected_advantage_clip),
        ("candidate_max_saturation_fraction", expected_candidate_max_saturation),
        ("candidate_max_delta_rms", expected_candidate_max_delta_rms),
    ):
        actual = float(metadata.get(name, float("nan")))
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12):
            raise SystemExit(
                f"PGC v4 continuation {name} mismatch: "
                f"checkpoint={actual} requested={expected}"
            )
    caps = [
        float(value)
        for value in metadata.get("action_chunk_residual_max_abs", [])
    ]
    if not caps or any(
        not math.isclose(
            value,
            expected_action_chunk_cap,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        )
        for value in caps
    ):
        raise SystemExit(
            "PGC v4 continuation action-chunk residual cap mismatch: "
            f"checkpoint={caps} requested={expected_action_chunk_cap}"
        )
    if int(metadata.get("rollout_num_inference_steps", -1)) != expected_rollout_steps:
        raise SystemExit("PGC v4 continuation rollout inference-step mismatch")
    if int(metadata.get("verifier_start_step", -1)) != expected_verifier_start:
        raise SystemExit("PGC v4 continuation verifier_start_step mismatch")
    if int(metadata.get("verifier_ramp_steps", -1)) != expected_verifier_ramp:
        raise SystemExit("PGC v4 continuation verifier_ramp_steps mismatch")
    if expected_version >= 5:
        if int(metadata.get("execution_prefix_steps", -1)) != expected_execution_prefix:
            raise SystemExit("PGC v5 continuation execution_prefix_steps mismatch")
        for name, expected in expected_v5_scalars.items():
            actual = float(metadata.get(name, float("nan")))
            if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12):
                raise SystemExit(
                    f"PGC v5 continuation {name} mismatch: "
                    f"checkpoint={actual} requested={expected}"
                )
print(
    "Validated PGC weight-only continuation: "
    f"step={expected_step} base={base_path}"
)
PY
  WEIGHT_ONLY_START_STEP="${CONTINUE_FROM_STEP}"
elif [[ "$(cd -- "$(dirname -- "${INIT_CHECKPOINT}")" && pwd -P)/$(basename -- "${INIT_CHECKPOINT}")" != "$(cd -- "$(dirname -- "${BASE_CHECKPOINT}")" && pwd -P)/$(basename -- "${BASE_CHECKPOINT}")" ]]; then
  echo "PGC_INIT_CHECKPOINT differs from BASE_CHECKPOINT but PGC_CONTINUE_FROM_STEP is unset." >&2
  exit 1
fi

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
    dataset_path = Path(dataset_dir)
    with (dataset_path / "meta/tasks.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                tasks.add(str(json.loads(line)["task"]))
    provenance_path = dataset_path / "meta/pgc_provenance.json"
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        for pair in provenance.get("pairs") or []:
            tasks.add(str(pair["source_instruction"]))
            tasks.add(str(pair["counterfactual_instruction"]))
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

echo "[PGC-FastWAM] version=${PGC_VERSION} scope-aware training"
echo "  training_scope=${TRAIN_SUITE}"
echo "  protected_base=${BASE_CHECKPOINT}"
echo "  initialization_checkpoint=${INIT_CHECKPOINT} weight_only_start_step=${WEIGHT_ONLY_START_STEP}"
echo "  native_datasets=${NATIVE_JSON}"
echo "  direct_counterfactual_datasets=${CF_JSON}"
echo "  counterfactual_oversample=${OVERSAMPLE_FACTOR} balanced_1to1=${BALANCE_NATIVE_COUNTERFACTUAL}"
echo "  seed=${TRAIN_SEED} max_steps=${MAX_STEPS} lr=${LEARNING_RATE}"
if [[ "${PGC_VERSION}" == "2" ]]; then
  echo "  tuning=action-only-lora rank=${LORA_RANK} alpha=${LORA_ALPHA} dropout=${LORA_DROPOUT}"
  LORA_ENABLED=true
elif [[ "${PGC_VERSION}" == "3" ]]; then
  echo "  tuning=bounded-velocity-residual cap=${RESIDUAL_MAX_ABS} regularization=${RESIDUAL_REG_WEIGHT} smoothness=${RESIDUAL_SMOOTHNESS_WEIGHT}"
  echo "  verifier_schedule=start:${VERIFIER_START_STEP} ramp:${VERIFIER_RAMP_STEPS}"
  LORA_ENABLED=false
else
  echo "  tuning=rollout-aligned-final-action-residual cap=${ACTION_CHUNK_RESIDUAL_MAX_ABS} rollout_steps=${ROLLOUT_INFERENCE_STEPS} gripper_weight=${ACTION_GRIPPER_WEIGHT}"
  echo "  advantage=temperature:${ADVANTAGE_TEMPERATURE} clip:${ADVANTAGE_CLIP} gate_threshold:${TRAIN_GATE_THRESHOLD}"
  echo "  candidate_support=max_saturation:${CANDIDATE_MAX_SATURATION_FRACTION} max_delta_rms:${CANDIDATE_MAX_DELTA_RMS}"
  if [[ "${PGC_VERSION}" == "5" ]]; then
    echo "  paired_language=source_zero:${SAME_STATE_SOURCE_ZERO_WEIGHT} goal_sep:${GOAL_SEPARATION_WEIGHT}/${GOAL_SEPARATION_MARGIN} residual_sep:${RESIDUAL_SEPARATION_WEIGHT}/${RESIDUAL_SEPARATION_MARGIN}"
    echo "  executed_prefix=${EXECUTION_PREFIX_STEPS} suffix_weight=${SUFFIX_LOSS_WEIGHT} verifier_negatives=wrong_language:${VERIFIER_WRONG_LANGUAGE_WEIGHT} bad_candidate:${VERIFIER_BAD_CANDIDATE_WEIGHT}"
  fi
  echo "  verifier_schedule=start:${VERIFIER_START_STEP} ramp:${VERIFIER_RAMP_STEPS}"
  LORA_ENABLED=false
fi
echo "  save_training_state=${SAVE_TRAINING_STATE}"

RUN_ID="pgc-${RUN_TAG}" bash scripts/train_zero1.sh "${NPROC_PER_NODE}" \
  task=libero_pgc_2cam224 \
  "resume=${INIT_CHECKPOINT}" \
  "weight_only_start_step=${WEIGHT_ONLY_START_STEP}" \
  "data.train.dataset_dirs=${NATIVE_JSON}" \
  "data.train.pgc_counterfactual_dataset_dirs=${CF_JSON}" \
  "data.train.pgc_counterfactual_oversample_factor=${OVERSAMPLE_FACTOR}" \
  "data.train.pgc_balance_native_counterfactual=${BALANCE_NATIVE_COUNTERFACTUAL}" \
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
  "model.policy_guard.version=${PGC_VERSION}" \
  model.policy_guard.require_direct_counterfactual_actions=true \
  "model.policy_guard.residual_regularization_weight=${RESIDUAL_REG_WEIGHT}" \
  "model.policy_guard.residual_smoothness_weight=${RESIDUAL_SMOOTHNESS_WEIGHT}" \
  "model.policy_guard.velocity_residual_max_abs=${RESIDUAL_MAX_ABS}" \
  "model.policy_guard.action_chunk_residual_max_abs=${ACTION_CHUNK_RESIDUAL_MAX_ABS}" \
  "model.policy_guard.rollout_num_inference_steps=${ROLLOUT_INFERENCE_STEPS}" \
  "model.policy_guard.action_gripper_weight=${ACTION_GRIPPER_WEIGHT}" \
  "model.policy_guard.advantage_temperature=${ADVANTAGE_TEMPERATURE}" \
  "model.policy_guard.advantage_clip=${ADVANTAGE_CLIP}" \
  "model.policy_guard.candidate_max_saturation_fraction=${CANDIDATE_MAX_SATURATION_FRACTION}" \
  "model.policy_guard.candidate_max_delta_rms=${CANDIDATE_MAX_DELTA_RMS}" \
  "model.policy_guard.gate_threshold=${TRAIN_GATE_THRESHOLD}" \
  "model.policy_guard.verifier_start_step=${VERIFIER_START_STEP}" \
  "model.policy_guard.verifier_ramp_steps=${VERIFIER_RAMP_STEPS}" \
  "model.policy_guard.execution_prefix_steps=${EXECUTION_PREFIX_STEPS}" \
  "model.policy_guard.suffix_loss_weight=${SUFFIX_LOSS_WEIGHT}" \
  "model.policy_guard.same_state_source_zero_weight=${SAME_STATE_SOURCE_ZERO_WEIGHT}" \
  "model.policy_guard.goal_separation_weight=${GOAL_SEPARATION_WEIGHT}" \
  "model.policy_guard.goal_separation_margin=${GOAL_SEPARATION_MARGIN}" \
  "model.policy_guard.residual_separation_weight=${RESIDUAL_SEPARATION_WEIGHT}" \
  "model.policy_guard.residual_separation_margin=${RESIDUAL_SEPARATION_MARGIN}" \
  "model.policy_guard.verifier_wrong_language_weight=${VERIFIER_WRONG_LANGUAGE_WEIGHT}" \
  "model.policy_guard.verifier_bad_candidate_weight=${VERIFIER_BAD_CANDIDATE_WEIGHT}" \
  "model.lora.enabled=${LORA_ENABLED}" \
  "model.lora.rank=${LORA_RANK}" \
  "model.lora.alpha=${LORA_ALPHA}" \
  "model.lora.dropout=${LORA_DROPOUT}" \
  "model.lora.experts=[action]" \
  "model.lora.extra_trainable_patterns=[]"

echo "[PGC-FastWAM] training complete."
