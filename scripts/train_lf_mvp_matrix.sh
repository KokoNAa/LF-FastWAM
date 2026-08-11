#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${1:?Usage: bash scripts/train_lf_mvp_matrix.sh <gpus> <task> <checkpoint> <steps> [seed] [hydra_overrides...]}"
TASK_NAME="${2:?Missing Hydra task name}"
BASE_CHECKPOINT="${3:?Missing common starting checkpoint}"
TRAIN_STEPS="${4:?Missing controlled fine-tuning step count}"
TRAIN_SEED="${5:-42}"
RUN_TAG="${RUN_TAG:-$(date +%Y-%m-%d_%H-%M-%S)}"
EXTRA_OVERRIDES=()
if (( $# > 5 )); then
  EXTRA_OVERRIDES=("${@:6}")
fi

COMMON_OVERRIDES=(
  "task=${TASK_NAME}"
  "resume=${BASE_CHECKPOINT}"
  "max_steps=${TRAIN_STEPS}"
  "seed=${TRAIN_SEED}"
  "${EXTRA_OVERRIDES[@]}"
)

echo "[LF-FastWAM] B0 baseline controlled fine-tuning"
RUN_ID="lf-b0-${RUN_TAG}" bash scripts/train_zero1.sh "${NPROC_PER_NODE}" \
  "${COMMON_OVERRIDES[@]}" \
  "model.action_dit_config.use_latent_action_queries=false" \
  "model.langforce_mvp.enabled=false"

echo "[LF-FastWAM] B1 query bottleneck only"
RUN_ID="lf-b1-${RUN_TAG}" bash scripts/train_zero1.sh "${NPROC_PER_NODE}" \
  "${COMMON_OVERRIDES[@]}" \
  "model.action_dit_config.use_latent_action_queries=true" \
  "model.langforce_mvp.enabled=true" \
  "model.langforce_mvp.enable_prior=false" \
  "model.langforce_mvp.enable_posterior_advantage=false"

echo "[LF-FastWAM] M1 full query + prior + posterior-advantage MVP"
RUN_ID="lf-m1-${RUN_TAG}" bash scripts/train_zero1.sh "${NPROC_PER_NODE}" \
  "${COMMON_OVERRIDES[@]}" \
  "model.action_dit_config.use_latent_action_queries=true" \
  "model.langforce_mvp.enabled=true" \
  "model.langforce_mvp.enable_prior=true" \
  "model.langforce_mvp.enable_posterior_advantage=true"
