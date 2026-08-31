# V9.30: V9.28 interface-preservation probe

## Decision and scope

Initialize from **V9.28 safe-gain step_010000**, not the old ERAF-unfrozen joint
checkpoint and not a trained V9.29 checkpoint. The zero-update comparison
preserved V9.28 behavior; degradation was already visible at V9.29 step500.
This motivates an early preservation probe, but does not by itself identify
whether data distribution, learning rate or the objective caused the regression.

- Frozen: released base, Video/Action LoRA, complete ERAF, GoalGraph and gain gate.
- Optimized: existing four-token compressor and context injector only.
- Deployment: still one Action Expert, no action blending and no teacher forward.
- Training teacher: copies only V9.28's compressor and injector; uses the same
  frozen ERAF queries, video cache and Action Expert as the student.
- Default: **250 optimizer steps**, LR **2e-6**, save every **50**. The launcher
  refuses budgets above500; do not extend before looking at closed-loop results.
- Data, bidirectional language ranking, two-noise supervision, base protection,
  closed-loop corrective weighting and sampling remain the V9.29 contract.
  The new launcher reuses its actual resolved dataset/sidecar paths.
- Gate is absent from optimizer groups, not merely in eval mode. Its **weights**
  stay fixed, but its outputs can still change when compressed features change.
  This is not gate recalibration, and guarded results are not the primary probe.

## Added objective and its limitation

For each of the same two noisy action/timestep draws, predict the fixed old
augmented policy's action flow with no gradient. On non-corrective valid rows,
retain the teacher only when its masked action MSE against the demonstration is
no worse than the base's MSE. Minimize masked student–teacher action-flow MSE on
these rows. Padding and empty masks are handled explicitly.

`L = L_V929_injector + preservation_weight * mean_noise(L_teacher)`

Default `preservation_weight=1.0`. There are no V9.28 closed-loop-success labels
in the current training data. Eligibility is an **offline error proxy**, not a
known-success mask or proof of deployment safety. Corrective rows are excluded
from distillation because they are intended to repair old behavior. Monitor
`pgc_v930_teacher_proxy_coverage`; near-zero coverage means the constraint is
mostly inactive. Small training loss alone is not an acceptance criterion.

An optional matched ablation sets `ERAF_PRESERVATION_WEIGHT=0.0`, keeping the
same V9.28 initialization, LR, frozen gate, data and short budget. Run sequentially
with a distinct `RUN_TAG`. This separates the preservation term from the lower
LR; comparing V9.30 only against historical V9.29 cannot isolate that effect.

## Checkpoint and freeze guarantees checked by code

The V9.28 checkpoint path, SHA256, objective, step and teacher tensor checksum
are persisted with the teacher. A V9.30 restore uses the saved old teacher,
never a fresh snapshot of the updated student. Missing/corrupt teachers,
V9.29 warm starts and non-step10000 V9.28 sources fail closed.

Exact trainable/optimizer parameter identities are checked. Frozen sidecar,
LoRA and teacher hashes are checked after load, at steps0/1, every50 steps and
before saving. The full released base is excluded from optimization but is not
copied to CPU for each hash. These checks complement, but do not replace, a
server-side ZeRO smoke run. No GPU/ZeRO or LIBERO run was performed locally.

## Complete server launch

First transfer/merge these code changes to the server; this document does not
imply they have been pushed. Run from a normal terminal, not by sourcing a script.
The nested Bash keeps a preflight failure from closing the interactive shell.
This starts a **new short run** with fresh optimizer/scheduler, not an exact
optimizer-state continuation. It neither stops another run nor overwrites it.

```bash
bash <<'BASH'
set -euo pipefail
REPO=/root/gpufree-data/LF-FastWAM
cd "$REPO"
CFG="$REPO/runs/libero_eraf_safe_gain_v929_2cam224/eraf-safe-gain-v929-libero10-v929-rollout-repair-10k-seed42-20260830-163813/config.yaml"
test -f "$CFG" || { echo "Missing actual V9.29 run-root config: $CFG"; exit 1; }
test -f scripts/train_libero_eraf_safe_gain_v930_from_config.py
RUN_TAG="libero10-preserve-v928-250-seed42-$(date +%Y%m%d-%H%M%S)"
LOG="/root/gpufree-data/${RUN_TAG}.log"
PID_FILE="/root/gpufree-data/${RUN_TAG}.pid"
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 \
  PYTHON_BIN=/opt/conda/bin/python \
  PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}" \
  DIFFSYNTH_MODEL_BASE_PATH=/root/gpufree-data/fastwam/FastWAM/checkpoints \
  ERAF_SAFE_GAIN_MAX_STEPS=250 ERAF_SAFE_GAIN_SAVE_EVERY=50 \
  ERAF_SAFE_GAIN_LEARNING_RATE=2.0e-6 ERAF_PRESERVATION_WEIGHT=1.0 \
  RUN_TAG="$RUN_TAG" \
  /opt/conda/bin/python -u scripts/train_libero_eraf_safe_gain_v930_from_config.py \
  "$CFG" 4 >"$LOG" 2>&1 </dev/null &
TRAIN_PID=$!
printf '%s\n' "$TRAIN_PID" > "$PID_FILE"
printf 'LOG=%s\nPID_FILE=%s\n' "$LOG" "$PID_FILE"
sleep 5
if kill -0 "$TRAIN_PID" 2>/dev/null; then
  echo 'Launcher alive; confirm optimizer steps in the log.'
else
  echo 'Launcher exited; inspect the error below.'
fi
tail -80 "$LOG"
BASH
```

If the exact source config path differs, locate the actual **run-root/config.yaml**
and replace `CFG`; do not invent `.hydra/config.yaml` paths or auto-select an
arbitrary latest run. The helper takes the **V9.28 resume path recorded in that
V9.29 config**, and the shared preflight validates it and every sidecar binding.

Monitor with the printed absolute log path:

```bash
tail -F /root/gpufree-data/REPLACE_WITH_PRINTED_RUN_TAG.log
```

## Acceptance sequence

1. Check step0/first update: finite differentiable loss, fixed teacher provenance,
   only two optimizer groups, nonzero proxy coverage where possible, and no
   frozen-tensor audit errors. Teacher-preservation loss starts at approximately
   zero because student and teacher initialize identically.
2. At steps50/100/250, run **forced ERAF** with seed42/trials5 on correct task5 and
   ordinary-counterfactual task9 using the existing comparison workflow.
   Compare paired trial IDs to the V9.28 warm results (5/5 and2/5 respectively).
   These are sentinel regression checks, not enough to claim overall gains.
3. If preserved, evaluate all10 tasks for correct + ordinary CF with the same
   manifest/init states. The existing `eval_pgc_libero.sh` accepts objective30 and
   reconstructs its schema and checkpoint contracts. Keep forced/base/guarded
   conditions separate; do not report a fallback-only result as ERAF improvement.
4. Do not tune gate thresholds on final test episodes. Only consider independent
   gate calibration after an augmented path with demonstrated gains exists.
