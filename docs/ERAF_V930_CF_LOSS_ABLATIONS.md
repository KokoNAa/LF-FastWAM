# V9.30 CF loss ablations: acquisition coverage versus language ranking

## What these experiments test

The audited corrective pool contains 30 episodes: 20 `target_lift` and 10
`counterfactual_goal`. This is the corrective pool, not the entire training set.
The acquisition prefixes are valid local action supervision; they are not
verified full-goal demonstrations. Full-goal verification also does not prove
instruction selectivity when source and counterfactual goals can both succeed.

These are diagnostics, not a claimed CF fix. In particular, the regressed
drawer-close pair already has five full-goal corrective episodes. A successful
ablation identifies a loss contribution worth investigating, not a unique cause
(removing a contribution also reduces gradient strength).

| Mode | Change relative to V9.30 |
| --- | --- |
| `none` | Unmodified control; optional fresh reproduction |
| `mask_lift_corrective` | A: zero action, non-regression and language-ranking contributions of audited lift-only corrective rows |
| `mask_corrective_ranking` | B: zero only language-ranking contributions of ALL corrective rows; retain their action/non-regression supervision |

Only one mode can be selected per run. Masks use the audited verification kind,
not suite/task IDs. V1 acquisition-only records are normalized after validation;
V2 missing/unknown kinds fail closed. Non-corrective rows have kind `0`, lift-only
rows `1`, full-goal rows `2`. The IDs are carried through the dataset and model
input builder; they do not enter inference.

## Matched controls

- Each arm starts independently from the V9.28 step10000 checkpoint recorded in
  the actual V9.30 control config. Never continue A from B, V9.29 or V9.30 step250.
- Clone the complete resolved V9.30 config: same native/historical-CF/strict-CF/
  corrective pools, 1:1:1:1 sample plan, seed, batch size, accumulation, learning
  rate, optimizer settings, two noise draws and preservation objective.
- In the current experiment: 250 optimizer steps per arm, LR 2e-6, save every50,
  batch size1, accumulation4, four training GPUs (effective batch16).
  The earlier three-GPU request was for evaluation; changing training world size
  changes the control. The runner prints its effective batch for inspection.
- Do not filter/resample data or skip forward passes. Apply multipliers to loss
  numerators while retaining ORIGINAL validity masks and mean denominators.
- Corrective rows remain excluded from V9.28 teacher preservation, exactly as in
  the control. Gate losses remain diagnostic only and the gate stays frozen.
- Only the existing four-token compressor and context injector are optimized.
  Complete ERAF, GoalGraph, base/LoRA and gain gate stay frozen. Deployment still
  has one Action Expert, no action blending and no inference teacher.
- An all-masked microbatch produces a differentiable zero loss. This does NOT
  mean AdamW parameters cannot move: momentum/weight decay still operate.

## Startup and provenance

`scripts/train_libero_eraf_cf_ablation.py` clones the actual control config,
reuses the existing checkpoint/sidecar/workspace/CUDA preflight, validates the
corrective index, then runs A and B sequentially. It refuses existing output
directories, records source-config/index/warm-checkpoint SHA256 and verification counts in
`experiment.json`, and checks final checkpoint step/mode before starting the
next arm. It rechecks warm-checkpoint/index hashes between arms. Failures stop
the sequence; original datasets/checkpoints are untouched.

The new `eraf_cf_ablation` and contract metadata are saved in checkpoints. The
standard evaluator validates and restores them. Historical V9.30 checkpoints
with neither field remain the unmasked control. Resuming a differently labelled
V9.30 checkpoint fails rather than silently changing an arm's identity.

## Complete server launch

Synchronize this change to the server first. The commands below do not imply
that code has already been committed or pushed. Run in a normal terminal; the
nested Bash confines a preflight failure to its subprocess.

```bash
bash <<'BASH'
set -euo pipefail
REPO=/root/gpufree-data/LF-FastWAM
cd "$REPO"
CFG="$REPO/runs/libero_eraf_safe_gain_v930_2cam224/eraf-safe-gain-v930-libero10-preserve-v928-250-seed42-20260831-152137/config.yaml"
test -f "$CFG" || { echo "Missing actual V9.30 control config: $CFG"; exit 1; }
test -f scripts/train_libero_eraf_cf_ablation.py || {
  echo 'Missing new ablation code: synchronize this change first.'; exit 1;
}
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
RUN_TAG="libero10-cf-loss-ablation-250-seed42-$(date +%Y%m%d-%H%M%S)"
ROOT="$REPO/runs/libero_eraf_cf_ablation/$RUN_TAG"
LOG="/root/gpufree-data/$RUN_TAG.log"
PID_FILE="/root/gpufree-data/$RUN_TAG.pid"

# Configuration-only inspection: no files written, no checkpoint/CUDA checks.
/opt/conda/bin/python scripts/train_libero_eraf_cf_ablation.py \
  "$CFG" 4 "$RUN_TAG" --dry-run

# Both arms run sequentially on the SAME four GPUs, with fresh optimizer state.
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 \
  DIFFSYNTH_MODEL_BASE_PATH=/root/gpufree-data/fastwam/FastWAM/checkpoints \
  HYDRA_FULL_ERROR=1 \
  /opt/conda/bin/python -u scripts/train_libero_eraf_cf_ablation.py \
  "$CFG" 4 "$RUN_TAG" >"$LOG" 2>&1 </dev/null &
TRAIN_PID=$!
printf '%s\n' "$TRAIN_PID" > "$PID_FILE"
printf 'ROOT=%s\nLOG=%s\nPID_FILE=%s\n' "$ROOT" "$LOG" "$PID_FILE"
sleep 5
if kill -0 "$TRAIN_PID" 2>/dev/null; then
  echo 'Launcher alive; confirm actual optimizer steps in the log.'
else
  echo 'Launcher exited; inspect the error below.'
fi
tail -80 "$LOG"
BASH
```

Monitor using the exact printed `LOG` path with `tail -F`. The single parent log
contains both arms' training output, delimited by `[TRAIN_START]` / `[TRAIN_DONE]`.
A live launcher alone is not proof training has passed preflight.

Output checkpoints, relative to the printed `ROOT`:

```text
mask_lift_corrective/checkpoints/weights/step_000250.pt
mask_corrective_ranking/checkpoints/weights/step_000250.pt
```

Optional `--modes none` runs a fresh control; it is not part of the default two
arms. Use a different run tag. `--modes mask_corrective_ranking` can run B alone
after inspecting a stopped A/B sequence, also under a new tag. Never overwrite a
partially completed experiment or silently resume its optimizer.

## Training diagnostics and evaluation

Inspect the valid-rate alongside loss metrics: a zero logged group loss may
simply mean that the microbatch contains no samples of that kind.

- `pgc_v930_cf_lift_valid_rate`, `pgc_v930_cf_goal_valid_rate`.
- `loss_pgc_v930_cf_{lift,goal}_{action,ranking,nonregression}_{raw,used}`.
- `pgc_v930_cf_action_kept_fraction`, `pgc_v930_cf_ranking_kept_fraction`.
- Existing teacher-proxy coverage and frozen-scope audits remain active.

A must have zero `lift_*_used` contributions; B must have zero corrective
`*_ranking_used`, with action/non-regression `used == raw`. The logged per-kind
means are diagnostic only, not additional terms in the objective.

After `[ALL_TRAIN_DONE]`, use the existing `scripts/eval_pgc_libero.sh` for each
step250 with:

- `PGC_GATE_MODE=counterfactual` (forced ERAF, not guarded/base).
- `PGC_EVAL_SUITES=[libero_10]`, ordinary manifest
  `/root/gpufree-data/pgc_libero_data_v1/manifests/libero_10_pgc.jsonl`.
- Five trials, seed42, ten inference steps, suite-default horizon, matching
  stats, no oracle/shadow/capture/state-memory ablations. Set diagnostics
  explicitly (`PGC_ERAF_DIAGNOSTICS=false` for the matched primary comparison).
- All10 tasks per arm: 20 subtasks /100 new CF episodes. Evaluation can use
  three GPUs without changing the per-episode protocol. It is NOT auto-started
  by the training runner.

Compare against the existing warm V9.28 (17/50) and V9.30 (21/50), including
paired LOST/GAINED counts and independent source/CF success flags. Do not score
co-success as failure or change the ordinary manifest. The historical no-ERAF
22/50 remains a separate reference; one episode is not evidence of a stable
performance difference. These reused episodes are diagnostic regression probes,
not a fresh generalization test; confirm any selected change on held-out trials.

CPU tests cover loss/gradient masks, unchanged denominators/forward counts,
full model teacher/checkpoint paths, historical compatibility and actual Hydra
override restoration. They do not substitute for a server CUDA/ZeRO smoke or
LIBERO rollout.

### Local verification (2026-08-31)

- 41 targeted tests passed: preservation/ablation masks and real CPU backward,
  checkpoint/evaluator contracts, config cloning/CLI dispatch, corrective-index
  audits, bidirectional-language and joint-training contracts.
- The input-label propagation, unchanged V9.29 sampler and V9.28 freeze/load
  checks also passed (three additional tests).
- The broader V9 ERAF suite still has three failures reproduced on unmodified
  commit `947d3c9`: the dataset audit fixture lacks `action_dim`, and the V9.23
  wrong-branch-gradient / V9.25 action-difference assertions fail. These unrelated
  behaviors were not changed. One pre-existing V8 error-message assertion was
  updated to accept the current replay-verification rejection wording.
- Tests used local CPU Torch 2.9.0, not the server's CUDA/ZeRO environment. GPU
  training and the 100 new CF evaluation episodes have not been run locally.
