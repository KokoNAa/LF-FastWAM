# PGC-FastWAM V9.26 on RoboTwin

This path connects the current single-path ERAF + shared Video/Action Expert
LoRA method to RoboTwin without reusing LIBERO's 7-D Cartesian assumptions.

## Contracts

- Policy action and proprio are native RoboTwin dual-arm qpos, both 14-D.
- RGB is the existing 384x320 mosaic: 256x320 head view above two 128x160
  wrist views. ERAF labels this layout `robotwin_mosaic`; it is not treated as
  three equal horizontal slices.
- A deployable checkpoint must be PGC V9.26, single ERAF path, no post-sampler
  residual or candidate gate, and contain shared Video + Action Expert LoRA.
- Counterfactual expert data is collected inside the source task's initialized
  scene. Independently sampled left/right scenes are invalid pairs.
- SAPIEN actor IDs (stored losslessly at the 24x20 ERAF token geometry), entity
  positions, qpos actions, seed, initial-state hash and action hash are
  training-only audited data. Deployment remains RGB, language, proprio and
  caller-owned completion memory.

## Server collection smoke test

```bash
cd /root/gpufree-data/LF-FastWAM
python scripts/collect_pgc_robotwin_pairs.py \
  --robotwin-root third_party/RoboTwin \
  --output-root data/pgc_robotwin_raw \
  --task-config demo_clean \
  --episodes 2 \
  --source-tasks place_a2b_left place_a2b_right

python scripts/validate_pgc_robotwin_raw.py \
  data/pgc_robotwin_raw/place_a2b_left_to_place_a2b_right/native \
  data/pgc_robotwin_raw/place_a2b_left_to_place_a2b_right/counterfactual \
  data/pgc_robotwin_raw/place_a2b_right_to_place_a2b_left/native \
  data/pgc_robotwin_raw/place_a2b_right_to_place_a2b_left/counterfactual
```

After the two-episode gate passes, rerun collection with `--episodes 50`.

Convert all four captures to LeRobot and bind each ERAF sidecar to its exact
converted dataset path:

```bash
python scripts/prepare_pgc_robotwin_datasets.py \
  --raw-root data/pgc_robotwin_raw \
  --dataset-root data/pgc_robotwin_lerobot \
  --sidecar-root data/pgc_robotwin_eraf
```

The command is resumable only after a complete dataset+sidecar pair exists. It
rejects partial output and verifies that native/counterfactual initial-state
hash sequences match within each direction pair.

## Matched CIS evaluation

The evaluator reconstructs the exact checkpoint architecture and rejects a
LIBERO or incompatible checkpoint before allocating the model:

```bash
export FASTWAM_EVAL_MODE=PGC
export CIS_TASKS=place_a2b_left,place_a2b_right
export CIS_TASK_CONFIGS=demo_clean
export CIS_CONDITIONS=correct,shuffled,counterfactual
export RUN_TAG=robotwin_cis_pgc_v926_clean_50_seed42

bash scripts/eval_robotwin_cis.sh \
  6 50 10 42 "$ROBOTWIN_PGC_CKPT" "$STATS_PATH"
```

Each `episodes.jsonl` record includes compact per-replan ERAF diagnostics:
active clauses, predicates, three-view attention mass, entity/anchor positions,
predicate truth, phase, execution probability and completion-memory state.
