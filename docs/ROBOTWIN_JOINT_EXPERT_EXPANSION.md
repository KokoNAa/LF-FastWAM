# Ten-task CF action and relation data expansion

The completed ten-task transfer screen found CF0 on all five added tasks.
For cup-front, the existing ERAF predicted `on` throughout all63 logged replans.
This expansion adds missing expert action and entity/relation supervision.
Its outcome is unmeasured until a trained checkpoint passes matched rollouts.
The current `20260907-1543-warmoff200-cf4` trial keeps its original frozen bank.

Collect `blocks_ranking_size`, `place_empty_cup`, `place_mouse_pad`,
`move_stapler_pad` and `move_pillbottle_pad` with the existing paired expert
collector and its explicit `--collection-profile joint_expert`. Plan16 accepted
scenes per task: first12 train, last4 whole-scene holdout. If a pilot uses smaller
counts, record those counts and use a separate output; do not call it16 scenes.
Both experts must plan and replay their respective goals from matched initial
states. This is ordinary expert data, not a policy-failure FG correction.

Use fresh reserved training seed starts84000000,84100000,84200000,84300000,
84400000 respectively. Do not read development or test outcomes to select scenes.
The preparer rejects seeds outside[84000000,85000000), parent-bank overlap,
missing paired episodes, invalid training provenance and changed initial states.
No existing capture is relabelled or overwritten.

Example for one task (set the actual fresh output and robotwin paths):

```sh
CUDA_VISIBLE_DEVICES=0 /opt/conda/bin/python scripts/collect_pgc_robotwin_pairs.py \
  --robotwin-root /root/gpufree-data/LF-FastWAM/third_party/RoboTwin \
  --output-root "$EXPANSION/raw/size" --task-config demo_clean \
  --episodes 16 --start-seed 84000000 --max-seed-attempts 100 \
  --collection-profile joint_expert --source-tasks blocks_ranking_size
```

After complete source/CF recordings, freeze a preparation `plan`, run its cache
`worker` shards, then `merge` with
`scripts/prepare_robotwin_joint_expert_replay.py`. Each mode takes the same
`--manifest`, `--checkpoint`, `--output`, `--collections` (pair-root paths),
`--train-per-task 12`, `--holdout-per-task 4`, `--stride 24`, `--shards N`.
Workers also take their `--shard` and physical `--gpu`. The plan records code,
source-file metadata, every scene split and the preparation checkpoint; changes
require a new output plan. Preparation needs complete original-bank metadata.

Cache only the frozen VAE/text/state inputs. Keep source and CF trajectories at
their own observations, cover each branch through its terminal frame, and mask
padding out of action losses. Only identical initial RGB/proprio observations
with two complete32-action windows qualify for extra paired-difference loss.
Explicit raw HDF5 paths and per-language frame indices feed the existing ERAF
grounding labels. Gold labels do not enter deployment observations or memory.

After preparation, preserve original-five retention and FG data while using
task-balanced expert sampling. Refresh ERAF semantics with the ten-task,
per-language holdout qualification added in commita0ff0f2; keep policy frozen
during semantic training. Evaluate action fine-tuning and matched controls in
a separate declared trial. Original-five scene gains and losses remain part of
selection; Correct is not a threshold. This data plan alone does not demonstrate
an ERAF or FG gain.

Launch collection only on verified idle GPUs after the running matrix releases
them, with the current work cutoff and adequate disk space. Preserve incomplete
records on failure; do not merge partial shards. Archive before server shutdown.
