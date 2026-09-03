# RoboTwin matched DTL/CIS evaluation

This pipeline evaluates whether FastWAM follows a changed instruction in the
same RoboTwin scene.  It does not treat a different task reset as a
counterfactual scene.

## Protocol

The checked-in manifest uses the native bidirectional task pair:

```text
place_a2b_left  <->  place_a2b_right
```

These tasks expose the same movable actor (`object`), reference actor
(`target_object`), instruction placeholders, and mutually exclusive native
success relations.  Each source task and accepted scene seed is evaluated
under three conditions:

| Condition | Policy instruction | Success predicate | Reported metric |
|---|---|---|---|
| `correct` | source | source | Correct SR |
| `shuffled` | alternate | source | DTL (lower is better) |
| `counterfactual` | alternate | alternate | CIS (higher is better) |

`shuffled` and `counterfactual` receive the exact same alternate instruction.
The manager runs each `correct` job first and saves its accepted episodes as a
canonical seed bank.  Only then are the corresponding `shuffled` and
`counterfactual` jobs launched; they replay the canonical seeds and exact
source/alternate instruction strings without independently rerunning expert
seed selection.  This two-stage schedule prevents nondeterministic CuRobo
planning from silently changing the scene matrix between conditions.
Before a rollout, both goals must be false.  During a rollout, source and
alternate predicates are observed independently and their ever-success and
final-success values are retained.

The declarative predicates mirror RoboTwin's native `place_a2b_left.py` and
`place_a2b_right.py` checks: planar distance in `(0.08, 0.20)`, orthogonal
distance below `0.05`, the requested x direction, and both grippers open.  An
expert rollout audits this declarative source predicate against native
`check_success`; any drift aborts the job.

The default matrix contains:

```text
2 source directions x 2 domains (clean/randomized) x 3 conditions = 12 jobs
```

This is a controlled spatial-relation CIS benchmark.  It must not be reported
as counterfactual coverage of all RoboTwin tasks; most other native tasks do
not provide an executable, mutually exclusive alternate goal in the same
scene.

## V9.39 four-task-family baseline

`configs/eval/robotwin_cis_v939_four_tasks.json` extends the matched protocol
to four task families.  It contains five executable source directions because
the native left/right family is audited in both directions:

| Family | Source | Counterfactual instruction/goal |
|---|---|---|
| spatial direction | `place_a2b_left`, `place_a2b_right` | reverse left/right relation |
| stack order | `stack_blocks_two` | red on green instead of green on red |
| row order | `blocks_ranking_rgb` | blue-green-red instead of red-green-blue |
| tray slots | `place_burger_fries` | hamburger/right and fries/left instead of native slots |

The final three are one-way interventions in their native RoboTwin scene;
RoboTwin does not ship reverse environment classes for them.  Their source
predicates mirror native `check_success`, while their alternate predicates are
independent executable same-scene goals.  Their alternate instructions are
fixed in the manifest so no nonexistent task description file is required.

Run a clean-domain one-episode smoke matrix first:

```bash
export MANIFEST_PATH="$PWD/configs/eval/robotwin_cis_v939_four_tasks.json"
export CIS_TASKS=place_a2b_left,place_a2b_right,stack_blocks_two,blocks_ranking_rgb,place_burger_fries
export CIS_TASK_CONFIGS=demo_clean
export CIS_CONDITIONS=correct,shuffled,counterfactual
export FASTWAM_EVAL_MODE=B0
export RUN_TAG=robotwin_v939_baseline_four_tasks_smoke_seed42

bash scripts/eval_robotwin_cis.sh \
  8 1 10 42 \
  "$ROBOTWIN_CKPT" \
  "$STATS_PATH"
```

After the smoke matrix validates, run both domains with 100 matched episodes
per cell:

```bash
export CIS_TASK_CONFIGS=demo_clean,demo_randomized
export RUN_TAG=robotwin_v939_baseline_four_tasks_formal_seed42

bash scripts/eval_robotwin_cis.sh \
  8 100 10 42 \
  "$ROBOTWIN_CKPT" \
  "$STATS_PATH"
```

The formal matrix contains `5 source directions x 2 domains x 3 conditions =
30 jobs`, or 3,000 matched rollouts.  This evaluation produces no training or
full-goal data.

## Server preflight

The server needs the normal RoboTwin environment, assets, `task_config`,
FastWAM checkpoint, and matching dataset statistics.  It does not need the
RoboTwin training dataset for evaluation.

On servers that cannot reach `huggingface.co` directly, install the release
checkpoint, pinned RoboTwin task configs, and required assets through a mirror:

```bash
HF_ENDPOINT=https://hf-mirror.com bash scripts/setup_robotwin_cis_server.sh
bash scripts/setup_robotwin_cis_python_env.sh
```

Downloads are resumable. Large asset archives use curl with retry and low-speed
timeouts so a stalled mirror connection does not hang forever. Successfully
extracted archives are marked under `third_party/RoboTwin/assets`; archives are
deleted after extraction by default to save disk space. Set
`KEEP_ASSET_ARCHIVES=1` to retain them.

The Python environment setup pins CuRobo `v0.7.8`, the upstream-recommended
version for the v1 API used by vendored RoboTwin. Building CuRobo requires a
CUDA toolkit with `nvcc`; a runtime-only CUDA container is insufficient.

```bash
cd /path/to/fast-WAM
git pull --ff-only origin main
pip install -e .

export ROBOTWIN_CKPT=/absolute/path/to/robotwin_checkpoint.pt
export STATS_PATH=/absolute/path/to/robotwin_dataset_stats.json

bash scripts/validate_robotwin_cis_server.sh \
  "$ROBOTWIN_CKPT" \
  "$STATS_PATH"
```

The released RoboTwin FastWAM checkpoint uses the B0 action path, which is the
launcher default.  Set `FASTWAM_EVAL_MODE=M1` for an M1 checkpoint.  For any
other architecture, use `FASTWAM_EVAL_MODE=CUSTOM` and append its exact Hydra
model overrides to the launcher command.

The single-task entrypoint persists its fully composed Hydra configuration and
passes that snapshot across the RoboTwin policy subprocess boundary.  Model
architecture overrides therefore apply to the process that actually constructs
FastWAM.  Evaluation also aborts if latent action queries are enabled but their
checkpoint tensor is absent, rather than using randomly initialized queries.

## Runs

One-episode, one-direction, clean-domain smoke test (three jobs):

```bash
CIS_TASKS=place_a2b_left \
CIS_TASK_CONFIGS=demo_clean \
RUN_TAG=robotwin_cis_smoke \
bash scripts/eval_robotwin_cis.sh \
  3 1 10 42 \
  "$ROBOTWIN_CKPT" \
  "$STATS_PATH"
```

Bidirectional five-episode pilot (12 jobs, 60 episodes):

```bash
RUN_TAG=robotwin_cis_pilot_seed42 \
bash scripts/eval_robotwin_cis.sh \
  8 5 10 42 \
  "$ROBOTWIN_CKPT" \
  "$STATS_PATH"
```

Formal 100-episode matrix (1,200 episodes):

```bash
RUN_TAG=robotwin_cis_formal_seed42 \
bash scripts/eval_robotwin_cis.sh \
  8 100 10 42 \
  "$ROBOTWIN_CKPT" \
  "$STATS_PATH"
```

Jobs use one model process per GPU by default.  A repeated command with the
same checkpoint, run tag, episode count, tasks, domains, and conditions skips
only fully validated job directories.  Partial or checkpoint-mismatched output
is rerun.

To evaluate only randomized scenes and the two conditions needed for CIS:

```bash
CIS_TASK_CONFIGS=demo_randomized \
CIS_CONDITIONS=correct,counterfactual \
RUN_TAG=robotwin_cis_only \
bash scripts/eval_robotwin_cis.sh \
  8 100 10 42 \
  "$ROBOTWIN_CKPT" \
  "$STATS_PATH"
```

Any run containing `shuffled` or `counterfactual` must also contain `correct`,
because the latter supplies the canonical episodes.  During the first stage,
some GPUs can intentionally remain idle until their matching `correct` job has
finished.

## Outputs and validation

The manager writes under:

```text
evaluate_results/robotwin/<checkpoint-tag>/<run-tag>/
├── <source-task>/<task-config>/<condition>/
│   ├── episodes.jsonl
│   ├── summary.json
│   ├── _result_clean.txt or _result_random.txt
│   └── rollout videos
├── cis_manager.log
├── cis_manager_state.json
├── failed_jobs.txt
├── cis_summary.json
└── cis_summary.csv
```

`cis_summary.json` is emitted only from validated job records.  Its matched
audit rejects seed drift, source/alternate instruction drift, unequal
Shuffle/Counterfactual policy instructions, initially satisfied goals,
duplicate cells, incomplete episode files, and mixed checkpoints.

Results can be revalidated independently:

```bash
python scripts/summarize_robotwin_cis.py \
  evaluate_results/robotwin/<checkpoint-tag>/<run-tag> \
  --expected-episodes 100 \
  --task place_a2b_left \
  --task place_a2b_right \
  --task-config demo_clean \
  --task-config demo_randomized \
  --condition correct \
  --condition shuffled \
  --condition counterfactual \
  --require-complete
```

Validated baseline and future method comparisons are recorded in
[`ROBOTWIN_CIS_RESULTS.md`](ROBOTWIN_CIS_RESULTS.md).
