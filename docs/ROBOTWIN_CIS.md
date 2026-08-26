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
The manager also requires the same ordered scene seeds across all conditions.
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

To evaluate only randomized scenes or only the CIS condition:

```bash
CIS_TASK_CONFIGS=demo_randomized \
CIS_CONDITIONS=counterfactual \
RUN_TAG=robotwin_cis_only \
bash scripts/eval_robotwin_cis.sh \
  8 100 10 42 \
  "$ROBOTWIN_CKPT" \
  "$STATS_PATH"
```

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
