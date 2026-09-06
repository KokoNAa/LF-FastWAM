# Warm RoboTwin policy repair with ERAF and Full Goal

This experiment starts from the selected shared Video/Action LoRA checkpoint
(`shared400`), whose existing matched development result is Correct11/15 and
CF5/15. Left placement and RGB-to-BGR ranking are both CF0/3. It asks whether
ERAF and complete corrective trajectories can improve those two tasks while
retaining the three already successful CF tasks.

This is a new checkpoint protocol, `robotwin_eraf_fg_adapter_v1`. Its constructor
uses the existing V9/objective26 ERAF architecture, but does **not** claim the
historical V9.28/V9.39 initialization, training milestones or source-only
failure lineage. Historical validators and entry points retain their contracts.

## Initialization and optimization

Bootstrap copies every shared400 policy adapter tensor exactly and initializes
ERAF separately. The action interface uses one shared policy for all tasks and
both instructions. It has no task-specific checkpoint selection or CF gate.

1. Train only ERAF semantic heads, global batch12, LR1e-4, evaluate every250
   optimizer steps, initial budget1500. A bounded extension can reach3000.
   Selection requires held-out role accuracy>=80% and relation accuracy>=90%.
   The scene split is audited independently of frame counts.
2. Train the action interface for100 optimizer steps at LR5e-5. This explicitly
   includes the fresh GoalGraph/query seeds, ERAF query routing, grounding
   bridge and context injector. Semantic heads and both policy experts freeze.
3. Jointly train those interfaces and shared Video/Action LoRA at LR1e-5;
   candidate joint checkpoints are200,400,800. Semantic heads remain frozen.

The optimizer keeps FP32 master parameters for BF16 modules so small updates
accumulate instead of rounding away. Stage transitions restart optimizer state;
weight checkpoints do not claim to be optimizer-resume checkpoints. Frozen
VAE/T5 inputs are cached before trainable Video operations; trainable Video and
action interfaces are recomputed for each gradient graph.

The global12-example mixture contains4 Correct retention states,2 CF retention
states,3 old expert pairs and3 FG corrections. Correct teacher weight is2, CF
teacher weight1, CF-positive weight2, endpoint weight1 and same-observation
conditional gain4. Teachers are Dense600 for Correct and shared400 for CF.
All temporary teacher weight swaps finish before any student graph is built.
Own-state expert pairs receive positive supervision; only exactly equal visual
latents and proprio may receive additional paired-difference supervision.

## Full Goal data

New collection uses reserved seeds[80000000,81000000), excluding all existing
replay scenes. Left and ranking each require24 training and6 data-holdout
scenes. The other three tasks each contribute10 successful shared400 CF scenes
for retention. These are separate data categories.

For each target-task scene:

1. Screen whether an ordinary CF expert can solve the initial scene. This is
   feasibility screening only; its trajectory is never an FG positive.
2. Run shared400 under the CF instruction and audit both complete goals.
3. For a failed rollout, replay a nonzero executed policy prefix into a fresh
   identical scene. Verify object and robot poses, velocities and qpos/qvel
   against the captured state, at absolute tolerance1e-7.
4. Continue physically from that state until the complete CF goal holds and
   both grippers are open. Preserve dense controls **and** RoboTwin's separate
   coordinated-arm planning path, which bypasses `take_dense_action`.
5. Require two independent prefix/control replays to complete the full goal.
   The coordinated-arm recorder also checks physical state after each control.

Failed rollouts that reached the source goal and those that reached neither
goal have different provenance labels. They are not silently conflated.
All accepted corrections carry checkpoint, initial/captured-state, prefix,
trajectory and archive hashes. No object teleport or fresh initial expert
trajectory is substituted for a failure-origin correction.

Full trajectories produce32-action windows at stride8, including the final
goal-reaching tail. Storage padding is masked out of the loss. The local
control sees only the same failure-start observation and first12 corrective
actions. Full and local controls draw exactly the same scene schedule and
preservation examples; later FG windows never enter the local control.

Gold simulator entity state is used for training labels and auditing only.
Deployment consumes three RGB cameras,14-D proprio and the instruction.
Offline action replay supplies no gold memory or privileged execution state.

## Evaluation and interpretation

`eval_robotwin_eraf_fg.py` reuses the existing CIS simulator loop. It replaces
only model construction and leaves the old policy symlink untouched. All
conditions use horizon32, replan24,10 denoising steps, model seed42 and the same
384x320 three-camera mosaic. Correct/CF initial physical states, scene seeds,
instructions, checkpoint identity and deployment settings are checked.

The existing30-episode regression requires one checkpoint with Correct>=10/15
and CF minima left1/3, right2/3, burger1/3, stack2/3, ranking1/3. It is reused
development data, not an independent test set. New development has6 scenes
per task and both instructions. The final locked test has10 new scenes per
task and both instructions. New catalogs use90M and91M seed ranges respectively,
select by native-expert feasibility, exclude initially satisfied goals and use
the existing unseen-instruction generator. Learned-policy outcomes do not
select catalog scenes. Locked-test outcomes must not choose checkpoints.

First attempt ERAF+FG, then compare no-ERAF/no-FG, ERAF-only, FG-only,
ERAF+FG and the local12-action control, with explicit checkpoint lineage,
optimizer counts and evaluation catalogs. A single improved arm does not
establish a module effect. Report incomplete controls and sample uncertainty.
Grounding accuracy and expert correction success are not policy CF success.

## Entry points

- `scripts/robotwin_eraf_fg.py`: bootstrap, inspect, acceptance.
- `scripts/train_robotwin_eraf_fg_grounding.py`: semantic pretraining.
- `scripts/collect_robotwin_eraf_fg.py`: verified failure corrections.
- `scripts/collect_robotwin_cf_retention.py`: successful CF retention states.
- `scripts/prepare_robotwin_eraf_fg_replay.py`: frozen-input preparation/merge.
- `scripts/train_robotwin_eraf_fg_action.py`: interface and joint training.
- `scripts/eval_robotwin_eraf_fg.py`: catalogs, matched evaluation, summary.

The campaign directory's `launch.json` records exact commands, process IDs and
code revisions. Each detached job writes its own log and exit marker. Formal
action training rejects incomplete per-task scene budgets. The six-GPU run
must stop experimental work before its reserved final archive/shutdown hour.
