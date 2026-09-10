# Generated-video review protocol

Freeze these rules before inspecting formal generated videos. The earlier
seen-texture smoke is an implementation test and is excluded from analysis.

## Measurements and interpretation

Report language-induced image differences over every generated clip, separately
for the full mosaic and the union of task-entity regions. Compare these with
between-noise differences under the same instruction. This measures sensitivity,
not goal understanding. The paired correct/wrong video-denoising margin is a
separate conditional-fit measurement and must not be relabeled success rate.

Use a predetermined qualitative sample: the first two accepted scenes of each
task, in frozen catalog order. Review initial, latest exactly shared decision
state (if present), source-late and target-late windows, all three noise seeds,
both checkpoints, both instructions. Missing shared-decision states are recorded
as missing, not substituted using model outcomes. Every selected generated clip
must receive a review record or an explicit unobservable/malformed classification.

Panels show both experts only for identical-observation windows. For a later
branch-specific observation, show only that branch's expert future. Comparing the
other branch at the same numerical frame index would silently compare different
world states. Keep the camera mosaic layout and the true frame indices visible.

For each generated clip record:

- visible physical change (object identity, direction, or relative geometry);
- whether the instructed final relation is visibly established in the window:
  yes, no, or unobservable (occlusion/artifacts prevent assessment);
- whether there is a clear visible contradiction with the instruction, and the
  concrete observation supporting it; otherwise use insufficient evidence;
- generation quality (usable, ambiguous, malformed), with evidence;
- comparison with the other instruction under identical noise: semantically
  relevant change, other visual change only, no visible change, or unobservable.

A relation not established in a 32-action prediction window is not automatically
a policy failure. Initial pickup choice is not sufficient to label ranking wrong:
either end block can be moved first while ultimately reaching RGB or BGR order.
Likewise, expert-trajectory similarity is not proof that an alternative trajectory
violates the goal. Do not infer gripper contact or physical grasp from pixels alone.
Verified contact/grasp/release outcomes come from the closed-loop simulator.

The qualitative sample is not a 50-scene video success-rate benchmark. Report its
actual sample coverage, uncertainty and limitations separately. Do not substitute
this review for the 1,200 closed-loop episodes or numerical all-state experiments.

The final report requires all four experiments, complete numerical coverage,
paired input/output verification, the completed predetermined visual review, and
scene-level aggregation. Pure output completion flags do not establish that.
