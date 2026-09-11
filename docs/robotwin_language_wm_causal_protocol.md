# Language acting inside the RoboTwin world model

Inference only; no training, action sampling or simulator rollout. Reuse the
completed world-language study's unmodified data, released checkpoint and no-eraf
checkpoint. Keep prior reports and their user-requested presentation transforms
out of the new measurements. New outputs report raw measurements and actual counts.

Freeze the first two catalog scenes per task (10 common states): initial for
ranking/stacking, shared post-grasp decision for placement. Require both raw
observations identical, dual references valid and future expert windows different.
Three paired noise seeds 42/43/44. Scene is the statistical unit; two scenes/task
make this a mechanistic pilot, not a population success-rate estimate.

At sigma 0.5 and 1.0, use both expert futures and identical noise per comparison.
Compare source/target text, both same-meaning paraphrases, empty task text, and
repeated source text. Keep the prefix, proprioception and initial-image latent.
Record correct/wrong-future velocity MSE, object/background regions, and all
30 layers' cross-attention residual differences with spatial distributions.

Necessity intervention: mask every text-context position (including text padding)
at layers 0-9, 10-19, 20-29, or all layers. Preserve the appended proprio token.
This removes all textual context, not only the changed relation word; the empty
task condition separately retains the usual prompt prefix.

Residual interchange: record source and target cross-attention outputs on the
same noisy video, then substitute the donor outputs in each layer band, both
directions. Verify same-source sham replacement and full-layer donor recovery.
This is a distributed residual intervention; donor activations carry earlier
contextual effects and do not establish individual-token causal attribution.

Generate 9-frame/32-action-window video-only futures with the original 20-step
flow schedule. Store exact initial-noise hashes and clamp the initial latent at
every step. Generate all conditions on the first selected scene per task, seed42:
five language controls, 8 layer text masks, 8 bidirectional residual patches, and
6 temporal text masks (steps 0-5,6-12,13-19). Recompute donor activations on each
recipient's CURRENT noisy latent, never reuse a diverged donor trajectory.
Store every PNG and latent output; save the exact times/deltas. Compare baseline
video-only velocity against the production joint predictor on a smoke case.

Artifact checks: repeat-source and self-patch numerical controls, preservation of
proprio and initial frame, input and checkpoint SHA256, complete layer coverage,
recoverable per-state outputs and non-overlapping shards. Do not claim complete
if any condition is missing. Unexpected values fail the worker rather than being
silently omitted. Keep smoke outputs separate from formal outputs.

Report paired margins, change relative to natural source-target differences, and
object/background sensitivity. Near-zero source-target differences make a
normalized mediation score undefined, not zero. Average noise within scenes;
report per-scene values and ranges, without inferential claims from two scenes.
Review all generated conditions for the frozen five scenes, comparing trajectories,
object/relationship consistency and visual artifacts with the expert references.
Image/latent distances are not task success. First-frame reconstruction differences
are excluded from future metrics. Short windows cannot establish whole-task success.
