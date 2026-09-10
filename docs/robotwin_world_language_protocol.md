# RoboTwin world-model language intervention

This is inference-only research. No optimizer steps, checkpoint changes, platform
shutdown, or wall-clock cutoff. Run one GPU worker at a time on the data disk.

Primary checkpoints: released robotwin_uncond_3cam_384 and current no_eraf
20260908-five-task-expert200 step_000200. Recompute/check their SHA256 identities.
The scope includes all four experiments below, not just representation distance.

Use five tasks (RGB ranking, two-block stack, left/right placement, burger/fries),
ten fresh demo_randomized scenes each. Select by feasibility of both expert goals,
never by policy success. Audit scene exclusion against available training manifests.
Require both goals false initially and eval_mode=true during collection and policy
evaluation, so both use the unseen texture partition. The earlier collection smoke
used the inherited default eval_mode=false and is excluded from the final study.
The three placement tasks use the existing paired shared-grasp planner to supply
identical post-grasp decision observations. Preserve that collection distinction.

1. At identical RGB/proprioception, compare source, target, source paraphrase,
   target paraphrase and empty instruction. Repeat identical inputs. Record all
   video-block hidden and K/V relative L2/cosine distances, including numerical
   floor and spatial distributions. First-layer pre-cross-attention K/V may be equal.
2. Cross source/target video K/V with independently fixed source/target action
   text. Keep all other inputs, masks and sampling schedules fixed. Seeds 42, 43,
   44 are repeated draws, not independent scenes. Same-cache reconstruction must
   match unmodified production inference. Report action differences, expert-goal
   preference where both references share the observation, and closed-loop goal,
   grasp and placement outcomes. Swap both directions; quantify interaction.
3. With paired true future clips, compute correct/wrong-language video velocity
   losses at low/mid/high noise using identical noisy tensors. Exclude initial
   and padded frames. Report full-image and operation-region results, paired
   margins and scene-level uncertainty. Do not confuse action denoising with video
   denoising, or low-noise reconstruction with unconditional goal selection.
4. Generate paired future videos with the same observation/noise, using the local
   joint inference entry point. Report semantic object/direction/relation adherence
   and model-quality failures, not only pixel differences. Check the configured
   temporal stride/horizon, retain inspectable clips, and distinguish this auxiliary
   mode from deployed action-only inference. Current baseline config has video
   action_conditioned=false and video tokens cannot read action tokens.

Use original full instructions bound to scene objects plus manually checked
paraphrases. At later states branch trajectories need not coincide: only compare
languages on the same fixed observation, and never call different branch states
same-state pairs. Do not demand different initial actions where goals share a prefix.

Report actual counts, no rate scaling or hypothetical episodes. Aggregate seeds
within scenes before confidence intervals. Archive code SHA, protocol/input hashes,
per-scene results, process exits and rendered figures. A completion flag alone does
not prove the four experiments or semantic scoring were completed.

Closed-loop termination retains the ordinary selected-goal rule associated with
the action-side instruction. Video-only comparisons hold that rule fixed. For
cross-action-language comparisons additionally report first-goal categories, using
the verified immediate termination at a selected success. If both goals were ever
true and only the selected goal is true finally, the opposite goal occurred first.
Simultaneous final goals or an unverifiable termination invariant are ambiguous.
Do not confuse these first-choice outcomes with full-rollout ever-success or grasp
counts. Confirm the simulator still immediately returns and the outer loop still
stops before another physics step when auditing this derivation.
