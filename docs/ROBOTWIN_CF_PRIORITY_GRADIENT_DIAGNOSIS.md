# Diagnose the executed CF-priority recipe

The 2026-09-07 warm-all5-off200 trial completed all three 400-step training arms.
Its complete six-scene RGB ranking cells are baseline 1/6, ordinary CF200 3/6,
ordinary CF400 2/6, FG200/400 0/6, and ERAF+FG200/400 0/6. Other task cells and
the selected-model ERAF bypass are still running at this decision point. These
are development results, not independent-test evidence or an overall ranking.

The historical gradient probe hardcoded ERAF off, Video+Action parameters,
Correct4/CF2, loss weights4/2, and language augmentation. Those gradients cannot
explain the current Action-only, Correct2/CF4, weights1/4, augmentation-off trial
without a fresh measurement. Its policy source and live bank must stay unchanged.

`probe_robotwin_objective_gradients.py --training-plan <arm>/joint/plan.json`
now recovers the recorded parameter scope, ERAF state, FG mode, sampler, seed,
augmentation and loss weights. The plan's optimization contract and mixture must
agree; supplied bank/teacher paths and the actual selected parameter names must
match the saved plan. Historical invocations retain their previous defaults.
The checkpoint being measured is supplied separately, so the same fixed batches
can be evaluated at different saved policy checkpoints.

Run three identical global batches at each arm's step200 first, using that arm's
own plan and `--skip-actions --batches 3 --max-seconds 600`. This is a bounded
diagnostic, not a new training or model-selection run. Record the requested
checkpoint, full training plan, its digest, code commit, exact sampled IDs,
per-example noise seeds and timestep, and completion status. Verify sampled IDs
against the corresponding executed rank journals before interpreting results.
Only launch after the evaluation controller releases the GPU; a momentary low
GPU utilization or memory reading does not establish that a GPU is free.

Inspect weighted group norms and cosines separately for Action LoRA and ERAF
interfaces. Ordinary CF control slots are reported separately from paired
experts. Compare FG against CF retention and paired experts; Correct is reported
for diagnosis but is not a performance threshold. These are raw gradients before
AdamW preconditioning, not proof of the optimizer's actual update direction.
The probe creates no optimizer, does not accumulate `.grad`, and verifies selected
parameters are unchanged. Closed-loop CF evaluation remains the efficacy test.

If the first three batches do not cover both FG target tasks or give inconsistent
directions, extend to the remaining predeclared batches within the current runtime
window. Do not select a favorable batch or interpret a partial probe as complete.
