# Serial ERAF and FG validation

The current experiment keeps the completed current no-ERAF policy fixed, trains
ERAF for 200 optimizer steps, then continues every learned ERAF and policy weight
for 200 further steps with FG supervision. FG starts a fresh optimizer for its
changed objective; it does not reset ERAF or load an independent FG donor.
The cumulative additional steps since the current control are **0 / 200 / 400**.
These models compare the serial training recipe, with different training budgets;
their scores alone do not isolate the effect of each module at an equal budget.

The source controller first runs 12 new development scenes in each of the five
fixed tasks for all three models (180 CF episodes). It records verified object
lifts, the task goal after lifting, strict release placement, and videos. All
tasks and failures remain in the equal-task macro CF result. Correct is not
evaluated in this protocol.

`run_robotwin_five_task_independent40.py` can nominate these candidates only after
the complete paired development matrix supports ERAF+FG > ERAF > no-ERAF. It
rechecks all checkpoint hashes, source bindings, completed process receipts,
training audits and the actual initialization proof that FG inherited the
completed ERAF checkpoint. The external no-ERAF checkpoint must remain the
audited current control. A full `robotwin_serial3_completion_audit_v1` receipt is
required; interrupted runs, smoke results and partial development cells cannot
be nominated.

The evaluator freezes all three candidates before accessing 40 new paired test
scenes per task (600 CF episodes). Training, development and historical formal
scenes are excluded. The test namespace must be unused; the evaluator does not
resample a namespace after observing outcomes. Model and inference runtime files
must match the source training checkout exactly. The serial training budget is
retained in the frozen protocol and paired report.

No independent test starts automatically merely because this code is installed.
It requires a completed positive development result, free GPUs, a new server
data-disk output and an authorized deadline long enough to finish the full test.
Model, optimizer and video files remain on the server data disk.
