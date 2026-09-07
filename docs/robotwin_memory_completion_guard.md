# Prevent unverified completion from entering clause memory

This is a separate experimental branch from the running paired1000 joint200
trial. That trial continues on `bb3bcce` with its original deployment behavior.
No CF success predicate, checkpoint tensor, dataset or optimizer is changed here.

## Reproduced defect

`PhaseSafeClauseMemory` computes a guarded `completed` condition requiring
predicate truth and release evidence for a new completion. However, its fallback
copied the raw classifier state into `next_state_ids`. A classifier prediction of
COMPLETED with false predicate truth and phase0 therefore yielded
`completed_sticky=False` but `next_state_ids=COMPLETED`. On the following replan,
the exported state qualified as previous completion and became sticky.

The new test fails on unmodified `bb3bcce` for the cold-start, false-predicate,
phase0 case: exported state3 contradicts the false completion condition. This is
a synthetic module reproduction, not evidence of its frequency in actual policy
rollouts or a demonstrated cause of all RoboTwin failures.

## Change and validation

When the guarded condition rejects a COMPLETED proposal, default to PENDING.
Existing holding and release-to-retry overrides then apply normally. Previously
recorded completion stays sticky, including during temporary inactive predictions.
The learned logits, scheduler and grounding outputs retain their existing paths.

Tests cover two consecutive replans, all three phases, four non-completed prior
states/cold start, existing sticky completion with active/inactive clauses,
positive-truth release/holding, and released-unsatisfied retry. The unchanged
residual-interface tests also pass:27 tests total on the Mac. The existing large
V9.13 test module cannot collect in the Mac venv because torchvision is absent;
server-side regression validation is recorded separately before deployment.

Do not mix this code into the active trial. A subsequent fixed-checkpoint
comparison must keep states/instructions and success rules matched, report both
ERAF-only and FG+ERAF outcomes, and retain the historical strongest controls.
No action-performance gain is claimed from these unit tests.
