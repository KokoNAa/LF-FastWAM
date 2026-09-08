# Matched controls for trajectory-target800

The completed ERAF+FG candidate (`20260908-trajectory-target800`, code
`f794d7176b3d5f4dcb3603040ee73c5d0309ac4f`) scored 31.667% on the ten-task
CF development matrix, below the historical strongest no-ERAF control (35%).
The user requested the missing matched controls. This run completes no-ERAF,
ERAF-only and FG-only using the exact 800-step trajectory recipe. The full
candidate is reused at its actual audited SHA; it is not retrained or selected
from a new checkpoint sweep.

## Matching

Each new arm runs on all six GPUs sequentially: no-ERAF, ERAF-only, FG-only.
All use global batch 12, seed42, 800 optimizer steps, final800 checkpoint only,
policy LR3e-6, interface LR3e-5, and the unchanged ten-step differentiable
sampler with first24 executed-action target-only loss. Each batch has four
old-task CF teacher examples at weight4, four complete cup/pill ordinary
trajectory examples and four FG or matched ordinary replacement examples at
weight1. No Correct or source-language conditional-difference loss is used.

No-ERAF and FG-only start from the exact strongest R action checkpoint.
ERAF-only starts from the previously audited ordinary S1500 semantic parent;
ERAF+FG used the corresponding FG S1500 parent. Both semantic parents have
verified zero-output action identity with R. Semantic training budgets match;
this run performs no new semantic training. FG-off replaces each FG slot by a
same-task/config ordinary target, while retaining exactly the same common
sample slots and noise-seed formula. ERAF-off freezes all interface parameters.
No existing trainer, sampler, architecture or evaluator files are modified.

Admission verifies actual runtime files against the candidate runtime, all
input hashes, prior complete evaluation and available GPUs/storage. First-step
and final audits verify actual rank journals. The final four-arm audit includes
the prior candidate's actual journals: 38,400 total examples, 6,400 shared and
3,200 matched FG/replacement slots per arm. Each arm saves its full model and
full optimizer on the server data disk. No model or optimizer is downloaded
to the Mac.

## Evaluation and interpretation

Run 135 new CF episodes (45 per new arm), retaining the existing candidate's
45 episodes. All use the same physical catalogs, task instructions, initial
seeds, ten-step inference, memory behavior and success predicates. Videos stay
off as in the candidate. Equal-task macro CF is primary; report individual
tasks, paired gains/losses and the burger strict final-slot supplement. The
comparison retains all historical methods, including the actual strongest
controls, for 33 total method entries. No independent-test data are used and
this ablation alone cannot establish independent generalization.

The operator deadline stops only owned jobs; it never shuts down the platform.
The server remains on under the user's continuing authorization. This run is
stored under the expanded data disk, not the system disk. Tests and preflight
must pass before launch; all commands, hashes and job exits are recorded.
