# Preparing the expanded FG bank

The actual source is the cup/pill formal60 collection at
`robotwin_expanded_fg/20260908-cup-pill-formal30`, based on the fixed paired1000
joint200 FG+ERAF checkpoint. The operator waits for that exact process and
kernel start time. It requires all six collection workers to exit0 and the
complete24train+6holdout per task dataset before claiming the GPUs.

Six workers cache the full corrective trajectories at stride8, with all real
terminal actions retained and padded actions masked. Existing parent rows are
preserved exactly in order. An audit compares every new normalized target to
the recorded raw action using the unchanged released z-score normalizer,
checks all source record fields and empty policy memory, and rejects duplicate
scenes, train/holdout leakage or missing goal-reaching tails.

The old mask binding explicitly permits only added expert data with identical
FG rows, so it cannot be reused for this extension. Instead, six replay workers
reproduce all old108 and new60corrections. Every RGB frame, proprioceptive state,
semantic geometry and physical control state must match. The aggregate index
is published only after exact full-bank coverage is checked. Phase/history
labels remain invalid for corrections; they are not invented from endpoints.

The action CLI now accepts the five explicitly registered expanded FG tasks,
retaining old defaults and minimum24train+6holdout per declared target task.
This does not start training or weaken the data admission requirement.

Preparation changes no model weights and uses no development/test scenes.
The server retains all payloads; only metadata is archived on the Mac.
The cutoff and3GiB reserve stop only this operator's children, never the source
collector or the platform. Preparation completion is not a performance claim.
