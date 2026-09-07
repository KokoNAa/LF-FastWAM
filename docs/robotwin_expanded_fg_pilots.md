# Expanded task full-goal correction pilots

The paired semantic1000/joint200 trial and fixed-weight memory guard trial both
ended at31.6667% ten-task CF macro for ERAF+FG, below the strongest35% control.
All five additional tasks remained0/3. The actual joint200 sample audit gives
each new task60expert row draws and0FG draws per arm. This is a coverage gap,
not proof of a single cause. This change tests whether real failed-policy
corrections can be collected for cup, size ranking, mouse, stapler and pill.

The opt-in `expanded_v2` protocol retains nonzero actual failed policy prefixes,
pose and velocity replay checks at1e-7, complete CF goals with both grippers open,
at least12real corrective actions and2independent full control replays. It
additionally binds canonical task instructions, the input manifest SHA256,
collector commit, rollout checkpoint SHA256 and explicit policy/ERAF/memory
configuration. Original v1 records still reject these new tasks. Historical
exclusions now resolve the authoritative ten-task registry.

Train seeds are[86000000,87000000), replay holdout seeds[87000000,88000000).
They are separate from old collection80m/82m/83m/84m/85m and all90m+development
or independent-test seeds. Pilots must scan existing metadata before reserving
actual task/seed ranges. Pilot trajectories are collection diagnostics and do
not establish a model gain or completed training dataset.

The collector uses the repair loader when given a repair checkpoint; the actual
task name/configuration and ERAF enablement are set explicitly. The default
legacy loader remains available for old collection. New-task pilots initially
reuse the physical expert continuation unchanged. Recovery failures remain
recorded; no object teleportation, success-tolerance loosening, new video input,
or relabeling of initial expert data as FG is allowed.

Each launch records a bounded scene/candidate budget, absolute cutoff and disk
reserve. A supervisor must enforce the hard cutoff on its own process groups,
because a single simulator call may run past an in-process check. Partial
manifests are written atomically during collection. Models stay on the server.
No controller shuts down the platform. Further expansion, fresh verified mask
replay and matched FG/no-FG training follow only after actual pilot evidence.
Performance validation keeps the historical controls and CF criteria unchanged.

## First expansion decision

All6initial pilots completed on2026-09-08 at04:52:38. Cup, mouse, stapler and
pill corrections succeeded at the first scene; size required an earlier
nonzero prefix. Raw camera/action/physical-state hash audits passed. Cup and
pill pilots were source-directed failures, directly matching the old-goal
behavior seen in development. Prior paired terminal probes also showed
instruction-dependent truth margins for these tasks. These observations
motivate prioritizing cup and pill; they do not establish a performance gain.

Opt-in `cup_pill_expansion` allocates24train+6replay_holdout scenes per task
across6GPUs. Each task has two12scene train shards and one6scene holdout shard;
all candidate ranges exclude the pilots and the original replay bank. The
operator requires complete pilot collections, identical collection checkpoint
and input manifest, and successful exactRGB/physical-state mask replays for
cup train/holdout and pill train before starting. It hashes the evidence.
The additional4GiB allocation is separate from a3GiB reserve; job cutoff remains
at most one hour and never powers off the server. Further failed scene attempts
are retained. No benchmark, baseline score, Correct criterion or model weight
is changed by collection. Matched training and unchanged ten-task CF evaluation
are required afterward, followed by independent verification of any lead.
