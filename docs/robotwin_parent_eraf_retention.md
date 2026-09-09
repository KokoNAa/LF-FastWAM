# Preserve the completed ERAF policy during FG continuation

The serial September 9 development run completed all 180 episodes. CF totals
were 40/60 (no ERAF), 43/60 (ERAF), and 33/60 (ERAF + FG). FG inherited the entire
completed ERAF model, and both training stages updated action LoRA and ERAF.
However, the retention target still came from the earlier no-ERAF model with
ERAF disabled. This mismatch is a candidate explanation, not a demonstrated
cause of the regression.

The opt-in `--cf-teacher-mode parent_eraf` changes only that retention teacher.
The CF teacher checkpoint must be the exact hash-bound completed ERAF parent
passed to `--checkpoint --continue-eraf-with-fg`. Its action/video adapters and
all ERAF parameters and persistent buffers are used for the ten-step retention
prediction. Student storage is restored before student autograd. The teacher
receives no gradients. Native/off behavior remains the default. This mode is
currently restricted to fresh joint five-task FG continuation, without optimizer
resume; unsupported configurations fail explicitly.

The student continues to jointly train action LoRA and all ERAF interfaces.
The backbone, video LoRA and semantic heads remain frozen. The mixture, data,
learning rates, seed, temporal sampling and FG gradient route remain unchanged.

## Proposed controlled experiment

- Parent: completed serial ERAF200 checkpoint, SHA256
  `26aea72064d7c17c39992ce2ece74ae63a7496d13f0931bcee3e2ba39e98fdcd`.
- Start a new FG200 arm from this parent, with the full parent as CF teacher.
  Retain the original FG200 (old no-ERAF teacher) as the matched comparison.
- Keep global batch 12 on GPUs 0, 1, 2; policy LR 3e-6, ERAF LR 3e-5;
  joint gradients; the same 4 retention / 4 expert / 4 FG mixture and seed 42.
- Predeclare final step 200. Do not select checkpoints from partial CF scores.
- Evaluate the new arm on all 60 existing canonical development scenes
  (12 per task). Reuse the unchanged reference evaluations on these exact
  scenes, with model, scene, configuration and code provenance preserved.
- Report all five tasks, aggregate CF, lift and placement, and paired effects.
  This reuses development scenes and is not an independent test. Correct/native
  performance remains unevaluated. A new 40-per-task test requires separate
  nomination after development evidence.

Before a full run, require the five-task production teacher identity probe and
an audited two-step three-GPU training smoke. The smoke checkpoint is never the
parent of the full experiment. The existing 14:50 HKT cutoff is binding; if the
full run cannot fit, a runtime extension is required. Finishing code or a smoke
test is not evidence of the desired model ordering.

The user subsequently removed the work cutoff on September 9. The controller
now accepts either an explicit `--deadline` or `--no-deadline`; the latter
records the absence of a cutoff in the run plan. It still stops its own jobs
on failure or insufficient data-disk space and never shuts down the platform.
