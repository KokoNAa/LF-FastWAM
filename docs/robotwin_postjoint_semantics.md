# Bounded post-action semantic diagnostic

The two ERAF action arms use two GPUs each and may finish before the two single-GPU controls. This diagnostic uses GPUs0and2 only after both ERAF workers exit successfully with final200-step outputs, while both slower controls still have at least50updates remaining. It confirms the primary controller is actually live and the requested GPUs have no compute owner.

Run `scripts/probe_robotwin_postjoint_semantics.py --source-root ACTION_RUN --output NEW_DIAGNOSTIC --seconds 240`. The command starts no training and never selects checkpoints. It applies the existing80-query held-out terminal semantic probe to each final ERAF action checkpoint, comparing against the same observations and labels used for the corresponding semantic1500 parent. The purpose is to detect whether joint action optimization retains or erases the earlier semantic changes.

The supervisor hashes its model/manifest inputs before and after, records all commands and process exits, and terminates only its own children at the four-minute limit or when the primary controller leaves joint training. It does not alter the primary training/evaluation script, task goals, loss, outputs or deadline. A completed semantic diagnostic is not a CF success-rate result or a reason to select an intermediate model.
