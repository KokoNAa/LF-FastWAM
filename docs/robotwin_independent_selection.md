# Freeze selection before creating an independent test

Use `scripts/freeze_robotwin_independent_selection.py --source-root COMPLETED_DEV --test-root NEW_TEST --output SELECTION_JSON` on the server only. This command hashes full checkpoints but does not load policies, acquire GPUs, collect scenes or consume test outcomes. The selection JSON must sit outside the new test directory, which must not yet exist.

The source controller must have completed and exited. The tool reconstructs its complete development comparison from episode and physical-state evidence and requires exact agreement with the archived comparison. It recomputes equal-task scores from counts and rejects any tie or loss of ERAF+FG against any declared method. It retains the three current matched ablations, the named historical strongest no-ERAF and FG-only models, the earlier best ERAF-only control, and any additional methods tied at the globally strongest control score. This is a fixed control-inclusion rule, not a search over test outcomes.

Every selected full checkpoint must still match the identity recorded in its actual development evaluation. The receipt records full SHA256 and metadata, deployment settings, all development scores, source evidence hashes, and expanded-training/development-catalog exclusion hashes. A missing or reserialized checkpoint is not silently accepted; it requires a separate identity-preserving archival workflow. Do not modify historical evaluation records to bypass that check.

The test controller must subsequently bind this receipt, verify its hashes before and after inference, generate the reserved test scenes with dual-expert feasibility, and retain a chronological record proving selection preceded test access. This utility cannot establish that no test was accessed in another directory or that every historical training source was supplied. Its receipt is one part of the completion evidence, and never claims model superiority on test data or goal achievement.

This tool is prepared while development training runs. No model is frozen and no test is opened merely by installing or testing this script.
