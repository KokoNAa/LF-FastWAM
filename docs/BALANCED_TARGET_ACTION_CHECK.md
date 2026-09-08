# Balanced target action check

The completed balanced-target200 trial gives ERAF+FG 30% ten-task macro CF versus 35% historical strongest no-ERAF. All five new tasks remain zero. This read-only diagnostic compares the two completed ERAF models against the previous deployed-action models and strongest control on the same 18 deterministic cup/pill initial and corrective observations (6 train and 12 replay holdout per model).

Run `scripts/probe_robotwin_balanced_postjoint_actions.py --source-root <completed balanced trial> --output <new path> --prior-diagnostic <deployed action fit check> --strongest-summary <expanded action fit strongest/summary.json> --seconds 240`.

Admission requires the completed 180-episode/29-method terminal receipt, all source processes gone, current actual model hashes matching that receipt, completed training audits, and idle GPUs 0 and 2. It reuses production 10-step actions with fixed noise 42, checks repeated inference and actual input hashes against archived references, and writes only JSON/logs. No updates or checkpoint files are created. Only the diagnostic's own process groups may be stopped at its deadline. No platform shutdown.

The comparison is action error, not closed-loop success; 18 observations cannot establish complete trajectory coverage or a unique failure cause. Source-language comparisons are diagnostics only and are not a proposed training objective. Training, datasets, CF metrics, and previous result files are unchanged.
