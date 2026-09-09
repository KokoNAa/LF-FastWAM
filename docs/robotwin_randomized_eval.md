# Latest three-model randomized evaluation

Evaluate the unchanged latest no-ERAF, ERAF200 and parent-teacher FG200 checkpoints on five tasks, 12 fresh randomized development scenes per task and model (180 CF episodes). This is evaluation only. There is no required performance ordering, checkpoint reselection, training, or platform shutdown.

`scripts/run_robotwin_randomized_eval.py` pins the three checkpoint hashes and the installed `demo_randomized.yml`. Both source and counterfactual experts must solve each catalog scene before any learned policy sees it. All three models use the same catalog, instructions, physical starts and raw initial camera/state inputs. The existing physical pickup/placement metrics and videos are retained. The randomized configuration includes backgrounds, clutter, lighting and table-height variation; camera-position randomization follows the installed configuration (currently zero).

The five seed ranges start at 91385000, 91386000, 91387000, 91388000 and 91389000, with at most 400 consecutive attempts each. These reserved development ranges must be unused in the archived scene inventory. Rejected scenes remain in the expert-screening log. No scene selection uses learned-policy outcomes. This is not an independent formal 40-episode test. Clean/random rate differences use distinct scene sets and must not be presented as paired domain effects.

Use a fresh output under the server data disk and an isolated committed worktree. The controller schedules at most one worker per selected GPU. It records process identities, fails on incomplete cells or changed input hashes, checks every video's metadata and physical event records, verifies initial RGB/state identity across models, and emits the complete paired report. It stops its own children on an error or signal. There is no wall-clock cutoff; each catalog and evaluation is bounded by its declared scene/episode count.

The generic evaluator and expert catalog generator now accept `--task-config demo_clean|demo_randomized`; clean remains the default for existing callers. Completion records include the actual task configuration, domain-randomization settings and configuration-file hash.
