# Recover CF evaluation after platform shutdown

On 2026-09-08 around 08:43 HKT the platform showed the six-card `wam 副本`
instance shut down and a negative available balance. The user replenished the
balance and authorized continuing the existing experiment. All four final
200-step weights and training audits had already completed before interruption.

`scripts/recover_robotwin_cf_evaluation.py` creates a new evaluation output. It
does not train, alter weights, change the original runtime checkout, or edit the
interrupted output. Before admission it checks source processes, all training
and audit exits, actual final weight SHA256 and file identities, the data
manifest, and the original committed runtime.

Complete task/arm cells are retained only after checking the exact checkpoint,
deployment, scene catalog, instructions, initial-state hashes, and counts. These
directories are linked into the new output. Incomplete cells are rerun in full
in the new output: their partial records remain in the interrupted source and
are not selectively merged. No seed or successful result is selected based on
outcome. The final comparison uses the same 180-episode matrix and all 21
historical/current methods.

Final audits rehash all four weights and all retained episode files. A process
deadline and a three-GiB disk reserve apply only to this recovery controller and
its children; the platform is never shut down by the controller.

Run `--plan-only` on the server first to inspect actual retained/pending cells.
Use the existing source runtime with `--runtime`; run the recovery controller
from its separately committed checkout. The exact source/runtime/output and
future deadline must be recorded in the launch receipt. No models are downloaded
to the Mac. A recovered evaluation is still a development comparison, not proof
of superiority or goal completion.
