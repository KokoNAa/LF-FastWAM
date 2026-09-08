# balanced-target200 formal five-task evaluation

User confirmed matched no_eraf, eraf_only and eraf_fg final200 checkpoints.
Five original tasks, 40 new physical scenes each, three models: 600 CF episodes.
The task subset is a user-selected scope after ten-task development results;
this is not a ten-task result or an unseen-task generalization claim.
No Correct evaluation or new training. Model selection is frozen before scenes.

All outputs and existing full weights remain on the server data disk. Videos,
original episode JSONL, object events, episode/cell CSV and paired CF gains/losses
are retained. Never download weights or optimizers to the laptop.

Catalog: seeds 91750000 + 1000 * task index, up to 400 attempted seeds/task.
Both experts must solve their goals from identical physical starts, initially
neither goal true. No learned policy screens scenes. Historical run manifests
and episode catalogs are scanned and hash-listed in frozen_protocol.json.
Any prior record in this namespace aborts before collection. Checkpoint and
input hashes are verified before and after evaluation. Each model uses the
same exact seed, instructions and initial-state hash, with 10 denoise steps,
32-action horizon, 24-action replan horizon and normal ERAF memory carry.

Metrics are read-only, sampled inside the existing physics-step goal check.
They add no environment steps and do not change the benchmark termination.
Object identity uses SAPIEN scene IDs: colored blocks can share the name box.

- Correct lift: correct instruction object, contact with at least two distinct
  finger links of one arm, nonzero contact impulse, gripper command <0.8,
  object center >=3cm above its initial height for >=0.10 simulation seconds.
  Any-object, all-object and per-object results are distinct. Sorting can be
  solved while leaving an already correct object unmoved; all-object lift is
  diagnostic, never a prerequisite for benchmark CF success.
- Placement after lift: at least one verified correct lift happened earlier,
  all selected goal relations hold, both grippers open, no finger contact,
  table objects within 2cm of initial height. Stack top uses benchmark vertical
  tolerance; the base additionally must be within 2.5cm of the expert's declared
  center. Burger/fries additionally must be nearer their intended slot than
  the opposite slot, with functional-point vertical difference <3cm.
- Existing CF success and CF-after-lift remain separate from this stricter
  physical placement diagnostic. Per-object placement and final placement are
  recorded. Conditional placement rate divides by episodes with any correct
  lift; denominator zero is null. No extra settling/stability is implied.

API smoke runs use a separate 91690000 namespace and expert render hooks to
validate actual contacts/IDs. These are excluded from formal outcomes. Formal
policy metrics use exactly one sample per physics-step success check; final
report generation does not add a second physical sample.

Run scripts/run_robotwin_formal_five40.py with --output on the server data disk,
--deadline an absolute timestamp, --gpus 0 1 2 3 4. The controller first freezes
models/exclusions, completes smoke, builds the catalog, runs the fixed matrix,
and verifies all 600 records. A failed child stops owned children and preserves
partial records without reporting them as complete. It never shuts down the
platform or touches unrelated jobs. Existing paused800 training stays paused.

Postprocessing: scripts/report_robotwin_formal_five40.py only accepts a complete,
verified600 matrix. It reports both the original CF-after-lift conditional rate
and the stricter physical-release confirmation rate, using the same lift
population as denominator. Strict nonconfirmation alone is not a placement
failure. This report preserves every frozen outcome and scoring threshold;
its source can be updated without altering the running evaluation checkout.
