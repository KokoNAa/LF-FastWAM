# ERAF-only gradients for full-goal correction examples

Experimental option: `--fg-gradient-route eraf_only`. Default `joint` retains
the historical training behavior. This option requires ERAF on and FG active;
it is not applicable to the ordinary-CF control.

Motivation is the complete 2026-09-07 CF-priority development screen: ordinary
CF200 achieved21/30, FG20017/30 and ERAF+FG20017/30. The same ERAF+FG200 weights
achieved16/30 when ERAF was bypassed. ERAF therefore had mixed, slightly positive
net effects in those scenes, while the FG-trained shared policy remained below
the ordinary-CF candidate. These small development results motivate a mechanism
test; they do not establish gradient conflict or future improvement.

The new route sends only `fg_correction` example gradients to currently
trainable ERAF parameters. In the declared interface/joint action stages those
are the interfaces, with semantic predictors frozen. Expert-pair and retention
losses retain their normal gradient routes. Inputs, teachers, FG targets, masks,
loss coefficients, sample order and inference are unchanged. Existing policy
gradients accumulated by earlier examples in the same batch are preserved.
No gradients are subtracted from `.grad`, and no optimizer step is added.

Production uses autograd's explicit backward inputs. The read-only gradient
observer selects the same guard parameter receivers. The route is stored in
the training plan, optimizer contract and checkpoint provenance. Resuming an
optimizer with a changed route is rejected; historical missing fields mean
`joint`. A new routing arm must use a fresh optimizer, so old FG momentum is
not mistaken for gradients from the new route.

Validation uses the actual checkpointed ERAF forward path. It checks that FG
retains exactly the same ERAF gradients, leaves pre-existing policy gradients
untouched, ordinary supervision still trains the policy, and the gradient
observer matches routed training without parameter or `.grad` mutation.

Before activation, inspect the current-recipe gradient diagnostics. A useful
subsequent comparison has four arms from the same policy/data parent: ordinary
CF, FG-only, standard ERAF+FG, and routed ERAF+FG. The two ERAF arms can share
an identical policy-frozen interface warmup because the routing options are
equivalent during that stage. Then use identical joint-step counts, sampling,
loss weights, policy/interface rates and checkpoint selection for the two ERAF
arms. Keep the ordinary and FG-only controls' normal routes; report parameter
and warmup differences. Evaluate complete CF goals, original-five gains/losses,
all ten tasks, and a same-checkpoint ERAF bypass. Correct has no hard threshold.
Use fresh independent scenes after development selection. No efficacy result
is claimed for this option until those experiments actually run.

This branch is separate from the running `42c7b42` diagnostics/data preparation.
Do not change that checkout's HEAD between its frozen cache plan and merge.

## Declared mechanism trial, 2026-09-07

The completed current-recipe diagnostic recovered exactly the executed sample
and noise schedule at step200, with no optimizer update. In three batches,
ERAF+FG Action FG/CF-retention mean gradient norms were0.082688/0.015122,
and ERAF interface norms were0.126217/0.010286. FG versus CF retention had
negative cosine in2/3 Action batches and3/3 interface batches. This motivates
lowering the correction coefficient to0.1 alongside the routing comparison;
it does not establish closed-loop improvement or AdamW update direction.

`run_robotwin_fg_routing_trial.py` declares four fresh-optimizer arms at200
steps: ordinary CF, FG-only, standard ERAF+FG, and routed ERAF+FG. All use
coefficient0.1, the existing five-task bank, primary2/4/3/3 global mixture,
policy LR3e-6, ERAF interface LR3e-5, seed42, and no language augmentation.
The ERAF arms reuse the exact same archived interface100 parent (its warmup
used the original coefficient1 and costs100 separately reported steps).
Controls start from all5_off200. Two ERAF arms each use two GPUs; controls
use one each. Global example/noise schedules match, while floating-point
reduction order and elapsed compute can differ across world sizes.

Step200 is selected in advance; intermediate saves every50 steps are only
recovery checkpoints for continuation across bounded power windows.
The GPU queue evaluates completed models while other training continues:
all original5 tasks at6 CF episodes each and extra5 at3 each,180 episodes
total. Compare initial states with the strongest archived ordinaryCF200,
whose original5 score is21/30 and equal-task ten-task macro is35.0%.
Do not replace that comparator with a weaker new control. The script does
not perform an independent test or ERAF bypass; those remain required
follow-up evidence for a promising candidate. It stops at the explicit
work cutoff and preserves partial records. Formal80-scene data expansion
is kept separate from this mechanism experiment.

Use `--resume-trial <stopped-source-root>` with a new output directory and
new explicit deadline to continue across power windows. The runner rejects
active producers, missing terminal records, changed recipe fields, changed
routes/world sizes, or changed checkpoint file identities. The action trainer
then validates FP32 master tensors against saved weights and restores AdamW
moments. It advances the deterministic sampler to the saved step and retains
the total target200. The policy is in eval mode during adapter training, and
sample/noise seeds are explicit per global step; dropout RNG is not active.
Already completed200-step arms are evaluated without additional updates.
Original journals remain intact; updates logged after the last save are
reported as discarded work and never added to the inherited optimizer budget.
