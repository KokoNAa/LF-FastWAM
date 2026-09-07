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
