# Expanded FG action diagnostic

The completed expanded60 four-arm trial gives ERAF+FG25%, ERAF30%, noERAF25%, FG-only23.33% ten-task DEV macro CF; strongest historical control remains35%. All added tasks have zero successful rollouts. These results do not support the larger correction weight or the current expanded recipe.

Before another optimization run, compare strongestR and all four final200 checkpoints on fixed cup/pill observations. Per task select one training and two replay-holdout scenes separately for ordinary and corrective data. Use the ordinary initial observation and the corrective start and middle observation, giving18 observations/model. Selection is deterministic and independent of prediction or success. No917 independent-test scenes are used.

Read original three-camera pixels and proprioception; freshly captured production video tokens must match prepared observations. Use only the current observation. References and real-action masks come from the audited replay cache. Record full10-step production CF, source-language and repeat outputs at fixed noise, their masked normalized action MSE for the first24 executable actions, and language-induced differences. Compare both instructions to the CF expert only to quantify language response; this does not provide a source expert at the corrective state.

Also measure random-time velocity reconstruction at sigma0.25/0.5 and pure-noise endpoint fit. Oracle-noised reconstruction includes target-action information and cannot show goal selection. A low action error is not CF success, and multiple valid trajectories may differ from this one expert. Compare training versus replay-holdout subsets and initial versus corrective states separately before drawing conclusions.

Models are frozen. Hash each checkpoint and the manifest before/after; record observation/reference hashes and exact IDs for cross-model pairing. Do not transfer weights or replay tensors to the Mac. Only code and JSON metadata are transferred. This diagnostic does not change training, checkpoint selection, evaluation predicates or shutdown policy.
