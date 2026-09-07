# Expanded Full Goal four-arm trial

This experiment tests whether actual failed-state corrections for cup and pill placement improve ten-task CF performance. The prepared bank retains every old row and adds 60 verified corrections: 24 training and 6 held-out scenes per new task. Left-placement and RGB corrections remain available. Preparation must complete the exact-target audit and fresh masks for all 168 correction scenes before this operator acquires GPUs.

Run `scripts/run_robotwin_expanded_fg_trial.py` with the preparation directory, its recorded PID, a new output directory and an absolute deadline. The operator verifies the source process command and kernel start time, all preparation exits, actual hashes and free GPUs. It stops only its own children at its deadline; it never shuts down the server. All checkpoints and optimizer files stay on the server.

## Prespecified training

The strongest retained action policy initializes every arm. Two retained semantic parents at cumulative step 1000 have exactly that action policy and zero output residual. Refresh each for 500 semantic optimizer steps, with the action policy frozen, then verify unchanged policy tensors and deployed OFF/full/repeated inference identity against the new manifest. Semantic batches contain six own-goal examples, three opposite-goal views of shared observations, and three matched FG or ordinary partial examples. The terminal diagnostic uses unchanged held-out queries and does not select checkpoints.

Train no-ERAF, FG-only, ERAF-only and ERAF+FG for 200 fresh joint optimizer steps, global batch 12. Every batch has two Correct retention, four CF retention, three shared expert and three FG or task/domain-matched ordinary target-only examples. Action learning rate is 3e-6; ERAF interface rate is 3e-5. The correction/replacement coefficient increases from 0.1 to 1 equally across arms. The actual-sample audit verifies 1,800 common examples and 600 matched replacement slots per arm, and identical batches within each FG setting.

This changes the data, semantic refresh and corrective coefficient together. It compares the resulting methods; it is not a data-only causal ablation or an equal-total-compute comparison. ERAF-off controls skip semantic computation that their deployed policy does not use. The no-FG arms never consume corrective pixels or actions.

## Evaluation and retention of evidence

Evaluate all four models on the same 45 development scenes each (180 total), using unchanged ten-task predicates and catalogs. Use exact equal-task macro CF, paired gains/losses and the supplementary strict burger slot metric. Retain every historical method in the previous comparison, including the strongest 35% control. The final semantic1500 and joint200 checkpoints are specified before evaluation; partial results cannot select a winner. Correct is not evaluated in this round, as authorized.

No development outcome alone establishes independent superiority. A promising winner still needs a frozen selection and independent test against actual strongest controls. Archive code commit, commands, input hashes, all optimizer-step journals, audit results and full episode evidence, including failures.
