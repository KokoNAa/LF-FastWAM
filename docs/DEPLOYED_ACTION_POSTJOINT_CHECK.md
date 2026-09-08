# Fixed-state check after deployed-rollout training

Use GPUs 0 and 2 only after both ERAF action workers exit 0 and while the same deployed-action primary controller remains in joint training with at least 30 updates remaining in each control. Budget: 240 seconds. This sidecar terminates only its own probes and yields if the primary stage changes. It neither changes the primary runtime nor writes its evaluation directory.

Reuse the existing production 10-step action diagnostic on the same 18 deterministically selected cup/pill observations: initial expert and initial/middle FG correction states, one training and two held-out scenes per task/kind. Compare each final200 ERAF model with its previous initial-anchor objective model and the strongest R control using archived reports. Bind manifests, checkpoints, primary protocol and all reference reports by SHA256; compare actual RGB, proprio, instructions, expert references and validity masks before aggregation. Archive the two full completed-stage parameter and sampling audits in this sidecar.

This tests whether direct deployed-action supervision improves actual output action fit. Flow/endpoint errors remain supplementary diagnostics. It cannot establish CF success, select a checkpoint, prove independent-test performance or justify shutdown. No weights or optimizer files are transferred to the Mac.
