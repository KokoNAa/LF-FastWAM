# RoboTwin matched CIS result ledger

This file is the comparison ledger for same-scene RoboTwin language
intervention experiments.  Add each new method as a row only when it uses the
fixed protocol below and its run passes the complete matched audit.

## Fixed comparison protocol

- Task pair: `place_a2b_left` and `place_a2b_right`.
- Domain: `demo_clean`.
- Conditions: Correct, Shuffled/DTL, and Counterfactual/CIS.
- Pairing: exact ordered scene seeds and exact source/counterfactual
  instructions are shared across conditions.
- Instruction split: `unseen`.
- Evaluation seed: `42`.
- Episodes: 50 per direction and condition; 100 episodes per aggregate metric.
- Inference: 10 denoising steps, B0 action path for the released baseline.
- `Correct SR` and `CIS` are higher-is-better.  `DTL source SR` is the source
  goal success rate after replacing the instruction, so lower is better.
- `LRG` is `Correct SR - DTL source SR`; higher means the source behavior is
  more sensitive to the changed instruction, but it must be interpreted
  alongside CIS because disruption alone is not successful redirection.

## Main comparison table

Directional cells are reported as `Left / Right / Aggregate`.  All deltas are
percentage points relative to the released FastWAM B0 row.  DTL improvement is
defined as `64% - method DTL`, so a positive value is better.

| Method | Checkpoint / mode | N per direction | Correct SR ↑ (L/R/A) | DTL source SR ↓ (L/R/A) | LRG ↑ (L/R/A) | CIS ↑ (L/R/A) | ΔCorrect vs B0 | DTL improvement vs B0 ↑ | ΔCIS vs B0 ↑ | Run tag | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| FastWAM B0 (released baseline) | `robotwin_uncond_3cam_384.pt`, B0 | 50 | 92% / 92% / **92%** | 56% / 72% / **64%** | 36pp / 20pp / **28pp** | 2% / 4% / **3%** | 0pp | 0pp | 0pp | `robotwin_cis_cfcore_b0_clean_50_seed42` | Complete, matched 6/6 |
| Our method (pending) | TBD | 50 | — | — | — | — | — | — | — | TBD | Pending |

Baseline aggregate counts and Wilson 95% confidence intervals:

- Correct: 92/100, 92.0%, CI `[85.0%, 95.9%]`.
- DTL source goal: 64/100, 64.0%, CI `[54.2%, 72.7%]`.
- CIS: 3/100, 3.0%, CI `[1.0%, 8.5%]`.
- Under the Counterfactual condition, the source-goal rate is 64% and the
  counterfactual-goal rate is 3%.

The released baseline run used code commit `3a5b167`.  Its server result root
is:

```text
evaluate_results/robotwin/robotwin_uncond_3cam_384/
robotwin_cis_cfcore_b0_clean_50_seed42/
```

Earlier all-zero smoke runs are excluded because model overrides were lost at
the RoboTwin subprocess boundary and the B0 checkpoint was evaluated with
randomly initialized latent action queries.  Commit `9cd7f21` fixed the config
handoff and made this mismatch fail closed.

This ledger covers the controlled bidirectional spatial-relation pair only; it
must not be presented as performance over the full RoboTwin task suite.
