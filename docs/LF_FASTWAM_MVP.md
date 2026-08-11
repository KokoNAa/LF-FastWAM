# LF-FastWAM MVP

This tree extends the official FastWAM `main` source with the MVP described in
`LF_FastWAM_MVP_Implementation_Spec.md`.

## Implemented information flow

- Action Expert sequence is `[Q, A]` when `langforce_mvp.enabled=true`.
- Query RoPE positions use `query_rope_offset`; action positions remain `0..H-1`.
- Query modulation uses video/action-flow timestep zero; action modulation keeps
  the sampled action timestep.
- Posterior Query reads current-frame video, language, and optional proprio.
- Prior Query reads current-frame video and optional proprio, but no language.
- Action tokens read Query and action tokens, but neither raw video nor language.
- Query cannot read future-video or action tokens.
- Prior uses a separate first-frame-only, state-only video prefill/cache.
- Posterior and prior reuse the same noisy action, action timestep, flow target,
  and padding mask.
- Inference runs posterior only and retains the video KV cache.

All-masked cross-attention rows are made backend-safe before attention and their
projected contribution is explicitly zeroed. This prevents both NaNs and linear
projection bias from acting as a masked-context signal.

## Experiment modes

The default `configs/model/fastwam.yaml` is full M1.

| Mode | Required overrides |
|---|---|
| B0 FastWAM-FT | `model.action_dit_config.use_latent_action_queries=false model.langforce_mvp.enabled=false` |
| B1 Query only | `model.langforce_mvp.enable_prior=false model.langforce_mvp.enable_posterior_advantage=false` |
| M1 full MVP | no overrides |

`langforce_mvp.enabled=false` is a complete baseline switch: it restores direct
current-frame and language access even though the inactive MVP config block keeps
its safety flags at `false`.

## Server validation

Run these commands in the server copy after transferring this source tree:

```bash
cd /path/to/fast-WAM
pip install -e .
bash scripts/validate_lf_mvp_server.sh
```

The tests cover baseline ActionDiT numerical compatibility, Query packing and
output shape, self/cross-attention structure, prior language leakage, posterior
language reachability, old action-backbone compatibility, and Query checkpoint
round trips.

For a short GPU smoke test, use a weight-only FastWAM checkpoint so all three
runs start at training step zero:

```bash
RUN_TAG=lf-smoke \
bash scripts/train_lf_mvp_matrix.sh \
  8 \
  libero_uncond_2cam224_1e-4 \
  /path/to/common_starting_fastwam.pt \
  200 \
  42
```

### Four-GPU LoRA path

When only the LIBERO-Spatial shard is available, use the dedicated LoRA task.
It freezes the Wan/FastWAM base, injects rank-16 adapters into attention, FFN,
and text projections of both experts, and additionally trains the latent Query,
action input/output heads, and proprio projection. The default micro-batch is
one per GPU with four-way gradient accumulation (effective global batch 16).

```bash
bash scripts/train_lf_lora_spatial_smoke.sh \
  4 \
  ./checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  2 \
  42
```

After the two-step smoke test succeeds, run the controlled B0/B1/M1 matrix:

```bash
RUN_TAG=lf-spatial-lora \
bash scripts/train_lf_mvp_matrix.sh \
  4 \
  libero_spatial_lf_lora_2cam224 \
  ./checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  200 \
  42
```

For a controlled one-epoch run on the 53,229-example Spatial shard (effective
global batch 16), use 3,328 optimizer steps and retain intermediate adapters
every 500 steps:

```bash
RUN_TAG=lf-spatial-lora-1epoch-v1 \
bash scripts/train_lf_mvp_matrix.sh \
  4 \
  libero_spatial_lf_lora_2cam224 \
  ./checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  3328 \
  42 \
  save_every=500
```

LoRA runs save `fastwam_lora_adapter_v1` weight files containing only adapter
parameters and the selected small action modules. The adapter records the
absolute base-checkpoint path and automatically loads that base before applying
its deltas. DeepSpeed optimizer/model state saving is disabled for this task to
avoid duplicating the frozen multi-gigabyte base model.

Inspect the first M1 logs for finite values of:

```text
loss_video
loss_action_post
loss_action_prior
loss_posterior_advantage
post_vs_prior_loss_ratio
fraction_post_better_than_prior
```

For a four-GPU preliminary evaluation of all ten LIBERO-Spatial tasks, run the
controlled Correct/Null matrix. The default is five trials per task, ten
inference steps, and seed 42:

```bash
PYTHON_BIN=/opt/conda/bin/python \
bash scripts/eval_lf_lora_spatial_matrix.sh 4 5 10 42
```

This runs B0, B1, and M1 sequentially by condition while parallelizing tasks
across four GPUs, then writes `lf_mvp_summary.json` and `.csv`. The Spatial
tasks all move the same black bowl to the same plate and differ only in the
bowl's initial location. Consequently, they support a preliminary
Correct/Null comparison but do not provide valid shuffled hard negatives or a
meaningful DTL estimate.

Before a full evaluation of a longer checkpoint, use a Correct-only recovery
gate with one trial per task by setting `EVAL_CONDITIONS=correct` and overriding
the three checkpoint paths and `OUTPUT_ROOT`. If B1/M1 remain at zero, do not
spend compute on the full Correct/Null matrix yet.

## Correct / Null / Shuffled evaluation

Copy the example manifest, replace placeholders with exact LIBERO task names and
same-scene hard negatives, then validate it:

```bash
cp configs/eval/language_intervention_eval.example.jsonl \
  configs/eval/language_intervention_eval.jsonl
python scripts/validate_language_intervention_manifest.py \
  configs/eval/language_intervention_eval.jsonl
```

Example M1 runs:

```bash
python experiments/libero/run_libero_manager.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=/path/to/m1.pt \
  EVALUATION.dataset_stats_path=/path/to/dataset_stats.json \
  EVALUATION.instruction_condition=correct \
  EVALUATION.output_dir=./evaluate_results/lf_m1/correct

python experiments/libero/run_libero_manager.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=/path/to/m1.pt \
  EVALUATION.dataset_stats_path=/path/to/dataset_stats.json \
  EVALUATION.instruction_condition=null \
  EVALUATION.output_dir=./evaluate_results/lf_m1/null

python experiments/libero/run_libero_manager.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=/path/to/m1.pt \
  EVALUATION.dataset_stats_path=/path/to/dataset_stats.json \
  EVALUATION.instruction_condition=shuffled \
  EVALUATION.language_intervention_manifest=./configs/eval/language_intervention_eval.jsonl \
  EVALUATION.output_dir=./evaluate_results/lf_m1/shuffled
```

For B0 evaluation, add both B0 overrides from the experiment table. For B1,
add both B1 overrides. Keep simulator seeds and trial counts identical.

Aggregate B0/B1/M1 outputs:

```bash
python scripts/summarize_language_interventions.py \
  --run B0=./evaluate_results/lf_b0 \
  --run B1=./evaluate_results/lf_b1 \
  --run M1=./evaluate_results/lf_m1 \
  --output-prefix ./evaluate_results/lf_mvp_summary
```

The summary contains `SR_correct`, `SR_null`, language-reliance gap,
default-task leakage under shuffled language, optional counterfactual success
from externally predicate-correct results, and latency p50/p95.

Paired counterfactual rollout is intentionally rejected by the single-task
LIBERO runner: changing only the instruction while retaining the original task
success predicate would produce an invalid CIS number. It should be enabled only
with an alternate executable task and its corresponding success predicate.

## Acceptance gates

- `SR_correct` drop versus B0 is at most 2 percentage points.
- At least one of: LRG +10pp, DTL -15pp, or predicate-correct CIS +10pp.
- Correct-instruction performance remains stable.
- Inference p50 overhead is at most 15%.
