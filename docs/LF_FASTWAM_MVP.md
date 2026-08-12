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

## DTL / CIS paired evaluation

The paired runner uses LIBERO-Object because each source scene contains one
basket and several manipulable objects. The manifest builder chooses a
different object that is present in the source scene, keeps the source
simulator and initial state unchanged, and imports only the paired task's BDDL
goal predicate for the `counterfactual` condition:

```bash
python scripts/prepare_libero_object_interventions.py \
  --output configs/eval/libero_object_dtl_cis.jsonl
python scripts/validate_language_intervention_manifest.py \
  configs/eval/libero_object_dtl_cis.jsonl
```

Run a one-trial gate on four GPUs before the formal five-trial matrix:

```bash
PYTHON_BIN=/opt/conda/bin/python \
bash scripts/eval_lf_lora_object_dtl_cis.sh 4 1 10 42
```

The three conditions have distinct semantics:

- `correct`: source instruction and source success predicate (`SR_correct`).
- `shuffled`: paired alternate instruction but source success predicate
  (`DTL_shuffle`, lower is better).
- `counterfactual`: the same paired instruction and alternate BDDL goal
  predicate (`CIS`, higher is better).

The counterfactual path fails closed unless the paired BDDL uses the same
LIBERO environment class, has a different goal, and every goal operand exists
in the instantiated source scene. This prevents an instruction-only swap from
being reported as CIS while still checking the original predicate.

### Object-only B0/M1 training

For a domain-controlled conclusion without mixing suites, download only the
official FastWAM LIBERO-Object archive and extract it under the configured data
root. Precompute the ten Object instruction embeddings before training:

```bash
huggingface-cli download yuanty/LIBERO-fastwam \
  libero_object_no_noops_lerobot.tar.gz \
  --repo-type dataset \
  --local-dir ./data/libero_mujoco3.3.2

tar -xzf ./data/libero_mujoco3.3.2/libero_object_no_noops_lerobot.tar.gz \
  -C ./data/libero_mujoco3.3.2

torchrun --standalone --nproc_per_node=4 scripts/precompute_text_embeds.py \
  task=libero_object_lf_lora_2cam224 \
  +overwrite=false
```

Train B0 and M1 from the same released base, with the same seed, rank-16 LoRA,
effective global batch 16, and exactly one Object epoch. The optimizer-step
count is derived from the extracted dataset length:

```bash
RUN_TAG=lf-object-lora-1epoch-v1 \
STATS_PATH=./checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
bash scripts/train_lf_lora_object_b0_m1.sh \
  4 \
  ./checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  42
```

The paired evaluation script now defaults to these Object-only B0/M1 runs and
automatically selects their final `step_*.pt` adapters:

```bash
bash scripts/eval_lf_lora_object_dtl_cis.sh 4 1 10 42
```

Only expand the gate from one to five trials after both models recover high
Object `SR_correct`. Report `DTL_shuffle` and CIS only alongside that recovery
check; a low DTL from a model with collapsed Correct success is not evidence of
language adherence.

## Acceptance gates

- `SR_correct` drop versus B0 is at most 2 percentage points.
- At least one of: LRG +10pp, DTL -15pp, or predicate-correct CIS +10pp.
- Correct-instruction performance remains stable.
- Inference p50 overhead is at most 15%.
