# PGC-FastWAM V9：Entity–Relation Affordance Field（ERAF）

## 1. 目标与贡献

PGC-FastWAM V9 在 V5 的“语言意图提取 + 反事实动作监督对齐”之外，引入独立的第二项贡献：**Entity–Relation Affordance Field（ERAF）**。

ERAF 将语言目标显式分解为最多四个谓词 clause，并在当前双相机 RGB 中分别定位 subject 与 reference，预测抓取、目标、交互 anchor、谓词当前状态和执行阶段，再把 clause-level 可执行表示注入冻结的 V5 ActionChunkProposal。它针对 LIBERO-10 中 V5 无法稳定处理的目标实体切换、方向/空间关系、关节状态和双物体合取目标。

部署端仍只读取 RGB、语言和 proprio。MuJoCo element mask、body pose、region、predicate truth 和 task phase 只存在于训练 sidecar 与离线评测中。

## 2. 模型结构

V9 保留一个不可变的 release FastWAM Base，以及冻结的 V5 GoalGraph、Goal Query Seeds 和保守 hard gate；不创建第二个 Action Expert，不使用 LoRA。

ERAF 包含三个部分：

1. `PredicateRoleDecoder`
   - 从 T5 token 解码最多四个 clause；
   - 输出 active、predicate、subject query 与 reference query；
   - 支持 `in/on/left/right/front/back/open/close/turnon/turnoff` 和多 clause。
2. `MultiViewEntityGrounder`
   - 使用 language-neutral 当前帧视觉 token；
   - 保留二维 patch 坐标和 camera ID；
   - 为每个 clause 的 subject/reference 分别产生双视角 heatmap、逐相机可见性/二维中心、实体 token 和归一化三维位置。
3. `RelationAffordanceReasoner`
   - 联合 clause、subject、reference、相对位置和预测的当前 predicate truth；
   - 输出 relation token、grasp/goal/interaction anchor 和 phase；
   - V5 Proposal 对 relation token 做 cross-attention。

ERAF 到 V5 Goal Query/Embedding 的两个 bridge 都是零初始化，因此 V5 checkpoint 第一次迁移到 V9 时，candidate 输出与 V5 数值一致。Base Expert、Video Expert、V5 GoalGraph 和 Verifier 在 grounding/action 阶段保持冻结；Verifier 只在最后校准阶段训练。

## 3. 数据合同

### 3.1 Strict-conflict manifest

`scripts/prepare_pgc_libero_strict_manifest.py` 会优先寻找 entity swap、relation swap、direction swap、articulated state、conjunction 与 compound conflict。每个候选必须通过：

- source 与 target goal 初始均为 false；
- 五条 source demo 达成 source、且不能达成 target；
- 五条 target demo 达成 target、且不能达成 source；
- 不通过即拒绝，不放宽为兼容目标；
- LIBERO-10 覆盖少于 8/10 时进程失败，禁止形成主结论。

### 3.2 ERAF sidecar

`scripts/build_pgc_libero_entity_relations.py` 为 native、历史 CF 和 strict CF 三个 LeRobot 数据集分别构建 `pgc_libero_entity_relation_v1` sidecar。每帧保存：

- target/source 两套 clause；
- predicate、subject/reference entity ID；
- 双相机 element mask、逐相机可见性与二维中心、实体三维位置；
- grasp、goal、interaction anchor；
- predicate truth 与 phase；
- episode/frame 对齐信息以及 state、action、sidecar SHA256。

sidecar 还必须声明并校验原始动作编码。release FastWAM native LeRobot
数据使用夹爪 `open=1/close=0`，而从 LIBERO HDF5 回放采集的 historical/strict
PGC 数据保留 MuJoCo 的 `open=-1/close=+1`。native sidecar 构建时先用
`g_env=1-2*g_fastwam` 回放；historical/strict 保持原值回放。训练加载器则只在
processor/normalizer 之前把 counterfactual 数据按
`g_fastwam=(1-g_env)/2` 对齐，原始 LeRobot 文件和 action SHA256 始终不改写。
因此三套数据进入 Action Proposal 时具有同一个 FastWAM 动作合同，同时仍可追溯
到磁盘上的原始采集记录。

BDDL region 必须从 `regions[name].target` 解析。加载训练数据时会再次校验 sidecar 文件 hash、LeRobot action hash、PGC 初始状态 hash、数组 shape/dtype/range、pair ID 和 strict 双向回放审计。

对于 LIBERO 将方向目标编码为 `on(subject, structural_region)` 的情况，V9 仍以 `regions[name].target` 确定真实 fixture 与 goal anchor，只使用指令中的显式 `left/right/front/back` 词恢复关系类别，并且只在 BDDL 已声明的 object/fixture catalog 中解析语义 reference；不会从 region 名称猜实体。

历史/strict counterfactual 轨迹执行的是 target 指令，因此只保留 source 的实体、关系、可见性、predicate truth 与静态 goal-anchor 监督；source 的 grasp/interaction anchor 和 phase 标签会显式置为无效，避免把 target 动作轨迹误标成 source 的执行监督。Native 轨迹保留完整执行标签。

### 3.3 采样比例

V9 action 训练池由数据加载器确定性构造：

- native : counterfactual = `1:1`；
- counterfactual 内 historical : strict = `1:1`；
- strict 内 conflict category 等频；
- 三个池必须非空且互不重叠。

## 4. LIBERO-10 数据构建

以下命令应在服务器仓库根目录执行。历史 manifest 必须与已经采集的 LIBERO-10 historical CF 数据逐 pair 对应。

```bash
cd /root/gpufree-data/LF-FastWAM

export MUJOCO_GL=egl
export PYTHON_BIN=/opt/conda/bin/python
export LIBERO_DATA_ROOT=/root/gpufree-data/fastwam/FastWAM/data/libero_mujoco3.3.2
export LIBERO_DEMO_ROOT=/root/gpufree-data/fastwam/third_party/LIBERO/libero/datasets

HISTORICAL_MANIFEST=/root/gpufree-data/pgc_libero_data_v1/manifests/libero_10_pgc_balanced_v2.jsonl
HISTORICAL_CF=/root/gpufree-data/pgc_libero_data_v1/libero_10_pgc_counterfactual_lerobot
V9_DATA_ROOT=/root/gpufree-data/pgc_libero_data_v1/v9/libero_10_seed42
BUILD_LOG=/root/gpufree-data/pgc_v9_libero10_data_build.log

nohup bash scripts/build_pgc_v9_libero_suite.sh \
  libero_10 \
  "$HISTORICAL_MANIFEST" \
  "$HISTORICAL_CF" \
  "$V9_DATA_ROOT" \
  42 \
  > "$BUILD_LOG" 2>&1 &

echo $! | tee /root/gpufree-data/pgc_v9_libero10_data_build.pid
tail -f "$BUILD_LOG"
```

构建结束后脚本会打印六个路径。正式训练需要其中五个：

```bash
STRICT_MANIFEST=$V9_DATA_ROOT/manifests/libero_10_strict_conflict.jsonl
STRICT_DATASET=$V9_DATA_ROOT/libero_10_pgc_strict_counterfactual_lerobot
NATIVE_SIDECAR=$V9_DATA_ROOT/sidecars/libero_10_native_eraf
HISTORICAL_SIDECAR=$V9_DATA_ROOT/sidecars/libero_10_historical_cf_eraf
STRICT_SIDECAR=$V9_DATA_ROOT/sidecars/libero_10_strict_cf_eraf

jq '{coverage, coverage_required, conflict_type_counts, uncovered}' \
  "$V9_DATA_ROOT/manifests/libero_10_strict_conflict.coverage.json"
```

## 5. 三阶段训练

### 5.1 公共环境

`V5_CKPT` 必须是 suite-specific LIBERO-10 V5 checkpoint，且它记录的 `base_checkpoint` 必须与下方 release Base 完全一致。

```bash
cd /root/gpufree-data/LF-FastWAM

export DIFFSYNTH_MODEL_BASE_PATH=/root/gpufree-data/fastwam/FastWAM/checkpoints
export LIBERO_DATA_ROOT=/root/gpufree-data/fastwam/FastWAM/data/libero_mujoco3.3.2
export TEXT_CACHE_DIR=/root/gpufree-data/LF-FastWAM/data/text_embeds_cache/libero

BASE=/root/gpufree-data/fastwam/FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224.pt
V5_CKPT=/absolute/path/to/libero_10_v5/checkpoints/weights/step_004000.pt
HISTORICAL_CF=/root/gpufree-data/pgc_libero_data_v1/libero_10_pgc_counterfactual_lerobot
V9_DATA_ROOT=/root/gpufree-data/pgc_libero_data_v1/v9/libero_10_seed42
STRICT_DATASET=$V9_DATA_ROOT/libero_10_pgc_strict_counterfactual_lerobot
NATIVE_SIDECAR=$V9_DATA_ROOT/sidecars/libero_10_native_eraf
HISTORICAL_SIDECAR=$V9_DATA_ROOT/sidecars/libero_10_historical_cf_eraf
STRICT_SIDECAR=$V9_DATA_ROOT/sidecars/libero_10_strict_cf_eraf
```

### 5.2 Stage 1：ERAF grounding，1500 steps

默认使用四卡与 `gradient_accumulation_steps=4`，有效 batch 为 16。若当前只
有三卡，可设置 `PGC_V9_GRADIENT_ACCUMULATION_STEPS=5`，得到最接近的有效
batch 15；这不会改变累计 step/checkpoint 合同。V9 仍保持 `model.lora.enabled=false`，
因为 grounding/action 阶段只训练 ERAF/Proposal sidecar，Base 与 V5 主体必须冻结。

```bash
GROUND_TAG=v9-libero10-eraf-grounding-1500-seed42-v1
GROUND_LOG=/root/gpufree-data/pgc_v9_libero10_grounding.log

nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 RUN_TAG="$GROUND_TAG" \
  bash scripts/train_pgc_v9_libero_stage.sh \
    libero_10 grounding 4 \
    "$BASE" "$V5_CKPT" \
    "$HISTORICAL_CF" "$STRICT_DATASET" \
    "$NATIVE_SIDECAR" "$HISTORICAL_SIDECAR" "$STRICT_SIDECAR" \
    42 full \
  > "$GROUND_LOG" 2>&1 &

echo $! | tee /root/gpufree-data/pgc_v9_libero10_grounding.pid
tail -f "$GROUND_LOG"
```

输出 checkpoint：

```bash
GROUND_CKPT=/root/gpufree-data/LF-FastWAM/runs/libero_pgc_2cam224/pgc-$GROUND_TAG/checkpoints/weights/step_001500.pt
```

Stage 1 结束后必须先过 grounding gate，失败时不得进入动作训练：

```bash
CUDA_VISIBLE_DEVICES=0 /opt/conda/bin/python \
  scripts/eval_pgc_v9_grounding_gate.py \
  --checkpoint "$GROUND_CKPT" \
  --output /root/gpufree-data/pgc_v9_libero10_grounding_gate.json \
  --num-samples 500 \
  --num-inference-steps 10 \
  --seed 42 \
  --device cuda:0 \
  --dtype bfloat16

echo "GROUNDING_GATE_EXIT=$?"
```

Gate 同时要求 subject/reference top-1 ≥80%、relation macro-F1 ≥90%、role-swap ≥90%、可见 goal anchor 中位误差 ≤5 cm、多 clause exact match ≥80%。失败返回码为 2。

### 5.3 Stage 2：Grounding–Action，新增 4000 steps

Stage 2 从累计 step 1500 开始，最终 checkpoint 为 step 5500。ERAF LR 为 `2e-5`，Proposal LR 为 `1e-4`。

```bash
ACTION_TAG=v9-libero10-eraf-action-4000-seed42-v1
ACTION_LOG=/root/gpufree-data/pgc_v9_libero10_action.log

nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 RUN_TAG="$ACTION_TAG" \
  bash scripts/train_pgc_v9_libero_stage.sh \
    libero_10 action 4 \
    "$BASE" "$GROUND_CKPT" \
    "$HISTORICAL_CF" "$STRICT_DATASET" \
    "$NATIVE_SIDECAR" "$HISTORICAL_SIDECAR" "$STRICT_SIDECAR" \
    42 full \
  > "$ACTION_LOG" 2>&1 &

echo $! | tee /root/gpufree-data/pgc_v9_libero10_action.pid
tail -f "$ACTION_LOG"

ACTION_CKPT=/root/gpufree-data/LF-FastWAM/runs/libero_pgc_2cam224/pgc-$ACTION_TAG/checkpoints/weights/step_005500.pt
```

### 5.4 Stage 3：Verifier calibration，新增 1000 steps

```bash
VERIFIER_TAG=v9-libero10-eraf-verifier-1000-seed42-v1
VERIFIER_LOG=/root/gpufree-data/pgc_v9_libero10_verifier.log

nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 RUN_TAG="$VERIFIER_TAG" \
  bash scripts/train_pgc_v9_libero_stage.sh \
    libero_10 verifier 4 \
    "$BASE" "$ACTION_CKPT" \
    "$HISTORICAL_CF" "$STRICT_DATASET" \
    "$NATIVE_SIDECAR" "$HISTORICAL_SIDECAR" "$STRICT_SIDECAR" \
    42 full \
  > "$VERIFIER_LOG" 2>&1 &

echo $! | tee /root/gpufree-data/pgc_v9_libero10_verifier.pid
tail -f "$VERIFIER_LOG"

V9_CKPT=/root/gpufree-data/LF-FastWAM/runs/libero_pgc_2cam224/pgc-$VERIFIER_TAG/checkpoints/weights/step_006500.pt
```

每 500 steps 保存一次 weights，不保存 DeepSpeed state。Stage 2/3 是 weights-only continuation，因此 Adam 与 scheduler 会按当前阶段重新初始化，这是预期行为。

### 5.5 两步四卡 smoke

环境变量 `PGC_V9_STAGE_STEPS=2` 只缩短当前阶段的实际 optimizer steps，不改变 continuation 起点。Grounding smoke 使用 V5；Action smoke 使用正式 step 1500；Verifier smoke 使用正式 step 5500。

```bash
PGC_V9_STAGE_STEPS=2 CUDA_VISIBLE_DEVICES=0,1,2,3 RUN_TAG=v9-grounding-smoke \
  bash scripts/train_pgc_v9_libero_stage.sh \
    libero_10 grounding 4 "$BASE" "$V5_CKPT" \
    "$HISTORICAL_CF" "$STRICT_DATASET" \
    "$NATIVE_SIDECAR" "$HISTORICAL_SIDECAR" "$STRICT_SIDECAR" \
    42 full
```

同样将 `grounding`/初始化 checkpoint 替换为 `action`/`$GROUND_CKPT` 和 `verifier`/`$ACTION_CKPT`，确认对应 loss、梯度范数和日志均有限。

## 6. 评测顺序

### 6.1 Base mode 与 Correct gate

```bash
export PGC_CHECKPOINT="$V9_CKPT"
export STATS_PATH=/root/gpufree-data/fastwam/FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json
export PGC_EVAL_SUITES='[libero_10]'
export PGC_MAX_POLICY_STEPS=600
export PGC_V9_ABLATION=full

# Base branch：应与 release FastWAM 保持同一不可变策略。
PGC_GATE_MODE=base \
OUTPUT_ROOT=/root/gpufree-data/LF-FastWAM/evaluate_results/pgc_v9_libero10_base_correct_seed42 \
bash scripts/eval_pgc_libero.sh 4 1 correct 42 10

# guarded Correct：先做 10/10 gate。
PGC_GATE_MODE=guarded \
OUTPUT_ROOT=/root/gpufree-data/LF-FastWAM/evaluate_results/pgc_v9_libero10_guarded_correct_gate_seed42 \
bash scripts/eval_pgc_libero.sh 4 1 correct 42 10

# gate 通过后再做每任务 5 trials。
PGC_GATE_MODE=guarded \
OUTPUT_ROOT=/root/gpufree-data/LF-FastWAM/evaluate_results/pgc_v9_libero10_guarded_correct_seed42_trials5 \
bash scripts/eval_pgc_libero.sh 4 5 correct 42 10
```

### 6.2 Historical raw CIS 与 strict CIS

```bash
HISTORICAL_MANIFEST=/root/gpufree-data/pgc_libero_data_v1/manifests/libero_10_pgc_balanced_v2.jsonl
STRICT_MANIFEST=$V9_DATA_ROOT/manifests/libero_10_strict_conflict.jsonl

PGC_GATE_MODE=guarded PGC_MANIFEST_PATH="$HISTORICAL_MANIFEST" \
OUTPUT_ROOT=/root/gpufree-data/LF-FastWAM/evaluate_results/pgc_v9_libero10_raw_cis_seed42_trials5 \
bash scripts/eval_pgc_libero.sh 4 5 counterfactual 42 10

PGC_GATE_MODE=guarded PGC_MANIFEST_PATH="$STRICT_MANIFEST" \
OUTPUT_ROOT=/root/gpufree-data/LF-FastWAM/evaluate_results/pgc_v9_libero10_strict_cis_seed42_trials5 \
bash scripts/eval_pgc_libero.sh 4 5 counterfactual 42 10
```

V9 评测会在 `ERAF_OVERLAY_DIR` 保存每次重规划的 subject/reference heatmap、predicate、grasp/goal/interaction anchor 和 phase 的 NPZ/PNG overlay。

## 7. 三 seed 主结果

正式结论使用 seed 42/43/44；每个 seed 独立训练，并对 Base/V9 的 correct、raw CIS、strict CIS 各跑每任务 5 trials。结果汇总器要求 18 个完全配对的 run：

```bash
/opt/conda/bin/python scripts/summarize_pgc_v9_evaluation.py \
  --run Base:correct:42:/path/base_correct_seed42 \
  --run V9:correct:42:/path/v9_correct_seed42 \
  --run Base:raw_cis:42:/path/base_raw_seed42 \
  --run V9:raw_cis:42:/path/v9_raw_seed42 \
  --run Base:strict_cis:42:/path/base_strict_seed42 \
  --run V9:strict_cis:42:/path/v9_strict_seed42 \
  --run Base:correct:43:/path/base_correct_seed43 \
  --run V9:correct:43:/path/v9_correct_seed43 \
  --run Base:raw_cis:43:/path/base_raw_seed43 \
  --run V9:raw_cis:43:/path/v9_raw_seed43 \
  --run Base:strict_cis:43:/path/base_strict_seed43 \
  --run V9:strict_cis:43:/path/v9_strict_seed43 \
  --run Base:correct:44:/path/base_correct_seed44 \
  --run V9:correct:44:/path/v9_correct_seed44 \
  --run Base:raw_cis:44:/path/base_raw_seed44 \
  --run V9:raw_cis:44:/path/v9_raw_seed44 \
  --run Base:strict_cis:44:/path/base_strict_seed44 \
  --run V9:strict_cis:44:/path/v9_strict_seed44 \
  --output /root/gpufree-data/pgc_v9_libero10_three_seed_report.json
```

所有成功率、target grasp/lift 和 per-task gate 都必须在每个 seed 独立通过；paired McNemar 在三 seed 的完全配对 episode 上汇总计算，并同时输出 Wilson 95% 区间。

其中 target grasp/lift 的分母只包含真实可抓取 object；`open/close/turnon/turnoff` 的 drawer、microwave、stove 等 unary fixture 不计作“未抓取目标”。

## 8. 消融

训练脚本最后一个参数支持：

- `full`：完整 ERAF；
- `entity-only`：只保留实体定位，不用 relation/anchor；
- `without-anchor`：保留实体与关系，不使用三维 anchor。

V5、V7 使用既有 checkpoint；V9 两个结构消融必须分别完成三阶段训练，评测设置与完整 V9 相同。checkpoint metadata 会记录消融结构，加载时配置不一致会直接失败。

## 9. 验证与 checkpoint 合同

本地或服务器回归测试：

```bash
cd /root/gpufree-data/LF-FastWAM
bash scripts/validate_pgc_server.sh
```

V9 checkpoint 格式为 `fastwam_policy_guard_v9`，并声明：

- `warm_start_contract=exact_pgc_v5_sidecars`；
- `grounding=predicate_entity_relation_affordance_field`；
- `privileged_supervision=training_only`；
- `deployment_inputs=rgb_language_proprio`；
- `policy_protection=single_immutable_base_plus_conservative_hard_gate`。

加载器严格验证 version、结构维度、温度、视觉比例、消融类型、Base 引用和所有 sidecar tensors；V5→V9 只允许新增 `entity_relation_affordance.*` 参数。Base、Video Expert 和冻结 sidecar 不会进入优化器。

LIBERO-10 通过后，使用同一脚本按 suite 独立构建和训练 Object、Spatial、Goal；在四个 suite 都通过之前，不启用联合训练。RoboTwin 后续复用同一“训练期实体/位姿标签，部署期纯视觉”合同。
