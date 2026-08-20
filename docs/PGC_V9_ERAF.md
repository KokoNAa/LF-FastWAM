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
- objective v2-v4 的 strict 池内 conflict category 等频；
- 三个池必须非空且互不重叠。

V9.4 grounding 改为在 native、historical CF、strict CF 三个池内部按
`pair_id/task` 等频采样，同时保持上述两个 `1:1` 外层比例。这样 LIBERO-10
的多 clause 和少数谓词不会被 native 的长轨迹数量淹没。

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

#### V9.1 gate-aligned grounding objective

最初的 V9 Stage 1 同时计算 sigmoid heatmap BCE/Dice 和 role-swap loss，但
role-swap 使用的是独立 sigmoid 响应的区域平均值；grounding gate 与部署端实际
使用的却是 temperature-softmax 后的 attention。两者不等价，因此可能出现训练
日志中的 mask/entity loss 已下降，但离线 gate 的 subject/reference top-1、
role-swap 和 multi-clause exact match 仍失败。

V9.1 将空间监督与 gate 的决策量严格对齐：

- 对 subject/reference 的 normalized attention 直接最小化 GT mask 内总概率质量的
  负对数；
- 使用同一 normalized attention 比较 own-role 与 wrong-role mask，并施加 `0.20`
  margin；
- 额外惩罚落在另一个角色独占区域的注意力，重叠区域不作为负样本；
- 将 BCE、Dice、visibility、view visibility 与 center loss 分开记录，避免聚合的
  `loss_pgc_v9_mask` 掩盖空间定位失败；
- checkpoint 记录 `grounding_objective_version=2` 及各空间权重，禁止与旧目标静默
  混用。

正式 V9.1 Stage 1 必须从相同的 V5 checkpoint 重新训练，不能从旧
`grounding_objective_version=1` 的 step 1500 继续。旧 checkpoint 仍可在显式使用
旧配置时加载，只用于复现实验。

```bash
GROUND_TAG=v9r1-libero10-eraf-grounding-1500-3gpu-seed42-v1
GROUND_LOG=/root/gpufree-data/pgc_v9r1_libero10_grounding_1500_3gpu.log

nohup env \
  CUDA_VISIBLE_DEVICES=0,1,2 \
  PGC_V9_GRADIENT_ACCUMULATION_STEPS=5 \
  PGC_V9_GROUNDING_OBJECTIVE_VERSION=2 \
  RUN_TAG="$GROUND_TAG" \
  bash scripts/train_pgc_v9_libero_stage.sh \
    libero_10 grounding 3 \
    "$BASE" "$V5_CKPT" \
    "$HISTORICAL_CF" "$STRICT_DATASET" \
    "$NATIVE_SIDECAR" "$HISTORICAL_SIDECAR" "$STRICT_SIDECAR" \
    42 full \
  > "$GROUND_LOG" 2>&1 &

echo $! | tee /root/gpufree-data/pgc_v9r1_libero10_grounding_1500_3gpu.pid
tail -f "$GROUND_LOG"
```

输出 checkpoint：

```bash
GROUND_CKPT=/root/gpufree-data/LF-FastWAM/runs/libero_pgc_2cam224/pgc-$GROUND_TAG/checkpoints/weights/step_001500.pt
```

训练中除原有指标外，还应持续检查：

- `loss_pgc_v9_attention_mask` 与 `loss_pgc_v9_role_overlap` 下降；
- `pgc_v9_subject_gt_attention_mass`、`pgc_v9_reference_gt_attention_mass` 上升；
- 两个 `pgc_v9_*_role_attention_margin` 转正并增大；
- `pgc_v9_role_swap_accuracy` 上升；
- `pgc_v9_role_swap_valid_fraction` 非零，确认该监督确实覆盖当前 batch。

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

V9.7 gate 同时要求 subject/reference full-mask top-1 ≥80%、relation macro-F1
≥90%、exclusive role accuracy ≥90%、exclusive evidence coverage ≥50%、可见 goal
anchor 中位误差 ≤5 cm、多 clause exact match ≥80%。失败返回码为 2。完整 mask
role-swap accuracy 继续输出作重叠诊断，但不再作为语义角色的硬准入项。

Gate 报告还包含 `role_residual_audit`，但该审计不会放宽上述准入标准。它使用
subject/reference mask 的逐 patch 交集构造 exclusive evidence，并分别报告：

- 完整 mask 和 exclusive mask 的 role-swap accuracy；
- 完整 mask 失败中有多少被 exclusive 判据恢复、仍然错误或缺少可分证据；
- mask IoU、subject/reference overlap fraction 和两种判据下的 margin 分布；
- 按 native/counterfactual、predicate 和任务指令拆分的同类指标。

`diagnosis=exclusive_role_gate_pass_full_mask_overlap_diagnostic` 表示 exclusive
角色准入已通过，但完整 mask 判据受到物体/容器重叠影响；它不会单独阻止进入
下一阶段。
`diagnosis=role_binding_generalization_failure` 表示 exclusive evidence 下仍会交换角色；
此时应增加显式 subject/reference assignment loss 和均衡 hard role-swap 采样，且不得
进入 Stage 2。

#### V9.7 exclusive-evidence calibration

V9.7 从 V9.6 objective-v7 step 3000 继续 250 个 grounding steps，仍只训练
`balanced_role_binding_adapter`。同一 clause 的 subject/reference mask 先删除逐 patch
交集，再计算 row/column assignment、hard/easy 全局梯度平衡和 multi-clause worst-role
loss。完整 mask 仍用于 BCE/Dice、attention mass 与 top-1 定位监督，因此不会通过
缩小 mask 来规避实体定位。

训练 stage 为 `grounding-exclusive-role`，objective version 为 8，默认学习率
`1e-5`，结束 step 为 3250。它继续使用 V9.3 审计得到的 hard-role index 和
V9.6 四池 curriculum。V9.7 checkpoint 额外声明：

- `eraf_role_evidence=exclusive_subject_reference_support`；
- `eraf_role_gate=exclusive_accuracy_with_full_mask_localization`；
- `eraf_exclusive_role_coverage_min=0.5`。

#### V9.2-B role-assignment repair

当 500-sample 审计确认 `role_binding_generalization_failure` 后，从 V9.1 step 1500
继续 1000 个纯 grounding steps，而不是回到 V5 重训。V9.2 使用 objective v3：

- 对 `subject-query → subject-entity`、`reference-query → reference-entity` 做 row CE；
- 对每个实体反向比较两个 query，增加 column CE，防止两个 query 塌缩到同一实体；
- 当前 assignment 错误的 clause 在线加权为普通 clause 的 3 倍；
- 保留 V9.1 attention-mask、role margin、mask、relation、anchor、position 和 phase
  监督；Base、V5 GoalGraph 和 Action Proposal 继续冻结；
- checkpoint 从 objective v2 单向迁移到 v3，禁止在 action/verifier 阶段偷偷改变
  grounding objective。

三卡正式续训：

```bash
V9R1_CKPT=/root/gpufree-data/LF-FastWAM/runs/libero_pgc_2cam224/pgc-v9r1-libero10-eraf-grounding-1500-3gpu-seed42-v1/checkpoints/weights/step_001500.pt
ROLE_TAG=v9r2-libero10-role-assignment-1000-3gpu-seed42-v1
ROLE_LOG=/root/gpufree-data/pgc_v9r2_libero10_role_assignment_1000_3gpu.log

nohup env \
  CUDA_VISIBLE_DEVICES=0,1,2 \
  PGC_V9_GRADIENT_ACCUMULATION_STEPS=5 \
  PGC_V9_GROUNDING_OBJECTIVE_VERSION=3 \
  PGC_V9_ROLE_ASSIGNMENT_WEIGHT=4.0 \
  PGC_V9_ROLE_ASSIGNMENT_TEMPERATURE=0.10 \
  PGC_V9_ROLE_ASSIGNMENT_HARD_WEIGHT=2.0 \
  RUN_TAG="$ROLE_TAG" \
  bash scripts/train_pgc_v9_libero_stage.sh \
    libero_10 grounding-role 3 \
    "$BASE" "$V9R1_CKPT" \
    "$HISTORICAL_CF" "$STRICT_DATASET" \
    "$NATIVE_SIDECAR" "$HISTORICAL_SIDECAR" "$STRICT_SIDECAR" \
    42 full \
  > "$ROLE_LOG" 2>&1 &

echo $! | tee /root/gpufree-data/pgc_v9r2_libero10_role_assignment.pid
tail -f "$ROLE_LOG"
```

V9.2 grounding checkpoint 的累计 step 为 2500。训练时必须同时监控
`loss_pgc_v9_role_assignment`、assignment row/column/total accuracy、hard fraction、
原 `pgc_v9_role_swap_accuracy`、两个 top-1 hit 和 anchor loss。之后仍运行相同的
500-sample grounding gate；objective v3 gate 只接受 step 2500。

#### V9.4 cross-clause structured-role repair

V9.3 的局部 role adapter 已把 subject/reference top-1 修复到较高水平，但最终
500-sample gate 在 role-swap `77.75%`、multi-clause exact `63.61%` 附近平台。
原因是每个 clause 只与自己的另一个 role 比较，仍可能绑定到其他 clause 的实体。

V9.4 从已完成的 V9.3 objective-v4 step 2500 单向迁移：

- 冻结完整 V9.3 ERAF，仅训练第二个零初始化 structured-role adapter；
- 在最多八个 subject/reference slot 之间做跨 clause self-attention；
- 每个 role 与同状态所有不同 entity ID 的 mask 对比；相同 entity ID（例如两个
  clause 共用 basket）作为合法共享目标，不构造假负样本；
- 对多 clause 样本额外优化最差 role，而非让容易 clause 抵消失败；
- 三个数据池内部按 task/pair 等频，同时严格保持 native:CF 和 historical:strict
  两级 `1:1`；
- 新输出层为全零，迁移第 0 步与 V9.3 数值完全一致。

正式训练新增 1000 steps，累计到 step 3500：

```bash
V9R3_CKPT=/absolute/path/to/v9r3/checkpoints/weights/step_002500.pt
V94_TAG=v9r4-libero10-structured-role-1000-4gpu-seed42-v1
V94_LOG=/root/gpufree-data/pgc_v9r4_libero10_structured_role_1000_4gpu.log

nohup env \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  PGC_V9_GROUNDING_OBJECTIVE_VERSION=5 \
  RUN_TAG="$V94_TAG" \
  bash scripts/train_pgc_v9_libero_stage.sh \
    libero_10 grounding-structured-role 4 \
    "$BASE" "$V9R3_CKPT" \
    "$HISTORICAL_CF" "$STRICT_DATASET" \
    "$NATIVE_SIDECAR" "$HISTORICAL_SIDECAR" "$STRICT_SIDECAR" \
    42 full \
  > "$V94_LOG" 2>&1 &
```

训练中重点监控 `loss_pgc_v9_structured_assignment`、
`loss_pgc_v9_multi_clause_consistency`、`pgc_v9_structured_assignment_accuracy`、
`pgc_v9_structured_multi_clause_accuracy` 与两个 structured adapter delta norm。
step 2750/3000/3250 可使用 `--allow-intermediate` 审计；只有 step 3500 的六项
grounding gate 全部通过，才允许进入 action stage。

### 5.3 Stage 2：Grounding–Action，新增 4000 steps

V9.1 Stage 2 从累计 step 1500 开始，最终 checkpoint 为 step 5500；V9.2/V9.3
从 step 2500 开始并最终保存为 step 6500；V9.4 从 step 3500 开始并最终保存为
step 7500。ERAF LR 为 `2e-5`，Proposal LR 为 `1e-4`，启动命令必须显式设置与
grounding checkpoint 一致的 `PGC_V9_GROUNDING_OBJECTIVE_VERSION`。

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
