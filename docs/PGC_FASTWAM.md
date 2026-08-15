# PGC-FastWAM v1

PGC（Policy-Guarded Counterfactual FastWAM）用于在不破坏原始 FastWAM 策略的前提下提升语言遵守，尤其针对 CIS 中“仍执行原指令 / 改变行为但抓错物体 / 什么也不抓”三类失败。

## 架构

PGC 包含两条物理隔离的动作路径：

1. **Base Policy**：官方 query-free FastWAM Video Expert + Action Expert。加载 release checkpoint 后全部冻结，不进入优化器。
2. **Counterfactual Policy**：从 Base Action Expert 初始化的一份独立、完整 Action Expert。该分支可以全量微调，并由 Goal Graph Queries 驱动。
3. **Goal Graph Encoder**：少量 goal slots 先读取语言，再读取当前帧视觉 token，产生状态相关的目标表示和 Action Query。它不使用 task-ID prototype，也不能读取未来帧。
4. **Action–Outcome Verifier**：在当前状态和目标表示下分别给 Base/Counterfactual action chunk 打分。
5. **Conservative Hard Gate**：只有当反事实候选分数达到绝对阈值，且比 Base 高出安全 margin 时才覆盖；否则逐元素原样返回 Base action。

`policy_guard.enabled=false` 时不会构建或执行 PGC 路径，旧 FastWAM/TC/LangForce 行为保持不变。PGC v1 与 TC、LangForce、Base LoRA 互斥。

## 监督信号

- `loss_pgc_action`：独立 Counterfactual Action Expert 对当前指令对应真实 action chunk 的 flow-matching 正监督。
- `loss_pgc_verifier`：用真实 action 作为正样本，并根据 Base/Counterfactual 候选到真实 action 的距离生成连续质量标签。这样不会把“其实正确的 Base 候选”强行标成负样本。
- `loss_pgc_verifier_ranking`：仅在直接反事实样本中、且 Counterfactual 候选确实比 Base 更接近真实 action 时，要求其评分超过 Base。
- `loss_pgc_goal_action_alignment`：Goal-State representation 与真实 action outcome 的对称对比对齐；同指令样本不会互相作为负样本。
- `loss_video` 与 `loss_pgc_base_action_monitor`：只用于监控，不参与优化。

训练日志还会记录候选质量目标、Verifier 分数、预测覆盖率、Goal Query 多样性及原策略冻结标记。

## 为什么必须准备新数据

仅将旧轨迹换成另一条文本不是反事实动作正监督。PGC 要求在**源任务当前状态**下，以替代指令执行并成功完成替代目标的真实轨迹。每个直接反事实 LeRobot 数据目录必须包含：

- `meta/pgc_provenance.json`：数据集级任务配对与监督来源；
- `meta/pgc_episodes.jsonl`：逐 episode 的源初始状态索引、初始状态 SHA-256、配对 ID 与成功谓词审计。

并通过：

```bash
python scripts/validate_pgc_counterfactual_datasets.py \
  /path/to/direct_counterfactual_lerobot
```

最小 provenance 示例：

```json
{
  "format": "pgc_counterfactual_actions_v1",
  "benchmark": "libero",
  "action_supervision": "executed_counterfactual_success_trajectory",
  "state_aligned": true,
  "successful_only": true,
  "successful_episode_count": 100,
  "source_suites": ["libero_object"],
  "pairs": [
    {
      "pair_id": "libero_object_00_to_01",
      "source_suite": "libero_object",
      "source_task_id": 0,
      "source_instruction": "pick up the alphabet soup and place it in the basket",
      "counterfactual_instruction": "pick up the cream cheese and place it in the basket",
      "counterfactual_goal_state": [["in", "cream_cheese_1", "basket_1_contain_region"]]
    }
  ]
}
```

逐 episode 审计示例：

```json
{"episode_index": 0, "pair_id": "libero_object_00_to_01", "source_initial_state_index": 3, "initial_state_sha256": "<64 lowercase hex characters>", "initial_state_match": true, "counterfactual_goal_satisfied": true}
```

任务文件 `meta/tasks.jsonl` 中的 instruction 必须是实际执行的替代指令；失败轨迹不能混入。

## LIBERO 全 suite 训练

创建文本文件（每行一个直接反事实数据目录）：

```text
/data/pgc/libero_spatial_cf_lerobot
/data/pgc/libero_object_cf_lerobot
/data/pgc/libero_goal_cf_lerobot
/data/pgc/libero_10_cf_lerobot
```

先为原始与反事实指令统一生成 T5 cache（`CF_JSON` 是 Hydra 可接受的目录数组）：

```bash
CF_JSON=$(/opt/conda/bin/python - /path/to/pgc_counterfactual_datasets.txt <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
items = []
for line in p.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#"):
        q = Path(line).expanduser()
        items.append(str((p.parent / q).resolve() if not q.is_absolute() else q.resolve()))
print(json.dumps(items, separators=(",", ":")))
PY
)

CUDA_VISIBLE_DEVICES=0 \
/opt/conda/bin/python scripts/precompute_text_embeds.py \
  task=libero_pgc_2cam224 \
  "data.train.pgc_counterfactual_dataset_dirs=${CF_JSON}" \
  +overwrite=false
```

训练命令：

```bash
export DIFFSYNTH_MODEL_BASE_PATH=/path/to/FastWAM/checkpoints
export LIBERO_DATA_ROOT=/path/to/FastWAM/data/libero_mujoco3.3.2

CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/train_pgc_libero.sh \
  4 \
  "$DIFFSYNTH_MODEL_BASE_PATH/fastwam_release/libero_uncond_2cam224.pt" \
  /path/to/pgc_counterfactual_datasets.txt \
  42 \
  4000
```

该入口会硬性检查：四个原始 LIBERO suite、四个 suite 的直接反事实覆盖、逐 episode provenance、任务文本缓存、release full checkpoint 和 GPU 数量。默认对 Counterfactual 数据做 2 倍采样，学习率为 `1e-5`，每卡 micro-batch 为 1、梯度累积为 4（四卡有效 batch 16），每 500 steps 保存权重。默认不保存 DeepSpeed state，以避免再次因磁盘空间不足在最后一步失败。

代码拉取后先在服务器执行完整回归：

```bash
bash scripts/validate_pgc_server.sh
```

输出 checkpoint 格式为 `fastwam_policy_guard_v1`，只保存独立 Action Expert、Goal Graph、Verifier 和外部 Base checkpoint 路径，不复制 6.8B Base 权重。

## 评估

Correct 全四 suite：

```bash
export PGC_CHECKPOINT=/path/to/step_004000.pt
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/eval_pgc_libero.sh 4 5 correct 42 10
```

先用强制 Base 门控做“原策略逐 action chunk 完全一致”的保护对照：

```bash
PGC_GATE_MODE=base \
OUTPUT_ROOT=/path/to/evaluate_results/pgc_base_exact_correct \
bash scripts/eval_pgc_libero.sh 4 5 correct 42 10
```

正式 guarded 门控可通过 `PGC_GATE_THRESHOLD` 和
`PGC_MIN_COUNTERFACTUAL_SCORE` 调整；必须在独立校准集上选择阈值，不能用最终
CIS 测试集调参。

Shuffled / CIS 必须提供经过审计且覆盖所选 suite/task 的 intervention manifest：

```bash
export PGC_MANIFEST_PATH=/path/to/audited_libero_interventions.jsonl
bash scripts/eval_pgc_libero.sh 4 5 shuffled 42 10
bash scripts/eval_pgc_libero.sh 4 5 counterfactual 42 10
```

可用 `PGC_EVAL_SUITES='[libero_object]'` 单独测试某个 suite。每个任务结果会额外保存：

- `policy_guard_decision_count`
- `policy_guard_override_count`
- `policy_guard_override_rate`
- `policy_guard_base_score_mean`
- `policy_guard_counterfactual_score_mean`
- 每个 episode、每次 replan 的选择和分数

正确的验收顺序是：先确认 `PGC_GATE_MODE=base` 与 release FastWAM 的逐 action chunk 输出一致，再检查 guarded 模式下四个 suite 的 Correct SR 是否保持，然后看 Shuffled 的 DTL，最后看 CIS 和覆盖率。CIS 上覆盖率为零说明 Gate 太保守；覆盖率高但 CIS 仍低说明 Counterfactual Action Expert 或目标表示仍未学对；Correct 上覆盖率高且 SR 下降说明 Gate 校准失败。

## RoboTwin

PGC 模型本身不依赖 LIBERO 的动作维度或相机数量，`RobotVideoDataset` 的 provenance 标记也可复用。RoboTwin 阶段仍需要单独定义可执行反事实目标、采集状态对齐成功轨迹、增加对应 validator/task config，并在其原始成功率门控通过后再测语言干预；不能直接拿 LIBERO 的 Goal/Verifier 权重宣称跨 benchmark 有效。
