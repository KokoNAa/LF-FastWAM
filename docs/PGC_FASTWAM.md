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

仅将旧轨迹换成另一条文本不是反事实动作正监督。PGC 要求在**源任务仿真模型中的已审计状态**下，以替代指令执行并成功完成请求结果的真实轨迹。该状态可以是布局完全一致时精确恢复的 donor flat state，也可以是按同名关节把 donor 的机器人、目标物体及公共物体迁移到 Source 模型后得到的 Source flat state；两者都必须在 Source 环境中成功回放。每个直接反事实 LeRobot 数据目录必须包含：

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
  "state_catalog": "meta/pgc_initial_states/episode_{episode_index:06d}.npy",
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
{"episode_index": 0, "pair_id": "libero_object_00_to_01", "source_initial_state_index": 0, "source_initial_state_catalog": "meta/pgc_initial_states/episode_000000.npy", "initial_state_sha256": "<64 lowercase hex characters>", "initial_state_match": true, "counterfactual_goal_satisfied": true}
```

任务文件 `meta/tasks.jsonl` 中的 instruction 必须是实际执行的替代指令；失败轨迹不能混入。

## 构建直接反事实动作数据

### 数据来源与有效性条件

构建器读取 LIBERO 官方原始 HDF5 人类示教（`data/demo_*/states` 与
`actions`），而不是把现有 LeRobot 轨迹替换一段文本。对每个
Source→Counterfactual 配对，它执行以下流程：

1. 从 Counterfactual 任务的成功 demo 取得 `states[0]` 和动作序列；
2. 创建 Source BDDL 环境并安装经过审计的请求谓词；
3. 若两个任务的有序 objects/fixtures 完全一致，直接精确恢复 `states[0]`；否则在相同环境类和 fixtures 下创建 Target 环境，按 MuJoCo joint name 把机器人、目标物体及公共物体的 qpos/qvel 迁移到 Source，保留 Source-only distractor，再把生成的 Source flat state 写回验证；
4. 按 LIBERO 数据再生成协议执行 10 个稳定化空动作并过滤 no-op；
5. 先做一次无记录验证；成功后从同一状态做第二次录制；
6. 第二次也完成替代谓词时，才写入双相机 LeRobot episode 和审计文件。

Spatial 的十个任务共享终点谓词，语言差异来自被描述的初始 bowl 位置，因此其正监督允许“终点谓词不变、目标任务初态和动作不同”，并明确写入 `counterfactual_goal_changed=false`。Object 的 donor 往往拥有不同 distractor 集合，必须使用 `named_joint_remap`，不能直接复制等长 flat state；否则相同 qpos 索引可能绑定到错误物体。

这里的 `source_initial_state_index` 指向数据集自己的
`meta/pgc_initial_states/` 状态目录。状态虽然来自目标任务 demo，但已经在 Source
环境中实际恢复或按名称迁移并验证；两次回放有任意一次不满足请求谓词都会拒绝，不能把失败重放伪装成正样本。

官方 HDF5 下载和格式说明见 [LIBERO 数据集说明](https://github.com/Lifelong-Robot-Learning/LIBERO#datasets)
及 [LIBERO 示教采集脚本](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/scripts/libero_100_collect_demonstrations.py)。

### 先检查服务器是否有原始 HDF5

现有的 `*_no_noops_lerobot` 目录不能还原完整 MuJoCo 初始状态；必须找到原始
HDF5：

```bash
find /root/gpufree-data/fastwam -type f \
  \( -name '*_demo.hdf5' -o -name '*_demo.h5' \) \
  | sort | tee /root/gpufree-data/libero_hdf5_files.txt

wc -l /root/gpufree-data/libero_hdf5_files.txt
sed -n '1,20p' /root/gpufree-data/libero_hdf5_files.txt
```

标准下载应包含 Spatial、Object、Goal 和 LIBERO-100 的 `*_demo.hdf5`；其中
LIBERO-100 同时提供 LIBERO-10 与 LIBERO-90 donor。若没有，需要先用 LIBERO
官方下载脚本补齐原始示教，不能从 MP4 或 Parquet 反推状态。

### 配对与只读规划检查

配对器会在四个源 suite 加 LIBERO-90 donor 池中寻找可执行 donor。完全相同的
有序 objects/fixtures 使用 `flat_exact`；相同环境类与 fixtures、但 distractor
物体清单不同的任务使用 `named_joint_remap`。加入 LIBERO-90 是
为了给 LIBERO-10 的 held-out 场景寻找同模型替代目标，并不会把 LIBERO-90
当作评估 source。Spatial 依据不同初态描述配对，不把其共同终点谓词误判成
无反事实信号。先只做规划检查，不启动 MuJoCo 录制：

```bash
cd /root/gpufree-data/LF-FastWAM

export PGC_HDF5_ROOT=/path/to/LIBERO/libero/datasets
export PGC_DATA_ROOT=/root/gpufree-data/pgc_libero_data_v1

PGC_PLAN_ONLY=true \
bash scripts/build_pgc_libero_datasets.sh \
  "$PGC_HDF5_ROOT" \
  "$PGC_DATA_ROOT" \
  5 \
  42
```

结果写到 `manifests/pgc_manifest_coverage.json`，并统计两种 state-transfer
mode。若某些任务仍显示 `no executable state-compatible donor task`，应为这些
场景采集人工/脚本 oracle demo。`PGC_RELAXED_SCENE_MATCH=true` 只适合诊断
fixture 也不同的候选，不用于正式数据。

### 四卡后台正式构建

规划通过后，每张卡负责一个 suite：

```bash
cd /root/gpufree-data/LF-FastWAM

nohup env \
  PGC_VIDEO_CODEC=h264 \
  PGC_MAX_DEMOS_PER_PAIR=50 \
  bash scripts/build_pgc_libero_datasets.sh \
    "$PGC_HDF5_ROOT" \
    "$PGC_DATA_ROOT" \
    5 \
    42 \
  > /root/gpufree-data/pgc_libero_data_build.log 2>&1 &

echo $! > /root/gpufree-data/pgc_libero_data_build.pid
```

监控：

```bash
tail -f /root/gpufree-data/pgc_libero_data_build.log

for f in "$PGC_DATA_ROOT"/logs/*.log; do
  echo "===== $f ====="
  tail -n 20 "$f"
done

find "$PGC_DATA_ROOT" -path '*/meta/pgc_collection_summary.json' \
  -exec sh -c 'echo "===== $1 ====="; cat "$1"' _ {} \;
```

网络或终端中断后使用相同目录续跑；已提交的 donor demo 不会重复：

```bash
PGC_RESUME=true \
bash scripts/build_pgc_libero_datasets.sh \
  "$PGC_HDF5_ROOT" \
  "$PGC_DATA_ROOT" \
  5 \
  42
```

完成后会生成 `pgc_counterfactual_datasets.txt`，并自动执行四 suite、每 suite
10 个源任务、每个配对至少一个成功 episode 的强校验。每个数据目录同时包含：

- `data/` 与 `videos/`：FastWAM 可直接读取的 Parquet/MP4；
- `meta/pgc_provenance.json`：配对和采集协议；
- `meta/pgc_episodes.jsonl`：逐 episode 成功与状态审计；
- `meta/pgc_initial_states/`：可复验的精确初始状态；
- `meta/pgc_collection_summary.json`：成功数、缺口和最近拒绝原因。

## LIBERO 分 suite 训练

正式联合训练前，固定按 `libero_object`、`libero_spatial`、`libero_goal`、
`libero_10` 四条独立实验线验证。每条线使用自己的原始数据、直接反事实数据、
checkpoint、日志和评估目录。suite 内的 10 个任务必须共同训练；若拆成单 task-ID
模型，语言在该模型内成为常量，DTL/CIS 无法判断模型是否真正使用语言。

先为当前 suite 的直接反事实指令生成 T5 cache。下面以 Object 为例：

```bash
PGC_CF_OBJECT=/path/to/pgc/libero_object_pgc_counterfactual_lerobot

CUDA_VISIBLE_DEVICES=0 \
/opt/conda/bin/python scripts/precompute_text_embeds.py \
  task=libero_pgc_2cam224 \
  "data.train.pgc_counterfactual_dataset_dirs=[${PGC_CF_OBJECT}]" \
  +overwrite=false
```

然后只启动这一条 suite 训练：

```bash
export DIFFSYNTH_MODEL_BASE_PATH=/path/to/FastWAM/checkpoints
export LIBERO_DATA_ROOT=/path/to/FastWAM/data/libero_mujoco3.3.2
BASE="$DIFFSYNTH_MODEL_BASE_PATH/fastwam_release/libero_uncond_2cam224.pt"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/train_pgc_libero_suite.sh \
  libero_object \
  4 \
  "$BASE" \
  "$PGC_CF_OBJECT" \
  42 \
  4000
```

其余三条线仅替换 suite 名和对应数据目录，不能把多个目录放入同一次训练：

```bash
bash scripts/train_pgc_libero_suite.sh libero_spatial 4 "$BASE" /path/to/pgc/libero_spatial_pgc_counterfactual_lerobot 42 4000
bash scripts/train_pgc_libero_suite.sh libero_goal    4 "$BASE" /path/to/pgc/libero_goal_pgc_counterfactual_lerobot    42 4000
bash scripts/train_pgc_libero_suite.sh libero_10      4 "$BASE" /path/to/pgc/libero_10_pgc_counterfactual_lerobot      42 4000
```

`train_pgc_libero_suite.sh` 会同时把原始数据和直接反事实数据限制到指定 suite，
并检查 `meta/pgc_provenance.json`。任何跨 suite 数据都会在创建训练进程前报错。
默认对 Counterfactual 数据做 2 倍采样，学习率为 `1e-5`，每卡 micro-batch 为
1、梯度累积为 4（四卡有效 batch 16），每 500 steps 保存权重。默认不保存
DeepSpeed state，以避免磁盘空间不足导致最后一步保存失败。

每条线必须依次通过以下门控，才进入下一条：

1. 数据覆盖指定 suite 的 10 个源任务，且所有 episode 都通过状态与成功审计；
2. `PGC_GATE_MODE=base` 与 release FastWAM 的 action chunk 数值一致；
3. guarded Correct SR 没有不可接受的策略回退；
4. DTL 与 CIS 相对基线有可复现提升，并报告 Gate 覆盖率；
5. 至少完成多 seed 或预先约定的重复实验，不能凭单 episode 放行。

四条线全部验收后，联合入口仍默认锁定。只有获得明确同意，才可创建包含四个
数据目录的列表并显式设置：

```bash
PGC_TRAIN_SUITE=all \
PGC_ALLOW_JOINT_TRAINING=true \
bash scripts/train_pgc_libero.sh 4 "$BASE" /path/to/pgc_counterfactual_datasets.txt 42 4000
```

代码拉取后先在服务器执行完整回归：

```bash
bash scripts/validate_pgc_server.sh
```

输出 checkpoint 格式为 `fastwam_policy_guard_v1`，只保存独立 Action Expert、Goal Graph、Verifier 和外部 Base checkpoint 路径，不复制 6.8B Base 权重。

## 评估

训练完成后只评估该 checkpoint 对应的 suite。下面继续以 Object 为例：

```bash
export PGC_CHECKPOINT=/path/to/step_004000.pt
PGC_EVAL_SUITES='[libero_object]' \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/eval_pgc_libero.sh 4 5 correct 42 10
```

先用强制 Base 门控做“原策略逐 action chunk 完全一致”的保护对照：

```bash
PGC_EVAL_SUITES='[libero_object]' \
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
PGC_EVAL_SUITES='[libero_object]' bash scripts/eval_pgc_libero.sh 4 5 shuffled 42 10
PGC_EVAL_SUITES='[libero_object]' bash scripts/eval_pgc_libero.sh 4 5 counterfactual 42 10
```

每个任务结果会额外保存：

- `policy_guard_decision_count`
- `policy_guard_override_count`
- `policy_guard_override_rate`
- `policy_guard_base_score_mean`
- `policy_guard_counterfactual_score_mean`
- 每个 episode、每次 replan 的选择和分数

正确的验收顺序是：先确认 `PGC_GATE_MODE=base` 与 release FastWAM 的逐
action chunk 输出一致，再检查 guarded 模式下当前 suite 的 Correct SR 是否
保持，然后看 Shuffled 的 DTL，最后看 CIS 和覆盖率。CIS 上覆盖率为零说明
Gate 太保守；覆盖率高但 CIS 仍低说明 Counterfactual Action Expert 或目标表示
仍未学对；Correct 上覆盖率高且 SR 下降说明 Gate 校准失败。当前 suite 通过后
才进入下一条实验线；不能用某个 suite 的结果替代其他 suite 的验证。

## RoboTwin

PGC 模型本身不依赖 LIBERO 的动作维度或相机数量，`RobotVideoDataset` 的 provenance 标记也可复用。RoboTwin 阶段仍需要单独定义可执行反事实目标、采集状态对齐成功轨迹、增加对应 validator/task config，并在其原始成功率门控通过后再测语言干预；不能直接拿 LIBERO 的 Goal/Verifier 权重宣称跨 benchmark 有效。
