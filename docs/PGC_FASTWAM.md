# PGC-FastWAM v4

PGC（Policy-Guarded Counterfactual FastWAM）用于在不破坏原始 FastWAM 策略的前提下提升语言遵守，尤其针对 CIS 中“仍执行原指令 / 改变行为但抓错物体 / 什么也不抓”三类失败。

## v4：最终动作空间的 rollout-aligned proposal

Object 实验确认 release Base 与 PGC v3 在相同 CIS manifest 上都是 30%，而 v3
Verifier 的 Base/CF 分数同时饱和在约 0.99、BF16 margin 接近量化零，guarded
模式没有发生 override。PGC v4 因此不再沿用“随机 diffusion timestep 上学习
velocity residual、部署时积分多个 residual”的训练路径，也不再使用 sigmoid
绝对分数门控。

v4 的部署路径是：

1. 冻结的 release Base 按正式评估相同的步数完成整段 action denoising，得到
   `a_base`；训练和部署都只编码当前观测帧，不读取示教未来帧。Base
   Video/Action Expert 不复制、不加 LoRA、不进入优化器。
2. Goal Graph 读取请求语言和当前帧视觉；小型时序 Proposal 读取
   `(a_base, goal)`，只执行一次
   `a_cf = stopgrad(a_base) + tanh(raw_delta) * cap`。输出层零初始化，因此新建
   v4 的候选逐元素等于 Base。
3. 直接反事实成功示教在**最终归一化 action chunk**上监督 `a_cf`；Native
   样本只监督 `delta=0`。夹爪维度可加权，另有残差能量、时序平滑和逐维硬 cap。
4. FP32 时序 Pairwise Verifier 同时比较完整的 Base/CF action 序列，输出未经过
   sigmoid 的 `advantage = value_cf - value_base`。训练目标来自两个候选到真实
   action 的加权 MSE 改善；相同候选的 advantage 严格为零。
5. guarded 模式只有在 raw advantage 达到 threshold，且 residual RMS/饱和比例
   通过 support guard 时才覆盖，否则返回 exact Base。`base` 和 `counterfactual`
   模式仍分别用于保护验证和强制候选诊断。

正式训练的 `PGC_ROLLOUT_INFERENCE_STEPS` 必须与评估命令最后一个
`num_inference_steps` 参数一致；评估脚本会从 checkpoint 元数据检查并拒绝不一致
的运行。v4 checkpoint 格式为 `fastwam_policy_guard_v4`，只保存 Goal Graph、
Final-Action Proposal、FP32 Pairwise Verifier 和元数据。

以下 v3 章节保留用于解释旧 checkpoint 与失败机制；v4 正式实验使用文末的 v4
启动命令。

## v3 历史架构

PGC v3 保留两个候选，但只存在一份 Action Expert：

1. **Base Policy**：官方 query-free FastWAM Video Expert + Action Expert。加载 release checkpoint 后全部冻结，不进入优化器。
2. **Counterfactual Candidate**：在自己的 diffusion latent 上再次调用同一份冻结 Base Action Expert，得到 Base flow velocity 与隐藏 action token；模型不再复制或微调第二个 Action Expert，也不再给动作主干注入 LoRA。
3. **Goal Graph + Bounded Velocity Residual**：少量 goal slots 先读取语言，再读取当前帧视觉 token；32 个 Goal Queries 与冻结 action hidden cross-attend 后，只输出 `Δv`。最终候选为 `v_cf = stopgrad(v_base) + tanh(raw_Δv) × cap`。输出层严格零初始化，因此新建 v3 在训练前逐元素等于 Base；每维硬上限阻止动作速度无限漂移。
4. **Action–Outcome Verifier**：在当前状态和目标表示下分别给 Base/Counterfactual action chunk 打分。
5. **Conservative Hard Gate**：只有当反事实候选分数达到绝对阈值，且比 Base 高出安全 margin 时才覆盖；否则逐元素原样返回 Base action。

`policy_guard.enabled=false` 时不会构建或执行 PGC 路径，旧 FastWAM/TC/LangForce 行为保持不变。PGC v3/v4 与 TC、LangForce、任何 Base LoRA 互斥。v1/v2 的独立专家与 LoRA 代码只为历史 checkpoint 兼容，不用于 v4 正式训练。

v3 训练优化器的严格白名单只有 Goal Graph、Goal Query Seeds、Bounded Velocity Residual 和 Verifier。整套 Base Video/Action Expert 全部 `requires_grad=false` 且保持 eval mode；代码会反向拒绝 v3 启用 LoRA，checkpoint 也不能含独立 Action Expert 或 LoRA tensor。

## 监督信号

- `loss_pgc_action`：仅在直接反事实样本上监督 `Δv_target = v_flow_target − stopgrad(v_base)`，使受限残差真正学习状态对齐的替代动作，而不是只学视觉语义。
- `loss_pgc_native_residual_zero`：仅在 Native 样本上要求 `Δv=0`。这比让另一套专家蒸馏 Base 更直接：Native 候选从结构上保持原始速度。
- `loss_pgc_residual_regularization`：限制反事实残差能量；`loss_pgc_residual_smoothness` 限制相邻 action step 的修正抖动；`tanh × cap` 再提供逐维硬边界。
- `loss_pgc_verifier`：用真实 action 作为正样本，并根据 Base/Counterfactual 候选到真实 action 的距离生成连续质量标签。这样不会把“其实正确的 Base 候选”强行标成负样本。
- `loss_pgc_verifier_ranking`：仅在直接反事实样本中、且 Counterfactual 候选确实比 Base 更接近真实 action 时，要求其**概率评分**超过 Base；训练 margin 与部署 Gate 使用同一概率尺度。
- `loss_pgc_goal_action_alignment`：Goal-State representation 与真实 action outcome 的跨卡对称对比对齐；同指令样本不会互相作为负样本。v3 默认前 1000 optimizer steps 将 Verifier 与该对齐项完全移出反向传播，随后 500 steps 线性接入，避免 Residual 与 Verifier 同时从随机状态追逐移动目标。
- `loss_video` 与 `loss_pgc_base_action_monitor`：只用于监控，不参与优化。

训练日志还会记录 `pgc_velocity_residual_norm/max_abs/saturation_fraction`、`pgc_native_residual_zero_mse`、`pgc_counterfactual_residual_target_mse`、`pgc_counterfactual_target_outside_cap_fraction`、`pgc_verifier_training_scale`、候选质量目标、预测覆盖率、Goal Query 多样性及原策略冻结标记。若 `target_outside_cap_fraction` 长期很高而残差饱和，才依据诊断调大 cap，不能直接解除边界。

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

分 suite 验证阶段只构建一个 suite。Object 使用下面的单卡命令，不会启动其他
三个 suite：

```bash
cd /root/gpufree-data/LF-FastWAM

nohup env \
  PGC_BUILD_SUITE=libero_object \
  PGC_VIDEO_CODEC=h264 \
  PGC_MAX_DEMOS_PER_PAIR=50 \
  bash scripts/build_pgc_libero_datasets.sh \
    "$PGC_HDF5_ROOT" \
    "$PGC_DATA_ROOT" \
    5 \
    42 \
  > /root/gpufree-data/pgc_object_data_build.log 2>&1 &

echo $! > /root/gpufree-data/pgc_object_data_build.pid
```

完成后只生成
`pgc_counterfactual_datasets.libero_object.txt`，并强制检查 Object 的 10 个源
任务。`PGC_BUILD_SUITE` 同样接受 `libero_spatial`、`libero_goal` 和
`libero_10`；省略时才构建全部四个 suite。

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

若某个 pair 已把当前 donor 的全部 HDF5 demo 试完但仍为 0 条成功轨迹，单纯
`PGC_RESUME=true` 不会重试出新结果。此时用候选排名覆盖选择下一个兼容 donor；
多个任务用逗号分隔：

```bash
PGC_BUILD_SUITE=libero_object \
PGC_RESUME=true \
PGC_CANDIDATE_RANK_OVERRIDES='libero_object:5=1,libero_object:8=1' \
bash scripts/build_pgc_libero_datasets.sh \
  "$PGC_HDF5_ROOT" \
  "$PGC_DATA_ROOT" \
  5 \
  42
```

排名 0 是默认 donor，1 是第二候选。恢复逻辑只允许替换从未产生成功 episode
的 pair；任何已有成功轨迹的 pair 都不可变，因此不会让旧 action、语言和
provenance 脱节。已有 episode 会保留，新 donor 从下一个 episode index 继续写。
若第二候选仍为 0，可把相应 rank 增加到 2 后再次安全续跑。

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
bash scripts/train_pgc_v4_libero_suite.sh \
  libero_object \
  4 \
  "$BASE" \
  "$PGC_CF_OBJECT" \
  42 \
  4000
```

其余三条线仅替换 suite 名和对应数据目录，不能把多个目录放入同一次训练：

```bash
bash scripts/train_pgc_v4_libero_suite.sh libero_spatial 4 "$BASE" /path/to/pgc/libero_spatial_pgc_counterfactual_lerobot 42 4000
bash scripts/train_pgc_v4_libero_suite.sh libero_goal    4 "$BASE" /path/to/pgc/libero_goal_pgc_counterfactual_lerobot    42 4000
bash scripts/train_pgc_v4_libero_suite.sh libero_10      4 "$BASE" /path/to/pgc/libero_10_pgc_counterfactual_lerobot      42 4000
```

`train_pgc_libero_suite.sh` 会同时把原始数据和直接反事实数据限制到指定 suite，
并检查 `meta/pgc_provenance.json`。任何跨 suite 数据都会在创建训练进程前报错。
默认按 frame index 把 Native 与 Counterfactual 数据构造成严格 1:1 采样。v4 不使用
LoRA，默认 final-action residual cap 为 `2.0`、能量/平滑权重均为 `0.01`，这些从
零初始化的小模块使用学习率 `1e-4`，每卡 micro-batch 为 1、
梯度累积为 4（四卡有效 batch 16），每 500 steps 保存权重。可通过
`PGC_ACTION_CHUNK_RESIDUAL_MAX_ABS`、`PGC_ROLLOUT_INFERENCE_STEPS`、
`PGC_ACTION_GRIPPER_WEIGHT`、`PGC_RESIDUAL_REGULARIZATION_WEIGHT`、
`PGC_RESIDUAL_SMOOTHNESS_WEIGHT`、`PGC_VERIFIER_START_STEP` 和
`PGC_VERIFIER_RAMP_STEPS` 修改 v4 超参。默认
不保存 DeepSpeed state，以避免磁盘空间不足导致最后一步保存失败。

每条线必须依次通过以下门控，才进入下一条：

1. 数据覆盖指定 suite 的 10 个源任务，且所有 episode 都通过状态与成功审计；
2. `PGC_GATE_MODE=base` 与 release FastWAM 的 action chunk 数值一致；
3. 500-step smoke 必须先通过 guarded Correct 策略保护并报告实际覆盖率；若强制
   Counterfactual Correct 尚未达到 8/10，只允许进行一次预先限定到 1500 step
   的欠训练诊断，不能直接扩展到完整 epoch；
4. 1500-step 诊断或完整训练后，强制 Counterfactual Correct 必须恢复闭环抓放
   能力，否则停止当前结构；
5. guarded Correct SR 没有不可接受的策略回退；
6. DTL 与 CIS 相对基线有可复现提升，并报告 Gate 覆盖率；
7. 至少完成多 seed 或预先约定的重复实验，不能凭单 episode 放行。

四个 suite 全部验收后，联合入口仍默认锁定。只有获得明确同意，才可创建包含四个
数据目录的列表并显式设置：

```bash
PGC_TRAIN_SUITE=all \
PGC_ALLOW_JOINT_TRAINING=true \
PGC_VERSION=4 \
bash scripts/train_pgc_libero.sh 4 "$BASE" /path/to/pgc_counterfactual_datasets.txt 42 4000
```

代码拉取后先在服务器执行完整回归：

```bash
bash scripts/validate_pgc_server.sh
```

输出 checkpoint 格式为 `fastwam_policy_guard_v4`，只保存 Goal Query Seeds、
Final-Action Proposal、Goal Graph、Pairwise Verifier、架构元数据和外部 Base checkpoint
路径；既不复制 6.8B Base 权重，也不存在第二套 Action Expert 或 LoRA。加载时先
恢复外部 Base，再严格恢复小型 PGC 模块。v1/v2/v3 checkpoint 仍按旧格式加载。

### 未保存 ZeRO 状态时续训

当原运行使用 `PGC_SAVE_TRAINING_STATE=false` 时，`.pt` 只包含模型权重，没有
DeepSpeed Adam moments 或 scheduler 状态。此时必须显式同时设置初始化
checkpoint 和其绝对 step；不能把 adapter 当成普通 Base 从 step 0 重新计数：

```bash
export PGC_INIT_CHECKPOINT=/path/to/step_000500.pt
export PGC_CONTINUE_FROM_STEP=500

# 最后一个参数 1500 是绝对目标 step，因此本次实际再训练 1000 steps。
bash scripts/train_pgc_v4_libero_suite.sh \
  libero_object 4 "$BASE" "$PGC_CF_OBJECT" 42 1500
```

启动器会校验匹配的 PGC v4 格式、保存步数、外部 Base 路径、rollout step 和
final-action residual
架构。Trainer 会把
`global_step` 恢复为 500，跳过同一随机序列中已经消费的数据，并为剩余 1000
steps 新建带短 warmup 的 scheduler。由于原运行没有保存 ZeRO state，优化器
动量无法恢复；日志会明确标记这是 fresh-optimizer weight-only continuation。
若有完整 `checkpoints/state/step_*`，仍应优先使用原有完整状态恢复路径。

## 评估

训练完成后只评估该 checkpoint 对应的 suite。下面继续以 Object 为例：

```bash
export PGC_CHECKPOINT=/path/to/step_004000.pt
PGC_EVAL_SUITES='[libero_object]' \
PGC_MAX_POLICY_STEPS=600 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/eval_pgc_libero.sh 4 5 correct 42 10
```

`PGC_MAX_POLICY_STEPS` 是可选诊断覆盖；不设置时继续使用各 suite 的标准
horizon。Object 长条盒子实验建议使用 600，以区分策略失败和未松手超时。

先用强制 Base 门控做“原策略逐 action chunk 完全一致”的保护对照：

```bash
PGC_EVAL_SUITES='[libero_object]' \
PGC_GATE_MODE=base \
OUTPUT_ROOT=/path/to/evaluate_results/pgc_base_exact_correct \
bash scripts/eval_pgc_libero.sh 4 5 correct 42 10
```

v4 正式 guarded 门控使用 FP32 raw advantage，只通过 `PGC_GATE_THRESHOLD`
调整；`PGC_MIN_COUNTERFACTUAL_SCORE` 仅供 v1-v3 兼容。阈值必须在独立校准集上
选择，不能用最终 CIS 测试集调参。

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
