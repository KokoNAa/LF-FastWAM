# RoboTwin no-ERAF 同状态诊断

用于排查 temporal-v2 splitfix 的 step500/1000 在 Correct 15/15、CF target 0/15、CF source 14/15 下为什么没有改变任务目标。这是现有 checkpoint 的部署采样诊断，不启动训练或仿真，不产生新的 CIS 成功率。

## 运行

在已有 RoboTwin 推理环境中运行，默认使用物理 GPU 0、1、2，分别加载 Base、step500、step1000。只有一张空闲卡时改为 `--gpus 0`，脚本会依次运行三个模型。先确认这些 GPU 可用。

```bash
cd /root/gpufree-data/LF-FastWAM
git pull --ff-only origin main
export DIFFSYNTH_MODEL_BASE_PATH=/root/gpufree-data/fastwam/FastWAM/checkpoints
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OUT="$PWD/runs/robotwin_same_state_probe/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$(dirname "$OUT")"

nohup /opt/conda/bin/python -u scripts/probe_robotwin_no_eraf.py run \
  --train-run "$PWD/runs/robotwin_lora_only_no_eraf_3cam384/robotwin-no-eraf-temporal-v2-splitfix-diagnostic1000-4gpu-seed42" \
  --expert-root /root/gpufree-data/pgc_robotwin_no_eraf_v1/formal/expert \
  --base-checkpoint /root/gpufree-data/fastwam/FastWAM/checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt \
  --stats-path /root/gpufree-data/fastwam/FastWAM/checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json \
  --steps 500 1000 --gpus 0 1 2 \
  --profiles historical strict --task-config demo_clean \
  --episodes-per-pair 1 --inference-steps 10 --seed 42 \
  --output "$OUT" --execute > "$OUT.driver.log" 2>&1 < /dev/null &
echo "PID=$! OUT=$OUT"
```

去掉 `--execute` 即只读预检：检查路径、训练配置及数据审计，不加载模型或创建输出目录。实际执行时输出目录必须不存在。

查看进度和结果：

```bash
tail -f "$OUT.driver.log"
tail -n 40 "$OUT"/base.log "$OUT"/step500.log "$OUT"/step1000.log
cat "$OUT/summary.json"
```

新登录会话需把 `OUT` 重新设为启动时打印的绝对路径。单模型日志在完成数据预检后才出现；`summary.json` 仅在全部模型成功完成后生成。失败时查看 driver 和对应模型日志，使用新的输出目录重跑。

## 实验约定

- 使用训练运行保存的配置、发布版 Base 和相同归一化统计。训练专家 episode 按实际训练配置的验证比例及数据集默认 seed 42 选择；strict native 仅作为同场景审计参考，输出明确标记它不属于训练池。
- 每个 pair/profile 先取一个符合训练划分的场景，取开头、有效起点范围的 25%/50%/75%，另取原任务与 CF 的 qpos 公共前缀最后一帧，以覆盖可能的决策分岔。全部窗口包含完整 32 步参考。
- 固定三相机 RGB、机器人状态、扩散 seed，切换原任务/CF 指令，比较三个模型。调用现有 `WorldActionRobotWinPolicy._infer_action_chunk`，沿用部署拼图、提示模板、状态处理、10 步采样和 9 视频帧。
- 图像来自 raw HDF5 的 JPEG，动作/状态与转换器相同，来自 `joint_action/vector`。这不是读取训练 MP4 的压缩后像素，也不是当前失败 rollout 的重放。
- 检查 raw 与转换后元数据、动作/初始状态 hash；保存配置、统计、checkpoint 和固定状态指纹。LoRA 检查运行时每个已保存 MoT 张量与 checkpoint 经 dtype 转换后精确一致、video/action LoRA-B 非零。每个模型首个状态重复推理，归一化动作最大差须不超过 `1e-5`。
- 只有原任务和 CF 轨迹在同帧的完整 RGB **及**状态完全一致，才计算双专家参考指标。中途不同观测的轨迹只与自己的专家参考比较。公共前缀未来 32 步仍相同的窗口标为不可区分，不计为语言失败。
- 动作指标使用同一发布版归一化空间；专家参考沿用训练 clamp，预测动作不重新裁剪。同时保存原始 qpos 及完整 32 步、部署前 24 步指标。

## 如何读结果

`summary.json` 按模型、historical/strict、native/CF 列出均值；`comparisons.csv` 保留逐状态对比。重点结合 pair 和前缀决策帧看结果，不能仅靠总均值判断。

| 观察 | 支持的解释 / 后续实验 |
| --- | --- |
| 权重审计失败 | 先排查 checkpoint、加载、基座或配置，不能解释为训练不足。 |
| 权重已变化，但 `target_delta_vs_base_rms` 和 `language_delta_rms` 都接近数值重复误差 | 部署生成基本没有响应适配器或语言；下一步检查运行时 LoRA 路径及 teacher-forced loss 与生成的差异。没有通用 RMS 阈值。 |
| 同状态语言差异增加，但目标专家距离不改善、决策帧投影为零或负 | 学到了条件差异，但缺少朝目标参考动作的变化；优先检查目标选择监督、分岔帧采样和 ranking 梯度。 |
| 只有 CF 中后段参考拟合改善，初始/分岔帧没有改善 | 与“学习轨迹续写，尚未学会从共同状态切换目标”的假设一致。 |
| 分岔帧和 CF 专家状态均改善，但已有闭环仍 CF 0/15 | 下一步采集相同 checkpoint 的失败 rollout 决策状态，检查执行时域、累积误差及 CF 恢复状态覆盖。 |

正的 `wrong_minus_correct_reference_rmse` 表示正确指令比错误指令更接近当前轨迹专家参考。双参考中的 `language_delta_projection_on_expert_delta` 是预测语言差在专家动作差上的比例：正数支持关节动作方向一致，负数表示相反；不能解释为末端在空间中朝目标移动。专家 RMSE 也不是成功率，多解动作和接触动力学仍需闭环验证。

结果回传至少包含 `summary.json`、`comparisons.csv` 和三个模型的 `checkpoint_audit.json`。完整 `records.jsonl` 含前 24 步细节；每个状态 NPZ 保留预测和专家动作，便于进一步绘图。第一轮只定位机制；扩大到 `--episodes-per-pair 2` 或增加独立 seed 后才能判断稳定性。

### 已有结果的双向动作审计（无需 GPU）

单独统计“CF 指令更接近 CF 专家”会包含两个指令都偏向 CF 专家的情况。下面的命令读取已完成 probe 的 records 和 NPZ，按 8/16/24/32 步重新计算双向偏好，区分 `both_correct`、`both_source`、`both_target`、`reversed`、`tie` 和不可区分的参考。它不加载模型，不重新推理，不改写原始结果。

```bash
/opt/conda/bin/python scripts/probe_robotwin_no_eraf.py audit-actions "$OUT"
cat "$OUT/action_audit.csv"
```

`action_audit_summary.json` 保存各窗口的计数，`action_audit.csv` 保存逐状态结果，`action_audit_details.jsonl` 还包含完整距离矩阵、参考轴坐标和变化最大的三个动作维度。参考轴上 source 为 0、target 为 1，最近参考分界为 0.5；不表示空间目标位置。共同更新量为两种指令相对 Base 更新的均值；条件差更新量为两种指令更新之差。夹爪/关节的贡献按归一化动作差的平方和计算。小于数值阈值之外的“参考可区分”不等于语义上的有效目标分岔。

## 本地 CPU 验证

```bash
python -m unittest -v tests.test_robotwin_no_eraf_probe
```

需要 numpy、h5py、Pillow、omegaconf；这些测试覆盖实际 HDF5 准备、严格池只有 raw source、训练划分、审计拒绝、专家参考可比性和完整结果汇总。GPU checkpoint 加载及部署采样须在服务器执行验证。
