# RoboTwin 部署观测缓存下的去噪对照

已有动作审计显示 historical stacking 在前 8/16/24/32 步均沿两条专家动作差的反方向变化；ranking 两种指令的动作几乎相同。下一步只检查两个 historical 初始场景的去噪行为，不训练、不仿真。

## 运行

```bash
cd /root/gpufree-data/LF-FastWAM
git pull --ff-only origin main
export DIFFSYNTH_MODEL_BASE_PATH=/root/gpufree-data/fastwam/FastWAM/checkpoints
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
DENOISE_OUT="$PWD/runs/robotwin_denoising_probe/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$(dirname "$DENOISE_OUT")"
nohup /opt/conda/bin/python -u scripts/probe_robotwin_denoising.py run \
  --source-probe latest --output "$DENOISE_OUT" --gpus 0 1 2 \
  --execute > "$DENOISE_OUT.driver.log" 2>&1 < /dev/null &
echo "PID=$! OUTPUT=$DENOISE_OUT"
tail -f "$DENOISE_OUT.driver.log"
```

`latest` 自动选择仓库 `runs/robotwin_same_state_probe` 下最近完成、格式匹配的 probe，启动时打印来源目录；也可传原 probe 的绝对路径。不依赖旧 shell 中的 OUT。只有一张空闲卡时改为 `--gpus 0`，依次运行三个模型。去掉 `--execute` 是只读计划检查。

默认 Base、step500、step1000 各一张 GPU，两个场景，source/target 两条专家参考，sigma=0.1/0.5/0.9/1.0，噪声 seed=42/43/44。每个模型先对每个状态、每种语言复现原始 10 步采样（共 4 次），再做 96 次缓存下的单次 Action 前向；无反向传播或优化器更新。

完成后回传：

```bash
cat "$DENOISE_OUT/denoising_summary.csv"
```

原始逐噪声记录在各模型的 `records.jsonl`；checkpoint 加载和部署重放检查在 `checkpoint_audit.json`、`*_replay.json`。任一 worker 失败时主日志会打印其最后 45 行，且不汇总部分结果。新会话需重新设置 DENOISE_OUT 为启动时打印的路径。

## 实验边界

1. 使用同一套经过服务器运行的加载流程，核对配置、统计和权重指纹。实际执行原来的 `_infer_action_chunk`，只观察第一次 Action predictor 调用，保留该语言的观测缓存；完整输出必须与原保存动作在归一化空间的最大差不超过 1e-5，否则停止。
2. 同一场景的 source/target 缓存分别来自各自指令。随后冻结这些缓存，以训练 scheduler 构造 `x_sigma = (1-sigma)*a_expert + sigma*noise`，用生产 `_predict_action_noise_with_cache` 比较正确/错误指令的速度预测。目标为训练 scheduler 的 `noise-a_expert`。同一个专家、sigma、seed 的两种语言接收完全相同的 noisy action；跨模型也检查输入 hash。
3. sigma<1 时输入包含专家动作信息。这是带专家线索的去噪重建诊断，不是自由生成目标选择测试。sigma=1 时两种专家参考的输入都为同一纯噪声；汇总会检查这一合同。
4. 缓存来自部署的当前观测，不输入未来专家视频。因此它使用训练的 Action flow 目标，但**不是完整训练 loss 的原样重放**；不能用它单独证明训练与部署的全部差异已经定位。后续若需要检查训练 Video 缓存、梯度冲突或训练采样覆盖，应单独设计对照。
5. 输出包括前 24/32 步的 flow RMSE，以及由 `x_sigma - sigma*velocity` 得到的单步 x0 估计误差。后者不是完整多步 rollout。正确与错误指令的误差差值为正，表示正确指令更接近专家标签。
6. 低 sigma 的 x0 误差天然随 sigma 缩小，不能把低噪声重建较好本身认定为“模型学会了”。应比较同 sigma 的 Base/500/1000，以及正确/错误语言的 flow 误差；同时看三次噪声抽样的范围。另报告 dtype 舍入导致的 x0 重建误差下限。sigma=1 的训练权重可能为零，保留未加权 flow 误差用于部署起点诊断。

## 如何使用结果

- 若同 sigma 下适配器的收益只出现在低噪声，且两种语言都能重建，说明动作线索可能承担了大部分重建工作；高噪声目标选择仍需加强。
- 若高噪声正确语言误差优于 Base 且显著优于错误语言，而原多步采样仍失败，进一步检查采样过程中的误差累积与语言条件。
- 若所有 sigma 的目标参考误差都几乎不改善，优先检查训练目标的有效监督、方向、梯度以及少场景可拟合性。单个场景不能代表整体泛化。

CPU 检查：`python -m unittest -q tests.test_robotwin_denoising_probe tests.test_robotwin_no_eraf_probe`。GPU 模型前向必须在服务器验证。
