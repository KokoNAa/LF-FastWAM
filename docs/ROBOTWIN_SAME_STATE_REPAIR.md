# RoboTwin 同状态双向 Action-LoRA 修复实验

现有诊断中，historical stacking 的指令与专家动作对应方向相反，ranking 的两个语言分支接近同一条专家轨迹。本实验直接做小规模训练干预，检查这两个初始状态能否通过正确分支监督得到纠正。

## 服务器运行

代码推送后，在服务器仓库执行下面整段命令。使用最近**已完成**的 same-state probe，自动继承原 checkpoint、统计文件、专家引用和训练配置，不依赖旧 shell 的 OUT。

```bash
cd /root/gpufree-data/LF-FastWAM
git pull --ff-only origin main
export DIFFSYNTH_MODEL_BASE_PATH=/root/gpufree-data/fastwam/FastWAM/checkpoints
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
REPAIR_OUT="$PWD/runs/robotwin_same_state_repair/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$(dirname "$REPAIR_OUT")"
nohup /opt/conda/bin/python -u scripts/train_robotwin_same_state_repair.py run \
  --source-probe latest --output "$REPAIR_OUT" \
  --gpus 0 1 --steps 300 --eval-every 50 --execute \
  > "$REPAIR_OUT.driver.log" 2>&1 < /dev/null &
echo "PID=$! OUTPUT=$REPAIR_OUT"
tail -f "$REPAIR_OUT.driver.log"
```

只有一张空闲卡时改成 `--gpus 0`，两组依次运行。去掉 `--execute` 仅检查并打印计划。已有输出目录一律拒绝复用；本脚本不恢复旧 optimizer。两个实验分别写入 `paired_flow.log` 和 `paired_flow_anchor.log`，可用 `tail -F` 查看训练步数与中间评估。

完成后，以下命令在**新 shell 中也能使用**，会自动打印最近完成的实验路径、checkpoint 选择和对比表：

```bash
cd /root/gpufree-data/LF-FastWAM
/opt/conda/bin/python scripts/train_robotwin_same_state_repair.py report latest
```

把这条命令的输出回传即可。任一 worker 失败，driver 会打印该 worker 日志末尾，不汇总部分结果；没有新完成的实验时，`report latest` 可能仍指向以前完成的实验，需检查打印的路径。

### 失败后的 CPU 诊断

2026-09-05 的 `20260905-164002` 实验中，两组到第 300 步仍无合格 checkpoint，修复状态的双向正确检查均为 0/12。12 是两个状态、三种噪声、两个动作时域的组合，不是 12 个独立场景。前 24 步的原指令平均 RMSE 分别下降 6.38% / 7.99%，CF 却上升 48.94% / 34.05%；两组从第 100 步起都出现 guard 退化。纯噪声项减轻了部分 CF 退化，但本轮没有证明修复有效。不要用这两组 checkpoint 替换原模型，也不要因平均原指令误差下降而直接延长训练。

先分析已经保存的逐状态训练日志和动作，不占 GPU、不重新加载 checkpoint：

```bash
cd /root/gpufree-data/LF-FastWAM
git pull --ff-only origin main
/opt/conda/bin/python scripts/inspect_robotwin_same_state_repair.py \
  /root/gpufree-data/LF-FastWAM/runs/robotwin_same_state_repair/20260905-164002
```

回传终端输出即可。同一实验目录还会生成 `repair_diagnostics.txt` 和完整的 `repair_diagnostics.json`，原始日志和动作不会改变；此命令只需要 Python 和 NumPy。传入明确的实验目录，避免 `latest` 选到另一次实验。它检查两组训练抽样一致、日志和评估覆盖完整、起始模型指标一致，并将初始/末次保存动作重新计算的 RMSE 与逐状态评估核对。任何缺失或冲突都报错，不输出部分诊断。

诊断包含每个修复状态前/后 50 步的 source/CF flow MSE、实际权重乘积、纯噪声 MSE、梯度范数与裁剪比例，以及每个状态初始/末次的动作误差、指令方向和误差分解。完整 JSON 还保留所有状态、所有检查点的逐状态平均指标。

对于两条预测动作 `s,t` 和各自专家 `r,q`，成对 MSE 可精确分成：

```text
pair_mse = mean((((s+t) - (r+q))/2)^2) + mean(((t-s) - (q-r))^2)/4
         = common_mse                    + conditional_mse
```

共同动作误差下降、指令差异误差不降，支持优先检查条件学习是否被共同动作拟合掩盖；若 source/CF 训练窗口误差都下降而完整生成恶化，应先用固定噪声复核这一分歧，再检查去噪轨迹上的训练与生成差异。若固定输入监督本身也不能拟合，再对单任务、Action/Video 条件通路分别做小规模干预。仅凭本轮汇总不能判定 Video 表示是瓶颈，也不能断言 Action LoRA 容量不足。

注意：前后训练窗口使用不同的随机噪声和 timestep；即使窗口平均 MSE 下降，也不能替代固定输入的去噪评估。`source_objective_share` 只是 loss 数值占比，不代表梯度占比或证明梯度冲突。梯度存在且有限只说明反向在运行。分解后的 `pair_mse` 是逐样本平方误差的均值，不等于汇总表中平均 RMSE 的平方。

```bash
python -m unittest -q tests.test_robotwin_same_state_repair_diagnostics
```

## 两个实验具体改变什么

两组都从原 step1000 的同一套完整 Video+Action LoRA 开始；默认学习率 5e-6，AdamW，无 weight decay，梯度范数上限 1。每个 optimizer step 遍历全部修复状态，每个状态的 source/CF 正确专家动作等权。默认原 probe 只有 stacking、ranking 各一个 historical 初始状态，因此每步含两个完整配对、四条动作标签。

| 组别 | 正确 Action flow 监督 | 额外纯噪声监督 |
| --- | --- | --- |
| paired_flow | 原 scheduler 的连续 sigma 分布和训练权重 | 无 |
| paired_flow_anchor | 与前组完全相同的普通噪声抽样 | sigma=1，两条指令同一噪声，独立权重 0.25 |

目标均为原训练 scheduler 的 `noise - expert_action`。source/CF 使用各自正确动作，不将两个动作平均成一个标签。sigma<1 的 noisy action 包含各自专家线索；sigma=1 两条指令的输入完全相同，因此该项直接约束不同语言产生不同的正确速度。纯噪声项不乘原 scheduler 的端点零权重。0.25 是待验证的实验起始值，可用 `--anchor-weight` 修改。

两组 optimizer 步数、初始权重、普通 sigma 和噪声一致，汇总时交叉核对。anchor 组每步有额外前向/反向，默认是另一组的两倍 Action 调用量；这不是计算量严格相等的正式消融。默认 train noise seeds 从 17000 段产生，评估使用 42/43/44；重叠时直接拒绝。

本实验先**只更新 Action LoRA A/B**，参数保持 FP32。Video LoRA、Base、VAE、文字编码器、proprio encoder 均冻结。所有模块保持 eval 模式以关闭 dropout，但 Action 前向显式启用梯度。它使用生产 Action predictor 的原函数体，移除的仅是该方法的 `no_grad` 包装，另用非重入 activation checkpointing 控制反向显存。原模型方法和推理器不被修改。

这里没有 Video loss、任何 ranking、ERAF 或 full-goal 数据；它是可学性干预，不是原四池目标的原样重放，也不能将任何收益单独归因于“关闭 ranking”。如果失败，只能说这个 Action-only 方案未通过，不能据此断言完整 Video+Action LoRA 无法学习。

## 数据与部署一致性

- 训练只选 historical、frame 0、RGB 和 proprio 完全一致、双方专家均处于已授权训练划分的状态。核对 pair 的成功审计、指令、scene seed 和非 final provenance；不将 strict native 审计数据用于训练。
- 专家引用采用原训练 normalizer，包括它对参考动作的 clamp；生成动作不额外 clamp。修复窗口必须在实际执行的前 24 步存在专家动作差异。
- 先真实执行原生产 `_infer_action_chunk`，对照原 probe 保存动作，最大归一化差须不超过 1e-5。缓存来自每条指令各自的当前观测，不包含未来专家视频。
- 缓存移至 CPU，仅当前状态的两个缓存进入 GPU。随后用原采样噪声、原 inference scheduler 和原 predictor 做完整 10 步生成；必须先与生产输出复现一致，训练结束后再重新走生产观测编码验证一次。
- 其他有双专家引用的初始状态只做 regression guard；与修复状态观测 hash 相同的条目不算独立 guard。这些 guard 没有进入本次更新，但可能在以前的训练中出现，不能称为未见过的泛化场景。
- 每步检查 Action LoRA 有有限、非零梯度；冻结参数的版本、存储、dtype 和梯度保持不变。保存时逐张量检查继承的 Video adapter 不变。

## 如何验收

在 step 0 和每 50 步，用三种固定评估噪声检查完整 10 步生成，分别统计前 24/32 个动作。`evaluation_*.json` 保留逐状态、逐噪声结果；`repair_summary.csv` 仅做便于回传的汇总。

`fit_pass` 要求每个修复状态、每个评估噪声、两个动作时域同时满足：

1. source/CF 两条预测各自更接近自己的专家，而不是都移向同一个专家。
2. 对每条专家参考，正确指令的误差均低于错误指令。
3. 两个正确分支的误差各自比原 step1000 降低至少 5%。不能用平均改善掩盖另一边退化。

`guard_pass` 要求存在 guard，且每个 guard 状态、每种语言、每个时域的三噪声平均误差都不超过初始值的 `1.10 倍 + 0.005`。这是离线回归筛查阈值，不是任务成功率容差。

只有同时通过两项的 checkpoint 才作为 `best` 候选，并按修复状态的平均正确动作误差选择；否则 `best=null`，仍保留全部中间 checkpoint 和失败指标供诊断。默认阈值可通过 `--minimum-improvement`、`--guard-relative`、`--guard-absolute` 明确修改，不应为了通过检查而事后放宽。

- 两组都能纠正：优先把同状态双向正监督扩展到更多合法训练场景和关键决策状态。
- 只有 anchor 组能纠正：支持继续验证显式高噪声条件监督，随后再做计算量匹配与更多场景的对照。
- 修复状态改善、guard 退化：需要扩大状态覆盖并加入原任务保留训练，不能直接采用局部拟合模型。
- 两组都失败：检查实际正确分支误差和梯度，下一步在同一小集合中检验 Video 条件表示及联合 LoRA；不直接加长全量训练。

即使通过，也仍需使用独立开发场景做 matched Correct/CF 闭环评估。专家关节轨迹误差下降不能替代成功谓词。

## Checkpoint 与测试

输出是标准 `fastwam_lora_adapter_v1`，包含继承的 Video adapter 和更新后的 Action adapter，可继续由普通推理加载器读取。`step` 记录初始 step1000 加本次 optimizer steps；`robotwin_same_state_repair` 单独记录修复目标、数据来源、更新范围和本次步数。原 `lora_config` 保留加载结构兼容性，实际实验目标以 repair metadata 和 plan 为准；这些小集合 checkpoint 不应直接当作正式四池训练的最佳模型。

```bash
PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}" python -m unittest -q \
  tests.test_robotwin_same_state_repair \
  tests.test_robotwin_denoising_probe \
  tests.test_robotwin_no_eraf_probe
```

CPU 测试使用实际的小型 ActionDiT、VideoDiT、MoT、scheduler 和 LoRA，以及当前 FastWAM 的 predictor 函数体，覆盖前向数值一致、完整反向、两分支等权梯度、冻结范围、端点非零专项监督、保留 Video adapter 的序列化，以及从 worker 到结果汇总的短流程。预训练大模型、CUDA 显存和仿真成功率需由服务器运行验证。
