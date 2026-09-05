# RoboTwin stacking 单任务与指令差异修复

## 这次诊断确认了什么

`20260905-164002` 的逐状态记录显示两种失败模式。ranking 在三种评估噪声下始终是 `both_target`；前 24 步约 99.82% / 99.88% 的更新能量属于共同动作更新，指令差异误差几乎不变。它的 CF 动作 RMSE 实际改善了，不能把上一轮平均 CF 退化解释成两个任务都退化。

stacking 才是本轮平均 CF 退化的主要来源：普通配对训练使其 CF RMSE 从 0.24737 升至 0.40896，anchor 组升至 0.36264；语言动作差异在专家差异方向上的投影从 −0.332 变成 −0.481 / −0.386，方向更偏离正确目标。对应的指令差异 MSE 增加约 20.0% / 5.7%。两组的共同动作 MSE 都下降，因此降低平均动作损失可以掩盖条件对应关系恶化。

梯度在全部 300 步存在；普通组未发生裁剪，anchor 组仅 2/300 步发生裁剪。ranking 的 source 正监督占该任务正监督 loss 的约 99.4%，在普通组前 50 步占两个任务总 loss 约 85.5%。这只是目标函数数值的占比，不能直接推导出相同比例的梯度贡献或证明任务间梯度冲突。

stacking 的 CF 纯噪声训练窗口 MSE 从 0.2980 升至 0.3297；这些窗口仍使用不同的随机噪声，因此下一轮增加固定输入评估，而不是直接断言损失已经拟合好、只有采样出了问题。

## 实验干预

两组都仅训练 historical stacking 的一个配对初始状态，另九个配对初始状态用于回归检查，包括原先参与修复的 historical ranking。训练资格、RGB/proprio 一致性、正确专家来源和非 final 划分仍沿用原审计。

| 组别 | 连续噪声正监督 | 纯噪声正监督 |
| --- | --- | --- |
| `paired_flow_anchor` | 原 source/CF 正确 flow loss | `0.25 × (C + D)` |
| `paired_flow_anchor_delta` | 完全相同 | `0.25 × (C + 4D)` |

`C` 是两条速度预测的平均值相对两条正确速度目标平均值的 MSE；`D` 是速度差误差的 MSE 除以 4。设 source/CF 速度预测为 `vs,vt`，正确速度为 `ys,yt`：

```text
C = mean(((vs+vt - ys-yt)/2)^2)
D = mean(((vt-vs - (yt-ys))/2)^2)
C + D = (MSE(vs,ys) + MSE(vt,yt))/2
```

因此新组是在已有正监督中提高差异部分的权重，没有增加新的标签、排名损失或错误分支惩罚。两条预测都接收梯度。sigma=1 时两个 noisy action 相同，速度差目标 `yt-ys` 等于专家动作差的负值（除 scheduler dtype 舍入）。差异项放大到 4 是预先固定的小规模干预值，尚未证明是合适的最终超参数。

两组均从原 step1000、全新的 AdamW 状态开始，学习率 5e-6，梯度裁剪上限 1，只更新 Action LoRA。Video/文字/观测编码全部冻结，部署缓存、32 个动作与 10 步生成保持不变。两组每步的前向/反向调用次数相同；额外差异权重会改变梯度大小和方向，因此不声称更新范数相同。

`--source-repair` 指向上一轮已完成实验，继承其原始 probe，以及 stacking 每一步已经记录的噪声、timestep 和 loss 归一化分母。原先有两个修复状态，隔离后仍保留分母 2，避免把 stacking 项的系数悄悄翻倍。重新生成噪声后核对 hash、time 和 scheduler weight。它不会加载上一轮退化的 repair checkpoint，也不会恢复旧 optimizer。该方式只允许选取原修复状态的子集，步数不超过原记录。

这样，旧混合训练与新单任务对照之间主要移除了 ranking 的训练贡献；两组新实验之间只改变差异项权重。由于新组轨迹会逐渐不同，结论仍需结合逐状态结果，而不能仅以总 loss 排名。

## 服务器命令

```bash
cd /root/gpufree-data/LF-FastWAM
git pull --ff-only origin main
export DIFFSYNTH_MODEL_BASE_PATH=/root/gpufree-data/fastwam/FastWAM/checkpoints
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
REPAIR_OUT="$PWD/runs/robotwin_same_state_repair/$(date +%Y%m%d-%H%M%S)-stack-delta"
nohup /opt/conda/bin/python -u scripts/train_robotwin_same_state_repair.py run \
  --source-repair /root/gpufree-data/LF-FastWAM/runs/robotwin_same_state_repair/20260905-164002 \
  --pairs stack_blocks_two_green_on_red_to_red_on_green \
  --arms paired_flow_anchor paired_flow_anchor_delta \
  --conditional-anchor-gain 4 --anchor-weight 0.25 --learning-rate 5e-6 \
  --fixed-flow-sigmas 0.5 0.9 1.0 --audit-anchor-gradients \
  --steps 300 --eval-every 50 --gpus 0 1 \
  --output "$REPAIR_OUT" --execute \
  > "$REPAIR_OUT.driver.log" 2>&1 < /dev/null &
echo "PID=$! OUTPUT=$REPAIR_OUT"
tail -f "$REPAIR_OUT.driver.log"
```

一张卡可用 `--gpus 0` 顺序执行。去掉 `--execute` 只检查计划。结束后在新 shell 运行：

```bash
cd /root/gpufree-data/LF-FastWAM
/opt/conda/bin/python scripts/train_robotwin_same_state_repair.py report latest
```

回传整段输出。检查 `[report]` 路径以 `-stack-delta` 结尾；如果新运行失败，`latest` 仍可能指向以前的已完成实验。driver 会打印失败 worker 的日志尾部。

## 新增观测与验收

- 在 step 0 和每 50 步，用相同的 42/43/44 噪声、sigma=0.5/0.9/1.0 做正确分支 flow 评估，分别记录前 24/32 步的 source/CF、共同误差和差异误差。`fixed_flow_summary.csv` 汇总逐状态结果，原始行在 `evaluation_*.json`。只有 sigma=1 的 noisy action 跨语言相同；低 sigma 的差异含有各自专家动作输入的影响，不能当成纯语言敏感性。
- 初始模型另记录一次 sigma=1、seed42 的 `C`/`D` 梯度范数和夹角，用于检查条件差异能否通过当前 Action LoRA 得到梯度。此探针不写入 optimizer 梯度，也不更新权重。它是局部、未加权的端点分量诊断，不能替代全训练目标或跨任务的梯度分析。
- 保留完整 10 步生成、两种动作时域、所有回归状态和生产 wrapper 重放检查。单状态的 `fit_rows` 为 6，即三种噪声 × 两个时域。`best` 仍要求两个正确分支都改善至少 5%、双向专家/语言对应正确、回归检查通过；固定 flow loss 下降不能替代这些条件。

若只有提高差异权重的一组通过，支持将条件差异监督继续扩展验证；若单任务对照也通过，则优先检查混合任务的训练干扰和权重。若固定输入差异误差已下降、完整生成仍错误，下一步检查去噪轨迹；若固定纯噪声监督也无法纠正差异，再检查 Action/Video 条件通路和优化能力。以上均是离线可学性实验，正式采用仍需独立场景闭环验收。

## 本地测试

```bash
PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}" python -m unittest -q \
  tests.test_robotwin_same_state_repair \
  tests.test_robotwin_same_state_repair_diagnostics \
  tests.test_robotwin_denoising_probe \
  tests.test_robotwin_no_eraf_probe
```

覆盖原监督与 gain=1 的梯度等价、差异目标的负号及两个分支的梯度、BF16 主干/FP32 LoRA、新旧输出兼容、固定输入可重复性、只读梯度探针、沿用记录噪声与归一化后重新训练的权重精确复现，以及不完整固定输入评估拒绝汇总。CUDA 大模型的显存和效果由服务器实验验证。
