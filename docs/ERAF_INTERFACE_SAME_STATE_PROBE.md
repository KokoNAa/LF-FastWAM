# CF 接口同状态诊断（不训练）

本实验用于定位 warm V9.28 → V9.30 B（`mask_corrective_ranking`）的接口变化。
不把 A/B 的相同总成功率解释为行为相同，也不把本实验当成新的训练对照组。

## 固定项与变化项

| 预测 | compressor | injector |
|---|---|---|
| old_old | warm | warm |
| new_old | B | warm |
| old_new | warm | B |
| new_new | B | B |

四种预测复用同一次真实的 Video/ERAF 前向、同一语言/观测/传入记忆、同一个
Video KV cache 和同一初始动作噪声。每次仅通过同一个 Action Expert 计算动作。
探针不重新运行 ERAF、不接受任何 simulator 真值输入、不更新 caller 的记忆、
不推进 simulator；旧接口只是两个小模块的内存副本，不是第二个专家。

普通 driver 的前向完成后，探针额外执行四次动作去噪；其中相同接口的重复预测
必须与普通 driver 一致（归一化动作 max absolute tolerance = 1e-5，rtol = 0）。
探针作用域保存/恢复 Python、NumPy、CPU/CUDA RNG。没有启用时原评测路径不变。
开启探针时记录的 inference latency 包含额外诊断计算，不可当作部署延迟。

仅 `old_old` 和 `new_new` 作为闭环 driver。两条轨迹可以分叉：**只有各自采样点
内部的四种预测是同状态比较，不能将不同 driver 同一时间下标的状态视为相同。**
hybrid 没有执行闭环，不能为它们报告成功率或声称某一模块已被证明导致失败。
瞬时动作差异、两模块交互项用于定位后续验证方向，不是闭环成功的代理标签。

## 范围及复现检查

默认 case 文件：`configs/eval/libero10_cf_interface_probe_cases.json`。

- 共同退化：task3/trial1、task3/trial4、task9/trial2。
- 共同新增成功：task5/trial1、task7/trial4。
- 保留原试验的 5 次 trial 顺序，避免只运行 trial4 却遗漏前面环境/RNG 调用。
- 两个 driver × 四个 task × 五个 trial = **8 个任务 / 40 个 episode**。
- 仅指定的五个 case 在每条 driver 轨迹上开探针；默认每 5 次 replan 采样一次，
  包含 replan0。其余 trial 是协议复现检查，不额外计算四组合。
- 三张卡，每卡最多一个进程。先完成 warm driver 的 4 个任务；任一 task 成功
  trial 集合不能复现历史 warm 结果，则写 `calibration_failure.json` 并停止 B 阶段。
- B 结果也会与历史 B 对照。不一致会在 summary 中显式标记；同状态探针仍是
  当前状态下有效的预测比较，但不能据此直接解释历史 lost case。

CPU 预检核对共同 Base 绑定、共享 LoRA、非接口 guard/ERAF/gate tensors，以及
B 的固定 teacher 与指定 warm checkpoint 的真实 SHA256/接口权重。不同即停止。
这不等于证明服务器历史 Base 文件从未被改写；闭环复现检查仍然必要。

每个 probe 保存观测、sim state、语言、输入记忆、ERAF 输出、初始动作噪声、
压缩 token、注入 token、第一步 flow 和四种归一化动作。执行动作另以环境单位保存。
原始 tensor dtype/shape 也记录；bfloat16 数值在 NPZ 内以 float32 保存。

被测 episode 每个 policy step 记录所有具名关节的 qpos 及 source/CF 谓词真值，
包括柜门、抽屉、微波炉等关节，不依赖 region 名能否解析成可抓取物体。
这些是真值观察记录，不进入策略。`no_object_manipulated` 不代表门/抽屉没动。

## 三卡服务器命令

代码提交并拉取后执行。整个命令在子 bash 中运行；预检失败不会退出登录终端。
不会修改原 checkpoint、数据、manifest、历史结果；不调用 tmux，也不停止其他任务。
请确认 GPU 0、1、2 空闲。时间取决于实际闭环长度；当前未做 GPU 速度测量。

```bash
bash <<'BASH'
set -euo pipefail
REPO=/root/gpufree-data/LF-FastWAM
cd "$REPO"
git pull --ff-only origin main

WARM="$REPO/runs/libero_eraf_safe_gain_2cam224/eraf-safe-gain-libero10-eraf-safe-gain-bidir-v928fix-10k-seed42-20260828-225913/checkpoints/weights/step_010000.pt"
B_RESULTS="$REPO/evaluate_results/cf-ablation-3gpu-step250-counterfactual-seed42-trials5-20260831-224506/mask_corrective_ranking"
WARM_RESULTS="$REPO/evaluate_results/v928_forced_eraf_step10000_counterfactual_seed42_trials5_20260829-222021"
STAMP=$(date +%Y%m%d-%H%M%S)
OUT="$REPO/evaluate_results/eraf_interface_probe_3gpu_$STAMP"
LOG="/root/gpufree-data/eraf-interface-probe-3gpu-$STAMP.log"
PID_FILE="/root/gpufree-data/eraf-interface-probe-3gpu-$STAMP.pid"
export PYTHONPATH="$REPO/src:$REPO:${PYTHONPATH:-}"

ARGS=(
  --source-config "$B_RESULTS/manager_config.yaml"
  --warm-checkpoint "$WARM"
  --warm-results "$WARM_RESULTS"
  --candidate-results "$B_RESULTS"
  --output "$OUT"
  --gpus 0 1 2
  --stride-replans 5
)

# Read-only plan; fails before background launch on absent inputs/protocol mismatch.
/opt/conda/bin/python scripts/eval_libero_eraf_interface_probe.py "${ARGS[@]}"

nohup env CUDA_VISIBLE_DEVICES=0,1,2 \
  DIFFSYNTH_MODEL_BASE_PATH=/root/gpufree-data/fastwam/FastWAM/checkpoints \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  /opt/conda/bin/python -u scripts/eval_libero_eraf_interface_probe.py \
  "${ARGS[@]}" --execute > "$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$PID_FILE"
printf 'PID=%s\nPID_FILE=%s\nLOG=%s\nOUT=%s\n' "$PID" "$PID_FILE" "$LOG" "$OUT"
printf 'Launched only; inspect the log for preflight/worker success.\n'
BASH
```

监控启动命令打印的 `LOG`（替换为完整实际路径）：

```bash
tail -F /root/gpufree-data/eraf-interface-probe-3gpu-实际时间戳.log
```

每个任务详情在 `OUT/old_old_taskN/worker.log` 或 `OUT/new_new_taskN/worker.log`。
失败时停止尚未启动的队列，已经运行的任务允许正常结束以保留诊断文件。
最终查看 `OUT/summary.json`；只有日志出现 `[ALL_DONE]` 才表示八个任务全部完成。
输出目录拒绝复用，不支持悄悄覆盖或自动续跑部分结果。

## 结果阅读顺序

1. 先确认 driver repeat 校验、warm/B 历史成功 trial 复现情况和 checkpoint 审计。
2. 查失败 case 的关节 qpos/CF 谓词变化时间，再查此前采样点的四组合动作。
3. 对照恢复单一旧接口能否让动作靠近 warm；同时查看新增成功 case 是否也被回滚。
4. 两个接口可能共同适配。较大 hybrid 差异或交互项不能单独证明某模块有错；
   真正的成功恢复仍需下一步受控闭环验证。本轮不调 gate，不追加训练，不选新部署模型。
