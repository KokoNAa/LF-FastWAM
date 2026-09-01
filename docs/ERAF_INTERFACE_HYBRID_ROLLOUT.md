# ERAF 接口 hybrid 闭环因果实验

该实验执行两个真实闭环 driver：

- `new_old`：V9.30 B compressor + warm V9.28 injector。
- `old_new`：warm V9.28 compressor + V9.30 B injector。

两者均从 V9.30 B 的完整 resolved config 和 checkpoint 加载，共享同一个
Action Expert、同一个冻结 ERAF/GoalGraph/gate/LoRA。只在内存中替换两个小接口
模块之一；不训练、不调 gate、不读取 simulator 真值作为策略输入、不覆盖文件。

范围是 task 3/5/7/9，每个 task 保留完整 5-trial 顺序：两个 driver × 四个 task ×
五个 trial，共 8 个任务、40 个 episode。三张 GPU 上每卡最多一个 worker，按
driver 分两波运行。

启动器首先核对 warm/candidate checkpoint 的 Base、共享 LoRA、冻结 ERAF/gate、
teacher SHA，以及历史 warm/B 的 manifest、trial 数和五个选定 case 的行为方向。
输出目录存在即停止，不自动覆盖或续跑。

## 三卡启动命令

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
OUT="$REPO/evaluate_results/eraf_interface_hybrids_3gpu_$STAMP"
LOG="/root/gpufree-data/eraf-interface-hybrids-3gpu-$STAMP.log"
PID_FILE="/root/gpufree-data/eraf-interface-hybrids-3gpu-$STAMP.pid"
export PYTHONPATH="$REPO/src:$REPO:${PYTHONPATH:-}"

ARGS=(
  --source-config "$B_RESULTS/manager_config.yaml"
  --warm-checkpoint "$WARM"
  --warm-results "$WARM_RESULTS"
  --candidate-results "$B_RESULTS"
  --output "$OUT"
  --gpus 0 1 2
)

/opt/conda/bin/python scripts/eval_libero_eraf_interface_hybrids.py "${ARGS[@]}"

nohup env CUDA_VISIBLE_DEVICES=0,1,2 \
  DIFFSYNTH_MODEL_BASE_PATH=/root/gpufree-data/fastwam/FastWAM/checkpoints \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  /opt/conda/bin/python -u scripts/eval_libero_eraf_interface_hybrids.py \
  "${ARGS[@]}" --execute > "$LOG" 2>&1 < /dev/null &

HYBRID_PID=$!
printf '%s\n' "$HYBRID_PID" > "$PID_FILE"
printf 'PID=%s\nPID_FILE=%s\nLOG=%s\nOUT=%s\n' \
  "$HYBRID_PID" "$PID_FILE" "$LOG" "$OUT"
BASH
```

监控：

```bash
tail -F "$(ls -t /root/gpufree-data/eraf-interface-hybrids-3gpu-*.log | head -n 1)"
```

只有 `[ALL_DONE] 8 hybrid tasks; 40 episodes` 表示完成。之后查看：

```bash
bash <<'BASH'
ROOT="$(ls -dt /root/gpufree-data/LF-FastWAM/evaluate_results/eraf_interface_hybrids_3gpu_* | head -n 1)"
printf 'ROOT=%s\n' "$ROOT"
/opt/conda/bin/python -m json.tool "$ROOT/summary.json"
BASH
```

`driver_scores` 直接报告三个共同退化 case 恢复了多少、两个共同新增成功 case
保留了多少。`jobs` 还报告每个 task 在全部五次 trial 上与 warm/B 的一致数。

判读边界：hybrid 的成功/失败是闭环因果结果；不同 driver 的轨迹仍不是同状态轨迹，
不能按相同 policy step 比较 simulator state。
