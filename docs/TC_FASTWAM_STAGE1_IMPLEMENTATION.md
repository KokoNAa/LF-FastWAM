# TC-FastWAM Stage 1 v2（TC-C）实现说明

本实现按任务书跳过 Stage 0，直接进入 Stage 1。v1 虽然 LF retrieval 达到 96.5%，但 Object Correct SR 只有 50%，说明 patch-level routing 破坏了策略 grounding。v2 将 Router 的视觉输入改成完整 Video Expert 最后一层 hidden，并从已训练 M1 LoRA 初始化，通过恢复调度平滑迁移到纯 Router。

## 模型路径

训练与推理共享的动作路径为：

```text
Language + proprio + learnable transition queries
                         ↓
              intended transition queries
                         ↓
Final-layer current-frame Video hidden → Transition Visual Router
                         ↓
                  routed prefix H_route
                         ↓
                Action Expert [H_route, A_t]
```

最终部署时，Action token 和 routed query 在 MoT 中都不能直接读取 raw video；Action Expert 的 cross-attention 也只保留 proprio，不直接读取 T5。训练恢复期前 10% 冻结已恢复的 M1 adapter、精确运行 M1 posterior-query 策略，同时 Contract 在后台训练完整 Router；随后 20% 解冻 LoRA 并并行计算共享同一 Video KV cache 的 M1 与纯 Router 动作流，线性混合两者的预测速度，30% 之后完全关闭 M1 shortcut。这样切换的是完整策略函数，而不只是输入 Query embedding。

训练期另以 clean GT current/future latents 经 Video Expert patch embedding 后的 hidden 均值差构造 `z_F`，以 routed intent 的池化表示构造 `z_L`。`z_F` 默认在进入 Outcome projection 前 stop-gradient，避免 contract 反向扰动 Video Expert；这只增加一次 patch embedding，不会复制 30 层 Video Transformer。两者归一化后使用对称 InfoNCE。跨卡训练会收集全局 in-batch negatives；单卡 batch size 1 返回有限的零 contract loss，不会崩溃。

Stage 1 明确不包含：`z_A`、Action–Future contract、counterfactual ranking、action-conditioned video、移除 Video Expert 的直接语言路径。这些分别属于 Stage 2/3。

## 训练与兼容性

- 默认配置仍关闭 Transition Contract，原 B0/M1 行为不变。
- TC-C 要求 prior/advantage 关闭。
- 保留 checkpoint key `latent_action_queries`，仅新增语义 alias `transition_queries`。
- TC LoRA checkpoint 保存 router、projection、outcome encoder 与架构 metadata。
- v2 正式训练必须从已完成的 Object M1 LoRA adapter 初始化；该 adapter 再引用官方 FastWAM base。
- 动作策略通过输出速度 blend 在恢复起点精确等价于 M1 posterior 路径；Router 本身保持标准初始化并从第一步接收 Contract 梯度。
- 推理不执行 Outcome Encoder，也不读取 future teacher。
- contract loss 在前 5% optimizer steps 为零，随后 5% 线性升到目标权重。

## 四卡训练

完整一轮：

```bash
cd /root/gpufree-data/LF-FastWAM
export DIFFSYNTH_MODEL_BASE_PATH=/root/gpufree-data/fastwam/FastWAM/checkpoints

CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/train_tc_stage1_object.sh \
  4 \
  /root/gpufree-data/LF-FastWAM/runs/libero_object_lf_lora_2cam224/lf-m1-lf-object-lora-1epoch-v1/checkpoints/weights/step_004207.pt \
  42 \
  null 2>&1 | tee /root/gpufree-data/tc_stage1_object.log
```

两步 smoke test：把最后一个参数 `null` 改为 `2`。

## 必看日志

除原 `loss_video`、`loss_action` 外，确认以下字段均为有限值：

```text
loss_transition_contract
sim_LF_positive
sim_LF_negative
sim_LF_margin
contract_retrieval_acc
transition_query_pairwise_cosine
router_attention_entropy
router_top1_mass
router_top5_mass
router_route_scale
router_language_residual_norm
router_visual_residual_norm
router_policy_residual_norm
policy_recovery_output_gap
transition_contract_scale
```

Stage 1 的正式 gate 仍需服务器实验确认：Correct SR 不低于 B0 5pp、LF retrieval 显著高于随机、hard-negative future-similarity margin 大于 0、query 不坍缩、推理 p50 不超过 B0 的 1.10 倍。
