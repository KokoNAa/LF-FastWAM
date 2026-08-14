# TC-FastWAM Stage 1 v2/v3（TC-C）实现说明

本实现按任务书跳过 Stage 0，直接进入 Stage 1。v1 虽然 LF retrieval 达到 96.5%，但 Object Correct SR 只有 50%，说明 patch-level routing 破坏了策略 grounding。v2 将 Router 的视觉输入改成完整 Video Expert 最后一层 hidden，并从已训练 M1 LoRA 初始化，通过恢复调度平滑迁移到纯 Router。v2 在 Object step 3500 的 5-trial Correct SR 为 70%，仍低于原 M1，因此 v3 增加冻结 M1 教师蒸馏和强策略保护，避免继续训练时教师策略本身发生漂移。

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

最终部署时，Action token 和 routed query 在 MoT 中都不能直接读取 raw video；Action Expert 的 cross-attention 也只保留 proprio，不直接读取 T5。训练恢复期前 10% 冻结已恢复的 M1 adapter，并直接运行原始 joint-MoT posterior-query 策略；Router 所需的最终 Video hidden 和过渡期所需的逐层 Video KV 均从同一次 joint 前向导出，因此 BF16 attention 和 LoRA dropout 下也保持函数级一致。同时 Contract 在后台训练完整 Router；随后 20% 解冻 LoRA，并以同一次 joint-M1 输出与纯 Router 动作流线性混合预测速度，30% 之后完全关闭 M1 shortcut、切换到顺序 Video→Router→Action 路径。这样切换的是完整策略函数，而不只是输入 Query embedding。

训练期另以 clean GT current/future latents 经 Video Expert patch embedding 后的 hidden 均值差构造 `z_F`，以 routed intent 的池化表示构造 `z_L`。`z_F` 默认在进入 Outcome projection 前 stop-gradient，避免 contract 反向扰动 Video Expert；这只增加一次 patch embedding，不会复制 30 层 Video Transformer。两者归一化后使用对称 InfoNCE。跨卡训练会收集全局 in-batch negatives；单卡 batch size 1 返回有限的零 contract loss，不会崩溃。

Stage 1 明确不包含：`z_A`、Action–Future contract、counterfactual ranking、action-conditioned video、移除 Video Expert 的直接语言路径。这些分别属于 Stage 2/3。

## v3：冻结 M1 教师、纯 Router 学生

v3 不复制第二个 6.8B 模型。每个训练 batch 先用加载后的 M1 LoRA 执行一次精确的 joint-MoT posterior 前向，并在 `torch.no_grad()` 下得到教师动作流速度、Video KV cache 与最终 Video hidden；随后用同一份 Video 表示执行纯 Router 学生动作路径。教师和学生使用完全相同的 noisy action、action timestep、语言和状态条件。

训练目标为：

```text
L = L_action(student, GT flow)
  + lambda_KD * MSE(student flow, frozen M1 teacher flow)
  + lambda_LF(step) * InfoNCE(z_L, z_F)
  + L_video
```

其中 `L_video` 在冻结 Video Expert 后只保留监控意义。M1 的 Video Expert、Action Expert、原 LoRA、latent queries 和 action head 全部 `requires_grad=false`，并且不进入 optimizer；optimizer 只包含 Router、intent projection 和 outcome encoder。Trainer 会在加载 checkpoint 后再次检查 policy optimizer tensor 数必须为 0，防止配置错误静默解冻教师。

与 v2 不同，v3 从第 1 步就训练纯 Router 学生，教师只负责提供保护目标，不再把教师输出作为学生输出进行 recovery blend。这样 `L_action` 和蒸馏损失从第 1 步就能给 Router 梯度。LF contract 仍保留独立的 5% warm-up + 5% ramp：`z_L` 的梯度更新 Router 与 intent projection，`z_F` 的梯度更新 outcome encoder；clean Video token 在进入 outcome encoder 前 stop-gradient，因此不会破坏冻结的 M1 policy。

## 训练与兼容性

- 默认配置仍关闭 Transition Contract，原 B0/M1 行为不变。
- TC-C 要求 prior/advantage 关闭。
- 保留 checkpoint key `latent_action_queries`，仅新增语义 alias `transition_queries`。
- TC LoRA checkpoint 保存 router、projection、outcome encoder 与架构 metadata。
- v2 正式训练必须从已完成的 Object M1 LoRA adapter 初始化；该 adapter 再引用官方 FastWAM base。
- v3 同样从 Object M1 LoRA adapter 初始化，但只训练 Transition Contract 模块；冻结的 M1 LoRA 权重仍会写入最终 adapter，使 checkpoint 可独立加载。
- 动作策略通过输出速度 blend 在恢复起点精确等价于 M1 posterior 路径；Router 本身保持标准初始化，并在 Contract 的 5% warm-up 结束后开始接收其梯度。
- v3 的 action/distillation 梯度从第 1 步优化 Router，LF representation 梯度按 contract warm-up 调度加入。
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

## v3 四卡训练

先在服务器拉取代码并运行门控测试：

```bash
cd /root/gpufree-data/LF-FastWAM
git pull --ff-only
/opt/conda/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

两步 smoke：

```bash
export DIFFSYNTH_MODEL_BASE_PATH=/root/gpufree-data/fastwam/FastWAM/checkpoints
M1_CKPT=/root/gpufree-data/LF-FastWAM/runs/libero_object_lf_lora_2cam224/lf-m1-lf-object-lora-1epoch-v1/checkpoints/weights/step_004207.pt

CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/train_tc_policy_distill_object.sh 4 "$M1_CKPT" 42 2
```

正式一轮使用 `nohup`：

```bash
RUN_TAG=v3-policy-distill-object-1epoch-seed42-v1 \
TC_POLICY_DISTILLATION_WEIGHT=1.0 \
TC_SAVE_TRAINING_STATE=true \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
nohup bash scripts/train_tc_policy_distill_object.sh \
  4 "$M1_CKPT" 42 null \
  > /root/gpufree-data/tc_v3_policy_distill_object.log 2>&1 &
```

如服务器关机，可从最近的完整 state 继续。`RUN_TAG` 必须保持与原训练一致：

```bash
RUN_TAG=v3-policy-distill-object-1epoch-seed42-v1 \
TC_SAVE_TRAINING_STATE=true \
TC_RESUME_STATE=/root/gpufree-data/LF-FastWAM/runs/libero_object_lf_lora_2cam224/tc-c-v3-policy-distill-object-1epoch-seed42-v1/checkpoints/state/step_XXXXXX \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
nohup bash scripts/train_tc_policy_distill_object.sh \
  4 "$M1_CKPT" 42 null \
  > /root/gpufree-data/tc_v3_policy_distill_object_resume.log 2>&1 &
```

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
policy_recovery_joint_m1
loss_policy_distillation
policy_student_teacher_mse
policy_teacher_action_norm
policy_teacher_frozen
policy_distillation_effective_weight
router_recovery_schedule_scale
transition_contract_scale
```

Stage 1 的正式 gate 仍需服务器实验确认：Correct SR 不低于 B0 5pp、LF retrieval 显著高于随机、hard-negative future-similarity margin 大于 0、query 不坍缩、推理 p50 不超过 B0 的 1.10 倍。
