# TC-FastWAM V5：反事实动作正监督

## 动机

TC-Full V4 在 LIBERO-Object 的 50 个 CIS episode 中得到：

- 反事实成功、目标物体抓取和目标物体抬升均为 `0/50`；
- `56%` episode 仍完成源目标；
- `28%` episode 操作无关物体。

V4 的 `L_CF` 只把替代语言从源任务未来推开，并不告诉部署策略替代语言应对应什么动作。V5 增加训练专用的 Counterfactual Action Positive（CAP）监督。

## 状态安全的正监督

不能把另一个 episode 的原始动作直接当作当前状态的标签：机器人位姿和物体位置可能不同。V5 使用真实示范建立任务级 EMA 原型：

```text
正确任务样本 (V_j, L_j, A_j)
  ├─ 部署 Router query residual ──> P_Q[j]
  └─ Action Effect Encoder(A_j) ──> P_A[j]
```

其中 `P_Q[j]` 直接处于 Action Expert 的 query 接口，且已经接受真实 `L_action` 与 M1 蒸馏；`P_A[j]` 来自任务 `j` 的真实动作示范。原型跨 GPU 汇总并按任务 ID 做 EMA 更新，不进入部署 checkpoint。

对源状态 `V_i` 和可执行替代指令 `L_j`，V5 运行与部署一致的第二条策略分支：

```text
same source state V_i + alternate language L_j
  -> frozen Video Expert
  -> trainable Transition Router
  -> frozen Action Expert
  -> alternate action prediction
```

监督由两部分组成：

1. 正向对齐：替代分支 query residual 对齐 `P_Q[j]`，intent embedding 对齐真实动作原型 `P_A[j]`；
2. 源动作分离：替代分支不能继续拟合源任务动作。使用有界 hinge，达到 margin 后不再鼓励无界偏移。

不同状态的 raw action chunk 从不互相复制。

## 总损失

```text
L_V5
= L_V4
 + lambda_CAP * schedule * L_CAP-positive
 + lambda_sep * schedule * L_action-separation
```

默认值：

```text
lambda_CAP       = 0.10
query weight     = 1.00
action weight    = 1.00
lambda_sep       = 0.05
separation margin= 0.05
prototype EMA    = 0.95
prototype slots  = 64
```

前 5% step 只收集动作原型；之后与 V4 contract schedule 同步启用。M1 LoRA、Video Expert、Action Expert 继续冻结，正确指令路径继续使用 M1 policy distillation。

## 关键日志

```text
loss_counterfactual_action_positive
loss_counterfactual_action_query_positive
loss_counterfactual_action_effect_positive
loss_counterfactual_action_separation
sim_CAP_query_positive
sim_CAP_action_positive
counterfactual_action_prototype_retrieval_acc
counterfactual_action_positive_valid_fraction
counterfactual_action_source_gap
counterfactual_action_separation_satisfied_fraction
counterfactual_action_policy_branch_active
policy_student_teacher_mse
loss_action
```

## 训练入口

V5 从策略保护的 V4 checkpoint 续训：

```bash
bash scripts/train_tc_counterfactual_action_object.sh \
  4 \
  /path/to/tc-full-v4-step_004000.pt \
  42 \
  4000
```

正式训练前将最后一个参数改成 `2` 做四卡 smoke。正式评估仍使用 Correct、Shuffle 与带行为分类的 50-episode CIS；不能只用 representation retrieval 指标作为语言遵守结论。
