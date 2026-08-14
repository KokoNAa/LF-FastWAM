# TC-FastWAM Stage 2 / TC-Full v4

## 范围

TC-Full v4 在已验证的策略保护版 TC-C v3 上增加：

- Action Effect Encoder，得到动作诱导转移表示 `z_A`；
- 对称 Action–Future InfoNCE：`L_AF`；
- 审计过的同场景反事实指令表示 `z_L-`；
- consequence-level margin ranking：`L_CF`；
- task-id-aware batch negative mask，避免把同任务轨迹误当作负样本。

本阶段仍属于任务书的 Phase T1：保留原 FastWAM future-video 分支及其直接语言路径，**不**启用 action-conditioned video。后者属于 Stage 3 / TC-Dyn。

## 训练数据流

```text
current observation + correct instruction
  -> Transition Router
  -> z_L+

current observation + audited same-scene wrong instruction
  -> shared Transition Router
  -> z_L-

current observation + clean GT action chunk + proprio
  -> training-only Action Effect Encoder
  -> z_A

clean current/future video latent change
  -> training-only Outcome Encoder
  -> z_F
```

策略 Router 继续读取 Video Expert final hidden，以保持 v3 已恢复的动作能力。用于 `L_LF/L_CF` 的正负 intent 表示则共同读取语言中性的 current patch token，防止正指令通过 Video Expert 的 T1 语言路径泄漏到视觉输入，使反事实 ranking 变成伪任务。

## 损失

```text
L_total
= L_video
 + L_action
 + lambda_distill * L_M1_distill
 + lambda_contract * schedule * (L_LF + lambda_AF * L_AF)
 + lambda_CF * schedule * L_CF
```

默认值：

```text
lambda_contract = 0.05
lambda_AF       = 1.0
lambda_CF       = 0.05
margin_CF       = 0.2
temperature     = 0.07
```

前 5% step 关闭 Contract/CF，随后 5% 线性打开。M1 joint-MoT 教师和原始 LoRA/Action Expert 继续严格冻结；优化器只能看到 Router、projection、Outcome Encoder 和 Action Effect Encoder。

## 反事实数据

`scripts/prepare_libero_object_interventions.py` 生成一对一、实体存在且可执行的 Object manifest。`RobotVideoDataset` 按默认 50% 概率启用显式 hard negative，并跳过没有有效 action/future transition 的 padding 样本。

训练样本新增：

```text
negative_context
negative_context_mask
negative_valid
negative_type
transition_task_id
```

`transition_task_id` 还用于屏蔽跨卡 InfoNCE 中的同任务 false negatives。

## Checkpoint

TC-Full 使用 `transition_contract.version=4`。它可以从保护策略的 TC-C v3 adapter 初始化：

- Router、intent/outcome projection、M1 LoRA 和 transition queries 精确恢复；
- 只初始化 v4 新增的 Action Effect Encoder；
- 保存后 metadata 标记 `use_action_effect=true`、`use_cf_ranking=true`；
- v2/v3/B0/M1 原有加载路径保持不变。

## 必看日志

```text
loss_action
loss_policy_distillation
policy_student_teacher_mse
loss_language_future_contract
loss_action_future_contract
loss_counterfactual_ranking
sim_LF_margin
sim_AF_margin
sim_CF_margin
contract_retrieval_acc
contract_retrieval_acc_AF
counterfactual_margin_satisfied_fraction
counterfactual_valid_fraction
contract_same_task_negative_fraction_LF
transition_query_pairwise_cosine
```

## 训练入口

```bash
bash scripts/train_tc_full_object.sh \
  4 \
  /path/to/tc-c-v3-step_004000.pt \
  42 \
  4000
```

正式训练前先将最后一个参数改为 `2` 完成四卡 smoke test。
