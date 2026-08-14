# TC-FastWAM V6：状态相关的目标物体监督

## 目标

V5 的反事实动作正监督使用任务级动作/Query EMA 原型。它能约束替代指令不要复现源动作，但原型不知道目标物体在**当前画面**中的位置，因此可能只把策略从源目标推开，仍无法把动作引向替代目标。

V6 保留 V4 的 LF/AF/CF、V5 的 CAP，以及冻结 M1 教师蒸馏，并新增一个状态相关的目标定位分支：

```text
当前双目视觉 patch + 当前指令
        -> StateConditionedTargetGrounder
        -> 当前状态上的目标 patch 分布
        -> Transition Router visual-attention bias
        -> Action Expert
```

部署时只增加一个小型 Grounder；不需要检测器、分割器、模拟器坐标或额外模型文件。

## 监督如何得到

现有 LeRobot 数据只包含双目 RGB、机器人 proprio 和动作，没有物体框或 mask。V6 因而使用训练期伪标签，而不虚构不存在的标注。

### 正确指令

对同一示范的 clean video latent，计算当前 latent 与未来 latent 的局部变化，池化到当前 Video patch 网格并保留变化最大的区域，得到语言无关的 interaction teacher：

```text
T_correct(V_0, V_future) -> p_teacher(patch | current state)
```

Grounder 在正确语言下输出的当前 patch 分布用 soft-label CE 对齐这个 teacher。

### 替代指令

正确示范按任务 ID 更新训练专用的目标外观 EMA 原型。对于同一源状态和替代任务 ID，将替代任务的外观原型逐 patch 匹配到**当前源画面**，形成替代目标的空间伪标签：

```text
prototype(target task) x current-state patches
    -> p_target(patch | same current state, alternate language)
```

它不是跨 episode 复制 raw action，也不是任务平均动作标签。EMA bank 不写入 checkpoint；部署只保存并使用 Grounder 参数。

## 损失与策略保护

```text
L_ground
= w_correct * CE(p_correct, interaction_teacher)
 + w_counter * CE(p_counter, current-state target teacher)
 + w_sep * ReLU(cos(p_correct, p_counter) - overlap_margin)

L_V6 = L_V5 + schedule * lambda_ground * L_ground
```

默认配置：

```text
lambda_ground       = 0.10
w_correct           = 1.00
w_counter           = 1.00
w_sep               = 0.25
overlap_margin      = 0.25
Router grounding bias = 2.00
teacher top-k       = 0.15
```

前 5% step 的 Grounder policy bias 和 Grounding loss 为 0，仅积累外观原型；随后在 5% step 内线性启用。无训练进度状态的推理模型始终使用完整 Grounder bias。V5 的冻结 M1 teacher 与 `policy_student_teacher_mse` 继续保护正确动作能力。

## 训练

V6 必须从策略保护的 V5 或 V6 LoRA adapter 初始化：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/train_tc_state_grounding_object.sh \
  4 \
  /path/to/tc-v5-step_004000.pt \
  42 \
  4000
```

先把最后一个参数改为 `2` 做 smoke。脚本复用现有 LIBERO-Object 数据和反事实 manifest，不需要重新制作数据集；如果文本 cache 已齐全，也不会下载新模型。

## 需要监控的指标

```text
loss_action
policy_student_teacher_mse
loss_state_grounding
loss_state_grounding_correct
loss_state_grounding_counterfactual
loss_state_grounding_separation
state_grounding_teacher_valid_fraction
state_grounding_target_valid_fraction
state_grounding_correct_top1_acc
state_grounding_counterfactual_top1_acc
state_grounding_positive_counterfactual_overlap
state_grounding_separation_satisfied_fraction
state_grounding_counterfactual_target_retrieval_acc
state_grounding_prototype_count
router_grounding_bias_scale
```

建议训练门控：prototype count 达到 10；target valid fraction 接近 1；正/反目标 overlap 持续下降；同时 `policy_student_teacher_mse` 不明显劣于 V5。最终结论仍以 Correct、Shuffle 和 50-episode CIS 行为评估为准，训练期定位指标不能替代真实成功率。

## 限制

interaction teacher 是弱监督：局部变化可能同时覆盖机械臂、被抓物体和放置区域。它能提供状态相关的位置约束，但不等同于真实物体 mask。若 V6 仍出现“能离开源物体、但无法稳定选择替代物体”，下一步应从 LIBERO simulator 导出目标物体 segmentation/pose，或离线生成可信的 object mask，再将本接口的伪标签替换为显式标签。
