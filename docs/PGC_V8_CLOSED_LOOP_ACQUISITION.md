# PGC-FastWAM V8：闭环目标获取修复

## 目标与实验依据

V8 只解决当前最优先的问题：提高反事实指令下正确目标物的抓取与拿起率，暂不继续优化拿起后的搬运和放置。

以 LIBERO-Object 的 PGC V5、50 个 CIS episode 为基线：

- CIS 成功：16/50（32%）
- 正确目标拿起：21/50（42%）
- 已拿起目标后的条件完成率：16/21（76.2%）
- Correct：10/10（100%）

V5 completion continuation 将条件完成率提高到 16/19（84.2%），但正确目标拿起率下降到 19/50，CIS 仍为 32%。因此 V8 不再训练 completion，而是回到原始 V5 checkpoint，修复目标获取阶段。

V7 的训练—部署残差审计进一步确认：Proposal 在离线训练状态上的 residual RMS 约为 0.722，而在闭环 LIBERO 状态上只有约 0.00549，闭环/训练比约为 0.0076。checkpoint 加载完全一致，文本上下文也基本一致，主要问题是闭环状态分布偏移，而不是保存、加载或文本缓存错误。

## 不变量

V8 保留以下 V5 部件和部署契约：

- released FastWAM Base Action Expert：冻结；
- V5 GoalGraph / language representation：冻结；
- V5 Verifier 和 hard gate：冻结；
- V5 ActionChunkProposal：从 V5 checkpoint 精确恢复后，作为唯一可训练模块；
- 不创建第二个 Action Expert，不启用 LoRA，不修改视频专家。

因此 V8 是一次小型 sidecar 续训，而不是全量微调。Correct 的最终保护仍由冻结 Base 和 conservative hard gate 提供；训练期间还保留 native/source 零残差约束。

## 数据管线

### 1. 采集真实闭环失败状态

使用 V5、`instruction_condition=counterfactual`、`gate_mode=counterfactual` 执行 CIS。每次策略重新规划前，若目标物尚未被抓取或拿起，则保存：

- 精确 MuJoCo simulator state；
- task、pair、source/counterfactual 指令与 goal predicate；
- episode、policy step、replan index；
- checkpoint 和 state SHA256。

episode 结束后，仅当正确目标从未被拿起时才提交这些状态。每个 GPU worker 写入独立的 task/trial 目录，避免并发覆盖。

### 2. 构建 replay-verified corrective trajectory

对每个失败闭环状态，构建器从原始、已审计的 PGC 成功轨迹中检索 pre-grasp expert action suffix。检索特征只用于提出候选，不能作为监督有效性的依据。候选必须满足：

1. 从捕获的精确 simulator state 恢复；
2. 第一次 oracle replay 确实拿起反事实目标；
3. 第二次录制 replay 再次拿起同一目标；
4. 轨迹、state digest、pair 和 action count 写入双重 audit/index。

只有两次 replay 都通过的 action suffix 才进入 LeRobot 训练集。没有失败 episode 的 pair 不强制生成纠正数据，继续由原始 V5 离线数据锚定。

### 3. 混合采样

训练数据分为三个互不重叠的 split：

- native LIBERO 成功数据；
- 原始 offline PGC counterfactual 成功数据；
- V8 exact-state closed-loop corrective 数据。

默认先在 counterfactual 半区内将 corrective 数据重复 4 倍，再将 native 半区重复到相同样本数，保持整体 native : counterfactual = 1 : 1。默认 loss 权重为：

- closed-loop corrective action：2.0；
- offline counterfactual action：1.0；
- native zero-residual guard multiplier：0.10；
- source-language same-state zero residual、goal/residual separation 和 residual regularization：沿用 V5。

## 训练与 checkpoint 契约

V8 必须从 suite-specific `fastwam_policy_guard_v5` checkpoint warm-start；不能从 Base 随机初始化 V8，也不能混用 V6/V7 checkpoint。加载时要求 V5 sidecar tensor 全量、同形状恢复，V8 本身不新增 tensor。

V8 checkpoint 格式为 `fastwam_policy_guard_v8`，metadata 必须包含：

- `counterfactual_tuning=closed_loop_replay_verified_target_acquisition_residual`
- `closed_loop_corrective_format=pgc_libero_closed_loop_corrective_v1`
- `acquisition_only=true`
- `closed_loop_trainable_scope=action_chunk_proposal_only`
- `warm_start_contract=exact_pgc_v5_sidecars`

## 训练监控

重点监控：

- `pgc_v8_closed_loop_action_loss`：真实闭环失败状态上的 action loss；
- `pgc_v8_offline_action_loss`：原始 PGC 成功状态上的 action loss；
- `pgc_v8_closed_loop_prefix_mse_improvement`：纠正 Proposal 相对 Base 在闭环状态上的前缀 MSE 改善，目标应为正；
- `pgc_v8_offline_prefix_mse_improvement`：不能持续显著恶化；
- `pgc_v5_same_state_source_residual_mse`：source language 下应保持低值；
- `pgc_native_residual_mse` 与 residual RMS/saturation：防止非反事实路径漂移；
- 启动日志中的 trainable parameter 名称必须全部属于 `policy_guard_modules.action_chunk_proposal.*`。

## 评估顺序与准入门槛

1. 2-step smoke，检查 V5→V8 精确恢复、数据契约和有限 loss；
2. 训练中间 checkpoint 做 Correct 10 episodes；
3. Correct 必须保持 10/10，才进入 CIS；
4. CIS 先跑 10 episodes 做行为诊断，再跑 50 episodes；
5. V8 的第一主指标是 `target_object_lifted`，必须高于 V5 的 21/50（42%）；
6. 同时报告 CIS、source goal、other object、no object 和 placement failure，不能用 lift 提升掩盖错误物体率上升；
7. 若 target lift 提升，再以该 V8 checkpoint 作为后续 post-grasp completion 阶段的冻结起点。

V8 不应以离线 action loss 下降作为成功结论；最终结论只由闭环 Correct 和 CIS 行为统计给出。
