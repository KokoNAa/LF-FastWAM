# LF/TC/PGC-FastWAM 全量审计交接报告

> 快照日期：2026-08-17（Asia/Shanghai）
>
> 代码仓库：`KokoNAa/LF-FastWAM`
>
> 审计基线：`main @ e6a0891`（`feat: add PGC residual deployment audit`）
>
> 当前候选：PGC-FastWAM v7；V8 尚未实现
>
> 当前正式验证范围：以 LIBERO-Object 为主；其余 LIBERO suites 与 RoboTwin 尚未完成 PGC 全流程验证

## 0. 文档用途与证据等级

本文用于把当前代码、服务器模型、数据、训练监督、实验结果、失败机制和未完成项交给新的审计人员。它不是论文结果稿，也不把训练日志中的内部指标当作机器人语言遵守结论。

本文采用以下证据等级：

- **C（Code-confirmed）**：可由 `e6a0891` 当前代码直接确认。
- **R（Result-reported）**：来自服务器输出、汇总 JSON 或本项目对话中保存的实验结果；审计时仍应回到原始 JSON/视频复核。
- **D（Diagnosed）**：由专门审计程序和数值证据支持的诊断。
- **H（Hypothesis）**：当前最合理但尚未通过受控实验排除替代解释的假设。
- **T（To verify）**：缺少原始文件、哈希、重复 seed 或跨 suite 实验，不能视为最终事实。

本地代码仓库不包含服务器上的大模型、数据集和 rollout 结果。下文所有 `/root/gpufree-data/...` 路径都需要在训练服务器上核验。创建本文前，本地工作区还有一个未跟踪的旧状态文档 `docs/LF_FASTWAM_MODEL_AND_EXPERIMENT_STATUS.md`；本文没有覆盖或吸收它进入代码快照。

## 1. 审计结论摘要

截至当前快照，最稳妥的结论是：

1. **LF-FastWAM MVP 已完整实现并完成 B0/B1/M1 受控训练。** Query、无语言 Prior、Posterior Advantage、信息访问 mask、checkpoint 兼容和 LIBERO 语言干预评估链路均已落地。M1 在 Object 上保持了 Correct，但 Object CIS 为 `0/50`，说明“让语言影响动作”没有自动转化为“按替代语言完成目标”。**[C/R]**
2. **TC-FastWAM 建立了 Language–Future、Action–Future、反事实 ranking、教师蒸馏、CAP 和状态目标 grounding 等多组表征监督。** 它能学出很强的离线检索指标，也曾把 Correct 恢复到较高水平；但已测版本没有稳定产生 CIS，部分版本还造成夹取或放置动作漂移。**[C/R]**
3. **PGC-FastWAM 将策略保护改为结构性保证。** v3 以后只有一份冻结的 release Base；可训练 sidecar 只提出受限候选，守门器不通过时逐元素返回 Base。Correct 成功因此可以由 fallback 保住，不再要求 CF 分支本身承担所有原策略能力。**[C]**
4. **PGC v5/v6/v7 尚未给出统计上可信的 CIS 提升。** 在同一 Object 50-episode 量级下，exact Base CIS 为 `26%`，v5 为 `32%`，v6 为 `32%`，v7 为 `30%`。相对 Base 分别为 `+6pp/+6pp/+4pp`，低于任务书建议的 `+10pp` 门槛；各自 Wilson 95% 区间高度重叠。**[R]**
5. **PGC v7 的主要故障边界已经被定位到闭环状态分布。** 172/172 个 checkpoint tensor 精确恢复；训练 cache 与在线文本编码几乎一致；Proposal 在示教状态上的 residual RMS 约 `0.722`，而 2,426 次闭环决策中只有 `0.00549`，部署/训练比值约 `0.00761`，即约 **131 倍幅度塌缩**。**[D]**
6. **因此当前首要问题不是继续增加离线 mask/binder loss，而是 Proposal 对闭环状态的动作泛化。** 当前最强假设是：只有 50 条成功 CF episode，加上广覆盖 native “残差归零”监督，使 Proposal 学成了示教流形检测器；离开示教状态后自然回到近零修正。V8 应围绕 DAgger 风格闭环纠正数据和部署态验收设计，但当前代码仍只支持 PGC v1–v7。**[C/D/H]**
7. **任务书的完整覆盖尚未完成。** PGC 正式数据、训练和 50-episode CIS 目前只覆盖 LIBERO-Object；Spatial 只完成过 LF 早期实验，Goal、LIBERO-10 没有完成 PGC suite-specific 闭环；RoboTwin 尚未开始。不能把 Object 结果外推为全 LIBERO 或跨 benchmark 结论。**[R/T]**

## 2. 原始任务与验收口径

原任务书是 [`LF_FastWAM_MVP_Implementation_Spec.md`](/Users/feng/Downloads/LF_FastWAM_MVP_Implementation_Spec.md)。核心目标是：在尽量保持 FastWAM 标准任务成功率和低延迟的同时，减少视觉捷径，提高模型对语言变化的行为依赖。该任务书当前位于本机 Downloads、未被 Git 版本化；正式审计包应复制一份只读快照并记录 SHA-256。

任务书建议的工程 gate 为：

| Gate | 建议门槛 | 当前状态 |
|---|---:|---|
| 原能力保持 | 相对 Fast-WAM-FT，`SR_correct` 下降不超过 2pp | PGC guarded Correct 可通过，但主要依赖 exact Base fallback；候选分支需单独审计 |
| 语言增强 | LRG `+10pp`，或 DTL `-15pp`，或 CIS `+10pp`，满足至少一项 | LF Object 的单次 DTL 曾 `80%→60%`；PGC Object CIS 最高只比 exact Base 高 6pp |
| 非随机敏感 | Correct 不下降 | guarded 路径满足；部分 TC/强制 CF 路径不满足 |
| 推理成本 | p50 overhead 不超过 15% | LF MVP 初测满足；PGC v7 尚缺统一 Base/guarded/forced-CF 延迟复核表 |

四个评估条件必须严格区分：

- **Correct**：源场景 + 源指令 + 源成功谓词。
- **Null**：源场景 + 空语言；用于 Language Reliance Gap。
- **Shuffled / DTL**：源场景 + 替代指令，但仍检查源成功谓词；越低越少无视新语言执行默认任务。
- **Counterfactual / CIS**：源场景 + 替代指令 + 替代成功谓词；越高越能真正完成替代语言目标。

PGC 还必须区分三种部署模式：

- `gate_mode=base`：强制 exact Base，用于建立相同初始状态、相同 manifest 下的基线。
- `gate_mode=counterfactual`：强制 Proposal 候选，用于诊断候选本身，不代表安全部署。
- `gate_mode=guarded`：由 Verifier/support guard 决定是否覆盖，否则返回 exact Base。

**审计红线：guarded Correct=100% 不能单独证明 Proposal 保持了原策略；它也可能只表示所有决策都回退到了 Base。**

## 3. FastWAM 基础模型与服务器资产

### 3.1 基础结构

本项目始终保留 FastWAM 的 Video/Action 双 Expert MoT 和世界模型训练路径：

| 组成 | 配置 |
|---|---|
| 视频模型 | Wan2.2-TI2V-5B |
| Video Expert | 30 层，hidden 3072，24 heads，FFN 14336 |
| Action Expert | 30 层，hidden 1024，24 heads，FFN 4096 |
| 文本 | UMT5-XXL embedding，4096 维，最大 128 tokens |
| 输入 | agent-view + wrist-view，两路 224×224 |
| 动作 | 32-step action chunk，每步 7 维 |
| Proprio | 8 维 |
| Scheduler | Video/Action 各 1000 train timesteps，shift 5.0 |
| 推理 | LIBERO 实验通常使用 10 个 action inference steps |

当前基础配置见 [`configs/model/fastwam.yaml`](../configs/model/fastwam.yaml)。需要注意：该 YAML 默认仍开启 LF MVP、关闭 PGC；正式 PGC 不是“直接运行默认配置”，而是通过 [`configs/task/libero_pgc_2cam224.yaml`](../configs/task/libero_pgc_2cam224.yaml) 和 suite launcher 覆盖为 query-free Base、关闭 LF/TC、开启指定 PGC version。

### 3.2 已知服务器资产

| 资产 | 服务器路径 | 状态 |
|---|---|---|
| release Base checkpoint | `/root/gpufree-data/fastwam/FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224.pt` | 已用于全部主实验 **[R]** |
| normalization stats | `/root/gpufree-data/fastwam/FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json` | 已用于训练/评估 **[R]** |
| ActionDiT pretrain | `/root/gpufree-data/fastwam/FastWAM/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt` | 已发现 **[R]** |
| Wan2.2 model root | `/root/gpufree-data/fastwam/FastWAM/checkpoints/Wan-AI/Wan2.2-TI2V-5B` | 已发现 **[R]** |
| tokenizer root | `/root/gpufree-data/fastwam/FastWAM/checkpoints/Wan-AI/Wan2.1-T2V-1.3B` | 已发现 **[R]** |

审计时必须补齐每个文件的 `sha256sum`、字节数和 mtime。当前交接记录只有路径和已用事实，没有不可抵赖的哈希清单。

## 4. 三条方法线的版本演进

### 4.1 LF-FastWAM MVP

LF MVP 在 Action Expert 中把 token 序列从 `[A]` 改为 `[Q, A]`：

- 32 个 latent action queries，RoPE offset 512；
- Posterior Query 可读当前帧视觉、语言和 proprio；
- Prior Query 可读当前帧视觉和 proprio，但严格不可读语言；
- Action token 只读 Query、Action token 和 proprio，不直接读 raw video/T5；
- Prior/Posterior 共用 noisy action、action timestep、flow target 和 padding mask；
- 推理只执行 Posterior，Prior 只用于训练。

M1 损失为：

```text
L = L_video + L_action_post
  + 0.1 * L_action_prior
  + 0.1 * L_posterior_advantage
```

受控模型：

| 名称 | Query | Prior | Advantage |
|---|---:|---:|---:|
| B0 FastWAM-FT | 否 | 否 | 否 |
| B1 Query-only | 是 | 否 | 否 |
| M1 LF-FastWAM MVP | 是 | 是 | 是 |

LF 训练采用 rank-16 LoRA，并额外训练 latent queries、action input/output head 和 proprio projection。详细设计见 [`docs/LF_FASTWAM_MVP.md`](LF_FASTWAM_MVP.md)。

### 4.2 TC-FastWAM

TC 线试图把“指令意图”与“未来变化/动作效果”绑定起来：

| 版本 | 主要新增 | 策略保护 |
|---|---|---|
| TC-C v1 | Transition Queries + Visual Router + Language–Future contract | 不足；Correct 明显下降 |
| TC-C v2 | Router 读取完整 Video final hidden；M1→Router recovery schedule | 过渡期恢复 M1，但训练后仍有动作漂移 |
| TC-C v3 | 冻结 M1 joint-MoT 教师；Router 学生做动作蒸馏 | M1 teacher 不进入 optimizer |
| TC-Full v4 | Action Effect Encoder、Action–Future contract、反事实 consequence ranking | 继续冻结 M1；Correct 恢复较好 |
| TC v5 | CAP：反事实动作正监督、任务级 Query/Action EMA prototype | 仍依赖间接动作原型 |
| TC v6 | 状态相关目标 Grounder；未来变化伪标签和 appearance prototype | 增加 grounding，但 Correct 与 CIS 未同时改善 |

核心表征：

- **LF**：语言意图 `z_L` 与未来状态变化 `z_F` 对齐。
- **AF**：动作效果 `z_A` 与未来状态变化 `z_F` 对齐。
- **CF**：正确语言结果优于同场景错误语言结果的 margin ranking。
- **CAP**：为替代语言提供动作侧正向原型，而不只把它从源动作推开。

TC 文档入口：

- [`docs/TC_FASTWAM_STAGE1_IMPLEMENTATION.md`](TC_FASTWAM_STAGE1_IMPLEMENTATION.md)
- [`docs/TC_FASTWAM_FULL_IMPLEMENTATION.md`](TC_FASTWAM_FULL_IMPLEMENTATION.md)
- [`docs/TC_FASTWAM_COUNTERFACTUAL_ACTION_POSITIVE.md`](TC_FASTWAM_COUNTERFACTUAL_ACTION_POSITIVE.md)
- [`docs/TC_FASTWAM_STATE_TARGET_GROUNDING.md`](TC_FASTWAM_STATE_TARGET_GROUNDING.md)

### 4.3 PGC-FastWAM

PGC 的原则是：**不再让语言增强分支直接改写唯一策略，而是保留 exact Base，额外提出候选并做保守覆盖。**

| 版本 | 候选形式 | 关键问题/改进 |
|---|---|---|
| PGC v1 | 独立 CF Action Expert + LoRA | 候选动作破坏严重；Correct 强制候选为 0% |
| PGC v2 | 与 Base 对齐初始化的 CF Action Expert + action LoRA | 500–1500 steps 后强制候选 Correct 仍低；CIS 可出现完全不动作 |
| PGC v3 | 共享冻结 Base，在 flow velocity 上加有界 residual | 消除第二套 6.8B expert 和 Base 漂移 |
| PGC v4 | Base 完整 denoise 后，在最终 action chunk 上一次性加 residual；FP32 pairwise verifier | 解决训练/部署 residual 积分错位；仍出现闭环 residual 过小 |
| PGC v5 | 同状态 Source/CF 双语言、Source residual=0、执行前缀对齐、hard verifier negatives | Object CIS 从 Base 26% 到 32%，但误抓也增加 |
| PGC v6 | 无 direct-language residual 的 visual target binder，未来变化 teacher + task prototype | target grasp/lift 未超过 v5 |
| PGC v7 | 当前帧显式 target/source/aux masks + 多空间 object tokens | 绑定监督更直接，但闭环 Proposal residual 塌缩，CIS 30% |

PGC 主文档见 [`docs/PGC_FASTWAM.md`](PGC_FASTWAM.md)。

## 5. 当前候选：PGC-FastWAM v7

### 5.1 部署计算图

PGC v7 的部署路径可概括为：

```text
RGB + proprio + requested language
             │
             ├──────────────► frozen query-free FastWAM Base ─► a_base
             │
             └► language-neutral current visual patches
                        + requested language
                         │
             Spatial Object-Token Target Binder
                         │  8 camera-aware object tokens
                         ▼
              per-action-query cross attention
                         │
              Final-Action Chunk Proposal
                         │
          a_cf = stopgrad(a_base) + bounded_delta
                         │
        FP32 Pairwise Verifier + support guard
                         │
             ┌───────────┴───────────┐
             │ pass                  │ reject
             ▼                       ▼
            a_cf              exact elementwise a_base
```

**C 级确认：**

- Base Video/Action Expert 全冻结，PGC v3+ 禁止 LoRA；
- Base action interface 是 query-free joint MoT；
- Proposal 在最终归一化 32-step action chunk 上工作；
- 默认只给实际执行的前 10 步 full weight，后 22 步 loss 权重为 0.1；
- Proposal residual 使用 `tanh × cap`，默认 cap 2.0；
- Verifier 为 FP32 时序 pairwise raw advantage，不使用两个接近 1 的 sigmoid 绝对分数；
- support guard 检查 residual RMS 和饱和率；
- 未通过时返回 exact Base；
- v7 mask 仅训练使用，部署不需要 simulator segmentation。

### 5.2 Target Binder

`SpatialObjectTokenTargetBinder`：

- 在语言注入前的当前帧视觉 patch 上做语言条件打分；
- patch 加入二维位置和相机身份；
- 默认选择 8 个空间 object tokens；
- 32 个 action query 各自 cross-attend object tokens；
- 没有 direct text-to-Proposal residual；
- object-token 到 action query 的末层零初始化，V5→V7 热启动时 Proposal 初始输出不变。

相关代码：

- [`src/fastwam/models/wan22/policy_guard.py`](../src/fastwam/models/wan22/policy_guard.py)
- [`src/fastwam/models/wan22/fastwam.py`](../src/fastwam/models/wan22/fastwam.py)

### 5.3 当前 checkpoint contract

v7 checkpoint：

- `format = fastwam_policy_guard_v7`；
- 外部引用 release Base，而不是重复保存 6.8B 主模型；
- 只保存 policy-guard sidecars 和 architecture metadata；
- 本次部署审计读到 172 个 sidecar tensors，172 个全部加载，严格相等；
- 不含 v6 prototype bank；
- metadata 必须声明 `object_token_mask_grounded_paired_action_residual`、`spatial_object_tokens_no_direct_language_residual` 和 `robosuite_element_current_frame_training_only`。

服务器候选路径按已使用 run tag 记录为：

```text
/root/gpufree-data/LF-FastWAM/runs/libero_pgc_2cam224/
  pgc-object-pgc-v7-mask-grounded-4000-seed42-v1/
  checkpoints/weights/step_004000.pt
```

该相对路径已由 step-4000 完成日志确认；审计人员仍需用 checkpoint metadata 和 SHA-256 确认文件内容，不能只依据目录名。

## 6. 数据情况

### 6.1 Native 数据

当前 Object 训练使用：

```text
/root/gpufree-data/fastwam/FastWAM/data/libero_mujoco3.3.2/
  libero_object_no_noops_lerobot
```

训练 launcher 默认执行 native/CF 1:1 平衡采样。Native 样本的历史作用包括保留 Base 行为、监督 Proposal residual 归零、训练 Verifier；这也是 v8 需要重新审计的关键，因为广覆盖 native zero 可能压倒狭窄 CF 正样本。

### 6.2 直接反事实动作数据

PGC 不把另一条文本贴到原动作上，而是从 LIBERO 官方 HDF5 成功示教恢复/迁移状态，在 Source 环境中重放替代任务动作，并且只保存两次回放都满足替代谓词的 episode。

Object 数据目录：

```text
/root/gpufree-data/pgc_libero_data_v1/
  libero_object_pgc_counterfactual_lerobot
```

Manifest：

```text
/root/gpufree-data/pgc_libero_data_v1/manifests/libero_object_pgc.jsonl
```

最终采集结果：50 条成功 CF episode，每个 Source task 5 条。数据集必须包含：

- `meta/pgc_provenance.json`；
- `meta/pgc_episodes.jsonl`；
- `meta/pgc_initial_states/`；
- LeRobot RGB/proprio/action 数据；
- v7 的 `meta/pgc_v7_target_masks/index.json` 和逐 episode `.npz` sidecars。

### 6.3 Object 配对表

最终实际采集配对为：

| Source task | Source object | Counterfactual task | Target object | Episodes |
|---:|---|---:|---|---:|
| 0 | alphabet soup | 1 | cream cheese | 5 |
| 1 | cream cheese | 0 | alphabet soup | 5 |
| 2 | salad dressing | 0 | alphabet soup | 5 |
| 3 | bbq sauce | 4 | ketchup | 5 |
| 4 | ketchup | 2 | salad dressing | 5 |
| 5 | tomato sauce | 7 | milk | 5 |
| 6 | butter | 5 | tomato sauce | 5 |
| 7 | milk | 1 | cream cheese | 5 |
| 8 | chocolate pudding | 9 | orange juice | 5 |
| 9 | orange juice | 6 | butter | 5 |

**重要审计项：**这是每个 Source 覆盖 5 条，但不是 target-balanced 的一一置换。task 0 和 1 各作为 target 两次，task 3 和 8 从未作为 target。这一不平衡可能影响按目标物体统计的 CIS，后续必须报告 macro-by-source、macro-by-target 和 pair-level 结果，不能只报告总平均。

### 6.4 v7 mask 监督

数据构建阶段在每个成功 episode 上重放动作，用 robosuite `element` segmentation 生成：

- 请求 target object mask；
- Source instruction object mask；
- 一个可见 auxiliary object mask。

mask bit-pack 到 sidecar，不修改 RGB、action、normalization stats 或 provenance。训练时将 mask area-resample 到视觉 patch 网格，监督 target/source/aux attention、mask 内 mass 和跨物体 margin。

### 6.5 尚未完成的数据范围

| Benchmark/suite | Native 数据 | Direct CF actions | v7 masks | PGC 正式训练/评估 |
|---|---:|---:|---:|---:|
| LIBERO-Spatial | 有，早期 LF 已用 | 未完成可审计正式集 | 未完成 | 未完成 |
| LIBERO-Object | 有 | 50 条成功 episode | 已构建 | 已完成 v1–v7 多轮 |
| LIBERO-Goal | 服务器曾生成 manifest，但无完整 CF action 集 | 未完成 | 未完成 | 未完成 |
| LIBERO-10 | 服务器曾生成 manifest，但无完整 CF action 集 | 未完成 | 未完成 | 未完成 |
| RoboTwin | FastWAM 原仓库支持 | 未定义 PGC contract | 未完成 | 未开始 |

## 7. 当前 v7 训练监督和调度

### 7.1 Proposal/动作监督

- `loss_pgc_action`：成功 CF 轨迹对最终 Proposal action 的执行前缀加权正监督。
- `loss_pgc_v5_same_state_source_zero`：同一视觉状态换回 Source language，要求 residual 为零。
- `loss_pgc_native_residual_zero`：Native 状态要求 residual 为零。
- `loss_pgc_residual_regularization`：限制残差能量。
- `loss_pgc_residual_smoothness`：限制相邻动作步修正抖动。
- action gripper 维度默认权重 2.0。

### 7.2 语言—目标绑定监督

- `loss_pgc_v7_target_mask`：请求语言 attention 对齐 target mask。
- `loss_pgc_v7_source_mask`：Source language attention 对齐 source mask。
- `loss_pgc_v7_aux_mask`：auxiliary language/object 辅助监督。
- `loss_pgc_v7_mask_mass`：鼓励 attention mass 落在请求对象 mask 内。
- `loss_pgc_v7_cross_object`：请求对象相对 source/aux 对象满足 margin。

### 7.3 表征与配对监督

- 同状态 `goal_cf` / `goal_source` cosine-margin 分离；
- 同状态 `delta_cf` / `delta_source` 自适应 margin 分离；
- Goal–Action distributed contrastive alignment；
- Verifier 的 wrong-language candidate 和 mirrored bad candidate negatives。

### 7.4 Verifier 监督

Verifier 比较 `a_base` 与 `a_cf` 的完整时序 chunk，但只按实际执行前缀形成主要监督。目标由两个候选相对真实成功 action 的加权 MSE 改善产生；不是把所有 CF 数据行无条件标成正类。

### 7.5 4000-step 默认阶段

| optimizer steps | 训练重点 |
|---:|---|
| 0–999 | 只训练显式 mask binder；Action/Verifier 图断开，避免 weight decay 改动已有 v5 sidecars |
| 1000–1499 | 线性打开 Proposal/action objectives |
| 1500–1999 | Proposal 全开；Verifier 线性 ramp |
| 2000–4000 | Binder + Proposal + Verifier 全部训练 |

v7 默认从 suite-specific、已验证的 PGC v5 checkpoint 热启动，optimizer/scheduler 从 step 0 开始。当前训练脚本是 [`scripts/train_pgc_v7_libero_suite.sh`](../scripts/train_pgc_v7_libero_suite.sh)。

### 7.5 当前正式 v7 run 记录

服务器日志确认：

```text
run tag       = pgc-object-pgc-v7-mask-grounded-4000-seed42-v1
max step      = 4000
final weights = ./runs/libero_pgc_2cam224/pgc-object-pgc-v7-mask-grounded-4000-seed42-v1/checkpoints/weights/step_004000.pt
state         = None
```

抽查训练日志可见 `pgc_base_policy_frozen=1`，后期 `pgc_v7_action_training_scale=1`、`pgc_verifier_training_scale=1`，记录值均为有限数。训练正常到达 `max_steps`；退出时有 `destroy_process_group()` 未调用的 NCCL warning，但训练脚本报告 complete，weight checkpoint 随后也通过精确加载审计。由于 `state=None`，该 run 没有可用于逐 step 完整恢复的 optimizer/scheduler state；只能从权重做新 schedule 的 warm start，不能宣称是 bitwise continuation。

## 8. Checkpoint 与训练谱系

| 模型 | 已知/推定路径 | 备注 |
|---|---|---|
| FastWAM release | `/root/gpufree-data/fastwam/FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224.pt` | 所有主线共同 Base |
| LF Object M1 | `/root/gpufree-data/LF-FastWAM/runs/libero_object_lf_lora_2cam224/lf-m1-lf-object-lora-1epoch-v1/checkpoints/weights/step_004207.pt` | `fastwam_lora_adapter_v1`，1213 trainable tensors，含 latent queries **[R]** |
| TC Stage1 v1 | `/root/gpufree-data/LF-FastWAM/runs/libero_object_lf_lora_2cam224/tc-c-object-stage1-1epoch-seed42-v1/checkpoints/weights/step_004207.pt` | 38 TC tensors + 1213 M1 tensors **[R]** |
| TC v3 | `/root/gpufree-data/LF-FastWAM/runs/libero_object_lf_lora_2cam224/tc-c-v3-policy-distill-object-4000-seed42-v1/checkpoints/weights/step_004000.pt` | 冻结 M1 教师蒸馏 **[R]** |
| TC-Full v4 | `/root/gpufree-data/LF-FastWAM/runs/libero_object_lf_lora_2cam224/tc-full-v4-full-object-4000-seed42-v1/checkpoints/weights/step_004000.pt` | weight checkpoint 已保存；DeepSpeed state 保存曾因磁盘写失败报错，须核验 **[R/T]** |
| TC v6 中止点 | `/root/gpufree-data/LF-FastWAM/runs/libero_object_lf_lora_2cam224/tc-ground-v6-state-grounding-object-4000-seed42-v1/checkpoints/weights/step_001500.pt` | Correct gate 80%，随后放弃 **[R]** |
| PGC v7 | `/root/gpufree-data/LF-FastWAM/runs/libero_pgc_2cam224/pgc-object-pgc-v7-mask-grounded-4000-seed42-v1/checkpoints/weights/step_004000.pt` | 当前候选，step-4000 日志确认；哈希待补 **[R/T]** |

审计人员必须从 checkpoint 本身恢复谱系，而不是信任目录名：

- `format`；
- `step`；
- `base_checkpoint`；
- `architecture_metadata`；
- policy-guard tensor count；
- saved rollout inference steps；
- execution prefix；
- target-binding contract；
- 是否意外包含 LoRA、独立 Action Expert 或 v6 prototype bank。

## 9. 已完成实验：LF-FastWAM

### 9.1 LIBERO-Spatial

200-step 初测：

| 模型 | Correct | Null | 解释 |
|---|---:|---:|---|
| B0 | 48% | 0% | 短训练尚能部分恢复 |
| B1 | 0% | 0% | Query 路径未恢复 |
| M1 | 0% | 0% | 不能用此结果判断方法失败 |

统一训练约 1 epoch（3328 optimizer steps）后：

| 模型 | Correct | Null | LRG |
|---|---:|---:|---:|
| B0 | 98% | 0% | 98pp |
| B1 | 90% | 0% | 90pp |
| M1 | 90% | 2% | 88pp |

Spatial 十个任务都是把同一个黑碗放到盘子，只改变初始位置，因此不适合作为替代目标 CIS 主结论。

### 9.2 LIBERO-Object

B0 与 M1 都在 Object suite 独立训练约 4207 steps，Correct 一次门控均为 10/10。

| 模型 | Correct | DTL Shuffle | CIS | 备注 |
|---|---:|---:|---:|---|
| B0 Object LoRA | 100% | 80% | 初期记录 0% | DTL 仅 10 episodes |
| M1 Object LoRA | 100% | 60% | 0/50 | DTL 相对 B0 -20pp，但 CIS 无提升 |

因此 LF MVP 支持“语言冲突更容易改变行为”的初步趋势，但不支持“模型会完成替代目标”。

## 10. 已完成实验：TC-FastWAM

以下结果均为 LIBERO-Object，且不同版本的 horizon、trial 数和 checkpoint 不完全相同；不能简单串成一条单调学习曲线。

| 版本 | Correct | Shuffle/DTL | CIS | 主要观察 |
|---|---:|---:|---:|---|
| TC-C v1 Stage1 | 50%（5 trials/task） | 未形成可信结论 | 未形成可信结论 | LF retrieval 很高，但 Router patch grounding 破坏策略 |
| TC-C v2 step 2500 | Correct 原始 summary 待回收 | 44%（一次语言测试记录） | 0% | 语言干预 checkpoint，与后续 step-3500 Correct 不是同一个点 |
| TC-C v2 step 3500 | 70%（35/50）；quick gate 80%（8/10） | 未在同 checkpoint 统一复测 | 未在同 checkpoint 统一复测 | 训练更久可恢复部分策略 |
| TC-C v3 + M1 distill | 88%（44/50，horizon 600） | 未统一复测 | 未证明提升 | target grasp 常正确；长盒放置/松手需要更长 horizon |
| TC-Full v4 | 94% | 56% | 0% | LF/AF/CF 内部指标良好，但替代动作未形成 |
| TC v5 CAP | Correct 受蒸馏保护 | 未统一复测 | 用户结论为无正向提升 | 语义改变仍不足以定位/执行当前状态动作 |
| TC v6 grounding step 1500 | 80%（8/10） | 未统一复测 | 用户判断无正向提升 | 同时影响原策略，训练被放弃 |

重要经验：

- 高 `contract_retrieval_acc` 或正 `sim_LF_margin` 只证明表征可分，不等于闭环任务成功；
- 动作 expert 的小偏移可表现为“找对物体但夹不住/放不稳/未及时松手”；
- horizon 从约 430 增到 600 后，TC v3 Correct 从表面失败显著恢复，说明必须把超时和真实动作失败分开；
- 但延长 horizon 不能解释 CIS=0，也不能替代反事实动作正监督。

## 11. 已完成实验：PGC-FastWAM

### 11.1 早期失败版本

- PGC v1 action-LoRA 分支在强制候选 Correct gate 为 0%，而强制 Base 为 100%；说明 Base 保护路径有效，但 CF expert 已被训练破坏。
- PGC v2 续训到 1500 steps 后，强制 CF Correct 为 2/10；随后 10 个 CIS episode 全部 `no_object_manipulated` 并 horizon timeout。继续堆训练步数没有解决控制语义。
- PGC v3 开始移除独立 CF expert，改成冻结 Base + sidecar residual；代码层面的策略保护从此稳定。

### 11.2 v3/v4 的关键诊断

- PGC v3 的 10-episode forced-CF CIS 为 3/10；随后 exact Base 在同样 setup 下也是 3/10，说明最初看到的 30% 不是 PGC 学习增益。
- PGC v4 Correct guarded 为 100%；forced-CF CIS 仍为 30%。
- v4 的 493/493 次 forced-CF 决策确实选择了候选，但闭环 residual RMS 均值仅 `0.00840`、最大 `0.0154`；候选行为自然接近 Base。

### 11.3 50-episode Object 主对照

下表使用 10 tasks × 5 trials 的行为汇总。Base 是相同语言干预 manifest 下的 exact Base 分支，不是早期错误的“Base CIS=0”记录。

| 行为/事件 | Exact Base | PGC v5 | PGC v6 | PGC v7 |
|---|---:|---:|---:|---:|
| Counterfactual goal success（CIS） | 13/50 = 26% | 16/50 = 32% | 16/50 = 32% | 15/50 = 30% |
| Source goal success | 28% | 24% | 26% | 26% |
| Target manipulated, placement failure | 6% | 10% | 8% | 4% |
| Source manipulated, no completion | 6% | 2% | 2% | 2% |
| Other object manipulated | 18% | 26% | 20% | 22% |
| No object manipulated | 16% | 6% | 12% | 16% |
| Target object grasped | 26% | 36% | 34% | 30% |
| Target object lifted | 34% | 42% | 40% | 32% |
| Horizon timeout | 74% | 68% | 68% | 70% |

描述性差值：

| 版本 | CIS 相对 Base | 是否达到任务书 +10pp 建议门槛 |
|---|---:|---:|
| PGC v5 | +6pp | 否 |
| PGC v6 | +6pp | 否 |
| PGC v7 | +4pp | 否 |

Wilson 95% 区间（只用于显示样本不确定性，不替代配对检验）：

| 模型 | CIS | Wilson 95% CI |
|---|---:|---:|
| Exact Base | 26% | 15.9%–39.6% |
| PGC v5 | 32% | 20.8%–45.8% |
| PGC v6 | 32% | 20.8%–45.8% |
| PGC v7 | 30% | 19.1%–43.8% |

这些区间高度重叠。若 50 episodes 使用相同初始状态和 seed，应从逐 episode JSON 做 McNemar/paired bootstrap；没有对应表时不能声称显著提升。

已知服务器汇总文件：

```text
/root/gpufree-data/LF-FastWAM/evaluate_results/
  pgc_v5_object_cis_forced_cf_seed42_trials5/counterfactual_behavior_summary.json
  pgc_v6_object_cis_forced_cf_seed42_trials5/counterfactual_behavior_summary.json
  pgc_v7_object_cis_forced_cf_seed42_trials5/counterfactual_behavior_summary.json
```

Exact Base 的 50-episode 汇总内容已被记录，但当前交接信息中没有确认其目录名；审计人员应依据 result JSON 中 checkpoint/gate metadata 搜索，不能把任意 26% 文件直接指定为 Base。v7 Correct 在实验过程中被报告为“通过”，但当前交接记录没有粘贴其完整 summary；也应从服务器原始结果补齐 trial 数、override rate 和 horizon。

### 11.4 v7 per-task 行为

| Source task | CF success | Source success | Target placement failure | Other object | No object |
|---:|---:|---:|---:|---:|---:|
| 0 | 0/5 | 4/5 | 0/5 | 0/5 | 0/5 |
| 1 | 5/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| 2 | 0/5 | 0/5 | 0/5 | 5/5 | 0/5 |
| 3 | 0/5 | 1/5 | 0/5 | 4/5 | 0/5 |
| 4 | 0/5 | 1/5 | 0/5 | 2/5 | 2/5 |
| 5 | 2/5 | 0/5 | 1/5 | 0/5 | 2/5 |
| 6 | 2/5 | 3/5 | 0/5 | 0/5 | 0/5 |
| 7 | 1/5 | 2/5 | 0/5 | 0/5 | 2/5 |
| 8 | 1/5 | 1/5 | 1/5 | 0/5 | 2/5 |
| 9 | 4/5 | 1/5 | 0/5 | 0/5 | 0/5 |

task 1/9 较好，task 2/3 完全转向其他物体，task 0 多数保持源目标。这说明故障不是单一“不会动”，而是 pair-specific 的目标绑定和动作泛化混合问题。

### 11.5 v7 binder 与 residual 诊断

50-episode forced-CF rollout 聚合：

```text
decisions                         = 2426
counterfactual overrides          = 2426 (forced mode, 100%)
candidate_delta_rms_mean          = 0.0054936661
candidate_delta_rms_max           = 0.1152163893
candidate_saturation_mean         = 0
target_binding_top1_mass_mean     = 0.4323893818
target_binding_entropy_mean       = 0.4100030254
target_binding_similarity_mean    = 0.4206851417
```

Binder 指标没有与成功率形成简单单调关系。例如 task 2 的 top-1 mass 约 0.665，却是 0/5 CF success 且 5/5 操作其他物体；task 1 的 top-1 mass 约 0.494，却是 5/5 成功。因此当前不能把较集中的 attention 当作目标绑定正确的充分证据。

## 12. v7 训练—部署残差审计

审计入口：

- [`scripts/audit_pgc_v7_residual_gap.sh`](../scripts/audit_pgc_v7_residual_gap.sh)
- [`scripts/audit_pgc_residual_gap.py`](../scripts/audit_pgc_residual_gap.py)

### 12.1 Checkpoint 恢复

```text
checkpoint_tensors = 172
loaded_tensors     = 172
missing            = []
unexpected         = []
shape_mismatches   = {}
unequal_tensors    = 0
max_abs_error      = 0
exact_match        = true
```

结论：不是 sidecar 漏载、错 shape 或 checkpoint 损坏。**[D]**

### 12.2 文本路径

```text
cached/live text context RMS   = 0.00097364
cached/live cosine             = 0.99955921
mask_equal_rate                = 1.0
```

结论：不是训练 cache 与部署在线 T5 编码的主要不一致。**[D]**

### 12.3 示教状态与闭环状态

| 测量 | n | mean | median | min | max |
|---|---:|---:|---:|---:|---:|
| Cached training-state residual RMS | 20 | 0.72218 | 0.74644 | 0.34810 | 0.75612 |
| Live-text training-state residual RMS | 20 | 0.72219 | 0.74630 | 0.34857 | 0.75612 |
| Closed-loop residual RMS | 2426 | 0.005494 | 0.005068 | 0.002123 | 0.11522 |

```text
live / cached ratio       = 1.0000089
closed-loop / live ratio  = 0.00760696
```

即使取闭环最大值 `0.115`，也低于 20 个示教样本的最小值 `0.349`。这不是小幅漂移，而是 Proposal 在部署状态上基本退化为近 Base。

### 12.4 离线动作改善

在 20 个示教状态上，live candidate 相对 Base 的 target-prefix MSE improvement：

```text
mean   = +0.62780
median = +0.57122
p05    = -0.78252
min    = -1.69283
max    = +1.71798
```

均值为正说明 Proposal 并非完全没学到；但负尾很重，说明即使在示教分布也不是每条候选都优于 Base。Verifier 的真实任务是识别这些负尾，而不是只看平均 improvement。

### 12.5 当前诊断

审计程序给出的分类是：

```text
closed_loop_state_distribution_shift
```

其证据链为：

1. checkpoint 精确恢复；
2. cached/live 文本一致；
3. 同一部署 `infer_action` 路径在示教状态产生大 residual；
4. 相同 checkpoint 在 LIBERO 闭环状态产生近零 residual；
5. 因此首个明确失败边界是 state distribution，而非保存/加载或文本路径。

## 13. 当前故障假设与替代解释

### 13.1 主假设

**H1：native residual-zero 监督 + 极少 CF 正轨迹，使 Proposal 学成示教流形检测器。**

- Native 数据覆盖广，要求 residual=0；
- CF 只有 50 条成功 episode，状态分布窄；
- 训练状态 residual 大，闭环状态 residual 小；
- 这与模型只在熟悉 CF 示教帧上激活高度一致。

### 13.2 仍需排除

- **H2：观测预处理或时间对齐差异。** 审计已排除文本，但还应逐项比较训练与 rollout 的图像裁剪、双相机顺序、proprio 时刻、action normalization 和 replan index。
- **H3：初始状态域偏差。** CF 数据状态来自 donor HDF5 的恢复/迁移，不一定覆盖官方 evaluation init-state catalog。
- **H4：Binder 选择了语义正确但控制不适用的 patch。** 当前只有 mask overlap/attention 指标，没有 rollout 中真实目标 pose-conditioned control 证据。
- **H5：Residual cap/architecture 过小。** 当前没有饱和，且示教 residual 可达 0.72；因此“cap 太小”不符合已有证据，但闭环的 representation scale 可能仍抑制输出。
- **H6：Verifier gate 错误。** forced-CF 2,426/2,426 overrides 仍只有 30% CIS，说明 v7 的主要问题先于 gate；gate 仍需单独校准，但不是这次 30% 的主因。

## 14. 结果解释中的已知混淆因素

1. **早期 Base CIS=0 是不可靠基线。** 在后续相同 manifest、50 episodes 的 exact Base 测试中 CIS 为 26%；旧 0% 可能来自小样本、不同配对、条件/分支选择或评估 bug，不能继续引用为当前 Base。
2. **单次 10-task gate 不等于稳定成功率。** 10/10 的 Wilson 下界约 72.2%；必须保留 5 trials/task 或多 seed。
3. **horizon 会显著改变结果。** 长条盒抓起后可能在 430 steps 前未松手，600-step 评估能恢复部分成功；每张表必须标注 `max_policy_steps`。
4. **forced-CF 与 guarded 不同。** forced-CF 用于测 Proposal，guarded 用于测部署组合；不能混表。
5. **Correct success 可能全部来自 Base fallback。** 必须同时报告 decision count、override count/rate 和强制候选 Correct。
6. **pair mapping 不平衡。** 当前 10 个 Source 有覆盖，但 target 频次不均。
7. **只有 seed 42 主线。** 当前没有充分多 seed 训练或环境复测。
8. **训练日志不是行为证据。** retrieval、mask mass、attention entropy、action MSE 都不能替代 Correct/DTL/CIS rollout。
9. **视频文件名可能仍显示 Source 描述。** 行为分析以结果 JSON 中的 `policy_instruction`、`pair_id` 和 counterfactual predicate 为准。
10. **结果多在服务器，未进入 Git。** 外部审计必须获取原始 JSON、CSV、MP4 和 log，而不仅是本文转录。

## 15. 当前完成度矩阵

| 项目 | 状态 | 证据/缺口 |
|---|---|---|
| LF MVP 结构、mask、Prior/Posterior | 完成 | 单元测试与训练 smoke 已通过 |
| B0/B1/M1 controlled FT | 完成 | Spatial；Object 主对照 B0/M1 |
| Correct/Null/Shuffled/CIS runner | 完成 | LIBERO Object 已多轮运行 |
| Counterfactual behavior taxonomy | 完成 | 抓取/抬升/源目标/其他物体/不动作/超时 |
| TC LF/AF/CF/CAP/grounding | 完成实验性实现 | 未获得稳定 CIS |
| PGC frozen Base + hard fallback | 完成 | v3+ 结构与测试确认 |
| Direct CF action dataset builder | 完成 | Object 50 条成功数据 |
| PGC v7 explicit mask binder | 完成 | Object 构建/训练/smoke/评估 |
| Training–deployment residual audit | 完成 | 定位 closed-loop state shift |
| PGC v8 | **未实现** | 当前代码 version 白名单只到 7 |
| LIBERO-Spatial PGC | 未完成 | 需独立 CF 数据、mask、训练、Correct/CIS |
| LIBERO-Goal PGC | 未完成 | 同上 |
| LIBERO-10 PGC | 未完成 | 同上 |
| 四 suite 联合训练 | 锁定/未批准 | launcher 默认拒绝 `all` |
| RoboTwin PGC | 未开始 | 需动作/相机/成功谓词/CF 数据 contract |
| 统计显著性 | 未满足 | 单 seed，50 episodes，缺 paired test |

## 16. V8 建议边界（尚未实现）

V8 的目标应是解决已经量化的闭环 residual collapse，而不是再加一组离线 representation loss。建议最小设计：

1. 保留 v7 的冻结 release Base、final-action Proposal、FP32 verifier、hard fallback 和显式 mask binder；
2. 收集官方 evaluation init states 和 on-policy rollout 中 Proposal residual 已塌缩的状态；
3. 在这些状态上获得可审计的替代目标成功动作，形成 DAgger-style corrective CF 数据；
4. Proposal 的 native 全域 zero 监督降权、分阶段或移出 Proposal；Native 主要用于 Verifier/Base-protection，保留同状态 Source-language zero 约束；
5. 训练 batch 明确包含 demonstration CF、closed-loop corrective CF、same-state Source 和 native guard 四类，并分别记录 residual；
6. checkpoint 选择增加部署态 gate：固定 held-out closed-loop states 上 residual RMS、target-action improvement 和 Correct protection 必须同时通过；
7. 先在 Object 单 suite 做 10-episode smoke、50-episode paired evaluation，再进入 Spatial/Goal/10；
8. 不在四个 suite 各自验收前解锁联合训练。

V8 必须先写数据 contract 和失败状态采样协议，再改模型。否则仍可能在相同 50 条离线 episode 上过拟合。

## 17. 外部审计所需原始材料

审计交付包至少应包含：

### 17.1 代码

- `git bundle` 或 commit `e6a0891` 的完整 clone；
- `git status --short`；
- `git submodule status`（如适用）；
- Python、CUDA、driver、torch、DeepSpeed、MuJoCo、robosuite、LIBERO 版本；
- 所有 Hydra resolved config 和启动环境变量。

### 17.2 Checkpoint

- release Base；
- B0/M1；
- TC v3/v4/v6 关键节点；
- PGC v3/v4/v5/v6/v7 final weights；
- 每个 checkpoint 的 SHA-256、metadata、tensor key/shape/dtype；
- 与外部 Base 的引用关系；
- 任何 optimizer/state checkpoint 的存在性和完整性。

### 17.3 数据

- Object native LeRobot dataset；
- 官方 HDF5 donor 路径和哈希；
- `libero_object_pgc_counterfactual_lerobot`；
- provenance、episode audit、initial state catalog；
- v7 mask index、每个 sidecar SHA-256；
- manifest 及每个 pair 的目标谓词；
- 被拒绝 demo、pair 替换历史和最终配对理由。

### 17.4 实验

- Base/v5/v6/v7 的 50-episode原始 task result JSON；
- behavior summary JSON/CSV；
- Correct/Null/Shuffled/CIS summary；
- 每个 episode MP4；
- policy-guard per-replan decisions；
- v7 training–deployment residual audit 完整 JSON（含逐样本记录，而非只给 summary）；
- 完整训练 log 和 GPU/墙钟信息。

## 18. 服务器复核命令

### 18.1 代码与环境

```bash
cd /root/gpufree-data/LF-FastWAM
git rev-parse HEAD
git status --short
/opt/conda/bin/python - <<'PY'
import sys, torch, numpy
print("python", sys.version)
print("torch", torch.__version__)
print("cuda", torch.version.cuda, torch.cuda.is_available())
print("numpy", numpy.__version__)
PY
bash scripts/validate_pgc_server.sh
```

### 18.2 资产哈希

```bash
BASE=/root/gpufree-data/fastwam/FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224.pt
V7=/root/gpufree-data/LF-FastWAM/runs/libero_pgc_2cam224/pgc-object-pgc-v7-mask-grounded-4000-seed42-v1/checkpoints/weights/step_004000.pt
sha256sum "$BASE" "$V7"
stat -c '%n %s %y' "$BASE" "$V7"
```

### 18.3 Checkpoint contract

只对本项目可信 checkpoint 使用 `weights_only=False`：

```bash
/opt/conda/bin/python - "$V7" <<'PY'
import json, sys, torch
p = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print("format:", p.get("format"))
print("step:", p.get("step"))
print("base:", p.get("base_checkpoint"))
print("guard_tensors:", len(p.get("policy_guard", {})))
print(json.dumps(p.get("architecture_metadata"), indent=2, ensure_ascii=False))
PY
```

### 18.4 数据 contract

```bash
PGC_DATA=/root/gpufree-data/pgc_libero_data_v1/libero_object_pgc_counterfactual_lerobot
/opt/conda/bin/python scripts/validate_pgc_counterfactual_datasets.py "$PGC_DATA"
/opt/conda/bin/python - "$PGC_DATA" <<'PY'
import sys
from fastwam.datasets.pgc_libero import load_pgc_target_mask_index
x = load_pgc_target_mask_index(sys.argv[1])
print("mask episodes:", len(x["episodes_by_index"]))
PY
```

### 18.5 三模式复核

```bash
export PGC_CHECKPOINT="$V7"
export PGC_EVAL_SUITES='[libero_object]'
export PGC_MANIFEST_PATH=/root/gpufree-data/pgc_libero_data_v1/manifests/libero_object_pgc.jsonl
export PGC_MAX_POLICY_STEPS=600

# 1. Exact Base counterfactual baseline
PGC_GATE_MODE=base \
OUTPUT_ROOT=/root/gpufree-data/LF-FastWAM/evaluate_results/audit_v7_base_cis \
bash scripts/eval_pgc_libero.sh 4 5 counterfactual 42 10

# 2. Forced Proposal
PGC_GATE_MODE=counterfactual \
OUTPUT_ROOT=/root/gpufree-data/LF-FastWAM/evaluate_results/audit_v7_forced_cf \
bash scripts/eval_pgc_libero.sh 4 5 counterfactual 42 10

# 3. Guarded Correct
PGC_GATE_MODE=guarded \
OUTPUT_ROOT=/root/gpufree-data/LF-FastWAM/evaluate_results/audit_v7_guarded_correct \
bash scripts/eval_pgc_libero.sh 4 5 correct 42 10
```

### 18.6 行为汇总与残差审计

```bash
FORCED=/root/gpufree-data/LF-FastWAM/evaluate_results/audit_v7_forced_cf
/opt/conda/bin/python scripts/summarize_counterfactual_behaviors.py \
  "$FORCED" \
  --expected-episodes 50 \
  --output-prefix "$FORCED/counterfactual_behavior_summary"

bash scripts/audit_pgc_v7_residual_gap.sh \
  "$V7" \
  "$FORCED" \
  20 \
  "$FORCED/training_deployment_residual_audit.json"
```

## 19. 代码审计入口

| 主题 | 文件 |
|---|---|
| FastWAM 主模型、三条方法线、checkpoint | [`src/fastwam/models/wan22/fastwam.py`](../src/fastwam/models/wan22/fastwam.py) |
| Action Query | [`src/fastwam/models/wan22/action_dit.py`](../src/fastwam/models/wan22/action_dit.py) |
| PGC Goal/Binder/Proposal/Verifier | [`src/fastwam/models/wan22/policy_guard.py`](../src/fastwam/models/wan22/policy_guard.py) |
| TC Router/Contract/Action Effect | [`src/fastwam/models/wan22/transition_contract.py`](../src/fastwam/models/wan22/transition_contract.py) |
| Trainer trainability/save schedule | [`src/fastwam/trainer.py`](../src/fastwam/trainer.py) |
| PGC dataset contract | [`src/fastwam/datasets/pgc_libero.py`](../src/fastwam/datasets/pgc_libero.py) |
| PGC task config | [`configs/task/libero_pgc_2cam224.yaml`](../configs/task/libero_pgc_2cam224.yaml) |
| PGC common launcher | [`scripts/train_pgc_libero.sh`](../scripts/train_pgc_libero.sh) |
| v7 launcher | [`scripts/train_pgc_v7_libero_suite.sh`](../scripts/train_pgc_v7_libero_suite.sh) |
| PGC eval | [`scripts/eval_pgc_libero.sh`](../scripts/eval_pgc_libero.sh) |
| CF dataset builder | [`scripts/build_pgc_libero_data.py`](../scripts/build_pgc_libero_data.py) |
| v7 mask builder | [`scripts/build_pgc_libero_target_masks.py`](../scripts/build_pgc_libero_target_masks.py) |
| behavior summary | [`scripts/summarize_counterfactual_behaviors.py`](../scripts/summarize_counterfactual_behaviors.py) |
| residual audit | [`scripts/audit_pgc_residual_gap.py`](../scripts/audit_pgc_residual_gap.py) |
| PGC tests | [`tests/test_policy_guard.py`](../tests/test_policy_guard.py) |
| data tests | [`tests/test_pgc_data_contract.py`](../tests/test_pgc_data_contract.py), [`tests/test_pgc_libero_data_builder.py`](../tests/test_pgc_libero_data_builder.py) |
| audit tests | [`tests/test_pgc_residual_audit.py`](../tests/test_pgc_residual_audit.py) |

## 20. 关键提交谱系

| Commit | 作用 |
|---|---|
| `e853e09` | LF-FastWAM MVP |
| `f9e332e` | 四卡 LoRA 训练路径 |
| `e99a966` | paired DTL/CIS eval |
| `2415607` | Object-only B0/M1 |
| `89bbe53` | TC Stage1 contract |
| `0dc749c` | M1 policy distillation |
| `6f95c11` | TC-Full |
| `762cb14` | Counterfactual action positive |
| `8edc019` | State-conditioned grounding |
| `d374fb0` | 初版 PGC |
| `392f8c8` | 可审计 PGC LIBERO datasets |
| `6fc1a49` | PGC action-LoRA |
| `1416af6` | PGC v2 redesign |
| `fbe3165` | PGC v3 protected velocity residual |
| `141ae3f` | PGC v4 rollout-aligned proposal |
| `aa3413d` | PGC v5 paired-language training |
| `6f8df5f` | PGC v6 visual target binding |
| `5522816` | PGC v7 spatial target binding |
| `e6a0891` | training–deployment residual audit |

## 21. 建议审计顺序

1. 固定代码 commit、环境和所有资产哈希；
2. 从 checkpoint metadata 证明 Base/sidecar 谱系，复核 172/172 exact load；
3. 验证 Object CF 数据 provenance、状态 SHA、成功谓词和 v7 mask sidecars；
4. 在相同 50 个初始状态上复跑 Base、forced-CF、guarded 三模式；
5. 逐 episode 配对比较 Base/v5/v6/v7，而不是只比较总比例；
6. 复现训练态/闭环态 residual gap，并按 replan time、task、camera、target、成功/失败分层；
7. 检查 native zero 与 CF positive 的 batch 比例、梯度量级和状态覆盖；
8. 先设计/审核 V8 closed-loop data contract，再允许实现；
9. Object 通过后依次完成 Spatial、Goal、LIBERO-10 suite-specific 数据和实验；
10. 四 suite 都通过后才考虑联合训练；最后单独为 RoboTwin 定义成功谓词和反事实采集协议。

## 22. 最终阶段性表述

建议对外使用以下表述：

> 本项目在 FastWAM 上实现了三代语言遵守增强路线：LF-MVP 通过 latent query 与无语言 prior 提高语言依赖，TC 通过 Language/Action/Future contracts 学习意图—结果表征，PGC 则冻结原始策略并用目标绑定、最终动作候选和保守门控探索反事实执行。当前 PGC 能结构性保留 exact Base，并建立了直接反事实动作、显式目标 mask 和行为级 CIS 审计链路；但在 LIBERO-Object 的 50-episode 对照中，PGC v5/v6/v7 的 CIS 仅比 exact Base 高 6/6/4 个百分点，尚不足以证明稳定提升。v7 审计进一步定位到 Proposal 在示教状态有效、在闭环状态 residual 缩小约 131 倍，因此下一阶段应优先补充闭环 corrective supervision，而不是继续增加离线表征损失。其余 LIBERO suites 与 RoboTwin 仍待逐套件验证。
