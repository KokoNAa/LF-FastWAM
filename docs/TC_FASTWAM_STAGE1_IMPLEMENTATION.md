# TC-FastWAM Stage 1（TC-C）实现说明

本实现按任务书跳过 Stage 0，直接进入 Stage 1。当前主方法是 **TC-C**：在保留 FastWAM 原始 Video Expert 与 future-video flow matching（Phase T1）的基础上，新增 intended transition、realized transition、Transition Router 和 Language–Future contrastive contract。

## 模型路径

训练与推理共享的动作路径为：

```text
Language + proprio + learnable transition queries
                         ↓
              intended transition queries
                         ↓
Current-frame Video tokens → Transition Visual Router
                         ↓
                  routed prefix H_route
                         ↓
                Action Expert [H_route, A_t]
```

Action token 和 routed query 在 MoT 中都不能直接读取 raw video；Action Expert 的 cross-attention 也只保留 proprio，不直接读取 T5。语言与视觉只通过 Router 汇入动作前缀。

训练期另以 clean GT current/future latents 经 Video Expert patch embedding 后的 hidden 均值差构造 `z_F`，以 routed intent 的池化表示构造 `z_L`。`z_F` 默认在进入 Outcome projection 前 stop-gradient，避免 contract 反向扰动 Video Expert；这只增加一次 patch embedding，不会复制 30 层 Video Transformer。两者归一化后使用对称 InfoNCE。跨卡训练会收集全局 in-batch negatives；单卡 batch size 1 返回有限的零 contract loss，不会崩溃。

Stage 1 明确不包含：`z_A`、Action–Future contract、counterfactual ranking、action-conditioned video、移除 Video Expert 的直接语言路径。这些分别属于 Stage 2/3。

## 训练与兼容性

- 默认配置仍关闭 Transition Contract，原 B0/M1 行为不变。
- TC-C 要求 prior/advantage 关闭。
- 保留 checkpoint key `latent_action_queries`，仅新增语义 alias `transition_queries`。
- TC LoRA checkpoint 保存 router、projection、outcome encoder 与架构 metadata。
- 旧 FastWAM/B0/M1 checkpoint 加载到 TC-C 时，新模块采用标准初始化。
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
  /root/gpufree-data/fastwam/FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224.pt \
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
transition_contract_scale
```

Stage 1 的正式 gate 仍需服务器实验确认：Correct SR 不低于 B0 5pp、LF retrieval 显著高于随机、hard-negative future-similarity margin 大于 0、query 不坍缩、推理 p50 不超过 B0 的 1.10 倍。
