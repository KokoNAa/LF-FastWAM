"""Stage-1 latent transition contract modules for TC-FastWAM.

The deployment path contains only the intent/router branch.  The outcome
encoder and contrastive loss are training-time teachers built from the clean
current/future video tokens.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def _masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    num_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Multi-head attention with an all-masked-row-safe boolean mask."""
    batch, query_len, hidden_dim = query.shape
    key_len = key.shape[1]
    if hidden_dim % num_heads != 0:
        raise ValueError(
            f"Attention hidden_dim={hidden_dim} must divide num_heads={num_heads}."
        )
    head_dim = hidden_dim // num_heads

    def _heads(x: torch.Tensor) -> torch.Tensor:
        return x.view(batch, -1, num_heads, head_dim).transpose(1, 2)

    q = _heads(query)
    k = _heads(key)
    v = _heads(value)
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) / math.sqrt(
        head_dim
    )
    if mask is None:
        valid = torch.ones(
            (batch, 1, 1, key_len), dtype=torch.bool, device=scores.device
        )
    else:
        if mask.shape != (batch, key_len):
            raise ValueError(
                f"Attention mask must be {(batch, key_len)}, got {tuple(mask.shape)}."
            )
        valid = mask.to(device=scores.device, dtype=torch.bool)[:, None, None, :]
    scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=-1)
    weights = weights * valid.to(weights.dtype)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    output = torch.matmul(weights.to(v.dtype), v)
    output = output.transpose(1, 2).reshape(batch, query_len, hidden_dim)
    return output, weights


class TransitionProjectionHead(nn.Module):
    """Small normalized projection used for transition-space embeddings."""

    def __init__(self, input_dim: int, projection_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.projection = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(projection_dim, projection_dim),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        projected = self.projection(self.norm(hidden))
        return F.normalize(projected.float(), dim=-1, eps=1e-6)


class TransitionVisualRouter(nn.Module):
    """Language-condition transition queries, then route current visual tokens."""

    def __init__(self, action_dim: int, video_dim: int, num_heads: int = 8):
        super().__init__()
        if action_dim % num_heads != 0:
            raise ValueError(
                f"action_dim={action_dim} must divide router num_heads={num_heads}."
            )
        self.action_dim = int(action_dim)
        self.video_dim = int(video_dim)
        self.num_heads = int(num_heads)

        self.language_q = nn.Linear(action_dim, action_dim)
        self.language_k = nn.Linear(action_dim, action_dim)
        self.language_v = nn.Linear(action_dim, action_dim)
        self.language_out = nn.Linear(action_dim, action_dim)

        self.visual_q = nn.Linear(action_dim, action_dim)
        self.visual_k = nn.Linear(video_dim, action_dim)
        self.visual_v = nn.Linear(video_dim, action_dim)
        self.visual_out = nn.Linear(action_dim, action_dim)

        self.language_norm = nn.LayerNorm(action_dim)
        self.visual_norm = nn.LayerNorm(action_dim)
        self.output_norm = nn.LayerNorm(action_dim)
        self.output_mlp = nn.Sequential(
            nn.Linear(action_dim, action_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(action_dim, action_dim),
        )
        # The external policy-output blend starts from the exact M1 posterior
        # policy, while these non-zero residuals can learn from the Contract
        # once its configured warm-up begins.

    @staticmethod
    def _diagnostics(
        routed: torch.Tensor, visual_weights: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        probs = visual_weights.float().clamp_min(1e-8)
        entropy = -(probs * probs.log()).sum(dim=-1)
        entropy = entropy / math.log(max(2, visual_weights.shape[-1]))
        top1 = probs.topk(k=1, dim=-1).values.sum(dim=-1)
        topk = probs.topk(k=min(5, probs.shape[-1]), dim=-1).values.sum(dim=-1)

        normalized_queries = F.normalize(routed.float(), dim=-1, eps=1e-6)
        similarity = torch.matmul(
            normalized_queries, normalized_queries.transpose(-1, -2)
        )
        query_count = similarity.shape[-1]
        if query_count > 1:
            off_diagonal = ~torch.eye(
                query_count, dtype=torch.bool, device=similarity.device
            )
            pairwise = similarity[:, off_diagonal].mean()
            pairwise_max = similarity[:, off_diagonal].max()
        else:
            pairwise = similarity.new_zeros(())
            pairwise_max = similarity.new_zeros(())
        return {
            "transition_query_norm": routed.float().norm(dim=-1).mean(),
            "transition_query_pairwise_cosine": pairwise,
            "transition_query_pairwise_cosine_max": pairwise_max,
            "router_attention_entropy": entropy.mean(),
            "router_top1_mass": top1.mean(),
            "router_top5_mass": topk.mean(),
        }

    def forward(
        self,
        transition_queries: torch.Tensor,
        language_hidden: torch.Tensor,
        language_mask: torch.Tensor,
        current_video_hidden: torch.Tensor,
        route_scale: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if transition_queries.ndim != 3:
            raise ValueError("`transition_queries` must be [B,K,D_action].")
        if language_hidden.ndim != 3 or current_video_hidden.ndim != 3:
            raise ValueError("Router language/video inputs must be 3D tensors.")
        if transition_queries.shape[0] != language_hidden.shape[0] or (
            transition_queries.shape[0] != current_video_hidden.shape[0]
        ):
            raise ValueError("Router inputs must have the same batch dimension.")

        language_query = self.language_q(self.language_norm(transition_queries))
        language_key = self.language_k(language_hidden)
        language_value = self.language_v(language_hidden)
        language_delta, _ = _masked_attention(
            language_query,
            language_key,
            language_value,
            language_mask,
            num_heads=self.num_heads,
        )
        route_scale = float(route_scale)
        if not 0.0 <= route_scale <= 1.0:
            raise ValueError(f"`route_scale` must be in [0,1], got {route_scale}.")
        language_residual = self.language_out(language_delta)
        intended_queries = transition_queries + route_scale * language_residual

        visual_query = self.visual_q(self.visual_norm(intended_queries))
        visual_key = self.visual_k(current_video_hidden)
        visual_value = self.visual_v(current_video_hidden)
        visual_delta, visual_weights = _masked_attention(
            visual_query,
            visual_key,
            visual_value,
            mask=None,
            num_heads=self.num_heads,
        )
        visual_residual = self.visual_out(visual_delta)
        route_residual = visual_residual + self.output_mlp(
            self.output_norm(intended_queries + visual_residual)
        )
        routed = intended_queries + route_scale * route_residual
        diagnostics = self._diagnostics(routed, visual_weights)
        diagnostics.update(
            {
                "router_route_scale": routed.new_tensor(route_scale),
                "router_language_residual_norm": language_residual.float()
                .norm(dim=-1)
                .mean(),
                "router_visual_residual_norm": route_residual.float()
                .norm(dim=-1)
                .mean(),
            }
        )
        return routed, diagnostics


class OutcomeTransitionEncoder(nn.Module):
    """Encode realized future change from clean Video-Expert patch tokens."""

    def __init__(self, video_dim: int, projection_dim: int):
        super().__init__()
        self.projection = TransitionProjectionHead(video_dim, projection_dim)

    def forward(
        self, video_hidden: torch.Tensor, *, tokens_per_frame: int
    ) -> torch.Tensor:
        tokens_per_frame = int(tokens_per_frame)
        if video_hidden.ndim != 3:
            raise ValueError("`video_hidden` must be [B,N,D_video].")
        if tokens_per_frame <= 0 or video_hidden.shape[1] <= tokens_per_frame:
            raise ValueError(
                "Outcome encoding requires current and future video tokens; "
                f"tokens={video_hidden.shape[1]}, tokens_per_frame={tokens_per_frame}."
            )
        return self.from_hidden_pair(
            video_hidden[:, :tokens_per_frame],
            video_hidden[:, tokens_per_frame:],
        )

    def from_hidden_pair(
        self, current_hidden: torch.Tensor, future_hidden: torch.Tensor
    ) -> torch.Tensor:
        if current_hidden.ndim != 3 or future_hidden.ndim != 3:
            raise ValueError("Current/future transition hidden must be 3D.")
        if current_hidden.shape[0] != future_hidden.shape[0] or (
            current_hidden.shape[2] != future_hidden.shape[2]
        ):
            raise ValueError("Current/future hidden batch and feature dims must match.")
        current = current_hidden.mean(dim=1)
        future = future_hidden.mean(dim=1)
        return self.projection(future - current)


def _gather_with_grad(tensor: torch.Tensor) -> tuple[torch.Tensor, int]:
    if not dist.is_available() or not dist.is_initialized():
        return tensor, 0
    try:
        from torch.distributed.nn.functional import all_gather

        local_size = torch.tensor(
            [tensor.shape[0]], dtype=torch.long, device=tensor.device
        )
        gathered_sizes = [
            torch.zeros_like(local_size) for _ in range(dist.get_world_size())
        ]
        dist.all_gather(gathered_sizes, local_size)
        sizes = [int(value.item()) for value in gathered_sizes]
        if len(set(sizes)) != 1:
            raise ValueError(
                "Transition Contract distributed InfoNCE requires equal local "
                f"batch sizes, got {sizes}. Set the training DataLoader to "
                "`drop_last=True`."
            )
        gathered = all_gather(tensor)
        return torch.cat(tuple(gathered), dim=0), dist.get_rank()
    except ImportError:
        gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, tensor.detach())
        gathered[dist.get_rank()] = tensor
        return torch.cat(gathered, dim=0), dist.get_rank()


class ContrastiveContractLoss(nn.Module):
    """Symmetric Language/Future InfoNCE with distributed in-batch negatives."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        if temperature <= 0:
            raise ValueError("`temperature` must be positive.")
        self.temperature = float(temperature)

    def forward(
        self, z_language: torch.Tensor, z_future: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if z_language.shape != z_future.shape or z_language.ndim != 2:
            raise ValueError(
                "Transition embeddings must have equal [B,D] shapes, got "
                f"{tuple(z_language.shape)} and {tuple(z_future.shape)}."
            )
        z_language = F.normalize(z_language.float(), dim=-1, eps=1e-6)
        z_future = F.normalize(z_future.float(), dim=-1, eps=1e-6)
        all_language, rank = _gather_with_grad(z_language)
        all_future, future_rank = _gather_with_grad(z_future)
        if rank != future_rank:
            raise RuntimeError("Distributed transition gather rank mismatch.")

        batch_size = z_language.shape[0]
        candidate_count = all_language.shape[0]
        labels = rank * batch_size + torch.arange(
            batch_size, device=z_language.device
        )
        logits_lf = torch.matmul(z_language, all_future.transpose(0, 1))
        logits_fl = torch.matmul(z_future, all_language.transpose(0, 1))
        if candidate_count > 1:
            loss = 0.5 * (
                F.cross_entropy(logits_lf / self.temperature, labels)
                + F.cross_entropy(logits_fl / self.temperature, labels)
            )
            negative_mask = torch.ones_like(logits_lf, dtype=torch.bool)
            negative_mask.scatter_(1, labels[:, None], False)
            negative = logits_lf.masked_fill(~negative_mask, -1.0).max(dim=1).values
        else:
            # Keep a differentiable scalar for single-card unit/smoke batches.
            loss = (z_language.sum() + z_future.sum()) * 0.0
            negative = logits_lf.new_zeros((batch_size,))

        positive = logits_lf.gather(1, labels[:, None]).squeeze(1)
        metrics = {
            "sim_LF_positive": positive.mean(),
            "sim_LF_negative": negative.mean(),
            "sim_LF_margin": (positive - negative).mean(),
            "contract_retrieval_acc": (logits_lf.argmax(dim=1) == labels)
            .float()
            .mean(),
            "contract_candidate_count": logits_lf.new_tensor(
                float(candidate_count)
            ),
        }
        return loss, metrics


def detached_metrics(metrics: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Convert scalar tensor metrics to logging-safe Python values."""
    return {
        key: float(value.detach().float().cpu().item())
        for key, value in metrics.items()
    }
