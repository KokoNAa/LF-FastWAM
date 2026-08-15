"""Policy-Guarded Counterfactual (PGC) modules for FastWAM.

PGC deliberately keeps the released FastWAM policy outside the optimization
graph. Historical v1/v2 checkpoints use a separate goal-conditioned Action
Expert. v3 instead evaluates the same frozen Base Expert and learns only a
bounded goal-conditioned correction in flow-velocity space. An
:class:`ActionOutcomeVerifier` decides whether that proposal is sufficiently
more compatible with the requested goal to override the exact Base candidate.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transition_contract import _gather_with_grad, _gather_without_grad


def _safe_key_padding_mask(
    values: torch.Tensor, valid_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return MHA-safe values and ``key_padding_mask``.

    PyTorch attention can produce NaNs when every key in a row is masked.  PGC
    treats such a row as a zero-valued sentinel token instead.  This is needed
    for the explicit null-language evaluation condition.
    """
    if values.ndim != 3 or valid_mask.ndim != 2:
        raise ValueError("Attention values/mask must be [B,S,D] and [B,S].")
    if values.shape[:2] != valid_mask.shape:
        raise ValueError(
            "Attention value/mask shape mismatch: "
            f"{tuple(values.shape)} vs {tuple(valid_mask.shape)}."
        )
    if values.shape[1] <= 0:
        raise ValueError("Attention requires at least one key token.")
    valid_mask = valid_mask.to(device=values.device, dtype=torch.bool)
    safe_values = values * valid_mask.unsqueeze(-1).to(values.dtype)
    safe_valid = valid_mask.clone()
    empty = ~safe_valid.any(dim=1)
    if bool(empty.any()):
        safe_valid[empty, 0] = True
        safe_values = safe_values.clone()
        safe_values[empty, 0] = 0
    return safe_values, ~safe_valid


def _query_diagnostics(
    queries: torch.Tensor, visual_attention: torch.Tensor
) -> dict[str, torch.Tensor]:
    normalized = F.normalize(queries.float(), dim=-1, eps=1.0e-6)
    similarity = torch.matmul(normalized, normalized.transpose(-1, -2))
    query_count = int(similarity.shape[-1])
    if query_count > 1:
        off_diagonal = ~torch.eye(
            query_count, dtype=torch.bool, device=similarity.device
        )
        pairwise = similarity[:, off_diagonal].mean()
        pairwise_max = similarity[:, off_diagonal].max()
    else:
        pairwise = similarity.new_zeros(())
        pairwise_max = similarity.new_zeros(())

    probabilities = visual_attention.float().clamp_min(1.0e-8)
    entropy = -(probabilities * probabilities.log()).sum(dim=-1)
    entropy = entropy / math.log(max(2, probabilities.shape[-1]))
    return {
        "pgc_query_norm": queries.float().norm(dim=-1).mean(),
        "pgc_query_pairwise_cosine": pairwise,
        "pgc_query_pairwise_cosine_max": pairwise_max,
        "pgc_visual_attention_entropy": entropy.mean(),
        "pgc_visual_top1_mass": probabilities.max(dim=-1).values.mean(),
    }


class GoalGraphEncoder(nn.Module):
    """Build state-grounded goal tokens and route counterfactual queries.

    The learned goal slots first read the requested language and then the
    *current* visual tokens.  The result is a compact goal graph: several slots
    can specialize to target, relation, destination, and execution context
    without relying on suite-specific task IDs or task-average prototypes.
    """

    def __init__(
        self,
        *,
        text_dim: int,
        video_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        projection_dim: int = 256,
        num_goal_tokens: int = 4,
        num_heads: int = 8,
    ):
        super().__init__()
        if min(text_dim, video_dim, action_dim, hidden_dim, projection_dim) <= 0:
            raise ValueError("PGC dimensions must be positive.")
        if num_goal_tokens <= 0:
            raise ValueError("`num_goal_tokens` must be positive.")
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"PGC hidden_dim={hidden_dim} must divide num_heads={num_heads}."
            )

        self.text_dim = int(text_dim)
        self.video_dim = int(video_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.projection_dim = int(projection_dim)
        self.num_goal_tokens = int(num_goal_tokens)

        self.goal_slots = nn.Parameter(
            torch.randn(1, self.num_goal_tokens, hidden_dim)
            / math.sqrt(hidden_dim)
        )
        self.text_projection = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, hidden_dim),
        )
        self.video_projection = nn.Sequential(
            nn.LayerNorm(video_dim),
            nn.Linear(video_dim, hidden_dim),
        )
        self.query_projection = nn.Sequential(
            nn.LayerNorm(action_dim),
            nn.Linear(action_dim, hidden_dim),
        )
        self.language_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.visual_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.query_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.language_norm = nn.LayerNorm(hidden_dim)
        self.visual_norm = nn.LayerNorm(hidden_dim)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.goal_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.query_output = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(action_dim, action_dim),
        )
        self.goal_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(
        self,
        *,
        base_queries: torch.Tensor,
        language_hidden: torch.Tensor,
        language_mask: torch.Tensor,
        current_video_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if base_queries.ndim != 3:
            raise ValueError("`base_queries` must be [B,K,D_action].")
        if language_hidden.ndim != 3 or current_video_hidden.ndim != 3:
            raise ValueError("PGC language/video inputs must be 3D tensors.")
        batch_size = int(base_queries.shape[0])
        if language_hidden.shape[0] != batch_size or (
            current_video_hidden.shape[0] != batch_size
        ):
            raise ValueError("PGC inputs must share their batch dimension.")
        if base_queries.shape[-1] != self.action_dim:
            raise ValueError("PGC base-query dimension mismatch.")
        if language_hidden.shape[-1] != self.text_dim:
            raise ValueError("PGC language dimension mismatch.")
        if current_video_hidden.shape[-1] != self.video_dim:
            raise ValueError("PGC video dimension mismatch.")

        text = self.text_projection(language_hidden)
        text, text_padding = _safe_key_padding_mask(text, language_mask)
        visual = self.video_projection(current_video_hidden)
        goal = self.goal_slots.expand(batch_size, -1, -1)

        language_delta, language_weights = self.language_attention(
            query=goal,
            key=text,
            value=text,
            key_padding_mask=text_padding,
            need_weights=True,
        )
        goal = self.language_norm(goal + language_delta)
        visual_delta, visual_weights = self.visual_attention(
            query=goal,
            key=visual,
            value=visual,
            need_weights=True,
        )
        goal = self.visual_norm(goal + visual_delta)
        goal = goal + self.goal_mlp(goal)

        query_hidden = self.query_projection(base_queries)
        query_delta, _ = self.query_attention(
            query=query_hidden,
            key=goal,
            value=goal,
            need_weights=False,
        )
        query_delta = self.query_output(self.query_norm(query_hidden + query_delta))
        routed_queries = base_queries + query_delta
        goal_embedding = F.normalize(
            self.goal_projection(goal.mean(dim=1)).float(), dim=-1, eps=1.0e-6
        )

        metrics = _query_diagnostics(routed_queries, visual_weights)
        language_probs = language_weights.float().clamp_min(1.0e-8)
        language_entropy = -(language_probs * language_probs.log()).sum(dim=-1)
        language_entropy = language_entropy / math.log(
            max(2, language_probs.shape[-1])
        )
        metrics["pgc_language_attention_entropy"] = language_entropy.mean()
        metrics["pgc_goal_embedding_norm"] = goal_embedding.norm(dim=-1).mean()
        return routed_queries, goal_embedding, metrics


class GoalResidualAdapter(nn.Module):
    """Inject goal information without replacing the pretrained visual path.

    PGC v2 keeps the counterfactual Action Expert query-free, exactly like the
    released FastWAM policy.  This adapter lets each action token attend to the
    state-grounded goal queries and adds the resulting language residual to the
    normal action-token stream.  The final projection is initialized to zero,
    so a freshly constructed v2 branch is numerically identical to the frozen
    base policy even though Goal Graph representations are already available.
    """

    def __init__(
        self,
        *,
        action_dim: int,
        num_heads: int = 8,
        residual_scale: float = 1.0,
    ):
        super().__init__()
        if action_dim <= 0:
            raise ValueError("`action_dim` must be positive.")
        if num_heads <= 0 or action_dim % num_heads != 0:
            raise ValueError(
                f"PGC residual action_dim={action_dim} must divide "
                f"num_heads={num_heads}."
            )
        if residual_scale < 0:
            raise ValueError("`residual_scale` must be non-negative.")

        self.action_dim = int(action_dim)
        self.num_heads = int(num_heads)
        self.residual_scale = float(residual_scale)
        self.action_norm = nn.LayerNorm(action_dim)
        self.goal_norm = nn.LayerNorm(action_dim)
        self.goal_attention = nn.MultiheadAttention(
            action_dim, num_heads, batch_first=True
        )
        self.output_projection = nn.Linear(action_dim, action_dim)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        action_tokens: torch.Tensor,
        goal_queries: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if action_tokens.ndim != 3 or goal_queries.ndim != 3:
            raise ValueError("PGC residual inputs must be [B,S,D] tensors.")
        if action_tokens.shape[0] != goal_queries.shape[0]:
            raise ValueError("PGC residual inputs must share a batch dimension.")
        if action_tokens.shape[-1] != self.action_dim or (
            goal_queries.shape[-1] != self.action_dim
        ):
            raise ValueError(
                "PGC residual inputs do not match the configured action dimension."
            )
        if goal_queries.shape[1] <= 0:
            raise ValueError("PGC residual requires at least one goal query.")

        residual_hidden, attention = self.goal_attention(
            query=self.action_norm(action_tokens),
            key=self.goal_norm(goal_queries),
            value=self.goal_norm(goal_queries),
            need_weights=True,
        )
        residual = self.output_projection(residual_hidden)
        applied_residual = residual * self.residual_scale
        output = action_tokens + applied_residual

        probabilities = attention.float().clamp_min(1.0e-8)
        entropy = -(probabilities * probabilities.log()).sum(dim=-1)
        entropy = entropy / math.log(max(2, probabilities.shape[-1]))
        metrics = {
            "pgc_goal_residual_hidden_norm": (
                residual_hidden.float().norm(dim=-1).mean()
            ),
            "pgc_goal_residual_norm": (
                applied_residual.float().norm(dim=-1).mean()
            ),
            "pgc_goal_residual_max_abs": applied_residual.float().abs().max(),
            "pgc_goal_residual_attention_entropy": entropy.mean(),
            "pgc_goal_residual_scale": output.new_tensor(self.residual_scale),
        }
        return output, metrics


class BoundedActionVelocityResidual(nn.Module):
    """Predict a bounded goal-conditioned correction to frozen Base velocity.

    PGC v3 never adapts or duplicates the released Action Expert. The frozen
    expert first produces its normal hidden action tokens and velocity; this
    module reads those tokens plus state-grounded goal queries and predicts a
    small correction in flow-velocity space. The final projection is exactly
    zero initialized and the output is bounded with ``tanh`` so a new v3 model
    is numerically identical to Base and cannot produce an unbounded policy
    perturbation during training.
    """

    def __init__(
        self,
        *,
        action_hidden_dim: int,
        action_dim: int,
        num_heads: int = 8,
        max_abs: float | Sequence[float] = 1.0,
    ):
        super().__init__()
        if action_hidden_dim <= 0 or action_dim <= 0:
            raise ValueError("PGC velocity-residual dimensions must be positive.")
        if num_heads <= 0 or action_hidden_dim % num_heads != 0:
            raise ValueError(
                f"PGC velocity-residual hidden_dim={action_hidden_dim} must "
                f"divide num_heads={num_heads}."
            )
        if isinstance(max_abs, Sequence) and not isinstance(max_abs, (str, bytes)):
            cap_values = [float(value) for value in max_abs]
            if len(cap_values) != int(action_dim):
                raise ValueError(
                    "Per-dimension PGC residual caps must match action_dim: "
                    f"{len(cap_values)} vs {action_dim}."
                )
        else:
            cap_values = [float(max_abs)] * int(action_dim)
        if any(value <= 0 for value in cap_values):
            raise ValueError("PGC velocity-residual caps must be positive.")

        self.action_hidden_dim = int(action_hidden_dim)
        self.action_dim = int(action_dim)
        self.num_heads = int(num_heads)
        self.action_norm = nn.LayerNorm(action_hidden_dim)
        self.goal_norm = nn.LayerNorm(action_hidden_dim)
        self.goal_attention = nn.MultiheadAttention(
            action_hidden_dim, num_heads, batch_first=True
        )
        self.fusion = nn.Sequential(
            nn.Linear(action_hidden_dim * 2, action_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.LayerNorm(action_hidden_dim),
        )
        self.output_projection = nn.Linear(action_hidden_dim, action_dim)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        self.register_buffer(
            "max_abs",
            torch.tensor(cap_values, dtype=torch.float32).view(1, 1, action_dim),
            persistent=True,
        )

    def forward(
        self,
        action_hidden: torch.Tensor,
        goal_queries: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if action_hidden.ndim != 3 or goal_queries.ndim != 3:
            raise ValueError("PGC v3 residual inputs must be [B,S,D] tensors.")
        if action_hidden.shape[0] != goal_queries.shape[0]:
            raise ValueError("PGC v3 residual inputs must share a batch dimension.")
        if action_hidden.shape[-1] != self.action_hidden_dim or (
            goal_queries.shape[-1] != self.action_hidden_dim
        ):
            raise ValueError("PGC v3 residual hidden dimensions do not match.")
        if goal_queries.shape[1] <= 0:
            raise ValueError("PGC v3 residual requires at least one goal query.")

        normalized_action = self.action_norm(action_hidden)
        normalized_goal = self.goal_norm(goal_queries)
        goal_delta, attention = self.goal_attention(
            query=normalized_action,
            key=normalized_goal,
            value=normalized_goal,
            need_weights=True,
        )
        residual_hidden = self.fusion(
            torch.cat([normalized_action, goal_delta], dim=-1)
        )
        raw_residual = self.output_projection(residual_hidden)
        cap = self.max_abs.to(
            device=raw_residual.device, dtype=raw_residual.dtype
        )
        residual = torch.tanh(raw_residual) * cap

        probabilities = attention.float().clamp_min(1.0e-8)
        entropy = -(probabilities * probabilities.log()).sum(dim=-1)
        entropy = entropy / math.log(max(2, probabilities.shape[-1]))
        saturation = residual.float().abs() >= (
            cap.float() * 0.95
        )
        metrics = {
            "pgc_velocity_residual_hidden_norm": (
                residual_hidden.float().norm(dim=-1).mean()
            ),
            "pgc_velocity_residual_raw_norm": (
                raw_residual.float().norm(dim=-1).mean()
            ),
            "pgc_velocity_residual_norm": (
                residual.float().norm(dim=-1).mean()
            ),
            "pgc_velocity_residual_max_abs": residual.float().abs().max(),
            "pgc_velocity_residual_cap_mean": cap.float().mean(),
            "pgc_velocity_residual_saturation_fraction": (
                saturation.float().mean()
            ),
            "pgc_velocity_residual_attention_entropy": entropy.mean(),
        }
        return residual, metrics


class ActionOutcomeVerifier(nn.Module):
    """Score action chunks against the current state and requested goal."""

    def __init__(
        self,
        *,
        action_dim: int,
        video_dim: int,
        goal_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        if min(action_dim, video_dim, goal_dim, hidden_dim) <= 0:
            raise ValueError("Verifier dimensions must be positive.")
        self.action_dim = int(action_dim)
        self.video_dim = int(video_dim)
        self.goal_dim = int(goal_dim)
        self.hidden_dim = int(hidden_dim)
        self.visual_encoder = nn.Sequential(
            nn.LayerNorm(video_dim),
            nn.Linear(video_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
        )
        self.goal_encoder = nn.Sequential(
            nn.LayerNorm(goal_dim),
            nn.Linear(goal_dim, hidden_dim),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.goal_state_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _masked_temporal_mean(
        hidden: torch.Tensor, action_is_pad: torch.Tensor | None
    ) -> torch.Tensor:
        if action_is_pad is None:
            return hidden.mean(dim=1)
        if action_is_pad.shape != hidden.shape[:2]:
            raise ValueError(
                "Verifier action padding mask mismatch: "
                f"{tuple(action_is_pad.shape)} vs {tuple(hidden.shape[:2])}."
            )
        valid = (~action_is_pad.to(device=hidden.device, dtype=torch.bool)).to(
            hidden.dtype
        )
        return (hidden * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)

    def encode_action(
        self,
        action: torch.Tensor,
        action_is_pad: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if action.ndim != 3 or action.shape[-1] != self.action_dim:
            raise ValueError(
                f"Verifier action must be [B,T,{self.action_dim}], got "
                f"{tuple(action.shape)}."
            )
        hidden = self.action_encoder(action)
        return F.normalize(
            self._masked_temporal_mean(hidden, action_is_pad).float(),
            dim=-1,
            eps=1.0e-6,
        )

    def encode_goal_state(
        self,
        current_video_hidden: torch.Tensor,
        goal_embedding: torch.Tensor,
    ) -> torch.Tensor:
        if current_video_hidden.ndim != 3 or (
            current_video_hidden.shape[-1] != self.video_dim
        ):
            raise ValueError("Verifier current visual tokens have invalid shape.")
        if goal_embedding.ndim != 2 or goal_embedding.shape[-1] != self.goal_dim:
            raise ValueError("Verifier goal embedding has invalid shape.")
        visual = self.visual_encoder(current_video_hidden).mean(dim=1)
        goal = self.goal_encoder(goal_embedding.to(dtype=visual.dtype))
        return F.normalize(
            self.goal_state_fusion(torch.cat([visual, goal], dim=-1)).float(),
            dim=-1,
            eps=1.0e-6,
        )

    def forward(
        self,
        *,
        current_video_hidden: torch.Tensor,
        goal_embedding: torch.Tensor,
        action: torch.Tensor,
        action_is_pad: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        goal_state = self.encode_goal_state(current_video_hidden, goal_embedding)
        action_embedding = self.encode_action(action, action_is_pad)
        logits = self.score_embeddings(goal_state, action_embedding)
        return logits, goal_state, action_embedding

    def score_embeddings(
        self,
        goal_state: torch.Tensor,
        action_embedding: torch.Tensor,
    ) -> torch.Tensor:
        if goal_state.shape != action_embedding.shape or goal_state.ndim != 2:
            raise ValueError("Verifier embeddings must share [B,D] shape.")
        goal_state_for_head = goal_state.to(dtype=action_embedding.dtype)
        interaction = torch.cat(
            [
                goal_state_for_head,
                action_embedding,
                (goal_state_for_head - action_embedding).abs(),
                goal_state_for_head * action_embedding,
            ],
            dim=-1,
        )
        score_dtype = next(self.score_head.parameters()).dtype
        return self.score_head(interaction.to(dtype=score_dtype)).squeeze(-1)


class GoalActionAlignmentLoss(nn.Module):
    """Distributed symmetric InfoNCE for goals and demonstrated action chunks."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        if temperature <= 0:
            raise ValueError("`temperature` must be positive.")
        self.temperature = float(temperature)

    def forward(
        self,
        goal_state: torch.Tensor,
        action_embedding: torch.Tensor,
        *,
        group_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if goal_state.shape != action_embedding.shape or goal_state.ndim != 2:
            raise ValueError("PGC alignment embeddings must share [B,D] shape.")
        batch_size = int(goal_state.shape[0])
        goal_state = F.normalize(goal_state.float(), dim=-1, eps=1.0e-6)
        action_embedding = F.normalize(
            action_embedding.float(), dim=-1, eps=1.0e-6
        )
        all_goal_state, rank = _gather_with_grad(goal_state)
        all_action_embedding, action_rank = _gather_with_grad(action_embedding)
        if rank != action_rank:
            raise RuntimeError("Distributed PGC alignment gather rank mismatch.")

        candidate_count = int(all_action_embedding.shape[0])
        if int(all_goal_state.shape[0]) != candidate_count:
            raise RuntimeError(
                "Distributed PGC alignment candidate count mismatch: "
                f"{int(all_goal_state.shape[0])} vs {candidate_count}."
            )
        labels = rank * batch_size + torch.arange(
            batch_size, device=goal_state.device
        )
        similarities_ga = torch.matmul(
            goal_state, all_action_embedding.transpose(0, 1)
        )
        similarities_ag = torch.matmul(
            action_embedding, all_goal_state.transpose(0, 1)
        )
        positive_mask = torch.zeros_like(similarities_ga, dtype=torch.bool)
        positive_mask.scatter_(1, labels[:, None], True)
        same_goal = torch.zeros_like(positive_mask)
        if group_ids is not None:
            if group_ids.shape != (batch_size,):
                raise ValueError("PGC `group_ids` must be [B].")
            group_ids = group_ids.to(device=goal_state.device, dtype=torch.long)
            all_group_ids = _gather_without_grad(group_ids)
            if int(all_group_ids.shape[0]) != candidate_count:
                raise RuntimeError(
                    "Distributed PGC alignment group count mismatch: "
                    f"{int(all_group_ids.shape[0])} vs {candidate_count}."
                )
            same_goal = (
                group_ids[:, None] == all_group_ids[None, :]
            ) & ~positive_mask
        mask_value = torch.finfo(similarities_ga.dtype).min
        masked_ga = similarities_ga.masked_fill(same_goal, mask_value)
        masked_ag = similarities_ag.masked_fill(same_goal, mask_value)
        negative_mask = ~positive_mask & ~same_goal
        if candidate_count > 1 and bool(negative_mask.any()):
            loss = 0.5 * (
                F.cross_entropy(masked_ga / self.temperature, labels)
                + F.cross_entropy(masked_ag / self.temperature, labels)
            )
            negative_values = similarities_ga.masked_fill(~negative_mask, -1.0)
            negative = negative_values.max(dim=1).values
            negative = torch.where(
                negative_mask.any(dim=1), negative, torch.zeros_like(negative)
            )
        else:
            # Keep a differentiable scalar for single-process smoke tests and
            # batches where every gathered sample represents the same goal.
            loss = (goal_state.sum() + action_embedding.sum()) * 0.0
            negative = similarities_ga.new_zeros((batch_size,))
        positive = similarities_ga.gather(1, labels[:, None]).squeeze(1)
        denominator = max(1, batch_size * max(1, candidate_count - 1))
        return loss, {
            "pgc_goal_action_positive_similarity": positive.mean(),
            "pgc_goal_action_negative_similarity": negative.mean(),
            "pgc_goal_action_margin": (positive - negative).mean(),
            "pgc_goal_action_retrieval_acc": (
                masked_ga.argmax(dim=1) == labels
            ).float().mean(),
            "pgc_goal_action_candidate_count": similarities_ga.new_tensor(
                float(candidate_count)
            ),
            "pgc_goal_action_effective_negative_count": (
                negative_mask.sum(dim=1).float().mean()
            ),
            "pgc_goal_action_same_goal_negative_fraction": (
                same_goal.float().sum() / denominator
            ),
        }


def detached_policy_guard_metrics(
    metrics: dict[str, Any]
) -> dict[str, float]:
    return {
        name: (
            float(value.detach().float().cpu().item())
            if isinstance(value, torch.Tensor)
            else float(value)
        )
        for name, value in metrics.items()
    }
