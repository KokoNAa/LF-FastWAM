"""Latent transition contract modules for TC-FastWAM.

The deployment path contains only the intent/router branch.  The outcome and
action-effect encoders plus their contract losses are training-time teachers.
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


class ActionEffectEncoder(nn.Module):
    """Encode the transition induced by a clean action chunk.

    The encoder is deliberately small and training-only.  It combines a
    language-neutral current-view token with the ordered action sequence and
    optional current proprioception, then projects the result into the shared
    transition space.
    """

    def __init__(
        self,
        *,
        action_dim: int,
        video_dim: int,
        projection_dim: int,
        proprio_dim: int | None = None,
        hidden_dim: int | None = None,
        num_heads: int = 8,
        num_layers: int = 2,
    ):
        super().__init__()
        hidden_dim = int(hidden_dim or projection_dim)
        if hidden_dim <= 0:
            raise ValueError("`hidden_dim` must be positive.")
        if num_heads <= 0 or hidden_dim % int(num_heads) != 0:
            raise ValueError(
                f"Action-effect hidden_dim={hidden_dim} must divide "
                f"num_heads={num_heads}."
            )
        if int(num_layers) <= 0:
            raise ValueError("`num_layers` must be positive.")

        self.action_dim = int(action_dim)
        self.video_dim = int(video_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        self.hidden_dim = hidden_dim
        self.action_projection = nn.Linear(self.action_dim, hidden_dim)
        self.visual_projection = nn.Sequential(
            nn.LayerNorm(self.video_dim),
            nn.Linear(self.video_dim, hidden_dim),
        )
        self.proprio_projection = (
            nn.Linear(self.proprio_dim, hidden_dim)
            if self.proprio_dim is not None
            else None
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=int(num_heads),
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.projection = TransitionProjectionHead(hidden_dim, projection_dim)

    @staticmethod
    def _position_embedding(
        length: int,
        dim: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        positions = torch.arange(length, device=device, dtype=torch.float32)[:, None]
        half_dim = max(1, (dim + 1) // 2)
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, device=device, dtype=torch.float32)
            / max(1, half_dim - 1)
        )[None, :]
        angles = positions * frequencies
        embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)[:, :dim]
        return embedding.to(dtype=dtype)

    def forward(
        self,
        *,
        current_video_hidden: torch.Tensor,
        action: torch.Tensor,
        proprio: torch.Tensor | None = None,
        action_is_pad: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if current_video_hidden.ndim != 3:
            raise ValueError("`current_video_hidden` must be [B,N,D_video].")
        if action.ndim != 3 or action.shape[-1] != self.action_dim:
            raise ValueError(
                "`action` must be [B,T,action_dim], got "
                f"{tuple(action.shape)} with action_dim={self.action_dim}."
            )
        batch_size, horizon, _ = action.shape
        if current_video_hidden.shape[0] != batch_size or (
            current_video_hidden.shape[-1] != self.video_dim
        ):
            raise ValueError("Action-effect video batch/feature dimensions mismatch.")
        if horizon <= 0:
            raise ValueError("Action-effect encoding requires a non-empty action chunk.")

        if action_is_pad is None:
            action_is_pad = torch.zeros(
                (batch_size, horizon), dtype=torch.bool, device=action.device
            )
        else:
            if action_is_pad.shape != (batch_size, horizon):
                raise ValueError(
                    "`action_is_pad` must match [B,T], got "
                    f"{tuple(action_is_pad.shape)}."
                )
            action_is_pad = action_is_pad.to(device=action.device, dtype=torch.bool)

        context_token = self.visual_projection(current_video_hidden.mean(dim=1))
        if self.proprio_projection is not None:
            if proprio is None or proprio.shape != (batch_size, self.proprio_dim):
                raise ValueError(
                    "Action-effect proprio must be [B,proprio_dim], got "
                    f"{None if proprio is None else tuple(proprio.shape)}."
                )
            context_token = context_token + self.proprio_projection(proprio)
        elif proprio is not None:
            raise ValueError("Received proprio but ActionEffectEncoder has no proprio_dim.")

        action_hidden = self.action_projection(action)
        action_hidden = action_hidden + self._position_embedding(
            horizon,
            self.hidden_dim,
            device=action.device,
            dtype=action_hidden.dtype,
        )[None, :, :]
        sequence = torch.cat([context_token[:, None, :], action_hidden], dim=1)
        padding_mask = torch.cat(
            [
                torch.zeros(
                    (batch_size, 1), dtype=torch.bool, device=action.device
                ),
                action_is_pad,
            ],
            dim=1,
        )
        encoded = self.encoder(sequence, src_key_padding_mask=padding_mask)
        valid = (~action_is_pad).to(dtype=encoded.dtype)
        action_pool = (
            encoded[:, 1:] * valid[:, :, None]
        ).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        effect_hidden = self.output_norm(encoded[:, 0] + action_pool)
        return self.projection(effect_hidden)


class CounterfactualActionPrototypeBank(nn.Module):
    """EMA task prototypes grounded by deployed queries and demonstrated actions.

    The bank is training-only.  It gives an executable alternate instruction a
    positive target without copying a raw action chunk from a different robot
    state into the source state.  Query prototypes act directly in the policy
    interface, while action-effect prototypes ground them in demonstrated
    behavior.
    """

    def __init__(
        self,
        *,
        num_slots: int,
        num_queries: int,
        query_dim: int,
        action_effect_dim: int,
        momentum: float = 0.95,
    ) -> None:
        super().__init__()
        if num_slots <= 0 or num_queries <= 0:
            raise ValueError("Prototype slots and query count must be positive.")
        if query_dim <= 0 or action_effect_dim <= 0:
            raise ValueError("Prototype feature dimensions must be positive.")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("Prototype momentum must be in [0, 1).")
        self.num_slots = int(num_slots)
        self.num_queries = int(num_queries)
        self.query_dim = int(query_dim)
        self.action_effect_dim = int(action_effect_dim)
        self.momentum = float(momentum)
        # These buffers are deliberately excluded from adapter checkpoints.
        # They are online training targets, not deployment parameters.
        self.register_buffer(
            "task_ids",
            torch.full((self.num_slots,), -1, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "counts",
            torch.zeros((self.num_slots,), dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "query_prototypes",
            torch.zeros(
                self.num_slots,
                self.num_queries,
                self.query_dim,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "action_effect_prototypes",
            torch.zeros(
                self.num_slots,
                self.action_effect_dim,
                dtype=torch.float32,
            ),
            persistent=False,
        )

    def _validate_embeddings(
        self,
        task_ids: torch.Tensor,
        query_residuals: torch.Tensor,
        action_effects: torch.Tensor,
    ) -> None:
        batch_size = task_ids.shape[0]
        if task_ids.ndim != 1:
            raise ValueError("Prototype task IDs must be [B].")
        if query_residuals.shape != (
            batch_size,
            self.num_queries,
            self.query_dim,
        ):
            raise ValueError(
                "Query prototype input shape mismatch: got "
                f"{tuple(query_residuals.shape)}."
            )
        if action_effects.shape != (batch_size, self.action_effect_dim):
            raise ValueError(
                "Action-effect prototype input shape mismatch: got "
                f"{tuple(action_effects.shape)}."
            )

    def _slots_for(self, task_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        task_ids = task_ids.to(device=self.task_ids.device, dtype=torch.long)
        matches = task_ids[:, None] == self.task_ids[None, :]
        available = matches.any(dim=1)
        slots = matches.to(dtype=torch.long).argmax(dim=1)
        return slots, available

    @torch.no_grad()
    def update(
        self,
        *,
        task_ids: torch.Tensor,
        query_residuals: torch.Tensor,
        action_effects: torch.Tensor,
    ) -> None:
        self._validate_embeddings(task_ids, query_residuals, action_effects)
        task_ids = _gather_without_grad(
            task_ids.detach().to(device=self.task_ids.device, dtype=torch.long)
        )
        query_residuals = _gather_without_grad(
            F.normalize(query_residuals.detach().float(), dim=-1, eps=1e-6)
        )
        action_effects = _gather_without_grad(
            F.normalize(action_effects.detach().float(), dim=-1, eps=1e-6)
        )
        for task_id, query_value, action_value in zip(
            task_ids,
            query_residuals,
            action_effects,
            strict=True,
        ):
            task_id_value = int(task_id.item())
            existing = torch.nonzero(
                self.task_ids == task_id_value,
                as_tuple=False,
            ).flatten()
            if existing.numel() > 0:
                slot = int(existing[0].item())
            else:
                empty = torch.nonzero(self.task_ids < 0, as_tuple=False).flatten()
                if empty.numel() == 0:
                    raise RuntimeError(
                        "Counterfactual action prototype bank is full; increase "
                        "`counterfactual_action_prototype_slots`."
                    )
                slot = int(empty[0].item())
                self.task_ids[slot] = task_id_value

            if int(self.counts[slot].item()) == 0:
                updated_query = query_value
                updated_action = action_value
            else:
                updated_query = torch.lerp(
                    query_value,
                    self.query_prototypes[slot].float(),
                    self.momentum,
                )
                updated_action = torch.lerp(
                    action_value,
                    self.action_effect_prototypes[slot].float(),
                    self.momentum,
                )
            self.query_prototypes[slot].copy_(
                F.normalize(updated_query, dim=-1, eps=1e-6).to(
                    dtype=self.query_prototypes.dtype
                )
            )
            self.action_effect_prototypes[slot].copy_(
                F.normalize(updated_action, dim=-1, eps=1e-6).to(
                    dtype=self.action_effect_prototypes.dtype
                )
            )
            self.counts[slot] += 1

    def available_mask(self, task_ids: torch.Tensor) -> torch.Tensor:
        return self._slots_for(task_ids)[1].to(device=task_ids.device)

    def positive_loss(
        self,
        *,
        counterfactual_task_ids: torch.Tensor,
        query_residuals: torch.Tensor,
        action_intents: torch.Tensor,
        valid_mask: torch.Tensor,
        query_weight: float = 1.0,
        action_effect_weight: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._validate_embeddings(
            counterfactual_task_ids,
            query_residuals,
            action_intents,
        )
        if valid_mask.shape != counterfactual_task_ids.shape:
            raise ValueError("Counterfactual prototype valid mask must be [B].")
        if query_weight < 0 or action_effect_weight < 0:
            raise ValueError("Counterfactual positive weights must be non-negative.")

        slots, available = self._slots_for(counterfactual_task_ids)
        available = available.to(device=query_residuals.device)
        valid_mask = valid_mask.to(
            device=query_residuals.device,
            dtype=torch.bool,
        ) & available
        target_queries = self.query_prototypes[slots].to(
            device=query_residuals.device,
            dtype=torch.float32,
        )
        target_actions = self.action_effect_prototypes[slots].to(
            device=action_intents.device,
            dtype=torch.float32,
        )
        normalized_queries = F.normalize(
            query_residuals.float(), dim=-1, eps=1e-6
        )
        normalized_actions = F.normalize(
            action_intents.float(), dim=-1, eps=1e-6
        )
        query_similarity = (
            normalized_queries * target_queries
        ).sum(dim=-1).mean(dim=-1)
        action_similarity = (
            normalized_actions * target_actions
        ).sum(dim=-1)
        valid = valid_mask.to(dtype=query_similarity.dtype)
        valid_count = valid.sum()
        query_loss = ((1.0 - query_similarity) * valid).sum() / (
            valid_count.clamp_min(1.0)
        )
        action_loss = ((1.0 - action_similarity) * valid).sum() / (
            valid_count.clamp_min(1.0)
        )
        loss = query_weight * query_loss + action_effect_weight * action_loss
        if not bool(valid_mask.any()):
            loss = (query_residuals.sum() + action_intents.sum()) * 0.0

        active_slots = torch.nonzero(self.task_ids >= 0, as_tuple=False).flatten()
        retrieval_accuracy = query_similarity.new_zeros(())
        if active_slots.numel() > 0 and bool(valid_mask.any()):
            active_queries = self.query_prototypes[active_slots].to(
                device=query_residuals.device,
                dtype=torch.float32,
            )
            active_actions = self.action_effect_prototypes[active_slots].to(
                device=action_intents.device,
                dtype=torch.float32,
            )
            query_logits = torch.einsum(
                "bkd,skd->bs",
                normalized_queries,
                active_queries,
            ) / self.num_queries
            action_logits = torch.matmul(
                normalized_actions,
                active_actions.transpose(0, 1),
            )
            predicted_slots = active_slots.to(query_logits.device)[
                (query_logits + action_logits).argmax(dim=1)
            ]
            retrieval_accuracy = (
                (predicted_slots == slots.to(predicted_slots.device)).float()
                * valid
            ).sum() / valid_count.clamp_min(1.0)

        def _valid_mean(value: torch.Tensor) -> torch.Tensor:
            return (value * valid).sum() / valid_count.clamp_min(1.0)

        return loss, {
            "loss_counterfactual_action_query_positive": query_loss,
            "loss_counterfactual_action_effect_positive": action_loss,
            "sim_CAP_query_positive": _valid_mean(query_similarity),
            "sim_CAP_action_positive": _valid_mean(action_similarity),
            "counterfactual_action_positive_valid_fraction": valid.mean(),
            "counterfactual_action_positive_valid_count": valid_count,
            "counterfactual_action_prototype_retrieval_acc": retrieval_accuracy,
            "counterfactual_action_prototype_count": (
                (self.task_ids >= 0).sum().to(dtype=query_similarity.dtype)
            ),
        }


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


def _gather_without_grad(tensor: torch.Tensor) -> torch.Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return tensor
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)


class ContrastiveContractLoss(nn.Module):
    """Symmetric transition InfoNCE with distributed in-batch negatives."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        if temperature <= 0:
            raise ValueError("`temperature` must be positive.")
        self.temperature = float(temperature)

    def forward(
        self,
        z_source: torch.Tensor,
        z_future: torch.Tensor,
        *,
        metric_prefix: str = "LF",
        group_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if z_source.shape != z_future.shape or z_source.ndim != 2:
            raise ValueError(
                "Transition embeddings must have equal [B,D] shapes, got "
                f"{tuple(z_source.shape)} and {tuple(z_future.shape)}."
            )
        if metric_prefix not in {"LF", "AF"}:
            raise ValueError(
                f"`metric_prefix` must be LF or AF, got {metric_prefix!r}."
            )
        z_source = F.normalize(z_source.float(), dim=-1, eps=1e-6)
        z_future = F.normalize(z_future.float(), dim=-1, eps=1e-6)
        all_source, rank = _gather_with_grad(z_source)
        all_future, future_rank = _gather_with_grad(z_future)
        if rank != future_rank:
            raise RuntimeError("Distributed transition gather rank mismatch.")

        batch_size = z_source.shape[0]
        candidate_count = all_source.shape[0]
        labels = rank * batch_size + torch.arange(
            batch_size, device=z_source.device
        )
        logits_sf = torch.matmul(z_source, all_future.transpose(0, 1))
        logits_fs = torch.matmul(z_future, all_source.transpose(0, 1))
        positive_mask = torch.zeros_like(logits_sf, dtype=torch.bool)
        positive_mask.scatter_(1, labels[:, None], True)
        same_group_negative = torch.zeros_like(logits_sf, dtype=torch.bool)
        if group_ids is not None:
            if group_ids.shape != (batch_size,):
                raise ValueError(
                    f"`group_ids` must be [B], got {tuple(group_ids.shape)}."
                )
            group_ids = group_ids.to(device=z_source.device, dtype=torch.long)
            all_group_ids = _gather_without_grad(group_ids)
            same_group_negative = (
                group_ids[:, None] == all_group_ids[None, :]
            ) & ~positive_mask
            mask_value = torch.finfo(logits_sf.dtype).min
            logits_sf = logits_sf.masked_fill(same_group_negative, mask_value)
            logits_fs = logits_fs.masked_fill(same_group_negative, mask_value)
        if candidate_count > 1:
            loss = 0.5 * (
                F.cross_entropy(logits_sf / self.temperature, labels)
                + F.cross_entropy(logits_fs / self.temperature, labels)
            )
            negative_mask = ~positive_mask & ~same_group_negative
            negative_values = logits_sf.masked_fill(~negative_mask, -1.0)
            negative = negative_values.max(dim=1).values
            negative = torch.where(
                negative_mask.any(dim=1), negative, torch.zeros_like(negative)
            )
        else:
            # Keep a differentiable scalar for single-card unit/smoke batches.
            loss = (z_source.sum() + z_future.sum()) * 0.0
            negative = logits_sf.new_zeros((batch_size,))

        positive = logits_sf.gather(1, labels[:, None]).squeeze(1)
        retrieval_key = (
            "contract_retrieval_acc"
            if metric_prefix == "LF"
            else "contract_retrieval_acc_AF"
        )
        candidate_key = (
            "contract_candidate_count"
            if metric_prefix == "LF"
            else "contract_candidate_count_AF"
        )
        metrics = {
            f"sim_{metric_prefix}_positive": positive.mean(),
            f"sim_{metric_prefix}_negative": negative.mean(),
            f"sim_{metric_prefix}_margin": (positive - negative).mean(),
            retrieval_key: (logits_sf.argmax(dim=1) == labels)
            .float()
            .mean(),
            candidate_key: logits_sf.new_tensor(
                float(candidate_count)
            ),
            f"contract_effective_negative_count_{metric_prefix}": (
                (~positive_mask & ~same_group_negative)
                .sum(dim=1)
                .float()
                .mean()
            ),
            f"contract_same_task_negative_fraction_{metric_prefix}": (
                same_group_negative.float().sum()
                / max(1, batch_size * max(1, candidate_count - 1))
            ),
        }
        return loss, metrics


class CounterfactualRankingLoss(nn.Module):
    """Rank the realized future above an executable same-scene instruction."""

    def __init__(self, margin: float = 0.2):
        super().__init__()
        if margin < 0:
            raise ValueError("`margin` must be non-negative.")
        self.margin = float(margin)

    def forward(
        self,
        z_positive: torch.Tensor,
        z_negative: torch.Tensor,
        z_future: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if (
            z_positive.shape != z_negative.shape
            or z_positive.shape != z_future.shape
            or z_positive.ndim != 2
        ):
            raise ValueError(
                "Counterfactual embeddings must share [B,D] shape, got "
                f"{tuple(z_positive.shape)}, {tuple(z_negative.shape)}, "
                f"{tuple(z_future.shape)}."
            )
        batch_size = z_positive.shape[0]
        if valid_mask is None:
            valid_mask = torch.ones(
                batch_size, dtype=torch.bool, device=z_positive.device
            )
        elif valid_mask.shape != (batch_size,):
            raise ValueError(
                f"`valid_mask` must be [B], got {tuple(valid_mask.shape)}."
            )
        valid_mask = valid_mask.to(device=z_positive.device, dtype=torch.bool)

        positive = F.cosine_similarity(
            z_positive.float(), z_future.float(), dim=-1, eps=1e-6
        )
        negative = F.cosine_similarity(
            z_negative.float(), z_future.float(), dim=-1, eps=1e-6
        )
        margin = positive - negative
        per_sample = torch.relu(self.margin - margin)
        valid = valid_mask.to(per_sample.dtype)
        valid_count = valid.sum()
        loss = (per_sample * valid).sum() / valid_count.clamp_min(1.0)
        # Retain a differentiable zero when a stochastic batch has no explicit
        # negative examples.
        if not bool(valid_mask.any()):
            loss = (z_positive.sum() + z_negative.sum() + z_future.sum()) * 0.0

        def _valid_mean(value: torch.Tensor) -> torch.Tensor:
            return (value * valid).sum() / valid_count.clamp_min(1.0)

        metrics = {
            "sim_CF_positive": _valid_mean(positive),
            "sim_CF_negative": _valid_mean(negative),
            "sim_CF_margin": _valid_mean(margin),
            "counterfactual_margin_satisfied_fraction": _valid_mean(
                (margin >= self.margin).to(valid.dtype)
            ),
            "counterfactual_valid_fraction": valid.mean(),
            "counterfactual_valid_count": valid_count,
        }
        return loss, metrics


def detached_metrics(metrics: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Convert scalar tensor metrics to logging-safe Python values."""
    return {
        key: float(value.detach().float().cpu().item())
        for key, value in metrics.items()
    }
