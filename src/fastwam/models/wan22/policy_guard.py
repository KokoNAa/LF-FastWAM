"""Policy-Guarded Counterfactual (PGC) modules for FastWAM.

PGC deliberately keeps the released FastWAM policy outside the optimization
graph. Historical v1/v2 checkpoints use a separate goal-conditioned Action
Expert. v3 instead evaluates the same frozen Base Expert and learns only a
bounded goal-conditioned correction in flow-velocity space. v4 removes that
train/deploy mismatch: it runs the Base sampler to completion, proposes one
bounded temporal correction in final action-chunk space, and uses an FP32
pairwise advantage verifier before any override of the exact Base candidate.
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


class LanguageVisualTargetBinder(nn.Module):
    """Route language through a visual-only target bottleneck.

    The language representation is used only to score current-frame visual
    patches.  Neither the Proposal query nor the deployment embedding receives
    a direct language residual: both are projected from the attention-pooled
    visual value.  This prevents a same-state language classifier from
    satisfying the grounding objective without locating a different object.
    """

    def __init__(
        self,
        *,
        text_dim: int,
        video_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        projection_dim: int = 256,
        num_heads: int = 8,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        if min(
            text_dim,
            video_dim,
            action_dim,
            hidden_dim,
            projection_dim,
            num_heads,
        ) <= 0:
            raise ValueError("PGC target-binding dimensions must be positive.")
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"PGC target-binding hidden_dim={hidden_dim} must divide "
                f"num_heads={num_heads}."
            )
        if temperature <= 0:
            raise ValueError("PGC target-binding temperature must be positive.")

        self.text_dim = int(text_dim)
        self.video_dim = int(video_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.projection_dim = int(projection_dim)
        self.num_heads = int(num_heads)
        self.temperature = float(temperature)

        self.target_language_seed = nn.Parameter(
            torch.randn(1, 1, hidden_dim) / math.sqrt(hidden_dim)
        )
        self.text_projection = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, hidden_dim),
        )
        self.language_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.language_norm = nn.LayerNorm(hidden_dim)
        self.visual_projection = nn.Sequential(
            nn.LayerNorm(video_dim),
            nn.Linear(video_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.binding_query_projection = nn.Sequential(
            nn.LayerNorm(video_dim),
            nn.Linear(video_dim, action_dim),
        )
        nn.init.zeros_(self.binding_query_projection[-1].weight)
        nn.init.zeros_(self.binding_query_projection[-1].bias)
        self.binding_embedding_projection = nn.Sequential(
            nn.LayerNorm(video_dim),
            nn.Linear(video_dim, projection_dim),
        )

    def forward(
        self,
        *,
        base_queries: torch.Tensor,
        language_hidden: torch.Tensor,
        language_mask: torch.Tensor,
        current_video_hidden: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        if (
            base_queries.ndim != 3
            or language_hidden.ndim != 3
            or current_video_hidden.ndim != 3
        ):
            raise ValueError("PGC target-binding inputs must be 3D tensors.")
        if language_mask.shape != language_hidden.shape[:2]:
            raise ValueError("PGC target-binding language mask must be [B,L].")
        if not (
            base_queries.shape[0]
            == language_hidden.shape[0]
            == current_video_hidden.shape[0]
        ):
            raise ValueError("PGC target-binding inputs must share batch size.")
        if base_queries.shape[-1] != self.action_dim:
            raise ValueError("PGC target-binding base-query dimension mismatch.")
        if language_hidden.shape[-1] != self.text_dim:
            raise ValueError("PGC target-binding language dimension mismatch.")
        if current_video_hidden.shape[-1] != self.video_dim:
            raise ValueError("PGC target-binding video dimension mismatch.")
        if current_video_hidden.shape[1] <= 0:
            raise ValueError("PGC target binding requires visual patches.")

        text = self.text_projection(language_hidden)
        text, text_padding = _safe_key_padding_mask(text, language_mask)
        seed = self.target_language_seed.expand(language_hidden.shape[0], -1, -1)
        language_delta, language_attention = self.language_attention(
            query=seed,
            key=text,
            value=text,
            key_padding_mask=text_padding,
            need_weights=True,
        )
        language_query = F.normalize(
            self.language_norm(seed + language_delta).float(),
            dim=-1,
            eps=1.0e-6,
        )[:, 0]
        visual_features = F.normalize(
            self.visual_projection(current_video_hidden).float(),
            dim=-1,
            eps=1.0e-6,
        )
        similarity = torch.einsum(
            "bd,bnd->bn", language_query, visual_features
        )
        target_attention = torch.softmax(
            similarity / self.temperature, dim=-1
        )

        # This pooled visual tensor is the only deployment path from language
        # to the Proposal.  Do not add language_query as a residual here.
        pooled_visual = torch.einsum(
            "bn,bnd->bd",
            target_attention.to(dtype=current_video_hidden.dtype),
            current_video_hidden,
        )
        visual_query_delta = self.binding_query_projection(pooled_visual).unsqueeze(1)
        binding_queries = base_queries + visual_query_delta
        binding_embedding = F.normalize(
            self.binding_embedding_projection(pooled_visual).float(),
            dim=-1,
            eps=1.0e-6,
        )

        visual_probs = target_attention.float().clamp_min(1.0e-8)
        language_probs = language_attention.float().clamp_min(1.0e-8)
        metrics = {
            "pgc_v6_target_attention_entropy": (
                -(visual_probs * visual_probs.log()).sum(dim=-1)
                / math.log(max(2, visual_probs.shape[-1]))
            ).mean(),
            "pgc_v6_target_attention_top1_mass": visual_probs.max(
                dim=-1
            ).values.mean(),
            "pgc_v6_target_similarity_max": similarity.max(dim=-1).values.mean(),
            "pgc_v6_target_language_attention_entropy": (
                -(language_probs * language_probs.log()).sum(dim=-1)
                / math.log(max(2, language_probs.shape[-1]))
            ).mean(),
            "pgc_v6_target_binding_query_norm": binding_queries.float()
            .norm(dim=-1)
            .mean(),
            "pgc_v6_target_binding_query_delta_norm": visual_query_delta.float()
            .norm(dim=-1)
            .mean(),
            "pgc_v6_target_binding_embedding_norm": binding_embedding.norm(
                dim=-1
            ).mean(),
        }
        return (
            binding_queries,
            binding_embedding,
            target_attention,
            visual_features,
            metrics,
        )


def infer_spatial_patch_grid(
    token_count: int, *, aspect_ratio: float
) -> tuple[int, int]:
    """Infer the rectangular patch grid while preserving token order."""
    token_count = int(token_count)
    aspect_ratio = float(aspect_ratio)
    if token_count <= 0 or aspect_ratio <= 0:
        raise ValueError("Patch token count and aspect ratio must be positive.")
    candidates: list[tuple[float, int, int]] = []
    for height in range(1, int(math.sqrt(token_count)) + 1):
        if token_count % height:
            continue
        width = token_count // height
        for candidate_height, candidate_width in (
            (height, width),
            (width, height),
        ):
            error = abs(
                math.log((candidate_width / candidate_height) / aspect_ratio)
            )
            candidates.append((error, candidate_height, candidate_width))
    if not candidates:
        raise ValueError(f"Cannot factor patch token count {token_count}.")
    _, grid_height, grid_width = min(candidates)
    return int(grid_height), int(grid_width)


def spatial_mask_to_patch_distribution(
    mask: torch.Tensor,
    *,
    token_count: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Area-resample a current-frame object mask onto visual patch tokens."""
    if mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask[:, 0]
    if mask.ndim != 3:
        raise ValueError("PGC object masks must be [B,H,W] or [B,1,H,W].")
    if mask.shape[-2] <= 0 or mask.shape[-1] <= 0:
        raise ValueError("PGC object masks require positive spatial dimensions.")
    aspect_ratio = float(mask.shape[-1]) / float(mask.shape[-2])
    grid_height, grid_width = infer_spatial_patch_grid(
        int(token_count), aspect_ratio=aspect_ratio
    )
    patch_mass = F.interpolate(
        mask.float().unsqueeze(1),
        size=(grid_height, grid_width),
        mode="area",
    ).flatten(1)
    total_mass = patch_mass.sum(dim=-1)
    valid = total_mass > 1.0e-6
    uniform = torch.full_like(patch_mass, 1.0 / max(1, int(token_count)))
    distribution = torch.where(
        valid.unsqueeze(-1),
        patch_mass / total_mass.clamp_min(1.0e-6).unsqueeze(-1),
        uniform,
    )
    metrics = {
        "pgc_v7_mask_valid_fraction": valid.float().mean(),
        "pgc_v7_mask_patch_fraction": (patch_mass > 0).float().mean(),
        "pgc_v7_mask_patch_mass": patch_mass.mean(),
        "pgc_v7_mask_grid_height": patch_mass.new_tensor(float(grid_height)),
        "pgc_v7_mask_grid_width": patch_mass.new_tensor(float(grid_width)),
    }
    return distribution, valid, metrics


class SpatialObjectTokenTargetBinder(nn.Module):
    """PGC V7 language-to-object binding with explicit spatial object tokens.

    Language only selects current-frame visual patches. The selected patch
    values, 2-D coordinates, and camera identities form a small object-token
    set. Every action query independently cross-attends to that set; there is
    no broadcast language residual and no direct text edge to the Proposal.
    """

    def __init__(
        self,
        *,
        text_dim: int,
        video_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        projection_dim: int = 256,
        num_heads: int = 8,
        num_object_tokens: int = 8,
        camera_count: int = 2,
        visual_aspect_ratio: float = 2.0,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        if min(
            text_dim,
            video_dim,
            action_dim,
            hidden_dim,
            projection_dim,
            num_heads,
            num_object_tokens,
            camera_count,
        ) <= 0:
            raise ValueError("PGC V7 binder dimensions must be positive.")
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"PGC V7 hidden_dim={hidden_dim} must divide heads={num_heads}."
            )
        if visual_aspect_ratio <= 0 or temperature <= 0:
            raise ValueError("PGC V7 aspect ratio and temperature must be positive.")
        self.text_dim = int(text_dim)
        self.video_dim = int(video_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.projection_dim = int(projection_dim)
        self.num_heads = int(num_heads)
        self.num_object_tokens = int(num_object_tokens)
        self.camera_count = int(camera_count)
        self.visual_aspect_ratio = float(visual_aspect_ratio)
        self.temperature = float(temperature)

        self.target_language_seed = nn.Parameter(
            torch.randn(1, 1, hidden_dim) / math.sqrt(hidden_dim)
        )
        self.text_projection = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, hidden_dim),
        )
        self.language_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.language_norm = nn.LayerNorm(hidden_dim)
        self.visual_projection = nn.Sequential(
            nn.LayerNorm(video_dim),
            nn.Linear(video_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.position_projection = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.camera_embedding = nn.Embedding(camera_count, hidden_dim)
        self.visual_norm = nn.LayerNorm(hidden_dim)
        self.action_query_projection = nn.Sequential(
            nn.LayerNorm(action_dim),
            nn.Linear(action_dim, hidden_dim),
        )
        self.object_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.action_query_norm = nn.LayerNorm(hidden_dim)
        self.query_output_projection = nn.Linear(hidden_dim, action_dim)
        nn.init.zeros_(self.query_output_projection.weight)
        nn.init.zeros_(self.query_output_projection.bias)
        self.binding_embedding_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, projection_dim),
        )

    def _spatial_features(
        self, current_video_hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        token_count = int(current_video_hidden.shape[1])
        grid_height, grid_width = infer_spatial_patch_grid(
            token_count, aspect_ratio=self.visual_aspect_ratio
        )
        y = torch.linspace(
            -1.0,
            1.0,
            grid_height,
            device=current_video_hidden.device,
            dtype=current_video_hidden.dtype,
        )
        x = torch.linspace(
            -1.0,
            1.0,
            grid_width,
            device=current_video_hidden.device,
            dtype=current_video_hidden.dtype,
        )
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coordinates = torch.stack((xx, yy), dim=-1).reshape(token_count, 2)
        column_ids = torch.arange(grid_width, device=current_video_hidden.device)
        camera_ids = torch.div(
            column_ids * self.camera_count,
            grid_width,
            rounding_mode="floor",
        ).clamp(max=self.camera_count - 1)
        camera_ids = camera_ids.unsqueeze(0).expand(grid_height, -1).reshape(-1)
        visual = self.visual_projection(current_video_hidden)
        position = self.position_projection(coordinates).unsqueeze(0)
        camera = self.camera_embedding(camera_ids).unsqueeze(0)
        return self.visual_norm(visual + position + camera), coordinates, camera_ids

    def forward(
        self,
        *,
        base_queries: torch.Tensor,
        language_hidden: torch.Tensor,
        language_mask: torch.Tensor,
        current_video_hidden: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        if (
            base_queries.ndim != 3
            or language_hidden.ndim != 3
            or current_video_hidden.ndim != 3
        ):
            raise ValueError("PGC V7 binder inputs must be 3D tensors.")
        if language_mask.shape != language_hidden.shape[:2]:
            raise ValueError("PGC V7 language mask must be [B,L].")
        if not (
            base_queries.shape[0]
            == language_hidden.shape[0]
            == current_video_hidden.shape[0]
        ):
            raise ValueError("PGC V7 binder inputs must share batch size.")
        if base_queries.shape[-1] != self.action_dim:
            raise ValueError("PGC V7 base-query dimension mismatch.")
        if language_hidden.shape[-1] != self.text_dim:
            raise ValueError("PGC V7 language dimension mismatch.")
        if current_video_hidden.shape[-1] != self.video_dim:
            raise ValueError("PGC V7 video dimension mismatch.")

        text = self.text_projection(language_hidden)
        text, text_padding = _safe_key_padding_mask(text, language_mask)
        seed = self.target_language_seed.expand(language_hidden.shape[0], -1, -1)
        language_delta, language_attention = self.language_attention(
            query=seed,
            key=text,
            value=text,
            key_padding_mask=text_padding,
            need_weights=True,
        )
        language_state = self.language_norm(seed + language_delta)[:, 0]
        language_query = F.normalize(language_state.float(), dim=-1, eps=1.0e-6)
        spatial_visual, _, camera_ids = self._spatial_features(
            current_video_hidden
        )
        visual_features = F.normalize(
            spatial_visual.float(), dim=-1, eps=1.0e-6
        )
        similarity = torch.einsum("bd,bnd->bn", language_query, visual_features)
        target_attention = torch.softmax(similarity / self.temperature, dim=-1)

        selected_count = min(self.num_object_tokens, int(spatial_visual.shape[1]))
        selected_mass, selected_indices = torch.topk(
            target_attention, k=selected_count, dim=-1, sorted=True
        )
        gather_index = selected_indices.unsqueeze(-1).expand(
            -1, -1, spatial_visual.shape[-1]
        )
        object_tokens = torch.gather(spatial_visual, dim=1, index=gather_index)
        # Preserve value magnitudes while exposing relative language confidence.
        confidence = selected_mass / selected_mass.sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0e-8)
        object_tokens = object_tokens * (
            1.0 + confidence.to(object_tokens.dtype).unsqueeze(-1)
        )

        action_queries = self.action_query_projection(base_queries)
        query_delta, query_attention = self.object_attention(
            query=action_queries,
            key=object_tokens,
            value=object_tokens,
            need_weights=True,
        )
        query_delta = self.query_output_projection(
            self.action_query_norm(action_queries + query_delta)
        )
        binding_queries = base_queries + query_delta
        pooled_visual = torch.einsum(
            "bn,bnd->bd",
            target_attention.to(spatial_visual.dtype),
            spatial_visual,
        )
        binding_embedding = F.normalize(
            self.binding_embedding_projection(pooled_visual).float(),
            dim=-1,
            eps=1.0e-6,
        )

        visual_probs = target_attention.float().clamp_min(1.0e-8)
        language_probs = language_attention.float().clamp_min(1.0e-8)
        query_probs = query_attention.float().clamp_min(1.0e-8)
        selected_camera_ids = camera_ids[selected_indices]
        metrics = {
            "pgc_v7_target_attention_entropy": (
                -(visual_probs * visual_probs.log()).sum(dim=-1)
                / math.log(max(2, visual_probs.shape[-1]))
            ).mean(),
            "pgc_v7_target_attention_top1_mass": visual_probs.max(
                dim=-1
            ).values.mean(),
            "pgc_v7_target_attention_topk_mass": selected_mass.sum(
                dim=-1
            ).mean(),
            "pgc_v7_target_similarity_max": similarity.max(dim=-1).values.mean(),
            "pgc_v7_language_attention_entropy": (
                -(language_probs * language_probs.log()).sum(dim=-1)
                / math.log(max(2, language_probs.shape[-1]))
            ).mean(),
            "pgc_v7_action_query_object_attention_entropy": (
                -(query_probs * query_probs.log()).sum(dim=-1)
                / math.log(max(2, query_probs.shape[-1]))
            ).mean(),
            "pgc_v7_selected_camera_diversity": (
                selected_camera_ids.float().var(dim=-1, unbiased=False).mean()
            ),
            "pgc_v7_binding_query_delta_norm": query_delta.float()
            .norm(dim=-1)
            .mean(),
            "pgc_v7_binding_embedding_norm": binding_embedding.norm(
                dim=-1
            ).mean(),
        }
        return (
            binding_queries,
            binding_embedding,
            target_attention,
            visual_features,
            metrics,
        )


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


def _sinusoidal_positions(
    length: int,
    dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return deterministic temporal positions without a horizon limit."""
    if length <= 0 or dim <= 0:
        raise ValueError("Temporal position length/dimension must be positive.")
    positions = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    frequencies = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / max(1, dim))
    )
    encoding = torch.zeros(length, dim, device=device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    if dim > 1:
        encoding[:, 1::2] = torch.cos(
            positions * frequencies[: encoding[:, 1::2].shape[1]]
        )
    return encoding.to(dtype=dtype).unsqueeze(0)


def _functional_call_fp32(module: nn.Module, *args: Any, **kwargs: Any) -> Any:
    """Run a small sidecar in FP32 without mutating ZeRO-owned parameters.

    DeepSpeed BF16 preparation may cast every parameter in the wrapped model,
    including a verifier that was explicitly constructed in FP32.  Casting
    only the verifier inputs back to FP32 then makes normalization layers fail
    because their weights remain BF16.  A stateless FP32 view keeps the
    verifier computation numerically stable while gradients still flow through
    the casts to the original parameters managed by the optimizer.
    """
    parameters_and_buffers = {
        name: parameter.float()
        for name, parameter in module.named_parameters()
    }
    parameters_and_buffers.update(
        {
            name: buffer.float() if buffer.is_floating_point() else buffer
            for name, buffer in module.named_buffers()
        }
    )
    try:
        device_type = next(module.parameters()).device.type
    except StopIteration as error:
        raise ValueError(
            "FP32 functional call requires module parameters."
        ) from error
    with torch.autocast(device_type=device_type, enabled=False):
        return torch.func.functional_call(
            module,
            parameters_and_buffers,
            args,
            kwargs,
        )


class RolloutAlignedActionProposal(nn.Module):
    """Correct a fully denoised Base action chunk in deployment space.

    Unlike the v3 velocity residual, this module is applied exactly once after
    the frozen Base sampler has completed. Training and inference therefore see
    the same kind of action candidate. The final projection is zero initialized
    so constructing v4 preserves the released policy bit-for-bit until the hard
    gate explicitly selects a learned proposal.
    """

    def __init__(
        self,
        *,
        action_dim: int,
        goal_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        max_abs: float | Sequence[float] = 2.0,
    ):
        super().__init__()
        if min(action_dim, goal_dim, hidden_dim, num_heads, num_layers) <= 0:
            raise ValueError("PGC v4 proposal dimensions/layers must be positive.")
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"PGC v4 proposal hidden_dim={hidden_dim} must divide "
                f"num_heads={num_heads}."
            )
        if isinstance(max_abs, Sequence) and not isinstance(max_abs, (str, bytes)):
            cap_values = [float(value) for value in max_abs]
            if len(cap_values) != int(action_dim):
                raise ValueError(
                    "PGC v4 per-dimension action caps must match action_dim: "
                    f"{len(cap_values)} vs {action_dim}."
                )
        else:
            cap_values = [float(max_abs)] * int(action_dim)
        if any(value <= 0 for value in cap_values):
            raise ValueError("PGC v4 action residual caps must be positive.")

        self.action_dim = int(action_dim)
        self.goal_dim = int(goal_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.action_projection = nn.Sequential(
            nn.LayerNorm(action_dim),
            nn.Linear(action_dim, hidden_dim),
        )
        self.goal_projection = nn.Sequential(
            nn.LayerNorm(goal_dim),
            nn.Linear(goal_dim, hidden_dim),
        )
        self.goal_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, action_dim)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        self.register_buffer(
            "max_abs",
            torch.tensor(cap_values, dtype=torch.float32).view(1, 1, action_dim),
            persistent=True,
        )

    def forward(
        self,
        *,
        base_action: torch.Tensor,
        goal_queries: torch.Tensor,
        action_is_pad: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if base_action.ndim != 3 or base_action.shape[-1] != self.action_dim:
            raise ValueError(
                "PGC v4 base action must be [B,T,A] with the configured "
                f"action_dim={self.action_dim}."
            )
        if goal_queries.ndim != 3 or goal_queries.shape[-1] != self.goal_dim:
            raise ValueError(
                "PGC v4 goal queries must be [B,K,D] with the configured "
                f"goal_dim={self.goal_dim}."
            )
        if base_action.shape[0] != goal_queries.shape[0]:
            raise ValueError("PGC v4 proposal inputs must share a batch dimension.")
        if action_is_pad is not None and action_is_pad.shape != base_action.shape[:2]:
            raise ValueError("PGC v4 action padding mask must be [B,T].")

        action = self.action_projection(base_action.detach())
        action = action + _sinusoidal_positions(
            int(action.shape[1]),
            int(action.shape[2]),
            device=action.device,
            dtype=action.dtype,
        )
        goal = self.goal_projection(goal_queries)
        goal_delta, attention = self.goal_attention(
            query=action,
            key=goal,
            value=goal,
            need_weights=True,
        )
        hidden = action + goal_delta
        output_padding = None
        encoder_padding = None
        if action_is_pad is not None:
            output_padding = action_is_pad.to(
                device=hidden.device, dtype=torch.bool
            )
            encoder_padding = output_padding.clone()
            empty = encoder_padding.all(dim=1)
            if bool(empty.any()):
                encoder_padding[empty, 0] = False
                hidden = hidden.clone()
                hidden[empty, 0] = 0
        hidden = self.temporal_encoder(
            hidden, src_key_padding_mask=encoder_padding
        )
        raw_residual = self.output_projection(self.output_norm(hidden))
        cap = self.max_abs.to(
            device=raw_residual.device, dtype=raw_residual.dtype
        )
        residual = torch.tanh(raw_residual) * cap
        if output_padding is not None:
            residual = residual.masked_fill(output_padding.unsqueeze(-1), 0.0)
        proposal = base_action.detach() + residual

        probabilities = attention.float().clamp_min(1.0e-8)
        entropy = -(probabilities * probabilities.log()).sum(dim=-1)
        entropy = entropy / math.log(max(2, probabilities.shape[-1]))
        saturation = residual.float().abs() >= cap.float() * 0.95
        per_sample_rms = residual.float().square().mean(dim=(1, 2)).sqrt()
        metrics = {
            "pgc_v4_action_residual_norm": residual.float().norm(dim=-1).mean(),
            "pgc_v4_action_residual_rms": per_sample_rms.mean(),
            "pgc_v4_action_residual_max_abs": residual.float().abs().max(),
            "pgc_v4_action_residual_saturation_fraction": saturation.float().mean(),
            "pgc_v4_action_residual_attention_entropy": entropy.mean(),
            "pgc_v4_action_residual_cap_mean": cap.float().mean(),
        }
        return proposal, residual, metrics


class PairwiseActionAdvantageVerifier(nn.Module):
    """FP32 temporal verifier that predicts CF advantage over Base directly."""

    def __init__(
        self,
        *,
        action_dim: int,
        video_dim: int,
        goal_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
    ):
        super().__init__()
        if min(
            action_dim,
            video_dim,
            goal_dim,
            hidden_dim,
            num_heads,
            num_layers,
        ) <= 0:
            raise ValueError("PGC v4 verifier dimensions/layers must be positive.")
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"PGC v4 verifier hidden_dim={hidden_dim} must divide "
                f"num_heads={num_heads}."
            )
        self.action_dim = int(action_dim)
        self.video_dim = int(video_dim)
        self.goal_dim = int(goal_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.action_projection = nn.Linear(action_dim, hidden_dim)
        action_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.action_encoder = nn.TransformerEncoder(
            action_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.action_pool_query = nn.Parameter(
            torch.randn(1, 1, hidden_dim) / math.sqrt(hidden_dim)
        )
        self.action_pool = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.visual_encoder = nn.Sequential(
            nn.LayerNorm(video_dim),
            nn.Linear(video_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
        )
        self.goal_encoder = nn.Sequential(
            nn.LayerNorm(goal_dim),
            nn.Linear(goal_dim, hidden_dim),
        )
        self.goal_state_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, 1),
        )

    def encode_action(
        self,
        action: torch.Tensor,
        action_is_pad: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if action.ndim != 3 or action.shape[-1] != self.action_dim:
            raise ValueError(
                f"PGC v4 verifier action must be [B,T,{self.action_dim}]."
            )
        action = action.float()
        hidden = _functional_call_fp32(self.action_projection, action)
        hidden = hidden + _sinusoidal_positions(
            int(hidden.shape[1]),
            int(hidden.shape[2]),
            device=hidden.device,
            dtype=hidden.dtype,
        )
        padding = None
        if action_is_pad is not None:
            if action_is_pad.shape != action.shape[:2]:
                raise ValueError("PGC v4 verifier padding mask must be [B,T].")
            padding = action_is_pad.to(
                device=hidden.device, dtype=torch.bool
            ).clone()
            empty = padding.all(dim=1)
            if bool(empty.any()):
                padding[empty, 0] = False
                hidden = hidden.clone()
                hidden[empty, 0] = 0
        hidden = _functional_call_fp32(
            self.action_encoder,
            hidden,
            src_key_padding_mask=padding,
        )
        query = self.action_pool_query.float().expand(hidden.shape[0], -1, -1)
        pooled, _ = _functional_call_fp32(
            self.action_pool,
            query=query,
            key=hidden,
            value=hidden,
            key_padding_mask=padding,
            need_weights=False,
        )
        return F.normalize(pooled[:, 0].float(), dim=-1, eps=1.0e-6)

    def encode_goal_state(
        self,
        current_video_hidden: torch.Tensor,
        goal_embedding: torch.Tensor,
    ) -> torch.Tensor:
        if current_video_hidden.ndim != 3 or (
            current_video_hidden.shape[-1] != self.video_dim
        ):
            raise ValueError("PGC v4 verifier visual tokens have invalid shape.")
        if goal_embedding.ndim != 2 or goal_embedding.shape[-1] != self.goal_dim:
            raise ValueError("PGC v4 verifier goal embedding has invalid shape.")
        visual = _functional_call_fp32(
            self.visual_encoder, current_video_hidden.float()
        ).mean(dim=1)
        goal = _functional_call_fp32(self.goal_encoder, goal_embedding.float())
        fused = _functional_call_fp32(
            self.goal_state_fusion, torch.cat([visual, goal], dim=-1)
        )
        return F.normalize(fused.float(), dim=-1, eps=1.0e-6)

    def score_candidate(
        self,
        goal_state: torch.Tensor,
        action_embedding: torch.Tensor,
    ) -> torch.Tensor:
        if goal_state.shape != action_embedding.shape or goal_state.ndim != 2:
            raise ValueError("PGC v4 verifier embeddings must share [B,D] shape.")
        interaction = torch.cat(
            [
                goal_state.float(),
                action_embedding.float(),
                (goal_state.float() - action_embedding.float()).abs(),
                goal_state.float() * action_embedding.float(),
            ],
            dim=-1,
        )
        return _functional_call_fp32(self.value_head, interaction).squeeze(-1)

    def forward(
        self,
        *,
        current_video_hidden: torch.Tensor,
        goal_embedding: torch.Tensor,
        base_action: torch.Tensor,
        counterfactual_action: torch.Tensor,
        action_is_pad: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        goal_state = self.encode_goal_state(current_video_hidden, goal_embedding)
        base_embedding = self.encode_action(base_action, action_is_pad)
        counterfactual_embedding = self.encode_action(
            counterfactual_action, action_is_pad
        )
        base_value = self.score_candidate(goal_state, base_embedding)
        counterfactual_value = self.score_candidate(
            goal_state, counterfactual_embedding
        )
        advantage = counterfactual_value - base_value
        return (
            advantage.float(),
            base_value.float(),
            counterfactual_value.float(),
            goal_state,
            base_embedding,
            counterfactual_embedding,
        )


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
