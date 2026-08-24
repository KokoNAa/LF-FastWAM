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
from typing import Any, Mapping, Sequence

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


class PhaseConditionedERAFActionBridge(nn.Module):
    """Route explicit, phase-conditioned ERAF geometry into Proposal queries.

    Subject, reference, and relation evidence remain distinct, and grasp,
    goal, interaction, and displacement geometry have direct paths.  The final
    projection is zero initialized, making a new V9.15 bridge exactly V9.14.
    """

    ROLE_COUNT = 3

    def __init__(
        self,
        *,
        goal_dim: int,
        eraf_hidden_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        max_clauses: int = 4,
    ):
        super().__init__()
        if min(
            goal_dim, eraf_hidden_dim, hidden_dim, num_heads, max_clauses
        ) <= 0:
            raise ValueError("V9.15 action-grounding dimensions must be positive.")
        if hidden_dim % num_heads != 0:
            raise ValueError(
                "V9.15 action-grounding hidden_dim must be divisible by "
                "num_heads."
            )
        self.goal_dim = int(goal_dim)
        self.eraf_hidden_dim = int(eraf_hidden_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_clauses = int(max_clauses)
        coordinate_dim = 3 * 5
        self.subject_projection = nn.Sequential(
            nn.LayerNorm(eraf_hidden_dim),
            nn.Linear(eraf_hidden_dim, hidden_dim),
        )
        self.reference_projection = nn.Sequential(
            nn.LayerNorm(eraf_hidden_dim),
            nn.Linear(eraf_hidden_dim, hidden_dim),
        )
        self.relation_projection = nn.Sequential(
            nn.LayerNorm(eraf_hidden_dim),
            nn.Linear(eraf_hidden_dim, hidden_dim),
        )
        self.grasp_geometry_projection = nn.Linear(
            coordinate_dim * 2, hidden_dim
        )
        self.goal_geometry_projection = nn.Linear(
            coordinate_dim * 2, hidden_dim
        )
        self.relation_geometry_projection = nn.Linear(
            coordinate_dim * 2, hidden_dim
        )
        self.phase_projection = nn.Linear(3, hidden_dim, bias=False)
        self.truth_projection = nn.Linear(1, hidden_dim, bias=False)
        self.role_embedding = nn.Parameter(
            torch.empty(self.ROLE_COUNT, hidden_dim)
        )
        nn.init.normal_(self.role_embedding, std=hidden_dim**-0.5)
        self.clause_embedding = nn.Parameter(
            torch.empty(max_clauses, hidden_dim)
        )
        nn.init.normal_(self.clause_embedding, std=hidden_dim**-0.5)
        self.query_projection = nn.Sequential(
            nn.LayerNorm(goal_dim),
            nn.Linear(goal_dim, hidden_dim),
        )
        self.role_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.query_delta_projection = nn.Linear(hidden_dim, goal_dim)
        nn.init.zeros_(self.query_delta_projection.weight)
        nn.init.zeros_(self.query_delta_projection.bias)

    @staticmethod
    def _coordinate_features(coordinates: torch.Tensor) -> torch.Tensor:
        value = coordinates.float().clamp(-2.0, 2.0)
        features = [value]
        for frequency in (1.0, 2.0):
            angle = math.pi * frequency * value
            features.extend((angle.sin(), angle.cos()))
        return torch.cat(features, dim=-1).to(coordinates.dtype)

    def forward(
        self,
        *,
        goal_queries: torch.Tensor,
        eraf_outputs: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        required = {
            "active_logits",
            "subject_token",
            "reference_token",
            "subject_position",
            "reference_position",
            "grasp_anchor",
            "goal_anchor",
            "interaction_anchor",
            "relation_hidden",
            "predicate_truth_logits",
            "phase_logits",
            "clause_execution_probability",
        }
        missing = sorted(required - set(eraf_outputs))
        if missing:
            raise ValueError(
                f"V9.15 action grounding is missing ERAF outputs: {missing}."
            )
        if goal_queries.ndim != 3 or goal_queries.shape[-1] != self.goal_dim:
            raise ValueError("V9.15 goal queries must be [B,K,goal_dim].")
        subject = eraf_outputs["subject_token"]
        reference = eraf_outputs["reference_token"]
        relation = eraf_outputs["relation_hidden"]
        if subject.shape != reference.shape or subject.shape != relation.shape:
            raise ValueError("V9.15 ERAF role tokens must have identical shapes.")
        if subject.ndim != 3 or subject.shape[-1] != self.eraf_hidden_dim:
            raise ValueError("V9.15 ERAF role token dimensions are invalid.")
        if subject.shape[1] != self.max_clauses:
            raise ValueError(
                "V9.15 ERAF clause count does not match the configured maximum."
            )
        if subject.shape[0] != goal_queries.shape[0]:
            raise ValueError("V9.15 ERAF and goal-query batches must match.")

        subject_position = eraf_outputs["subject_position"]
        reference_position = eraf_outputs["reference_position"]
        grasp_anchor = eraf_outputs["grasp_anchor"]
        goal_anchor = eraf_outputs["goal_anchor"]
        interaction_anchor = eraf_outputs["interaction_anchor"]
        displacement = goal_anchor.float() - grasp_anchor.float()
        subject_geometry = torch.cat(
            (
                self._coordinate_features(subject_position),
                self._coordinate_features(grasp_anchor),
            ),
            dim=-1,
        ).to(subject.dtype)
        reference_geometry = torch.cat(
            (
                self._coordinate_features(reference_position),
                self._coordinate_features(goal_anchor),
            ),
            dim=-1,
        ).to(subject.dtype)
        relation_geometry = torch.cat(
            (
                self._coordinate_features(interaction_anchor),
                self._coordinate_features(displacement),
            ),
            dim=-1,
        ).to(subject.dtype)
        phase_probability = torch.softmax(
            eraf_outputs["phase_logits"].float(), dim=-1
        ).to(subject.dtype)
        phase = self.phase_projection(phase_probability)
        truth = self.truth_projection(
            torch.sigmoid(eraf_outputs["predicate_truth_logits"].float())
            .to(subject.dtype)
            .unsqueeze(-1)
        )
        tokens = torch.stack(
            (
                self.subject_projection(subject)
                + self.grasp_geometry_projection(subject_geometry)
                + self.role_embedding[0],
                self.reference_projection(reference)
                + self.goal_geometry_projection(reference_geometry)
                + self.role_embedding[1],
                self.relation_projection(relation)
                + self.relation_geometry_projection(relation_geometry)
                + truth
                + self.role_embedding[2],
            ),
            dim=2,
        )
        tokens = (
            tokens
            + phase.unsqueeze(2)
            + self.clause_embedding[None, :, None, :]
        )
        routing = eraf_outputs["clause_execution_probability"].float()
        if routing.shape != subject.shape[:2]:
            raise ValueError("V9.15 clause routing shape is invalid.")
        active = torch.sigmoid(eraf_outputs["active_logits"].float())
        routing = routing * active
        no_route = routing.sum(dim=-1, keepdim=True) <= 1.0e-6
        fallback = torch.softmax(eraf_outputs["active_logits"].float(), dim=-1)
        routing = torch.where(no_route, fallback, routing)
        tokens = tokens * routing.to(tokens.dtype).unsqueeze(-1).unsqueeze(-1)
        tokens = tokens.reshape(tokens.shape[0], -1, tokens.shape[-1])
        valid_clause = active > 0.5
        valid_token = valid_clause.unsqueeze(-1).expand(
            -1, -1, self.ROLE_COUNT
        ).reshape(valid_clause.shape[0], -1)
        safe_tokens, key_padding_mask = _safe_key_padding_mask(
            tokens, valid_token
        )
        query_delta, attention = self.role_attention(
            query=self.query_projection(goal_queries),
            key=safe_tokens,
            value=safe_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=True,
        )
        projected_delta = self.query_delta_projection(query_delta).to(
            goal_queries.dtype
        )
        routed_queries = goal_queries + projected_delta
        probabilities = attention.float().clamp_min(1.0e-8)
        entropy = -(probabilities * probabilities.log()).sum(dim=-1)
        entropy = entropy / math.log(max(2, probabilities.shape[-1]))
        metrics = {
            "pgc_v915_action_grounding_query_delta_rms": (
                projected_delta.float().square().mean().sqrt()
            ),
            "pgc_v915_action_grounding_attention_entropy": entropy.mean(),
            "pgc_v915_action_grounding_goal_anchor_norm": (
                goal_anchor.float().norm(dim=-1).mean()
            ),
            "pgc_v915_action_grounding_displacement_norm": (
                displacement.float().norm(dim=-1).mean()
            ),
        }
        return routed_queries, metrics


class PhaseConditionedERAFGeometryActionAdapter(nn.Module):
    """Apply ERAF geometry directly to a deployed action chunk.

    V9.15/V9.16 route geometry through goal queries and two attention
    bottlenecks before it reaches the Proposal.  This adapter provides a short
    causal path from phase-selected, EEF-relative anchors to every action
    timestep.  Its final projection is zero initialized, so a fresh V9.17
    module is exactly equivalent to V9.16.

    LIBERO proprio is normalized independently from ERAF's canonical workspace.
    A small learned calibration maps the current proprio vector into the same
    bounded coordinate frame before relative vectors are formed.  This keeps
    the interface deployment-only (RGB, language, proprio) and also supports
    benchmarks with a different proprio dimension.
    """

    def __init__(
        self,
        *,
        action_dim: int,
        proprio_dim: int,
        hidden_dim: int = 256,
        max_clauses: int = 4,
        max_abs: float | Sequence[float] = 0.25,
    ) -> None:
        super().__init__()
        if min(action_dim, proprio_dim, hidden_dim, max_clauses) <= 0:
            raise ValueError("V9.17 geometry-action dimensions must be positive.")
        if isinstance(max_abs, Sequence) and not isinstance(max_abs, (str, bytes)):
            cap_values = [float(value) for value in max_abs]
            if len(cap_values) != int(action_dim):
                raise ValueError(
                    "V9.17 per-dimension residual caps must match action_dim."
                )
        else:
            cap_values = [float(max_abs)] * int(action_dim)
        if any(value <= 0 for value in cap_values):
            raise ValueError("V9.17 geometry residual caps must be positive.")

        self.action_dim = int(action_dim)
        self.proprio_dim = int(proprio_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_clauses = int(max_clauses)
        self.eef_position_projection = nn.Sequential(
            nn.LayerNorm(self.proprio_dim),
            nn.Linear(self.proprio_dim, self.hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.hidden_dim, 3),
            nn.Tanh(),
        )
        # Four three-dimensional relative vectors, each expanded to
        # [x, sin(pi*x), cos(pi*x), sin(2*pi*x), cos(2*pi*x)].  Phase, truth,
        # and two predicted visibility confidences remain explicit scalars.
        geometry_dim = 4 * 3 * 5 + 3 + 1 + 2
        self.geometry_projection = nn.Sequential(
            nn.LayerNorm(geometry_dim),
            nn.Linear(geometry_dim, self.hidden_dim * 2),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )
        self.clause_embedding = nn.Parameter(
            torch.empty(self.max_clauses, self.hidden_dim)
        )
        nn.init.normal_(self.clause_embedding, std=self.hidden_dim**-0.5)
        self.action_projection = nn.Sequential(
            nn.LayerNorm(self.action_dim),
            nn.Linear(self.action_dim, self.hidden_dim),
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.output_projection = nn.Linear(self.hidden_dim, self.action_dim)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        self.register_buffer(
            "max_abs",
            torch.tensor(cap_values, dtype=torch.float32).view(1, 1, -1),
            persistent=True,
        )

    @staticmethod
    def _coordinate_features(coordinates: torch.Tensor) -> torch.Tensor:
        value = coordinates.float().clamp(-2.0, 2.0)
        features = [value]
        for frequency in (1.0, 2.0):
            angle = math.pi * frequency * value
            features.extend((angle.sin(), angle.cos()))
        return torch.cat(features, dim=-1).to(coordinates.dtype)

    def forward(
        self,
        *,
        candidate_action: torch.Tensor,
        eraf_outputs: Mapping[str, torch.Tensor],
        proprio: torch.Tensor,
        action_is_pad: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        required = {
            "active_logits",
            "subject_position",
            "reference_position",
            "grasp_anchor",
            "goal_anchor",
            "interaction_anchor",
            "phase_logits",
            "predicate_truth_logits",
            "clause_execution_probability",
            "subject_visibility_logits",
            "reference_visibility_logits",
        }
        missing = sorted(required - set(eraf_outputs))
        if missing:
            raise ValueError(
                f"V9.17 geometry-action adapter is missing ERAF outputs: {missing}."
            )
        if candidate_action.ndim != 3 or candidate_action.shape[-1] != self.action_dim:
            raise ValueError("V9.17 candidate action must be [B,T,action_dim].")
        if proprio.ndim != 2 or proprio.shape != (
            candidate_action.shape[0],
            self.proprio_dim,
        ):
            raise ValueError(
                "V9.17 proprio must be [B,proprio_dim] and match the action batch."
            )
        if action_is_pad is not None and action_is_pad.shape != candidate_action.shape[:2]:
            raise ValueError("V9.17 action padding mask must be [B,T].")
        bypass = eraf_outputs.get("audit_bypass_bridge")
        if bypass is not None and bool(torch.as_tensor(bypass).all()):
            residual = torch.zeros_like(candidate_action)
            zero = residual.float().sum()
            return candidate_action, residual, {
                "pgc_v917_geometry_action_residual_rms": zero,
                "pgc_v917_geometry_action_residual_max_abs": zero,
                "pgc_v917_geometry_route_confidence": zero,
                "pgc_v917_geometry_eef_position_norm": zero,
                "pgc_v917_geometry_goal_relative_norm": zero,
            }

        subject_position = eraf_outputs["subject_position"]
        reference_position = eraf_outputs["reference_position"]
        grasp_anchor = eraf_outputs["grasp_anchor"]
        goal_anchor = eraf_outputs["goal_anchor"]
        interaction_anchor = eraf_outputs["interaction_anchor"]
        clause_shape = subject_position.shape[:2]
        expected_position_shape = (*clause_shape, 3)
        if clause_shape != (candidate_action.shape[0], self.max_clauses):
            raise ValueError("V9.17 ERAF clause shape is invalid.")
        if any(
            value.shape != expected_position_shape
            for value in (
                subject_position,
                reference_position,
                grasp_anchor,
                goal_anchor,
                interaction_anchor,
            )
        ):
            raise ValueError("V9.17 ERAF anchor/position tensors must be [B,C,3].")

        module_dtype = self.action_projection[1].weight.dtype
        # Newly introduced trainable sidecars remain FP32 under the ZeRO2
        # bf16 training path while frozen ERAF outputs arrive as bf16.  Keep
        # every normalization/projection in the adapter parameter dtype and
        # cast only the bounded residual back to the deployed action dtype.
        # In particular, nn.LayerNorm requires its input and affine
        # parameters to agree outside autocast.
        subject_position = subject_position.to(dtype=module_dtype)
        reference_position = reference_position.to(dtype=module_dtype)
        grasp_anchor = grasp_anchor.to(dtype=module_dtype)
        goal_anchor = goal_anchor.to(dtype=module_dtype)
        interaction_anchor = interaction_anchor.to(dtype=module_dtype)
        eef_position = self.eef_position_projection(
            proprio.to(dtype=module_dtype)
        )
        eef_position = eef_position.unsqueeze(1).expand(-1, self.max_clauses, -1)
        phase = torch.softmax(eraf_outputs["phase_logits"].float(), dim=-1).to(
            module_dtype
        )
        if phase.shape != (*clause_shape, 3):
            raise ValueError("V9.17 phase probabilities must be [B,C,3].")
        # Preserve a small cross-phase signal while emphasizing the geometry
        # relevant to approach, transport, and release respectively.
        approach_scale = 0.1 + 0.9 * phase[..., 0:1]
        transport_scale = 0.1 + 0.9 * phase[..., 1:2]
        release_scale = 0.1 + 0.9 * phase[..., 2:3]
        grasp_relative = (grasp_anchor - eef_position) * approach_scale
        # Once grasped, the controlled object follows the end effector and may
        # be visually occluded.  Use the EEF as the transport origin so the
        # placement vector remains meaningful after pickup.
        goal_relative = (goal_anchor - eef_position) * transport_scale
        interaction_relative = (interaction_anchor - eef_position) * release_scale
        relation_relative = reference_position - subject_position
        truth = torch.sigmoid(
            eraf_outputs["predicate_truth_logits"].float()
        ).to(module_dtype).unsqueeze(-1)
        subject_visibility = torch.sigmoid(
            eraf_outputs["subject_visibility_logits"].float()
        ).to(module_dtype).unsqueeze(-1)
        reference_visibility = torch.sigmoid(
            eraf_outputs["reference_visibility_logits"].float()
        ).to(module_dtype).unsqueeze(-1)
        geometry = torch.cat(
            (
                self._coordinate_features(grasp_relative),
                self._coordinate_features(goal_relative),
                self._coordinate_features(interaction_relative),
                self._coordinate_features(relation_relative),
                phase,
                truth,
                subject_visibility,
                reference_visibility,
            ),
            dim=-1,
        )
        clause_tokens = self.geometry_projection(geometry) + self.clause_embedding

        active = torch.sigmoid(eraf_outputs["active_logits"].float())
        execution = eraf_outputs["clause_execution_probability"].float()
        if active.shape != clause_shape or execution.shape != clause_shape:
            raise ValueError("V9.17 active/execution routing must be [B,C].")
        # Predicted visibility is a soft confidence, never a hard occlusion cut.
        visibility = 0.25 + 0.375 * (
            subject_visibility.squeeze(-1) + reference_visibility.squeeze(-1)
        )
        routing = execution.clamp_min(0.0) * active * visibility
        routing_sum = routing.sum(dim=-1, keepdim=True)
        normalized_routing = routing / routing_sum.clamp_min(1.0e-6)
        geometry_context = (
            clause_tokens * normalized_routing.to(clause_tokens.dtype).unsqueeze(-1)
        ).sum(dim=1)
        route_confidence = routing_sum.clamp(0.0, 1.0).to(module_dtype)

        action_hidden = self.action_projection(
            candidate_action.detach().to(dtype=module_dtype)
        )
        action_hidden = action_hidden + _sinusoidal_positions(
            int(action_hidden.shape[1]),
            int(action_hidden.shape[2]),
            device=action_hidden.device,
            dtype=action_hidden.dtype,
        )
        conditioned = self.output_norm(
            action_hidden + geometry_context.to(action_hidden.dtype).unsqueeze(1)
        )
        cap = self.max_abs.to(device=candidate_action.device, dtype=module_dtype)
        residual = torch.tanh(self.output_projection(conditioned)) * cap
        residual = residual * route_confidence.unsqueeze(-1)
        if action_is_pad is not None:
            residual = residual.masked_fill(action_is_pad.bool().unsqueeze(-1), 0.0)
        residual = residual.to(dtype=candidate_action.dtype)
        action = candidate_action + residual
        metrics = {
            "pgc_v917_geometry_action_residual_rms": (
                residual.float().square().mean().sqrt()
            ),
            "pgc_v917_geometry_action_residual_max_abs": residual.float().abs().max(),
            "pgc_v917_geometry_route_confidence": route_confidence.float().mean(),
            "pgc_v917_geometry_eef_position_norm": (
                eef_position[:, 0].float().norm(dim=-1).mean()
            ),
            "pgc_v917_geometry_goal_relative_norm": (
                goal_relative.float().norm(dim=-1).mean()
            ),
        }
        return action, residual, metrics


class HardRoutedERAFPhaseServo(nn.Module):
    """Convert one unfinished ERAF clause into a phase-specific action servo.

    The V9.17 adapter softly averages every active clause and then mixes the
    resulting geometry with the candidate action through a LayerNorm.  That
    path is connected, but it does not preserve which clause or Cartesian
    direction caused an action.  V9.19 instead selects exactly one clause,
    keeps the desired Cartesian direction explicit, and learns only a positive
    temporal gain plus bounded rotation/gripper corrections.

    ``legacy_suppression_raw`` and both output heads are zero initialized.  A
    newly-created servo is therefore bitwise equivalent to V9.18: the complete
    legacy geometry residual is retained and the new servo contributes zero.
    Training can subsequently suppress the legacy path per phase while the
    direction-preserving path takes over.
    """

    ARTICULATED_PREDICATE_IDS = (7, 8, 9, 10)

    def __init__(
        self,
        *,
        action_dim: int,
        proprio_dim: int,
        hidden_dim: int = 256,
        max_clauses: int = 4,
        max_abs: float | Sequence[float] = 0.25,
        eef_scale: Sequence[float] = (1.0, 1.0, 1.0),
        eef_bias: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> None:
        super().__init__()
        if min(action_dim, proprio_dim, hidden_dim, max_clauses) <= 0:
            raise ValueError("V9.19 phase-servo dimensions must be positive.")
        if action_dim < 3 or proprio_dim < 3:
            raise ValueError(
                "V9.19 phase servo requires three action and proprio xyz dims."
            )
        if isinstance(max_abs, Sequence) and not isinstance(max_abs, (str, bytes)):
            cap_values = [float(value) for value in max_abs]
            if len(cap_values) != int(action_dim):
                raise ValueError(
                    "V9.19 per-dimension residual caps must match action_dim."
                )
        else:
            cap_values = [float(max_abs)] * int(action_dim)
        if any(value <= 0 for value in cap_values):
            raise ValueError("V9.19 phase-servo residual caps must be positive.")
        eef_scale_values = [float(value) for value in eef_scale]
        eef_bias_values = [float(value) for value in eef_bias]
        if len(eef_scale_values) != 3 or len(eef_bias_values) != 3:
            raise ValueError("V9.19 EEF affine scale/bias must each have 3 values.")
        if not all(
            math.isfinite(value) and value > 0 for value in eef_scale_values
        ) or not all(math.isfinite(value) for value in eef_bias_values):
            raise ValueError("V9.19 EEF affine calibration must be finite and positive.")

        self.action_dim = int(action_dim)
        self.proprio_dim = int(proprio_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_clauses = int(max_clauses)
        # FastWAM proprio stores dataset-normalized EEF xyz in the first three
        # channels.  This affine maps it to ERAF's canonical workspace frame
        # and stays inspectable instead of relearning xyz through an opaque MLP.
        self.eef_scale = nn.Parameter(torch.tensor(eef_scale_values))
        self.eef_bias = nn.Parameter(torch.tensor(eef_bias_values))
        self.register_buffer(
            "initial_eef_scale", torch.tensor(eef_scale_values), persistent=True
        )
        self.register_buffer(
            "initial_eef_bias", torch.tensor(eef_bias_values), persistent=True
        )
        # The matrix is robot/action-frame calibration, not a task-specific
        # policy.  Identity is correct for LIBERO's Cartesian delta convention
        # and remains trainable for another robot frame.
        self.workspace_to_action = nn.Parameter(torch.eye(3))
        gain_input_dim = self.action_dim + 1 + 3 + 1
        self.translation_gain = nn.Sequential(
            nn.LayerNorm(gain_input_dim),
            nn.Linear(gain_input_dim, self.hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.zeros_(self.translation_gain[-1].weight)
        nn.init.zeros_(self.translation_gain[-1].bias)
        aux_input_dim = self.action_dim + 3 + 3 + 1
        self.auxiliary_residual = nn.Sequential(
            nn.LayerNorm(aux_input_dim),
            nn.Linear(aux_input_dim, self.hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.hidden_dim, self.action_dim),
        )
        nn.init.zeros_(self.auxiliary_residual[-1].weight)
        nn.init.zeros_(self.auxiliary_residual[-1].bias)
        self.legacy_suppression_raw = nn.Parameter(torch.zeros(3))
        self.register_buffer(
            "max_abs",
            torch.tensor(cap_values, dtype=torch.float32).view(1, 1, -1),
            persistent=True,
        )

    @staticmethod
    def _gather_clause(value: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
        if value.ndim < 2 or value.shape[0] != selected.shape[0]:
            raise ValueError("V9.19 clause tensor and selection batch mismatch.")
        gather_shape = (selected.shape[0], 1) + (1,) * (value.ndim - 2)
        index = selected.view(gather_shape).expand(
            selected.shape[0], 1, *value.shape[2:]
        )
        return value.gather(1, index).squeeze(1)

    def forward(
        self,
        *,
        candidate_action: torch.Tensor,
        legacy_residual: torch.Tensor,
        eraf_outputs: Mapping[str, torch.Tensor],
        proprio: torch.Tensor,
        eef_position: torch.Tensor | None = None,
        action_is_pad: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        required = {
            "active_logits",
            "predicate_logits",
            "grasp_anchor",
            "goal_anchor",
            "interaction_anchor",
            "phase_logits",
            "predicate_truth_logits",
            "clause_execution_probability",
        }
        missing = sorted(required - set(eraf_outputs))
        if missing:
            raise ValueError(f"V9.19 phase servo is missing ERAF outputs: {missing}.")
        if candidate_action.shape != legacy_residual.shape or (
            candidate_action.ndim != 3
            or candidate_action.shape[-1] != self.action_dim
        ):
            raise ValueError(
                "V9.19 candidate and legacy residual must share [B,T,A]."
            )
        if proprio.shape != (candidate_action.shape[0], self.proprio_dim):
            raise ValueError("V9.19 proprio must be [B,proprio_dim].")
        if action_is_pad is not None and action_is_pad.shape != candidate_action.shape[:2]:
            raise ValueError("V9.19 action padding mask must be [B,T].")

        module_dtype = self.translation_gain[1].weight.dtype
        active = torch.sigmoid(eraf_outputs["active_logits"].float())
        execution = eraf_outputs["clause_execution_probability"].float()
        if active.shape != execution.shape or active.shape != (
            candidate_action.shape[0],
            self.max_clauses,
        ):
            raise ValueError("V9.19 active/execution routing must be [B,C].")
        route_score = execution.clamp_min(0.0) * active
        fallback = torch.softmax(eraf_outputs["active_logits"].float(), dim=-1)
        route_score = torch.where(
            (route_score.sum(dim=-1, keepdim=True) > 1.0e-6),
            route_score,
            fallback,
        )
        selected_clause = route_score.argmax(dim=-1)
        selected_confidence = self._gather_clause(
            route_score, selected_clause
        ).clamp(0.0, 1.0)

        phase_probability = torch.softmax(
            eraf_outputs["phase_logits"].float(), dim=-1
        )
        selected_phase_probability = self._gather_clause(
            phase_probability, selected_clause
        ).to(module_dtype)
        selected_phase = selected_phase_probability.argmax(dim=-1)
        selected_truth = torch.sigmoid(
            self._gather_clause(
                eraf_outputs["predicate_truth_logits"].float(), selected_clause
            )
        )
        # A true predicate while holding is release-ready, matching V9.18's
        # audited phase convention.
        control_phase = torch.where(
            (selected_phase == 1) & (selected_truth >= 0.5),
            torch.full_like(selected_phase, 2),
            selected_phase,
        )
        predicate_id = self._gather_clause(
            eraf_outputs["predicate_logits"].argmax(dim=-1), selected_clause
        )
        articulated = torch.zeros_like(predicate_id, dtype=torch.bool)
        for value in self.ARTICULATED_PREDICATE_IDS:
            articulated |= predicate_id == value

        if eef_position is None:
            effective_eef_scale = self.eef_scale.to(module_dtype).clamp_min(
                1.0e-4
            )
            calibrated_eef = (
                proprio[:, :3].to(dtype=module_dtype)
                * effective_eef_scale
                + self.eef_bias.to(module_dtype)
            ).clamp(-1.0, 1.0)
            calibrated_from_state = calibrated_eef.new_zeros(())
        else:
            if eef_position.shape != (candidate_action.shape[0], 3):
                raise ValueError("V9.19 canonical EEF position must be [B,3].")
            calibrated_eef = eef_position.to(dtype=module_dtype)
            calibrated_from_state = calibrated_eef.new_ones(())

        grasp = self._gather_clause(
            eraf_outputs["grasp_anchor"].to(module_dtype), selected_clause
        )
        goal = self._gather_clause(
            eraf_outputs["goal_anchor"].to(module_dtype), selected_clause
        )
        interaction = self._gather_clause(
            eraf_outputs["interaction_anchor"].to(module_dtype), selected_clause
        )
        approach_vector = grasp - calibrated_eef
        transport_vector = goal - calibrated_eef
        release_vector = torch.where(
            articulated.unsqueeze(-1),
            interaction - calibrated_eef,
            transport_vector,
        )
        desired = torch.where(
            (control_phase == 0).unsqueeze(-1),
            approach_vector,
            torch.where(
                (control_phase == 1).unsqueeze(-1),
                transport_vector,
                release_vector,
            ),
        )
        desired_norm = desired.norm(dim=-1, keepdim=True)
        workspace_direction = desired / desired_norm.clamp_min(1.0e-6)
        action_direction = torch.matmul(
            workspace_direction, self.workspace_to_action.to(module_dtype).T
        )
        action_direction = F.normalize(action_direction, dim=-1, eps=1.0e-6)

        batch, horizon, _ = candidate_action.shape
        phase_features = F.one_hot(control_phase.clamp(0, 2), 3).to(module_dtype)
        candidate = candidate_action.detach().to(module_dtype)
        gain_features = torch.cat(
            (
                candidate,
                desired_norm.unsqueeze(1).expand(-1, horizon, -1),
                phase_features.unsqueeze(1).expand(-1, horizon, -1),
                selected_truth.to(module_dtype).view(batch, 1, 1).expand(
                    -1, horizon, -1
                ),
            ),
            dim=-1,
        )
        # The forward value is non-negative and bounded, so the servo cannot
        # reverse the desired direction.  A straight-through term guarantees
        # a nonzero first-step gradient at the exact zero-init boundary across
        # PyTorch versions (whose clamp boundary subgradient may differ).
        translation_cap = float(self.max_abs[..., :3].min().item())
        raw_gain = self.translation_gain(gain_features)
        bounded_gain = raw_gain.clamp(
            min=0.0, max=translation_cap
        )
        gain = bounded_gain + raw_gain - raw_gain.detach()
        gain = gain * selected_confidence.to(module_dtype).view(batch, 1, 1)
        translation = action_direction.unsqueeze(1) * gain

        aux_features = torch.cat(
            (
                candidate,
                desired.unsqueeze(1).expand(-1, horizon, -1),
                phase_features.unsqueeze(1).expand(-1, horizon, -1),
                selected_truth.to(module_dtype).view(batch, 1, 1).expand(
                    -1, horizon, -1
                ),
            ),
            dim=-1,
        )
        cap = self.max_abs.to(device=candidate.device, dtype=module_dtype)
        auxiliary = torch.tanh(self.auxiliary_residual(aux_features)) * cap
        auxiliary = auxiliary.clone()
        auxiliary[..., :3] = 0.0
        servo_residual = auxiliary
        servo_residual[..., :3] = translation

        bounded_suppression = self.legacy_suppression_raw.clamp(0.0, 1.0)
        straight_through_suppression = (
            bounded_suppression
            + self.legacy_suppression_raw
            - self.legacy_suppression_raw.detach()
        )
        suppression = straight_through_suppression[control_phase].to(
            module_dtype
        )
        retained_legacy = legacy_residual.to(module_dtype) * (
            1.0 - suppression.view(batch, 1, 1)
        )
        total_residual = (retained_legacy + servo_residual).clamp(
            min=-cap, max=cap
        )
        if action_is_pad is not None:
            total_residual = total_residual.masked_fill(
                action_is_pad.bool().unsqueeze(-1), 0.0
            )
            servo_residual = servo_residual.masked_fill(
                action_is_pad.bool().unsqueeze(-1), 0.0
            )
        total_residual = total_residual.to(candidate_action.dtype)
        action = candidate_action + total_residual
        hard_route = F.one_hot(selected_clause, self.max_clauses).float()
        return action, total_residual, {
            "pgc_v919_hard_clause_route_max": hard_route.max(dim=-1).values.mean(),
            "pgc_v919_selected_clause_mean": selected_clause.float().mean(),
            "pgc_v919_selected_phase_mean": control_phase.float().mean(),
            "pgc_v919_route_confidence": selected_confidence.float().mean(),
            "pgc_v919_canonical_eef_from_state": calibrated_from_state,
            "pgc_v919_eef_scale_min": self.eef_scale.float().min(),
            "pgc_v919_eef_bias_max_abs": self.eef_bias.float().abs().max(),
            "pgc_v919_desired_distance": desired_norm.float().mean(),
            "pgc_v919_desired_distance_per_sample": (
                desired_norm.squeeze(-1)
            ),
            "pgc_v919_translation_gain": gain.float().mean(),
            "pgc_v919_servo_residual_rms": (
                servo_residual.float().square().mean().sqrt()
            ),
            "pgc_v919_total_residual_rms": (
                total_residual.float().square().mean().sqrt()
            ),
            "pgc_v919_legacy_suppression": suppression.float().mean(),
            "pgc_v919_workspace_to_action_orthogonality_error": (
                (
                    self.workspace_to_action.float().T
                    @ self.workspace_to_action.float()
                    - torch.eye(3, device=self.workspace_to_action.device)
                )
                .square()
                .mean()
                .sqrt()
            ),
            "pgc_v919_workspace_to_action_determinant": torch.linalg.det(
                self.workspace_to_action.float()
            ),
            "pgc_v919_selected_clause": selected_clause,
            "pgc_v919_selected_control_phase": control_phase,
            "pgc_v919_route_confidence_per_sample": selected_confidence,
            "pgc_v919_desired_direction": action_direction,
            "pgc_v919_retained_legacy_residual": retained_legacy,
            "pgc_v919_servo_residual": servo_residual,
        }


class PhaseCompatibleERAFWaypointAdapter(nn.Module):
    """Learn a local action vector only where ERAF geometry is executable.

    A terminal anchor is not necessarily the next collision-free Cartesian
    direction.  This adapter keeps positive progress toward the hard-routed
    ERAF anchor, while learning a bounded tangent component that represents a
    local waypoint.  A separate compatibility predictor decides whether the
    current phase should use this geometric route at all.  The gain and tangent
    heads are zero initialized, so adding V9.20 is exactly V9.19 at warm start.
    """

    def __init__(
        self,
        *,
        action_dim: int,
        hidden_dim: int = 256,
        max_abs: float | Sequence[float] = 0.25,
        tangent_max_ratio: float = 0.75,
    ) -> None:
        super().__init__()
        if action_dim < 3 or hidden_dim <= 0:
            raise ValueError("V9.20 waypoint adapter requires xyz actions.")
        if not 0 < tangent_max_ratio <= 1:
            raise ValueError("V9.20 tangent_max_ratio must be in (0,1].")
        if isinstance(max_abs, Sequence) and not isinstance(max_abs, (str, bytes)):
            caps = [float(value) for value in max_abs]
            if len(caps) != int(action_dim):
                raise ValueError("V9.20 residual caps must match action_dim.")
        else:
            caps = [float(max_abs)] * int(action_dim)
        if any(value <= 0 for value in caps):
            raise ValueError("V9.20 residual caps must be positive.")

        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.tangent_max_ratio = float(tangent_max_ratio)
        feature_dim = self.action_dim + 3 + 3 + 1
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.feature_projection = nn.Sequential(
            nn.Linear(feature_dim, self.hidden_dim),
            nn.GELU(approximate="tanh"),
        )
        self.compatibility_head = nn.Linear(self.hidden_dim, 1)
        self.tangent_head = nn.Linear(self.hidden_dim, 3)
        self.gain_head = nn.Linear(self.hidden_dim, 1)
        nn.init.zeros_(self.compatibility_head.weight)
        nn.init.zeros_(self.compatibility_head.bias)
        nn.init.zeros_(self.tangent_head.weight)
        nn.init.zeros_(self.tangent_head.bias)
        nn.init.zeros_(self.gain_head.weight)
        nn.init.zeros_(self.gain_head.bias)
        self.register_buffer(
            "max_abs",
            torch.tensor(caps, dtype=torch.float32).view(1, 1, -1),
            persistent=True,
        )

    def forward(
        self,
        *,
        candidate_action: torch.Tensor,
        legacy_residual: torch.Tensor,
        inherited_servo_residual: torch.Tensor,
        desired_direction: torch.Tensor,
        control_phase: torch.Tensor,
        route_confidence: torch.Tensor,
        action_is_pad: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if (
            candidate_action.shape != legacy_residual.shape
            or candidate_action.shape != inherited_servo_residual.shape
            or (
            candidate_action.ndim != 3
            or candidate_action.shape[-1] != self.action_dim
            )
        ):
            raise ValueError(
                "V9.20 candidate/legacy/inherited actions must share [B,T,A]."
            )
        batch, horizon, _ = candidate_action.shape
        if desired_direction.shape != (batch, 3):
            raise ValueError("V9.20 desired_direction must be [B,3].")
        if control_phase.shape != (batch,) or route_confidence.shape != (batch,):
            raise ValueError("V9.20 phase/confidence must be [B].")
        if action_is_pad is not None and action_is_pad.shape != (batch, horizon):
            raise ValueError("V9.20 action padding mask must be [B,T].")

        module_dtype = self.feature_projection[0].weight.dtype
        candidate = candidate_action.to(module_dtype)
        anchor_direction = F.normalize(
            desired_direction.to(module_dtype), dim=-1, eps=1.0e-6
        )
        phase = F.one_hot(control_phase.long().clamp(0, 2), 3).to(module_dtype)
        features = torch.cat(
            (
                candidate,
                anchor_direction.unsqueeze(1).expand(-1, horizon, -1),
                phase.unsqueeze(1).expand(-1, horizon, -1),
                route_confidence.to(module_dtype)
                .view(batch, 1, 1)
                .expand(-1, horizon, -1),
            ),
            dim=-1,
        )
        hidden = self.feature_projection(self.feature_norm(features))
        compatibility_logits = self.compatibility_head(hidden).squeeze(-1)
        compatibility = torch.sigmoid(compatibility_logits)
        raw_retention = compatibility * 2.0
        bounded_retention = raw_retention.clamp(max=1.0)
        inherited_retention = (
            bounded_retention.detach() + raw_retention - raw_retention.detach()
        )

        raw_tangent = self.tangent_head(hidden)
        anchor_step = anchor_direction.unsqueeze(1)
        tangent = raw_tangent - (
            raw_tangent * anchor_step
        ).sum(dim=-1, keepdim=True) * anchor_step
        # Smooth vector-norm saturation. Unlike tanh(||t||) * t / ||t|| with
        # an epsilon clamp, this map has Jacobian tangent_max_ratio * I at the
        # zero-init point, so the local-waypoint head can actually leave zero.
        tangent = self.tangent_max_ratio * tangent / torch.sqrt(
            1.0 + tangent.square().sum(dim=-1, keepdim=True)
        )
        local_direction = F.normalize(
            anchor_step + tangent, dim=-1, eps=1.0e-6
        )

        translation_cap = self.max_abs[..., :3].to(
            device=candidate.device, dtype=module_dtype
        ).min()
        raw_gain = self.gain_head(hidden)
        bounded_gain = raw_gain.clamp(min=0.0, max=translation_cap)
        gain = bounded_gain.detach() + raw_gain - raw_gain.detach()
        translation = local_direction * gain * compatibility.unsqueeze(-1)
        servo_residual = torch.zeros_like(candidate)
        servo_residual[..., :3] = translation
        if action_is_pad is not None:
            servo_residual = servo_residual.masked_fill(
                action_is_pad.bool().unsqueeze(-1), 0.0
            )
        cap = self.max_abs.to(device=candidate.device, dtype=module_dtype)
        retained_inherited_servo = inherited_servo_residual.to(module_dtype) * (
            inherited_retention.unsqueeze(-1)
        )
        effective_servo_residual = retained_inherited_servo + servo_residual
        total_residual = (
            legacy_residual.to(module_dtype)
            + effective_servo_residual
        ).clamp(min=-cap, max=cap)
        if action_is_pad is not None:
            pad = action_is_pad.bool().unsqueeze(-1)
            effective_servo_residual = effective_servo_residual.masked_fill(
                pad, 0.0
            )
            total_residual = total_residual.masked_fill(pad, 0.0)
        action = candidate + total_residual
        return action.to(candidate_action.dtype), total_residual.to(
            candidate_action.dtype
        ), {
            "pgc_v920_compatibility_probability": compatibility.float().mean(),
            "pgc_v920_inherited_servo_retention": (
                bounded_retention.float().mean()
            ),
            "pgc_v920_waypoint_tangent_rms": tangent.float().square().mean().sqrt(),
            "pgc_v920_translation_gain": bounded_gain.float().mean(),
            "pgc_v920_servo_residual_rms": (
                servo_residual.float().square().mean().sqrt()
            ),
            "pgc_v920_total_residual_rms": (
                total_residual.float().square().mean().sqrt()
            ),
            "pgc_v920_compatibility_logits": compatibility_logits,
            "pgc_v920_compatibility_probability_per_step": compatibility,
            "pgc_v920_local_direction": local_direction,
            "pgc_v920_servo_translation": translation,
            "pgc_v920_effective_servo_residual": effective_servo_residual,
        }


class PhaseSpecificERAFExpertResidualAdapter(nn.Module):
    """Map routed ERAF geometry to an expert-aligned local action residual.

    A terminal grasp/goal/interaction anchor does not uniquely determine the
    next action: transport may first lift or detour, and release also needs
    rotation and gripper commands.  The preceding waypoint field can only add
    a positive-progress xyz vector.  This adapter therefore predicts a bounded
    full-action correction with a separate output head for approach,
    transport, and release.  Its heads are zero initialized, making a newly
    added adapter exactly equivalent to the frozen waypoint checkpoint.

    Training may call the same module a second time with privileged anchors and
    phase labels.  Those labels never enter the deployed call; they only teach
    the shared geometry-to-action map and provide a distillation target for the
    learned ERAF route.
    """

    def __init__(
        self,
        *,
        action_dim: int,
        hidden_dim: int = 256,
        max_abs: float | Sequence[float] = 0.25,
    ) -> None:
        super().__init__()
        if action_dim < 3 or hidden_dim <= 0:
            raise ValueError(
                "ERAF expert residual adapter requires xyz actions and a "
                "positive hidden dimension."
            )
        if isinstance(max_abs, Sequence) and not isinstance(max_abs, (str, bytes)):
            caps = [float(value) for value in max_abs]
            if len(caps) != int(action_dim):
                raise ValueError(
                    "ERAF expert residual caps must match action_dim."
                )
        else:
            caps = [float(max_abs)] * int(action_dim)
        if any(value <= 0 for value in caps):
            raise ValueError("ERAF expert residual caps must be positive.")

        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        # current action + current residual + direction + distance + route
        # confidence + waypoint compatibility + normalized horizon position.
        feature_dim = 2 * self.action_dim + 3 + 4
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.feature_projection = nn.Sequential(
            nn.Linear(feature_dim, self.hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(approximate="tanh"),
        )
        self.phase_output = nn.Linear(self.hidden_dim, 3 * self.action_dim)
        nn.init.zeros_(self.phase_output.weight)
        nn.init.zeros_(self.phase_output.bias)
        self.register_buffer(
            "max_abs",
            torch.tensor(caps, dtype=torch.float32).view(1, 1, 1, -1),
            persistent=True,
        )

    def forward(
        self,
        *,
        candidate_action: torch.Tensor,
        current_residual: torch.Tensor,
        desired_direction: torch.Tensor,
        desired_distance: torch.Tensor,
        control_phase: torch.Tensor,
        route_confidence: torch.Tensor,
        waypoint_compatibility: torch.Tensor,
        action_is_pad: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if (
            candidate_action.ndim != 3
            or candidate_action.shape[-1] != self.action_dim
            or current_residual.shape != candidate_action.shape
        ):
            raise ValueError(
                "ERAF expert candidate/current residual must share [B,T,A]."
            )
        batch, horizon, _ = candidate_action.shape
        if desired_direction.shape != (batch, 3):
            raise ValueError("ERAF expert desired_direction must be [B,3].")
        if desired_distance.shape not in {(batch,), (batch, 1)}:
            raise ValueError("ERAF expert desired_distance must be [B] or [B,1].")
        if control_phase.shape != (batch,) or route_confidence.shape != (batch,):
            raise ValueError("ERAF expert phase/confidence must be [B].")
        if waypoint_compatibility.shape not in {
            (batch,),
            (batch, horizon),
        }:
            raise ValueError(
                "ERAF expert waypoint compatibility must be [B] or [B,T]."
            )
        if action_is_pad is not None and action_is_pad.shape != (batch, horizon):
            raise ValueError("ERAF expert action padding mask must be [B,T].")

        module_dtype = self.feature_projection[0].weight.dtype
        candidate = candidate_action.to(module_dtype)
        current = current_residual.to(module_dtype)
        current_action = candidate + current
        direction = F.normalize(
            desired_direction.to(module_dtype), dim=-1, eps=1.0e-6
        ).unsqueeze(1).expand(-1, horizon, -1)
        distance = desired_distance.to(module_dtype).reshape(batch, 1, 1).expand(
            -1, horizon, -1
        )
        route = route_confidence.to(module_dtype).reshape(batch, 1, 1).expand(
            -1, horizon, -1
        )
        compatibility = waypoint_compatibility.to(module_dtype)
        if compatibility.ndim == 1:
            compatibility = compatibility[:, None].expand(-1, horizon)
        compatibility = compatibility.unsqueeze(-1)
        if horizon == 1:
            progress = candidate.new_zeros((batch, 1, 1))
        else:
            progress = torch.linspace(
                0.0,
                1.0,
                horizon,
                device=candidate.device,
                dtype=module_dtype,
            ).view(1, horizon, 1).expand(batch, -1, -1)
        features = torch.cat(
            (
                current_action,
                current,
                direction,
                distance,
                route,
                compatibility,
                progress,
            ),
            dim=-1,
        )
        hidden = self.feature_projection(self.feature_norm(features))
        raw_candidates = self.phase_output(hidden).reshape(
            batch, horizon, 3, self.action_dim
        )
        cap = self.max_abs.to(device=candidate.device, dtype=module_dtype)
        phase_candidates = torch.tanh(raw_candidates) * cap
        phase_index = control_phase.long().clamp(0, 2).view(
            batch, 1, 1, 1
        ).expand(-1, horizon, 1, self.action_dim)
        selected_correction = phase_candidates.gather(2, phase_index).squeeze(2)
        if action_is_pad is not None:
            pad = action_is_pad.bool().unsqueeze(-1)
            selected_correction = selected_correction.masked_fill(pad, 0.0)
            phase_candidates = phase_candidates.masked_fill(
                pad.unsqueeze(2), 0.0
            )
        total_cap = cap.squeeze(2)
        total_residual = (current + selected_correction).clamp(
            min=-total_cap, max=total_cap
        )
        if action_is_pad is not None:
            total_residual = total_residual.masked_fill(pad, 0.0)
        effective_correction = total_residual - current
        if action_is_pad is not None:
            effective_correction = effective_correction.masked_fill(pad, 0.0)
        action = candidate + total_residual
        return action.to(candidate_action.dtype), total_residual.to(
            candidate_action.dtype
        ), {
            "pgc_v921_expert_correction_rms": (
                effective_correction.float().square().mean().sqrt()
            ),
            "pgc_v921_raw_expert_correction_rms": (
                selected_correction.float().square().mean().sqrt()
            ),
            "pgc_v921_total_residual_rms": (
                total_residual.float().square().mean().sqrt()
            ),
            "pgc_v921_phase_residual_candidates": phase_candidates,
            "pgc_v921_selected_expert_correction": selected_correction,
            "pgc_v921_effective_expert_correction": effective_correction,
            "pgc_v921_selected_control_phase": control_phase,
        }


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
    """Detach scalar training metrics without consuming audit tensors.

    Proposal modules may return per-sample routing decisions alongside scalar
    summaries.  Those tensors are useful to causal audits and rollout traces,
    but they are not valid trainer log values.  Keep them in the proposal
    output while omitting them from the scalar loss dictionary.
    """

    detached: dict[str, float] = {}
    for name, value in metrics.items():
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                continue
            detached[name] = float(value.detach().float().cpu().item())
        else:
            detached[name] = float(value)
    return detached
