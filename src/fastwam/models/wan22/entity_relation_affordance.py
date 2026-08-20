"""Predicate-grounded entity--relation affordance fields for PGC V9.

ERAF is deliberately a deployment-safe sidecar.  Simulator masks, entity
positions, and predicate state are accepted only by the loss helper below;
the forward path consumes the same language and RGB-derived video tokens as
the released policy.  A zero-initialized bridge injects the structured
representation into the already trained V5 goal queries, which makes a
V5->V9 migration numerically exact before optimization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .policy_guard import _safe_key_padding_mask, infer_spatial_patch_grid


ERAF_PREDICATES = (
    "pad",
    "in",
    "on",
    "left",
    "right",
    "front",
    "back",
    "open",
    "close",
    "turnon",
    "turnoff",
)
ERAF_PREDICATE_TO_ID = {name: index for index, name in enumerate(ERAF_PREDICATES)}


def masks_to_patch_distributions(
    masks: torch.Tensor, *, token_count: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Area-resample ``[B,C,H,W]`` masks onto the concatenated patch grid."""
    if masks.ndim != 4:
        raise ValueError("ERAF role masks must have shape [B,C,H,W].")
    batch_size, clause_count, height, width = masks.shape
    if min(batch_size, clause_count, height, width, int(token_count)) <= 0:
        raise ValueError("ERAF role masks and token count must be non-empty.")
    grid_height, grid_width = infer_spatial_patch_grid(
        int(token_count), aspect_ratio=float(width) / float(height)
    )
    mass = F.interpolate(
        masks.float().reshape(batch_size * clause_count, 1, height, width),
        size=(grid_height, grid_width),
        mode="area",
    ).reshape(batch_size, clause_count, int(token_count))
    total = mass.sum(dim=-1)
    valid = total > 1.0e-6
    uniform = torch.full_like(mass, 1.0 / max(1, int(token_count)))
    distribution = torch.where(
        valid.unsqueeze(-1),
        mass / total.clamp_min(1.0e-6).unsqueeze(-1),
        uniform,
    )
    return distribution, valid


def masks_to_patch_targets(
    masks: torch.Tensor, *, token_count: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Area-resample masks to per-patch occupancy targets for Dice+BCE."""
    if masks.ndim != 4:
        raise ValueError("ERAF role masks must have shape [B,C,H,W].")
    batch_size, clause_count, height, width = masks.shape
    grid_height, grid_width = infer_spatial_patch_grid(
        int(token_count), aspect_ratio=float(width) / float(height)
    )
    target = (
        F.interpolate(
            masks.float().reshape(batch_size * clause_count, 1, height, width),
            size=(grid_height, grid_width),
            mode="area",
        )
        .reshape(batch_size, clause_count, int(token_count))
        .clamp_(0.0, 1.0)
    )
    return target, target.sum(dim=-1) > 1.0e-6


class PredicateRoleDecoder(nn.Module):
    """Decode a bounded set of predicate clauses and semantic role queries."""

    def __init__(
        self,
        *,
        text_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        max_clauses: int = 4,
        num_predicates: int = len(ERAF_PREDICATES),
    ) -> None:
        super().__init__()
        if min(text_dim, hidden_dim, num_heads, max_clauses, num_predicates) <= 0:
            raise ValueError("ERAF role-decoder dimensions must be positive.")
        if hidden_dim % num_heads:
            raise ValueError(
                "ERAF role-decoder hidden_dim must be divisible by num_heads."
            )
        self.text_dim = int(text_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.max_clauses = int(max_clauses)
        self.num_predicates = int(num_predicates)
        self.clause_seeds = nn.Parameter(
            torch.randn(1, max_clauses, hidden_dim) / math.sqrt(hidden_dim)
        )
        self.subject_role = nn.Parameter(
            torch.randn(1, 1, hidden_dim) / math.sqrt(hidden_dim)
        )
        self.reference_role = nn.Parameter(
            torch.randn(1, 1, hidden_dim) / math.sqrt(hidden_dim)
        )
        self.text_projection = nn.Sequential(
            nn.LayerNorm(text_dim), nn.Linear(text_dim, hidden_dim)
        )
        self.language_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.clause_norm = nn.LayerNorm(hidden_dim)
        self.clause_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.active_head = nn.Linear(hidden_dim, 1)
        self.predicate_head = nn.Linear(hidden_dim, num_predicates)
        self.predicate_embedding = nn.Embedding(num_predicates, hidden_dim)
        self.subject_norm = nn.LayerNorm(hidden_dim)
        self.reference_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self, language_hidden: torch.Tensor, language_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if language_hidden.ndim != 3 or language_hidden.shape[-1] != self.text_dim:
            raise ValueError("ERAF language_hidden must be [B,L,text_dim].")
        if language_mask.shape != language_hidden.shape[:2]:
            raise ValueError("ERAF language_mask must be [B,L].")
        text = self.text_projection(language_hidden)
        text, text_padding = _safe_key_padding_mask(text, language_mask)
        clauses = self.clause_seeds.expand(language_hidden.shape[0], -1, -1)
        delta, attention = self.language_attention(
            query=clauses,
            key=text,
            value=text,
            key_padding_mask=text_padding,
            need_weights=True,
        )
        clauses = self.clause_norm(clauses + delta)
        clauses = clauses + self.clause_mlp(clauses)
        predicate_logits = self.predicate_head(clauses)
        predicate_probabilities = torch.softmax(predicate_logits.float(), dim=-1)
        predicate_state = torch.matmul(
            predicate_probabilities.to(clauses.dtype),
            self.predicate_embedding.weight,
        )
        clauses = clauses + predicate_state
        return {
            "clause_hidden": clauses,
            "active_logits": self.active_head(clauses).squeeze(-1),
            "predicate_logits": predicate_logits,
            "subject_queries": self.subject_norm(clauses + self.subject_role),
            "reference_queries": self.reference_norm(clauses + self.reference_role),
            "language_attention": attention,
        }


class MultiViewEntityGrounder(nn.Module):
    """Ground semantic role queries in camera-aware spatial video patches."""

    def __init__(
        self,
        *,
        video_dim: int,
        hidden_dim: int = 256,
        camera_count: int = 2,
        visual_aspect_ratio: float = 2.0,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        if min(video_dim, hidden_dim, camera_count) <= 0:
            raise ValueError("ERAF entity-grounder dimensions must be positive.")
        if visual_aspect_ratio <= 0 or temperature <= 0:
            raise ValueError("ERAF aspect ratio and temperature must be positive.")
        self.video_dim = int(video_dim)
        self.hidden_dim = int(hidden_dim)
        self.camera_count = int(camera_count)
        self.visual_aspect_ratio = float(visual_aspect_ratio)
        self.temperature = float(temperature)
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
        self.role_norm = nn.LayerNorm(hidden_dim)
        self.position_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, 3),
            nn.Tanh(),
        )
        self.visibility_head = nn.Linear(hidden_dim, 1)
        self.view_visibility_head = nn.Sequential(
            nn.LayerNorm(hidden_dim + 1),
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, 1),
        )

    def _spatial_visual(
        self, visual_hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        token_count = int(visual_hidden.shape[1])
        grid_height, grid_width = infer_spatial_patch_grid(
            token_count, aspect_ratio=self.visual_aspect_ratio
        )
        y = torch.linspace(
            -1.0,
            1.0,
            grid_height,
            device=visual_hidden.device,
            dtype=visual_hidden.dtype,
        )
        x = torch.linspace(
            -1.0,
            1.0,
            grid_width,
            device=visual_hidden.device,
            dtype=visual_hidden.dtype,
        )
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coordinates = torch.stack((xx, yy), dim=-1).reshape(token_count, 2)
        columns = torch.arange(grid_width, device=visual_hidden.device)
        if grid_width % self.camera_count:
            raise ValueError(
                "ERAF horizontally concatenated patch width must be divisible "
                "by camera_count."
            )
        camera_ids = torch.div(
            columns * self.camera_count, grid_width, rounding_mode="floor"
        ).clamp(max=self.camera_count - 1)
        camera_ids = camera_ids.unsqueeze(0).expand(grid_height, -1).reshape(-1)
        camera_width = grid_width // self.camera_count
        local_x = torch.linspace(
            -1.0,
            1.0,
            camera_width,
            device=visual_hidden.device,
            dtype=visual_hidden.dtype,
        )
        local_x = local_x[columns.remainder(camera_width)]
        local_x = local_x.unsqueeze(0).expand(grid_height, -1).reshape(-1)
        local_y = yy.reshape(-1)
        view_coordinates = torch.stack((local_x, local_y), dim=-1)
        visual = self.visual_projection(visual_hidden)
        visual = visual + self.position_projection(coordinates).unsqueeze(0)
        visual = visual + self.camera_embedding(camera_ids).unsqueeze(0)
        return self.visual_norm(visual), coordinates, camera_ids, view_coordinates

    def ground(
        self, role_queries: torch.Tensor, visual_hidden: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if role_queries.ndim != 3 or role_queries.shape[-1] != self.hidden_dim:
            raise ValueError("ERAF role queries must be [B,C,hidden_dim].")
        if visual_hidden.ndim != 3 or visual_hidden.shape[-1] != self.video_dim:
            raise ValueError("ERAF visual hidden must be [B,N,video_dim].")
        if role_queries.shape[0] != visual_hidden.shape[0]:
            raise ValueError("ERAF role/video batches must match.")
        visual, coordinates, camera_ids, view_coordinates = self._spatial_visual(
            visual_hidden
        )
        query = F.normalize(self.role_norm(role_queries).float(), dim=-1, eps=1e-6)
        key = F.normalize(visual.float(), dim=-1, eps=1e-6)
        similarity = torch.einsum("bcd,bnd->bcn", query, key)
        attention = torch.softmax(similarity / self.temperature, dim=-1)
        tokens = torch.einsum("bcn,bnd->bcd", attention.to(visual.dtype), visual)
        camera_membership = (
            F.one_hot(camera_ids, num_classes=self.camera_count)
            .transpose(0, 1)
            .to(attention.dtype)
        )
        view_attention = attention.unsqueeze(2) * camera_membership[None, None]
        view_mass = view_attention.sum(dim=-1)
        normalized_view_attention = view_attention / view_mass.clamp_min(
            1.0e-8
        ).unsqueeze(-1)
        view_tokens = torch.einsum(
            "bcvn,bnd->bcvd",
            normalized_view_attention.to(visual.dtype),
            visual,
        )
        view_centers = torch.einsum(
            "bcvn,nd->bcvd",
            normalized_view_attention.float(),
            view_coordinates.float(),
        )
        view_visibility_logits = self.view_visibility_head(
            torch.cat(
                (view_tokens, view_mass.to(view_tokens.dtype).unsqueeze(-1)),
                dim=-1,
            )
        ).squeeze(-1)
        return {
            "attention": attention,
            "token": tokens,
            # Kept only inside the transient ERAF output dictionary so the
            # verifier can form a same-state wrong-entity candidate.  This is
            # still an RGB-derived deployment feature, never a simulator label.
            "visual_tokens": visual,
            "position": self.position_head(tokens).float(),
            "visibility_logits": self.visibility_head(tokens).squeeze(-1),
            "view_visibility_logits": view_visibility_logits,
            "view_centers": view_centers,
            "view_attention_mass": view_mass,
            # Mask BCE needs unbounded logits.  Cosine similarity itself is
            # confined to [-1,1], so expose the same temperature-scaled logits
            # used by the spatial softmax.
            "similarity": similarity / self.temperature,
            "coordinates": coordinates,
            "camera_ids": camera_ids,
            "view_coordinates": view_coordinates,
        }


class RelationAffordanceReasoner(nn.Module):
    """Turn grounded role pairs into executable clause-level affordances."""

    def __init__(
        self,
        *,
        hidden_dim: int = 256,
        action_dim: int,
        projection_dim: int,
        phase_count: int = 3,
    ) -> None:
        super().__init__()
        if min(hidden_dim, action_dim, projection_dim, phase_count) <= 0:
            raise ValueError("ERAF relation-reasoner dimensions must be positive.")
        self.hidden_dim = int(hidden_dim)
        self.action_dim = int(action_dim)
        self.projection_dim = int(projection_dim)
        self.phase_count = int(phase_count)
        self.relation_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 6, hidden_dim * 2),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.predicate_truth_head = nn.Linear(hidden_dim, 1)
        self.predicate_state_projection = nn.Linear(1, hidden_dim)
        self.relation_refinement = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.phase_head = nn.Linear(hidden_dim, phase_count)
        self.interaction_anchor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
            nn.Tanh(),
        )
        self.grasp_anchor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
            nn.Tanh(),
        )
        self.goal_anchor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
            nn.Tanh(),
        )
        self.action_projection = nn.Linear(hidden_dim, action_dim)
        self.embedding_projection = nn.Linear(hidden_dim, projection_dim)

    def decode_relation_hidden(self, relation: torch.Tensor) -> dict[str, torch.Tensor]:
        """Decode a clause representation into executable affordance heads."""
        return {
            "relation_hidden": relation,
            "action_tokens": self.action_projection(relation),
            "embedding_tokens": self.embedding_projection(relation),
            "predicate_truth_logits": self.predicate_truth_head(relation).squeeze(-1),
            "phase_logits": self.phase_head(relation),
            "grasp_anchor": self.grasp_anchor_head(relation).float(),
            "interaction_anchor": self.interaction_anchor_head(relation).float(),
            "goal_anchor": self.goal_anchor_head(relation).float(),
        }

    def forward(
        self,
        *,
        clause_hidden: torch.Tensor,
        subject_token: torch.Tensor,
        reference_token: torch.Tensor,
        subject_position: torch.Tensor,
        reference_position: torch.Tensor,
        active_logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        delta = subject_position - reference_position
        relation = self.relation_mlp(
            torch.cat(
                (
                    clause_hidden,
                    subject_token,
                    reference_token,
                    subject_position.to(clause_hidden.dtype),
                    delta.to(clause_hidden.dtype),
                ),
                dim=-1,
            )
        )
        # Infer the current predicate truth from the grounded scene, then feed
        # that state back into the executable relation representation.  This
        # separates, for example, an already-open drawer from the same fixture
        # before opening and lets the phase/anchor heads condition on state.
        truth_probability = torch.sigmoid(
            self.predicate_truth_head(relation).float()
        ).to(relation.dtype)
        relation = relation + self.relation_refinement(
            relation
            + self.predicate_state_projection(
                # predicate_truth_head already preserves a singleton feature
                # axis: [batch, clauses, 1].  Adding another axis would turn
                # this into [batch, clauses, 1, 1] and accidentally broadcast
                # the batch axis against the clause axis whenever B != C.
                truth_probability
            )
        )
        active_probability = torch.sigmoid(active_logits.float()).to(relation.dtype)
        relation = relation * active_probability.unsqueeze(-1)
        return self.decode_relation_hidden(relation)


class RoleAssignmentResidualAdapter(nn.Module):
    """A zero-init role-only repair on top of a frozen ERAF backbone.

    V9.2 updated the complete ERAF module while trying to separate subject and
    reference queries.  That also moved the already-good predicate, visual,
    relation, and anchor representations.  V9.3 instead learns two small
    residuals from the frozen V9.1 role/clause states.  Zero-initializing the
    output projections makes a freshly migrated V9.3 checkpoint numerically
    identical to V9.1 before the first optimizer step.
    """

    def __init__(self, *, hidden_dim: int, adapter_hidden_dim: int) -> None:
        super().__init__()
        if min(hidden_dim, adapter_hidden_dim) <= 0:
            raise ValueError("ERAF role-adapter dimensions must be positive.")
        self.hidden_dim = int(hidden_dim)
        self.adapter_hidden_dim = int(adapter_hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim * 3)
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim * 3, adapter_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(adapter_hidden_dim, adapter_hidden_dim),
            nn.GELU(approximate="tanh"),
        )
        self.subject_output = nn.Linear(adapter_hidden_dim, hidden_dim)
        self.reference_output = nn.Linear(adapter_hidden_dim, hidden_dim)
        nn.init.zeros_(self.subject_output.weight)
        nn.init.zeros_(self.subject_output.bias)
        nn.init.zeros_(self.reference_output.weight)
        nn.init.zeros_(self.reference_output.bias)

    def forward(
        self,
        *,
        clause_hidden: torch.Tensor,
        subject_queries: torch.Tensor,
        reference_queries: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if not (
            clause_hidden.shape == subject_queries.shape == reference_queries.shape
        ):
            raise ValueError(
                "ERAF role-adapter clause/subject/reference shapes must match."
            )
        features = self.input_norm(
            torch.cat((clause_hidden, subject_queries, reference_queries), dim=-1)
        )
        hidden = self.shared(features)
        subject_delta = self.subject_output(hidden)
        reference_delta = self.reference_output(hidden)
        return {
            "subject_queries": subject_queries + subject_delta,
            "reference_queries": reference_queries + reference_delta,
            "subject_delta": subject_delta,
            "reference_delta": reference_delta,
        }


class StructuredRoleAssignmentAdapter(nn.Module):
    """Zero-init cross-clause repair layered on the frozen V9.3 adapter.

    V9.3 repairs each clause independently, so a subject can beat its local
    reference while still binding to an entity owned by another clause.  V9.4
    exposes every subject/reference slot to one shared transformer before
    predicting a second residual.  The final projections remain zero at
    migration time, preserving the complete V9.3 deployment path exactly.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        adapter_hidden_dim: int,
        num_heads: int,
        max_clauses: int,
    ) -> None:
        super().__init__()
        if min(hidden_dim, adapter_hidden_dim, num_heads, max_clauses) <= 0:
            raise ValueError("ERAF structured-role dimensions must be positive.")
        if adapter_hidden_dim % num_heads:
            raise ValueError(
                "ERAF structured-role hidden dimension must be divisible by heads."
            )
        self.hidden_dim = int(hidden_dim)
        self.adapter_hidden_dim = int(adapter_hidden_dim)
        self.max_clauses = int(max_clauses)
        self.input_norm = nn.LayerNorm(hidden_dim * 3)
        self.input_projection = nn.Linear(hidden_dim * 3, adapter_hidden_dim)
        self.clause_embedding = nn.Parameter(
            torch.zeros(max_clauses, adapter_hidden_dim)
        )
        self.role_embedding = nn.Parameter(torch.zeros(2, adapter_hidden_dim))
        nn.init.normal_(self.clause_embedding, std=0.02)
        nn.init.normal_(self.role_embedding, std=0.02)
        self.encoder = nn.TransformerEncoderLayer(
            d_model=adapter_hidden_dim,
            nhead=num_heads,
            dim_feedforward=adapter_hidden_dim * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.output_norm = nn.LayerNorm(adapter_hidden_dim)
        self.subject_output = nn.Linear(adapter_hidden_dim, hidden_dim)
        self.reference_output = nn.Linear(adapter_hidden_dim, hidden_dim)
        nn.init.zeros_(self.subject_output.weight)
        nn.init.zeros_(self.subject_output.bias)
        nn.init.zeros_(self.reference_output.weight)
        nn.init.zeros_(self.reference_output.bias)

    def forward(
        self,
        *,
        clause_hidden: torch.Tensor,
        subject_queries: torch.Tensor,
        reference_queries: torch.Tensor,
        active_logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if not (
            clause_hidden.shape == subject_queries.shape == reference_queries.shape
        ):
            raise ValueError(
                "ERAF structured-role clause/subject/reference shapes must match."
            )
        if clause_hidden.shape[1] != self.max_clauses:
            raise ValueError(
                "ERAF structured-role input does not match configured clauses."
            )
        if active_logits.shape != clause_hidden.shape[:2]:
            raise ValueError(
                "ERAF structured-role active logits must match [B,clauses]."
            )
        shared = self.input_projection(
            self.input_norm(
                torch.cat((clause_hidden, subject_queries, reference_queries), dim=-1)
            )
        )
        clause = self.clause_embedding.unsqueeze(0)
        active = torch.sigmoid(active_logits.float()).to(shared.dtype).unsqueeze(-1)
        subject_slot = shared + clause + self.role_embedding[0]
        reference_slot = shared + clause + self.role_embedding[1]
        slots = torch.cat((subject_slot, reference_slot), dim=1)
        encoded = self.encoder(slots)
        encoded = self.output_norm(encoded)
        subject_hidden, reference_hidden = encoded.split(self.max_clauses, dim=1)
        subject_delta = self.subject_output(subject_hidden) * active
        reference_delta = self.reference_output(reference_hidden) * active
        return {
            "subject_queries": subject_queries + subject_delta,
            "reference_queries": reference_queries + reference_delta,
            "subject_delta": subject_delta,
            "reference_delta": reference_delta,
        }


class BalancedRoleBindingAdapter(nn.Module):
    """Visual-candidate-aware V9.5 repair on top of the frozen V9.3 path.

    V9.4 only exchanged language-side clause slots.  That leaves the adapter
    unable to distinguish two clauses whose linguistic structure is similar
    but whose same-state visual candidates differ.  V9.5 injects the frozen
    V9.3 subject/reference visual tokens into the cross-clause slots before a
    new zero-initialized residual is predicted.  The frozen tokens are
    detached by the caller, so assignment gradients cannot update the V9.3
    grounder, position head, relation reasoner, or anchor heads.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        adapter_hidden_dim: int,
        num_heads: int,
        max_clauses: int,
    ) -> None:
        super().__init__()
        if min(hidden_dim, adapter_hidden_dim, num_heads, max_clauses) <= 0:
            raise ValueError("ERAF balanced-role dimensions must be positive.")
        if adapter_hidden_dim % num_heads:
            raise ValueError(
                "ERAF balanced-role hidden dimension must be divisible by heads."
            )
        self.hidden_dim = int(hidden_dim)
        self.adapter_hidden_dim = int(adapter_hidden_dim)
        self.max_clauses = int(max_clauses)
        self.input_norm = nn.LayerNorm(hidden_dim * 4)
        self.input_projection = nn.Linear(hidden_dim * 4, adapter_hidden_dim)
        self.clause_embedding = nn.Parameter(
            torch.zeros(max_clauses, adapter_hidden_dim)
        )
        self.role_embedding = nn.Parameter(torch.zeros(2, adapter_hidden_dim))
        nn.init.normal_(self.clause_embedding, std=0.02)
        nn.init.normal_(self.role_embedding, std=0.02)
        self.encoder = nn.TransformerEncoderLayer(
            d_model=adapter_hidden_dim,
            nhead=num_heads,
            dim_feedforward=adapter_hidden_dim * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.output_norm = nn.LayerNorm(adapter_hidden_dim)
        self.subject_output = nn.Linear(adapter_hidden_dim, hidden_dim)
        self.reference_output = nn.Linear(adapter_hidden_dim, hidden_dim)
        nn.init.zeros_(self.subject_output.weight)
        nn.init.zeros_(self.subject_output.bias)
        nn.init.zeros_(self.reference_output.weight)
        nn.init.zeros_(self.reference_output.bias)

    def forward(
        self,
        *,
        clause_hidden: torch.Tensor,
        subject_queries: torch.Tensor,
        reference_queries: torch.Tensor,
        subject_candidates: torch.Tensor,
        reference_candidates: torch.Tensor,
        active_logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        expected = clause_hidden.shape
        if not all(
            value.shape == expected
            for value in (
                subject_queries,
                reference_queries,
                subject_candidates,
                reference_candidates,
            )
        ):
            raise ValueError(
                "ERAF balanced-role clause/query/candidate shapes must match."
            )
        if clause_hidden.shape[1] != self.max_clauses:
            raise ValueError(
                "ERAF balanced-role input does not match configured clauses."
            )
        if active_logits.shape != clause_hidden.shape[:2]:
            raise ValueError("ERAF balanced-role active logits must match [B,clauses].")

        def slot(
            query: torch.Tensor, candidate: torch.Tensor, role_index: int
        ) -> torch.Tensor:
            projected = self.input_projection(
                self.input_norm(
                    torch.cat(
                        (
                            clause_hidden,
                            query,
                            candidate.detach(),
                            query - candidate.detach(),
                        ),
                        dim=-1,
                    )
                )
            )
            return (
                projected
                + self.clause_embedding.unsqueeze(0)
                + self.role_embedding[role_index]
            )

        subject_slot = slot(subject_queries, subject_candidates, 0)
        reference_slot = slot(reference_queries, reference_candidates, 1)
        slots = torch.cat((subject_slot, reference_slot), dim=1)
        encoded = self.output_norm(self.encoder(slots))
        subject_hidden, reference_hidden = encoded.split(self.max_clauses, dim=1)
        active = torch.sigmoid(active_logits.float()).to(encoded.dtype).unsqueeze(-1)
        subject_delta = self.subject_output(subject_hidden) * active
        reference_delta = self.reference_output(reference_hidden) * active
        return {
            "subject_queries": subject_queries + subject_delta,
            "reference_queries": reference_queries + reference_delta,
            "subject_delta": subject_delta,
            "reference_delta": reference_delta,
        }


class ClauseActivationCalibrationAdapter(nn.Module):
    """Zero-init V9.8 calibration for clause activity and cardinality.

    V9.7 fixes subject/reference binding but deliberately freezes the
    ``PredicateRoleDecoder``.  Consequently its multi-clause gate cannot move:
    that gate is computed from the decoder's active logits, not from the role
    adapter.  This sidecar reads the frozen clause states and base active logits,
    exchanges information across clause slots, and predicts a residual on the
    active logits plus an auxiliary clause-count distribution.

    Only the final projections are zero initialized.  A freshly migrated V9.8
    therefore has exactly the V9.7 deployment output while still receiving
    gradients on the first optimizer step.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        adapter_hidden_dim: int,
        num_heads: int,
        max_clauses: int,
        residual_max_abs: float = 4.0,
    ) -> None:
        super().__init__()
        if min(hidden_dim, adapter_hidden_dim, num_heads, max_clauses) <= 0:
            raise ValueError("ERAF clause-calibration dimensions must be positive.")
        if adapter_hidden_dim % num_heads:
            raise ValueError(
                "ERAF clause-calibration hidden dimension must be divisible by heads."
            )
        if residual_max_abs <= 0:
            raise ValueError(
                "ERAF clause-calibration residual bound must be positive."
            )
        self.hidden_dim = int(hidden_dim)
        self.adapter_hidden_dim = int(adapter_hidden_dim)
        self.max_clauses = int(max_clauses)
        self.residual_max_abs = float(residual_max_abs)
        self.input_norm = nn.LayerNorm(hidden_dim + 1)
        self.input_projection = nn.Linear(hidden_dim + 1, adapter_hidden_dim)
        self.clause_embedding = nn.Parameter(
            torch.zeros(max_clauses, adapter_hidden_dim)
        )
        nn.init.normal_(self.clause_embedding, std=0.02)
        self.encoder = nn.TransformerEncoderLayer(
            d_model=adapter_hidden_dim,
            nhead=num_heads,
            dim_feedforward=adapter_hidden_dim * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.output_norm = nn.LayerNorm(adapter_hidden_dim)
        self.active_output = nn.Linear(adapter_hidden_dim, 1)
        self.cardinality_output = nn.Linear(adapter_hidden_dim, max_clauses + 1)
        nn.init.zeros_(self.active_output.weight)
        nn.init.zeros_(self.active_output.bias)
        nn.init.zeros_(self.cardinality_output.weight)
        nn.init.zeros_(self.cardinality_output.bias)

    def forward(
        self,
        *,
        clause_hidden: torch.Tensor,
        active_logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if clause_hidden.ndim != 3 or clause_hidden.shape[-1] != self.hidden_dim:
            raise ValueError(
                "ERAF clause-calibration hidden states must be [B,C,hidden_dim]."
            )
        if clause_hidden.shape[1] != self.max_clauses:
            raise ValueError(
                "ERAF clause-calibration input does not match configured clauses."
            )
        if active_logits.shape != clause_hidden.shape[:2]:
            raise ValueError(
                "ERAF clause-calibration active logits must match [B,clauses]."
            )
        base_active = active_logits.to(clause_hidden.dtype).unsqueeze(-1)
        hidden = self.input_projection(
            self.input_norm(torch.cat((clause_hidden, base_active), dim=-1))
        )
        hidden = hidden + self.clause_embedding.unsqueeze(0)
        hidden = self.output_norm(self.encoder(hidden))
        raw_residual = self.active_output(hidden).squeeze(-1)
        active_residual = (
            torch.tanh(raw_residual.float()) * self.residual_max_abs
        ).to(active_logits.dtype)
        cardinality_logits = self.cardinality_output(hidden.mean(dim=1))
        return {
            "active_logits": active_logits + active_residual,
            "base_active_logits": active_logits,
            "active_residual": active_residual,
            "cardinality_logits": cardinality_logits,
        }


class EntityRelationAffordanceField(nn.Module):
    """Complete RGB+language ERAF deployment path used by PGC V9."""

    def __init__(
        self,
        *,
        text_dim: int,
        video_dim: int,
        action_dim: int,
        projection_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        max_clauses: int = 4,
        camera_count: int = 2,
        visual_aspect_ratio: float = 2.0,
        temperature: float = 0.07,
        entity_only: bool = False,
        use_anchors: bool = True,
        role_adapter_enabled: bool = False,
        role_adapter_hidden_dim: int = 256,
        role_adapter_teacher_enabled: bool = False,
        structured_role_adapter_enabled: bool = False,
        structured_role_adapter_hidden_dim: int = 256,
        balanced_role_adapter_enabled: bool = False,
        balanced_role_adapter_hidden_dim: int = 256,
        clause_activation_adapter_enabled: bool = False,
        clause_activation_adapter_hidden_dim: int = 256,
        clause_activation_residual_max_abs: float = 4.0,
    ) -> None:
        super().__init__()
        self.text_dim = int(text_dim)
        self.video_dim = int(video_dim)
        self.action_dim = int(action_dim)
        self.projection_dim = int(projection_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.max_clauses = int(max_clauses)
        self.camera_count = int(camera_count)
        self.entity_only = bool(entity_only)
        self.use_anchors = bool(use_anchors)
        self.role_adapter_enabled = bool(role_adapter_enabled)
        self.role_adapter_hidden_dim = int(role_adapter_hidden_dim)
        self.role_adapter_teacher_enabled = bool(role_adapter_teacher_enabled)
        self.structured_role_adapter_enabled = bool(structured_role_adapter_enabled)
        self.structured_role_adapter_hidden_dim = int(
            structured_role_adapter_hidden_dim
        )
        self.balanced_role_adapter_enabled = bool(balanced_role_adapter_enabled)
        self.balanced_role_adapter_hidden_dim = int(balanced_role_adapter_hidden_dim)
        self.clause_activation_adapter_enabled = bool(
            clause_activation_adapter_enabled
        )
        self.clause_activation_adapter_hidden_dim = int(
            clause_activation_adapter_hidden_dim
        )
        self.clause_activation_residual_max_abs = float(
            clause_activation_residual_max_abs
        )
        self.role_decoder = PredicateRoleDecoder(
            text_dim=text_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            max_clauses=max_clauses,
        )
        self.entity_grounder = MultiViewEntityGrounder(
            video_dim=video_dim,
            hidden_dim=hidden_dim,
            camera_count=camera_count,
            visual_aspect_ratio=visual_aspect_ratio,
            temperature=temperature,
        )
        self.relation_reasoner = RelationAffordanceReasoner(
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            projection_dim=projection_dim,
        )
        self.role_assignment_adapter = (
            RoleAssignmentResidualAdapter(
                hidden_dim=hidden_dim,
                adapter_hidden_dim=role_adapter_hidden_dim,
            )
            if self.role_adapter_enabled
            else None
        )
        self.structured_role_assignment_adapter = (
            StructuredRoleAssignmentAdapter(
                hidden_dim=hidden_dim,
                adapter_hidden_dim=structured_role_adapter_hidden_dim,
                num_heads=num_heads,
                max_clauses=max_clauses,
            )
            if self.structured_role_adapter_enabled
            else None
        )
        self.balanced_role_binding_adapter = (
            BalancedRoleBindingAdapter(
                hidden_dim=hidden_dim,
                adapter_hidden_dim=balanced_role_adapter_hidden_dim,
                num_heads=num_heads,
                max_clauses=max_clauses,
            )
            if self.balanced_role_adapter_enabled
            else None
        )
        self.clause_activation_adapter = (
            ClauseActivationCalibrationAdapter(
                hidden_dim=hidden_dim,
                adapter_hidden_dim=clause_activation_adapter_hidden_dim,
                num_heads=num_heads,
                max_clauses=max_clauses,
                residual_max_abs=clause_activation_residual_max_abs,
            )
            if self.clause_activation_adapter_enabled
            else None
        )
        self.entity_only_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.base_query_projection = nn.Sequential(
            nn.LayerNorm(action_dim), nn.Linear(action_dim, hidden_dim)
        )
        self.relation_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.query_delta_projection = nn.Linear(hidden_dim, action_dim)
        self.embedding_delta_projection = nn.Linear(projection_dim, projection_dim)
        nn.init.zeros_(self.query_delta_projection.weight)
        nn.init.zeros_(self.query_delta_projection.bias)
        nn.init.zeros_(self.embedding_delta_projection.weight)
        nn.init.zeros_(self.embedding_delta_projection.bias)

    def _decode_affordance(
        self,
        *,
        clause_hidden: torch.Tensor,
        subject_token: torch.Tensor,
        reference_token: torch.Tensor,
        subject_position: torch.Tensor,
        reference_position: torch.Tensor,
        active_logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.entity_only:
            entity_hidden = self.entity_only_projection(subject_token)
            entity_hidden = entity_hidden * torch.sigmoid(active_logits.float()).to(
                entity_hidden.dtype
            ).unsqueeze(-1)
            return self.relation_reasoner.decode_relation_hidden(entity_hidden)
        if not self.use_anchors:
            subject_position = torch.zeros_like(subject_position)
            reference_position = torch.zeros_like(reference_position)
        return self.relation_reasoner(
            clause_hidden=clause_hidden,
            subject_token=subject_token,
            reference_token=reference_token,
            subject_position=subject_position,
            reference_position=reference_position,
            active_logits=active_logits,
        )

    def _route_relation(
        self,
        *,
        base_goal_queries: torch.Tensor,
        base_goal_embedding: torch.Tensor,
        relation_hidden: torch.Tensor,
        embedding_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = self.base_query_projection(base_goal_queries)
        query_delta, query_attention = self.relation_attention(
            query=query,
            key=relation_hidden,
            value=relation_hidden,
            need_weights=True,
        )
        routed_queries = base_goal_queries + self.query_delta_projection(query_delta)
        pooled_embedding = embedding_tokens.mean(dim=1)
        routed_embedding = base_goal_embedding + self.embedding_delta_projection(
            pooled_embedding
        ).to(base_goal_embedding.dtype)
        return routed_queries, routed_embedding, query_attention

    def negative_goal_queries(
        self,
        *,
        base_goal_queries: torch.Tensor,
        base_goal_embedding: torch.Tensor,
        outputs: Mapping[str, torch.Tensor],
        kind: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Construct explicit wrong-entity or wrong-relation verifier negatives."""
        kind = str(kind).strip().lower()
        if kind == "entity":
            clause_hidden = outputs["clause_hidden"]
            # For binary predicates the grounded reference is a real,
            # same-state wrong subject (for example the basket instead of the
            # soup). Unary fixture predicates bind subject/reference to the
            # same fixture, so they fall back to the least-compatible RGB
            # patch. This keeps every negative label-free at deployment while
            # avoiding the old no-op unary role swap.
            visual_tokens = outputs["visual_tokens"]
            wrong_indices = outputs["subject_attention"].argmin(dim=-1)
            gather_indices = wrong_indices.unsqueeze(-1).expand(
                -1, -1, visual_tokens.shape[-1]
            )
            expanded_visual = visual_tokens.unsqueeze(1).expand(
                -1, wrong_indices.shape[1], -1, -1
            )
            fallback_subject = torch.gather(
                expanded_visual, 2, gather_indices.unsqueeze(2)
            ).squeeze(2)
            reference_token = outputs["reference_token"]
            reference_position = outputs["reference_position"]
            predicted_ids = outputs["predicate_logits"].argmax(dim=-1)
            is_binary = (predicted_ids >= ERAF_PREDICATE_TO_ID["in"]) & (
                predicted_ids <= ERAF_PREDICATE_TO_ID["back"]
            )
            subject_token = torch.where(
                is_binary.unsqueeze(-1), reference_token, fallback_subject
            )
            predicted_subject_position = self.entity_grounder.position_head(
                subject_token
            ).float()
            subject_position = torch.where(
                is_binary.unsqueeze(-1),
                reference_position,
                predicted_subject_position,
            )
        elif kind == "relation":
            probabilities = torch.softmax(outputs["predicate_logits"].float(), dim=-1)
            current_predicate = torch.matmul(
                probabilities.to(outputs["clause_hidden"].dtype),
                self.role_decoder.predicate_embedding.weight,
            )
            predicted_ids = outputs["predicate_logits"].argmax(dim=-1)
            # Rotate among real predicates and never introduce the padding ID.
            wrong_ids = 1 + (predicted_ids.clamp_min(1) % (len(ERAF_PREDICATES) - 1))
            wrong_predicate = self.role_decoder.predicate_embedding(wrong_ids)
            clause_hidden = (
                outputs["clause_hidden"] - current_predicate + wrong_predicate
            )
            subject_token = outputs["subject_token"]
            reference_token = outputs["reference_token"]
            subject_position = outputs["subject_position"]
            reference_position = outputs["reference_position"]
        else:
            raise ValueError("ERAF negative kind must be entity or relation.")
        negative = self._decode_affordance(
            clause_hidden=clause_hidden,
            subject_token=subject_token,
            reference_token=reference_token,
            subject_position=subject_position,
            reference_position=reference_position,
            active_logits=outputs["active_logits"],
        )
        queries, embedding, _ = self._route_relation(
            base_goal_queries=base_goal_queries,
            base_goal_embedding=base_goal_embedding,
            relation_hidden=negative["relation_hidden"],
            embedding_tokens=negative["embedding_tokens"],
        )
        return queries, embedding

    def forward(
        self,
        *,
        base_goal_queries: torch.Tensor,
        base_goal_embedding: torch.Tensor,
        language_hidden: torch.Tensor,
        language_mask: torch.Tensor,
        current_video_hidden: torch.Tensor,
    ) -> tuple[
        torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]
    ]:
        roles = self.role_decoder(language_hidden, language_mask)
        teacher: dict[str, torch.Tensor] | None = None
        base_subject_queries = roles["subject_queries"]
        base_reference_queries = roles["reference_queries"]
        if self.role_assignment_adapter is not None:
            adapted_roles = self.role_assignment_adapter(
                clause_hidden=roles["clause_hidden"],
                subject_queries=base_subject_queries,
                reference_queries=base_reference_queries,
            )
            roles = {
                **roles,
                "subject_queries": adapted_roles["subject_queries"],
                "reference_queries": adapted_roles["reference_queries"],
            }
            subject_role_delta = adapted_roles["subject_delta"]
            reference_role_delta = adapted_roles["reference_delta"]
        else:
            subject_role_delta = torch.zeros_like(base_subject_queries)
            reference_role_delta = torch.zeros_like(base_reference_queries)
        teacher_subject: dict[str, torch.Tensor] | None = None
        teacher_reference: dict[str, torch.Tensor] | None = None
        needs_visual_candidates = self.balanced_role_binding_adapter is not None
        needs_teacher_outputs = torch.is_grad_enabled()
        if self.role_adapter_teacher_enabled and (
            needs_teacher_outputs or needs_visual_candidates
        ):
            # V9.3 preserves the exact V9.1 path before its local adapter.
            # V9.4 layers a second adapter on top and instead freezes the
            # complete validated V9.3 role path as its same-state teacher.
            if (
                self.structured_role_assignment_adapter is not None
                or self.balanced_role_binding_adapter is not None
            ):
                teacher_subject_queries = roles["subject_queries"]
                teacher_reference_queries = roles["reference_queries"]
            else:
                teacher_subject_queries = base_subject_queries
                teacher_reference_queries = base_reference_queries
            with torch.no_grad():
                teacher_subject = self.entity_grounder.ground(
                    teacher_subject_queries, current_video_hidden
                )
                teacher_reference = self.entity_grounder.ground(
                    teacher_reference_queries, current_video_hidden
                )
                teacher_affordance = (
                    self._decode_affordance(
                        clause_hidden=roles["clause_hidden"],
                        subject_token=teacher_subject["token"],
                        reference_token=teacher_reference["token"],
                        subject_position=teacher_subject["position"],
                        reference_position=teacher_reference["position"],
                        active_logits=roles["active_logits"],
                    )
                    if needs_teacher_outputs
                    else None
                )
            if teacher_affordance is not None:
                teacher = {
                    "subject_attention": teacher_subject["attention"].detach(),
                    "reference_attention": teacher_reference["attention"].detach(),
                    "subject_position": teacher_subject["position"].detach(),
                    "reference_position": teacher_reference["position"].detach(),
                    "relation_hidden": teacher_affordance["relation_hidden"].detach(),
                    "grasp_anchor": teacher_affordance["grasp_anchor"].detach(),
                    "goal_anchor": teacher_affordance["goal_anchor"].detach(),
                    "interaction_anchor": teacher_affordance[
                        "interaction_anchor"
                    ].detach(),
                    "active_logits": roles["active_logits"].detach(),
                    "predicate_logits": roles["predicate_logits"].detach(),
                }
        if self.structured_role_assignment_adapter is not None:
            structured_roles = self.structured_role_assignment_adapter(
                clause_hidden=roles["clause_hidden"],
                subject_queries=roles["subject_queries"],
                reference_queries=roles["reference_queries"],
                active_logits=roles["active_logits"],
            )
            roles = {
                **roles,
                "subject_queries": structured_roles["subject_queries"],
                "reference_queries": structured_roles["reference_queries"],
            }
            structured_subject_role_delta = structured_roles["subject_delta"]
            structured_reference_role_delta = structured_roles["reference_delta"]
        else:
            structured_subject_role_delta = torch.zeros_like(base_subject_queries)
            structured_reference_role_delta = torch.zeros_like(base_reference_queries)
        if self.balanced_role_binding_adapter is not None:
            if teacher_subject is None or teacher_reference is None:
                raise RuntimeError(
                    "V9.5 balanced binding requires frozen V9.3 visual candidates."
                )
            balanced_roles = self.balanced_role_binding_adapter(
                clause_hidden=roles["clause_hidden"],
                subject_queries=roles["subject_queries"],
                reference_queries=roles["reference_queries"],
                subject_candidates=teacher_subject["token"],
                reference_candidates=teacher_reference["token"],
                active_logits=roles["active_logits"],
            )
            roles = {
                **roles,
                "subject_queries": balanced_roles["subject_queries"],
                "reference_queries": balanced_roles["reference_queries"],
            }
            balanced_subject_role_delta = balanced_roles["subject_delta"]
            balanced_reference_role_delta = balanced_roles["reference_delta"]
        else:
            balanced_subject_role_delta = torch.zeros_like(base_subject_queries)
            balanced_reference_role_delta = torch.zeros_like(base_reference_queries)
        # V9.8 calibrates clause activity only after every frozen role adapter
        # has produced its visual query.  This prevents the new clause loss
        # from moving the V9.7 subject/reference heatmaps that already passed
        # their localization and exclusive-role gates.
        if self.clause_activation_adapter is not None:
            calibrated = self.clause_activation_adapter(
                clause_hidden=roles["clause_hidden"],
                active_logits=roles["active_logits"],
            )
            roles = {**roles, "active_logits": calibrated["active_logits"]}
            base_active_logits = calibrated["base_active_logits"]
            clause_active_residual = calibrated["active_residual"]
            clause_cardinality_logits = calibrated["cardinality_logits"]
        else:
            base_active_logits = roles["active_logits"]
            clause_active_residual = torch.zeros_like(roles["active_logits"])
            clause_cardinality_logits = roles["active_logits"].new_zeros(
                roles["active_logits"].shape[0], self.max_clauses + 1
            )
        subject = self.entity_grounder.ground(
            roles["subject_queries"], current_video_hidden
        )
        reference = self.entity_grounder.ground(
            roles["reference_queries"], current_video_hidden
        )
        affordance = self._decode_affordance(
            clause_hidden=roles["clause_hidden"],
            subject_token=subject["token"],
            reference_token=reference["token"],
            subject_position=subject["position"],
            reference_position=reference["position"],
            active_logits=roles["active_logits"],
        )
        # Do not renormalize here: both zero bridge projections must make a
        # freshly migrated V9 bitwise identical to the V5 sidecar path.
        routed_queries, routed_embedding, query_attention = self._route_relation(
            base_goal_queries=base_goal_queries,
            base_goal_embedding=base_goal_embedding,
            relation_hidden=affordance["relation_hidden"],
            embedding_tokens=affordance["embedding_tokens"],
        )
        subject_probs = subject["attention"].float().clamp_min(1e-8)
        reference_probs = reference["attention"].float().clamp_min(1e-8)
        query_probs = query_attention.float().clamp_min(1e-8)
        metrics = {
            "pgc_v9_subject_top1_mass": subject_probs.max(dim=-1).values.mean(),
            "pgc_v9_reference_top1_mass": reference_probs.max(dim=-1).values.mean(),
            "pgc_v9_subject_attention_entropy": (
                -(subject_probs * subject_probs.log()).sum(dim=-1)
                / math.log(max(2, subject_probs.shape[-1]))
            ).mean(),
            "pgc_v9_reference_attention_entropy": (
                -(reference_probs * reference_probs.log()).sum(dim=-1)
                / math.log(max(2, reference_probs.shape[-1]))
            ).mean(),
            "pgc_v9_clause_active_probability": torch.sigmoid(
                roles["active_logits"].float()
            ).mean(),
            "pgc_v9_relation_query_attention_entropy": (
                -(query_probs * query_probs.log()).sum(dim=-1)
                / math.log(max(2, query_probs.shape[-1]))
            ).mean(),
            "pgc_v9_query_delta_norm": (routed_queries - base_goal_queries)
            .float()
            .norm(dim=-1)
            .mean(),
            "pgc_v9_role_adapter_subject_delta_norm": subject_role_delta.float()
            .norm(dim=-1)
            .mean(),
            "pgc_v9_role_adapter_reference_delta_norm": reference_role_delta.float()
            .norm(dim=-1)
            .mean(),
            "pgc_v9_structured_role_adapter_subject_delta_norm": (
                structured_subject_role_delta.float().norm(dim=-1).mean()
            ),
            "pgc_v9_structured_role_adapter_reference_delta_norm": (
                structured_reference_role_delta.float().norm(dim=-1).mean()
            ),
            "pgc_v9_balanced_role_adapter_subject_delta_norm": (
                balanced_subject_role_delta.float().norm(dim=-1).mean()
            ),
            "pgc_v9_balanced_role_adapter_reference_delta_norm": (
                balanced_reference_role_delta.float().norm(dim=-1).mean()
            ),
            "pgc_v9_clause_active_residual_rms": (
                clause_active_residual.float().pow(2).mean().sqrt()
            ),
            "pgc_v9_clause_active_residual_max_abs": (
                clause_active_residual.float().abs().max()
            ),
            "pgc_v9_clause_active_residual_saturation": (
                (
                    clause_active_residual.float().abs()
                    >= 0.95 * self.clause_activation_residual_max_abs
                )
                .float()
                .mean()
            ),
        }
        outputs = {
            **roles,
            "base_active_logits": base_active_logits,
            "clause_active_residual": clause_active_residual,
            "clause_cardinality_logits": clause_cardinality_logits,
            "subject_attention": subject["attention"],
            "reference_attention": reference["attention"],
            "subject_similarity": subject["similarity"],
            "reference_similarity": reference["similarity"],
            "subject_token": subject["token"],
            "reference_token": reference["token"],
            "visual_tokens": subject["visual_tokens"],
            "subject_position": subject["position"],
            "reference_position": reference["position"],
            "subject_visibility_logits": subject["visibility_logits"],
            "reference_visibility_logits": reference["visibility_logits"],
            "subject_view_visibility_logits": subject["view_visibility_logits"],
            "reference_view_visibility_logits": reference["view_visibility_logits"],
            "subject_view_centers": subject["view_centers"],
            "reference_view_centers": reference["view_centers"],
            "subject_view_attention_mass": subject["view_attention_mass"],
            "reference_view_attention_mass": reference["view_attention_mass"],
            "spatial_coordinates": subject["coordinates"],
            "camera_ids": subject["camera_ids"],
            "subject_role_delta": subject_role_delta,
            "reference_role_delta": reference_role_delta,
            "structured_subject_role_delta": structured_subject_role_delta,
            "structured_reference_role_delta": structured_reference_role_delta,
            "balanced_subject_role_delta": balanced_subject_role_delta,
            "balanced_reference_role_delta": balanced_reference_role_delta,
            **affordance,
        }
        if teacher is not None:
            outputs.update(
                {f"teacher_{name}": value for name, value in teacher.items()}
            )
        return routed_queries, routed_embedding, outputs, metrics


@dataclass(frozen=True)
class ERAFLossWeights:
    objective_version: int = 1
    mask: float = 1.0
    attention_mask: float = 0.0
    entity: float = 1.0
    relation: float = 1.0
    anchor: float = 1.0
    position: float = 0.5
    role_swap: float = 0.5
    role_overlap: float = 0.0
    role_swap_margin: float = 0.20
    role_assignment: float = 0.0
    role_assignment_temperature: float = 0.10
    role_assignment_hard_weight: float = 0.0
    role_attention_preservation: float = 0.0
    role_position_preservation: float = 0.0
    role_anchor_preservation: float = 0.0
    role_relation_preservation: float = 0.0
    role_adapter_energy: float = 0.0
    structured_assignment: float = 0.0
    structured_assignment_temperature: float = 0.10
    structured_assignment_hard_weight: float = 0.0
    multi_clause_consistency: float = 0.0
    clause_activation_balance: float = 0.0
    clause_cardinality: float = 0.0
    clause_worst_slot: float = 0.0
    clause_multi_group_weight: float = 1.0
    clause_adapter_energy: float = 0.0
    phase: float = 0.25


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=values.device, dtype=torch.bool)
    if not bool(mask.any()):
        return torch.nan_to_num(
            values, nan=0.0, posinf=0.0, neginf=0.0
        ).sum() * 0.0
    return values[mask].mean()


def _masked_weighted_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    sample_weights: torch.Tensor,
) -> torch.Tensor:
    mask = mask.to(device=values.device, dtype=torch.bool)
    if not bool(mask.any()):
        return torch.nan_to_num(
            values, nan=0.0, posinf=0.0, neginf=0.0
        ).sum() * 0.0
    selected_weights = sample_weights.to(device=values.device, dtype=values.dtype)[mask]
    return (values[mask] * selected_weights).sum() / selected_weights.sum().clamp_min(
        1.0e-8
    )


def _supervised_role_contrastive_loss(
    queries: torch.Tensor,
    visual_tokens: torch.Tensor,
    entity_ids: torch.Tensor,
    valid: torch.Tensor,
    *,
    temperature: float = 0.07,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match role queries to same-entity visual tokens across the batch.

    Multiple clauses may name the same object.  They are treated as multiple
    positives rather than false negatives, which is important for conjunction
    tasks and repeated fixture references.
    """
    flat_valid = valid.reshape(-1).bool() & (entity_ids.reshape(-1) >= 0)
    if int(flat_valid.sum()) < 2:
        zero = queries.sum() * 0.0
        return zero, zero.detach()
    q = F.normalize(
        queries.reshape(-1, queries.shape[-1])[flat_valid].float(),
        dim=-1,
        eps=1.0e-6,
    )
    v = F.normalize(
        visual_tokens.reshape(-1, visual_tokens.shape[-1])[flat_valid].float(),
        dim=-1,
        eps=1.0e-6,
    )
    ids = entity_ids.reshape(-1)[flat_valid].long()
    logits = torch.matmul(q, v.transpose(0, 1)) / float(temperature)
    positive = ids[:, None] == ids[None, :]
    log_prob = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    loss = -torch.logsumexp(log_prob.masked_fill(~positive, -torch.inf), dim=-1)
    accuracy = positive.gather(1, logits.argmax(dim=-1, keepdim=True)).float().mean()
    return loss.mean(), accuracy


def _structured_all_entity_assignment_loss(
    *,
    role_attentions: Mapping[str, torch.Tensor],
    role_targets: Mapping[str, torch.Tensor],
    role_entity_ids: Mapping[str, torch.Tensor],
    role_valid: Mapping[str, torch.Tensor],
    clause_valid: torch.Tensor,
    temperature: float,
    hard_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Assign every role query against every visible same-state entity.

    Candidate slots are built from all subject/reference masks in the sample.
    Slots carrying the query's entity ID are positives and excluded from the
    negative bank, so conjunctions may legally share one basket/reference.
    Every different entity, including entities owned by another clause, is a
    hard negative.  The multi-clause term backpropagates through the worst role
    in each compound instruction rather than allowing easy clauses to hide it.
    """
    if temperature <= 0:
        raise ValueError("ERAF structured-assignment temperature must be positive.")
    if hard_weight < 0:
        raise ValueError("ERAF structured-assignment hard weight must be non-negative.")
    candidates = torch.cat((role_targets["subject"], role_targets["reference"]), dim=1)
    candidate_ids = torch.cat(
        (role_entity_ids["subject"], role_entity_ids["reference"]), dim=1
    ).long()
    candidate_valid = torch.cat(
        (role_valid["subject"], role_valid["reference"]), dim=1
    ).bool() & (candidate_ids >= 0)

    query_losses: list[torch.Tensor] = []
    query_valids: list[torch.Tensor] = []
    query_corrects: list[torch.Tensor] = []
    query_margins: list[torch.Tensor] = []
    query_negative_counts: list[torch.Tensor] = []
    for role in ("subject", "reference"):
        attention = role_attentions[role].float()
        own_target = role_targets[role].float()
        entity_ids = role_entity_ids[role].long()
        valid = role_valid[role].bool() & (entity_ids >= 0)
        own_mass = (attention * own_target).sum(dim=-1)
        candidate_mass = torch.einsum("bct,bkt->bck", attention, candidates)
        negative_valid = (
            valid.unsqueeze(-1)
            & candidate_valid.unsqueeze(1)
            & (candidate_ids.unsqueeze(1) != entity_ids.unsqueeze(-1))
        )
        has_negative = negative_valid.any(dim=-1)
        valid = valid & has_negative
        wrong_logits = (candidate_mass / float(temperature)).masked_fill(
            ~negative_valid, -torch.inf
        )
        own_logits = own_mass / float(temperature)
        loss = (
            torch.logaddexp(own_logits, torch.logsumexp(wrong_logits, dim=-1))
            - own_logits
        )
        hardest_wrong = candidate_mass.masked_fill(~negative_valid, -torch.inf).amax(
            dim=-1
        )
        hardest_wrong = torch.where(
            has_negative, hardest_wrong, torch.zeros_like(hardest_wrong)
        )
        correct = own_mass > hardest_wrong
        query_losses.append(loss)
        query_valids.append(valid)
        query_corrects.append(correct)
        query_margins.append(own_mass - hardest_wrong)
        query_negative_counts.append(negative_valid.float().sum(dim=-1))

    losses = torch.cat(query_losses, dim=1)
    valid = torch.cat(query_valids, dim=1)
    correct = torch.cat(query_corrects, dim=1)
    margins = torch.cat(query_margins, dim=1)
    negative_counts = torch.cat(query_negative_counts, dim=1)
    hard = (~correct).float().detach()
    sample_weights = 1.0 + float(hard_weight) * hard
    assignment_loss = _masked_weighted_mean(losses, valid, sample_weights)

    # A compound task succeeds only if every grounded role is correct.  Focus
    # its extra loss on the weakest role, matching the all-clauses gate rather
    # than averaging the failure away.
    multi_sample = (clause_valid.bool().sum(dim=-1) > 1) & valid.any(dim=-1)
    worst_query_loss = losses.masked_fill(~valid, -torch.inf).amax(dim=-1)
    worst_query_loss = torch.where(
        valid.any(dim=-1), worst_query_loss, torch.zeros_like(worst_query_loss)
    )
    multi_clause_loss = _masked_mean(worst_query_loss, multi_sample)
    sample_correct = (correct | ~valid).all(dim=-1)
    zero = losses.sum() * 0.0
    metrics = {
        "accuracy": _masked_mean(correct.float(), valid).detach(),
        "column_accuracy": zero.detach(),
        "hard_fraction": _masked_mean(hard, valid).detach(),
        "margin": _masked_mean(margins, valid).detach(),
        "negative_count": _masked_mean(negative_counts, valid).detach(),
        "multi_clause_accuracy": (
            _masked_mean(sample_correct.float(), multi_sample).detach()
            if bool(multi_sample.any())
            else zero.detach()
        ),
        "multi_clause_fraction": multi_sample.float().mean().detach(),
        "hard_gradient_fraction": zero.detach(),
        "global_easy_count": zero.detach(),
        "global_hard_count": zero.detach(),
        "exclusive_coverage": zero.detach(),
    }
    return assignment_loss, multi_clause_loss, metrics


def _balanced_group_mean(
    values: torch.Tensor,
    valid: torch.Tensor,
    correct: torch.Tensor,
    *,
    hard_group_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Give easy/hard groups exact global DDP mass without NaN empty groups.

    DDP averages gradients across ranks.  Scaling each local sum by
    ``world_size / global_count`` therefore produces the gradient of the true
    global group mean, even when a rank has no member of that group.  The
    straight-through detached all-reduce keeps the forward loss identical on
    every rank while preserving that local autograd path.
    """
    if hard_group_weight <= 0:
        raise ValueError("ERAF balanced hard-group weight must be positive.")
    easy_valid = valid.bool() & correct.bool()
    hard_valid = valid.bool() & ~correct.bool()
    graph_zero = torch.nan_to_num(
        values, nan=0.0, posinf=0.0, neginf=0.0
    ).sum() * 0.0

    def global_group_mean(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        local_count = mask.sum().to(dtype=values.dtype)
        global_count = local_count.detach().clone()
        distributed = (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        world_size = 1
        if distributed:
            torch.distributed.all_reduce(
                global_count, op=torch.distributed.ReduceOp.SUM
            )
            world_size = torch.distributed.get_world_size()
        if float(global_count.item()) <= 0.0:
            return graph_zero, global_count
        if bool(mask.any()) and not bool(torch.isfinite(values[mask]).all()):
            raise FloatingPointError(
                "ERAF V9.6 received NaN/Inf in a valid assignment group."
            )
        local_sum = torch.where(
            mask,
            torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0),
            torch.zeros_like(values),
        ).sum()
        local_contribution = (
            local_sum * float(world_size) / global_count.clamp_min(1.0)
        )
        if not distributed:
            return local_contribution, global_count
        detached_global_sum = local_sum.detach().clone()
        torch.distributed.all_reduce(
            detached_global_sum, op=torch.distributed.ReduceOp.SUM
        )
        detached_global_mean = detached_global_sum / global_count.clamp_min(1.0)
        return (
            local_contribution
            + detached_global_mean
            - local_contribution.detach(),
            global_count,
        )

    easy_mean, global_easy_count = global_group_mean(easy_valid)
    hard_mean, global_hard_count = global_group_mean(hard_valid)
    easy_present = float(global_easy_count.item()) > 0.0
    hard_present = float(global_hard_count.item()) > 0.0
    if easy_present and hard_present:
        denominator = 1.0 + float(hard_group_weight)
        value = (
            easy_mean + float(hard_group_weight) * hard_mean
        ) / denominator
        hard_gradient_fraction = values.new_tensor(
            float(hard_group_weight) / denominator
        )
        return value, hard_gradient_fraction, global_easy_count, global_hard_count
    if hard_present:
        return hard_mean, values.new_tensor(1.0), global_easy_count, global_hard_count
    if easy_present:
        return easy_mean, values.new_tensor(0.0), global_easy_count, global_hard_count
    return graph_zero, graph_zero.detach(), global_easy_count, global_hard_count


def _clause_activation_calibration_loss(
    *,
    active_logits: torch.Tensor,
    cardinality_logits: torch.Tensor,
    clause_valid: torch.Tensor,
    multi_group_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Train V9.8 against clause activity, count, and exact-match failures.

    Single-clause and multi-clause samples receive fixed global DDP mass, so a
    large native single-clause pool cannot drown out the conjunction examples
    that define the hard gate.  The worst-slot term directly targets the
    all-slots-correct metric instead of allowing easy inactive padding slots to
    dominate mean BCE.
    """
    if active_logits.shape != clause_valid.shape:
        raise ValueError("ERAF clause activity must match clause_valid.")
    expected_cardinality_shape = (
        active_logits.shape[0],
        active_logits.shape[1] + 1,
    )
    if cardinality_logits.shape != expected_cardinality_shape:
        raise ValueError(
            "ERAF cardinality logits must be [B,max_clauses+1]."
        )
    if multi_group_weight <= 0:
        raise ValueError("ERAF multi-clause group weight must be positive.")
    target = clause_valid.bool()
    count_target = target.sum(dim=-1).long()
    sample_valid = count_target > 0
    multi_sample = count_target > 1
    slot_bce = F.binary_cross_entropy_with_logits(
        active_logits.float(), target.float(), reduction="none"
    )
    sample_bce = slot_bce.mean(dim=-1)
    balanced_active, multi_gradient_fraction, single_count, multi_count = (
        _balanced_group_mean(
            sample_bce,
            correct=~multi_sample,
            valid=sample_valid,
            hard_group_weight=float(multi_group_weight),
        )
    )
    cardinality_loss = _masked_mean(
        F.cross_entropy(
            cardinality_logits.float(), count_target, reduction="none"
        ),
        sample_valid,
    )
    worst_slot_loss = _masked_mean(slot_bce.amax(dim=-1), multi_sample)
    prediction = active_logits > 0
    exact = (prediction == target).all(dim=-1)
    cardinality_prediction = cardinality_logits.argmax(dim=-1)
    zero = active_logits.sum() * 0.0
    metrics = {
        "exact": _masked_mean(exact.float(), sample_valid).detach(),
        "multi_exact": (
            _masked_mean(exact.float(), multi_sample).detach()
            if bool(multi_sample.any())
            else zero.detach()
        ),
        "cardinality_accuracy": _masked_mean(
            (cardinality_prediction == count_target).float(), sample_valid
        ).detach(),
        "multi_fraction": _masked_mean(
            multi_sample.float(), sample_valid
        ).detach(),
        "multi_gradient_fraction": multi_gradient_fraction.detach(),
        "global_single_count": single_count.detach(),
        "global_multi_count": multi_count.detach(),
    }
    return balanced_active, cardinality_loss, worst_slot_loss, metrics


def _balanced_bipartite_assignment_loss(
    *,
    role_attentions: Mapping[str, torch.Tensor],
    role_targets: Mapping[str, torch.Tensor],
    role_entity_ids: Mapping[str, torch.Tensor],
    role_valid: Mapping[str, torch.Tensor],
    clause_valid: torch.Tensor,
    temperature: float,
    hard_group_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """V9.5 row/column assignment with balanced online hard mining.

    Rows are semantic role queries and columns are all visible same-state role
    entities.  Multi-positive IDs allow shared containers while the column
    term prevents unrelated queries from collapsing onto one entity.  Hard
    and easy rows/columns receive fixed group-level mass, so a 10% failure
    rate still supplies 50% of the gradient when ``hard_group_weight=1``.
    """
    if temperature <= 0:
        raise ValueError("ERAF bipartite-assignment temperature must be positive.")
    if hard_group_weight <= 0:
        raise ValueError(
            "ERAF bipartite-assignment hard-group weight must be positive."
        )
    queries = torch.cat(
        (role_attentions["subject"], role_attentions["reference"]), dim=1
    ).float()
    candidates = torch.cat(
        (role_targets["subject"], role_targets["reference"]), dim=1
    ).float()
    query_ids = torch.cat(
        (role_entity_ids["subject"], role_entity_ids["reference"]), dim=1
    ).long()
    candidate_ids = query_ids
    query_valid = torch.cat(
        (role_valid["subject"], role_valid["reference"]), dim=1
    ).bool() & (query_ids >= 0)
    candidate_valid = query_valid
    mass = torch.einsum("brt,bkt->brk", queries, candidates)
    pair_valid = query_valid.unsqueeze(-1) & candidate_valid.unsqueeze(1)
    positive = pair_valid & (query_ids.unsqueeze(-1) == candidate_ids.unsqueeze(1))
    negative = pair_valid & ~positive

    def directional(
        scores: torch.Tensor,
        valid_pairs: torch.Tensor,
        positive_pairs: torch.Tensor,
        negative_pairs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        valid = positive_pairs.any(dim=-1) & negative_pairs.any(dim=-1)
        logits = (scores / float(temperature)).masked_fill(~valid_pairs, -torch.inf)
        positive_logits = logits.masked_fill(~positive_pairs, -torch.inf)
        nll = torch.logsumexp(logits, dim=-1) - torch.logsumexp(positive_logits, dim=-1)
        selected = logits.argmax(dim=-1, keepdim=True)
        correct = positive_pairs.gather(-1, selected).squeeze(-1) & valid
        negative_count = negative_pairs.float().sum(dim=-1)
        return nll, valid, correct, negative_count

    row_loss, row_valid, row_correct, row_negative_count = directional(
        mass, pair_valid, positive, negative
    )
    column_loss, column_valid, column_correct, column_negative_count = directional(
        mass.transpose(1, 2),
        pair_valid.transpose(1, 2),
        positive.transpose(1, 2),
        negative.transpose(1, 2),
    )
    # Balance all row and column decisions together.  Balancing each direction
    # separately would silently reduce the hard mass when one direction has no
    # mistakes, which is exactly the long-tail failure V9.5 is meant to repair.
    (
        assignment_loss,
        hard_gradient_fraction,
        global_easy_count,
        global_hard_count,
    ) = _balanced_group_mean(
        torch.cat((row_loss, column_loss), dim=1),
        torch.cat((row_valid, column_valid), dim=1),
        torch.cat((row_correct, column_correct), dim=1),
        hard_group_weight=hard_group_weight,
    )

    # The gate requires every active role in a conjunction to be right.  Use
    # the worst row and worst column rather than an average over easy clauses.
    multi_sample = (clause_valid.bool().sum(dim=-1) > 1) & row_valid.any(dim=-1)
    worst_row = torch.where(row_valid, row_loss, torch.zeros_like(row_loss)).amax(
        dim=-1
    )
    worst_column = torch.where(
        column_valid, column_loss, torch.zeros_like(column_loss)
    ).amax(dim=-1)
    worst_row = torch.where(
        row_valid.any(dim=-1), worst_row, torch.zeros_like(worst_row)
    )
    worst_column = torch.where(
        column_valid.any(dim=-1), worst_column, torch.zeros_like(worst_column)
    )
    multi_clause_loss = _masked_mean(0.5 * (worst_row + worst_column), multi_sample)
    sample_correct = (row_correct | ~row_valid).all(dim=-1)
    zero = mass.sum() * 0.0
    metrics = {
        "accuracy": _masked_mean(row_correct.float(), row_valid).detach(),
        "column_accuracy": _masked_mean(column_correct.float(), column_valid).detach(),
        "hard_fraction": _masked_mean((~row_correct).float(), row_valid).detach(),
        "margin": zero.detach(),
        "negative_count": _masked_mean(
            0.5 * (row_negative_count + column_negative_count), row_valid
        ).detach(),
        "multi_clause_accuracy": (
            _masked_mean(sample_correct.float(), multi_sample).detach()
            if bool(multi_sample.any())
            else zero.detach()
        ),
        "multi_clause_fraction": multi_sample.float().mean().detach(),
        "hard_gradient_fraction": hard_gradient_fraction.detach(),
        "global_easy_count": global_easy_count.detach(),
        "global_hard_count": global_hard_count.detach(),
        "exclusive_coverage": zero.detach(),
    }
    return assignment_loss, multi_clause_loss, metrics


def _balanced_exclusive_role_assignment_loss(
    *,
    role_attentions: Mapping[str, torch.Tensor],
    role_targets: Mapping[str, torch.Tensor],
    role_entity_ids: Mapping[str, torch.Tensor],
    role_valid: Mapping[str, torch.Tensor],
    clause_valid: torch.Tensor,
    temperature: float,
    hard_group_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Assign subject/reference roles using only unambiguous visual evidence.

    Projected masks can overlap when an object is inside or on a reference
    fixture.  Shared patches cannot identify which semantic role an attention
    query selected, so V9.7 removes them from both the positive and negative
    role evidence.  Full masks remain supervised by the localization losses;
    this objective only changes the subject/reference competition used by the
    role gate.
    """
    if temperature <= 0:
        raise ValueError("ERAF exclusive-assignment temperature must be positive.")
    if hard_group_weight <= 0:
        raise ValueError(
            "ERAF exclusive-assignment hard-group weight must be positive."
        )

    subject_target = role_targets["subject"].float()
    reference_target = role_targets["reference"].float()
    shared = torch.minimum(subject_target, reference_target)
    subject_exclusive = (subject_target - shared).clamp_min(0.0)
    reference_exclusive = (reference_target - shared).clamp_min(0.0)
    distinct_entities = (
        role_entity_ids["subject"].long()
        != role_entity_ids["reference"].long()
    )
    pair_valid = (
        clause_valid.bool()
        & role_valid["subject"].bool()
        & role_valid["reference"].bool()
        & distinct_entities
        & (subject_exclusive.sum(dim=-1) > 1.0e-8)
        & (reference_exclusive.sum(dim=-1) > 1.0e-8)
    )

    subject_attention = role_attentions["subject"].float()
    reference_attention = role_attentions["reference"].float()
    subject_own = (subject_attention * subject_exclusive).sum(dim=-1)
    subject_wrong = (subject_attention * reference_exclusive).sum(dim=-1)
    reference_own = (reference_attention * reference_exclusive).sum(dim=-1)
    reference_wrong = (reference_attention * subject_exclusive).sum(dim=-1)

    def binary_nll(own: torch.Tensor, wrong: torch.Tensor) -> torch.Tensor:
        own_logit = own / float(temperature)
        wrong_logit = wrong / float(temperature)
        return torch.logaddexp(own_logit, wrong_logit) - own_logit

    row_losses = torch.stack(
        (
            binary_nll(subject_own, subject_wrong),
            binary_nll(reference_own, reference_wrong),
        ),
        dim=-1,
    )
    row_correct = torch.stack(
        (
            subject_own > subject_wrong,
            reference_own > reference_wrong,
        ),
        dim=-1,
    )
    row_margins = torch.stack(
        (
            subject_own - subject_wrong,
            reference_own - reference_wrong,
        ),
        dim=-1,
    )
    # Each exclusive entity should also prefer its corresponding semantic
    # query, preventing both queries from collapsing onto the same role.
    column_losses = torch.stack(
        (
            binary_nll(subject_own, reference_wrong),
            binary_nll(reference_own, subject_wrong),
        ),
        dim=-1,
    )
    column_correct = torch.stack(
        (
            subject_own > reference_wrong,
            reference_own > subject_wrong,
        ),
        dim=-1,
    )
    decision_valid = pair_valid.unsqueeze(-1).expand_as(row_correct)
    (
        assignment_loss,
        hard_gradient_fraction,
        global_easy_count,
        global_hard_count,
    ) = _balanced_group_mean(
        torch.cat((row_losses, column_losses), dim=1),
        torch.cat((decision_valid, decision_valid), dim=1),
        torch.cat((row_correct, column_correct), dim=1),
        hard_group_weight=hard_group_weight,
    )

    multi_sample = (clause_valid.bool().sum(dim=-1) > 1) & pair_valid.any(dim=-1)
    worst_row = torch.where(
        decision_valid, row_losses, torch.zeros_like(row_losses)
    ).flatten(1).amax(dim=-1)
    worst_column = torch.where(
        decision_valid, column_losses, torch.zeros_like(column_losses)
    ).flatten(1).amax(dim=-1)
    multi_clause_loss = _masked_mean(
        0.5 * (worst_row + worst_column), multi_sample
    )
    sample_correct = (row_correct | ~decision_valid).flatten(1).all(dim=-1)
    zero = row_losses.sum() * 0.0
    metrics = {
        "accuracy": _masked_mean(
            row_correct.float(), decision_valid
        ).detach(),
        "column_accuracy": _masked_mean(
            column_correct.float(), decision_valid
        ).detach(),
        "hard_fraction": _masked_mean(
            (~row_correct).float(), decision_valid
        ).detach(),
        "margin": _masked_mean(row_margins, decision_valid).detach(),
        "negative_count": _masked_mean(
            torch.ones_like(row_losses), decision_valid
        ).detach(),
        "multi_clause_accuracy": (
            _masked_mean(sample_correct.float(), multi_sample).detach()
            if bool(multi_sample.any())
            else zero.detach()
        ),
        "multi_clause_fraction": multi_sample.float().mean().detach(),
        "hard_gradient_fraction": hard_gradient_fraction.detach(),
        "global_easy_count": global_easy_count.detach(),
        "global_hard_count": global_hard_count.detach(),
        "exclusive_coverage": _masked_mean(
            pair_valid.float(), clause_valid.bool()
        ).detach(),
    }
    return assignment_loss, multi_clause_loss, metrics


def entity_relation_affordance_loss(
    outputs: Mapping[str, torch.Tensor],
    labels: Mapping[str, torch.Tensor],
    *,
    weights: ERAFLossWeights,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute V9 training-only privileged supervision losses and metrics."""
    predicate_ids = labels["predicate_ids"].long()
    clause_valid = labels["clause_valid"].bool()
    if predicate_ids.shape != outputs["active_logits"].shape:
        raise ValueError("ERAF predicate labels must match [B,max_clauses].")
    active_loss = F.binary_cross_entropy_with_logits(
        outputs["active_logits"].float(), clause_valid.float()
    )
    predicate_loss = _masked_mean(
        F.cross_entropy(
            outputs["predicate_logits"].float().transpose(1, 2),
            predicate_ids,
            reduction="none",
        ),
        clause_valid,
    )

    spatial_bce_losses: list[torch.Tensor] = []
    spatial_dice_losses: list[torch.Tensor] = []
    visibility_losses: list[torch.Tensor] = []
    view_visibility_losses: list[torch.Tensor] = []
    view_center_losses: list[torch.Tensor] = []
    attention_mask_losses: list[torch.Tensor] = []
    hit_metrics: list[torch.Tensor] = []
    role_heatmap_predictions: dict[str, torch.Tensor] = {}
    role_attentions: dict[str, torch.Tensor] = {}
    role_targets: dict[str, torch.Tensor] = {}
    role_mask_valid: dict[str, torch.Tensor] = {}
    role_attention_masses: dict[str, torch.Tensor] = {}
    for role in ("subject", "reference"):
        target, teacher_valid = masks_to_patch_targets(
            labels[f"{role}_masks"],
            token_count=int(outputs[f"{role}_similarity"].shape[-1]),
        )
        valid = clause_valid & labels[f"{role}_mask_valid"].bool() & teacher_valid
        logits = outputs[f"{role}_similarity"].float()
        prediction = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(
            logits, target.float(), reduction="none"
        ).mean(dim=-1)
        intersection = (prediction * target.float()).sum(dim=-1)
        dice = 1.0 - (2.0 * intersection + 1.0) / (
            prediction.sum(dim=-1) + target.float().sum(dim=-1) + 1.0
        )
        spatial_bce_losses.append(_masked_mean(bce, valid))
        spatial_dice_losses.append(_masked_mean(dice, valid))
        attention = outputs[f"{role}_attention"].float()
        # Deployment chooses entities from this normalized attention, not the
        # independent sigmoid heatmap above.  Optimizing its probability mass
        # inside the audited mask closes the old train/deploy objective gap.
        support = target > 0
        attention_mass = (attention * support.to(attention.dtype)).sum(dim=-1)
        attention_mask_losses.append(
            _masked_mean(-attention_mass.clamp_min(1.0e-8).log(), valid)
        )
        top1 = attention.argmax(dim=-1)
        hit = torch.gather((target > 0).float(), -1, top1.unsqueeze(-1)).squeeze(-1)
        hit_metrics.append(_masked_mean(hit, valid))
        role_heatmap_predictions[role] = prediction
        role_attentions[role] = attention
        role_targets[role] = target.float()
        role_mask_valid[role] = valid
        role_attention_masses[role] = attention_mass
        visibility_loss = F.binary_cross_entropy_with_logits(
            outputs[f"{role}_visibility_logits"].float(),
            labels[f"{role}_mask_valid"].float(),
            reduction="none",
        )
        visibility_losses.append(_masked_mean(visibility_loss, clause_valid))
        view_visible = labels[f"{role}_view_visible"].bool()
        view_valid = clause_valid.unsqueeze(-1).expand_as(view_visible)
        view_visibility_loss = F.binary_cross_entropy_with_logits(
            outputs[f"{role}_view_visibility_logits"].float(),
            view_visible.float(),
            reduction="none",
        )
        center_loss = F.smooth_l1_loss(
            outputs[f"{role}_view_centers"].float(),
            labels[f"{role}_view_centers"].float(),
            reduction="none",
        ).mean(dim=-1)
        view_visibility_losses.append(_masked_mean(view_visibility_loss, view_valid))
        view_center_losses.append(_masked_mean(center_loss, view_valid & view_visible))
    spatial_bce_loss = torch.stack(spatial_bce_losses).mean()
    spatial_dice_loss = torch.stack(spatial_dice_losses).mean()
    visibility_loss = torch.stack(visibility_losses).mean()
    view_visibility_loss = torch.stack(view_visibility_losses).mean()
    view_center_loss = torch.stack(view_center_losses).mean()
    # Preserve the original composite mask-loss scale: the previous code
    # averaged three terms per role, where two terms were themselves sums.
    mask_loss = (
        spatial_bce_loss
        + spatial_dice_loss
        + visibility_loss
        + view_visibility_loss
        + view_center_loss
    ) / 3.0
    attention_mask_loss = torch.stack(attention_mask_losses).mean()

    position_losses = []
    for role in ("subject", "reference"):
        valid = clause_valid & labels[f"{role}_position_valid"].bool()
        value = F.smooth_l1_loss(
            outputs[f"{role}_position"].float(),
            labels[f"{role}_positions"].float(),
            reduction="none",
        ).mean(dim=-1)
        position_losses.append(_masked_mean(value, valid))
    position_loss = torch.stack(position_losses).mean()

    anchor_losses = []
    for name in ("grasp", "goal", "interaction"):
        valid = clause_valid & labels[f"{name}_anchor_valid"].bool()
        value = F.smooth_l1_loss(
            outputs[f"{name}_anchor"].float(),
            labels[f"{name}_anchors"].float(),
            reduction="none",
        ).mean(dim=-1)
        anchor_losses.append(_masked_mean(value, valid))
    anchor_loss = torch.stack(anchor_losses).mean()

    truth_loss = _masked_mean(
        F.binary_cross_entropy_with_logits(
            outputs["predicate_truth_logits"].float(),
            labels["predicate_truth"].float(),
            reduction="none",
        ),
        clause_valid & labels["predicate_truth_valid"].bool(),
    )
    phase_loss = _masked_mean(
        F.cross_entropy(
            outputs["phase_logits"].float().transpose(1, 2),
            labels["phase_ids"].long(),
            reduction="none",
        ),
        clause_valid & labels["phase_valid"].bool(),
    )
    entity_losses = []
    entity_accuracies = []
    for role in ("subject", "reference"):
        role_valid = role_mask_valid[role]
        role_loss, role_accuracy = _supervised_role_contrastive_loss(
            outputs[f"{role}_queries"],
            outputs[f"{role}_token"],
            labels[f"{role}_entity_ids"],
            role_valid,
        )
        entity_losses.append(role_loss)
        entity_accuracies.append(role_accuracy)
    role_entity_loss = torch.stack(entity_losses).mean()

    # Same-state role-swap negative.  V9.1 uses the exact normalized attention
    # and occupancy-weighted mass consumed by the offline gate.  Historical v1
    # remains available below only for explicit reproduction.
    role_swap_valid = (
        role_mask_valid["subject"]
        & role_mask_valid["reference"]
        & (labels["subject_entity_ids"].long() != labels["reference_entity_ids"].long())
    )
    subject_attention_own = (role_attentions["subject"] * role_targets["subject"]).sum(
        dim=-1
    )
    subject_attention_wrong = (
        role_attentions["subject"] * role_targets["reference"]
    ).sum(dim=-1)
    reference_attention_own = (
        role_attentions["reference"] * role_targets["reference"]
    ).sum(dim=-1)
    reference_attention_wrong = (
        role_attentions["reference"] * role_targets["subject"]
    ).sum(dim=-1)
    shared_role_target = torch.minimum(
        role_targets["subject"], role_targets["reference"]
    )
    subject_exclusive_target = (
        role_targets["subject"] - shared_role_target
    ).clamp_min(0.0)
    reference_exclusive_target = (
        role_targets["reference"] - shared_role_target
    ).clamp_min(0.0)
    exclusive_role_valid = (
        role_swap_valid
        & (subject_exclusive_target.sum(dim=-1) > 1.0e-8)
        & (reference_exclusive_target.sum(dim=-1) > 1.0e-8)
    )
    subject_exclusive_own = (
        role_attentions["subject"] * subject_exclusive_target
    ).sum(dim=-1)
    subject_exclusive_wrong = (
        role_attentions["subject"] * reference_exclusive_target
    ).sum(dim=-1)
    reference_exclusive_own = (
        role_attentions["reference"] * reference_exclusive_target
    ).sum(dim=-1)
    reference_exclusive_wrong = (
        role_attentions["reference"] * subject_exclusive_target
    ).sum(dim=-1)
    if int(weights.objective_version) >= 8:
        # V9.7 aligns role competition with the exclusive-evidence gate.
        # Full-mask localization remains optimized above and is still audited.
        subject_own = subject_exclusive_own
        subject_wrong = subject_exclusive_wrong
        reference_own = reference_exclusive_own
        reference_wrong = reference_exclusive_wrong
        role_supervision_valid = exclusive_role_valid
    elif int(weights.objective_version) >= 2:
        subject_own = subject_attention_own
        subject_wrong = subject_attention_wrong
        reference_own = reference_attention_own
        reference_wrong = reference_attention_wrong
        role_supervision_valid = role_swap_valid
    else:
        # Preserve the historical V9 objective for explicit reproduction.
        # Formal V9.1 runs set objective_version=2 and never take this branch.
        subject_own = (
            role_heatmap_predictions["subject"] * role_targets["subject"]
        ).sum(dim=-1) / role_targets["subject"].sum(dim=-1).clamp_min(1.0)
        subject_wrong = (
            role_heatmap_predictions["subject"] * role_targets["reference"]
        ).sum(dim=-1) / role_targets["reference"].sum(dim=-1).clamp_min(1.0)
        reference_own = (
            role_heatmap_predictions["reference"] * role_targets["reference"]
        ).sum(dim=-1) / role_targets["reference"].sum(dim=-1).clamp_min(1.0)
        reference_wrong = (
            role_heatmap_predictions["reference"] * role_targets["subject"]
        ).sum(dim=-1) / role_targets["subject"].sum(dim=-1).clamp_min(1.0)
        role_supervision_valid = role_swap_valid
    role_swap_margin = float(weights.role_swap_margin)
    role_swap_loss = _masked_mean(
        0.5
        * (
            torch.relu(role_swap_margin + subject_wrong - subject_own)
            + torch.relu(role_swap_margin + reference_wrong - reference_own)
        ),
        role_supervision_valid,
    )
    # V9.2 adds an explicit two-query/two-entity assignment objective.  The
    # row terms require each semantic query to prefer its own entity; the
    # column terms prevent both queries from collapsing onto the same entity.
    # Incorrect clauses are upweighted online, which preserves the audited
    # native/CF data mixture while concentrating gradient on hard role swaps.
    assignment_temperature = float(weights.role_assignment_temperature)
    if assignment_temperature <= 0:
        raise ValueError("ERAF role-assignment temperature must be positive.")
    assignment_target = torch.zeros_like(subject_own, dtype=torch.long)

    def assignment_ce(own: torch.Tensor, wrong: torch.Tensor) -> torch.Tensor:
        logits = torch.stack((own, wrong), dim=-1) / assignment_temperature
        return F.cross_entropy(
            logits.reshape(-1, 2),
            assignment_target.reshape(-1),
            reduction="none",
        ).reshape_as(own)

    assignment_per_clause = 0.25 * (
        assignment_ce(subject_own, subject_wrong)
        + assignment_ce(reference_own, reference_wrong)
        + assignment_ce(subject_own, reference_wrong)
        + assignment_ce(reference_own, subject_wrong)
    )
    row_correct = (subject_own > subject_wrong) & (reference_own > reference_wrong)
    column_correct = (subject_own > reference_wrong) & (reference_own > subject_wrong)
    assignment_correct = row_correct & column_correct
    hard_assignment = (~assignment_correct).float().detach()
    hard_weight = float(weights.role_assignment_hard_weight)
    if hard_weight < 0:
        raise ValueError("ERAF role-assignment hard weight must be non-negative.")
    assignment_sample_weights = 1.0 + hard_weight * hard_assignment
    role_assignment_loss = _masked_weighted_mean(
        assignment_per_clause,
        role_supervision_valid,
        assignment_sample_weights,
    )
    zero = outputs["active_logits"].sum() * 0.0
    structured_assignment_loss = zero
    multi_clause_consistency_loss = zero
    structured_metrics = {
        "accuracy": zero.detach(),
        "column_accuracy": zero.detach(),
        "hard_fraction": zero.detach(),
        "margin": zero.detach(),
        "negative_count": zero.detach(),
        "multi_clause_accuracy": zero.detach(),
        "multi_clause_fraction": zero.detach(),
        "hard_gradient_fraction": zero.detach(),
        "global_easy_count": zero.detach(),
        "global_hard_count": zero.detach(),
        "exclusive_coverage": zero.detach(),
    }
    if int(weights.objective_version) >= 8:
        (
            structured_assignment_loss,
            multi_clause_consistency_loss,
            structured_metrics,
        ) = _balanced_exclusive_role_assignment_loss(
            role_attentions=role_attentions,
            role_targets=role_targets,
            role_entity_ids={
                role: labels[f"{role}_entity_ids"]
                for role in ("subject", "reference")
            },
            role_valid=role_mask_valid,
            clause_valid=clause_valid,
            temperature=float(weights.structured_assignment_temperature),
            hard_group_weight=float(weights.structured_assignment_hard_weight),
        )
    elif int(weights.objective_version) >= 6:
        (
            structured_assignment_loss,
            multi_clause_consistency_loss,
            structured_metrics,
        ) = _balanced_bipartite_assignment_loss(
            role_attentions=role_attentions,
            role_targets=role_targets,
            role_entity_ids={
                role: labels[f"{role}_entity_ids"] for role in ("subject", "reference")
            },
            role_valid=role_mask_valid,
            clause_valid=clause_valid,
            temperature=float(weights.structured_assignment_temperature),
            hard_group_weight=float(weights.structured_assignment_hard_weight),
        )
    elif int(weights.objective_version) >= 5:
        (
            structured_assignment_loss,
            multi_clause_consistency_loss,
            structured_metrics,
        ) = _structured_all_entity_assignment_loss(
            role_attentions=role_attentions,
            role_targets=role_targets,
            role_entity_ids={
                role: labels[f"{role}_entity_ids"] for role in ("subject", "reference")
            },
            role_valid=role_mask_valid,
            clause_valid=clause_valid,
            temperature=float(weights.structured_assignment_temperature),
            hard_weight=float(weights.structured_assignment_hard_weight),
        )
    clause_activation_balance_loss = zero
    clause_cardinality_loss = zero
    clause_worst_slot_loss = zero
    clause_adapter_energy_loss = zero
    clause_metrics = {
        "exact": zero.detach(),
        "multi_exact": zero.detach(),
        "cardinality_accuracy": zero.detach(),
        "multi_fraction": zero.detach(),
        "multi_gradient_fraction": zero.detach(),
        "global_single_count": zero.detach(),
        "global_multi_count": zero.detach(),
        "base_exact": zero.detach(),
        "base_multi_exact": zero.detach(),
        "exact_gain": zero.detach(),
        "multi_exact_gain": zero.detach(),
    }
    if int(weights.objective_version) >= 9:
        required_clause_outputs = {
            "base_active_logits",
            "clause_active_residual",
            "clause_cardinality_logits",
        }
        missing_clause_outputs = sorted(required_clause_outputs.difference(outputs))
        if missing_clause_outputs:
            raise ValueError(
                "V9.8 clause calibration requires its zero-init adapter outputs; "
                f"missing={missing_clause_outputs}."
            )
        (
            clause_activation_balance_loss,
            clause_cardinality_loss,
            clause_worst_slot_loss,
            clause_metrics,
        ) = _clause_activation_calibration_loss(
            active_logits=outputs["active_logits"],
            cardinality_logits=outputs["clause_cardinality_logits"],
            clause_valid=clause_valid,
            multi_group_weight=float(weights.clause_multi_group_weight),
        )
        clause_adapter_energy_loss = outputs["clause_active_residual"].float().pow(
            2
        ).mean()
        base_exact = (
            (outputs["base_active_logits"] > 0) == clause_valid
        ).all(dim=-1)
        base_multi = clause_valid.sum(dim=-1) > 1
        sample_valid = clause_valid.any(dim=-1)
        clause_metrics["base_exact"] = _masked_mean(
            base_exact.float(), sample_valid
        ).detach()
        clause_metrics["base_multi_exact"] = _masked_mean(
            base_exact.float(), base_multi
        ).detach()
        clause_metrics["exact_gain"] = (
            clause_metrics["exact"] - clause_metrics["base_exact"]
        ).detach()
        clause_metrics["multi_exact_gain"] = (
            clause_metrics["multi_exact"] - clause_metrics["base_multi_exact"]
        ).detach()
    # Penalize attention assigned exclusively to the other semantic role.
    # Pixels shared by overlapping objects/containers are excluded so an
    # object entering a basket is not trained as a false negative.
    subject_exclusive_negative = role_targets["reference"] * (
        role_targets["subject"] <= 0
    ).to(role_targets["reference"].dtype)
    reference_exclusive_negative = role_targets["subject"] * (
        role_targets["reference"] <= 0
    ).to(role_targets["subject"].dtype)
    subject_overlap = (
        role_attentions["subject"] * subject_exclusive_negative
    ).sum(dim=-1)
    reference_overlap = (
        role_attentions["reference"] * reference_exclusive_negative
    ).sum(dim=-1)
    role_overlap_loss = _masked_mean(
        0.5 * (subject_overlap + reference_overlap), role_swap_valid
    )
    role_attention_preservation_loss = zero
    role_position_preservation_loss = zero
    role_anchor_preservation_loss = zero
    role_relation_preservation_loss = zero
    role_adapter_energy_loss = zero
    teacher_preservation_fraction = zero.detach()
    teacher_predicate_max_abs_error = zero.detach()
    teacher_active_max_abs_error = zero.detach()
    if int(weights.objective_version) >= 4:
        required_teacher = {
            "teacher_subject_attention",
            "teacher_reference_attention",
            "teacher_subject_position",
            "teacher_reference_position",
            "teacher_relation_hidden",
            "teacher_grasp_anchor",
            "teacher_goal_anchor",
            "teacher_interaction_anchor",
            "teacher_active_logits",
            "teacher_predicate_logits",
            "subject_role_delta",
            "reference_role_delta",
        }
        missing_teacher = sorted(required_teacher.difference(outputs))
        if missing_teacher:
            raise ValueError(
                "V9.3+ role-adapter training requires its frozen teacher "
                f"bypass outputs; missing={missing_teacher}."
            )
        teacher_subject_attention = outputs["teacher_subject_attention"].float()
        teacher_reference_attention = outputs["teacher_reference_attention"].float()
        teacher_subject_own = (teacher_subject_attention * role_targets["subject"]).sum(
            dim=-1
        )
        teacher_subject_wrong = (
            teacher_subject_attention * role_targets["reference"]
        ).sum(dim=-1)
        teacher_reference_own = (
            teacher_reference_attention * role_targets["reference"]
        ).sum(dim=-1)
        teacher_reference_wrong = (
            teacher_reference_attention * role_targets["subject"]
        ).sum(dim=-1)
        # Preserve only clauses V9.1 already bound correctly.  Incorrect
        # clauses remain free to move under the supervised assignment loss.
        role_pair_valid = role_mask_valid["subject"] & role_mask_valid["reference"]
        unary_role = (
            labels["subject_entity_ids"].long() == labels["reference_entity_ids"].long()
        )
        teacher_preservation_valid = role_pair_valid & (
            unary_role
            | (
                (teacher_subject_own > teacher_subject_wrong)
                & (teacher_reference_own > teacher_reference_wrong)
            )
        )
        teacher_preservation_fraction = _masked_mean(
            teacher_preservation_valid.float(), role_pair_valid
        ).detach()
        # V9.6 separates identity correction from geometry preservation. Hard
        # role queries may change their attention, but position/relation and
        # all affordance anchors retain the clean V9.3 teacher on every active
        # clause. This prevents a binding repair from moving a correct goal
        # anchor beyond the 5 cm gate.
        geometry_preservation_valid = (
            clause_valid
            if int(weights.objective_version) >= 7
            else teacher_preservation_valid
        )

        attention_preservation_terms = []
        for role, teacher_attention in (
            ("subject", teacher_subject_attention),
            ("reference", teacher_reference_attention),
        ):
            student_attention = role_attentions[role].float().clamp_min(1.0e-8)
            teacher_probability = teacher_attention.clamp_min(1.0e-8)
            teacher_probability = teacher_probability / teacher_probability.sum(
                dim=-1, keepdim=True
            ).clamp_min(1.0e-8)
            kl = (
                teacher_probability
                * (teacher_probability.log() - student_attention.log())
            ).sum(dim=-1)
            attention_preservation_terms.append(
                _masked_mean(kl, teacher_preservation_valid)
            )
        role_attention_preservation_loss = torch.stack(
            attention_preservation_terms
        ).mean()

        position_preservation_terms = []
        for role in ("subject", "reference"):
            value = F.smooth_l1_loss(
                outputs[f"{role}_position"].float(),
                outputs[f"teacher_{role}_position"].float(),
                reduction="none",
            ).mean(dim=-1)
            position_preservation_terms.append(
                _masked_mean(value, geometry_preservation_valid)
            )
        role_position_preservation_loss = torch.stack(
            position_preservation_terms
        ).mean()

        anchor_preservation_terms = []
        for name in ("grasp", "goal", "interaction"):
            value = F.smooth_l1_loss(
                outputs[f"{name}_anchor"].float(),
                outputs[f"teacher_{name}_anchor"].float(),
                reduction="none",
            ).mean(dim=-1)
            anchor_preservation_terms.append(
                _masked_mean(value, geometry_preservation_valid)
            )
        role_anchor_preservation_loss = torch.stack(anchor_preservation_terms).mean()
        role_relation_preservation_loss = _masked_mean(
            F.mse_loss(
                outputs["relation_hidden"].float(),
                outputs["teacher_relation_hidden"].float(),
                reduction="none",
            ).mean(dim=-1),
            geometry_preservation_valid,
        )
        if int(weights.objective_version) >= 6:
            required_balanced = {
                "balanced_subject_role_delta",
                "balanced_reference_role_delta",
            }
            missing_balanced = sorted(required_balanced.difference(outputs))
            if missing_balanced:
                raise ValueError(
                    "V9.5 balanced-role training requires its zero-init "
                    f"adapter outputs; missing={missing_balanced}."
                )
            energy_subject_delta = outputs["balanced_subject_role_delta"]
            energy_reference_delta = outputs["balanced_reference_role_delta"]
        elif int(weights.objective_version) >= 5:
            required_structured = {
                "structured_subject_role_delta",
                "structured_reference_role_delta",
            }
            missing_structured = sorted(required_structured.difference(outputs))
            if missing_structured:
                raise ValueError(
                    "V9.4 structured-role training requires its zero-init "
                    f"adapter outputs; missing={missing_structured}."
                )
            energy_subject_delta = outputs["structured_subject_role_delta"]
            energy_reference_delta = outputs["structured_reference_role_delta"]
        else:
            energy_subject_delta = outputs["subject_role_delta"]
            energy_reference_delta = outputs["reference_role_delta"]
        role_adapter_energy_loss = _masked_mean(
            0.5
            * (
                energy_subject_delta.float().pow(2).mean(dim=-1)
                + energy_reference_delta.float().pow(2).mean(dim=-1)
            ),
            clause_valid,
        )
        teacher_predicate_max_abs_error = (
            (
                outputs["predicate_logits"].float()
                - outputs["teacher_predicate_logits"].float()
            )
            .abs()
            .max()
            .detach()
        )
        teacher_active_max_abs_error = (
            (
                outputs["active_logits"].float()
                - outputs["teacher_active_logits"].float()
            )
            .abs()
            .max()
            .detach()
        )
    entity_loss = active_loss + role_entity_loss
    relation_loss = predicate_loss + truth_loss
    total = (
        weights.mask * mask_loss
        + weights.attention_mask * attention_mask_loss
        + weights.entity * entity_loss
        + weights.relation * relation_loss
        + weights.anchor * anchor_loss
        + weights.position * position_loss
        + weights.role_swap * role_swap_loss
        + weights.role_overlap * role_overlap_loss
        + weights.role_assignment * role_assignment_loss
        + weights.role_attention_preservation * role_attention_preservation_loss
        + weights.role_position_preservation * role_position_preservation_loss
        + weights.role_anchor_preservation * role_anchor_preservation_loss
        + weights.role_relation_preservation * role_relation_preservation_loss
        + weights.role_adapter_energy * role_adapter_energy_loss
        + weights.structured_assignment * structured_assignment_loss
        + weights.multi_clause_consistency * multi_clause_consistency_loss
        + weights.clause_activation_balance * clause_activation_balance_loss
        + weights.clause_cardinality * clause_cardinality_loss
        + weights.clause_worst_slot * clause_worst_slot_loss
        + weights.clause_adapter_energy * clause_adapter_energy_loss
        + weights.phase * phase_loss
    )
    predicate_prediction = outputs["predicate_logits"].argmax(dim=-1)
    clause_prediction = outputs["active_logits"] > 0
    exact = (
        (clause_prediction == clause_valid)
        & ((predicate_prediction == predicate_ids) | ~clause_valid)
    ).all(dim=-1)
    metrics = {
        "loss_pgc_v9_mask": mask_loss.detach(),
        "loss_pgc_v9_spatial_bce": spatial_bce_loss.detach(),
        "loss_pgc_v9_spatial_dice": spatial_dice_loss.detach(),
        "loss_pgc_v9_visibility": visibility_loss.detach(),
        "loss_pgc_v9_view_visibility": view_visibility_loss.detach(),
        "loss_pgc_v9_view_center": view_center_loss.detach(),
        "loss_pgc_v9_attention_mask": attention_mask_loss.detach(),
        "loss_pgc_v9_entity": entity_loss.detach(),
        "loss_pgc_v9_relation": relation_loss.detach(),
        "loss_pgc_v9_anchor": anchor_loss.detach(),
        "loss_pgc_v9_position": position_loss.detach(),
        "loss_pgc_v9_role_swap": role_swap_loss.detach(),
        "loss_pgc_v9_role_overlap": role_overlap_loss.detach(),
        "loss_pgc_v9_role_assignment": role_assignment_loss.detach(),
        "loss_pgc_v9_role_attention_preservation": (
            role_attention_preservation_loss.detach()
        ),
        "loss_pgc_v9_role_position_preservation": (
            role_position_preservation_loss.detach()
        ),
        "loss_pgc_v9_role_anchor_preservation": (
            role_anchor_preservation_loss.detach()
        ),
        "loss_pgc_v9_role_relation_preservation": (
            role_relation_preservation_loss.detach()
        ),
        "loss_pgc_v9_role_adapter_energy": role_adapter_energy_loss.detach(),
        "loss_pgc_v9_structured_assignment": (structured_assignment_loss.detach()),
        "loss_pgc_v9_multi_clause_consistency": (
            multi_clause_consistency_loss.detach()
        ),
        "loss_pgc_v9_clause_activation_balance": (
            clause_activation_balance_loss.detach()
        ),
        "loss_pgc_v9_clause_cardinality": clause_cardinality_loss.detach(),
        "loss_pgc_v9_clause_worst_slot": clause_worst_slot_loss.detach(),
        "loss_pgc_v9_clause_adapter_energy": clause_adapter_energy_loss.detach(),
        "loss_pgc_v9_phase": phase_loss.detach(),
        "loss_pgc_v9_predicate": predicate_loss.detach(),
        "loss_pgc_v9_role_entity_contrastive": role_entity_loss.detach(),
        "pgc_v9_subject_top1_hit": hit_metrics[0].detach(),
        "pgc_v9_reference_top1_hit": hit_metrics[1].detach(),
        "pgc_v9_subject_gt_attention_mass": _masked_mean(
            role_attention_masses["subject"], role_mask_valid["subject"]
        ).detach(),
        "pgc_v9_reference_gt_attention_mass": _masked_mean(
            role_attention_masses["reference"], role_mask_valid["reference"]
        ).detach(),
        "pgc_v9_subject_wrong_role_attention_mass": _masked_mean(
            subject_attention_wrong, role_swap_valid
        ).detach(),
        "pgc_v9_reference_wrong_role_attention_mass": _masked_mean(
            reference_attention_wrong, role_swap_valid
        ).detach(),
        "pgc_v9_subject_role_attention_margin": _masked_mean(
            subject_attention_own - subject_attention_wrong, role_swap_valid
        ).detach(),
        "pgc_v9_reference_role_attention_margin": _masked_mean(
            reference_attention_own - reference_attention_wrong,
            role_swap_valid,
        ).detach(),
        "pgc_v9_role_swap_accuracy": _masked_mean(
            (
                (subject_attention_own > subject_attention_wrong)
                & (reference_attention_own > reference_attention_wrong)
            ).float(),
            role_swap_valid,
        ).detach(),
        "pgc_v9_role_swap_valid_fraction": _masked_mean(
            role_swap_valid.float(), clause_valid
        ).detach(),
        "pgc_v9_exclusive_role_swap_accuracy": _masked_mean(
            (
                (subject_exclusive_own > subject_exclusive_wrong)
                & (reference_exclusive_own > reference_exclusive_wrong)
            ).float(),
            exclusive_role_valid,
        ).detach(),
        "pgc_v9_exclusive_role_coverage": _masked_mean(
            exclusive_role_valid.float(), role_swap_valid
        ).detach(),
        "pgc_v9_exclusive_subject_role_attention_margin": _masked_mean(
            subject_exclusive_own - subject_exclusive_wrong,
            exclusive_role_valid,
        ).detach(),
        "pgc_v9_exclusive_reference_role_attention_margin": _masked_mean(
            reference_exclusive_own - reference_exclusive_wrong,
            exclusive_role_valid,
        ).detach(),
        "pgc_v9_role_assignment_accuracy": _masked_mean(
            assignment_correct.float(), role_supervision_valid
        ).detach(),
        "pgc_v9_role_assignment_row_accuracy": _masked_mean(
            row_correct.float(), role_supervision_valid
        ).detach(),
        "pgc_v9_role_assignment_column_accuracy": _masked_mean(
            column_correct.float(), role_supervision_valid
        ).detach(),
        "pgc_v9_role_assignment_hard_fraction": _masked_mean(
            hard_assignment, role_supervision_valid
        ).detach(),
        "pgc_v9_structured_assignment_accuracy": structured_metrics["accuracy"],
        "pgc_v9_structured_assignment_column_accuracy": structured_metrics[
            "column_accuracy"
        ],
        "pgc_v9_structured_assignment_hard_fraction": structured_metrics[
            "hard_fraction"
        ],
        "pgc_v9_structured_assignment_margin": structured_metrics["margin"],
        "pgc_v9_structured_assignment_negative_count": structured_metrics[
            "negative_count"
        ],
        "pgc_v9_structured_multi_clause_accuracy": structured_metrics[
            "multi_clause_accuracy"
        ],
        "pgc_v9_structured_multi_clause_fraction": structured_metrics[
            "multi_clause_fraction"
        ],
        "pgc_v9_structured_hard_gradient_fraction": structured_metrics[
            "hard_gradient_fraction"
        ],
        "pgc_v9_structured_global_easy_count": structured_metrics[
            "global_easy_count"
        ],
        "pgc_v9_structured_global_hard_count": structured_metrics[
            "global_hard_count"
        ],
        "pgc_v9_structured_exclusive_coverage": structured_metrics[
            "exclusive_coverage"
        ],
        "pgc_v9_clause_activation_exact": clause_metrics["exact"],
        "pgc_v9_clause_activation_multi_exact": clause_metrics["multi_exact"],
        "pgc_v9_clause_cardinality_accuracy": clause_metrics[
            "cardinality_accuracy"
        ],
        "pgc_v9_clause_multi_fraction": clause_metrics["multi_fraction"],
        "pgc_v9_clause_multi_gradient_fraction": clause_metrics[
            "multi_gradient_fraction"
        ],
        "pgc_v9_clause_global_single_count": clause_metrics[
            "global_single_count"
        ],
        "pgc_v9_clause_global_multi_count": clause_metrics[
            "global_multi_count"
        ],
        "pgc_v9_base_clause_activation_exact": clause_metrics["base_exact"],
        "pgc_v9_base_clause_activation_multi_exact": clause_metrics[
            "base_multi_exact"
        ],
        "pgc_v9_clause_activation_exact_gain": clause_metrics["exact_gain"],
        "pgc_v9_clause_activation_multi_exact_gain": clause_metrics[
            "multi_exact_gain"
        ],
        "pgc_v9_teacher_preservation_fraction": teacher_preservation_fraction,
        "pgc_v9_teacher_predicate_max_abs_error": (teacher_predicate_max_abs_error),
        "pgc_v9_teacher_active_max_abs_error": teacher_active_max_abs_error,
        "pgc_v9_subject_entity_retrieval_acc": entity_accuracies[0].detach(),
        "pgc_v9_reference_entity_retrieval_acc": entity_accuracies[1].detach(),
        "pgc_v9_relation_accuracy": _masked_mean(
            (predicate_prediction == predicate_ids).float(), clause_valid
        ).detach(),
        "pgc_v9_clause_exact_match": exact.float().mean().detach(),
        "pgc_v9_clause_valid_fraction": clause_valid.float().mean().detach(),
    }
    return total, metrics
