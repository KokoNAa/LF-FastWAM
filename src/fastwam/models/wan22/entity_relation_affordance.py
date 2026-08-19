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
ERAF_PREDICATE_TO_ID = {
    name: index for index, name in enumerate(ERAF_PREDICATES)
}


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
    target = F.interpolate(
        masks.float().reshape(batch_size * clause_count, 1, height, width),
        size=(grid_height, grid_width),
        mode="area",
    ).reshape(batch_size, clause_count, int(token_count)).clamp_(0.0, 1.0)
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
            "reference_queries": self.reference_norm(
                clauses + self.reference_role
            ),
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
            -1.0, 1.0, grid_height,
            device=visual_hidden.device, dtype=visual_hidden.dtype,
        )
        x = torch.linspace(
            -1.0, 1.0, grid_width,
            device=visual_hidden.device, dtype=visual_hidden.dtype,
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
        tokens = torch.einsum(
            "bcn,bnd->bcd", attention.to(visual.dtype), visual
        )
        camera_membership = F.one_hot(
            camera_ids, num_classes=self.camera_count
        ).transpose(0, 1).to(attention.dtype)
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
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 3), nn.Tanh()
        )
        self.grasp_anchor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 3), nn.Tanh()
        )
        self.goal_anchor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 3), nn.Tanh()
        )
        self.action_projection = nn.Linear(hidden_dim, action_dim)
        self.embedding_projection = nn.Linear(hidden_dim, projection_dim)

    def decode_relation_hidden(
        self, relation: torch.Tensor
    ) -> dict[str, torch.Tensor]:
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
            entity_hidden = entity_hidden * torch.sigmoid(
                active_logits.float()
            ).to(entity_hidden.dtype).unsqueeze(-1)
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
            is_binary = (
                (predicted_ids >= ERAF_PREDICATE_TO_ID["in"])
                & (predicted_ids <= ERAF_PREDICATE_TO_ID["back"])
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
            probabilities = torch.softmax(
                outputs["predicate_logits"].float(), dim=-1
            )
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
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        roles = self.role_decoder(language_hidden, language_mask)
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
        }
        outputs = {
            **roles,
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
            "subject_view_visibility_logits": subject[
                "view_visibility_logits"
            ],
            "reference_view_visibility_logits": reference[
                "view_visibility_logits"
            ],
            "subject_view_centers": subject["view_centers"],
            "reference_view_centers": reference["view_centers"],
            "subject_view_attention_mass": subject["view_attention_mass"],
            "reference_view_attention_mass": reference["view_attention_mass"],
            "spatial_coordinates": subject["coordinates"],
            "camera_ids": subject["camera_ids"],
            **affordance,
        }
        return routed_queries, routed_embedding, outputs, metrics


@dataclass(frozen=True)
class ERAFLossWeights:
    mask: float = 1.0
    entity: float = 1.0
    relation: float = 1.0
    anchor: float = 1.0
    position: float = 0.5
    role_swap: float = 0.5
    phase: float = 0.25


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=values.device, dtype=torch.bool)
    if not bool(mask.any()):
        return values.sum() * 0.0
    return values[mask].mean()


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

    mask_losses: list[torch.Tensor] = []
    hit_metrics: list[torch.Tensor] = []
    role_predictions: dict[str, torch.Tensor] = {}
    role_targets: dict[str, torch.Tensor] = {}
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
        dice = 1.0 - (
            2.0 * intersection + 1.0
        ) / (
            prediction.sum(dim=-1) + target.float().sum(dim=-1) + 1.0
        )
        mask_losses.append(_masked_mean(bce + dice, valid))
        top1 = outputs[f"{role}_attention"].float().argmax(dim=-1)
        hit = torch.gather((target > 0).float(), -1, top1.unsqueeze(-1)).squeeze(-1)
        hit_metrics.append(_masked_mean(hit, valid))
        role_predictions[role] = prediction
        role_targets[role] = target.float()
        visibility_loss = F.binary_cross_entropy_with_logits(
            outputs[f"{role}_visibility_logits"].float(),
            labels[f"{role}_mask_valid"].float(),
            reduction="none",
        )
        mask_losses.append(_masked_mean(visibility_loss, clause_valid))
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
        mask_losses.append(
            _masked_mean(view_visibility_loss, view_valid)
            + _masked_mean(center_loss, view_valid & view_visible)
        )
    mask_loss = torch.stack(mask_losses).mean()

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
        role_valid = clause_valid & labels[f"{role}_mask_valid"].bool()
        role_loss, role_accuracy = _supervised_role_contrastive_loss(
            outputs[f"{role}_queries"],
            outputs[f"{role}_token"],
            labels[f"{role}_entity_ids"],
            role_valid,
        )
        entity_losses.append(role_loss)
        entity_accuracies.append(role_accuracy)
    role_entity_loss = torch.stack(entity_losses).mean()

    # Same-state role-swap negative: each role's heatmap must place more mass
    # on its own GT mask than on the other role's GT mask.
    role_swap_valid = (
        clause_valid
        & labels["subject_mask_valid"].bool()
        & labels["reference_mask_valid"].bool()
        & (
            labels["subject_entity_ids"].long()
            != labels["reference_entity_ids"].long()
        )
    )
    subject_own = (
        role_predictions["subject"] * role_targets["subject"]
    ).sum(dim=-1) / role_targets["subject"].sum(dim=-1).clamp_min(1.0)
    subject_wrong = (
        role_predictions["subject"] * role_targets["reference"]
    ).sum(dim=-1) / role_targets["reference"].sum(dim=-1).clamp_min(1.0)
    reference_own = (
        role_predictions["reference"] * role_targets["reference"]
    ).sum(dim=-1) / role_targets["reference"].sum(dim=-1).clamp_min(1.0)
    reference_wrong = (
        role_predictions["reference"] * role_targets["subject"]
    ).sum(dim=-1) / role_targets["subject"].sum(dim=-1).clamp_min(1.0)
    role_swap_loss = _masked_mean(
        0.5
        * (
            torch.relu(0.20 + subject_wrong - subject_own)
            + torch.relu(0.20 + reference_wrong - reference_own)
        ),
        role_swap_valid,
    )
    entity_loss = active_loss + role_entity_loss
    relation_loss = predicate_loss + truth_loss
    total = (
        weights.mask * mask_loss
        + weights.entity * entity_loss
        + weights.relation * relation_loss
        + weights.anchor * anchor_loss
        + weights.position * position_loss
        + weights.role_swap * role_swap_loss
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
        "loss_pgc_v9_entity": entity_loss.detach(),
        "loss_pgc_v9_relation": relation_loss.detach(),
        "loss_pgc_v9_anchor": anchor_loss.detach(),
        "loss_pgc_v9_position": position_loss.detach(),
        "loss_pgc_v9_role_swap": role_swap_loss.detach(),
        "loss_pgc_v9_phase": phase_loss.detach(),
        "loss_pgc_v9_predicate": predicate_loss.detach(),
        "loss_pgc_v9_role_entity_contrastive": role_entity_loss.detach(),
        "pgc_v9_subject_top1_hit": hit_metrics[0].detach(),
        "pgc_v9_reference_top1_hit": hit_metrics[1].detach(),
        "pgc_v9_subject_entity_retrieval_acc": entity_accuracies[0].detach(),
        "pgc_v9_reference_entity_retrieval_acc": entity_accuracies[1].detach(),
        "pgc_v9_relation_accuracy": _masked_mean(
            (predicate_prediction == predicate_ids).float(), clause_valid
        ).detach(),
        "pgc_v9_clause_exact_match": exact.float().mean().detach(),
        "pgc_v9_clause_valid_fraction": clause_valid.float().mean().detach(),
    }
    return total, metrics
