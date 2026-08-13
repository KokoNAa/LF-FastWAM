import os
import torch
import torch.nn as nn
from typing import Any, Dict, Optional

from fastwam.utils.logging_config import get_logger

from .helpers.gradient import gradient_checkpoint_forward
from .wan_video_dit import (
    DiTBlock,
    sinusoidal_embedding_1d,
    precompute_freqs_cis,
)

logger = get_logger(__name__)


class ActionHead(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int, eps: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.proj = nn.Linear(hidden_dim, out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, hidden_dim) / hidden_dim**0.5)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        shift, scale = (self.modulation.to(dtype=t.dtype, device=t.device) + t.unsqueeze(1)).chunk(2, dim=1)
        shift = shift.squeeze(1)
        scale = scale.squeeze(1)
        return self.proj(self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1))


class ActionDiT(nn.Module):
    ACTION_BACKBONE_SKIP_PREFIXES = (
        "action_encoder.",
        "head.",
        "latent_action_queries",
    )
    ACTION_BACKBONE_META_KEYS = (
        "hidden_dim",
        "ffn_dim",
        "num_layers",
        "num_heads",
        "attn_head_dim",
        "text_dim",
        "freq_dim",
        "eps",
    )

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        ffn_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        num_heads: int,
        attn_head_dim: int,
        num_layers: int,
        use_gradient_checkpointing: bool = False,
        use_latent_action_queries: bool = False,
        num_latent_queries: int = 32,
        query_rope_offset: int = 512,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.ffn_dim = ffn_dim
        self.text_dim = text_dim
        self.freq_dim = freq_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.use_latent_action_queries = bool(use_latent_action_queries)
        self.num_latent_queries = int(num_latent_queries)
        self.query_rope_offset = int(query_rope_offset)

        if num_heads <= 0:
            raise ValueError(f"`num_heads` must be > 0, got {num_heads}")
        if attn_head_dim <= 0:
            raise ValueError(f"`attn_head_dim` must be > 0, got {attn_head_dim}")
        if attn_head_dim % 2 != 0:
            raise ValueError(f"`attn_head_dim` must be even for RoPE, got {attn_head_dim}")
        if self.num_latent_queries <= 0:
            raise ValueError(
                f"`num_latent_queries` must be > 0, got {self.num_latent_queries}"
            )
        if self.query_rope_offset < 0:
            raise ValueError(
                f"`query_rope_offset` must be >= 0, got {self.query_rope_offset}"
            )

        self.action_encoder = nn.Linear(action_dim, hidden_dim)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_dim=hidden_dim,
                    attn_head_dim=attn_head_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    eps=eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.head = nn.Linear(hidden_dim, action_dim)
        self.freqs = precompute_freqs_cis(attn_head_dim, end=1024)
        if self.use_latent_action_queries:
            if self.query_rope_offset + self.num_latent_queries > self.freqs.shape[0]:
                raise ValueError(
                    "Latent-query RoPE positions exceed the cache: "
                    f"offset={self.query_rope_offset}, queries={self.num_latent_queries}, "
                    f"cache={self.freqs.shape[0]}."
                )
            self.latent_action_queries = nn.Parameter(
                torch.randn(1, self.num_latent_queries, hidden_dim) / hidden_dim**0.5
            )

        self.use_gradient_checkpointing = use_gradient_checkpointing

    @classmethod
    def backbone_key_set(cls, keys) -> set[str]:
        return {
            key
            for key in keys
            if not any(key.startswith(prefix) for prefix in cls.ACTION_BACKBONE_SKIP_PREFIXES)
        }

    @classmethod
    def from_pretrained(
        cls,
        action_dit_config: dict[str, Any],
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "ActionDiT":
        if action_dit_config is None:
            raise ValueError("`action_dit_config` is required for ActionDiT.from_pretrained().")
        if skip_dit_load_from_pretrain:
            logger.info(
                "Skipping ActionDiT pretrained load (`skip_dit_load_from_pretrain=True`); "
                "initializing action expert randomly and expecting checkpoint override."
            )
            return cls(**action_dit_config).to(device=device, dtype=torch_dtype)
        if not action_dit_pretrained_path:
            logger.info("No `action_dit_pretrained_path` provided, initializing ActionDiT with random weights.")
            return cls(**action_dit_config).to(device=device, dtype=torch_dtype)
        from pathlib import Path
        p = Path(action_dit_pretrained_path)
        if not p.is_absolute():
            p = Path(__file__).resolve().parents[4] / p
        action_dit_pretrained_path = str(p)
        if not os.path.isfile(action_dit_pretrained_path):
            raise FileNotFoundError(
                f"`action_dit_pretrained_path` does not exist: {action_dit_pretrained_path}"
            )

        action_cfg = dict(action_dit_config)
        action_expert = cls(**action_cfg).to(device=device, dtype=torch_dtype)
        action_state = action_expert.state_dict()
        expected_backbone_keys = cls.backbone_key_set(action_state.keys())

        payload = torch.load(action_dit_pretrained_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(
                f"Invalid action backbone payload type from {action_dit_pretrained_path}: {type(payload)}"
            )
        
        policy = payload.get("policy", {})
        if policy:
            logger.info(f"ActionDiT backbone payload policy: {policy}")

        meta = payload.get("meta")
        expected_meta = {
            "hidden_dim": int(action_cfg["hidden_dim"]),
            "ffn_dim": int(action_cfg["ffn_dim"]),
            "num_layers": int(action_cfg["num_layers"]),
            "num_heads": int(action_cfg["num_heads"]),
            "attn_head_dim": int(action_cfg["attn_head_dim"]),
            "text_dim": int(action_cfg["text_dim"]),
            "freq_dim": int(action_cfg["freq_dim"]),
            "eps": float(action_cfg["eps"]),
        }
        for key in cls.ACTION_BACKBONE_META_KEYS:
            if key not in meta:
                raise ValueError(f"`meta.{key}` missing in {action_dit_pretrained_path}")
            expected_value = expected_meta[key]
            got_value = meta[key]
            if key == "eps":
                if abs(float(got_value) - float(expected_value)) > 1e-12:
                    raise ValueError(
                        f"`meta.{key}` mismatch in {action_dit_pretrained_path}: "
                        f"expected {expected_value}, got {got_value}"
                    )
            elif int(got_value) != int(expected_value):
                raise ValueError(
                    f"`meta.{key}` mismatch in {action_dit_pretrained_path}: "
                    f"expected {expected_value}, got {got_value}"
                )

        backbone_state_dict = payload.get("backbone_state_dict")
        if not isinstance(backbone_state_dict, dict):
            raise ValueError(
                f"`backbone_state_dict` must be a dict in {action_dit_pretrained_path}, "
                f"got {type(backbone_state_dict)}"
            )

        provided_keys = set(backbone_state_dict.keys())
        missing_keys = sorted(expected_backbone_keys - provided_keys)
        unexpected_keys = sorted(provided_keys - expected_backbone_keys)
        if missing_keys or unexpected_keys:
            raise ValueError(
                "Action backbone key mismatch in preprocessed payload. "
                f"missing={missing_keys[:10]}{'...' if len(missing_keys) > 10 else ''}, "
                f"unexpected={unexpected_keys[:10]}{'...' if len(unexpected_keys) > 10 else ''}"
            )

        merged_state = dict(action_state)
        for key in expected_backbone_keys:
            value = backbone_state_dict[key]
            if not isinstance(value, torch.Tensor):
                raise ValueError(
                    f"`backbone_state_dict[{key}]` must be torch.Tensor in {action_dit_pretrained_path}, "
                    f"got {type(value)}"
                )
            target = merged_state[key]
            if tuple(value.shape) != tuple(target.shape):
                raise ValueError(
                    f"Shape mismatch for `{key}` in {action_dit_pretrained_path}: "
                    f"expected {tuple(target.shape)}, got {tuple(value.shape)}"
                )
            merged_state[key] = value.to(device=target.device, dtype=target.dtype)

        action_expert.load_state_dict(merged_state, strict=True)
        logger.info(
            "Loaded ActionDiT backbone from %s (keys=%d; random_kept_prefixes=%s).",
            action_dit_pretrained_path,
            len(expected_backbone_keys),
            list(cls.ACTION_BACKBONE_SKIP_PREFIXES),
        )
        return action_expert.to(device=device, dtype=torch_dtype)

    def pre_dit(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        *,
        use_queries: Optional[bool] = None,
        query_context_mask: Optional[torch.Tensor] = None,
        action_context_mask: Optional[torch.Tensor] = None,
        transition_query_tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if action_tokens.ndim != 3:
            raise ValueError(
                f"`action_tokens` must be 3D [B, T, action_dim], got shape {tuple(action_tokens.shape)}"
            )
        if action_tokens.shape[2] != self.action_dim:
            raise ValueError(
                f"`action_tokens` last dim must be {self.action_dim}, got {action_tokens.shape[2]}"
            )
        if timestep.ndim != 1:
            raise ValueError(f"`timestep` must be 1D [B] or [1], got shape {tuple(timestep.shape)}")
        if context.ndim != 3:
            raise ValueError(
                f"`context` must be 3D [B, L, D], got shape {tuple(context.shape)}"
            )

        batch_size = action_tokens.shape[0]
        if context.shape[0] != batch_size:
            raise ValueError(
                f"Batch mismatch between action tokens and text context: {batch_size} vs {context.shape[0]}"
            )
        if timestep.shape[0] not in (1, batch_size):
            raise ValueError(
                f"`timestep` length must be 1 or batch_size({batch_size}), got {timestep.shape[0]}"
            )
        if timestep.shape[0] == 1 and batch_size > 1:
            if self.training:
                raise ValueError("During training, action timestep length must match batch_size.")
            timestep = timestep.expand(batch_size)

        if context_mask is None:
            context_mask = torch.ones(
                (batch_size, context.shape[1]), dtype=torch.bool, device=context.device
            )
        else:
            if context_mask.ndim != 2:
                raise ValueError(f"`context_mask` must be 2D [B, L], got shape {tuple(context_mask.shape)}")
            if context_mask.shape[0] != batch_size or context_mask.shape[1] != context.shape[1]:
                raise ValueError(
                    f"`context_mask` shape must match `context` shape [B, L], got {tuple(context_mask.shape)} vs {tuple(context.shape)}"
                )

        action_seq_len = action_tokens.shape[1]
        if action_seq_len > self.freqs.shape[0]:
            raise ValueError(
                f"Action token length {action_seq_len} exceeds RoPE cache {self.freqs.shape[0]}."
            )

        if use_queries is None:
            use_queries = self.use_latent_action_queries
        use_queries = bool(use_queries)
        if use_queries and not self.use_latent_action_queries:
            raise ValueError(
                "`use_queries=True` requires ActionDiT to be constructed with "
                "`use_latent_action_queries=True`."
            )
        if not use_queries and (
            query_context_mask is not None
            or action_context_mask is not None
            or transition_query_tokens is not None
        ):
            raise ValueError(
                "Query-specific inputs require `use_queries=True`."
            )

        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod_action = self.time_projection(t).unflatten(1, (6, self.hidden_dim))

        action_emb = self.action_encoder(action_tokens)
        context_emb = self.text_embedding(context)

        num_queries = self.num_latent_queries if use_queries else 0
        if use_queries:
            if self.query_rope_offset + num_queries > self.freqs.shape[0]:
                raise ValueError(
                    "Latent-query RoPE positions exceed the cache: "
                    f"offset={self.query_rope_offset}, queries={num_queries}, "
                    f"cache={self.freqs.shape[0]}."
                )
            if transition_query_tokens is None:
                query_tokens = self.latent_action_queries.expand(
                    batch_size, -1, -1
                )
            else:
                expected = (batch_size, num_queries, self.hidden_dim)
                if tuple(transition_query_tokens.shape) != expected:
                    raise ValueError(
                        "`transition_query_tokens` shape must be "
                        f"{expected}, got {tuple(transition_query_tokens.shape)}."
                    )
                query_tokens = transition_query_tokens.to(
                    device=action_emb.device, dtype=action_emb.dtype
                )
            tokens = torch.cat([query_tokens, action_emb], dim=1)

            zero_timestep = torch.zeros_like(timestep)
            t_query = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, zero_timestep)
            )
            t_mod_query = self.time_projection(t_query).unflatten(
                1, (6, self.hidden_dim)
            )
            t_mod = torch.cat(
                [
                    t_mod_query[:, None].expand(-1, num_queries, -1, -1),
                    t_mod_action[:, None].expand(-1, action_seq_len, -1, -1),
                ],
                dim=1,
            )

            def _expand_context_mask(
                mask: Optional[torch.Tensor],
                *,
                seq_len: int,
                name: str,
            ) -> torch.Tensor:
                if mask is None:
                    mask = context_mask
                if mask.ndim == 2:
                    if mask.shape != (batch_size, context.shape[1]):
                        raise ValueError(
                            f"`{name}` shape must be [B,L], got {tuple(mask.shape)}."
                        )
                    return mask.unsqueeze(1).expand(-1, seq_len, -1)
                if mask.ndim == 3:
                    expected = (batch_size, seq_len, context.shape[1])
                    if tuple(mask.shape) != expected:
                        raise ValueError(
                            f"`{name}` shape must be {expected}, got {tuple(mask.shape)}."
                        )
                    return mask
                raise ValueError(
                    f"`{name}` must be 2D [B,L] or 3D [B,S,L], "
                    f"got shape {tuple(mask.shape)}."
                )

            query_attn_mask = _expand_context_mask(
                query_context_mask,
                seq_len=num_queries,
                name="query_context_mask",
            )
            action_attn_mask = _expand_context_mask(
                action_context_mask,
                seq_len=action_seq_len,
                name="action_context_mask",
            )
            context_attn_mask = torch.cat(
                [query_attn_mask, action_attn_mask], dim=1
            )
            query_freqs = self.freqs[
                self.query_rope_offset : self.query_rope_offset + num_queries
            ]
            action_freqs = self.freqs[:action_seq_len]
            freqs = torch.cat([query_freqs, action_freqs], dim=0).view(
                num_queries + action_seq_len, 1, -1
            ).to(tokens.device)
        else:
            tokens = action_emb
            t_mod = t_mod_action
            context_attn_mask = context_mask.unsqueeze(1).expand(
                -1, action_seq_len, -1
            )
            freqs = self.freqs[:action_seq_len].view(
                action_seq_len, 1, -1
            ).to(tokens.device)

        return {
            "tokens": tokens,
            "freqs": freqs,
            "t": t,
            "t_mod": t_mod,
            "context": context_emb,
            "context_mask": context_attn_mask,
            "meta": {
                "batch_size": batch_size,
                "seq_len": num_queries + action_seq_len,
                "num_queries": num_queries,
                "action_seq_len": action_seq_len,
                "uses_transition_query_override": (
                    transition_query_tokens is not None
                ),
            },
        }

    @property
    def transition_queries(self) -> torch.nn.Parameter:
        """Semantic alias that preserves the legacy checkpoint key."""
        if not self.use_latent_action_queries:
            raise AttributeError(
                "Transition queries are unavailable when latent queries are disabled."
            )
        return self.latent_action_queries

    def post_dit(self, tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        num_queries = int(pre_state.get("meta", {}).get("num_queries", 0))
        action_seq_len = int(
            pre_state.get("meta", {}).get(
                "action_seq_len", tokens.shape[1] - num_queries
            )
        )
        if tokens.shape[1] != num_queries + action_seq_len:
            raise ValueError(
                "ActionDiT output length does not match pre-DiT metadata: "
                f"tokens={tokens.shape[1]}, queries={num_queries}, "
                f"actions={action_seq_len}."
            )
        return self.head(tokens[:, num_queries:])

    def post_dit_with_hidden(
        self, tokens: torch.Tensor, pre_state: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """Return predictions plus hidden states without changing legacy callers."""
        num_queries = int(pre_state.get("meta", {}).get("num_queries", 0))
        action_hidden = tokens[:, num_queries:]
        transition_hidden = tokens[:, :num_queries]
        return {
            "action_pred": self.head(action_hidden),
            "transition_query_hidden": transition_hidden,
            "action_hidden": action_hidden,
        }

    def forward(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        *,
        use_queries: Optional[bool] = None,
        query_context_mask: Optional[torch.Tensor] = None,
        action_context_mask: Optional[torch.Tensor] = None,
        transition_query_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pre_state = self.pre_dit(
            action_tokens=action_tokens,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            use_queries=use_queries,
            query_context_mask=query_context_mask,
            action_context_mask=action_context_mask,
            transition_query_tokens=transition_query_tokens,
        )
        x = pre_state["tokens"]
        context = pre_state["context"]
        t_mod = pre_state["t_mod"]
        freqs = pre_state["freqs"]
        context_mask = pre_state["context_mask"]

        for block in self.blocks:
            if self.use_gradient_checkpointing:
                x = gradient_checkpoint_forward(
                    block,
                    self.use_gradient_checkpointing,
                    x,
                    context,
                    t_mod,
                    freqs,
                    context_mask=context_mask,
                )
            else:
                x = block(x, context, t_mod, freqs, context_mask=context_mask)

        return self.post_dit(x, pre_state)
