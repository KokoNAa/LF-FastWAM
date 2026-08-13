from pathlib import Path
from typing import Any, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from fastwam.utils.logging_config import get_logger

from .action_dit import ActionDiT
from .helpers.loader import load_wan22_ti2v_5b_components
from .lora import (
    inject_lora,
    is_lora_parameter_name,
    matches_any_pattern,
    normalize_lora_config,
)
from .mot import MoT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from .transition_contract import (
    ContrastiveContractLoss,
    OutcomeTransitionEncoder,
    TransitionProjectionHead,
    TransitionVisualRouter,
    detached_metrics,
)

logger = get_logger(__name__)


class FastWAM(torch.nn.Module):
    """MoT world model with video/action experts."""

    def __init__(
        self,
        video_expert,
        action_expert: ActionDiT,
        mot: MoT,
        vae,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        langforce_mvp_config: Optional[dict[str, Any]] = None,
        transition_contract_config: Optional[dict[str, Any]] = None,
    ):
        super().__init__()
        self.video_expert = video_expert
        self.action_expert = action_expert
        self.mot = mot
        # Keep trainer compatibility: optimizer and freeze logic use `model.dit`.
        self.dit = self.mot

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if text_dim is None:
            if self.text_encoder is None:
                raise ValueError("`text_dim` is required when `text_encoder` is not loaded.")
            text_dim = int(self.text_encoder.dim)
        self.text_dim = int(text_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        if self.proprio_dim is not None:
            self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim).to(torch_dtype)
        else:
            self.proprio_encoder = None

        self.train_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_train_shift,
        )
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_infer_shift,
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_train_shift,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_infer_shift,
        )
        # Optional aliases for consistency with Wan22Core naming.
        self.train_scheduler = self.train_video_scheduler
        self.infer_scheduler = self.infer_video_scheduler

        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)

        mvp_config = dict(langforce_mvp_config or {})
        self.langforce_mvp_enabled = bool(mvp_config.get("enabled", False))
        self.langforce_prior_enabled = self.langforce_mvp_enabled and bool(
            mvp_config.get("enable_prior", True)
        )
        self.langforce_advantage_enabled = self.langforce_prior_enabled and bool(
            mvp_config.get("enable_posterior_advantage", True)
        )
        self.lambda_prior_action = float(
            mvp_config.get("lambda_prior_action", 0.10)
        )
        self.lambda_posterior_advantage = float(
            mvp_config.get("lambda_posterior_advantage", 0.10)
        )
        self.posterior_advantage_margin_ratio = float(
            mvp_config.get("posterior_advantage_margin_ratio", 0.05)
        )
        if self.langforce_mvp_enabled:
            self.action_reads_raw_video = bool(
                mvp_config.get("action_reads_raw_video", False)
            )
            self.action_reads_language = bool(
                mvp_config.get("action_reads_language", False)
            )
        else:
            # `enabled=false` is a complete baseline switch even when the same
            # config block retains MVP-only false values for these flags.
            self.action_reads_raw_video = True
            self.action_reads_language = True
        self.detach_prior_video_cache = bool(
            mvp_config.get("detach_prior_video_cache", True)
        )

        if self.langforce_mvp_enabled:
            if not bool(getattr(self.action_expert, "use_latent_action_queries", False)):
                raise ValueError(
                    "LangForce MVP requires ActionDiT "
                    "`use_latent_action_queries=True`."
                )
            if self.action_reads_raw_video:
                raise ValueError(
                    "LangForce MVP requires `action_reads_raw_video=false`; "
                    "otherwise the visual shortcut remains open."
                )
            if self.action_reads_language:
                raise ValueError(
                    "LangForce MVP requires `action_reads_language=false`; "
                    "otherwise latent queries are not the task-semantic bottleneck."
                )
        if bool(mvp_config.get("enable_posterior_advantage", True)) and (
            self.langforce_mvp_enabled
            and not bool(mvp_config.get("enable_prior", True))
        ):
            raise ValueError(
                "Posterior-advantage loss requires `enable_prior=true`."
            )
        if self.lambda_prior_action < 0 or self.lambda_posterior_advantage < 0:
            raise ValueError("LangForce loss weights must be non-negative.")
        if not 0.0 <= self.posterior_advantage_margin_ratio < 1.0:
            raise ValueError(
                "`posterior_advantage_margin_ratio` must be in [0, 1)."
            )

        contract_config = dict(transition_contract_config or {})
        self.transition_contract_enabled = bool(
            contract_config.get("enabled", False)
        )
        self.transition_contract_version = int(contract_config.get("version", 2))
        self.transition_contract_weight = float(
            contract_config.get("contract_weight", 0.05)
        )
        self.transition_contract_temperature = float(
            contract_config.get("temperature", 0.07)
        )
        self.transition_contract_warmup_ratio = float(
            contract_config.get("warmup_ratio", 0.05)
        )
        self.transition_contract_ramp_ratio = float(
            contract_config.get("ramp_ratio", 0.05)
        )
        self.transition_policy_recovery_ratio = float(
            contract_config.get("policy_recovery_ratio", 0.10)
        )
        self.transition_router_ramp_ratio = float(
            contract_config.get("router_ramp_ratio", 0.20)
        )
        self.transition_freeze_m1_during_recovery = bool(
            contract_config.get("freeze_m1_during_recovery", True)
        )
        self.outcome_stop_gradient = bool(
            contract_config.get("outcome_stop_gradient", True)
        )
        self.use_transition_router = bool(
            contract_config.get("use_transition_router", True)
        )
        self.transition_contract_modules = nn.ModuleDict()
        self.transition_contract_loss = None
        self._transition_training_step = 0
        self._transition_training_max_steps = 1
        self._transition_training_progress_active = False
        self.transition_policy_init_checkpoint: Optional[str] = None

        if self.transition_contract_enabled:
            if self.transition_contract_version != 2:
                raise ValueError(
                    "The current TC-C implementation requires "
                    "`transition_contract.version=2`."
                )
            if not bool(
                getattr(self.action_expert, "use_latent_action_queries", False)
            ):
                raise ValueError(
                    "Transition Contract requires ActionDiT "
                    "`use_latent_action_queries=True`."
                )
            if not self.use_transition_router:
                raise ValueError("TC-C Stage 1 requires `use_transition_router=true`.")
            if bool(contract_config.get("direct_action_video_access", False)):
                raise ValueError(
                    "TC-C forbids `direct_action_video_access=true`; the Action "
                    "Expert must consume routed transition tokens only."
                )
            if bool(contract_config.get("direct_action_text_access", False)):
                raise ValueError(
                    "TC-C forbids `direct_action_text_access=true`; language must "
                    "reach actions through the Transition Router."
                )
            if not bool(contract_config.get("direct_video_text_access", True)):
                raise ValueError(
                    "Stage 1 is Phase T1 and requires `direct_video_text_access=true`."
                )
            if bool(contract_config.get("action_conditioned_video", False)):
                raise ValueError(
                    "`action_conditioned_video` belongs to Stage 3, not TC-C Stage 1."
                )
            if bool(contract_config.get("use_action_effect", False)):
                raise ValueError("`use_action_effect` belongs to Stage 2.")
            if bool(contract_config.get("use_counterfactual_ranking", False)):
                raise ValueError("Counterfactual ranking belongs to Stage 2.")
            if self.langforce_prior_enabled or self.langforce_advantage_enabled:
                raise ValueError(
                    "TC-C mainline requires LangForce prior/advantage ablations off."
                )
            if self.transition_contract_weight < 0:
                raise ValueError("`contract_weight` must be non-negative.")
            if not 0.0 <= self.transition_contract_warmup_ratio < 1.0:
                raise ValueError("`warmup_ratio` must be in [0,1).")
            if not 0.0 <= self.transition_contract_ramp_ratio < 1.0:
                raise ValueError("`ramp_ratio` must be in [0,1).")
            if not 0.0 <= self.transition_policy_recovery_ratio < 1.0:
                raise ValueError("`policy_recovery_ratio` must be in [0,1).")
            if not 0.0 <= self.transition_router_ramp_ratio < 1.0:
                raise ValueError("`router_ramp_ratio` must be in [0,1).")
            if (
                self.transition_policy_recovery_ratio
                + self.transition_router_ramp_ratio
                > 1.0
            ):
                raise ValueError(
                    "`policy_recovery_ratio + router_ramp_ratio` must be <= 1."
                )

            projection_dim = int(contract_config.get("projection_dim", 512))
            router_num_heads = int(contract_config.get("router_num_heads", 8))
            action_hidden_dim = int(self.action_expert.hidden_dim)
            video_hidden_dim = int(self.video_expert.hidden_dim)
            self.transition_contract_modules.update(
                {
                    "router": TransitionVisualRouter(
                        action_dim=action_hidden_dim,
                        video_dim=video_hidden_dim,
                        num_heads=router_num_heads,
                    ),
                    "intent_projection": TransitionProjectionHead(
                        action_hidden_dim, projection_dim
                    ),
                    "outcome_encoder": OutcomeTransitionEncoder(
                        video_hidden_dim, projection_dim
                    ),
                }
            )
            self.transition_contract_modules.to(dtype=self.torch_dtype)
            self.transition_contract_loss = ContrastiveContractLoss(
                temperature=self.transition_contract_temperature
            )

        self.uses_transition_queries = bool(
            self.langforce_mvp_enabled or self.transition_contract_enabled
        )
        if self.transition_contract_enabled:
            self.action_reads_raw_video = False
            self.action_reads_language = False

        self.lora_config = normalize_lora_config(None)
        self.lora_enabled = False
        self.lora_base_checkpoint: Optional[str] = None

        self.to(self.device)

    def set_training_progress(self, step: int, max_steps: int) -> None:
        """Set optimizer-step progress used by Stage-1 contract warm-up."""
        self._transition_training_step = max(0, int(step))
        self._transition_training_max_steps = max(1, int(max_steps))
        self._transition_training_progress_active = True

    def _transition_contract_scale(self) -> float:
        progress = self._transition_training_step / self._transition_training_max_steps
        warmup_end = self.transition_contract_warmup_ratio
        ramp_end = warmup_end + self.transition_contract_ramp_ratio
        if progress < warmup_end:
            return 0.0
        if self.transition_contract_ramp_ratio <= 0 or progress >= ramp_end:
            return 1.0
        scale = (progress - warmup_end) / self.transition_contract_ramp_ratio
        # Snap arithmetic representations of an exact schedule endpoint (for
        # example 0.3 vs 0.1 + 0.2) to the intended closed interval boundary.
        if scale >= 1.0 - 1.0e-12:
            return 1.0
        if scale <= 1.0e-12:
            return 0.0
        return float(scale)

    def _transition_router_scale(self) -> float:
        """Return the explicit training recovery scale; deployment is pure Router."""
        if not self._transition_training_progress_active:
            return 1.0
        progress = self._transition_training_step / self._transition_training_max_steps
        recovery_end = self.transition_policy_recovery_ratio
        ramp_end = recovery_end + self.transition_router_ramp_ratio
        if progress < recovery_end:
            return 0.0
        if self.transition_router_ramp_ratio <= 0 or progress >= ramp_end:
            return 1.0
        scale = (progress - recovery_end) / self.transition_router_ramp_ratio
        if scale >= 1.0 - 1.0e-12:
            return 1.0
        if scale <= 1.0e-12:
            return 0.0
        return float(scale)

    def configure_lora(self, config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        normalized = normalize_lora_config(config)
        if not normalized["enabled"]:
            if self.lora_enabled:
                raise ValueError("Cannot disable LoRA after adapters have been injected.")
            self.lora_config = normalized
            return {"enabled": False, "modules": [], "parameters": 0}

        if self.lora_enabled:
            comparable_keys = (
                "rank",
                "alpha",
                "dropout",
                "experts",
                "target_modules",
                "extra_trainable_patterns",
            )
            mismatch = {
                key: (self.lora_config.get(key), normalized.get(key))
                for key in comparable_keys
                if self.lora_config.get(key) != normalized.get(key)
            }
            if mismatch:
                raise ValueError(f"LoRA is already configured differently: {mismatch}")
            return self.lora_report()

        injected: list[str] = []
        expert_modules = {
            "video": self.video_expert,
            "action": self.action_expert,
        }
        for expert_name in normalized["experts"]:
            names = inject_lora(
                expert_modules[expert_name],
                target_modules=normalized["target_modules"],
                rank=normalized["rank"],
                alpha=normalized["alpha"],
                dropout=normalized["dropout"],
            )
            injected.extend(f"{expert_name}_expert.{name}" for name in names)

        self.lora_config = normalized
        self.lora_enabled = True
        report = self.lora_report()
        logger.info(
            "Injected LoRA: experts=%s rank=%d alpha=%.2f dropout=%.3f "
            "modules=%d adapter_parameters=%d",
            normalized["experts"],
            normalized["rank"],
            normalized["alpha"],
            normalized["dropout"],
            len(report["modules"]),
            report["parameters"],
        )
        return report

    def lora_report(self) -> dict[str, Any]:
        modules = []
        parameter_count = 0
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and hasattr(module, "lora_A"):
                modules.append(name)
                parameter_count += module.lora_A.numel() + module.lora_B.numel()
        return {
            "enabled": self.lora_enabled,
            "modules": modules,
            "parameters": int(parameter_count),
        }

    def _adapter_parameter_ids(self) -> set[int]:
        if not self.lora_enabled:
            return set()
        extra_patterns = list(self.lora_config["extra_trainable_patterns"])
        if self.transition_contract_enabled:
            extra_patterns.append("transition_contract_modules.*")
        return {
            id(parameter)
            for name, parameter in self.named_parameters()
            if is_lora_parameter_name(name)
            or matches_any_pattern(name, extra_patterns)
        }

    def prepare_trainable_parameters(self) -> dict[str, int]:
        """Freeze the base model and expose only configured adapter parameters."""
        self.eval()
        self.requires_grad_(False)
        self.dit.train()

        if self.lora_enabled:
            adapter_ids = self._adapter_parameter_ids()
            for parameter in self.parameters():
                if id(parameter) in adapter_ids:
                    parameter.requires_grad_(True)
        else:
            self.dit.requires_grad_(True)
            if self.transition_contract_enabled:
                self.transition_contract_modules.train()
                self.transition_contract_modules.requires_grad_(True)
            if self.proprio_encoder is not None:
                self.proprio_encoder.train()
                self.proprio_encoder.requires_grad_(True)

        if self.transition_contract_enabled:
            self.transition_contract_modules.train()

        trainable = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        total = sum(parameter.numel() for parameter in self.parameters())
        if trainable <= 0:
            raise ValueError("Training configuration produced zero trainable parameters.")
        return {
            "trainable": int(trainable),
            "total": int(total),
        }

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        langforce_mvp_config: Optional[dict[str, Any]] = None,
        transition_contract_config: Optional[dict[str, Any]] = None,
        lora_config: Optional[dict[str, Any]] = None,
    ):
        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required for FastWAM.from_wan22_pretrained().")
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required for FastWAM.")

        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )

        video_expert = components.dit
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        if int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError("ActionDiT `num_heads` must match video expert for MoT mixed attention.")
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError("ActionDiT `attn_head_dim` must match video expert for MoT mixed attention.")
        if int(len(action_expert.blocks)) != int(len(video_expert.blocks)):
            raise ValueError("ActionDiT `num_layers` must match video expert.")

        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            langforce_mvp_config=langforce_mvp_config,
            transition_contract_config=transition_contract_config,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path
            ),
        }
        model.configure_lora(lora_config)
        return model

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.mot.to(*args, **kwargs)
        if self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
        self.vae.to(*args, **kwargs)
        return self

    @staticmethod
    def _check_resize_height_width(height, width, num_frames):
        if height % 16 != 0:
            height = (height + 15) // 16 * 16
        if width % 16 != 0:
            width = (width + 15) // 16 * 16
        if num_frames % 4 != 1:
            num_frames = (num_frames + 3) // 4 * 4 + 1
        return height, width, num_frames

    @torch.no_grad()
    def encode_prompt(self, prompt: Union[str, Sequence[str]]):
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError(
                "Prompt encoding requires loaded text encoder/tokenizer. "
                "Set `load_text_encoder=true` or provide precomputed `context/context_mask`."
            )
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        # FIXME: original implementation's zero padding is visible in cross-attn.
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        mask = torch.ones_like(mask)
        return prompt_emb.to(device=self.device), mask

    def _append_proprio_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None or proprio is None:
            return context, context_mask
        if proprio.ndim != 2:
            raise ValueError(f"`proprio` must be 2D [B, D], got shape {tuple(proprio.shape)}")
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(
                f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
            )
        proprio_token = self.proprio_encoder(
            proprio.to(device=self.device, dtype=context.dtype).unsqueeze(1)
        ).to(dtype=context.dtype) # [B, 1, D]
        proprio_mask = torch.ones((context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device)
        return (
            torch.cat([context, proprio_token], dim=1),
            torch.cat([context_mask, proprio_mask], dim=1),
        )

    @staticmethod
    def _build_context_masks(
        full_context_mask: torch.Tensor,
        language_context_len: int,
        has_proprio: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Separate language validity from appended state-token validity."""
        if full_context_mask.ndim != 2:
            raise ValueError(
                "`full_context_mask` must be [B,L], got "
                f"{tuple(full_context_mask.shape)}."
            )
        if full_context_mask.dtype != torch.bool:
            raise ValueError("`full_context_mask` must have boolean dtype.")
        context_len = int(full_context_mask.shape[1])
        language_context_len = int(language_context_len)
        if not 0 <= language_context_len <= context_len:
            raise ValueError(
                "`language_context_len` must lie within the context sequence, "
                f"got {language_context_len} for length {context_len}."
            )
        if has_proprio and language_context_len >= context_len:
            raise ValueError(
                "`has_proprio=True` requires state tokens after the language prefix."
            )
        if not has_proprio and language_context_len != context_len:
            raise ValueError(
                "Context contains a non-language suffix while `has_proprio=False`."
            )

        full_mask = full_context_mask.clone()
        state_only_mask = torch.zeros_like(full_mask)
        if has_proprio:
            state_only_mask[:, language_context_len:] = full_mask[
                :, language_context_len:
            ]
        return full_mask, state_only_mask

    @staticmethod
    def _build_action_context_mask(
        *,
        full_mask: torch.Tensor,
        state_only_mask: torch.Tensor,
        num_queries: int,
        action_horizon: int,
        mode: str,
    ) -> torch.Tensor:
        if full_mask.ndim != 2 or state_only_mask.ndim != 2:
            raise ValueError("Action context source masks must be 2D [B,L].")
        if full_mask.shape != state_only_mask.shape:
            raise ValueError("Full and state-only context masks must have equal shape.")
        if full_mask.dtype != torch.bool or state_only_mask.dtype != torch.bool:
            raise ValueError("Action context masks must have boolean dtype.")
        num_queries = int(num_queries)
        action_horizon = int(action_horizon)
        if num_queries <= 0 or action_horizon <= 0:
            raise ValueError("Query count and action horizon must both be positive.")
        if mode not in {"posterior", "prior"}:
            raise ValueError(f"Unsupported action context mode: {mode}")

        query_source = full_mask if mode == "posterior" else state_only_mask
        query_mask = query_source[:, None, :].expand(-1, num_queries, -1)
        action_mask = state_only_mask[:, None, :].expand(
            -1, action_horizon, -1
        )
        return torch.cat([query_mask, action_mask], dim=1)

    def _prepare_action_tokens(
        self,
        *,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        full_context_mask: torch.Tensor,
        state_only_context_mask: torch.Tensor,
        mode: str,
        transition_query_tokens: Optional[torch.Tensor] = None,
        policy_recovery: bool = False,
    ) -> dict[str, Any]:
        if not self.uses_transition_queries:
            return self.action_expert.pre_dit(
                action_tokens=action_tokens,
                timestep=timestep,
                context=context,
                context_mask=full_context_mask,
                use_queries=False,
            )

        num_queries = int(self.action_expert.num_latent_queries)
        if self.transition_contract_enabled and not policy_recovery:
            if transition_query_tokens is None:
                raise ValueError(
                    "TC-C action preparation requires routed transition query tokens."
                )
            # Language and raw visual evidence have already been fused into the
            # routed prefix. The Action Expert may still read robot state.
            query_mask = state_only_context_mask[:, None, :].expand(
                -1, num_queries, -1
            )
            action_mask = state_only_context_mask[:, None, :].expand(
                -1, int(action_tokens.shape[1]), -1
            )
            token_context_mask = torch.cat([query_mask, action_mask], dim=1)
        else:
            token_context_mask = self._build_action_context_mask(
                full_mask=full_context_mask,
                state_only_mask=state_only_context_mask,
                num_queries=num_queries,
                action_horizon=int(action_tokens.shape[1]),
                mode=mode,
            )
        return self.action_expert.pre_dit(
            action_tokens=action_tokens,
            timestep=timestep,
            context=context,
            context_mask=full_context_mask,
            use_queries=True,
            query_context_mask=token_context_mask[:, :num_queries],
            action_context_mask=token_context_mask[:, num_queries:],
            transition_query_tokens=transition_query_tokens,
        )

    def encode_intended_transition(
        self,
        *,
        video_tokens: torch.Tensor,
        video_tokens_per_frame: int,
        context: torch.Tensor,
        full_context_mask: torch.Tensor,
        route_scale: Optional[float] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Encode language intent and route current visual evidence."""
        if not self.transition_contract_enabled:
            raise RuntimeError("Transition Contract is disabled.")
        current_video = video_tokens[:, : int(video_tokens_per_frame)]
        if route_scale is None:
            route_scale = self._transition_router_scale()
        batch_size = current_video.shape[0]
        transition_queries = self.action_expert.transition_queries.expand(
            batch_size, -1, -1
        )
        language_hidden = self.action_expert.text_embedding(context)
        routed, router_metrics = self.transition_contract_modules["router"](
            transition_queries=transition_queries,
            language_hidden=language_hidden,
            language_mask=full_context_mask,
            current_video_hidden=current_video,
            route_scale=route_scale,
        )
        z_language = self.transition_contract_modules["intent_projection"](
            routed.mean(dim=1)
        )
        return routed, z_language, router_metrics

    def encode_realized_transition(
        self,
        *,
        clean_input_latents: torch.Tensor,
        context: torch.Tensor,
        full_context_mask: torch.Tensor,
        action: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
    ) -> torch.Tensor:
        """Encode clean current/future Video-Expert patch change for training."""
        if not self.transition_contract_enabled:
            raise RuntimeError("Transition Contract is disabled.")
        teacher_timestep = torch.zeros(
            clean_input_latents.shape[0],
            device=clean_input_latents.device,
            dtype=clean_input_latents.dtype,
        )
        grad_context = (
            torch.no_grad()
            if self.outcome_stop_gradient
            else torch.enable_grad()
        )
        with grad_context:
            teacher_pre = self.video_expert.pre_dit(
                x=clean_input_latents,
                timestep=teacher_timestep,
                context=context,
                context_mask=full_context_mask,
                action=action,
                fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            )
        teacher_tokens = teacher_pre["tokens"]
        if self.outcome_stop_gradient:
            teacher_tokens = teacher_tokens.detach()
        current_hidden, future_hidden = (
            self.video_expert.split_current_future_hidden(
                teacher_tokens,
                tokens_per_frame=int(
                    teacher_pre["meta"]["tokens_per_frame"]
                ),
            )
        )
        return self.transition_contract_modules[
            "outcome_encoder"
        ].from_hidden_pair(
            current_hidden, future_hidden
        )

    def compute_transition_contract_loss(
        self, z_language: torch.Tensor, z_future: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.transition_contract_loss is None:
            raise RuntimeError("Transition Contract loss is unavailable.")
        return self.transition_contract_loss(z_language, z_future)

    def _run_video_expert_to_final_hidden(
        self,
        video_pre: dict[str, Any],
    ) -> tuple[list[dict[str, torch.Tensor]], torch.Tensor]:
        """Run the full Video Expert once and expose its semantic hidden state."""
        video_seq_len = int(video_pre["tokens"].shape[1])
        video_attention_mask = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=int(
                video_pre["meta"]["tokens_per_frame"]
            ),
            device=video_pre["tokens"].device,
        )
        result = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=video_attention_mask,
            return_final_hidden=True,
        )
        if not isinstance(result, tuple):
            raise RuntimeError("Video Expert did not return final hidden states.")
        return result

    def _forward_tc_v2_action_from_video_hidden(
        self,
        *,
        video_pre: dict[str, Any],
        final_video_hidden: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        action_tokens: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        full_context_mask: torch.Tensor,
        state_only_context_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Route final Video hidden and recover M1 before pure-Router training.

        Recovery is interpolated on predicted action velocity, not only on the
        input query embeddings. The M1 interface also lets posterior queries
        read current-frame Video K/V and T5 context at every Action-Expert
        layer, so query-only interpolation would remove those paths abruptly.
        """
        route_scale = self._transition_router_scale()
        routed_full, z_language, router_metrics = self.encode_intended_transition(
            video_tokens=final_video_hidden,
            video_tokens_per_frame=int(
                video_pre["meta"]["tokens_per_frame"]
            ),
            context=context,
            full_context_mask=full_context_mask,
            # Contract supervision always sees the full Router output once its
            # configured warm-up begins. Only the policy output is blended.
            route_scale=1.0,
        )
        base_queries = self.action_expert.transition_queries.expand(
            routed_full.shape[0], -1, -1
        )
        router_metrics["router_route_scale"] = routed_full.new_tensor(route_scale)
        router_metrics["router_policy_residual_norm"] = (
            (routed_full - base_queries).float().norm(dim=-1).mean()
        )

        def _run_policy(
            transition_tokens: torch.Tensor,
            *,
            m1_posterior_interface: bool,
        ) -> torch.Tensor:
            action_pre = self._prepare_action_tokens(
                action_tokens=action_tokens,
                timestep=timestep_action,
                context=context,
                full_context_mask=full_context_mask,
                state_only_context_mask=state_only_context_mask,
                mode="posterior",
                transition_query_tokens=transition_tokens,
                policy_recovery=m1_posterior_interface,
            )
            attention_mask = self._build_mot_attention_mask(
                video_seq_len=int(video_pre["tokens"].shape[1]),
                action_seq_len=int(action_pre["tokens"].shape[1]),
                video_tokens_per_frame=int(
                    video_pre["meta"]["tokens_per_frame"]
                ),
                device=video_pre["tokens"].device,
                num_queries=int(action_pre["meta"]["num_queries"]),
                action_reads_raw_video=False,
                queries_read_raw_video=m1_posterior_interface,
            )
            action_hidden = self.mot.forward_action_with_video_cache(
                action_tokens=action_pre["tokens"],
                action_freqs=action_pre["freqs"],
                action_t_mod=action_pre["t_mod"],
                action_context_payload={
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=int(video_pre["tokens"].shape[1]),
            )
            return self.action_expert.post_dit(action_hidden, action_pre)

        if route_scale <= 0.0:
            # Exact M1 posterior-query policy.
            pred_action = _run_policy(
                base_queries,
                m1_posterior_interface=True,
            )
            router_metrics["policy_recovery_output_gap"] = pred_action.new_zeros(())
            return pred_action, z_language, router_metrics

        if route_scale >= 1.0:
            # Final TC-C policy: Router is the only language/visual interface.
            pred_action = _run_policy(
                routed_full,
                m1_posterior_interface=False,
            )
            router_metrics["policy_recovery_output_gap"] = pred_action.new_zeros(())
            return pred_action, z_language, router_metrics

        # Both branches share the same Video cache. This linear flow-velocity
        # blend makes the policy function continuous while shortcuts disappear.
        pred_action_m1 = _run_policy(
            base_queries,
            m1_posterior_interface=True,
        )
        pred_action_router = _run_policy(
            routed_full,
            m1_posterior_interface=False,
        )
        pred_action = torch.lerp(
            pred_action_m1,
            pred_action_router,
            float(route_scale),
        )
        router_metrics["policy_recovery_output_gap"] = (
            (pred_action_router - pred_action_m1)
            .float()
            .norm(dim=-1)
            .mean()
        )
        return pred_action, z_language, router_metrics

    @torch.no_grad()
    def _encode_video_latents(self, video_tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        z = self.vae.encode(
            video_tensor,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        return z

    @torch.no_grad()
    def _encode_input_image_latents_tensor(self, input_image: torch.Tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        image = input_image.to(device=self.device)[0].unsqueeze(1)
        z = self.vae.encode([image], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if isinstance(z, list):
            z = z[0].unsqueeze(0)
        return z

    def _decode_latents(self, latents, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        video_tensor = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video_tensor = video_tensor.squeeze(0).detach().float().clamp(-1, 1)
        video_tensor = ((video_tensor + 1.0) * 127.5).to(torch.uint8).cpu()
        frames = []
        for t in range(video_tensor.shape[1]):
            frame = video_tensor[:, t].permute(1, 2, 0).numpy()
            frames.append(Image.fromarray(frame))
        return frames

    def build_inputs(self, sample, tiled: bool = False):
        video = sample["video"]
        if "context" not in sample or "context_mask" not in sample:
            raise ValueError(
                "FastWAM training requires `sample['context']` and `sample['context_mask']`."
            )
        context = sample["context"]
        context_mask = sample["context_mask"]
        proprio = sample.get("proprio", None)
        if video.ndim != 5:
            raise ValueError(f"`sample['video']` must be 5D [B, 3, T, H, W], got shape {tuple(video.shape)}")
        if video.shape[1] != 3:
            raise ValueError(f"`sample['video']` channel dimension must be 3, got shape {tuple(video.shape)}")

        batch_size, _, num_frames, height, width = video.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"Video spatial dims must be multiples of 16, got H={height}, W={width}"
            )
        if num_frames % 4 != 1:
            raise ValueError(f"Video T must satisfy T % 4 == 1, got T={num_frames}")
        if num_frames <= 1:
            raise ValueError(f"Video T must be > 1 for action-conditioned training, got T={num_frames}")

        if "action" not in sample:
            raise ValueError("`sample['action']` is required for FastWAM training.")

        action = sample["action"]
        if action.ndim != 3:
            raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
        action_horizon = int(action.shape[1])
        if action_horizon % (num_frames - 1) != 0:
            raise ValueError(
                f"`sample['action']` temporal dimension must be divisible by video transitions ({num_frames - 1}), got {action_horizon}"
            )

        action_is_pad = sample.get("action_is_pad", None)
        if action_is_pad is not None:
            if action_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['action_is_pad']` must be 2D [B, T], got shape {tuple(action_is_pad.shape)}"
                )
            if action_is_pad.shape[0] != batch_size or action_is_pad.shape[1] != action_horizon:
                raise ValueError(
                    "`sample['action_is_pad']` shape mismatch: "
                    f"got {tuple(action_is_pad.shape)} vs expected ({batch_size}, {action_horizon})"
                )

        image_is_pad = sample.get("image_is_pad", None)
        if image_is_pad is not None:
            if image_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['image_is_pad']` must be 2D [B, T], got shape {tuple(image_is_pad.shape)}"
                )
            if image_is_pad.shape[0] != batch_size or image_is_pad.shape[1] != num_frames:
                raise ValueError(
                    "`sample['image_is_pad']` shape mismatch: "
                    f"got {tuple(image_is_pad.shape)} vs expected ({batch_size}, {num_frames})"
                )
        
        input_video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        input_latents = self._encode_video_latents(input_video, tiled=tiled)

        first_frame_latents = None
        fuse_flag = False
        if getattr(self.video_expert, "fuse_vae_embedding_in_latents", False):
            first_frame_latents = input_latents[:, :, 0:1]
            fuse_flag = True

        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        language_context_len = int(context.shape[1])
        has_proprio = False
        if self.proprio_encoder is not None:
            if proprio is None:
                raise ValueError("`sample['proprio']` is required when `proprio_dim` is enabled.")
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")
            if proprio.shape[2] != self.proprio_dim:
                raise ValueError(
                    f"`sample['proprio']` last dim must be {self.proprio_dim}, got {proprio.shape[2]}"
                )
            proprio = proprio[:, 0, :] # [B, D]
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
            )
            has_proprio = True
        action = action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)

        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)

        return {
            "context": context,
            "context_mask": context_mask,
            "language_context_len": language_context_len,
            "has_proprio": has_proprio,
            "input_latents": input_latents,
            "first_frame_latents": first_frame_latents,
            "fuse_vae_embedding_in_latents": fuse_flag,
            "action": action,
            "action_is_pad": action_is_pad,
            "image_is_pad": image_is_pad,
        }

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
        *,
        num_queries: int = 0,
        action_reads_raw_video: bool = True,
        queries_read_raw_video: bool = True,
    ) -> torch.Tensor:
        video_seq_len = int(video_seq_len)
        action_seq_len = int(action_seq_len)
        num_queries = int(num_queries)
        if video_seq_len <= 0 or action_seq_len <= 0:
            raise ValueError("Video and action-expert sequence lengths must be positive.")
        if not 0 <= num_queries < action_seq_len:
            raise ValueError(
                "`num_queries` must be in [0, action_seq_len), got "
                f"{num_queries} and {action_seq_len}."
            )
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)

        # video -> video
        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
        if num_queries == 0:
            # Baseline: action tokens read each other and the current frame.
            mask[video_seq_len:, video_seq_len:] = True
            if action_reads_raw_video:
                mask[video_seq_len:, :first_frame_tokens] = True
            return mask

        query_start = video_seq_len
        query_end = query_start + num_queries
        action_start = query_end
        action_end = total_seq_len

        # Latent queries read only the current frame and other queries. They
        # cannot see privileged future-video or noisy action tokens.
        if queries_read_raw_video:
            mask[query_start:query_end, :first_frame_tokens] = True
        mask[query_start:query_end, query_start:query_end] = True

        # Actions consume the bottleneck plus other action tokens. MVP keeps
        # direct raw-video access disabled by construction.
        mask[action_start:action_end, query_start:query_end] = True
        mask[action_start:action_end, action_start:action_end] = True
        if action_reads_raw_video:
            mask[action_start:action_end, :first_frame_tokens] = True
        return mask

    def _compute_video_loss_per_sample(
        self,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        image_is_pad: Optional[torch.Tensor],
        include_initial_video_step: bool,
    ) -> torch.Tensor:
        video_loss_token = F.mse_loss(pred_video.float(), target_video.float(), reduction="none").mean(dim=(1, 3, 4))
        if image_is_pad is None:
            return video_loss_token.mean(dim=1)

        temporal_factor = int(self.vae.temporal_downsample_factor)
        if temporal_factor <= 0:
            raise ValueError(f"`vae.temporal_downsample_factor` must be positive, got {temporal_factor}.")
        if image_is_pad.shape[1] < 1:
            raise ValueError("`image_is_pad` must contain at least one frame.")
        if (image_is_pad.shape[1] - 1) % temporal_factor != 0:
            raise ValueError(
                "Cannot align `image_is_pad` with video latent steps: "
                f"num_frames={image_is_pad.shape[1]}, temporal_downsample_factor={temporal_factor}."
            )

        tail_is_pad = image_is_pad[:, 1:]
        latent_tail_is_pad = tail_is_pad.view(image_is_pad.shape[0], -1, temporal_factor).all(dim=2)
        if include_initial_video_step:
            video_is_pad = torch.cat([image_is_pad[:, :1], latent_tail_is_pad], dim=1)
        else:
            video_is_pad = latent_tail_is_pad

        if video_is_pad.shape[1] != video_loss_token.shape[1]:
            raise ValueError(
                "Video-loss mask shape mismatch: "
                f"mask steps={video_is_pad.shape[1]}, loss steps={video_loss_token.shape[1]}."
            )

        valid = (~video_is_pad).to(device=video_loss_token.device, dtype=video_loss_token.dtype)
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return (video_loss_token * valid).sum(dim=1) / valid_sum

    @staticmethod
    def _compute_action_loss_per_sample(
        pred_action: torch.Tensor,
        target_action: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if pred_action.shape != target_action.shape:
            raise ValueError(
                "Predicted and target action shapes must match, got "
                f"{tuple(pred_action.shape)} and {tuple(target_action.shape)}."
            )
        action_loss_token = F.mse_loss(
            pred_action.float(), target_action.float(), reduction="none"
        ).mean(dim=2)
        if action_is_pad is None:
            return action_loss_token.mean(dim=1)
        if action_is_pad.shape != action_loss_token.shape:
            raise ValueError(
                "Action padding mask shape mismatch: "
                f"mask={tuple(action_is_pad.shape)}, "
                f"loss={tuple(action_loss_token.shape)}."
            )
        valid = (~action_is_pad).to(
            device=action_loss_token.device, dtype=action_loss_token.dtype
        )
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return (action_loss_token * valid).sum(dim=1) / valid_sum

    def _forward_prior_action_train(
        self,
        *,
        first_frame_latents: torch.Tensor,
        noisy_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        state_only_context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
    ) -> torch.Tensor:
        """Vision/state-only prior with an independently prefetched video cache."""
        if not self.langforce_mvp_enabled:
            raise RuntimeError("Prior action forward requires LangForce MVP mode.")

        timestep_video_zero = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=first_frame_latents.device,
        )
        video_pre_prior = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video_zero,
            context=context,
            context_mask=state_only_context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre_prior = self._prepare_action_tokens(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            full_context_mask=state_only_context_mask,
            state_only_context_mask=state_only_context_mask,
            mode="prior",
        )

        video_seq_len = int(video_pre_prior["tokens"].shape[1])
        num_queries = int(action_pre_prior["meta"]["num_queries"])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=int(action_pre_prior["tokens"].shape[1]),
            video_tokens_per_frame=int(
                video_pre_prior["meta"]["tokens_per_frame"]
            ),
            device=video_pre_prior["tokens"].device,
            num_queries=num_queries,
            action_reads_raw_video=False,
        )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre_prior["tokens"],
            video_freqs=video_pre_prior["freqs"],
            video_t_mod=video_pre_prior["t_mod"],
            video_context_payload={
                "context": video_pre_prior["context"],
                "mask": video_pre_prior["context_mask"],
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )
        if self.detach_prior_video_cache:
            video_kv_cache = [
                {"k": layer["k"].detach(), "v": layer["v"].detach()}
                for layer in video_kv_cache
            ]

        action_tokens_prior = self.mot.forward_action_with_video_cache(
            action_tokens=action_pre_prior["tokens"],
            action_freqs=action_pre_prior["freqs"],
            action_t_mod=action_pre_prior["t_mod"],
            action_context_payload={
                "context": action_pre_prior["context"],
                "mask": action_pre_prior["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )
        return self.action_expert.post_dit(
            action_tokens_prior, action_pre_prior
        )

    def training_loss(self, sample, tiled: bool = False):
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        full_context_mask, state_only_context_mask = self._build_context_masks(
            full_context_mask=context_mask,
            language_context_len=inputs["language_context_len"],
            has_proprio=inputs["has_proprio"],
        )
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]

        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)

        if inputs["first_frame_latents"] is not None:
            latents[:, :, 0:1] = inputs["first_frame_latents"]

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        video_pre = self.video_expert.pre_dit(
            x=latents,
            timestep=timestep_video,
            context=context,
            context_mask=full_context_mask,
            action=action,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )

        z_language = None
        router_metrics: dict[str, torch.Tensor] = {}
        if self.transition_contract_enabled:
            (
                video_kv_cache,
                final_video_hidden,
            ) = self._run_video_expert_to_final_hidden(video_pre)
            pred_action_post, z_language, router_metrics = (
                self._forward_tc_v2_action_from_video_hidden(
                    video_pre=video_pre,
                    final_video_hidden=final_video_hidden,
                    video_kv_cache=video_kv_cache,
                    action_tokens=noisy_action,
                    timestep_action=timestep_action,
                    context=context,
                    full_context_mask=full_context_mask,
                    state_only_context_mask=state_only_context_mask,
                )
            )
            pred_video = self.video_expert.post_dit(
                final_video_hidden, video_pre
            )
        else:
            action_pre = self._prepare_action_tokens(
                action_tokens=noisy_action,
                timestep=timestep_action,
                context=context,
                full_context_mask=full_context_mask,
                state_only_context_mask=state_only_context_mask,
                mode="posterior",
            )
            video_tokens = video_pre["tokens"]
            action_tokens = action_pre["tokens"]
            attention_mask = self._build_mot_attention_mask(
                video_seq_len=video_tokens.shape[1],
                action_seq_len=action_tokens.shape[1],
                video_tokens_per_frame=int(
                    video_pre["meta"]["tokens_per_frame"]
                ),
                device=video_tokens.device,
                num_queries=int(action_pre["meta"]["num_queries"]),
                action_reads_raw_video=self.action_reads_raw_video,
            )
            tokens_out = self.mot(
                embeds_all={"video": video_tokens, "action": action_tokens},
                attention_mask=attention_mask,
                freqs_all={
                    "video": video_pre["freqs"],
                    "action": action_pre["freqs"],
                },
                context_all={
                    "video": {
                        "context": video_pre["context"],
                        "mask": video_pre["context_mask"],
                    },
                    "action": {
                        "context": action_pre["context"],
                        "mask": action_pre["context_mask"],
                    },
                },
                t_mod_all={
                    "video": video_pre["t_mod"],
                    "action": action_pre["t_mod"],
                },
            )
            pred_video = self.video_expert.post_dit(
                tokens_out["video"], video_pre
            )
            pred_action_post = self.action_expert.post_dit(
                tokens_out["action"], action_pre
            )

        include_initial_video_step = inputs["first_frame_latents"] is None
        if inputs["first_frame_latents"] is not None:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]

        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

        e_post = self._compute_action_loss_per_sample(
            pred_action=pred_action_post,
            target_action=target_action,
            action_is_pad=action_is_pad,
        )

        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            e_post.device, dtype=e_post.dtype
        )
        loss_action_post = (e_post * action_weight).mean()

        loss_total = (
            self.loss_lambda_video * loss_video
            + self.loss_lambda_action * loss_action_post
        )
        loss_dict = {
            "loss_video": self.loss_lambda_video
            * float(loss_video.detach().item()),
            # Preserve the upstream metric name for existing dashboards.
            "loss_action": self.loss_lambda_action
            * float(loss_action_post.detach().item()),
            "loss_action_post": float(loss_action_post.detach().item()),
        }

        if self.transition_contract_enabled:
            if z_language is None:
                raise RuntimeError("TC-C failed to produce z_language.")
            z_future = self.encode_realized_transition(
                clean_input_latents=input_latents,
                context=context,
                full_context_mask=full_context_mask,
                action=action,
                fuse_vae_embedding_in_latents=inputs[
                    "fuse_vae_embedding_in_latents"
                ],
            )
            loss_contract, contract_metrics = (
                self.compute_transition_contract_loss(z_language, z_future)
            )
            contract_scale = self._transition_contract_scale()
            effective_contract_weight = (
                self.transition_contract_weight * contract_scale
            )
            loss_total = loss_total + effective_contract_weight * loss_contract
            loss_dict.update(
                {
                    "loss_transition_contract": float(
                        loss_contract.detach().item()
                    ),
                    "loss_language_future_contract": float(
                        loss_contract.detach().item()
                    ),
                    "transition_contract_scale": float(contract_scale),
                    "transition_contract_effective_weight": float(
                        effective_contract_weight
                    ),
                    "transition_embedding_norm": float(
                        z_language.detach().float().norm(dim=-1).mean().item()
                    ),
                }
            )
            loss_dict.update(detached_metrics(contract_metrics))
            loss_dict.update(detached_metrics(router_metrics))

        if self.langforce_prior_enabled:
            pred_action_prior = self._forward_prior_action_train(
                first_frame_latents=input_latents[:, :, 0:1],
                noisy_action=noisy_action,
                timestep_action=timestep_action,
                context=context,
                state_only_context_mask=state_only_context_mask,
                fuse_vae_embedding_in_latents=inputs[
                    "fuse_vae_embedding_in_latents"
                ],
            )
            e_prior = self._compute_action_loss_per_sample(
                pred_action=pred_action_prior,
                target_action=target_action,
                action_is_pad=action_is_pad,
            )
            loss_action_prior = (e_prior * action_weight).mean()
            loss_total = (
                loss_total + self.lambda_prior_action * loss_action_prior
            )
            loss_dict["loss_action_prior"] = float(
                loss_action_prior.detach().item()
            )

            if self.langforce_advantage_enabled:
                target_prior = (
                    1.0 - self.posterior_advantage_margin_ratio
                ) * e_prior.detach()
                loss_advantage = torch.relu(e_post - target_prior).mean()
                loss_total = (
                    loss_total
                    + self.lambda_posterior_advantage * loss_advantage
                )
            else:
                loss_advantage = e_post.new_zeros(())
            loss_dict["loss_posterior_advantage"] = float(
                loss_advantage.detach().item()
            )
            loss_dict["post_vs_prior_loss_ratio"] = float(
                (e_post.mean() / (e_prior.mean() + 1e-8)).detach().item()
            )
            loss_dict["fraction_post_better_than_prior"] = float(
                (e_post < e_prior).float().mean().detach().item()
            )

        return loss_total, loss_dict

    @torch.no_grad()
    def _predict_joint_noise(
        self,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        state_only_context_mask: Optional[torch.Tensor],
        fuse_vae_embedding_in_latents: bool,
        gt_action: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_pre = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=gt_action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        if state_only_context_mask is None:
            state_only_context_mask = context_mask
        if self.transition_contract_enabled:
            video_kv_cache, final_video_hidden = (
                self._run_video_expert_to_final_hidden(video_pre)
            )
            pred_action, _, _ = self._forward_tc_v2_action_from_video_hidden(
                video_pre=video_pre,
                final_video_hidden=final_video_hidden,
                video_kv_cache=video_kv_cache,
                action_tokens=latents_action,
                timestep_action=timestep_action,
                context=context,
                full_context_mask=context_mask,
                state_only_context_mask=state_only_context_mask,
            )
            pred_video = self.video_expert.post_dit(
                final_video_hidden, video_pre
            )
            return pred_video, pred_action

        action_pre = self._prepare_action_tokens(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            full_context_mask=context_mask,
            state_only_context_mask=state_only_context_mask,
            mode="posterior",
        )
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
            num_queries=int(action_pre["meta"]["num_queries"]),
            action_reads_raw_video=self.action_reads_raw_video,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )
        return (
            self.video_expert.post_dit(tokens_out["video"], video_pre),
            self.action_expert.post_dit(tokens_out["action"], action_pre),
        )

    @torch.no_grad()
    def _predict_action_noise(
        self,
        first_frame_latents: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        state_only_context_mask: Optional[torch.Tensor],
        fuse_vae_embedding_in_latents: bool,
    ) -> torch.Tensor:
        timestep_video = torch.zeros_like(timestep_action, dtype=first_frame_latents.dtype, device=self.device)
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        if state_only_context_mask is None:
            state_only_context_mask = context_mask
        if self.transition_contract_enabled:
            video_kv_cache, final_video_hidden = (
                self._run_video_expert_to_final_hidden(video_pre)
            )
            pred_action, _, _ = self._forward_tc_v2_action_from_video_hidden(
                video_pre=video_pre,
                final_video_hidden=final_video_hidden,
                video_kv_cache=video_kv_cache,
                action_tokens=latents_action,
                timestep_action=timestep_action,
                context=context,
                full_context_mask=context_mask,
                state_only_context_mask=state_only_context_mask,
            )
            return pred_action
        action_pre = self._prepare_action_tokens(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            full_context_mask=context_mask,
            state_only_context_mask=state_only_context_mask,
            mode="posterior",
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
            num_queries=int(action_pre["meta"]["num_queries"]),
            action_reads_raw_video=self.action_reads_raw_video,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_action

    @torch.no_grad()
    def _predict_action_noise_with_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        state_only_context_mask: Optional[torch.Tensor],
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
        routed_transition_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if state_only_context_mask is None:
            state_only_context_mask = context_mask
        action_pre = self._prepare_action_tokens(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            full_context_mask=context_mask,
            state_only_context_mask=state_only_context_mask,
            mode="posterior",
            transition_query_tokens=routed_transition_tokens,
        )
        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )
        return self.action_expert.post_dit(action_tokens, action_pre)

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None, # NOTE: this is gt action for conditioning videos, not for action expert
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        test_action_with_infer_action: bool = True,
        mask_language: bool = False,
    ) -> dict[str, Any]:
        self.eval()
        if test_action_with_infer_action:
            if seed is None:
                raise ValueError("`test_action_with_infer_action=True` requires non-null `seed`.")
            action_only_out = self.infer_action(
                prompt=prompt,
                input_image=input_image.clone(),
                action_horizon=action_horizon,
                context=context.clone() if context is not None else None,
                context_mask=context_mask.clone() if context_mask is not None else None,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
                proprio=proprio.clone() if proprio is not None else None,
                mask_language=mask_language,
            )["action"]
        
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(height, width, num_video_frames)
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if checked_t != num_video_frames:
            raise ValueError(
                f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}"
            )
        if action is not None:
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3 or action.shape[0] != 1 or action.shape[1] != action_horizon:
                # NOTE: This enforces action condition to have the same shape as action horizon to predict, which may be unnecessary
                raise ValueError(
                    f"`action` must have shape [1, T, a_dim] or [T, a_dim], got {tuple(action.shape)} with action_horizon={action_horizon}"
                )
            action = action.to(device=self.device, dtype=self.torch_dtype)
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor

        video_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        latents_video[:, :, 0:1] = first_frame_latents.clone()
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        language_context_len = int(context.shape[1])
        has_proprio = False
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )
            has_proprio = True
        full_context_mask, state_only_context_mask = self._build_context_masks(
            full_context_mask=context_mask,
            language_context_len=language_context_len,
            has_proprio=has_proprio,
        )
        if mask_language:
            full_context_mask[:, :language_context_len] = False
        context_mask = full_context_mask

        infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_video, step_delta_video, step_t_action, step_delta_action in zip(
            infer_timesteps_video,
            infer_deltas_video,
            infer_timesteps_action,
            infer_deltas_action,
        ):
            timestep_video = step_t_video.unsqueeze(0).to(dtype=latents_video.dtype, device=self.device)
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_video_posi, pred_action_posi = self._predict_joint_noise(
                latents_video=latents_video,
                latents_action=latents_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                state_only_context_mask=state_only_context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                gt_action=action,
            )
            pred_video = pred_video_posi
            pred_action = pred_action_posi

            latents_video = self.infer_video_scheduler.step(pred_video, step_delta_video, latents_video)
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
            latents_video[:, :, 0:1] = first_frame_latents.clone()

        action_out = latents_action[0].detach().to(device="cpu", dtype=torch.float32)
        if test_action_with_infer_action:
            if not torch.allclose(action_out, action_only_out, atol=1e-2, rtol=1e-2):
                max_abs_diff = (action_out - action_only_out).abs().max().item()
                logger.warning(
                    f"Action from infer_joint and infer_action differ with max abs diff {max_abs_diff:.6f}. "
                )

        return {
            "video": self._decode_latents(latents_video, tiled=tiled),
            "action": action_out,
        }

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        mask_language: bool = False,
    ) -> dict[str, Any]:
        self.eval()
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError(
                "`infer_action` requires `video_attention_mask_mode='first_frame_causal'`."
            )

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        language_context_len = int(context.shape[1])
        has_proprio = False
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )
            has_proprio = True
        full_context_mask, state_only_context_mask = self._build_context_masks(
            full_context_mask=context_mask,
            language_context_len=language_context_len,
            has_proprio=has_proprio,
        )
        if mask_language:
            full_context_mask[:, :language_context_len] = False
        context_mask = full_context_mask

        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        routed_transition_tokens = None
        if self.transition_contract_enabled:
            prefill_result = self.mot.prefill_video_cache(
                video_tokens=video_pre["tokens"],
                video_freqs=video_pre["freqs"],
                video_t_mod=video_pre["t_mod"],
                video_context_payload={
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                video_attention_mask=(
                    self.video_expert.build_video_to_video_mask(
                        video_seq_len=int(video_pre["tokens"].shape[1]),
                        video_tokens_per_frame=int(
                            video_pre["meta"]["tokens_per_frame"]
                        ),
                        device=video_pre["tokens"].device,
                    )
                ),
                return_final_hidden=True,
            )
            if not isinstance(prefill_result, tuple):
                raise RuntimeError("TC-C v2 Video prefill did not return hidden.")
            video_kv_cache, final_video_hidden = prefill_result
            routed_transition_tokens, _, _ = self.encode_intended_transition(
                video_tokens=final_video_hidden,
                video_tokens_per_frame=int(
                    video_pre["meta"]["tokens_per_frame"]
                ),
                context=context,
                full_context_mask=context_mask,
                route_scale=1.0,
            )
        video_seq_len = int(video_pre["tokens"].shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=(
                int(latents_action.shape[1])
                + (
                    int(self.action_expert.num_latent_queries)
                    if self.uses_transition_queries
                    else 0
                )
            ),
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
            num_queries=(
                int(self.action_expert.num_latent_queries)
                if self.uses_transition_queries
                else 0
            ),
            action_reads_raw_video=self.action_reads_raw_video,
            queries_read_raw_video=False,
        )
        if not self.transition_contract_enabled:
            video_kv_cache = self.mot.prefill_video_cache(
                video_tokens=video_pre["tokens"],
                video_freqs=video_pre["freqs"],
                video_t_mod=video_pre["t_mod"],
                video_context_payload={
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                video_attention_mask=attention_mask[
                    :video_seq_len, :video_seq_len
                ],
            )

        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_action_posi = self._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                state_only_context_mask=state_only_context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
                routed_transition_tokens=routed_transition_tokens,
            )
            pred_action = pred_action_posi

            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)

        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }

    @torch.no_grad()
    def infer(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_frames: int,
        action: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 5.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        mask_language: bool = False,
    ):
        return self.infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_frames,
            action_horizon=action_horizon,
            action=action,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
            mask_language=mask_language,
        )

    def _lora_adapter_state_dict(self) -> dict[str, torch.Tensor]:
        adapter_ids = self._adapter_parameter_ids()
        state = self.mot.state_dict()
        names = {
            name
            for name, parameter in self.mot.named_parameters()
            if id(parameter) in adapter_ids
        }
        return {
            name: state[name].detach().to(device="cpu")
            for name in sorted(names)
        }

    def _transition_contract_metadata(self) -> dict[str, Any]:
        return {
            "architecture": "tc_fastwam",
            "transition_contract_version": self.transition_contract_version,
            "num_transition_queries": int(
                getattr(self.action_expert, "num_latent_queries", 0)
            ),
            "use_router": bool(self.use_transition_router),
            "use_contract": bool(self.transition_contract_enabled),
            "use_cf_ranking": False,
            "use_action_effect": False,
            "action_conditioned_video": False,
            "router_visual_source": "video_expert_final_hidden",
            "policy_recovery_ratio": self.transition_policy_recovery_ratio,
            "router_ramp_ratio": self.transition_router_ramp_ratio,
            "policy_recovery_blend": "action_flow_velocity",
            "freeze_m1_during_recovery": (
                self.transition_freeze_m1_during_recovery
            ),
            "policy_init_checkpoint": self.transition_policy_init_checkpoint,
        }

    def _transition_contract_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            name: value.detach().to(device="cpu")
            for name, value in self.transition_contract_modules.state_dict().items()
        }

    def save_checkpoint(self, path, optimizer=None, step=None):
        if self.lora_enabled:
            payload = {
                "format": "fastwam_lora_adapter_v1",
                "mot_trainable": self._lora_adapter_state_dict(),
                "lora_config": dict(self.lora_config),
                "base_checkpoint": self.lora_base_checkpoint,
                "step": step,
                "torch_dtype": str(self.torch_dtype),
            }
            if self.transition_contract_enabled:
                payload["transition_contract"] = (
                    self._transition_contract_state_dict()
                )
                payload["architecture_metadata"] = (
                    self._transition_contract_metadata()
                )
            adapter_ids = self._adapter_parameter_ids()
            if self.proprio_encoder is not None and any(
                id(parameter) in adapter_ids
                for parameter in self.proprio_encoder.parameters()
            ):
                payload["proprio_encoder"] = {
                    name: value.detach().to(device="cpu")
                    for name, value in self.proprio_encoder.state_dict().items()
                }
            if optimizer is not None:
                payload["optimizer"] = optimizer.state_dict()
            torch.save(payload, path)
            return

        payload = {
            "format": "fastwam_full_v1",
            "mot": self.mot.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if self.transition_contract_enabled:
            payload["transition_contract"] = self._transition_contract_state_dict()
            payload["architecture_metadata"] = self._transition_contract_metadata()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    @staticmethod
    def _resolve_adapter_base_checkpoint(
        adapter_path: str, base_checkpoint: str
    ) -> str:
        candidate = Path(base_checkpoint).expanduser()
        if not candidate.is_absolute():
            candidate = Path(adapter_path).resolve().parent / candidate
        candidate = candidate.resolve()
        if not candidate.is_file():
            raise FileNotFoundError(
                "LoRA adapter base checkpoint was not found: "
                f"{candidate} (adapter={adapter_path})"
            )
        return str(candidate)

    def _load_lora_adapter(self, path: str, payload: dict, optimizer=None):
        adapter_source_path = str(Path(path).expanduser().resolve())
        transition_state = payload.get("transition_contract")
        if transition_state is not None:
            if not self.transition_contract_enabled:
                raise ValueError(
                    "This adapter contains TC-FastWAM weights, but the current "
                    "model has `transition_contract.enabled=false`. Enable the "
                    "matching Stage-1 config before loading it."
                )
            metadata = payload.get("architecture_metadata") or {}
            saved_version = metadata.get("transition_contract_version")
            if saved_version != self.transition_contract_version:
                raise ValueError(
                    "TC checkpoint version mismatch: "
                    f"checkpoint={saved_version}, model={self.transition_contract_version}."
                )

        base_checkpoint = payload.get("base_checkpoint")
        if base_checkpoint:
            resolved_base = self._resolve_adapter_base_checkpoint(
                path, str(base_checkpoint)
            )
            self.load_checkpoint(resolved_base, optimizer=None)

        saved_lora_config = payload.get("lora_config")
        if not isinstance(saved_lora_config, dict):
            raise ValueError(f"LoRA adapter missing `lora_config`: {path}")
        self.configure_lora(saved_lora_config)

        adapter_state = payload.get("mot_trainable")
        if not isinstance(adapter_state, dict) or not adapter_state:
            raise ValueError(f"LoRA adapter missing non-empty `mot_trainable`: {path}")
        current_state = self.mot.state_dict()
        unexpected = sorted(set(adapter_state) - set(current_state))
        if unexpected:
            raise ValueError(
                f"LoRA adapter contains unknown MoT keys: {unexpected[:20]}"
            )
        shape_mismatches = {
            name: (tuple(value.shape), tuple(current_state[name].shape))
            for name, value in adapter_state.items()
            if tuple(value.shape) != tuple(current_state[name].shape)
        }
        if shape_mismatches:
            raise ValueError(
                f"LoRA adapter shape mismatches: {shape_mismatches}"
            )
        self.mot.load_state_dict(adapter_state, strict=False)

        if transition_state is not None:
            self.transition_contract_modules.load_state_dict(
                transition_state, strict=True
            )
        elif self.transition_contract_enabled:
            logger.info(
                "Adapter has no Transition Contract tensors; keeping initialized "
                "TC-C modules for backward-compatible B0/M1 loading."
            )
        if self.transition_contract_enabled and transition_state is None:
            # The loaded M1 adapter is the policy initialization, but the new
            # TC adapter must remain self-contained with respect to the
            # official FastWAM base rather than recursively depending on M1.
            self.transition_policy_init_checkpoint = adapter_source_path
        else:
            self.transition_policy_init_checkpoint = None

        if self.proprio_encoder is not None and "proprio_encoder" in payload:
            self.proprio_encoder.load_state_dict(
                payload["proprio_encoder"], strict=True
            )
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        logger.info(
            "Loaded LoRA adapter from %s (trainable_tensors=%d base=%s).",
            path,
            len(adapter_state),
            self.lora_base_checkpoint,
        )
        return payload

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu")
        if payload.get("format") == "fastwam_lora_adapter_v1":
            return self._load_lora_adapter(str(path), payload, optimizer=optimizer)

        if "mot" in payload:
            incompatible = self.mot.load_state_dict(payload["mot"], strict=False)
            missing_lora = [
                key for key in incompatible.missing_keys if is_lora_parameter_name(key)
            ]
            missing_other = [
                key for key in incompatible.missing_keys if not is_lora_parameter_name(key)
            ]
            if missing_lora:
                logger.info(
                    "Base checkpoint has no LoRA tensors; keeping zero-init adapters "
                    "(missing=%d).",
                    len(missing_lora),
                )
            if missing_other or incompatible.unexpected_keys:
                logger.warning(
                    "Loaded MoT checkpoint with strict=False. missing=%s unexpected=%s",
                    missing_other,
                    incompatible.unexpected_keys,
                )
        elif "dit" in payload:
            logger.warning("Loading legacy `dit` checkpoint into video expert only.")
            self.video_expert.load_state_dict(payload["dit"], strict=False)
        else:
            raise ValueError(f"Checkpoint missing both `mot` and `dit` keys: {path}")
        if self.proprio_encoder is not None:
            if "proprio_encoder" in payload:
                self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
            else:
                logger.warning("Checkpoint has no `proprio_encoder` weights; keeping current `proprio_encoder` params.")
        elif "proprio_encoder" in payload:
            logger.warning("Checkpoint contains `proprio_encoder` weights but current model has `proprio_dim=None`; ignoring.")

        transition_state = payload.get("transition_contract")
        if transition_state is not None:
            if not self.transition_contract_enabled:
                raise ValueError(
                    "Checkpoint contains TC-FastWAM weights; enable the matching "
                    "`transition_contract` config before loading."
                )
            metadata = payload.get("architecture_metadata") or {}
            saved_version = metadata.get("transition_contract_version")
            if saved_version != self.transition_contract_version:
                raise ValueError(
                    "TC checkpoint version mismatch: "
                    f"checkpoint={saved_version}, model={self.transition_contract_version}."
                )
            self.transition_contract_modules.load_state_dict(
                transition_state, strict=True
            )
        elif self.transition_contract_enabled:
            logger.info(
                "Checkpoint has no Transition Contract tensors; keeping standard "
                "initialization for backward-compatible B0/M1 loading."
            )

        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        self.lora_base_checkpoint = str(Path(path).expanduser().resolve())
        return payload

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)
