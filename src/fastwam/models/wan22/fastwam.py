import copy
import math
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
from .policy_guard import (
    ActionOutcomeVerifier,
    GoalActionAlignmentLoss,
    GoalGraphEncoder,
    GoalResidualAdapter,
    detached_policy_guard_metrics,
)
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from .transition_contract import (
    ActionEffectEncoder,
    ContrastiveContractLoss,
    CounterfactualActionPrototypeBank,
    CounterfactualRankingLoss,
    OutcomeTransitionEncoder,
    StateConditionedTargetGrounder,
    StateTargetPrototypeBank,
    TransitionProjectionHead,
    TransitionVisualRouter,
    detached_metrics,
    interaction_patch_distribution,
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
        policy_guard_config: Optional[dict[str, Any]] = None,
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
        self.transition_action_future_weight = float(
            contract_config.get("action_future_weight", 1.0)
        )
        self.transition_counterfactual_weight = float(
            contract_config.get("counterfactual_weight", 0.05)
        )
        self.transition_counterfactual_margin = float(
            contract_config.get("counterfactual_margin", 0.2)
        )
        self.transition_counterfactual_action_positive_weight = float(
            contract_config.get("counterfactual_action_positive_weight", 0.10)
        )
        self.transition_counterfactual_action_query_weight = float(
            contract_config.get("counterfactual_action_query_weight", 1.0)
        )
        self.transition_counterfactual_action_effect_weight = float(
            contract_config.get("counterfactual_action_effect_weight", 1.0)
        )
        self.transition_counterfactual_action_separation_weight = float(
            contract_config.get("counterfactual_action_separation_weight", 0.05)
        )
        self.transition_counterfactual_action_separation_margin = float(
            contract_config.get("counterfactual_action_separation_margin", 0.05)
        )
        self.transition_state_grounding_weight = float(
            contract_config.get("state_grounding_weight", 0.10)
        )
        self.transition_state_grounding_correct_weight = float(
            contract_config.get("state_grounding_correct_weight", 1.0)
        )
        self.transition_state_grounding_counterfactual_weight = float(
            contract_config.get("state_grounding_counterfactual_weight", 1.0)
        )
        self.transition_state_grounding_separation_weight = float(
            contract_config.get("state_grounding_separation_weight", 0.25)
        )
        self.transition_state_grounding_overlap_margin = float(
            contract_config.get("state_grounding_overlap_margin", 0.25)
        )
        self.transition_state_grounding_router_bias = float(
            contract_config.get("state_grounding_router_bias", 2.0)
        )
        self.transition_state_grounding_teacher_topk = float(
            contract_config.get("state_grounding_teacher_topk", 0.15)
        )
        self.transition_state_grounding_teacher_temperature = float(
            contract_config.get("state_grounding_teacher_temperature", 0.25)
        )
        self.transition_state_grounding_hidden_dim = int(
            contract_config.get("state_grounding_hidden_dim", 256)
        )
        self.transition_state_grounding_temperature = float(
            contract_config.get("state_grounding_temperature", 0.07)
        )
        self.transition_state_grounding_prototype_slots = int(
            contract_config.get("state_grounding_prototype_slots", 64)
        )
        self.transition_state_grounding_prototype_momentum = float(
            contract_config.get("state_grounding_prototype_momentum", 0.95)
        )
        self.transition_state_grounding_prototype_temperature = float(
            contract_config.get("state_grounding_prototype_temperature", 0.07)
        )
        self.transition_state_grounding_prototype_topk = float(
            contract_config.get("state_grounding_prototype_topk", 0.10)
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
        self.transition_policy_distillation_enabled = bool(
            contract_config.get("policy_distillation_enabled", False)
        )
        self.transition_policy_distillation_weight = float(
            contract_config.get("policy_distillation_weight", 1.0)
        )
        self.transition_freeze_m1_policy = bool(
            contract_config.get("freeze_m1_policy", False)
        )
        self.outcome_stop_gradient = bool(
            contract_config.get("outcome_stop_gradient", True)
        )
        self.use_transition_router = bool(
            contract_config.get("use_transition_router", True)
        )
        self.transition_use_action_effect = bool(
            contract_config.get("use_action_effect", False)
        )
        self.transition_use_counterfactual_ranking = bool(
            contract_config.get("use_counterfactual_ranking", False)
        )
        self.transition_use_counterfactual_action_positive = bool(
            contract_config.get("use_counterfactual_action_positive", False)
        )
        self.transition_use_state_conditioned_grounding = bool(
            contract_config.get("use_state_conditioned_grounding", False)
        )
        self.transition_contract_modules = nn.ModuleDict()
        self.transition_contract_loss = None
        self.transition_counterfactual_loss = None
        self._transition_training_step = 0
        self._transition_training_max_steps = 1
        self._transition_training_progress_active = False
        self.transition_policy_init_checkpoint: Optional[str] = None

        if self.transition_contract_enabled:
            if self.transition_contract_version not in {2, 3, 4, 5, 6}:
                raise ValueError(
                    "The current TC implementation supports "
                    "`transition_contract.version` 2, 3, 4, 5, or 6."
                )
            if not bool(
                getattr(self.action_expert, "use_latent_action_queries", False)
            ):
                raise ValueError(
                    "Transition Contract requires ActionDiT "
                    "`use_latent_action_queries=True`."
                )
            if not self.use_transition_router:
                raise ValueError("TC-FastWAM requires `use_transition_router=true`.")
            if bool(contract_config.get("direct_action_video_access", False)):
                raise ValueError(
                    "TC-FastWAM forbids `direct_action_video_access=true`; the Action "
                    "Expert must consume routed transition tokens only."
                )
            if bool(contract_config.get("direct_action_text_access", False)):
                raise ValueError(
                    "TC-FastWAM forbids `direct_action_text_access=true`; language must "
                    "reach actions through the Transition Router."
                )
            if not bool(contract_config.get("direct_video_text_access", True)):
                raise ValueError(
                    "TC-C/TC-Full use Phase T1 and require "
                    "`direct_video_text_access=true`."
                )
            if bool(contract_config.get("action_conditioned_video", False)):
                raise ValueError(
                    "`action_conditioned_video` belongs to Stage 3 / TC-Dyn."
                )
            if self.transition_contract_version < 4 and (
                self.transition_use_action_effect
                or self.transition_use_counterfactual_ranking
            ):
                raise ValueError(
                    "Action-effect and counterfactual ranking require "
                    "TC-Full `transition_contract.version>=4`."
                )
            if self.transition_contract_version < 5 and (
                self.transition_use_counterfactual_action_positive
            ):
                raise ValueError(
                    "Counterfactual action positive supervision requires "
                    "`transition_contract.version>=5`."
                )
            if self.transition_contract_version < 6 and (
                self.transition_use_state_conditioned_grounding
            ):
                raise ValueError(
                    "State-conditioned target grounding requires "
                    "`transition_contract.version=6`."
                )
            if self.langforce_prior_enabled or self.langforce_advantage_enabled:
                raise ValueError(
                    "TC mainline requires LangForce prior/advantage ablations off."
                )
            if self.transition_contract_weight < 0:
                raise ValueError("`contract_weight` must be non-negative.")
            if self.transition_action_future_weight < 0:
                raise ValueError("`action_future_weight` must be non-negative.")
            if self.transition_counterfactual_weight < 0:
                raise ValueError("`counterfactual_weight` must be non-negative.")
            if self.transition_counterfactual_margin < 0:
                raise ValueError("`counterfactual_margin` must be non-negative.")
            if self.transition_counterfactual_action_positive_weight < 0:
                raise ValueError(
                    "`counterfactual_action_positive_weight` must be non-negative."
                )
            if self.transition_counterfactual_action_query_weight < 0:
                raise ValueError(
                    "`counterfactual_action_query_weight` must be non-negative."
                )
            if self.transition_counterfactual_action_effect_weight < 0:
                raise ValueError(
                    "`counterfactual_action_effect_weight` must be non-negative."
                )
            if self.transition_counterfactual_action_separation_weight < 0:
                raise ValueError(
                    "`counterfactual_action_separation_weight` must be non-negative."
                )
            if self.transition_counterfactual_action_separation_margin < 0:
                raise ValueError(
                    "`counterfactual_action_separation_margin` must be non-negative."
                )
            if min(
                self.transition_state_grounding_weight,
                self.transition_state_grounding_correct_weight,
                self.transition_state_grounding_counterfactual_weight,
                self.transition_state_grounding_separation_weight,
                self.transition_state_grounding_router_bias,
            ) < 0:
                raise ValueError("State-grounding weights must be non-negative.")
            if not 0.0 <= self.transition_state_grounding_overlap_margin <= 1.0:
                raise ValueError(
                    "`state_grounding_overlap_margin` must be in [0,1]."
                )
            if not 0.0 < self.transition_state_grounding_teacher_topk <= 1.0:
                raise ValueError(
                    "`state_grounding_teacher_topk` must be in (0,1]."
                )
            if self.transition_state_grounding_teacher_temperature <= 0:
                raise ValueError(
                    "`state_grounding_teacher_temperature` must be positive."
                )
            if self.transition_state_grounding_hidden_dim <= 0:
                raise ValueError("`state_grounding_hidden_dim` must be positive.")
            if self.transition_state_grounding_temperature <= 0:
                raise ValueError("`state_grounding_temperature` must be positive.")
            if self.transition_state_grounding_prototype_slots <= 0:
                raise ValueError(
                    "`state_grounding_prototype_slots` must be positive."
                )
            if not 0.0 <= self.transition_state_grounding_prototype_momentum < 1.0:
                raise ValueError(
                    "`state_grounding_prototype_momentum` must be in [0,1)."
                )
            if self.transition_state_grounding_prototype_temperature <= 0:
                raise ValueError(
                    "`state_grounding_prototype_temperature` must be positive."
                )
            if not 0.0 < self.transition_state_grounding_prototype_topk <= 1.0:
                raise ValueError(
                    "`state_grounding_prototype_topk` must be in (0,1]."
                )
            if self.transition_policy_distillation_weight < 0:
                raise ValueError(
                    "`policy_distillation_weight` must be non-negative."
                )
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
            if self.transition_contract_version == 2 and (
                self.transition_policy_distillation_enabled
                or self.transition_freeze_m1_policy
            ):
                raise ValueError(
                    "M1 policy distillation/protection requires "
                    "`transition_contract.version>=3`."
                )
            if self.transition_contract_version >= 3:
                if not self.transition_policy_distillation_enabled:
                    raise ValueError(
                        "Protected TC v3+ requires "
                        "`policy_distillation_enabled=true`."
                    )
                if self.transition_policy_distillation_weight <= 0:
                    raise ValueError(
                        "Protected TC v3+ requires a positive "
                        "`policy_distillation_weight`."
                    )
                if not self.transition_freeze_m1_policy:
                    raise ValueError(
                        "Protected TC v3+ requires `freeze_m1_policy=true` so the "
                        "joint-MoT teacher cannot drift."
                    )
            if self.transition_contract_version >= 4:
                if not self.transition_use_action_effect:
                    raise ValueError("TC-Full v4+ requires `use_action_effect=true`.")
                if not self.transition_use_counterfactual_ranking:
                    raise ValueError(
                        "TC-Full v4+ requires `use_counterfactual_ranking=true`."
                    )
            if self.transition_contract_version >= 5 and (
                not self.transition_use_counterfactual_action_positive
            ):
                raise ValueError(
                    "TC-Full v5+ requires "
                    "`use_counterfactual_action_positive=true`."
                )
            if self.transition_contract_version >= 6 and (
                not self.transition_use_state_conditioned_grounding
            ):
                raise ValueError(
                    "TC-Full v6 requires "
                    "`use_state_conditioned_grounding=true`."
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
            if self.transition_use_action_effect:
                self.transition_contract_modules["action_effect_encoder"] = (
                    ActionEffectEncoder(
                        action_dim=int(self.action_expert.action_dim),
                        video_dim=video_hidden_dim,
                        projection_dim=projection_dim,
                        proprio_dim=self.proprio_dim,
                        hidden_dim=int(
                            contract_config.get(
                                "action_effect_hidden_dim", projection_dim
                            )
                        ),
                        num_heads=int(
                            contract_config.get("action_effect_num_heads", 8)
                        ),
                        num_layers=int(
                            contract_config.get("action_effect_num_layers", 2)
                        ),
                    )
                )
            if self.transition_use_counterfactual_action_positive:
                self.transition_contract_modules[
                    "counterfactual_action_prototypes"
                ] = CounterfactualActionPrototypeBank(
                    num_slots=int(
                        contract_config.get(
                            "counterfactual_action_prototype_slots", 64
                        )
                    ),
                    num_queries=int(self.action_expert.num_latent_queries),
                    query_dim=action_hidden_dim,
                    action_effect_dim=projection_dim,
                    momentum=float(
                        contract_config.get(
                            "counterfactual_action_prototype_momentum", 0.95
                        )
                    ),
                )
            if self.transition_use_state_conditioned_grounding:
                grounding_dim = self.transition_state_grounding_hidden_dim
                self.transition_contract_modules["state_target_grounder"] = (
                    StateConditionedTargetGrounder(
                        language_dim=action_hidden_dim,
                        video_dim=video_hidden_dim,
                        hidden_dim=grounding_dim,
                        temperature=self.transition_state_grounding_temperature,
                    )
                )
                self.transition_contract_modules["state_target_prototypes"] = (
                    StateTargetPrototypeBank(
                        num_slots=self.transition_state_grounding_prototype_slots,
                        feature_dim=grounding_dim,
                        momentum=self.transition_state_grounding_prototype_momentum,
                        temperature=(
                            self.transition_state_grounding_prototype_temperature
                        ),
                        topk_fraction=(
                            self.transition_state_grounding_prototype_topk
                        ),
                    )
                )
            self.transition_contract_modules.to(dtype=self.torch_dtype)
            self.transition_contract_loss = ContrastiveContractLoss(
                temperature=self.transition_contract_temperature
            )
            if self.transition_use_counterfactual_ranking:
                self.transition_counterfactual_loss = CounterfactualRankingLoss(
                    margin=self.transition_counterfactual_margin
                )

        guard_config = dict(policy_guard_config or {})
        self.policy_guard_enabled = bool(guard_config.get("enabled", False))
        self.policy_guard_version = int(guard_config.get("version", 2))
        self.policy_guard_action_weight = float(
            guard_config.get("counterfactual_action_weight", 1.0)
        )
        self.policy_guard_native_distillation_weight = float(
            guard_config.get("native_distillation_weight", 1.0)
        )
        self.policy_guard_goal_residual_scale = float(
            guard_config.get("goal_residual_scale", 1.0)
        )
        self.policy_guard_verifier_weight = float(
            guard_config.get("verifier_weight", 0.25)
        )
        self.policy_guard_alignment_weight = float(
            guard_config.get("goal_action_alignment_weight", 0.10)
        )
        self.policy_guard_verifier_margin = float(
            guard_config.get("verifier_margin", 0.20)
        )
        self.policy_guard_verifier_action_mse_temperature = float(
            guard_config.get("verifier_action_mse_temperature", 0.25)
        )
        self.policy_guard_gate_threshold = float(
            guard_config.get("gate_threshold", 0.20)
        )
        self.policy_guard_min_counterfactual_score = float(
            guard_config.get("min_counterfactual_score", 0.60)
        )
        self.policy_guard_gate_mode = str(
            guard_config.get("gate_mode", "guarded")
        ).strip().lower()
        self.policy_guard_require_direct_counterfactual_actions = bool(
            guard_config.get("require_direct_counterfactual_actions", True)
        )
        self.policy_guard_modules = nn.ModuleDict()
        self.policy_guard_action_expert: Optional[ActionDiT] = None
        self.policy_guard_alignment_loss: Optional[GoalActionAlignmentLoss] = None
        self.policy_guard_base_checkpoint: Optional[str] = None
        self.policy_guard_legacy_full_loaded = False

        if self.policy_guard_enabled:
            if self.policy_guard_version not in {1, 2}:
                raise ValueError(
                    "The current PGC implementation supports version=1 or 2."
                )
            if self.langforce_mvp_enabled or self.transition_contract_enabled:
                raise ValueError(
                    "PGC is an independent policy-protection mainline; disable "
                    "LangForce MVP and Transition Contract when it is enabled."
                )
            if bool(
                getattr(self.action_expert, "use_latent_action_queries", False)
            ):
                raise ValueError(
                    "PGC requires the protected base Action Expert to use the "
                    "released query-free interface "
                    "(`action_dit_config.use_latent_action_queries=false`)."
                )
            if min(
                self.policy_guard_action_weight,
                self.policy_guard_native_distillation_weight,
                self.policy_guard_goal_residual_scale,
                self.policy_guard_verifier_weight,
                self.policy_guard_alignment_weight,
                self.policy_guard_verifier_margin,
                self.policy_guard_verifier_action_mse_temperature,
                self.policy_guard_gate_threshold,
                self.policy_guard_min_counterfactual_score,
            ) < 0:
                raise ValueError("PGC weights, margins, and thresholds must be non-negative.")
            if self.policy_guard_min_counterfactual_score > 1:
                raise ValueError("`min_counterfactual_score` must be <= 1.")
            if self.policy_guard_verifier_action_mse_temperature <= 0:
                raise ValueError(
                    "`verifier_action_mse_temperature` must be positive."
                )
            if self.policy_guard_gate_mode not in {
                "guarded",
                "base",
                "counterfactual",
            }:
                raise ValueError(
                    "`policy_guard.gate_mode` must be guarded, base, or "
                    "counterfactual."
                )

            num_action_queries = int(
                guard_config.get("num_action_queries", 32)
            )
            query_rope_offset = int(
                guard_config.get("query_rope_offset", 512)
            )
            self.policy_guard_num_action_queries = num_action_queries
            self.policy_guard_query_rope_offset = query_rope_offset
            if num_action_queries <= 0 or query_rope_offset < 0:
                raise ValueError("PGC action-query count/offset are invalid.")
            if self.policy_guard_version == 1 and (
                query_rope_offset + num_action_queries
                > int(self.action_expert.freqs.shape[0])
            ):
                raise ValueError(
                    "PGC action queries exceed the ActionDiT RoPE cache."
                )

            # This is a physically independent Action Expert.  Only its common
            # backbone is initialized from the base; the base module itself is
            # never put in the optimizer or mutated by PGC training.
            counterfactual_action_expert = copy.deepcopy(self.action_expert)
            if self.policy_guard_version == 1:
                counterfactual_action_expert.use_latent_action_queries = True
                counterfactual_action_expert.num_latent_queries = num_action_queries
                counterfactual_action_expert.query_rope_offset = query_rope_offset
                reference_parameter = next(
                    counterfactual_action_expert.parameters()
                )
                counterfactual_action_expert.register_parameter(
                    "latent_action_queries",
                    nn.Parameter(
                        torch.randn(
                            1,
                            num_action_queries,
                            int(counterfactual_action_expert.hidden_dim),
                            device=reference_parameter.device,
                            dtype=reference_parameter.dtype,
                        )
                        / math.sqrt(int(counterfactual_action_expert.hidden_dim))
                    ),
                )
            else:
                # PGC v2 preserves the released query-free Action Expert
                # interface. Goal tokens are injected by a strictly zero-init
                # residual adapter, so the untrained branch equals Base.
                counterfactual_action_expert.use_latent_action_queries = False
            self.policy_guard_action_expert = counterfactual_action_expert

            hidden_dim = int(guard_config.get("hidden_dim", 512))
            projection_dim = int(guard_config.get("projection_dim", 256))
            verifier_hidden_dim = int(
                guard_config.get("verifier_hidden_dim", 256)
            )
            self.policy_guard_modules.update(
                {
                    "goal_graph": GoalGraphEncoder(
                        text_dim=self.text_dim,
                        video_dim=int(self.video_expert.hidden_dim),
                        action_dim=int(counterfactual_action_expert.hidden_dim),
                        hidden_dim=hidden_dim,
                        projection_dim=projection_dim,
                        num_goal_tokens=int(
                            guard_config.get("num_goal_tokens", 4)
                        ),
                        num_heads=int(guard_config.get("num_heads", 8)),
                    ),
                    "verifier": ActionOutcomeVerifier(
                        action_dim=int(counterfactual_action_expert.action_dim),
                        video_dim=int(self.video_expert.hidden_dim),
                        goal_dim=projection_dim,
                        hidden_dim=verifier_hidden_dim,
                    ),
                }
            )
            if self.policy_guard_version >= 2:
                goal_query_seeds = nn.Embedding(
                    num_action_queries,
                    int(counterfactual_action_expert.hidden_dim),
                )
                nn.init.normal_(
                    goal_query_seeds.weight,
                    std=(
                        1.0
                        / math.sqrt(
                            int(counterfactual_action_expert.hidden_dim)
                        )
                    ),
                )
                self.policy_guard_modules.update(
                    {
                        "goal_query_seeds": goal_query_seeds,
                        "goal_residual_adapter": GoalResidualAdapter(
                            action_dim=int(
                                counterfactual_action_expert.hidden_dim
                            ),
                            num_heads=int(guard_config.get("num_heads", 8)),
                            residual_scale=(
                                self.policy_guard_goal_residual_scale
                            ),
                        ),
                    }
                )
            self.policy_guard_modules.to(dtype=self.torch_dtype)
            self.policy_guard_alignment_loss = GoalActionAlignmentLoss(
                temperature=float(
                    guard_config.get("alignment_temperature", 0.07)
                )
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
        """Set optimizer-step progress used by TC contract warm-up."""
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

    def _transition_state_grounding_scale(self) -> float:
        """Warm up the new v6 policy bias while keeping deployment fully active."""
        if not self._transition_training_progress_active:
            return 1.0
        return self._transition_contract_scale()

    def configure_lora(self, config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        normalized = normalize_lora_config(config)
        if self.policy_guard_enabled and normalized["enabled"]:
            if normalized["experts"] != ["action"]:
                raise ValueError(
                    "PGC LoRA must target only its independent counterfactual "
                    "Action Expert (`lora.experts=[action]`)."
                )
            if normalized["extra_trainable_patterns"]:
                raise ValueError(
                    "PGC does not accept `lora.extra_trainable_patterns`; its "
                    "only trainable Action-Expert tensors are LoRA A/B"
                    + (
                        " and `latent_action_queries`."
                        if self.policy_guard_version == 1
                        else "."
                    )
                )
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
        if self.policy_guard_enabled:
            if self.policy_guard_action_expert is None:
                raise RuntimeError("PGC Action Expert was not initialized.")
            names = inject_lora(
                self.policy_guard_action_expert,
                target_modules=normalized["target_modules"],
                rank=normalized["rank"],
                alpha=normalized["alpha"],
                dropout=normalized["dropout"],
            )
            injected.extend(
                f"policy_guard_action_expert.{name}" for name in names
            )
        else:
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
                injected.extend(
                    f"{expert_name}_expert.{name}" for name in names
                )

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

    def _policy_guard_action_adapter_parameter_ids(self) -> set[int]:
        """Return the hard allowlist for the PGC Action Expert optimizer."""
        if (
            not self.policy_guard_enabled
            or not self.lora_enabled
            or self.policy_guard_action_expert is None
        ):
            return set()
        return {
            id(parameter)
            for name, parameter in self.policy_guard_action_expert.named_parameters()
            if (
                self.policy_guard_version == 1
                and name == "latent_action_queries"
            )
            or is_lora_parameter_name(name)
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

    def _transition_parameter_ids(self) -> set[int]:
        if not self.transition_contract_enabled:
            return set()
        return {
            id(parameter)
            for parameter in self.transition_contract_modules.parameters()
        }

    def prepare_trainable_parameters(self) -> dict[str, int]:
        """Freeze the base model and expose only configured adapter parameters."""
        self.eval()
        self.requires_grad_(False)
        if self.policy_guard_enabled:
            if self.policy_guard_action_expert is None:
                raise RuntimeError("PGC Action Expert was not initialized.")
            if not self.lora_enabled:
                raise ValueError(
                    "PGC full Action-Expert fine-tuning is disabled. Enable "
                    "action-only LoRA before preparing trainable parameters."
                )
            if self.policy_guard_legacy_full_loaded:
                raise ValueError(
                    "Legacy full-PGC checkpoints are evaluation-only and "
                    "cannot be resumed for LoRA training. Start from the "
                    "released FastWAM base checkpoint."
                )
            adapter_ids = self._policy_guard_action_adapter_parameter_ids()
            if not adapter_ids:
                raise ValueError("PGC LoRA produced zero Action-Expert parameters.")
            self.policy_guard_action_expert.train()
            for parameter in self.policy_guard_action_expert.parameters():
                if id(parameter) in adapter_ids:
                    parameter.requires_grad_(True)
            self.policy_guard_modules.train()
            self.policy_guard_modules.requires_grad_(True)
            trainable = sum(
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            )
            total = sum(parameter.numel() for parameter in self.parameters())
            if trainable <= 0:
                raise ValueError("PGC produced zero trainable parameters.")
            return {"trainable": int(trainable), "total": int(total)}

        if not self.transition_freeze_m1_policy:
            self.dit.train()

        if self.lora_enabled:
            adapter_ids = self._adapter_parameter_ids()
            if self.transition_freeze_m1_policy:
                adapter_ids &= self._transition_parameter_ids()
            for parameter in self.parameters():
                if id(parameter) in adapter_ids:
                    parameter.requires_grad_(True)
        else:
            if not self.transition_freeze_m1_policy:
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
        policy_guard_config: Optional[dict[str, Any]] = None,
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
            policy_guard_config=policy_guard_config,
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

    def _encode_policy_guard_goal(
        self,
        *,
        final_video_hidden: torch.Tensor,
        video_tokens_per_frame: int,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if not self.policy_guard_enabled or self.policy_guard_action_expert is None:
            raise RuntimeError("PGC goal encoding requires policy_guard.enabled=true.")
        current_token_count = min(
            int(video_tokens_per_frame), int(final_video_hidden.shape[1])
        )
        if current_token_count <= 0:
            raise ValueError("PGC requires at least one current visual token.")
        if self.policy_guard_version >= 2:
            seed_module = self.policy_guard_modules["goal_query_seeds"]
            base_queries = seed_module.weight.unsqueeze(0).expand(
                final_video_hidden.shape[0], -1, -1
            )
        else:
            base_queries = (
                self.policy_guard_action_expert.transition_queries.expand(
                    final_video_hidden.shape[0], -1, -1
                )
            )
        return self.policy_guard_modules["goal_graph"](
            base_queries=base_queries,
            language_hidden=context,
            language_mask=context_mask,
            current_video_hidden=final_video_hidden[:, :current_token_count].detach(),
        )

    def _prepare_policy_guard_action_tokens(
        self,
        *,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        full_context_mask: torch.Tensor,
        state_only_context_mask: torch.Tensor,
        routed_goal_queries: torch.Tensor,
    ) -> dict[str, Any]:
        if not self.policy_guard_enabled or self.policy_guard_action_expert is None:
            raise RuntimeError("PGC action preparation requires an initialized branch.")
        if self.policy_guard_version >= 2:
            action_pre = self.policy_guard_action_expert.pre_dit(
                action_tokens=action_tokens,
                timestep=timestep,
                context=context,
                context_mask=full_context_mask,
                use_queries=False,
            )
            residual_tokens, residual_metrics = self.policy_guard_modules[
                "goal_residual_adapter"
            ](
                action_pre["tokens"],
                routed_goal_queries,
            )
            action_pre["tokens"] = residual_tokens
            action_pre["policy_guard_residual_metrics"] = residual_metrics
            return action_pre

        num_queries = int(self.policy_guard_action_expert.num_latent_queries)
        query_context_mask = state_only_context_mask[:, None, :].expand(
            -1, num_queries, -1
        )
        action_context_mask = state_only_context_mask[:, None, :].expand(
            -1, int(action_tokens.shape[1]), -1
        )
        return self.policy_guard_action_expert.pre_dit(
            action_tokens=action_tokens,
            timestep=timestep,
            context=context,
            context_mask=state_only_context_mask,
            use_queries=True,
            query_context_mask=query_context_mask,
            action_context_mask=action_context_mask,
            transition_query_tokens=routed_goal_queries,
        )

    def _forward_policy_guard_action_from_cache(
        self,
        *,
        action_tokens: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        full_context_mask: torch.Tensor,
        state_only_context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        video_seq_len: int,
        video_tokens_per_frame: int,
        routed_goal_queries: torch.Tensor,
        return_metrics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.policy_guard_action_expert is None:
            raise RuntimeError("PGC Action Expert is unavailable.")
        action_pre = self._prepare_policy_guard_action_tokens(
            action_tokens=action_tokens,
            timestep=timestep_action,
            context=context,
            full_context_mask=full_context_mask,
            state_only_context_mask=state_only_context_mask,
            routed_goal_queries=routed_goal_queries,
        )
        num_queries = int(action_pre["meta"]["num_queries"])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=int(action_pre["tokens"].shape[1]),
            video_tokens_per_frame=video_tokens_per_frame,
            device=action_pre["tokens"].device,
            num_queries=num_queries,
            action_reads_raw_video=(self.policy_guard_version >= 2),
            queries_read_raw_video=False,
        )
        output_tokens = self.mot.forward_action_with_video_cache(
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
            action_expert=self.policy_guard_action_expert,
        )
        output = self.policy_guard_action_expert.post_dit(
            output_tokens, action_pre
        )
        if return_metrics:
            return output, dict(
                action_pre.get("policy_guard_residual_metrics", {})
            )
        return output

    def _policy_guard_clean_action_from_velocity(
        self,
        *,
        noisy_action: torch.Tensor,
        predicted_velocity: torch.Tensor,
        timestep_action: torch.Tensor,
    ) -> torch.Tensor:
        sigma = (
            timestep_action / float(self.train_action_scheduler.num_train_timesteps)
        ).to(device=noisy_action.device, dtype=noisy_action.dtype)
        sigma = sigma.view(-1, *([1] * (noisy_action.ndim - 1)))
        return noisy_action - sigma * predicted_velocity

    @staticmethod
    def _masked_policy_guard_mean(
        values: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        valid = valid_mask.to(device=values.device, dtype=values.dtype)
        result = (values * valid).sum() / valid.sum().clamp_min(1.0)
        if not bool(valid_mask.any()):
            result = values.sum() * 0.0
        return result

    def _compute_policy_guard_v2_action_losses(
        self,
        *,
        predicted_action: torch.Tensor,
        base_action_teacher: torch.Tensor,
        target_action: torch.Tensor,
        action_weight: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
        is_counterfactual: torch.Tensor,
        direct_action_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Separate native policy anchoring from counterfactual supervision."""
        if self.policy_guard_version < 2:
            raise RuntimeError("PGC v2 action losses require version >= 2.")
        is_counterfactual = is_counterfactual.to(
            device=predicted_action.device, dtype=torch.bool
        )
        direct_action_valid = direct_action_valid.to(
            device=predicted_action.device, dtype=torch.bool
        )
        if is_counterfactual.shape != direct_action_valid.shape or (
            is_counterfactual.ndim != 1
        ):
            raise ValueError("PGC v2 sample masks must share [B] shape.")
        batch_size = int(is_counterfactual.shape[0])
        # The scheduler deliberately returns a scalar for a one-sample local
        # micro-batch. Other scheduler implementations may retain singleton
        # dimensions. Normalize either representation to one weight/sample
        # before combining it with the provenance masks.
        if action_weight.ndim == 0:
            action_weight = action_weight.expand(batch_size)
        elif action_weight.numel() == batch_size:
            action_weight = action_weight.reshape(batch_size)
        else:
            raise ValueError(
                "PGC v2 action weights must contain exactly one value per "
                f"sample, got shape {tuple(action_weight.shape)} for "
                f"batch size {batch_size}."
            )
        action_weight = action_weight.to(
            device=predicted_action.device,
            dtype=torch.float32,
        )

        native_valid = direct_action_valid & ~is_counterfactual
        counterfactual_valid = direct_action_valid & is_counterfactual
        flow_error = self._compute_action_loss_per_sample(
            pred_action=predicted_action,
            target_action=target_action,
            action_is_pad=action_is_pad,
        )
        distillation_error = self._compute_action_loss_per_sample(
            pred_action=predicted_action,
            target_action=base_action_teacher.detach(),
            action_is_pad=action_is_pad,
        )
        counterfactual_action_loss = self._masked_policy_guard_mean(
            flow_error * action_weight,
            counterfactual_valid,
        )
        native_distillation_loss = self._masked_policy_guard_mean(
            distillation_error * action_weight,
            native_valid,
        )
        metrics = {
            "pgc_native_fraction": (~is_counterfactual).float().mean(),
            "pgc_counterfactual_fraction": is_counterfactual.float().mean(),
            "pgc_native_valid_fraction": native_valid.float().mean(),
            "pgc_counterfactual_valid_fraction": (
                counterfactual_valid.float().mean()
            ),
            "pgc_native_student_teacher_mse": (
                self._masked_policy_guard_mean(
                    distillation_error,
                    native_valid,
                ).detach()
            ),
            "pgc_counterfactual_flow_mse": (
                self._masked_policy_guard_mean(
                    flow_error,
                    counterfactual_valid,
                ).detach()
            ),
        }
        return counterfactual_action_loss, native_distillation_loss, metrics

    def _compute_policy_guard_verifier_loss(
        self,
        *,
        current_video_hidden: torch.Tensor,
        goal_embedding: torch.Tensor,
        demonstrated_action: torch.Tensor,
        base_candidate_action: torch.Tensor,
        counterfactual_candidate_action: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
        is_counterfactual: torch.Tensor,
        direct_action_valid: torch.Tensor,
        goal_ids: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        verifier = self.policy_guard_modules["verifier"]
        goal_state = verifier.encode_goal_state(
            current_video_hidden.detach(), goal_embedding
        )
        demonstrated_embedding = verifier.encode_action(
            demonstrated_action, action_is_pad
        )
        base_embedding = verifier.encode_action(
            base_candidate_action.detach(), action_is_pad
        )
        counterfactual_embedding = verifier.encode_action(
            counterfactual_candidate_action.detach(), action_is_pad
        )
        demonstrated_logits = verifier.score_embeddings(
            goal_state, demonstrated_embedding
        )
        base_logits = verifier.score_embeddings(goal_state, base_embedding)
        counterfactual_logits = verifier.score_embeddings(
            goal_state, counterfactual_embedding
        )

        valid = direct_action_valid.to(
            device=demonstrated_logits.device, dtype=torch.bool
        )
        is_counterfactual = is_counterfactual.to(
            device=demonstrated_logits.device, dtype=torch.bool
        )
        positive = torch.ones_like(demonstrated_logits)
        loss_demonstrated = self._masked_policy_guard_mean(
            F.binary_cross_entropy_with_logits(
                demonstrated_logits, positive, reduction="none"
            ),
            valid,
        )
        def _candidate_quality(candidate: torch.Tensor) -> torch.Tensor:
            per_step = F.mse_loss(
                candidate.float(), demonstrated_action.float(), reduction="none"
            ).mean(dim=-1)
            if action_is_pad is None:
                error = per_step.mean(dim=1)
            else:
                action_valid = (~action_is_pad).to(
                    device=per_step.device, dtype=per_step.dtype
                )
                error = (per_step * action_valid).sum(dim=1) / action_valid.sum(
                    dim=1
                ).clamp_min(1.0)
            return torch.exp(
                -error / self.policy_guard_verifier_action_mse_temperature
            ).clamp(0.0, 1.0)

        base_target = _candidate_quality(base_candidate_action.detach())
        counterfactual_target = _candidate_quality(
            counterfactual_candidate_action.detach()
        )
        loss_counterfactual_candidate = self._masked_policy_guard_mean(
            F.binary_cross_entropy_with_logits(
                counterfactual_logits,
                counterfactual_target.to(dtype=counterfactual_logits.dtype),
                reduction="none",
            ),
            valid,
        )
        loss_base_candidate = self._masked_policy_guard_mean(
            F.binary_cross_entropy_with_logits(
                base_logits,
                base_target.to(dtype=base_logits.dtype),
                reduction="none",
            ),
            valid,
        )
        counterfactual_valid = valid & is_counterfactual
        ranking_valid = counterfactual_valid & (
            counterfactual_target > base_target
        )
        base_score = torch.sigmoid(base_logits)
        counterfactual_score = torch.sigmoid(counterfactual_logits)
        ranking_margin = (
            counterfactual_score - base_score
            if self.policy_guard_version >= 2
            else counterfactual_logits - base_logits
        )
        ranking = torch.relu(
            self.policy_guard_verifier_margin
            - ranking_margin
        )
        loss_ranking = self._masked_policy_guard_mean(
            ranking, ranking_valid
        )
        verifier_loss = (
            loss_demonstrated
            + loss_counterfactual_candidate
            + loss_base_candidate
        ) / 3.0 + loss_ranking

        if self.policy_guard_alignment_loss is None:
            raise RuntimeError("PGC alignment loss was not initialized.")
        alignment_loss, alignment_metrics = self.policy_guard_alignment_loss(
            goal_state,
            demonstrated_embedding,
            group_ids=goal_ids,
        )
        predicted_override = (
            (counterfactual_score >= self.policy_guard_min_counterfactual_score)
            & (
                counterfactual_score - base_score
                >= self.policy_guard_gate_threshold
            )
        )
        metrics = {
            "loss_pgc_verifier_demonstrated": loss_demonstrated.detach(),
            "loss_pgc_verifier_base": loss_base_candidate.detach(),
            "loss_pgc_verifier_counterfactual": (
                loss_counterfactual_candidate.detach()
            ),
            "loss_pgc_verifier_ranking": loss_ranking.detach(),
            "pgc_verifier_base_score": base_score.detach().mean(),
            "pgc_verifier_counterfactual_score": (
                counterfactual_score.detach().mean()
            ),
            "pgc_verifier_base_quality_target": base_target.detach().mean(),
            "pgc_verifier_counterfactual_quality_target": (
                counterfactual_target.detach().mean()
            ),
            "pgc_verifier_score_margin": (
                counterfactual_score.detach() - base_score.detach()
            ).mean(),
            "pgc_direct_counterfactual_fraction": is_counterfactual.float().mean(),
            "pgc_direct_action_valid_fraction": valid.float().mean(),
            "pgc_verifier_ranking_valid_fraction": ranking_valid.float().mean(),
            "pgc_verifier_probability_margin_training": (
                base_score.new_tensor(float(self.policy_guard_version >= 2))
            ),
            "pgc_predicted_override_rate": predicted_override.float().mean(),
            "pgc_predicted_override_rate_on_counterfactual": (
                self._masked_policy_guard_mean(
                    predicted_override.float(), counterfactual_valid
                ).detach()
            ),
            "pgc_predicted_override_rate_on_native": (
                self._masked_policy_guard_mean(
                    predicted_override.float(), valid & ~is_counterfactual
                ).detach()
            ),
        }
        metrics.update(detached_policy_guard_metrics(alignment_metrics))
        return verifier_loss, alignment_loss, metrics

    def _select_policy_guard_action(
        self,
        *,
        base_action: torch.Tensor,
        counterfactual_action: torch.Tensor,
        base_score: torch.Tensor,
        counterfactual_score: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if base_action.shape != counterfactual_action.shape:
            raise ValueError("PGC action candidates must share shape.")
        if base_score.shape != counterfactual_score.shape or base_score.ndim != 1:
            raise ValueError("PGC candidate scores must share [B] shape.")
        if self.policy_guard_gate_mode == "base":
            selected = torch.zeros_like(base_score, dtype=torch.bool)
        elif self.policy_guard_gate_mode == "counterfactual":
            selected = torch.ones_like(base_score, dtype=torch.bool)
        else:
            selected = (
                counterfactual_score >= self.policy_guard_min_counterfactual_score
            ) & (
                counterfactual_score - base_score
                >= self.policy_guard_gate_threshold
            )
        while selected.ndim < base_action.ndim:
            selected = selected.unsqueeze(-1)
        output = torch.where(selected, counterfactual_action, base_action)
        return output, selected.reshape(base_score.shape[0], -1)[:, 0]

    def encode_intended_transition(
        self,
        *,
        video_tokens: torch.Tensor,
        video_tokens_per_frame: int,
        context: torch.Tensor,
        full_context_mask: torch.Tensor,
        route_scale: Optional[float] = None,
        grounding_video_tokens: Optional[torch.Tensor] = None,
        return_grounding_state: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]
        | tuple[
            torch.Tensor,
            torch.Tensor,
            dict[str, torch.Tensor],
            dict[str, torch.Tensor],
        ]
    ):
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
        grounding_state: dict[str, torch.Tensor] = {}
        visual_attention_bias = None
        visual_attention_bias_scale = 0.0
        if self.transition_use_state_conditioned_grounding:
            if grounding_video_tokens is None:
                grounding_video_tokens = current_video
            grounding_video = grounding_video_tokens[
                :, : int(video_tokens_per_frame)
            ]
            # Cached WAN contexts deliberately expose an all-true attention
            # mask after zeroing padded rows. Mean pooling that mask would
            # dilute short instructions with projection bias from many zero
            # rows, so recover the non-zero language support for grounding.
            grounding_language_mask = full_context_mask & (
                context.detach().float().abs().amax(dim=-1) > 0
            )
            grounding_language_mask = torch.where(
                ~grounding_language_mask.any(dim=-1, keepdim=True),
                full_context_mask,
                grounding_language_mask,
            )
            (
                grounding_similarity,
                grounding_attention,
                grounding_visual_features,
                grounding_metrics,
            ) = self.transition_contract_modules["state_target_grounder"](
                language_hidden=language_hidden,
                language_mask=grounding_language_mask,
                current_video_hidden=grounding_video,
            )
            visual_attention_bias = grounding_similarity
            visual_attention_bias_scale = (
                self.transition_state_grounding_router_bias
                * self._transition_state_grounding_scale()
            )
            grounding_state = {
                "attention": grounding_attention,
                "similarity": grounding_similarity,
                "visual_features": grounding_visual_features,
            }
        else:
            grounding_metrics = {}
        routed, router_metrics = self.transition_contract_modules["router"](
            transition_queries=transition_queries,
            language_hidden=language_hidden,
            language_mask=full_context_mask,
            current_video_hidden=current_video,
            route_scale=route_scale,
            visual_attention_bias=visual_attention_bias,
            visual_attention_bias_scale=visual_attention_bias_scale,
        )
        router_metrics.update(grounding_metrics)
        z_language = self.transition_contract_modules["intent_projection"](
            routed.mean(dim=1)
        )
        if return_grounding_state:
            return routed, z_language, router_metrics, grounding_state
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
        self,
        z_language: torch.Tensor,
        z_future: torch.Tensor,
        *,
        group_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.transition_contract_loss is None:
            raise RuntimeError("Transition Contract loss is unavailable.")
        return self.transition_contract_loss(
            z_language,
            z_future,
            metric_prefix="LF",
            group_ids=group_ids,
        )

    def encode_action_effect_transition(
        self,
        *,
        current_video_hidden: torch.Tensor,
        action: torch.Tensor,
        proprio: Optional[torch.Tensor],
        action_is_pad: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not self.transition_use_action_effect:
            raise RuntimeError("Action-effect transition encoding is disabled.")
        return self.transition_contract_modules["action_effect_encoder"](
            current_video_hidden=current_video_hidden,
            action=action,
            proprio=proprio,
            action_is_pad=action_is_pad,
        )

    def compute_action_future_contract_loss(
        self,
        z_action: torch.Tensor,
        z_future: torch.Tensor,
        *,
        group_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.transition_contract_loss is None:
            raise RuntimeError("Transition Contract loss is unavailable.")
        if not self.transition_use_action_effect:
            raise RuntimeError("Action-Future Contract is disabled.")
        return self.transition_contract_loss(
            z_action,
            z_future,
            metric_prefix="AF",
            group_ids=group_ids,
        )

    def compute_counterfactual_ranking_loss(
        self,
        z_positive: torch.Tensor,
        z_negative: torch.Tensor,
        z_future: torch.Tensor,
        *,
        valid_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.transition_counterfactual_loss is None:
            raise RuntimeError("Counterfactual ranking loss is unavailable.")
        return self.transition_counterfactual_loss(
            z_positive,
            z_negative,
            z_future,
            valid_mask=valid_mask,
        )

    def compute_counterfactual_action_separation_loss(
        self,
        *,
        positive_error: torch.Tensor,
        counterfactual_source_error: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Prevent alternate language from reproducing the source action.

        The hinge is bounded: once the counterfactual branch is sufficiently
        worse on the source target, it receives no incentive to diverge
        further.  Direction comes from the positive task prototype loss.
        """
        if positive_error.shape != counterfactual_source_error.shape or (
            positive_error.ndim != 1
        ):
            raise ValueError("Counterfactual action errors must share [B] shape.")
        if valid_mask.shape != positive_error.shape:
            raise ValueError("Counterfactual action valid mask must be [B].")
        valid_mask = valid_mask.to(
            device=positive_error.device,
            dtype=torch.bool,
        )
        source_gap = counterfactual_source_error - positive_error.detach()
        per_sample = torch.relu(
            self.transition_counterfactual_action_separation_margin - source_gap
        )
        valid = valid_mask.to(dtype=per_sample.dtype)
        valid_count = valid.sum()
        loss = (per_sample * valid).sum() / valid_count.clamp_min(1.0)
        if not bool(valid_mask.any()):
            loss = (
                positive_error.sum() + counterfactual_source_error.sum()
            ) * 0.0

        def _valid_mean(value: torch.Tensor) -> torch.Tensor:
            return (value * valid).sum() / valid_count.clamp_min(1.0)

        return loss, {
            "counterfactual_action_source_mse": _valid_mean(
                counterfactual_source_error
            ),
            "counterfactual_action_source_gap": _valid_mean(source_gap),
            "counterfactual_action_separation_satisfied_fraction": (
                _valid_mean(
                    (
                        source_gap
                        >= self.transition_counterfactual_action_separation_margin
                    ).to(valid.dtype)
                )
            ),
        }

    @staticmethod
    def _state_grounding_cross_entropy(
        *,
        student_attention: torch.Tensor,
        teacher_attention: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Normalized soft-label CE with an all-invalid-row-safe reduction."""
        if student_attention.shape != teacher_attention.shape:
            raise ValueError("State-grounding attention tensors must share [B,N].")
        if valid_mask.shape != student_attention.shape[:1]:
            raise ValueError("State-grounding valid mask must be [B].")
        token_count = int(student_attention.shape[-1])
        normalizer = max(1.0, math.log(max(2, token_count)))
        per_sample = -(
            teacher_attention.float()
            * student_attention.float().clamp_min(1e-8).log()
        ).sum(dim=-1) / normalizer
        valid = valid_mask.to(device=per_sample.device, dtype=per_sample.dtype)
        loss = (per_sample * valid).sum() / valid.sum().clamp_min(1.0)
        if not bool(valid_mask.any()):
            loss = (student_attention.sum() + teacher_attention.sum()) * 0.0
        return loss

    def compute_state_conditioned_grounding_loss(
        self,
        *,
        clean_input_latents: torch.Tensor,
        tokens_per_frame: int,
        positive_state: dict[str, torch.Tensor],
        counterfactual_state: dict[str, torch.Tensor],
        transition_task_ids: torch.Tensor,
        counterfactual_task_ids: torch.Tensor,
        counterfactual_valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Supervise language targets in the current visual state.

        Correct demonstrations provide a language-free interaction-region
        teacher from local latent change.  Their grounded visual features
        update a training-only task appearance bank.  The requested alternate
        task prototype is then matched against patches in this exact source
        state, yielding a counterfactual spatial positive without simulator
        coordinates or task-average action labels.
        """
        if not self.transition_use_state_conditioned_grounding:
            raise RuntimeError("State-conditioned target grounding is disabled.")
        required = {"attention", "visual_features"}
        if not required.issubset(positive_state) or not required.issubset(
            counterfactual_state
        ):
            raise RuntimeError("State-grounding branches did not return their state.")

        positive_attention = positive_state["attention"]
        counterfactual_attention = counterfactual_state["attention"]
        positive_visual = positive_state["visual_features"]
        counterfactual_visual = counterfactual_state["visual_features"]
        teacher_attention, teacher_valid, teacher_metrics = (
            interaction_patch_distribution(
                clean_input_latents.detach(),
                tokens_per_frame=int(tokens_per_frame),
                topk_fraction=self.transition_state_grounding_teacher_topk,
                temperature=self.transition_state_grounding_teacher_temperature,
            )
        )
        teacher_attention = teacher_attention.to(
            device=positive_attention.device, dtype=positive_attention.dtype
        )
        teacher_valid = teacher_valid.to(
            device=positive_attention.device, dtype=torch.bool
        )
        prototype_bank = self.transition_contract_modules[
            "state_target_prototypes"
        ]
        prototype_bank.update(
            task_ids=transition_task_ids,
            visual_features=positive_visual,
            teacher_attention=teacher_attention,
            valid_mask=teacher_valid,
        )
        target_attention, target_valid, target_metrics = (
            prototype_bank.target_distribution(
                task_ids=counterfactual_task_ids,
                visual_features=counterfactual_visual,
                valid_mask=counterfactual_valid_mask,
            )
        )
        target_attention = target_attention.to(
            device=counterfactual_attention.device,
            dtype=counterfactual_attention.dtype,
        )
        target_valid = target_valid.to(
            device=counterfactual_attention.device, dtype=torch.bool
        )

        loss_correct = self._state_grounding_cross_entropy(
            student_attention=positive_attention,
            teacher_attention=teacher_attention,
            valid_mask=teacher_valid,
        )
        loss_counterfactual = self._state_grounding_cross_entropy(
            student_attention=counterfactual_attention,
            teacher_attention=target_attention,
            valid_mask=target_valid,
        )
        overlap = F.cosine_similarity(
            positive_attention.float(),
            counterfactual_attention.float(),
            dim=-1,
            eps=1e-8,
        )
        separation_per_sample = torch.relu(
            overlap - self.transition_state_grounding_overlap_margin
        )
        target_valid_float = target_valid.to(dtype=separation_per_sample.dtype)
        loss_separation = (
            separation_per_sample * target_valid_float
        ).sum() / target_valid_float.sum().clamp_min(1.0)
        if not bool(target_valid.any()):
            loss_separation = (
                positive_attention.sum() + counterfactual_attention.sum()
            ) * 0.0

        loss = (
            self.transition_state_grounding_correct_weight * loss_correct
            + self.transition_state_grounding_counterfactual_weight
            * loss_counterfactual
            + self.transition_state_grounding_separation_weight
            * loss_separation
        )

        def _valid_mean(
            value: torch.Tensor, valid_mask: torch.Tensor
        ) -> torch.Tensor:
            valid = valid_mask.to(device=value.device, dtype=value.dtype)
            return (value * valid).sum() / valid.sum().clamp_min(1.0)

        correct_top1_match = (
            positive_attention.argmax(dim=-1)
            == teacher_attention.argmax(dim=-1)
        ).to(dtype=positive_attention.dtype)
        target_top1_match = (
            counterfactual_attention.argmax(dim=-1)
            == target_attention.argmax(dim=-1)
        ).to(dtype=counterfactual_attention.dtype)
        metrics = {
            "loss_state_grounding_correct": loss_correct.detach(),
            "loss_state_grounding_counterfactual": loss_counterfactual.detach(),
            "loss_state_grounding_separation": loss_separation.detach(),
            "state_grounding_correct_top1_acc": _valid_mean(
                correct_top1_match, teacher_valid
            ),
            "state_grounding_counterfactual_top1_acc": _valid_mean(
                target_top1_match, target_valid
            ),
            "state_grounding_positive_counterfactual_overlap": _valid_mean(
                overlap, target_valid
            ),
            "state_grounding_separation_satisfied_fraction": _valid_mean(
                (
                    overlap <= self.transition_state_grounding_overlap_margin
                ).to(dtype=overlap.dtype),
                target_valid,
            ),
            "state_grounding_counterfactual_target_retrieval_acc": (
                prototype_bank.retrieval_accuracy(
                    task_ids=counterfactual_task_ids,
                    visual_features=counterfactual_visual,
                    attention=counterfactual_attention,
                    valid_mask=target_valid,
                )
            ),
        }
        metrics.update(teacher_metrics)
        metrics.update(target_metrics)
        return loss, metrics

    def _forward_counterfactual_action_positive_train(
        self,
        *,
        latents: torch.Tensor,
        timestep_video: torch.Tensor,
        action: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        context: torch.Tensor,
        full_context_mask: torch.Tensor,
        state_only_context_mask: torch.Tensor,
        noisy_action: torch.Tensor,
        timestep_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the deployment policy under alternate language on the same state."""
        if not self.transition_use_counterfactual_action_positive:
            raise RuntimeError(
                "Counterfactual action positive supervision is disabled."
            )
        # The protected Video/MoT policy is frozen in v5+. This second
        # language-conditioned Video pass is needed for deployment fidelity,
        # but its activations never need gradients.  Gradients begin at the
        # Router query consumed by the frozen Action Expert.
        with torch.no_grad():
            video_pre = self.video_expert.pre_dit(
                x=latents.detach(),
                timestep=timestep_video,
                context=context,
                context_mask=full_context_mask,
                action=action,
                fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            )
            video_kv_cache, final_video_hidden = (
                self._run_video_expert_to_final_hidden(video_pre)
            )
        pred_action, z_action_intent, _ = (
            self._forward_tc_v2_action_from_video_hidden(
                video_pre=video_pre,
                final_video_hidden=final_video_hidden,
                video_kv_cache=video_kv_cache,
                pred_action_m1=None,
                action_tokens=noisy_action,
                timestep_action=timestep_action,
                context=context,
                full_context_mask=full_context_mask,
                state_only_context_mask=state_only_context_mask,
            )
        )
        routed_queries, _, _ = self.encode_intended_transition(
            video_tokens=final_video_hidden,
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            context=context,
            full_context_mask=full_context_mask,
            route_scale=1.0,
            grounding_video_tokens=video_pre["tokens"],
        )
        base_queries = self.action_expert.transition_queries.expand(
            routed_queries.shape[0], -1, -1
        )
        return pred_action, routed_queries - base_queries, z_action_intent

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
        pred_action_m1: Optional[torch.Tensor] = None,
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
        scheduled_route_scale = self._transition_router_scale()
        # Protected TC v3+ separates policy protection from student optimization. The
        # frozen joint-MoT path is the teacher at every step, while the pure
        # Router student is optimized from step one.  Reusing the v2 recovery
        # blend here would make the first recovery window teacher-only and
        # therefore give the Router no action/distillation gradient.
        route_scale = (
            1.0 if self.transition_freeze_m1_policy else scheduled_route_scale
        )
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
            # Grounding must be learned from language-neutral current-state
            # patch embeddings, while the Router still reads semantic final
            # Video-Expert hidden states.
            grounding_video_tokens=video_pre["tokens"],
        )
        base_queries = self.action_expert.transition_queries.expand(
            routed_full.shape[0], -1, -1
        )
        router_metrics["router_route_scale"] = routed_full.new_tensor(route_scale)
        router_metrics["router_recovery_schedule_scale"] = (
            routed_full.new_tensor(scheduled_route_scale)
        )
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

        if route_scale < 1.0 and pred_action_m1 is None:
            raise RuntimeError(
                "TC-C recovery requires the exact joint-MoT M1 action output; "
                "the sequential Video-cache path cannot synthesize it."
            )

        if route_scale <= 0.0:
            # The caller ran the original joint-MoT posterior policy. Returning
            # that exact tensor makes recovery a function-level invariant,
            # including under BF16 SDPA and stochastic LoRA dropout.
            router_metrics["policy_recovery_output_gap"] = (
                pred_action_m1.new_zeros(())
            )
            router_metrics["policy_recovery_joint_m1"] = (
                pred_action_m1.new_ones(())
            )
            return pred_action_m1, z_language, router_metrics

        if route_scale >= 1.0:
            # Final TC-C policy: Router is the only language/visual interface.
            pred_action = _run_policy(
                routed_full,
                m1_posterior_interface=False,
            )
            if pred_action_m1 is None:
                output_gap = pred_action.new_zeros(())
            else:
                output_gap = (
                    (pred_action - pred_action_m1)
                    .float()
                    .norm(dim=-1)
                    .mean()
                )
                router_metrics["policy_recovery_joint_m1"] = (
                    pred_action_m1.new_ones(())
                )
            router_metrics["policy_recovery_output_gap"] = output_gap
            return pred_action, z_language, router_metrics

        # Both branches share the same Video cache. This linear flow-velocity
        # blend makes the policy function continuous while shortcuts disappear.
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
        router_metrics["policy_recovery_joint_m1"] = (
            pred_action_m1.new_ones(())
        )
        return pred_action, z_language, router_metrics

    def _run_joint_m1_policy_with_video_cache(
        self,
        *,
        video_pre: dict[str, Any],
        action_tokens: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        full_context_mask: torch.Tensor,
        state_only_context_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]], torch.Tensor]:
        """Run the original M1 posterior policy and expose its Video state.

        Recovery must preserve the actual joint-MoT computation. Replaying the
        Action Expert later from a sequential Video prefill is algebraically
        similar, but changes BF16 attention kernels and LoRA-dropout RNG order
        on the full model. Those differences are large enough to invalidate a
        pretrained policy.
        """
        base_queries = self.action_expert.transition_queries.expand(
            action_tokens.shape[0], -1, -1
        )
        action_pre = self._prepare_action_tokens(
            action_tokens=action_tokens,
            timestep=timestep_action,
            context=context,
            full_context_mask=full_context_mask,
            state_only_context_mask=state_only_context_mask,
            mode="posterior",
            transition_query_tokens=base_queries,
            policy_recovery=True,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=int(action_pre["tokens"].shape[1]),
            video_tokens_per_frame=int(
                video_pre["meta"]["tokens_per_frame"]
            ),
            device=video_pre["tokens"].device,
            num_queries=int(action_pre["meta"]["num_queries"]),
            action_reads_raw_video=False,
            queries_read_raw_video=True,
        )
        joint_result = self.mot(
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
            return_video_cache=True,
        )
        if not isinstance(joint_result, tuple):
            raise RuntimeError("Joint M1 forward did not return Video K/V cache.")
        tokens_out, video_kv_cache = joint_result
        pred_action_m1 = self.action_expert.post_dit(
            tokens_out["action"], action_pre
        )
        return pred_action_m1, video_kv_cache, tokens_out["video"]

    def _forward_tc_v2_train(
        self,
        *,
        video_pre: dict[str, Any],
        action_tokens: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        full_context_mask: torch.Tensor,
        state_only_context_mask: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        dict[str, torch.Tensor],
    ]:
        """Run TC recovery/student policy and return an optional M1 teacher.

        TC-C v2 only evaluates the joint M1 path during recovery. TC v3+
        freezes that path and evaluates it at every step, so the pure Router
        policy can be distilled without allocating a second 6.8B model.
        """
        route_scale = self._transition_router_scale()
        pred_action_m1 = None
        teacher_required = bool(
            route_scale < 1.0 or self.transition_policy_distillation_enabled
        )
        if teacher_required:
            if self.transition_freeze_m1_policy:
                with torch.no_grad():
                    (
                        pred_action_m1,
                        video_kv_cache,
                        final_video_hidden,
                    ) = self._run_joint_m1_policy_with_video_cache(
                        video_pre=video_pre,
                        action_tokens=action_tokens,
                        timestep_action=timestep_action,
                        context=context,
                        full_context_mask=full_context_mask,
                        state_only_context_mask=state_only_context_mask,
                    )
            else:
                (
                    pred_action_m1,
                    video_kv_cache,
                    final_video_hidden,
                ) = self._run_joint_m1_policy_with_video_cache(
                    video_pre=video_pre,
                    action_tokens=action_tokens,
                    timestep_action=timestep_action,
                    context=context,
                    full_context_mask=full_context_mask,
                    state_only_context_mask=state_only_context_mask,
                )
        else:
            (
                video_kv_cache,
                final_video_hidden,
            ) = self._run_video_expert_to_final_hidden(video_pre)

        pred_action, z_language, router_metrics = (
            self._forward_tc_v2_action_from_video_hidden(
                video_pre=video_pre,
                final_video_hidden=final_video_hidden,
                video_kv_cache=video_kv_cache,
                pred_action_m1=pred_action_m1,
                action_tokens=action_tokens,
                timestep_action=timestep_action,
                context=context,
                full_context_mask=full_context_mask,
                state_only_context_mask=state_only_context_mask,
            )
        )
        return (
            pred_action,
            final_video_hidden,
            z_language,
            pred_action_m1,
            router_metrics,
        )

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
        negative_context = sample.get("negative_context")
        negative_context_mask = sample.get("negative_context_mask")
        negative_valid = sample.get("negative_valid")
        transition_task_id = sample.get("transition_task_id")
        counterfactual_task_id = sample.get("counterfactual_task_id")
        pgc_is_counterfactual = sample.get("pgc_is_counterfactual")
        pgc_direct_action_valid = sample.get("pgc_direct_action_valid")
        pgc_goal_id = sample.get("pgc_goal_id")
        proprio = sample.get("proprio", None)
        if video.ndim != 5:
            raise ValueError(f"`sample['video']` must be 5D [B, 3, T, H, W], got shape {tuple(video.shape)}")
        if video.shape[1] != 3:
            raise ValueError(f"`sample['video']` channel dimension must be 3, got shape {tuple(video.shape)}")

        batch_size, _, num_frames, height, width = video.shape
        if self.policy_guard_enabled:
            missing_pgc = [
                name
                for name, value in (
                    ("pgc_is_counterfactual", pgc_is_counterfactual),
                    ("pgc_direct_action_valid", pgc_direct_action_valid),
                    ("pgc_goal_id", pgc_goal_id),
                )
                if value is None
            ]
            if missing_pgc:
                raise ValueError(
                    "PGC training requires dataset provenance fields: "
                    f"{missing_pgc}. Use RobotVideoDataset with the PGC data options."
                )
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
        if self.transition_use_counterfactual_ranking:
            if negative_context is None or negative_context_mask is None:
                raise ValueError(
                    "TC-Full training requires `negative_context` and "
                    "`negative_context_mask` from the audited intervention manifest."
                )
            if negative_valid is None:
                raise ValueError("TC-Full training requires `negative_valid`.")
            if transition_task_id is None:
                raise ValueError("TC-Full training requires `transition_task_id`.")
        if self.transition_use_counterfactual_action_positive and (
            counterfactual_task_id is None
        ):
            raise ValueError(
                "TC-Full v5+ training requires `counterfactual_task_id`."
            )
        if (negative_context is None) != (negative_context_mask is None):
            raise ValueError(
                "`negative_context` and `negative_context_mask` must appear together."
            )
        if negative_context is not None:
            if negative_context.ndim != 3 or negative_context_mask.ndim != 2:
                raise ValueError(
                    "`negative_context/negative_context_mask` must be "
                    "[B,L,D]/[B,L]."
                )
            if negative_context.shape != context.shape or (
                negative_context_mask.shape != context_mask.shape
            ):
                raise ValueError(
                    "Positive and negative text-cache tensor shapes must match, got "
                    f"{tuple(context.shape)}/{tuple(context_mask.shape)} and "
                    f"{tuple(negative_context.shape)}/"
                    f"{tuple(negative_context_mask.shape)}."
                )
        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if negative_context is not None:
            negative_context = negative_context.to(
                device=self.device, dtype=self.torch_dtype, non_blocking=True
            )
            negative_context_mask = negative_context_mask.to(
                device=self.device, dtype=torch.bool, non_blocking=True
            )
        if negative_valid is not None:
            negative_valid = torch.as_tensor(
                negative_valid, device=self.device, dtype=torch.bool
            )
            if negative_valid.ndim == 0:
                negative_valid = negative_valid.expand(batch_size)
            if negative_valid.shape != (batch_size,):
                raise ValueError(
                    f"`negative_valid` must be [B], got {tuple(negative_valid.shape)}."
                )
        if transition_task_id is not None:
            transition_task_id = torch.as_tensor(
                transition_task_id, device=self.device, dtype=torch.long
            )
            if transition_task_id.ndim == 0:
                transition_task_id = transition_task_id.expand(batch_size)
            if transition_task_id.shape != (batch_size,):
                raise ValueError(
                    "`transition_task_id` must be [B], got "
                    f"{tuple(transition_task_id.shape)}."
                )
        if counterfactual_task_id is not None:
            counterfactual_task_id = torch.as_tensor(
                counterfactual_task_id,
                device=self.device,
                dtype=torch.long,
            )
            if counterfactual_task_id.ndim == 0:
                counterfactual_task_id = counterfactual_task_id.expand(
                    batch_size
                )
            if counterfactual_task_id.shape != (batch_size,):
                raise ValueError(
                    "`counterfactual_task_id` must be [B], got "
                    f"{tuple(counterfactual_task_id.shape)}."
                )
        if pgc_is_counterfactual is not None:
            pgc_is_counterfactual = torch.as_tensor(
                pgc_is_counterfactual, device=self.device, dtype=torch.bool
            )
            if pgc_is_counterfactual.ndim == 0:
                pgc_is_counterfactual = pgc_is_counterfactual.expand(batch_size)
            if pgc_is_counterfactual.shape != (batch_size,):
                raise ValueError("`pgc_is_counterfactual` must be [B].")
        if pgc_direct_action_valid is not None:
            pgc_direct_action_valid = torch.as_tensor(
                pgc_direct_action_valid, device=self.device, dtype=torch.bool
            )
            if pgc_direct_action_valid.ndim == 0:
                pgc_direct_action_valid = pgc_direct_action_valid.expand(batch_size)
            if pgc_direct_action_valid.shape != (batch_size,):
                raise ValueError("`pgc_direct_action_valid` must be [B].")
        if pgc_goal_id is not None:
            pgc_goal_id = torch.as_tensor(
                pgc_goal_id, device=self.device, dtype=torch.long
            )
            if pgc_goal_id.ndim == 0:
                pgc_goal_id = pgc_goal_id.expand(batch_size)
            if pgc_goal_id.shape != (batch_size,):
                raise ValueError("`pgc_goal_id` must be [B].")
        language_context_len = int(context.shape[1])
        has_proprio = False
        proprio_current = None
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
            proprio_current = proprio.to(
                device=self.device, dtype=self.torch_dtype, non_blocking=True
            )
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio_current,
            )
            if negative_context is not None:
                negative_context, negative_context_mask = (
                    self._append_proprio_to_context(
                        context=negative_context,
                        context_mask=negative_context_mask,
                        proprio=proprio_current,
                    )
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
            "negative_context": negative_context,
            "negative_context_mask": negative_context_mask,
            "negative_valid": negative_valid,
            "transition_task_id": transition_task_id,
            "counterfactual_task_id": counterfactual_task_id,
            "pgc_is_counterfactual": pgc_is_counterfactual,
            "pgc_direct_action_valid": pgc_direct_action_valid,
            "pgc_goal_id": pgc_goal_id,
            "language_context_len": language_context_len,
            "has_proprio": has_proprio,
            "proprio_current": proprio_current,
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

        if self.policy_guard_enabled:
            with torch.no_grad():
                video_pre = self.video_expert.pre_dit(
                    x=latents,
                    timestep=timestep_video,
                    context=context,
                    context_mask=full_context_mask,
                    action=action,
                    fuse_vae_embedding_in_latents=inputs[
                        "fuse_vae_embedding_in_latents"
                    ],
                )
        else:
            video_pre = self.video_expert.pre_dit(
                x=latents,
                timestep=timestep_video,
                context=context,
                context_mask=full_context_mask,
                action=action,
                fuse_vae_embedding_in_latents=inputs[
                    "fuse_vae_embedding_in_latents"
                ],
            )

        z_language = None
        pred_action_teacher = None
        router_metrics: dict[str, torch.Tensor] = {}
        pred_action_base = None
        policy_guard_goal_embedding = None
        policy_guard_current_video_hidden = None
        policy_guard_metrics: dict[str, torch.Tensor] = {}
        if self.transition_contract_enabled:
            (
                pred_action_post,
                final_video_hidden,
                z_language,
                pred_action_teacher,
                router_metrics,
            ) = self._forward_tc_v2_train(
                video_pre=video_pre,
                action_tokens=noisy_action,
                timestep_action=timestep_action,
                context=context,
                full_context_mask=full_context_mask,
                state_only_context_mask=state_only_context_mask,
            )
            pred_video = self.video_expert.post_dit(
                final_video_hidden, video_pre
            )
        elif self.policy_guard_enabled:
            action_pre_base = self.action_expert.pre_dit(
                action_tokens=noisy_action,
                timestep=timestep_action,
                context=context,
                context_mask=full_context_mask,
                use_queries=False,
            )
            video_tokens = video_pre["tokens"]
            base_action_tokens = action_pre_base["tokens"]
            base_attention_mask = self._build_mot_attention_mask(
                video_seq_len=int(video_tokens.shape[1]),
                action_seq_len=int(base_action_tokens.shape[1]),
                video_tokens_per_frame=int(
                    video_pre["meta"]["tokens_per_frame"]
                ),
                device=video_tokens.device,
                num_queries=0,
                action_reads_raw_video=True,
            )
            with torch.no_grad():
                base_result = self.mot(
                    embeds_all={
                        "video": video_tokens,
                        "action": base_action_tokens,
                    },
                    attention_mask=base_attention_mask,
                    freqs_all={
                        "video": video_pre["freqs"],
                        "action": action_pre_base["freqs"],
                    },
                    context_all={
                        "video": {
                            "context": video_pre["context"],
                            "mask": video_pre["context_mask"],
                        },
                        "action": {
                            "context": action_pre_base["context"],
                            "mask": action_pre_base["context_mask"],
                        },
                    },
                    t_mod_all={
                        "video": video_pre["t_mod"],
                        "action": action_pre_base["t_mod"],
                    },
                    return_video_cache=True,
                )
                if not isinstance(base_result, tuple):
                    raise RuntimeError("PGC base forward did not return Video cache.")
                base_tokens_out, video_kv_cache = base_result
                pred_video = self.video_expert.post_dit(
                    base_tokens_out["video"], video_pre
                )
                pred_action_base = self.action_expert.post_dit(
                    base_tokens_out["action"], action_pre_base
                )

            (
                routed_goal_queries,
                policy_guard_goal_embedding,
                policy_guard_metrics,
            ) = self._encode_policy_guard_goal(
                final_video_hidden=base_tokens_out["video"],
                video_tokens_per_frame=int(
                    video_pre["meta"]["tokens_per_frame"]
                ),
                context=context,
                context_mask=full_context_mask,
            )
            policy_guard_current_video_hidden = base_tokens_out["video"][:, : int(
                video_pre["meta"]["tokens_per_frame"]
            )].detach()
            (
                pred_action_post,
                policy_guard_residual_metrics,
            ) = self._forward_policy_guard_action_from_cache(
                action_tokens=noisy_action,
                timestep_action=timestep_action,
                context=context,
                full_context_mask=full_context_mask,
                state_only_context_mask=state_only_context_mask,
                video_kv_cache=video_kv_cache,
                video_seq_len=int(video_tokens.shape[1]),
                video_tokens_per_frame=int(
                    video_pre["meta"]["tokens_per_frame"]
                ),
                routed_goal_queries=routed_goal_queries,
                return_metrics=True,
            )
            policy_guard_metrics.update(policy_guard_residual_metrics)
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

        if self.policy_guard_enabled:
            if (
                pred_action_base is None
                or policy_guard_goal_embedding is None
                or policy_guard_current_video_hidden is None
            ):
                raise RuntimeError("PGC training branch did not produce candidates.")
            e_base = self._compute_action_loss_per_sample(
                pred_action=pred_action_base,
                target_action=target_action,
                action_is_pad=action_is_pad,
            )
            loss_action_base = (e_base * action_weight).mean()
            base_clean_action = self._policy_guard_clean_action_from_velocity(
                noisy_action=noisy_action,
                predicted_velocity=pred_action_base,
                timestep_action=timestep_action,
            )
            counterfactual_clean_action = (
                self._policy_guard_clean_action_from_velocity(
                    noisy_action=noisy_action,
                    predicted_velocity=pred_action_post,
                    timestep_action=timestep_action,
                )
            )
            verifier_loss, alignment_loss, verifier_metrics = (
                self._compute_policy_guard_verifier_loss(
                    current_video_hidden=policy_guard_current_video_hidden,
                    goal_embedding=policy_guard_goal_embedding,
                    demonstrated_action=action,
                    base_candidate_action=base_clean_action,
                    counterfactual_candidate_action=(
                        counterfactual_clean_action
                    ),
                    action_is_pad=action_is_pad,
                    is_counterfactual=inputs["pgc_is_counterfactual"],
                    direct_action_valid=inputs["pgc_direct_action_valid"],
                    goal_ids=inputs["pgc_goal_id"],
                )
            )
            policy_action_metrics: dict[str, torch.Tensor] = {}
            if self.policy_guard_version >= 2:
                (
                    counterfactual_action_loss,
                    native_distillation_loss,
                    policy_action_metrics,
                ) = self._compute_policy_guard_v2_action_losses(
                    predicted_action=pred_action_post,
                    base_action_teacher=pred_action_base,
                    target_action=target_action,
                    action_weight=action_weight,
                    action_is_pad=action_is_pad,
                    is_counterfactual=inputs["pgc_is_counterfactual"],
                    direct_action_valid=inputs["pgc_direct_action_valid"],
                )
                policy_action_objective = (
                    self.policy_guard_action_weight
                    * counterfactual_action_loss
                    + self.policy_guard_native_distillation_weight
                    * native_distillation_loss
                )
            else:
                counterfactual_action_loss = loss_action_post
                native_distillation_loss = loss_action_post.detach() * 0.0
                policy_action_objective = (
                    self.policy_guard_action_weight * loss_action_post
                )
            loss_total = (
                policy_action_objective
                + self.policy_guard_verifier_weight * verifier_loss
                + self.policy_guard_alignment_weight * alignment_loss
            )
            loss_dict.update(
                {
                    "loss_action": float(
                        policy_action_objective.detach().item()
                    ),
                    "loss_pgc_action": float(
                        counterfactual_action_loss.detach().item()
                    ),
                    "loss_pgc_all_action_monitor": float(
                        loss_action_post.detach().item()
                    ),
                    "loss_pgc_native_policy_distillation": float(
                        native_distillation_loss.detach().item()
                    ),
                    "loss_pgc_base_action_monitor": float(
                        loss_action_base.detach().item()
                    ),
                    "loss_pgc_verifier": float(verifier_loss.detach().item()),
                    "loss_pgc_goal_action_alignment": float(
                        alignment_loss.detach().item()
                    ),
                    "pgc_action_effective_weight": float(
                        self.policy_guard_action_weight
                    ),
                    "pgc_native_distillation_effective_weight": float(
                        self.policy_guard_native_distillation_weight
                    ),
                    "pgc_verifier_effective_weight": float(
                        self.policy_guard_verifier_weight
                    ),
                    "pgc_alignment_effective_weight": float(
                        self.policy_guard_alignment_weight
                    ),
                    "pgc_base_policy_frozen": 1.0,
                    "pgc_video_loss_optimization_weight": 0.0,
                }
            )
            loss_dict.update(detached_policy_guard_metrics(policy_guard_metrics))
            loss_dict.update(detached_policy_guard_metrics(verifier_metrics))
            loss_dict.update(detached_policy_guard_metrics(policy_action_metrics))

        if self.transition_contract_enabled:
            if self.transition_policy_distillation_enabled:
                if pred_action_teacher is None:
                    raise RuntimeError(
                        "TC-C policy distillation did not produce an M1 teacher."
                    )
                teacher_action = pred_action_teacher.detach()
                e_distill = self._compute_action_loss_per_sample(
                    pred_action=pred_action_post,
                    target_action=teacher_action,
                    action_is_pad=action_is_pad,
                )
                loss_policy_distillation = (
                    e_distill * action_weight
                ).mean()
                loss_total = (
                    loss_total
                    + self.transition_policy_distillation_weight
                    * loss_policy_distillation
                )
                loss_dict.update(
                    {
                        "loss_policy_distillation": float(
                            loss_policy_distillation.detach().item()
                        ),
                        "policy_distillation_effective_weight": float(
                            self.transition_policy_distillation_weight
                        ),
                        "policy_student_teacher_mse": float(
                            e_distill.detach().mean().item()
                        ),
                        "policy_teacher_action_norm": float(
                            teacher_action.float()
                            .norm(dim=-1)
                            .mean()
                            .item()
                        ),
                        "policy_teacher_frozen": float(
                            self.transition_freeze_m1_policy
                        ),
                    }
                )
            if z_language is None:
                raise RuntimeError("TC-C failed to produce z_language.")
            z_negative = None
            positive_grounding_state: dict[str, torch.Tensor] = {}
            counterfactual_grounding_state: dict[str, torch.Tensor] = {}
            if self.transition_contract_version >= 4:
                # TC-Full contracts must not identify the positive instruction
                # through the Video Expert's direct T1 text path.  Re-encode
                # positive and negative intents from the same language-neutral
                # current patch tokens; the deployment policy continues using
                # final-Video-hidden routing protected by M1 distillation.
                contract_visual_tokens = video_pre["tokens"].detach()
                if self.transition_use_state_conditioned_grounding:
                    (
                        _,
                        z_language,
                        _,
                        positive_grounding_state,
                    ) = self.encode_intended_transition(
                        video_tokens=contract_visual_tokens,
                        video_tokens_per_frame=int(
                            video_pre["meta"]["tokens_per_frame"]
                        ),
                        context=context,
                        full_context_mask=full_context_mask,
                        route_scale=1.0,
                        grounding_video_tokens=contract_visual_tokens,
                        return_grounding_state=True,
                    )
                else:
                    _, z_language, _ = self.encode_intended_transition(
                        video_tokens=contract_visual_tokens,
                        video_tokens_per_frame=int(
                            video_pre["meta"]["tokens_per_frame"]
                        ),
                        context=context,
                        full_context_mask=full_context_mask,
                        route_scale=1.0,
                    )
                negative_context = inputs["negative_context"]
                negative_context_mask = inputs["negative_context_mask"]
                if negative_context is None or negative_context_mask is None:
                    raise RuntimeError(
                        "TC-Full failed to receive counterfactual text context."
                    )
                if self.transition_use_state_conditioned_grounding:
                    (
                        _,
                        z_negative,
                        _,
                        counterfactual_grounding_state,
                    ) = self.encode_intended_transition(
                        video_tokens=contract_visual_tokens,
                        video_tokens_per_frame=int(
                            video_pre["meta"]["tokens_per_frame"]
                        ),
                        context=negative_context,
                        full_context_mask=negative_context_mask,
                        route_scale=1.0,
                        grounding_video_tokens=contract_visual_tokens,
                        return_grounding_state=True,
                    )
                else:
                    _, z_negative, _ = self.encode_intended_transition(
                        video_tokens=contract_visual_tokens,
                        video_tokens_per_frame=int(
                            video_pre["meta"]["tokens_per_frame"]
                        ),
                        context=negative_context,
                        full_context_mask=negative_context_mask,
                        route_scale=1.0,
                    )
            z_future = self.encode_realized_transition(
                clean_input_latents=input_latents,
                context=context,
                full_context_mask=full_context_mask,
                action=action,
                fuse_vae_embedding_in_latents=inputs[
                    "fuse_vae_embedding_in_latents"
                ],
            )
            loss_language_future, contract_metrics = (
                self.compute_transition_contract_loss(
                    z_language,
                    z_future,
                    group_ids=inputs["transition_task_id"],
                )
            )
            loss_action_future = z_language.sum() * 0.0
            z_action = None
            if self.transition_use_action_effect:
                z_action = self.encode_action_effect_transition(
                    current_video_hidden=video_pre["tokens"][:, : int(
                        video_pre["meta"]["tokens_per_frame"]
                    )].detach(),
                    action=action,
                    proprio=inputs["proprio_current"],
                    action_is_pad=action_is_pad,
                )
                loss_action_future, action_future_metrics = (
                    self.compute_action_future_contract_loss(
                        z_action,
                        z_future,
                        group_ids=inputs["transition_task_id"],
                    )
                )
                contract_metrics.update(action_future_metrics)
                loss_dict["action_transition_embedding_norm"] = float(
                    z_action.detach().float().norm(dim=-1).mean().item()
                )
            loss_contract = (
                loss_language_future
                + self.transition_action_future_weight * loss_action_future
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
                        loss_language_future.detach().item()
                    ),
                    "loss_action_future_contract": float(
                        loss_action_future.detach().item()
                    ),
                    "action_future_contract_weight": float(
                        self.transition_action_future_weight
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
            if self.transition_use_counterfactual_ranking:
                if z_negative is None:
                    raise RuntimeError(
                        "TC-Full counterfactual branch did not produce z_negative."
                    )
                loss_counterfactual, counterfactual_metrics = (
                    self.compute_counterfactual_ranking_loss(
                        z_language,
                        z_negative,
                        z_future,
                        valid_mask=inputs["negative_valid"],
                    )
                )
                effective_counterfactual_weight = (
                    self.transition_counterfactual_weight * contract_scale
                )
                loss_total = (
                    loss_total
                    + effective_counterfactual_weight * loss_counterfactual
                )
                loss_dict.update(
                    {
                        "loss_counterfactual_ranking": float(
                            loss_counterfactual.detach().item()
                        ),
                        "counterfactual_scale": float(contract_scale),
                        "counterfactual_effective_weight": float(
                            effective_counterfactual_weight
                        ),
                        "counterfactual_transition_embedding_norm": float(
                            z_negative.detach().float().norm(dim=-1).mean().item()
                        ),
                    }
                )
                loss_dict.update(detached_metrics(counterfactual_metrics))
            if self.transition_use_state_conditioned_grounding:
                transition_task_id = inputs["transition_task_id"]
                counterfactual_task_id = inputs["counterfactual_task_id"]
                negative_valid = inputs["negative_valid"]
                if (
                    transition_task_id is None
                    or counterfactual_task_id is None
                    or negative_valid is None
                ):
                    raise RuntimeError(
                        "TC-Full v6 requires source/target task IDs and a "
                        "valid counterfactual mask."
                    )
                loss_state_grounding, state_grounding_metrics = (
                    self.compute_state_conditioned_grounding_loss(
                        clean_input_latents=input_latents,
                        tokens_per_frame=int(
                            video_pre["meta"]["tokens_per_frame"]
                        ),
                        positive_state=positive_grounding_state,
                        counterfactual_state=(
                            counterfactual_grounding_state
                        ),
                        transition_task_ids=transition_task_id,
                        counterfactual_task_ids=counterfactual_task_id,
                        counterfactual_valid_mask=negative_valid,
                    )
                )
                effective_state_grounding_weight = (
                    self.transition_state_grounding_weight * contract_scale
                )
                loss_total = (
                    loss_total
                    + effective_state_grounding_weight * loss_state_grounding
                )
                loss_dict.update(
                    {
                        "loss_state_grounding": float(
                            loss_state_grounding.detach().item()
                        ),
                        "state_grounding_scale": float(contract_scale),
                        "state_grounding_effective_weight": float(
                            effective_state_grounding_weight
                        ),
                    }
                )
                loss_dict.update(detached_metrics(state_grounding_metrics))
            if self.transition_use_counterfactual_action_positive:
                if z_action is None:
                    raise RuntimeError(
                        "TC-Full v5+ requires an action-effect embedding."
                    )
                counterfactual_task_id = inputs["counterfactual_task_id"]
                if counterfactual_task_id is None:
                    raise RuntimeError(
                        "TC-Full v5+ did not receive counterfactual task IDs."
                    )
                prototype_bank = self.transition_contract_modules[
                    "counterfactual_action_prototypes"
                ]
                base_queries = self.action_expert.transition_queries.expand(
                    batch_size, -1, -1
                )
                # The correct-policy query is already grounded by L_action and
                # M1 distillation.  Store its detached task-level residual as a
                # positive policy target for other same-scene instructions.
                with torch.no_grad():
                    positive_queries, _, _ = self.encode_intended_transition(
                        video_tokens=final_video_hidden.detach(),
                        video_tokens_per_frame=int(
                            video_pre["meta"]["tokens_per_frame"]
                        ),
                        context=context,
                        full_context_mask=full_context_mask,
                        route_scale=1.0,
                        grounding_video_tokens=contract_visual_tokens,
                    )
                positive_query_residual = positive_queries - base_queries
                prototype_bank.update(
                    task_ids=inputs["transition_task_id"],
                    query_residuals=positive_query_residual,
                    action_effects=z_action,
                )
                positive_valid = inputs["negative_valid"] & (
                    prototype_bank.available_mask(counterfactual_task_id)
                )
                run_counterfactual_policy = bool(
                    contract_scale > 0.0 and positive_valid.any()
                )
                if run_counterfactual_policy:
                    negative_context = inputs["negative_context"]
                    negative_context_mask = inputs["negative_context_mask"]
                    if (
                        negative_context is None
                        or negative_context_mask is None
                    ):
                        raise RuntimeError(
                            "TC-Full v5+ requires alternate language context."
                        )
                    (
                        negative_full_context_mask,
                        negative_state_only_context_mask,
                    ) = self._build_context_masks(
                        full_context_mask=negative_context_mask,
                        language_context_len=inputs["language_context_len"],
                        has_proprio=inputs["has_proprio"],
                    )
                    (
                        pred_action_counterfactual,
                        counterfactual_query_residual,
                        counterfactual_action_intent,
                    ) = self._forward_counterfactual_action_positive_train(
                        latents=latents,
                        timestep_video=timestep_video,
                        action=action,
                        fuse_vae_embedding_in_latents=inputs[
                            "fuse_vae_embedding_in_latents"
                        ],
                        context=negative_context,
                        full_context_mask=negative_full_context_mask,
                        state_only_context_mask=(
                            negative_state_only_context_mask
                        ),
                        noisy_action=noisy_action,
                        timestep_action=timestep_action,
                    )
                    loss_counterfactual_action_positive, cap_metrics = (
                        prototype_bank.positive_loss(
                            counterfactual_task_ids=counterfactual_task_id,
                            query_residuals=counterfactual_query_residual,
                            action_intents=counterfactual_action_intent,
                            valid_mask=positive_valid,
                            query_weight=(
                                self.transition_counterfactual_action_query_weight
                            ),
                            action_effect_weight=(
                                self.transition_counterfactual_action_effect_weight
                            ),
                        )
                    )
                    e_counterfactual_source = (
                        self._compute_action_loss_per_sample(
                            pred_action=pred_action_counterfactual,
                            target_action=target_action,
                            action_is_pad=action_is_pad,
                        )
                    )
                    (
                        loss_counterfactual_action_separation,
                        separation_metrics,
                    ) = self.compute_counterfactual_action_separation_loss(
                        positive_error=e_post,
                        counterfactual_source_error=e_counterfactual_source,
                        valid_mask=positive_valid,
                    )
                else:
                    no_positive = torch.zeros_like(positive_valid)
                    (
                        loss_counterfactual_action_positive,
                        cap_metrics,
                    ) = prototype_bank.positive_loss(
                        counterfactual_task_ids=counterfactual_task_id,
                        query_residuals=positive_query_residual,
                        action_intents=z_action,
                        valid_mask=no_positive,
                        query_weight=(
                            self.transition_counterfactual_action_query_weight
                        ),
                        action_effect_weight=(
                            self.transition_counterfactual_action_effect_weight
                        ),
                    )
                    (
                        loss_counterfactual_action_separation,
                        separation_metrics,
                    ) = self.compute_counterfactual_action_separation_loss(
                        positive_error=e_post,
                        counterfactual_source_error=e_post,
                        valid_mask=no_positive,
                    )

                effective_cap_weight = (
                    self.transition_counterfactual_action_positive_weight
                    * contract_scale
                )
                effective_separation_weight = (
                    self.transition_counterfactual_action_separation_weight
                    * contract_scale
                )
                loss_total = (
                    loss_total
                    + effective_cap_weight
                    * loss_counterfactual_action_positive
                    + effective_separation_weight
                    * loss_counterfactual_action_separation
                )
                loss_dict.update(
                    {
                        "loss_counterfactual_action_positive": float(
                            loss_counterfactual_action_positive.detach().item()
                        ),
                        "loss_counterfactual_action_separation": float(
                            loss_counterfactual_action_separation.detach().item()
                        ),
                        "counterfactual_action_positive_effective_weight": float(
                            effective_cap_weight
                        ),
                        "counterfactual_action_separation_effective_weight": float(
                            effective_separation_weight
                        ),
                        "counterfactual_action_policy_branch_active": float(
                            run_counterfactual_policy
                        ),
                    }
                )
                loss_dict.update(detached_metrics(cap_metrics))
                loss_dict.update(detached_metrics(separation_metrics))

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
        action_only_pred = None
        if test_action_with_infer_action:
            if seed is None:
                raise ValueError("`test_action_with_infer_action=True` requires non-null `seed`.")
            action_only_pred = self.infer_action(
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
            )
            action_only_out = action_only_pred["action"]
        
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
        if self.policy_guard_enabled and test_action_with_infer_action:
            action_out = action_only_out
        if test_action_with_infer_action:
            if not torch.allclose(action_out, action_only_out, atol=1e-2, rtol=1e-2):
                max_abs_diff = (action_out - action_only_out).abs().max().item()
                logger.warning(
                    f"Action from infer_joint and infer_action differ with max abs diff {max_abs_diff:.6f}. "
                )

        result = {
            "video": self._decode_latents(latents_video, tiled=tiled),
            "action": action_out,
        }
        if self.policy_guard_enabled and action_only_pred is not None:
            for key in (
                "policy_guard_selected_counterfactual",
                "policy_guard_base_score",
                "policy_guard_counterfactual_score",
                "policy_guard_score_margin",
                "policy_guard_gate_mode",
                "policy_guard_base_action",
                "policy_guard_counterfactual_action",
            ):
                if key in action_only_pred:
                    result[key] = action_only_pred[key]
        return result

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
        policy_guard_latents_action = (
            latents_action.clone() if self.policy_guard_enabled else None
        )

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
        final_video_hidden = None
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
            prefill_result = self.mot.prefill_video_cache(
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
                return_final_hidden=self.policy_guard_enabled,
            )
            if self.policy_guard_enabled:
                if not isinstance(prefill_result, tuple):
                    raise RuntimeError("PGC Video prefill did not return hidden tokens.")
                video_kv_cache, final_video_hidden = prefill_result
            else:
                video_kv_cache = prefill_result

        policy_guard_goal_queries = None
        policy_guard_goal_embedding = None
        policy_guard_current_video_hidden = None
        if self.policy_guard_enabled:
            if final_video_hidden is None:
                raise RuntimeError("PGC inference requires final Video hidden tokens.")
            (
                policy_guard_goal_queries,
                policy_guard_goal_embedding,
                _,
            ) = self._encode_policy_guard_goal(
                final_video_hidden=final_video_hidden,
                video_tokens_per_frame=int(
                    video_pre["meta"]["tokens_per_frame"]
                ),
                context=context,
                context_mask=context_mask,
            )
            policy_guard_current_video_hidden = final_video_hidden[:, : int(
                video_pre["meta"]["tokens_per_frame"]
            )].detach()

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

            if self.policy_guard_enabled:
                if (
                    policy_guard_latents_action is None
                    or policy_guard_goal_queries is None
                ):
                    raise RuntimeError("PGC inference candidate was not initialized.")
                pred_action_counterfactual = (
                    self._forward_policy_guard_action_from_cache(
                        action_tokens=policy_guard_latents_action,
                        timestep_action=timestep_action,
                        context=context,
                        full_context_mask=context_mask,
                        state_only_context_mask=state_only_context_mask,
                        video_kv_cache=video_kv_cache,
                        video_seq_len=video_seq_len,
                        video_tokens_per_frame=int(
                            video_pre["meta"]["tokens_per_frame"]
                        ),
                        routed_goal_queries=policy_guard_goal_queries,
                    )
                )
                policy_guard_latents_action = self.infer_action_scheduler.step(
                    pred_action_counterfactual,
                    step_delta_action,
                    policy_guard_latents_action,
                )

        if not self.policy_guard_enabled:
            return {
                "action": latents_action[0].detach().to(
                    device="cpu", dtype=torch.float32
                ),
            }

        if (
            policy_guard_latents_action is None
            or policy_guard_goal_embedding is None
            or policy_guard_current_video_hidden is None
        ):
            raise RuntimeError("PGC inference did not produce both candidates.")
        verifier = self.policy_guard_modules["verifier"]
        base_logits, _, _ = verifier(
            current_video_hidden=policy_guard_current_video_hidden,
            goal_embedding=policy_guard_goal_embedding,
            action=latents_action,
        )
        counterfactual_logits, _, _ = verifier(
            current_video_hidden=policy_guard_current_video_hidden,
            goal_embedding=policy_guard_goal_embedding,
            action=policy_guard_latents_action,
        )
        base_score = torch.sigmoid(base_logits)
        counterfactual_score = torch.sigmoid(counterfactual_logits)
        selected_action, selected_counterfactual = (
            self._select_policy_guard_action(
                base_action=latents_action,
                counterfactual_action=policy_guard_latents_action,
                base_score=base_score,
                counterfactual_score=counterfactual_score,
            )
        )
        return {
            "action": selected_action[0].detach().to(
                device="cpu", dtype=torch.float32
            ),
            "policy_guard_selected_counterfactual": bool(
                selected_counterfactual[0].item()
            ),
            "policy_guard_base_score": float(base_score[0].item()),
            "policy_guard_counterfactual_score": float(
                counterfactual_score[0].item()
            ),
            "policy_guard_score_margin": float(
                (counterfactual_score[0] - base_score[0]).item()
            ),
            "policy_guard_gate_mode": self.policy_guard_gate_mode,
            "policy_guard_base_action": latents_action[0].detach().to(
                device="cpu", dtype=torch.float32
            ),
            "policy_guard_counterfactual_action": (
                policy_guard_latents_action[0]
                .detach()
                .to(device="cpu", dtype=torch.float32)
            ),
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
            "use_cf_ranking": bool(
                self.transition_use_counterfactual_ranking
            ),
            "use_action_effect": bool(self.transition_use_action_effect),
            "use_cf_action_positive": bool(
                self.transition_use_counterfactual_action_positive
            ),
            "use_state_conditioned_grounding": bool(
                self.transition_use_state_conditioned_grounding
            ),
            "action_conditioned_video": False,
            "router_visual_source": (
                "video_expert_final_hidden_with_current_state_target_bias"
                if self.transition_use_state_conditioned_grounding
                else "video_expert_final_hidden"
            ),
            "contract_visual_source": (
                "language_neutral_video_patch_tokens"
                if self.transition_contract_version >= 4
                else "video_expert_final_hidden"
            ),
            "action_future_weight": self.transition_action_future_weight,
            "counterfactual_weight": self.transition_counterfactual_weight,
            "counterfactual_margin": self.transition_counterfactual_margin,
            "counterfactual_action_positive_weight": (
                self.transition_counterfactual_action_positive_weight
            ),
            "counterfactual_action_query_weight": (
                self.transition_counterfactual_action_query_weight
            ),
            "counterfactual_action_effect_weight": (
                self.transition_counterfactual_action_effect_weight
            ),
            "counterfactual_action_separation_weight": (
                self.transition_counterfactual_action_separation_weight
            ),
            "counterfactual_action_separation_margin": (
                self.transition_counterfactual_action_separation_margin
            ),
            "state_grounding_weight": self.transition_state_grounding_weight,
            "state_grounding_correct_weight": (
                self.transition_state_grounding_correct_weight
            ),
            "state_grounding_counterfactual_weight": (
                self.transition_state_grounding_counterfactual_weight
            ),
            "state_grounding_separation_weight": (
                self.transition_state_grounding_separation_weight
            ),
            "state_grounding_overlap_margin": (
                self.transition_state_grounding_overlap_margin
            ),
            "state_grounding_router_bias": (
                self.transition_state_grounding_router_bias
            ),
            "state_grounding_teacher": (
                "clean_latent_local_interaction_change"
                if self.transition_use_state_conditioned_grounding
                else None
            ),
            "state_grounding_hidden_dim": (
                self.transition_state_grounding_hidden_dim
            ),
            "policy_recovery_ratio": self.transition_policy_recovery_ratio,
            "router_ramp_ratio": self.transition_router_ramp_ratio,
            "policy_recovery_blend": (
                "disabled_frozen_teacher_distillation"
                if self.transition_contract_version >= 3
                else "action_flow_velocity"
            ),
            "policy_recovery_source": "joint_mot_posterior",
            "freeze_m1_during_recovery": (
                self.transition_freeze_m1_during_recovery
            ),
            "policy_distillation_enabled": (
                self.transition_policy_distillation_enabled
            ),
            "policy_distillation_weight": (
                self.transition_policy_distillation_weight
            ),
            "policy_teacher_source": "frozen_joint_mot_posterior",
            "freeze_m1_policy": self.transition_freeze_m1_policy,
            "student_policy_path": (
                "pure_router_from_step_zero"
                if self.transition_contract_version >= 3
                else "scheduled_m1_to_router_recovery"
            ),
            "policy_protection": (
                "requires_grad_false_and_optimizer_exclusion"
                if self.transition_freeze_m1_policy
                else "recovery_window_only"
            ),
            "policy_init_checkpoint": self.transition_policy_init_checkpoint,
        }

    def _transition_contract_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            name: value.detach().to(device="cpu")
            for name, value in self.transition_contract_modules.state_dict().items()
        }

    def _policy_guard_metadata(self) -> dict[str, Any]:
        if not self.policy_guard_enabled or self.policy_guard_action_expert is None:
            raise RuntimeError("PGC metadata requested while policy guard is disabled.")
        is_v2 = self.policy_guard_version >= 2
        return {
            "architecture": "pgc_fastwam",
            "policy_guard_version": self.policy_guard_version,
            "base_policy": "frozen_released_fastwam",
            "base_action_interface": "query_free_joint_mot",
            "counterfactual_policy": (
                "base_equivalent_visual_residual_action_expert_lora"
                if is_v2
                else "independent_action_expert_lora"
            ),
            "counterfactual_tuning": "lora",
            "counterfactual_action_interface": (
                "query_free_raw_current_visual"
                if is_v2
                else "latent_query_goal_bottleneck"
            ),
            "goal_injection": (
                "zero_initialized_action_token_residual"
                if is_v2
                else "latent_action_query_replacement"
            ),
            "native_policy_teacher": (
                "frozen_base_velocity_same_noise_timestep"
                if is_v2
                else None
            ),
            "native_distillation_weight": (
                self.policy_guard_native_distillation_weight
            ),
            "goal_residual_scale": self.policy_guard_goal_residual_scale,
            "lora_rank": int(self.lora_config["rank"]),
            "lora_alpha": float(self.lora_config["alpha"]),
            "lora_dropout": float(self.lora_config["dropout"]),
            "lora_target_modules": list(self.lora_config["target_modules"]),
            "num_action_queries": int(self.policy_guard_num_action_queries),
            "query_rope_offset": (
                int(self.policy_guard_query_rope_offset)
                if not is_v2
                else None
            ),
            "goal_graph_tokens": int(
                self.policy_guard_modules["goal_graph"].num_goal_tokens
            ),
            "gate_mode": self.policy_guard_gate_mode,
            "gate_threshold": self.policy_guard_gate_threshold,
            "min_counterfactual_score": (
                self.policy_guard_min_counterfactual_score
            ),
            "verifier_action_mse_temperature": (
                self.policy_guard_verifier_action_mse_temperature
            ),
            "require_direct_counterfactual_actions": (
                self.policy_guard_require_direct_counterfactual_actions
            ),
            "policy_protection": "immutable_base_plus_conservative_hard_gate",
            "representation_supervision": "direct_goal_action_alignment",
            "verifier_margin_space": "probability" if is_v2 else "logit",
        }

    @staticmethod
    def _cpu_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
        return {
            name: value.detach().to(device="cpu")
            for name, value in module.state_dict().items()
        }

    def _policy_guard_action_adapter_state_dict(
        self,
    ) -> dict[str, torch.Tensor]:
        if self.policy_guard_action_expert is None:
            raise RuntimeError("PGC Action Expert is unavailable.")
        adapter_ids = self._policy_guard_action_adapter_parameter_ids()
        state = self.policy_guard_action_expert.state_dict()
        names = {
            name
            for name, parameter in self.policy_guard_action_expert.named_parameters()
            if id(parameter) in adapter_ids
        }
        return {
            name: state[name].detach().to(device="cpu")
            for name in sorted(names)
        }

    def _sync_policy_guard_action_from_base(self) -> None:
        if not self.policy_guard_enabled or self.policy_guard_action_expert is None:
            return
        incompatible = self.policy_guard_action_expert.load_state_dict(
            self.action_expert.state_dict(), strict=False
        )
        disallowed_missing = [
            key
            for key in incompatible.missing_keys
            if key != "latent_action_queries"
            and not is_lora_parameter_name(key)
        ]
        if disallowed_missing or incompatible.unexpected_keys:
            raise ValueError(
                "Could not initialize the PGC Action Expert from the protected "
                f"base: missing={disallowed_missing}, "
                f"unexpected={list(incompatible.unexpected_keys)}."
            )
        logger.info(
            "Initialized independent PGC Action Expert from frozen base "
            "(new_query_tensors=%d).",
            len(incompatible.missing_keys),
        )

    def save_checkpoint(self, path, optimizer=None, step=None):
        if self.policy_guard_enabled:
            if self.policy_guard_action_expert is None:
                raise RuntimeError("PGC Action Expert is unavailable for saving.")
            if not self.lora_enabled:
                raise ValueError(
                    "PGC checkpoints require action-only LoRA; full "
                    "Action-Expert checkpoints are disabled."
                )
            if self.policy_guard_legacy_full_loaded:
                raise ValueError(
                    "Cannot convert a legacy full-PGC checkpoint into a "
                    "partial LoRA checkpoint."
                )
            if not self.policy_guard_base_checkpoint:
                raise ValueError(
                    "PGC checkpoint cannot be saved before a protected base "
                    "checkpoint has been loaded."
                )
            action_adapter = self._policy_guard_action_adapter_state_dict()
            if not action_adapter:
                raise ValueError("PGC checkpoint has no Action-Expert adapter tensors.")
            payload = {
                "format": f"fastwam_policy_guard_v{self.policy_guard_version}",
                "base_checkpoint": self.policy_guard_base_checkpoint,
                "counterfactual_action_adapter": action_adapter,
                "counterfactual_lora_config": dict(self.lora_config),
                "policy_guard": self._cpu_state_dict(
                    self.policy_guard_modules
                ),
                "architecture_metadata": self._policy_guard_metadata(),
                "step": step,
                "torch_dtype": str(self.torch_dtype),
            }
            if optimizer is not None:
                payload["optimizer"] = optimizer.state_dict()
            torch.save(payload, path)
            return

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

    def _validate_transition_checkpoint_version(
        self, saved_version: Any
    ) -> int:
        try:
            saved_version = int(saved_version)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"TC checkpoint has invalid version: {saved_version!r}."
            ) from exc
        if saved_version == self.transition_contract_version:
            return saved_version
        # Stage 2 is initialized from the policy-protected Stage-1 v3
        # checkpoint.  Only the new training-time ActionEffectEncoder is
        # absent; Router, intent/outcome projections, LoRA, and M1 policy are
        # restored exactly.
        if self.transition_contract_version == 4 and saved_version == 3:
            return saved_version
        # v5 adds a training-only online prototype bank and new losses, but no
        # persistent deployment parameters.  A protected v4 adapter therefore
        # migrates exactly.
        if self.transition_contract_version == 5 and saved_version == 4:
            return saved_version
        # v6 adds a small deployment grounder plus a training-only appearance
        # prototype bank.  All protected v5 policy tensors remain compatible.
        if self.transition_contract_version == 6 and saved_version == 5:
            return saved_version
        raise ValueError(
            "TC checkpoint version mismatch: "
            f"checkpoint={saved_version}, model={self.transition_contract_version}."
        )

    def _load_transition_contract_checkpoint_state(
        self,
        transition_state: dict[str, torch.Tensor],
        *,
        saved_version: int,
    ) -> None:
        if saved_version == self.transition_contract_version:
            self.transition_contract_modules.load_state_dict(
                transition_state, strict=True
            )
            return
        if self.transition_contract_version == 5 and saved_version == 4:
            self.transition_contract_modules.load_state_dict(
                transition_state,
                strict=True,
            )
            logger.info(
                "Migrated TC checkpoint v4 -> v5; initialized the empty "
                "training-only counterfactual action prototype bank."
            )
            return
        if self.transition_contract_version == 6 and saved_version == 5:
            incompatible = self.transition_contract_modules.load_state_dict(
                transition_state, strict=False
            )
            unexpected = list(incompatible.unexpected_keys)
            disallowed_missing = [
                key
                for key in incompatible.missing_keys
                if not key.startswith("state_target_grounder.")
            ]
            if unexpected or disallowed_missing:
                raise ValueError(
                    "Invalid TC v5 -> v6 state-grounding migration: "
                    f"missing={disallowed_missing}, unexpected={unexpected}."
                )
            logger.info(
                "Migrated TC checkpoint v5 -> v6; initialized %d new "
                "StateConditionedTargetGrounder tensors and an empty "
                "training-only appearance prototype bank.",
                len(incompatible.missing_keys),
            )
            return
        incompatible = self.transition_contract_modules.load_state_dict(
            transition_state, strict=False
        )
        unexpected = list(incompatible.unexpected_keys)
        disallowed_missing = [
            key
            for key in incompatible.missing_keys
            if not key.startswith("action_effect_encoder.")
        ]
        if unexpected or disallowed_missing:
            raise ValueError(
                "Invalid TC v3 -> TC-Full v4 migration state: "
                f"missing={disallowed_missing}, unexpected={unexpected}."
            )
        logger.info(
            "Migrated TC checkpoint v%d -> v%d; initialized %d new "
            "ActionEffectEncoder tensors.",
            saved_version,
            self.transition_contract_version,
            len(incompatible.missing_keys),
        )

    def _load_lora_adapter(self, path: str, payload: dict, optimizer=None):
        adapter_source_path = str(Path(path).expanduser().resolve())
        transition_state = payload.get("transition_contract")
        if transition_state is not None:
            if not self.transition_contract_enabled:
                raise ValueError(
                    "This adapter contains TC-FastWAM weights, but the current "
                    "model has `transition_contract.enabled=false`. Enable the "
                    "matching TC config before loading it."
                )
            metadata = payload.get("architecture_metadata") or {}
            saved_version = metadata.get("transition_contract_version")
            saved_version = self._validate_transition_checkpoint_version(
                saved_version
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
            self._load_transition_contract_checkpoint_state(
                transition_state,
                saved_version=saved_version,
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
        elif transition_state is not None:
            metadata = payload.get("architecture_metadata") or {}
            self.transition_policy_init_checkpoint = metadata.get(
                "policy_init_checkpoint"
            )
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

    def _load_policy_guard_checkpoint(
        self, path: str, payload: dict, optimizer=None
    ):
        if not self.policy_guard_enabled or self.policy_guard_action_expert is None:
            raise ValueError(
                "This checkpoint contains PGC weights; enable the matching "
                "`policy_guard` model config before loading it."
            )
        metadata = payload.get("architecture_metadata") or {}
        saved_policy_guard_version = int(
            metadata.get("policy_guard_version", -1)
        )
        if saved_policy_guard_version != int(self.policy_guard_version):
            raise ValueError(
                "PGC checkpoint version mismatch: "
                f"checkpoint={metadata.get('policy_guard_version')}, "
                f"model={self.policy_guard_version}."
            )
        expected_format = f"fastwam_policy_guard_v{saved_policy_guard_version}"
        if payload.get("format") != expected_format:
            raise ValueError(
                "PGC checkpoint format/version mismatch: "
                f"format={payload.get('format')!r}, expected={expected_format!r}."
            )
        base_checkpoint = payload.get("base_checkpoint")
        if not base_checkpoint:
            raise ValueError("PGC checkpoint is missing `base_checkpoint`.")
        resolved_base = self._resolve_adapter_base_checkpoint(
            path, str(base_checkpoint)
        )
        if Path(resolved_base).resolve() == Path(path).expanduser().resolve():
            raise ValueError("PGC checkpoint cannot name itself as its base.")

        action_adapter = payload.get("counterfactual_action_adapter")
        legacy_action_state = payload.get("counterfactual_action_expert")
        if action_adapter is not None:
            saved_lora_config = payload.get("counterfactual_lora_config")
            if not isinstance(saved_lora_config, dict):
                raise ValueError(
                    "PGC LoRA checkpoint is missing "
                    "`counterfactual_lora_config`."
                )
            self.configure_lora(saved_lora_config)
        elif legacy_action_state is None:
            raise ValueError(
                "PGC checkpoint has neither a counterfactual Action-Expert "
                "adapter nor a legacy full Action Expert."
            )

        base_payload = self.load_checkpoint(resolved_base, optimizer=None)
        if base_payload.get("format") in {
            "fastwam_policy_guard_v1",
            "fastwam_policy_guard_v2",
        }:
            raise ValueError("Nested PGC checkpoints are not supported as bases.")

        guard_state = payload.get("policy_guard")
        if not isinstance(guard_state, dict) or not guard_state:
            raise ValueError("PGC checkpoint has no Goal-Graph/Verifier state.")

        if action_adapter is not None:
            if not isinstance(action_adapter, dict) or not action_adapter:
                raise ValueError(
                    "PGC checkpoint has an empty counterfactual Action-Expert "
                    "adapter."
                )
            current_state = self.policy_guard_action_expert.state_dict()
            expected_adapter = set(
                self._policy_guard_action_adapter_state_dict()
            )
            saved_adapter = set(action_adapter)
            missing = sorted(expected_adapter - saved_adapter)
            unexpected = sorted(saved_adapter - expected_adapter)
            if missing or unexpected:
                raise ValueError(
                    "PGC Action-Expert adapter key mismatch: "
                    f"missing={missing[:20]}, unexpected={unexpected[:20]}."
                )
            shape_mismatches = {
                name: (tuple(value.shape), tuple(current_state[name].shape))
                for name, value in action_adapter.items()
                if tuple(value.shape) != tuple(current_state[name].shape)
            }
            if shape_mismatches:
                raise ValueError(
                    "PGC Action-Expert adapter shape mismatches: "
                    f"{shape_mismatches}"
                )
            incompatible = self.policy_guard_action_expert.load_state_dict(
                action_adapter, strict=False
            )
            if incompatible.unexpected_keys:
                raise ValueError(
                    "PGC Action-Expert adapter contains unexpected tensors: "
                    f"{list(incompatible.unexpected_keys)}"
                )
            action_tensor_count = len(action_adapter)
            self.policy_guard_legacy_full_loaded = False
        else:
            if not isinstance(legacy_action_state, dict) or not legacy_action_state:
                raise ValueError("Legacy PGC checkpoint has no full Action Expert.")
            incompatible = self.policy_guard_action_expert.load_state_dict(
                legacy_action_state, strict=not self.lora_enabled
            )
            if self.lora_enabled:
                disallowed_missing = [
                    key
                    for key in incompatible.missing_keys
                    if not is_lora_parameter_name(key)
                ]
                if disallowed_missing or incompatible.unexpected_keys:
                    raise ValueError(
                        "Legacy PGC Action Expert is incompatible with the "
                        "current model: "
                        f"missing={disallowed_missing}, "
                        f"unexpected={list(incompatible.unexpected_keys)}."
                    )
            action_tensor_count = len(legacy_action_state)
            self.policy_guard_legacy_full_loaded = True
            logger.warning(
                "Loaded a legacy full-PGC Action Expert. It is supported for "
                "evaluation only; new PGC training requires action-only LoRA."
            )
        self.policy_guard_modules.load_state_dict(guard_state, strict=True)
        self.policy_guard_base_checkpoint = resolved_base
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        logger.info(
            "Loaded PGC checkpoint from %s (base=%s action_tensors=%d "
            "guard_tensors=%d).",
            path,
            resolved_base,
            action_tensor_count,
            len(guard_state),
        )
        return payload

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu")
        if payload.get("format") in {
            "fastwam_policy_guard_v1",
            "fastwam_policy_guard_v2",
        }:
            return self._load_policy_guard_checkpoint(
                str(path), payload, optimizer=optimizer
            )
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
            saved_version = self._validate_transition_checkpoint_version(
                saved_version
            )
            self._load_transition_contract_checkpoint_state(
                transition_state,
                saved_version=saved_version,
            )
        elif self.transition_contract_enabled:
            logger.info(
                "Checkpoint has no Transition Contract tensors; keeping standard "
                "initialization for backward-compatible B0/M1 loading."
            )

        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        resolved_checkpoint = str(Path(path).expanduser().resolve())
        self.lora_base_checkpoint = resolved_checkpoint
        if self.policy_guard_enabled:
            self._sync_policy_guard_action_from_base()
            self.policy_guard_base_checkpoint = resolved_checkpoint
            self.policy_guard_legacy_full_loaded = False
        return payload

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)
