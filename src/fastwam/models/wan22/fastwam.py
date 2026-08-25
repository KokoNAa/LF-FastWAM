import copy
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from fastwam.datasets.pgc_libero import PGC_ENTITY_RELATION_ARRAY_NAMES
from fastwam.utils.logging_config import get_logger

from .action_dit import ActionDiT
from .entity_relation_affordance import (
    ERAFLossWeights,
    EntityRelationAffordanceField,
    entity_relation_affordance_loss,
)
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
    BoundedActionVelocityResidual,
    ClauseSemanticRetentionResidual,
    ERAFActionContextInjector,
    GoalActionAlignmentLoss,
    GoalGraphEncoder,
    GoalResidualAdapter,
    HardRoutedERAFPhaseServo,
    LanguageVisualTargetBinder,
    PairwiseActionAdvantageVerifier,
    PhaseCompatibleERAFWaypointAdapter,
    PhaseConditionedERAFActionBridge,
    PhaseConditionedERAFGeometryActionAdapter,
    PhaseSpecificERAFExpertResidualAdapter,
    RolloutAlignedActionProposal,
    SpatialObjectTokenTargetBinder,
    detached_policy_guard_metrics,
    spatial_mask_to_patch_distribution,
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


PGC_PHASE_SAFE_MEMORY_CLAUSE_LABEL_NAMES = (
    "phase_safe_memory_previous_state_ids",
    "phase_safe_memory_target_state_ids",
    "phase_safe_memory_state_valid",
)
PGC_PHASE_SAFE_MEMORY_SAMPLE_LABEL_NAMES = (
    "phase_safe_memory_execution_target",
    "phase_safe_memory_execution_valid",
    "phase_safe_memory_stage_id",
    "phase_safe_memory_stage_valid",
)
PGC_PHASE_SAFE_MEMORY_LABEL_NAMES = (
    *PGC_PHASE_SAFE_MEMORY_CLAUSE_LABEL_NAMES,
    *PGC_PHASE_SAFE_MEMORY_SAMPLE_LABEL_NAMES,
)
PGC_ERAF_ACTION_BRIDGE_MODULE_NAMES = (
    "base_query_projection",
    "relation_attention",
    "query_delta_projection",
    "embedding_delta_projection",
)


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
        eraf_config = dict(guard_config.get("entity_relation_grounding", {}) or {})
        self.policy_guard_eraf_training_stage = str(
            eraf_config.get("training_stage", "grounding")
        ).strip().lower()
        self.policy_guard_eraf_initialization_contract = str(
            eraf_config.get(
                "initialization_contract", "exact_pgc_v5_sidecars"
            )
        ).strip()
        self.policy_guard_eraf_hidden_dim = int(
            eraf_config.get("hidden_dim", 256)
        )
        self.policy_guard_eraf_num_heads = int(
            eraf_config.get("num_heads", 8)
        )
        self.policy_guard_eraf_max_clauses = int(
            eraf_config.get("max_clauses", 4)
        )
        self.policy_guard_eraf_camera_count = int(
            eraf_config.get("camera_count", 2)
        )
        self.policy_guard_eraf_visual_aspect_ratio = float(
            eraf_config.get("visual_aspect_ratio", 2.0)
        )
        self.policy_guard_eraf_temperature = float(
            eraf_config.get("temperature", 0.07)
        )
        self.policy_guard_eraf_entity_only = bool(
            eraf_config.get("entity_only", False)
        )
        self.policy_guard_eraf_use_anchors = bool(
            eraf_config.get("use_anchors", True)
        )
        self.policy_guard_eraf_learning_rate = float(
            eraf_config.get("learning_rate", 2.0e-5)
        )
        self.policy_guard_eraf_grounding_aux_weight = float(
            eraf_config.get("grounding_aux_weight", 0.25)
        )
        # V9.14 retains only a monotonic completed-clause bitset across
        # replans.  ``action_joint_training`` is intentionally separate from
        # that deployment contract so a later verifier stage can preserve the
        # same memory semantics without reopening the ERAF action bridge.
        self.policy_guard_eraf_completion_only_memory = bool(
            eraf_config.get("completion_only_memory", False)
        )
        self.policy_guard_eraf_action_joint_training = bool(
            eraf_config.get("action_joint_training", False)
        )
        self.policy_guard_eraf_action_grounding_hidden_dim = int(
            eraf_config.get("action_grounding_hidden_dim", 256)
        )
        self.policy_guard_eraf_action_grounding_num_heads = int(
            eraf_config.get("action_grounding_num_heads", 8)
        )
        self.policy_guard_eraf_action_grounding_learning_rate = float(
            eraf_config.get("action_grounding_learning_rate", 1.0e-4)
        )
        self.policy_guard_eraf_action_causal_ranking_weight = float(
            eraf_config.get("action_causal_ranking_weight", 1.0)
        )
        self.policy_guard_eraf_action_causal_margin = float(
            eraf_config.get("action_causal_margin", 0.01)
        )
        self.policy_guard_eraf_action_geometry_hidden_dim = int(
            eraf_config.get("action_geometry_hidden_dim", 256)
        )
        self.policy_guard_eraf_action_geometry_learning_rate = float(
            eraf_config.get("action_geometry_learning_rate", 2.0e-5)
        )
        self.policy_guard_eraf_action_geometry_residual_max_abs = float(
            eraf_config.get("action_geometry_residual_max_abs", 0.25)
        )
        self.policy_guard_eraf_action_phase_residual_imitation_weight = float(
            eraf_config.get("action_phase_residual_imitation_weight", 2.0)
        )
        self.policy_guard_eraf_action_phase_direction_weight = float(
            eraf_config.get("action_phase_direction_weight", 0.5)
        )
        self.policy_guard_eraf_action_phase_approach_weight = float(
            eraf_config.get("action_phase_approach_weight", 1.0)
        )
        self.policy_guard_eraf_action_phase_transport_weight = float(
            eraf_config.get("action_phase_transport_weight", 2.0)
        )
        self.policy_guard_eraf_action_phase_release_weight = float(
            eraf_config.get("action_phase_release_weight", 3.0)
        )
        self.policy_guard_eraf_action_phase_direction_min_norm = float(
            eraf_config.get("action_phase_direction_min_norm", 1.0e-3)
        )
        self.policy_guard_eraf_action_servo_frame_weight = float(
            eraf_config.get("action_servo_frame_weight", 0.01)
        )
        self.policy_guard_eraf_action_eef_scale = tuple(
            float(value)
            for value in eraf_config.get("action_eef_scale", (1.0, 1.0, 1.0))
        )
        self.policy_guard_eraf_action_eef_bias = tuple(
            float(value)
            for value in eraf_config.get("action_eef_bias", (0.0, 0.0, 0.0))
        )
        self.policy_guard_eraf_action_waypoint_compatibility_weight = float(
            eraf_config.get("action_waypoint_compatibility_weight", 1.0)
        )
        self.policy_guard_eraf_action_waypoint_imitation_weight = float(
            eraf_config.get("action_waypoint_imitation_weight", 2.0)
        )
        self.policy_guard_eraf_action_waypoint_direction_weight = float(
            eraf_config.get("action_waypoint_direction_weight", 0.5)
        )
        self.policy_guard_eraf_action_waypoint_zero_weight = float(
            eraf_config.get("action_waypoint_zero_weight", 1.0)
        )
        self.policy_guard_eraf_action_waypoint_min_cosine = float(
            eraf_config.get("action_waypoint_min_cosine", 0.20)
        )
        self.policy_guard_eraf_action_waypoint_tangent_max_ratio = float(
            eraf_config.get("action_waypoint_tangent_max_ratio", 0.75)
        )
        self.policy_guard_eraf_action_expert_imitation_weight = float(
            eraf_config.get("action_expert_imitation_weight", 2.0)
        )
        self.policy_guard_eraf_action_expert_direction_weight = float(
            eraf_config.get("action_expert_direction_weight", 0.5)
        )
        self.policy_guard_eraf_action_expert_deployed_weight = float(
            eraf_config.get("action_expert_deployed_weight", 1.0)
        )
        self.policy_guard_eraf_action_expert_distillation_weight = float(
            eraf_config.get("action_expert_distillation_weight", 0.5)
        )
        self.policy_guard_eraf_action_expert_native_zero_weight = float(
            eraf_config.get("action_expert_native_zero_weight", 1.0)
        )
        self.policy_guard_eraf_action_clause_ranking_weight = float(
            eraf_config.get("action_clause_ranking_weight", 4.0)
        )
        self.policy_guard_eraf_action_clause_ranking_margin = float(
            eraf_config.get("action_clause_ranking_margin", 0.02)
        )
        self.policy_guard_eraf_action_clause_teacher_weight = float(
            eraf_config.get("action_clause_teacher_weight", 4.0)
        )
        self.policy_guard_eraf_action_clause_alignment_guard_weight = float(
            eraf_config.get("action_clause_alignment_guard_weight", 8.0)
        )
        self.policy_guard_eraf_action_clause_wrong_suppression_weight = float(
            eraf_config.get("action_clause_wrong_suppression_weight", 4.0)
        )
        self.policy_guard_eraf_expert_lora_world_language_weight = float(
            eraf_config.get("expert_lora_world_language_weight", 0.10)
        )
        self.policy_guard_eraf_expert_lora_world_language_margin = float(
            eraf_config.get("expert_lora_world_language_margin", 0.01)
        )
        self.policy_guard_eraf_expert_lora_native_action_weight = float(
            eraf_config.get("expert_lora_native_action_weight", 1.0)
        )
        self.policy_guard_eraf_expert_lora_counterfactual_action_weight = float(
            eraf_config.get("expert_lora_counterfactual_action_weight", 1.0)
        )
        self.policy_guard_eraf_expert_lora_regularization_weight = float(
            eraf_config.get("expert_lora_regularization_weight", 1.0e-6)
        )
        self.policy_guard_eraf_grounding_objective_version = int(
            eraf_config.get("grounding_objective_version", 1)
        )
        gate_aligned_grounding = (
            self.policy_guard_eraf_grounding_objective_version >= 2
        )
        assignment_grounding = (
            self.policy_guard_eraf_grounding_objective_version >= 3
        )
        role_adapter_grounding = (
            self.policy_guard_eraf_grounding_objective_version >= 4
        )
        structured_role_grounding = (
            self.policy_guard_eraf_grounding_objective_version >= 5
        )
        balanced_role_grounding = (
            self.policy_guard_eraf_grounding_objective_version >= 6
        )
        clause_activation_grounding = (
            self.policy_guard_eraf_grounding_objective_version >= 9
        )
        view_scheduler_grounding = (
            self.policy_guard_eraf_grounding_objective_version >= 10
        )
        clause_tuple_grounding = (
            self.policy_guard_eraf_grounding_objective_version >= 12
        )
        closed_loop_rebinding_grounding = (
            self.policy_guard_eraf_grounding_objective_version == 13
        )
        phase_safe_memory_grounding = (
            self.policy_guard_eraf_grounding_objective_version >= 14
        )
        self.policy_guard_eraf_role_adapter_hidden_dim = int(
            eraf_config.get("role_adapter_hidden_dim", 256)
        )
        self.policy_guard_eraf_structured_role_adapter_hidden_dim = int(
            eraf_config.get("structured_role_adapter_hidden_dim", 256)
        )
        self.policy_guard_eraf_balanced_role_adapter_hidden_dim = int(
            eraf_config.get("balanced_role_adapter_hidden_dim", 256)
        )
        self.policy_guard_eraf_clause_activation_adapter_hidden_dim = int(
            eraf_config.get("clause_activation_adapter_hidden_dim", 256)
        )
        self.policy_guard_eraf_clause_activation_residual_max_abs = float(
            eraf_config.get("clause_activation_residual_max_abs", 4.0)
        )
        self.policy_guard_eraf_view_fusion_adapter_hidden_dim = int(
            eraf_config.get("view_fusion_adapter_hidden_dim", 256)
        )
        self.policy_guard_eraf_view_fusion_residual_max_abs = float(
            eraf_config.get("view_fusion_residual_max_abs", 4.0)
        )
        self.policy_guard_eraf_clause_scheduler_hidden_dim = int(
            eraf_config.get("clause_scheduler_hidden_dim", 256)
        )
        self.policy_guard_eraf_clause_scheduler_residual_max_abs = float(
            eraf_config.get("clause_scheduler_residual_max_abs", 1.0)
        )
        self.policy_guard_eraf_closed_loop_rebinding_hidden_dim = int(
            eraf_config.get("closed_loop_rebinding_hidden_dim", 256)
        )
        self.policy_guard_eraf_closed_loop_query_residual_max_abs = float(
            eraf_config.get("closed_loop_query_residual_max_abs", 1.0)
        )
        self.policy_guard_eraf_closed_loop_state_residual_max_abs = float(
            eraf_config.get("closed_loop_state_residual_max_abs", 2.0)
        )
        self.policy_guard_eraf_phase_safe_memory_hidden_dim = int(
            eraf_config.get("phase_safe_memory_hidden_dim", 256)
        )
        self.policy_guard_eraf_phase_safe_memory_state_count = int(
            eraf_config.get("phase_safe_memory_state_count", 4)
        )
        self.policy_guard_eraf_phase_safe_memory_routing_residual_max_abs = float(
            eraf_config.get(
                "phase_safe_memory_routing_residual_max_abs", 1.0
            )
        )
        self.policy_guard_eraf_loss_weights = ERAFLossWeights(
            objective_version=(
                self.policy_guard_eraf_grounding_objective_version
            ),
            mask=float(eraf_config.get("mask_weight", 1.0)),
            attention_mask=float(
                eraf_config.get(
                    "attention_mask_weight",
                    2.0 if gate_aligned_grounding else 0.0,
                )
            ),
            entity=float(eraf_config.get("entity_weight", 1.0)),
            relation=float(eraf_config.get("relation_weight", 1.0)),
            anchor=float(eraf_config.get("anchor_weight", 1.0)),
            position=float(eraf_config.get("position_weight", 0.5)),
            role_swap=float(
                eraf_config.get(
                    "role_swap_weight", 2.0 if gate_aligned_grounding else 0.5
                )
            ),
            role_overlap=float(
                eraf_config.get(
                    "role_overlap_weight",
                    1.0 if gate_aligned_grounding else 0.0,
                )
            ),
            role_swap_margin=float(
                eraf_config.get("role_swap_margin", 0.20)
            ),
            role_assignment=float(
                eraf_config.get(
                    "role_assignment_weight",
                    (
                        1.0
                        if role_adapter_grounding
                        else (4.0 if assignment_grounding else 0.0)
                    ),
                )
            ),
            role_assignment_temperature=float(
                eraf_config.get("role_assignment_temperature", 0.10)
            ),
            role_assignment_hard_weight=float(
                eraf_config.get(
                    "role_assignment_hard_weight",
                    (
                        0.5
                        if role_adapter_grounding
                        else (2.0 if assignment_grounding else 0.0)
                    ),
                )
            ),
            role_attention_preservation=float(
                eraf_config.get(
                    "role_attention_preservation_weight",
                    (
                        5.0
                        if balanced_role_grounding
                        else (1.0 if role_adapter_grounding else 0.0)
                    ),
                )
            ),
            role_position_preservation=float(
                eraf_config.get(
                    "role_position_preservation_weight",
                    (
                        2.0
                        if balanced_role_grounding
                        else (0.5 if role_adapter_grounding else 0.0)
                    ),
                )
            ),
            role_anchor_preservation=float(
                eraf_config.get(
                    "role_anchor_preservation_weight",
                    (
                        10.0
                        if balanced_role_grounding
                        else (1.0 if role_adapter_grounding else 0.0)
                    ),
                )
            ),
            role_relation_preservation=float(
                eraf_config.get(
                    "role_relation_preservation_weight",
                    (
                        2.0
                        if balanced_role_grounding
                        else (0.5 if role_adapter_grounding else 0.0)
                    ),
                )
            ),
            role_adapter_energy=float(
                eraf_config.get(
                    "role_adapter_energy_weight",
                    0.01 if role_adapter_grounding else 0.0,
                )
            ),
            structured_assignment=float(
                eraf_config.get(
                    "structured_assignment_weight",
                    2.0 if structured_role_grounding else 0.0,
                )
            ),
            structured_assignment_temperature=float(
                eraf_config.get("structured_assignment_temperature", 0.10)
            ),
            structured_assignment_hard_weight=float(
                eraf_config.get(
                    "structured_assignment_hard_weight",
                    2.0 if structured_role_grounding else 0.0,
                )
            ),
            multi_clause_consistency=float(
                eraf_config.get(
                    "multi_clause_consistency_weight",
                    1.0 if structured_role_grounding else 0.0,
                )
            ),
            clause_tuple_assignment=float(
                eraf_config.get(
                    "clause_tuple_assignment_weight",
                    4.0 if clause_tuple_grounding else 0.0,
                )
            ),
            clause_tuple_temperature=float(
                eraf_config.get("clause_tuple_temperature", 0.10)
            ),
            clause_tuple_hard_weight=float(
                eraf_config.get(
                    "clause_tuple_hard_weight",
                    1.0 if clause_tuple_grounding else 0.0,
                )
            ),
            clause_tuple_multi_consistency=float(
                eraf_config.get(
                    "clause_tuple_multi_consistency_weight",
                    2.0 if clause_tuple_grounding else 0.0,
                )
            ),
            clause_activation_balance=float(
                eraf_config.get(
                    "clause_activation_balance_weight",
                    1.0 if clause_activation_grounding else 0.0,
                )
            ),
            clause_cardinality=float(
                eraf_config.get(
                    "clause_cardinality_weight",
                    1.0 if clause_activation_grounding else 0.0,
                )
            ),
            clause_worst_slot=float(
                eraf_config.get(
                    "clause_worst_slot_weight",
                    2.0 if clause_activation_grounding else 0.0,
                )
            ),
            clause_multi_group_weight=float(
                eraf_config.get("clause_multi_group_weight", 1.0)
            ),
            clause_adapter_energy=float(
                eraf_config.get(
                    "clause_adapter_energy_weight",
                    0.01 if clause_activation_grounding else 0.0,
                )
            ),
            view_fusion=float(
                eraf_config.get(
                    "view_fusion_weight",
                    2.0 if view_scheduler_grounding else 0.0,
                )
            ),
            view_fusion_energy=float(
                eraf_config.get(
                    "view_fusion_energy_weight",
                    0.01 if view_scheduler_grounding else 0.0,
                )
            ),
            clause_scheduler=float(
                eraf_config.get(
                    "clause_scheduler_weight",
                    1.0 if view_scheduler_grounding else 0.0,
                )
            ),
            clause_scheduler_energy=float(
                eraf_config.get(
                    "clause_scheduler_energy_weight",
                    0.01 if view_scheduler_grounding else 0.0,
                )
            ),
            phase_rebinding_energy=float(
                eraf_config.get(
                    "phase_rebinding_energy_weight",
                    0.01 if closed_loop_rebinding_grounding else 0.0,
                )
            ),
            phase_safe_memory_state=float(
                eraf_config.get(
                    "phase_safe_memory_state_weight",
                    1.0 if phase_safe_memory_grounding else 0.0,
                )
            ),
            phase_safe_memory_scheduler=float(
                eraf_config.get(
                    "phase_safe_memory_scheduler_weight",
                    1.0 if phase_safe_memory_grounding else 0.0,
                )
            ),
            phase_safe_memory_energy=float(
                eraf_config.get(
                    "phase_safe_memory_energy_weight",
                    0.01 if phase_safe_memory_grounding else 0.0,
                )
            ),
            phase=float(eraf_config.get("phase_weight", 1.0)),
        )
        self.policy_guard_action_weight = float(
            guard_config.get("counterfactual_action_weight", 1.0)
        )
        self.policy_guard_native_distillation_weight = float(
            guard_config.get("native_distillation_weight", 1.0)
        )
        self.policy_guard_residual_regularization_weight = float(
            guard_config.get("residual_regularization_weight", 0.01)
        )
        self.policy_guard_residual_smoothness_weight = float(
            guard_config.get("residual_smoothness_weight", 0.01)
        )
        self.policy_guard_velocity_residual_max_abs = guard_config.get(
            "velocity_residual_max_abs", 1.0
        )
        self.policy_guard_action_chunk_residual_max_abs = guard_config.get(
            "action_chunk_residual_max_abs", 2.0
        )
        self.policy_guard_rollout_num_inference_steps = int(
            guard_config.get("rollout_num_inference_steps", 10)
        )
        self.policy_guard_action_gripper_weight = float(
            guard_config.get("action_gripper_weight", 2.0)
        )
        self.policy_guard_advantage_temperature = float(
            guard_config.get("advantage_temperature", 0.25)
        )
        self.policy_guard_advantage_clip = float(
            guard_config.get("advantage_clip", 4.0)
        )
        self.policy_guard_candidate_max_saturation_fraction = float(
            guard_config.get("candidate_max_saturation_fraction", 0.25)
        )
        self.policy_guard_candidate_max_delta_rms = float(
            guard_config.get("candidate_max_delta_rms", 2.0)
        )
        self.policy_guard_execution_prefix_steps = int(
            guard_config.get("execution_prefix_steps", 10)
        )
        self.policy_guard_suffix_loss_weight = float(
            guard_config.get("suffix_loss_weight", 0.10)
        )
        self.policy_guard_completion_phase_enabled = bool(
            guard_config.get("completion_phase_enabled", False)
        )
        self.policy_guard_completion_transport_weight = float(
            guard_config.get("completion_transport_weight", 2.0)
        )
        self.policy_guard_completion_release_weight = float(
            guard_config.get("completion_release_weight", 3.0)
        )
        self.policy_guard_completion_train_proposal_only = bool(
            guard_config.get("completion_train_proposal_only", True)
        )
        self.policy_guard_closed_loop_corrective_enabled = bool(
            guard_config.get("closed_loop_corrective_enabled", False)
        )
        self.policy_guard_closed_loop_corrective_weight = float(
            guard_config.get("closed_loop_corrective_weight", 2.0)
        )
        self.policy_guard_offline_acquisition_weight = float(
            guard_config.get("offline_acquisition_weight", 1.0)
        )
        self.policy_guard_native_guard_weight = float(
            guard_config.get("native_guard_weight", 0.10)
        )
        self.policy_guard_acquisition_only = bool(
            guard_config.get("acquisition_only", True)
        )
        self.policy_guard_closed_loop_train_proposal_only = bool(
            guard_config.get("closed_loop_train_proposal_only", True)
        )
        self.policy_guard_same_state_source_zero_weight = float(
            guard_config.get("same_state_source_zero_weight", 1.0)
        )
        self.policy_guard_goal_separation_weight = float(
            guard_config.get("goal_separation_weight", 0.25)
        )
        self.policy_guard_goal_separation_margin = float(
            guard_config.get("goal_separation_margin", 0.20)
        )
        self.policy_guard_residual_separation_weight = float(
            guard_config.get("residual_separation_weight", 0.25)
        )
        self.policy_guard_residual_separation_margin = float(
            guard_config.get("residual_separation_margin", 0.05)
        )
        self.policy_guard_verifier_wrong_language_weight = float(
            guard_config.get("verifier_wrong_language_weight", 0.50)
        )
        self.policy_guard_verifier_bad_candidate_weight = float(
            guard_config.get("verifier_bad_candidate_weight", 0.50)
        )
        self.policy_guard_verifier_wrong_entity_weight = float(
            guard_config.get("verifier_wrong_entity_weight", 0.50)
        )
        self.policy_guard_verifier_wrong_relation_weight = float(
            guard_config.get("verifier_wrong_relation_weight", 0.50)
        )
        self.policy_guard_target_binding_interaction_weight = float(
            guard_config.get("target_binding_interaction_weight", 1.0)
        )
        self.policy_guard_target_binding_prototype_weight = float(
            guard_config.get("target_binding_prototype_weight", 0.50)
        )
        self.policy_guard_target_binding_source_weight = float(
            guard_config.get("target_binding_source_weight", 0.50)
        )
        self.policy_guard_target_binding_hard_negative_weight = float(
            guard_config.get("target_binding_hard_negative_weight", 0.50)
        )
        self.policy_guard_target_binding_separation_weight = float(
            guard_config.get("target_binding_separation_weight", 0.25)
        )
        self.policy_guard_target_binding_hard_negative_margin = float(
            guard_config.get("target_binding_hard_negative_margin", 0.20)
        )
        self.policy_guard_target_binding_separation_margin = float(
            guard_config.get("target_binding_separation_margin", 0.15)
        )
        self.policy_guard_target_binding_teacher_topk = float(
            guard_config.get("target_binding_teacher_topk", 0.15)
        )
        self.policy_guard_target_binding_teacher_temperature = float(
            guard_config.get("target_binding_teacher_temperature", 0.25)
        )
        self.policy_guard_target_binding_hidden_dim = int(
            guard_config.get("target_binding_hidden_dim", 256)
        )
        self.policy_guard_target_binding_temperature = float(
            guard_config.get("target_binding_temperature", 0.07)
        )
        self.policy_guard_target_binding_prototype_slots = int(
            guard_config.get("target_binding_prototype_slots", 64)
        )
        self.policy_guard_target_binding_prototype_momentum = float(
            guard_config.get("target_binding_prototype_momentum", 0.95)
        )
        self.policy_guard_target_binding_prototype_temperature = float(
            guard_config.get("target_binding_prototype_temperature", 0.07)
        )
        self.policy_guard_target_binding_prototype_topk = float(
            guard_config.get("target_binding_prototype_topk", 0.10)
        )
        self.policy_guard_target_binding_num_object_tokens = int(
            guard_config.get("target_binding_num_object_tokens", 8)
        )
        self.policy_guard_target_binding_camera_count = int(
            guard_config.get("target_binding_camera_count", 2)
        )
        self.policy_guard_target_mask_weight = float(
            guard_config.get("target_mask_weight", 1.0)
        )
        self.policy_guard_source_mask_weight = float(
            guard_config.get("source_mask_weight", 0.5)
        )
        self.policy_guard_aux_mask_weight = float(
            guard_config.get("aux_mask_weight", 0.5)
        )
        self.policy_guard_mask_mass_weight = float(
            guard_config.get("mask_mass_weight", 0.5)
        )
        self.policy_guard_cross_object_weight = float(
            guard_config.get("cross_object_weight", 0.5)
        )
        self.policy_guard_cross_object_margin = float(
            guard_config.get("cross_object_margin", 0.25)
        )
        self.policy_guard_target_binding_action_start_step = int(
            guard_config.get("target_binding_action_start_step", 1000)
        )
        self.policy_guard_target_binding_action_ramp_steps = int(
            guard_config.get("target_binding_action_ramp_steps", 500)
        )
        self.policy_guard_verifier_start_step = int(
            guard_config.get("verifier_start_step", 1000)
        )
        self.policy_guard_verifier_ramp_steps = int(
            guard_config.get("verifier_ramp_steps", 500)
        )
        self._policy_guard_training_step = 0
        self._policy_guard_training_max_steps = 1
        self._policy_guard_training_progress_active = False
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
        self.policy_guard_target_prototype_bank: Optional[
            StateTargetPrototypeBank
        ] = None
        self.policy_guard_base_checkpoint: Optional[str] = None
        self.policy_guard_legacy_full_loaded = False
        self._policy_guard_last_eraf_diagnostics: Optional[
            dict[str, torch.Tensor]
        ] = None
        # Ephemeral inference-only tensors used by the V9.17 direct geometry
        # route and causal audit. They are never serialized or returned by the
        # ordinary rollout API.
        self._policy_guard_last_eraf_outputs: Optional[
            dict[str, torch.Tensor]
        ] = None

        if self.policy_guard_enabled:
            if self.policy_guard_version not in {1, 2, 3, 4, 5, 6, 7, 8, 9}:
                raise ValueError(
                    "The current PGC implementation supports version=1, 2, "
                    "3, 4, 5, 6, 7, 8, or 9."
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
                self.policy_guard_residual_regularization_weight,
                self.policy_guard_residual_smoothness_weight,
                self.policy_guard_goal_residual_scale,
                self.policy_guard_verifier_weight,
                self.policy_guard_alignment_weight,
                self.policy_guard_verifier_margin,
                self.policy_guard_verifier_action_mse_temperature,
                self.policy_guard_gate_threshold,
                self.policy_guard_min_counterfactual_score,
                self.policy_guard_action_gripper_weight,
                self.policy_guard_advantage_temperature,
                self.policy_guard_advantage_clip,
                self.policy_guard_candidate_max_saturation_fraction,
                self.policy_guard_candidate_max_delta_rms,
                self.policy_guard_suffix_loss_weight,
                self.policy_guard_same_state_source_zero_weight,
                self.policy_guard_goal_separation_weight,
                self.policy_guard_goal_separation_margin,
                self.policy_guard_residual_separation_weight,
                self.policy_guard_residual_separation_margin,
                self.policy_guard_verifier_wrong_language_weight,
                self.policy_guard_verifier_bad_candidate_weight,
                self.policy_guard_verifier_wrong_entity_weight,
                self.policy_guard_verifier_wrong_relation_weight,
                self.policy_guard_target_binding_interaction_weight,
                self.policy_guard_target_binding_prototype_weight,
                self.policy_guard_target_binding_source_weight,
                self.policy_guard_target_binding_hard_negative_weight,
                self.policy_guard_target_binding_separation_weight,
                self.policy_guard_target_binding_hard_negative_margin,
                self.policy_guard_target_binding_separation_margin,
                self.policy_guard_target_mask_weight,
                self.policy_guard_source_mask_weight,
                self.policy_guard_aux_mask_weight,
                self.policy_guard_mask_mass_weight,
                self.policy_guard_cross_object_weight,
                self.policy_guard_cross_object_margin,
                self.policy_guard_closed_loop_corrective_weight,
                self.policy_guard_offline_acquisition_weight,
                self.policy_guard_native_guard_weight,
            ) < 0:
                raise ValueError("PGC weights, margins, and thresholds must be non-negative.")
            if self.policy_guard_execution_prefix_steps <= 0:
                raise ValueError("PGC execution_prefix_steps must be positive.")
            if self.policy_guard_completion_phase_enabled:
                if self.policy_guard_version != 5:
                    raise ValueError(
                        "PGC completion-phase recovery is intentionally "
                        "restricted to the V5 baseline."
                    )
                if min(
                    self.policy_guard_completion_transport_weight,
                    self.policy_guard_completion_release_weight,
                ) < 1.0:
                    raise ValueError(
                        "PGC completion transport/release weights must be >= 1."
                    )
            if self.policy_guard_closed_loop_corrective_enabled:
                if self.policy_guard_version != 8:
                    raise ValueError(
                        "Closed-loop corrective supervision is restricted to PGC v8."
                    )
                if not self.policy_guard_acquisition_only:
                    raise ValueError(
                        "PGC v8 currently requires acquisition_only=true."
                    )
            elif self.policy_guard_version == 8:
                raise ValueError(
                    "PGC v8 requires closed_loop_corrective_enabled=true."
                )
            if self.policy_guard_version == 9:
                if self.policy_guard_eraf_initialization_contract not in {
                    "exact_pgc_v5_sidecars",
                    "released_base_fresh_eraf",
                }:
                    raise ValueError(
                        "PGC v9 ERAF initialization_contract must be "
                        "'exact_pgc_v5_sidecars' or "
                        "'released_base_fresh_eraf'."
                    )
                if self.policy_guard_eraf_grounding_objective_version not in {
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                    20,
                    21,
                    22,
                    23,
                    24,
                    25,
                    26,
                }:
                    raise ValueError(
                        "PGC v9 ERAF grounding_objective_version must be "
                        "between 1 and 26 inclusive."
                    )
                if min(
                    self.policy_guard_eraf_loss_weights.role_assignment,
                    self.policy_guard_eraf_loss_weights.role_assignment_hard_weight,
                    self.policy_guard_eraf_loss_weights.structured_assignment,
                    self.policy_guard_eraf_loss_weights.structured_assignment_hard_weight,
                    self.policy_guard_eraf_loss_weights.multi_clause_consistency,
                    self.policy_guard_eraf_loss_weights.clause_tuple_assignment,
                    self.policy_guard_eraf_loss_weights.clause_tuple_hard_weight,
                    self.policy_guard_eraf_loss_weights.clause_tuple_multi_consistency,
                    self.policy_guard_eraf_loss_weights.clause_activation_balance,
                    self.policy_guard_eraf_loss_weights.clause_cardinality,
                    self.policy_guard_eraf_loss_weights.clause_worst_slot,
                    self.policy_guard_eraf_loss_weights.clause_adapter_energy,
                    self.policy_guard_eraf_loss_weights.view_fusion,
                    self.policy_guard_eraf_loss_weights.view_fusion_energy,
                    self.policy_guard_eraf_loss_weights.clause_scheduler,
                    self.policy_guard_eraf_loss_weights.clause_scheduler_energy,
                    self.policy_guard_eraf_loss_weights.phase_rebinding_energy,
                    self.policy_guard_eraf_loss_weights.phase_safe_memory_state,
                    self.policy_guard_eraf_loss_weights.phase_safe_memory_scheduler,
                    self.policy_guard_eraf_loss_weights.phase_safe_memory_energy,
                ) < 0:
                    raise ValueError(
                        "PGC v9 ERAF role-assignment weights must be non-negative."
                    )
                if self.policy_guard_eraf_loss_weights.clause_tuple_temperature <= 0:
                    raise ValueError(
                        "PGC v9 ERAF clause-tuple temperature must be positive."
                    )
                if self.policy_guard_eraf_loss_weights.clause_multi_group_weight <= 0:
                    raise ValueError(
                        "PGC v9 ERAF clause multi-group weight must be positive."
                    )
                if self.policy_guard_eraf_clause_activation_residual_max_abs <= 0:
                    raise ValueError(
                        "PGC v9 ERAF clause activation residual bound must be positive."
                    )
                if self.policy_guard_eraf_view_fusion_residual_max_abs <= 0:
                    raise ValueError(
                        "PGC v9 ERAF view-fusion residual bound must be positive."
                    )
                if not (
                    0
                    < self.policy_guard_eraf_clause_scheduler_residual_max_abs
                    <= 1
                ):
                    raise ValueError(
                        "PGC v9 ERAF scheduler residual bound must be in (0,1]."
                    )
                if (
                    self.policy_guard_eraf_loss_weights.role_assignment_temperature
                    <= 0
                ):
                    raise ValueError(
                        "PGC v9 ERAF role-assignment temperature must be positive."
                    )
                if (
                    self.policy_guard_eraf_loss_weights.structured_assignment_temperature
                    <= 0
                ):
                    raise ValueError(
                        "PGC v9 ERAF structured-assignment temperature must be positive."
                    )
                if self.policy_guard_eraf_training_stage not in {
                    "grounding",
                    "action",
                    "verifier",
                }:
                    raise ValueError(
                        "PGC v9 ERAF training_stage must be grounding, action, "
                        "or verifier."
                    )
                if (
                    self.policy_guard_eraf_completion_only_memory
                    and self.policy_guard_eraf_grounding_objective_version < 14
                ):
                    raise ValueError(
                        "PGC V9.14 completion-only memory requires the "
                        "objective-v14 phase-memory architecture."
                    )
                if self.policy_guard_eraf_action_joint_training and not (
                    self.policy_guard_eraf_completion_only_memory
                    and self.policy_guard_eraf_grounding_objective_version >= 14
                    and self.policy_guard_eraf_training_stage == "action"
                ):
                    raise ValueError(
                        "PGC V9.14 ERAF--Proposal joint training requires "
                        "objective-v14, action stage, and completion-only memory."
                    )
                if (
                    self.policy_guard_eraf_action_joint_training
                    and self.policy_guard_eraf_grounding_aux_weight != 0.0
                    and self.policy_guard_eraf_grounding_objective_version < 26
                ):
                    raise ValueError(
                        "PGC V9.14 freezes the ERAF grounding core; "
                        "grounding_aux_weight must be zero during joint action "
                        "training."
                    )
                if (
                    self.policy_guard_eraf_grounding_objective_version >= 26
                    and self.policy_guard_eraf_grounding_aux_weight <= 0.0
                ):
                    raise ValueError(
                        "PGC V9.26 shared Expert LoRA training requires a "
                        "positive grounding_aux_weight so frozen ERAF behavior "
                        "is preserved while Video features adapt."
                    )
                if (
                    self.policy_guard_eraf_grounding_objective_version >= 15
                    and not (
                        self.policy_guard_eraf_action_joint_training
                        and self.policy_guard_eraf_completion_only_memory
                        and self.policy_guard_eraf_training_stage == "action"
                    )
                ):
                    raise ValueError(
                        "PGC V9.15 geometry-action grounding requires the "
                        "completion-only joint action stage."
                    )
                if min(
                    self.policy_guard_eraf_action_grounding_hidden_dim,
                    self.policy_guard_eraf_action_grounding_num_heads,
                    self.policy_guard_eraf_action_geometry_hidden_dim,
                ) <= 0:
                    raise ValueError(
                        "PGC V9.15 action-grounding dimensions must be positive."
                    )
                if self.policy_guard_eraf_action_geometry_residual_max_abs <= 0:
                    raise ValueError(
                        "PGC V9.17 geometry-action residual bound must be positive."
                    )
                if min(
                    self.policy_guard_eraf_action_phase_residual_imitation_weight,
                    self.policy_guard_eraf_action_phase_direction_weight,
                    self.policy_guard_eraf_action_phase_approach_weight,
                    self.policy_guard_eraf_action_phase_transport_weight,
                    self.policy_guard_eraf_action_phase_release_weight,
                    self.policy_guard_eraf_action_phase_direction_min_norm,
                ) <= 0:
                    raise ValueError(
                        "PGC V9.18 phase-residual weights and direction norm "
                        "threshold must be positive."
                    )
                if (
                    self.policy_guard_eraf_grounding_objective_version >= 26
                    and (
                        min(
                            self.policy_guard_eraf_expert_lora_world_language_weight,
                            self.policy_guard_eraf_expert_lora_world_language_margin,
                            self.policy_guard_eraf_expert_lora_native_action_weight,
                            self.policy_guard_eraf_expert_lora_counterfactual_action_weight,
                        )
                        <= 0
                        or self.policy_guard_eraf_expert_lora_regularization_weight
                        < 0
                    )
                ):
                    raise ValueError(
                        "PGC V9.26 Expert-LoRA world/action weights and margin "
                        "must be positive; LoRA regularization must be non-negative."
                    )
                if self.policy_guard_eraf_action_servo_frame_weight < 0:
                    raise ValueError(
                        "PGC V9.19 servo frame regularization must be "
                        "non-negative."
                    )
                if min(
                    self.policy_guard_eraf_action_waypoint_compatibility_weight,
                    self.policy_guard_eraf_action_waypoint_imitation_weight,
                    self.policy_guard_eraf_action_waypoint_direction_weight,
                    self.policy_guard_eraf_action_waypoint_zero_weight,
                ) <= 0:
                    raise ValueError("PGC V9.20 waypoint loss weights must be positive.")
                if not 0 <= self.policy_guard_eraf_action_waypoint_min_cosine <= 1:
                    raise ValueError("PGC V9.20 waypoint minimum cosine must be in [0,1].")
                if not 0 < self.policy_guard_eraf_action_waypoint_tangent_max_ratio <= 1:
                    raise ValueError("PGC V9.20 tangent ratio must be in (0,1].")
                if min(
                    self.policy_guard_eraf_action_expert_imitation_weight,
                    self.policy_guard_eraf_action_expert_direction_weight,
                    self.policy_guard_eraf_action_expert_deployed_weight,
                    self.policy_guard_eraf_action_expert_distillation_weight,
                    self.policy_guard_eraf_action_expert_native_zero_weight,
                ) <= 0:
                    raise ValueError(
                        "PGC V9.21 expert-alignment loss weights must be positive."
                    )
                if min(
                    self.policy_guard_eraf_action_clause_ranking_weight,
                    self.policy_guard_eraf_action_clause_ranking_margin,
                ) <= 0:
                    raise ValueError(
                        "PGC V9.22 clause-action ranking weight and margin "
                        "must be positive."
                    )
                if min(
                    self.policy_guard_eraf_action_clause_teacher_weight,
                    self.policy_guard_eraf_action_clause_alignment_guard_weight,
                ) <= 0:
                    raise ValueError(
                        "PGC V9.23 clause teacher and alignment-guard weights "
                        "must be positive."
                    )
                if (
                    self.policy_guard_eraf_action_clause_wrong_suppression_weight
                    <= 0
                ):
                    raise ValueError(
                        "PGC V9.24 wrong-clause suppression weight must be "
                        "positive."
                    )
                if (
                    self.policy_guard_eraf_action_grounding_hidden_dim
                    % self.policy_guard_eraf_action_grounding_num_heads
                    != 0
                ):
                    raise ValueError(
                        "PGC V9.15 action-grounding hidden dimension must be "
                        "divisible by its attention-head count."
                    )
                if self.policy_guard_eraf_action_grounding_learning_rate <= 0:
                    raise ValueError(
                        "PGC V9.15 action-grounding learning rate must be positive."
                    )
                if min(
                    self.policy_guard_eraf_action_causal_ranking_weight,
                    self.policy_guard_eraf_action_causal_margin,
                ) < 0:
                    raise ValueError(
                        "PGC V9.15 causal action weights must be non-negative."
                    )
                if min(
                    self.policy_guard_eraf_hidden_dim,
                    self.policy_guard_eraf_num_heads,
                    self.policy_guard_eraf_max_clauses,
                    self.policy_guard_eraf_camera_count,
                    self.policy_guard_eraf_role_adapter_hidden_dim,
                    self.policy_guard_eraf_structured_role_adapter_hidden_dim,
                    self.policy_guard_eraf_balanced_role_adapter_hidden_dim,
                    self.policy_guard_eraf_closed_loop_rebinding_hidden_dim,
                    self.policy_guard_eraf_phase_safe_memory_hidden_dim,
                    self.policy_guard_eraf_phase_safe_memory_state_count,
                ) <= 0:
                    raise ValueError("PGC v9 ERAF dimensions must be positive.")
                if (
                    self.policy_guard_eraf_hidden_dim
                    % self.policy_guard_eraf_num_heads
                ):
                    raise ValueError(
                        "PGC v9 ERAF hidden_dim must be divisible by num_heads."
                    )
                if min(
                    self.policy_guard_eraf_visual_aspect_ratio,
                    self.policy_guard_eraf_temperature,
                    self.policy_guard_eraf_learning_rate,
                    self.policy_guard_eraf_closed_loop_query_residual_max_abs,
                    self.policy_guard_eraf_closed_loop_state_residual_max_abs,
                    self.policy_guard_eraf_phase_safe_memory_routing_residual_max_abs,
                ) <= 0:
                    raise ValueError(
                        "PGC v9 ERAF aspect ratio, temperature, and LR must "
                        "be positive."
                    )
                if self.policy_guard_eraf_phase_safe_memory_state_count != 4:
                    raise ValueError(
                        "PGC V9.13 phase-safe memory requires four states."
                    )
                if not (
                    0
                    < self.policy_guard_eraf_phase_safe_memory_routing_residual_max_abs
                    <= 1
                ):
                    raise ValueError(
                        "PGC V9.13 phase-safe routing residual bound must be in (0,1]."
                    )
                if min(
                    self.policy_guard_eraf_grounding_aux_weight,
                    *self.policy_guard_eraf_loss_weights.__dict__.values(),
                ) < 0:
                    raise ValueError("PGC v9 ERAF loss weights must be non-negative.")
            if self.policy_guard_version in {6, 7}:
                if (
                    self.policy_guard_target_binding_action_start_step < 0
                    or self.policy_guard_target_binding_action_ramp_steps < 0
                ):
                    raise ValueError(
                        "PGC v6 target-binding action start/ramp must be "
                        "non-negative."
                    )
                if self.policy_guard_target_binding_hidden_dim <= 0:
                    raise ValueError(
                        "PGC v6/v7 target-binding dimensions must be positive."
                    )
                if self.policy_guard_target_binding_temperature <= 0:
                    raise ValueError(
                        "PGC v6/v7 target-binding temperature must be positive."
                    )
                if self.policy_guard_version == 6:
                    if self.policy_guard_target_binding_prototype_slots <= 0:
                        raise ValueError(
                            "PGC v6 target-binding prototype slots must be positive."
                        )
                    if min(
                        self.policy_guard_target_binding_teacher_temperature,
                        self.policy_guard_target_binding_prototype_temperature,
                    ) <= 0:
                        raise ValueError(
                            "PGC v6 target-binding teacher/prototype "
                            "temperatures must be positive."
                        )
                    if not 0.0 < self.policy_guard_target_binding_teacher_topk <= 1.0:
                        raise ValueError(
                            "PGC v6 target-binding teacher top-k must be in (0,1]."
                        )
                    if not 0.0 < self.policy_guard_target_binding_prototype_topk <= 1.0:
                        raise ValueError(
                            "PGC v6 target-binding prototype top-k must be in (0,1]."
                        )
                    if not (
                        0.0
                        <= self.policy_guard_target_binding_prototype_momentum
                        < 1.0
                    ):
                        raise ValueError(
                            "PGC v6 target-binding prototype momentum must be in [0,1)."
                        )
                if self.policy_guard_version == 7 and min(
                    self.policy_guard_target_binding_num_object_tokens,
                    self.policy_guard_target_binding_camera_count,
                ) <= 0:
                    raise ValueError(
                        "PGC v7 object-token count and camera count must be positive."
                    )
            if self.policy_guard_suffix_loss_weight > 1:
                raise ValueError("PGC suffix_loss_weight must be in [0, 1].")
            if self.policy_guard_rollout_num_inference_steps <= 0:
                raise ValueError(
                    "PGC v4 rollout_num_inference_steps must be positive."
                )
            if self.policy_guard_advantage_temperature <= 0:
                raise ValueError("PGC v4 advantage_temperature must be positive.")
            if self.policy_guard_advantage_clip <= 0:
                raise ValueError("PGC v4 advantage_clip must be positive.")
            if self.policy_guard_action_gripper_weight <= 0:
                raise ValueError("PGC v4 action_gripper_weight must be positive.")
            if not 0.0 <= self.policy_guard_candidate_max_saturation_fraction <= 1.0:
                raise ValueError(
                    "PGC v4 candidate_max_saturation_fraction must be in [0, 1]."
                )
            if self.policy_guard_candidate_max_delta_rms <= 0:
                raise ValueError(
                    "PGC v4 candidate_max_delta_rms must be positive."
                )
            if (
                self.policy_guard_verifier_start_step < 0
                or self.policy_guard_verifier_ramp_steps < 0
            ):
                raise ValueError(
                    "PGC verifier start/ramp steps must be non-negative."
                )
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

            # v1/v2 retain their historical physically independent Action
            # Expert for checkpoint compatibility. v3 has no trainable Action
            # Expert copy: both candidates use the single frozen released
            # expert and the counterfactual candidate receives only a bounded
            # flow-velocity residual after the frozen post-DiT head.
            policy_action_expert = self.action_expert
            if self.policy_guard_version <= 2:
                counterfactual_action_expert = copy.deepcopy(
                    self.action_expert
                )
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
                            / math.sqrt(
                                int(counterfactual_action_expert.hidden_dim)
                            )
                        ),
                    )
                else:
                    counterfactual_action_expert.use_latent_action_queries = False
                self.policy_guard_action_expert = counterfactual_action_expert
                policy_action_expert = counterfactual_action_expert

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
                        action_dim=int(policy_action_expert.hidden_dim),
                        hidden_dim=hidden_dim,
                        projection_dim=projection_dim,
                        num_goal_tokens=int(
                            guard_config.get("num_goal_tokens", 4)
                        ),
                        num_heads=int(guard_config.get("num_heads", 8)),
                    ),
                    "verifier": (
                        PairwiseActionAdvantageVerifier(
                            action_dim=int(policy_action_expert.action_dim),
                            video_dim=int(self.video_expert.hidden_dim),
                            goal_dim=projection_dim,
                            hidden_dim=verifier_hidden_dim,
                            num_heads=int(
                                guard_config.get("verifier_num_heads", 8)
                            ),
                            num_layers=int(
                                guard_config.get("verifier_num_layers", 2)
                            ),
                        )
                        if self.policy_guard_version >= 4
                        else ActionOutcomeVerifier(
                            action_dim=int(policy_action_expert.action_dim),
                            video_dim=int(self.video_expert.hidden_dim),
                            goal_dim=projection_dim,
                            hidden_dim=verifier_hidden_dim,
                        )
                    ),
                }
            )
            if self.policy_guard_version >= 2:
                goal_query_seeds = nn.Embedding(
                    num_action_queries,
                    int(policy_action_expert.hidden_dim),
                )
                nn.init.normal_(
                    goal_query_seeds.weight,
                    std=(
                        1.0
                        / math.sqrt(
                            int(policy_action_expert.hidden_dim)
                        )
                    ),
                )
                self.policy_guard_modules["goal_query_seeds"] = goal_query_seeds
                if self.policy_guard_version == 2:
                    self.policy_guard_modules["goal_residual_adapter"] = (
                        GoalResidualAdapter(
                            action_dim=int(policy_action_expert.hidden_dim),
                            num_heads=int(guard_config.get("num_heads", 8)),
                            residual_scale=(
                                self.policy_guard_goal_residual_scale
                            ),
                        )
                    )
                elif self.policy_guard_version == 3:
                    self.policy_guard_modules["action_velocity_residual"] = (
                        BoundedActionVelocityResidual(
                            action_hidden_dim=int(
                                policy_action_expert.hidden_dim
                            ),
                            action_dim=int(policy_action_expert.action_dim),
                            num_heads=int(guard_config.get("num_heads", 8)),
                            max_abs=(
                                self.policy_guard_velocity_residual_max_abs
                            ),
                        )
                    )
                else:
                    self.policy_guard_modules["action_chunk_proposal"] = (
                        RolloutAlignedActionProposal(
                            action_dim=int(policy_action_expert.action_dim),
                            goal_dim=int(policy_action_expert.hidden_dim),
                            hidden_dim=int(
                                guard_config.get("proposal_hidden_dim", 256)
                            ),
                            num_heads=int(
                                guard_config.get("proposal_num_heads", 8)
                            ),
                            num_layers=int(
                                guard_config.get("proposal_num_layers", 2)
                            ),
                            max_abs=(
                                self.policy_guard_action_chunk_residual_max_abs
                            ),
                        )
                    )
                    if self.policy_guard_version == 9:
                        self.policy_guard_modules[
                            "entity_relation_affordance"
                        ] = EntityRelationAffordanceField(
                            text_dim=self.text_dim,
                            video_dim=int(self.video_expert.hidden_dim),
                            action_dim=int(policy_action_expert.hidden_dim),
                            projection_dim=projection_dim,
                            hidden_dim=self.policy_guard_eraf_hidden_dim,
                            num_heads=self.policy_guard_eraf_num_heads,
                            max_clauses=self.policy_guard_eraf_max_clauses,
                            camera_count=self.policy_guard_eraf_camera_count,
                            visual_aspect_ratio=(
                                self.policy_guard_eraf_visual_aspect_ratio
                            ),
                            temperature=self.policy_guard_eraf_temperature,
                            entity_only=self.policy_guard_eraf_entity_only,
                            use_anchors=self.policy_guard_eraf_use_anchors,
                            role_adapter_enabled=(
                                self.policy_guard_eraf_grounding_objective_version
                                >= 4
                            ),
                            role_adapter_hidden_dim=(
                                self.policy_guard_eraf_role_adapter_hidden_dim
                            ),
                            role_adapter_teacher_enabled=(
                                self.policy_guard_eraf_grounding_objective_version
                                >= 4
                            ),
                            structured_role_adapter_enabled=(
                                self.policy_guard_eraf_grounding_objective_version
                                == 5
                            ),
                            structured_role_adapter_hidden_dim=(
                                self.policy_guard_eraf_structured_role_adapter_hidden_dim
                            ),
                            balanced_role_adapter_enabled=(
                                self.policy_guard_eraf_grounding_objective_version
                                >= 6
                            ),
                            balanced_role_adapter_hidden_dim=(
                                self.policy_guard_eraf_balanced_role_adapter_hidden_dim
                            ),
                            clause_activation_adapter_enabled=(
                                self.policy_guard_eraf_grounding_objective_version
                                >= 9
                            ),
                            clause_activation_adapter_hidden_dim=(
                                self.policy_guard_eraf_clause_activation_adapter_hidden_dim
                            ),
                            clause_activation_residual_max_abs=(
                                self.policy_guard_eraf_clause_activation_residual_max_abs
                            ),
                            view_fusion_enabled=(
                                self.policy_guard_eraf_grounding_objective_version
                                >= 10
                            ),
                            view_fusion_adapter_hidden_dim=(
                                self.policy_guard_eraf_view_fusion_adapter_hidden_dim
                            ),
                            view_fusion_residual_max_abs=(
                                self.policy_guard_eraf_view_fusion_residual_max_abs
                            ),
                            clause_scheduler_enabled=(
                                self.policy_guard_eraf_grounding_objective_version
                                >= 10
                            ),
                            clause_scheduler_hidden_dim=(
                                self.policy_guard_eraf_clause_scheduler_hidden_dim
                            ),
                            clause_scheduler_residual_max_abs=(
                                self.policy_guard_eraf_clause_scheduler_residual_max_abs
                            ),
                            closed_loop_rebinding_enabled=(
                                self.policy_guard_eraf_grounding_objective_version
                                == 13
                            ),
                            closed_loop_rebinding_hidden_dim=(
                                self.policy_guard_eraf_closed_loop_rebinding_hidden_dim
                            ),
                            closed_loop_query_residual_max_abs=(
                                self.policy_guard_eraf_closed_loop_query_residual_max_abs
                            ),
                            closed_loop_state_residual_max_abs=(
                                self.policy_guard_eraf_closed_loop_state_residual_max_abs
                            ),
                            phase_safe_memory_enabled=(
                                self.policy_guard_eraf_grounding_objective_version
                                >= 14
                            ),
                            phase_safe_memory_hidden_dim=(
                                self.policy_guard_eraf_phase_safe_memory_hidden_dim
                            ),
                            phase_safe_memory_state_count=(
                                self.policy_guard_eraf_phase_safe_memory_state_count
                            ),
                            phase_safe_memory_routing_residual_max_abs=(
                                self.policy_guard_eraf_phase_safe_memory_routing_residual_max_abs
                            ),
                        )
                        if (
                            self.policy_guard_eraf_grounding_objective_version
                            >= 15
                        ):
                            self.policy_guard_modules[
                                "eraf_action_grounding_bridge"
                            ] = PhaseConditionedERAFActionBridge(
                                goal_dim=int(policy_action_expert.hidden_dim),
                                eraf_hidden_dim=self.policy_guard_eraf_hidden_dim,
                                hidden_dim=(
                                    self.policy_guard_eraf_action_grounding_hidden_dim
                                ),
                                num_heads=(
                                    self.policy_guard_eraf_action_grounding_num_heads
                                ),
                                max_clauses=self.policy_guard_eraf_max_clauses,
                            )
                        if (
                            self.policy_guard_eraf_grounding_objective_version
                            >= 17
                        ):
                            if self.proprio_dim is None:
                                raise ValueError(
                                    "PGC V9.17 direct geometry-action routing "
                                    "requires proprio_dim."
                                )
                            self.policy_guard_modules[
                                "eraf_geometry_action_adapter"
                            ] = PhaseConditionedERAFGeometryActionAdapter(
                                action_dim=int(policy_action_expert.action_dim),
                                proprio_dim=int(self.proprio_dim),
                                hidden_dim=(
                                    self.policy_guard_eraf_action_geometry_hidden_dim
                                ),
                                max_clauses=self.policy_guard_eraf_max_clauses,
                                max_abs=(
                                    self.policy_guard_eraf_action_geometry_residual_max_abs
                                ),
                            )
                        if (
                            self.policy_guard_eraf_grounding_objective_version
                            >= 19
                        ):
                            if self.proprio_dim is None:
                                raise ValueError(
                                    "PGC V9.19 hard-routed phase servo requires "
                                    "proprio_dim."
                                )
                            self.policy_guard_modules[
                                "eraf_hard_routed_phase_servo"
                            ] = HardRoutedERAFPhaseServo(
                                action_dim=int(policy_action_expert.action_dim),
                                proprio_dim=int(self.proprio_dim),
                                hidden_dim=(
                                    self.policy_guard_eraf_action_geometry_hidden_dim
                                ),
                                max_clauses=self.policy_guard_eraf_max_clauses,
                                max_abs=(
                                    self.policy_guard_eraf_action_geometry_residual_max_abs
                                ),
                                eef_scale=self.policy_guard_eraf_action_eef_scale,
                                eef_bias=self.policy_guard_eraf_action_eef_bias,
                            )
                        if self.policy_guard_eraf_grounding_objective_version >= 20:
                            self.policy_guard_modules[
                                "eraf_phase_compatible_waypoint_adapter"
                            ] = PhaseCompatibleERAFWaypointAdapter(
                                action_dim=int(policy_action_expert.action_dim),
                                hidden_dim=self.policy_guard_eraf_action_geometry_hidden_dim,
                                max_abs=self.policy_guard_eraf_action_geometry_residual_max_abs,
                                tangent_max_ratio=(
                                    self.policy_guard_eraf_action_waypoint_tangent_max_ratio
                                ),
                            )
                        if self.policy_guard_eraf_grounding_objective_version >= 21:
                            self.policy_guard_modules[
                                "eraf_phase_expert_residual_adapter"
                            ] = PhaseSpecificERAFExpertResidualAdapter(
                                action_dim=int(policy_action_expert.action_dim),
                                hidden_dim=(
                                    self.policy_guard_eraf_action_geometry_hidden_dim
                                ),
                                max_abs=(
                                    self.policy_guard_eraf_action_geometry_residual_max_abs
                                ),
                            )
                        if self.policy_guard_eraf_grounding_objective_version == 23:
                            # A checkpointed, training-only copy of the admitted
                            # V9.21 adapter.  It is never used by rollout and is
                            # kept immutable while the deployed adapter learns
                            # clause discrimination.
                            self.policy_guard_modules[
                                "eraf_phase_expert_residual_teacher"
                            ] = PhaseSpecificERAFExpertResidualAdapter(
                                action_dim=int(policy_action_expert.action_dim),
                                hidden_dim=(
                                    self.policy_guard_eraf_action_geometry_hidden_dim
                                ),
                                max_abs=(
                                    self.policy_guard_eraf_action_geometry_residual_max_abs
                                ),
                            )
                        if self.policy_guard_eraf_grounding_objective_version >= 24:
                            self.policy_guard_modules[
                                "eraf_clause_semantic_retention_residual"
                            ] = ClauseSemanticRetentionResidual(
                                action_dim=int(policy_action_expert.action_dim),
                                goal_dim=int(policy_action_expert.hidden_dim),
                                max_clauses=self.policy_guard_eraf_max_clauses,
                                hidden_dim=(
                                    self.policy_guard_eraf_action_geometry_hidden_dim
                                ),
                            )
                        if self.policy_guard_eraf_grounding_objective_version >= 25:
                            self.policy_guard_modules[
                                "eraf_action_context_injector"
                            ] = ERAFActionContextInjector(
                                goal_dim=int(policy_action_expert.hidden_dim),
                                text_dim=self.text_dim,
                                hidden_dim=(
                                    self.policy_guard_eraf_action_geometry_hidden_dim
                                ),
                            )
                    if self.policy_guard_version in {6, 7}:
                        binder_kwargs = {
                            "text_dim": self.text_dim,
                            "video_dim": int(self.video_expert.hidden_dim),
                            "action_dim": int(policy_action_expert.hidden_dim),
                            "hidden_dim": self.policy_guard_target_binding_hidden_dim,
                            "projection_dim": projection_dim,
                            "num_heads": int(
                                guard_config.get("target_binding_num_heads", 8)
                            ),
                            "temperature": self.policy_guard_target_binding_temperature,
                        }
                        if self.policy_guard_version == 7:
                            target_binder = SpatialObjectTokenTargetBinder(
                                **binder_kwargs,
                                num_object_tokens=(
                                    self.policy_guard_target_binding_num_object_tokens
                                ),
                                camera_count=(
                                    self.policy_guard_target_binding_camera_count
                                ),
                                visual_aspect_ratio=float(
                                    guard_config.get(
                                        "target_binding_visual_aspect_ratio", 2.0
                                    )
                                ),
                            )
                        else:
                            target_binder = LanguageVisualTargetBinder(
                                **binder_kwargs
                            )
                        self.policy_guard_modules[
                            "target_binder"
                        ] = target_binder
                        if self.policy_guard_version == 6:
                            self.policy_guard_target_prototype_bank = (
                                StateTargetPrototypeBank(
                                    num_slots=(
                                        self.policy_guard_target_binding_prototype_slots
                                    ),
                                    feature_dim=int(target_binder.hidden_dim),
                                    momentum=(
                                        self.policy_guard_target_binding_prototype_momentum
                                    ),
                                    temperature=(
                                        self.policy_guard_target_binding_prototype_temperature
                                    ),
                                    topk_fraction=(
                                        self.policy_guard_target_binding_prototype_topk
                                    ),
                                )
                            )
            self.policy_guard_modules.to(dtype=self.torch_dtype)
            if self.policy_guard_version >= 4:
                # The v3 probability gate collapsed in BF16 near sigmoid=1.
                # Keep the pairwise value head and temporal encoders in FP32
                # and gate on their raw advantage instead.
                self.policy_guard_modules["verifier"].float()
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
        self._policy_guard_training_step = max(0, int(step))
        self._policy_guard_training_max_steps = max(1, int(max_steps))
        self._policy_guard_training_progress_active = True

    def _policy_guard_verifier_scale(self) -> float:
        """Delay v3 verifier/alignment until the action residual is useful."""
        if self.policy_guard_version < 3:
            return 1.0
        if self.policy_guard_version == 9:
            return float(self.policy_guard_eraf_training_stage == "verifier")
        if (
            self.policy_guard_version == 8
            and self.policy_guard_closed_loop_train_proposal_only
        ):
            return 0.0
        if not self._policy_guard_training_progress_active:
            return 1.0
        step = self._policy_guard_training_step
        start = self.policy_guard_verifier_start_step
        if self.policy_guard_version in {6, 7}:
            # V6/V7 are deliberately staged: first learn the visual target map,
            # then let that map shape the Proposal, and only then fit the
            # moving candidate Verifier.  Taking the maximum also protects a
            # direct Hydra launch that forgot the wrapper's later default.
            start = max(
                start,
                self.policy_guard_target_binding_action_start_step
                + self.policy_guard_target_binding_action_ramp_steps,
            )
        ramp = self.policy_guard_verifier_ramp_steps
        if step <= start:
            return 0.0
        if ramp <= 0 or step >= start + ramp:
            return 1.0
        scale = (step - start) / ramp
        if scale >= 1.0 - 1.0e-12:
            return 1.0
        if scale <= 1.0e-12:
            return 0.0
        return float(scale)

    def _policy_guard_target_binding_action_scale(self) -> float:
        """Keep V5 action sidecars frozen while V6 first learns its target map."""
        if self.policy_guard_version == 9:
            return float(self.policy_guard_eraf_training_stage == "action")
        if self.policy_guard_version not in {6, 7}:
            return 1.0
        if not self._policy_guard_training_progress_active:
            return 1.0
        step = self._policy_guard_training_step
        start = self.policy_guard_target_binding_action_start_step
        ramp = self.policy_guard_target_binding_action_ramp_steps
        if step <= start:
            return 0.0
        if ramp <= 0 or step >= start + ramp:
            return 1.0
        scale = (step - start) / ramp
        if scale >= 1.0 - 1.0e-12:
            return 1.0
        if scale <= 1.0e-12:
            return 0.0
        return float(scale)

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
        eraf_shared_expert_lora = bool(
            self.policy_guard_enabled
            and self.policy_guard_version == 9
            and self.policy_guard_eraf_grounding_objective_version >= 26
        )
        if self.policy_guard_enabled and normalized["enabled"]:
            if self.policy_guard_version >= 3 and not eraf_shared_expert_lora:
                raise ValueError(
                    "PGC v3+ policy-guard training does not permit LoRA; "
                    "the single Base Action Expert remains immutable."
                )
            if eraf_shared_expert_lora and set(normalized["experts"]) != {
                "video",
                "action",
            }:
                raise ValueError(
                    "PGC V9.26 requires LoRA on the shared Video and Action "
                    "Experts (`lora.experts=[video,action]`)."
                )
            if not eraf_shared_expert_lora and normalized["experts"] != ["action"]:
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
        if self.policy_guard_enabled and not eraf_shared_expert_lora:
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
            if self.policy_guard_version >= 3:
                shared_expert_lora = bool(
                    self.policy_guard_version == 9
                    and self.policy_guard_eraf_grounding_objective_version >= 26
                )
                if self.lora_enabled and not shared_expert_lora:
                    raise ValueError(
                        "PGC v3+ cannot prepare trainable parameters with LoRA enabled."
                    )
                if shared_expert_lora and not self.lora_enabled:
                    raise ValueError(
                        "PGC V9.26 requires shared Video/Action Expert LoRA."
                    )
                if self.policy_guard_action_expert is not None:
                    raise RuntimeError(
                        "PGC v3+ must not instantiate an independent Action Expert."
                    )
                self.policy_guard_modules.train()
                self.policy_guard_modules.requires_grad_(True)
                if self.policy_guard_version == 9:
                    if self.policy_guard_eraf_grounding_objective_version >= 4:
                        # Grounding repairs keep the policy sidecars frozen.
                        # V9.9 jointly calibrates the previously trained role
                        # query with its new visibility fusion and scheduler;
                        # action/verifier continue to freeze the entire ERAF.
                        for module in self.policy_guard_modules.values():
                            module.eval()
                            module.requires_grad_(False)
                        if self.policy_guard_eraf_training_stage == "grounding":
                            eraf = self.policy_guard_modules[
                                "entity_relation_affordance"
                            ]
                            if (
                                self.policy_guard_eraf_grounding_objective_version
                                >= 14
                            ):
                                role_adapter = eraf.phase_safe_clause_memory
                                if role_adapter is None:
                                    raise RuntimeError(
                                        "V9.13 grounding requires its phase-safe "
                                        "clause memory."
                                    )
                            elif (
                                self.policy_guard_eraf_grounding_objective_version
                                >= 13
                            ):
                                role_adapter = (
                                    eraf.closed_loop_phase_rebinding_adapter
                                )
                                if role_adapter is None:
                                    raise RuntimeError(
                                        "V9.12 grounding requires its closed-loop "
                                        "phase-rebinding adapter."
                                    )
                            elif (
                                self.policy_guard_eraf_grounding_objective_version
                                >= 11
                            ):
                                role_adapter = eraf.balanced_role_binding_adapter
                                if role_adapter is None:
                                    raise RuntimeError(
                                        "V9.10/V9.11 grounding requires its balanced "
                                        "visual role-binding adapter."
                                    )
                            elif (
                                self.policy_guard_eraf_grounding_objective_version
                                >= 10
                            ):
                                trainable_repairs = [
                                    eraf.balanced_role_binding_adapter,
                                    eraf.clause_activation_adapter,
                                    eraf.entity_grounder.view_visibility_head,
                                    eraf.entity_grounder.view_fusion_adapter,
                                    eraf.clause_execution_scheduler,
                                ]
                                if any(
                                    repair is None for repair in trainable_repairs
                                ):
                                    raise RuntimeError(
                                        "V9.9 grounding requires balanced role, "
                                        "clause-activation, view-fusion, visibility, "
                                        "and clause-scheduler modules."
                                    )
                                for repair in trainable_repairs:
                                    repair.train()
                                    repair.requires_grad_(True)
                                role_adapter = None
                            elif (
                                self.policy_guard_eraf_grounding_objective_version
                                >= 9
                            ):
                                role_adapter = eraf.clause_activation_adapter
                                if role_adapter is None:
                                    raise RuntimeError(
                                        "V9.8 grounding requires its clause "
                                        "activation calibration adapter."
                                    )
                            elif (
                                self.policy_guard_eraf_grounding_objective_version
                                >= 6
                            ):
                                role_adapter = eraf.balanced_role_binding_adapter
                                if role_adapter is None:
                                    raise RuntimeError(
                                        "V9.5/V9.6/V9.7 grounding requires its balanced "
                                        "visual role-binding adapter."
                                    )
                            elif (
                                self.policy_guard_eraf_grounding_objective_version
                                >= 5
                            ):
                                role_adapter = (
                                    eraf.structured_role_assignment_adapter
                                )
                                if role_adapter is None:
                                    raise RuntimeError(
                                        "V9.4 grounding requires its structured "
                                        "role adapter."
                                    )
                            else:
                                role_adapter = eraf.role_assignment_adapter
                                if role_adapter is None:
                                    raise RuntimeError(
                                        "V9.3 grounding requires its role adapter."
                                    )
                            if role_adapter is not None:
                                role_adapter.train()
                                role_adapter.requires_grad_(True)
                        elif self.policy_guard_eraf_training_stage == "action":
                            if (
                                self.policy_guard_eraf_grounding_objective_version
                                >= 26
                            ):
                                # ERAF itself and all legacy post-action
                                # residuals stay frozen. The one deployed path
                                # updates only shared Video/Action LoRA plus the
                                # internal ERAF context projection.
                                context_injector = self.policy_guard_modules[
                                    "eraf_action_context_injector"
                                ]
                                context_injector.train()
                                context_injector.requires_grad_(True)
                                adapter_ids = self._adapter_parameter_ids()
                                if not adapter_ids:
                                    raise ValueError(
                                        "PGC V9.26 produced no shared Expert LoRA "
                                        "parameters."
                                    )
                                self.mot.train()
                                self.video_expert.train()
                                self.action_expert.train()
                                for parameter in self.mot.parameters():
                                    if id(parameter) in adapter_ids:
                                        parameter.requires_grad_(True)
                            elif (
                                self.policy_guard_eraf_grounding_objective_version
                                >= 25
                            ):
                                # V9.25 changes only the injection boundary:
                                # ERAF tokens condition the shared frozen Action
                                # Expert during denoising. Every post-sampler
                                # Proposal/geometry/servo residual stays frozen
                                # and is bypassed by training and deployment.
                                context_injector = self.policy_guard_modules[
                                    "eraf_action_context_injector"
                                ]
                                context_injector.train()
                                context_injector.requires_grad_(True)
                            elif (
                                self.policy_guard_eraf_grounding_objective_version
                                >= 24
                            ):
                                # Preserve the admitted V9.21 action function
                                # exactly and train only an identity-initialized
                                # clause residual that may fall back toward Base.
                                clause_residual = self.policy_guard_modules[
                                    "eraf_clause_semantic_retention_residual"
                                ]
                                clause_residual.train()
                                clause_residual.requires_grad_(True)
                            elif (
                                self.policy_guard_eraf_grounding_objective_version
                                >= 21
                            ):
                                # Freeze the admitted semantic/geometry stack and
                                # learn only a zero-init, phase-specific full-action
                                # correction against expert action prefixes.
                                expert_adapter = self.policy_guard_modules[
                                    "eraf_phase_expert_residual_adapter"
                                ]
                                expert_adapter.train()
                                expert_adapter.requires_grad_(True)
                                if (
                                    self.policy_guard_eraf_grounding_objective_version
                                    == 23
                                ):
                                    teacher = self.policy_guard_modules[
                                        "eraf_phase_expert_residual_teacher"
                                    ]
                                    teacher.eval()
                                    teacher.requires_grad_(False)
                            elif (
                                self.policy_guard_eraf_grounding_objective_version
                                >= 20
                            ):
                                # V9.20 freezes the complete V9.19 fallback and
                                # learns only compatibility-gated local waypoints.
                                waypoint = self.policy_guard_modules[
                                    "eraf_phase_compatible_waypoint_adapter"
                                ]
                                waypoint.train()
                                waypoint.requires_grad_(True)
                            elif (
                                self.policy_guard_eraf_grounding_objective_version
                                >= 19
                            ):
                                # V9.19 freezes the admitted V9.18 stack and
                                # learns only the hard-clause, phase-specific
                                # Cartesian servo that can suppress the legacy
                                # soft geometry residual per phase.
                                phase_servo = self.policy_guard_modules[
                                    "eraf_hard_routed_phase_servo"
                                ]
                                phase_servo.train()
                                phase_servo.requires_grad_(True)
                            elif (
                                self.policy_guard_eraf_grounding_objective_version
                                >= 17
                            ):
                                # V9.17 preserves the validated ERAF, Proposal,
                                # and V9.16 semantic bridge and trains only the
                                # short EEF-relative geometry-to-action path.
                                geometry_adapter = self.policy_guard_modules[
                                    "eraf_geometry_action_adapter"
                                ]
                                geometry_adapter.train()
                                geometry_adapter.requires_grad_(True)
                            elif (
                                self.policy_guard_eraf_grounding_objective_version
                                >= 16
                            ):
                                # V9.16 is a narrowly scoped semantic calibration:
                                # preserve the V9.15 Proposal and every ERAF path,
                                # and update only the final causal action bridge.
                                action_grounding_bridge = self.policy_guard_modules[
                                    "eraf_action_grounding_bridge"
                                ]
                                action_grounding_bridge.train()
                                action_grounding_bridge.requires_grad_(True)
                            else:
                                proposal = self.policy_guard_modules[
                                    "action_chunk_proposal"
                                ]
                                proposal.train()
                                proposal.requires_grad_(True)
                            if (
                                self.policy_guard_eraf_action_joint_training
                                and self.policy_guard_eraf_grounding_objective_version
                                < 16
                            ):
                                eraf = self.policy_guard_modules[
                                    "entity_relation_affordance"
                                ]
                                for module_name in PGC_ERAF_ACTION_BRIDGE_MODULE_NAMES:
                                    bridge = getattr(eraf, module_name)
                                    bridge.train()
                                    bridge.requires_grad_(True)
                                if (
                                    self.policy_guard_eraf_grounding_objective_version
                                    >= 15
                                ):
                                    action_grounding_bridge = (
                                        self.policy_guard_modules[
                                            "eraf_action_grounding_bridge"
                                        ]
                                    )
                                    action_grounding_bridge.train()
                                    action_grounding_bridge.requires_grad_(True)
                        else:
                            verifier = self.policy_guard_modules["verifier"]
                            verifier.train()
                            verifier.requires_grad_(True)
                    else:
                        trainable_modules = {
                            "grounding": {"entity_relation_affordance"},
                            "action": {
                                "entity_relation_affordance",
                                "action_chunk_proposal",
                            },
                            "verifier": {"verifier"},
                        }[self.policy_guard_eraf_training_stage]
                        for name, module in self.policy_guard_modules.items():
                            if name in trainable_modules:
                                module.train()
                                module.requires_grad_(True)
                            else:
                                module.eval()
                                module.requires_grad_(False)
                if (
                    self.policy_guard_version == 5
                    and self.policy_guard_completion_phase_enabled
                    and self.policy_guard_completion_train_proposal_only
                ):
                    for name, module in self.policy_guard_modules.items():
                        if name == "action_chunk_proposal":
                            module.train()
                            module.requires_grad_(True)
                        else:
                            module.eval()
                            module.requires_grad_(False)
                if (
                    self.policy_guard_version == 8
                    and self.policy_guard_closed_loop_train_proposal_only
                ):
                    # V8 repairs the deployed V5 candidate only. The released
                    # Base plus V5 language and gate sidecars remain immutable.
                    for name, module in self.policy_guard_modules.items():
                        if name == "action_chunk_proposal":
                            module.train()
                            module.requires_grad_(True)
                        else:
                            module.eval()
                            module.requires_grad_(False)
                if self.policy_guard_version in {6, 7}:
                    # V6/V7 have no deployment edge through the legacy Goal Graph.
                    # Keep it only for checkpoint migration/history, but exclude
                    # it from the optimizer so language cannot regain a direct
                    # shortcut around the visual target bottleneck.
                    self.policy_guard_modules["goal_graph"].eval()
                    self.policy_guard_modules["goal_graph"].requires_grad_(False)
                trainable = sum(
                    parameter.numel()
                    for parameter in self.parameters()
                    if parameter.requires_grad
                )
                total = sum(parameter.numel() for parameter in self.parameters())
                if trainable <= 0:
                    raise ValueError("PGC v3+ produced zero trainable parameters.")
                return {"trainable": int(trainable), "total": int(total)}
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

    def policy_guard_optimizer_groups(
        self, default_learning_rate: float
    ) -> Optional[list[dict[str, Any]]]:
        """Return the explicit V9 stage-wise optimizer contract."""
        if not (self.policy_guard_enabled and self.policy_guard_version == 9):
            return None
        default_learning_rate = float(default_learning_rate)
        if default_learning_rate <= 0:
            raise ValueError("PGC v9 optimizer LR must be positive.")
        groups: list[dict[str, Any]] = []
        optimizer_modules = [
            (
                "entity_relation_affordance",
                (
                    default_learning_rate
                    if self.policy_guard_eraf_training_stage == "grounding"
                    else self.policy_guard_eraf_learning_rate
                ),
            ),
            ("action_chunk_proposal", default_learning_rate),
            ("verifier", default_learning_rate),
        ]
        if "eraf_action_grounding_bridge" in self.policy_guard_modules:
            optimizer_modules.append(
                (
                    "eraf_action_grounding_bridge",
                    self.policy_guard_eraf_action_grounding_learning_rate,
                )
            )
        if "eraf_geometry_action_adapter" in self.policy_guard_modules:
            optimizer_modules.append(
                (
                    "eraf_geometry_action_adapter",
                    self.policy_guard_eraf_action_geometry_learning_rate,
                )
            )
        if "eraf_hard_routed_phase_servo" in self.policy_guard_modules:
            optimizer_modules.append(
                (
                    "eraf_hard_routed_phase_servo",
                    self.policy_guard_eraf_action_geometry_learning_rate,
                )
            )
        if "eraf_phase_compatible_waypoint_adapter" in self.policy_guard_modules:
            optimizer_modules.append(
                (
                    "eraf_phase_compatible_waypoint_adapter",
                    self.policy_guard_eraf_action_geometry_learning_rate,
                )
            )
        if "eraf_phase_expert_residual_adapter" in self.policy_guard_modules:
            optimizer_modules.append(
                (
                    "eraf_phase_expert_residual_adapter",
                    self.policy_guard_eraf_action_geometry_learning_rate,
                )
            )
        if "eraf_clause_semantic_retention_residual" in self.policy_guard_modules:
            optimizer_modules.append(
                (
                    "eraf_clause_semantic_retention_residual",
                    self.policy_guard_eraf_action_geometry_learning_rate,
                )
            )
        if "eraf_action_context_injector" in self.policy_guard_modules:
            optimizer_modules.append(
                (
                    "eraf_action_context_injector",
                    self.policy_guard_eraf_action_geometry_learning_rate,
                )
            )
        for module_name, learning_rate in optimizer_modules:
            parameters = [
                parameter
                for parameter in self.policy_guard_modules[module_name].parameters()
                if parameter.requires_grad
            ]
            if parameters:
                groups.append(
                    {
                        "params": parameters,
                        "lr": float(learning_rate),
                        "pgc_v9_group": module_name,
                    }
                )
        if (
            self.policy_guard_eraf_grounding_objective_version >= 26
            and self.lora_enabled
        ):
            adapter_ids = self._adapter_parameter_ids()
            adapter_parameters = [
                parameter
                for parameter in self.parameters()
                if parameter.requires_grad and id(parameter) in adapter_ids
            ]
            if not adapter_parameters:
                raise RuntimeError(
                    "PGC V9.26 optimizer received no trainable shared Expert "
                    "LoRA parameters."
                )
            groups.append(
                {
                    "params": adapter_parameters,
                    "lr": default_learning_rate,
                    "pgc_v9_group": "shared_video_action_lora",
                }
            )
        grouped_ids = {
            id(parameter)
            for group in groups
            for parameter in group["params"]
        }
        expected_ids = {
            id(parameter)
            for parameter in self.parameters()
            if parameter.requires_grad
        }
        if grouped_ids != expected_ids:
            raise RuntimeError(
                "PGC v9 optimizer groups do not exactly cover the trainable "
                "sidecars."
            )
        return groups

    def _policy_guard_completion_only_state(
        self,
        policy_state: Optional[Mapping[str, torch.Tensor]],
        *,
        previous_state: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> Optional[dict[str, torch.Tensor]]:
        """Enforce the V9.14 cross-replan state contract inside the model.

        The learned four-state PSCM remains a current-frame diagnostic and
        routing feature, but HOLDING/RETRY/PENDING predictions are never fed
        back at the next replan.  Only an explicitly valid COMPLETED bit is
        allowed through this boundary.  Keeping this enforcement in FastWAM
        makes LIBERO and downstream RoboTwin callers share identical behavior
        instead of relying on a benchmark-specific wrapper.
        """
        if policy_state is None:
            return None
        if not self.policy_guard_eraf_completion_only_memory:
            return dict(policy_state)
        state_ids = policy_state.get("phase_safe_memory_state_ids")
        state_valid = policy_state.get("phase_safe_memory_valid")
        if state_ids is None or state_valid is None:
            raise ValueError(
                "V9.14 completion-only policy state requires state IDs and "
                "validity."
            )
        state_ids = torch.as_tensor(state_ids).long()
        state_valid = torch.as_tensor(state_valid).bool()
        if state_ids.shape != state_valid.shape:
            raise ValueError(
                "V9.14 completion-only state IDs and validity must have the "
                "same shape."
            )
        completed = state_valid & (state_ids == 3)
        if previous_state is not None:
            previous_ids = previous_state.get("phase_safe_memory_state_ids")
            previous_valid = previous_state.get("phase_safe_memory_valid")
            if previous_ids is None or previous_valid is None:
                raise ValueError(
                    "V9.14 previous completion-only state requires state IDs "
                    "and validity."
                )
            previous_ids = torch.as_tensor(previous_ids).to(
                device=state_ids.device, dtype=torch.long
            )
            previous_valid = torch.as_tensor(previous_valid).to(
                device=state_valid.device, dtype=torch.bool
            )
            if previous_ids.shape != state_ids.shape or (
                previous_valid.shape != state_valid.shape
            ):
                raise ValueError(
                    "V9.14 current and previous completion-only states must "
                    "have identical shapes."
                )
            completed = completed | (
                previous_valid & (previous_ids == 3)
            )
        sanitized = dict(policy_state)
        sanitized["phase_safe_memory_state_ids"] = torch.where(
            completed,
            torch.full_like(state_ids, 3),
            torch.zeros_like(state_ids),
        )
        sanitized["phase_safe_memory_valid"] = completed
        return sanitized

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
        language_context_len: Optional[int] = None,
        current_visual_hidden: Optional[torch.Tensor] = None,
        policy_guard_state: Optional[Mapping[str, torch.Tensor]] = None,
        policy_guard_eraf_oracle: Optional[Mapping[str, torch.Tensor]] = None,
        proprio: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if not self.policy_guard_enabled:
            raise RuntimeError("PGC goal encoding requires policy_guard.enabled=true.")
        current_token_count = min(
            int(video_tokens_per_frame), int(final_video_hidden.shape[1])
        )
        if current_token_count <= 0:
            raise ValueError("PGC requires at least one current visual token.")
        if self.policy_guard_version == 9:
            if current_visual_hidden is None:
                raise ValueError(
                    "PGC v9 requires language-neutral current visual tokens."
                )
            if language_context_len is None:
                language_context_len = int(context.shape[1])
            goal_queries, goal_embedding, _, goal_metrics = (
                self._encode_policy_guard_eraf(
                    final_video_hidden=final_video_hidden,
                    current_visual_hidden=current_visual_hidden,
                    video_tokens_per_frame=video_tokens_per_frame,
                    context=context,
                    context_mask=context_mask,
                    language_context_len=int(language_context_len),
                    policy_guard_state=policy_guard_state,
                    policy_guard_eraf_oracle=policy_guard_eraf_oracle,
                    proprio=proprio,
                )
            )
            return goal_queries, goal_embedding, goal_metrics
        if self.policy_guard_version in {6, 7}:
            if current_visual_hidden is None:
                raise ValueError(
                    "PGC v6/v7 requires language-neutral current visual tokens."
                )
            if language_context_len is None:
                language_context_len = int(context.shape[1])
            language_context_len = int(language_context_len)
            if not 0 < language_context_len <= int(context.shape[1]):
                raise ValueError(
                    "PGC v6/v7 language_context_len must select a non-empty "
                    "language prefix."
                )
            (
                binding_queries,
                binding_embedding,
                _,
                _,
                binding_metrics,
            ) = self._encode_policy_guard_target_binding(
                current_visual_hidden=current_visual_hidden,
                video_tokens_per_frame=video_tokens_per_frame,
                language_hidden=context[:, :language_context_len],
                language_mask=context_mask[:, :language_context_len],
            )
            return binding_queries, binding_embedding, binding_metrics
        if self.policy_guard_version >= 2:
            seed_module = self.policy_guard_modules["goal_query_seeds"]
            base_queries = seed_module.weight.unsqueeze(0).expand(
                final_video_hidden.shape[0], -1, -1
            )
        else:
            if self.policy_guard_action_expert is None:
                raise RuntimeError("PGC v1 goal queries require its Action Expert.")
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

    def _encode_policy_guard_eraf(
        self,
        *,
        final_video_hidden: torch.Tensor,
        current_visual_hidden: torch.Tensor,
        video_tokens_per_frame: int,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        language_context_len: int,
        policy_guard_state: Optional[Mapping[str, torch.Tensor]] = None,
        policy_guard_eraf_oracle: Optional[Mapping[str, torch.Tensor]] = None,
        proprio: Optional[torch.Tensor] = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
    ]:
        """Encode the frozen V5 goal, then add the zero-init ERAF bridge."""
        if not (self.policy_guard_enabled and self.policy_guard_version == 9):
            raise RuntimeError("ERAF goal encoding requires PGC v9.")
        current_token_count = min(
            int(video_tokens_per_frame),
            int(final_video_hidden.shape[1]),
            int(current_visual_hidden.shape[1]),
        )
        if current_token_count <= 0:
            raise ValueError("PGC v9 requires current-frame visual patches.")
        language_context_len = int(language_context_len)
        if not 0 < language_context_len <= int(context.shape[1]):
            raise ValueError(
                "PGC v9 language_context_len must select a non-empty language "
                "prefix."
            )
        seed_module = self.policy_guard_modules["goal_query_seeds"]
        base_queries = seed_module.weight.unsqueeze(0).expand(
            final_video_hidden.shape[0], -1, -1
        )
        detach_visual_backbone = (
            self.policy_guard_eraf_grounding_objective_version < 26
        )
        goal_graph_visual = final_video_hidden[:, :current_token_count]
        if detach_visual_backbone:
            goal_graph_visual = goal_graph_visual.detach()
        base_goal_queries, base_goal_embedding, base_metrics = (
            self.policy_guard_modules["goal_graph"](
                base_queries=base_queries,
                language_hidden=context,
                language_mask=context_mask,
                current_video_hidden=goal_graph_visual,
            )
        )
        eraf_module = self.policy_guard_modules["entity_relation_affordance"]
        policy_guard_state = self._policy_guard_completion_only_state(
            policy_guard_state
        )
        (
            routed_queries,
            routed_embedding,
            eraf_outputs,
            eraf_metrics,
        ) = eraf_module(
            base_goal_queries=(
                base_goal_queries.detach()
                if detach_visual_backbone
                else base_goal_queries
            ),
            base_goal_embedding=(
                base_goal_embedding.detach()
                if detach_visual_backbone
                else base_goal_embedding
            ),
            language_hidden=context[:, :language_context_len],
            language_mask=context_mask[:, :language_context_len],
            current_video_hidden=current_visual_hidden[
                :, :current_token_count
            ].detach(),
            policy_state=policy_guard_state,
            proprio=proprio,
        )
        if policy_guard_eraf_oracle is not None:
            if torch.is_grad_enabled():
                raise RuntimeError(
                    "Oracle ERAF labels are forbidden during training."
                )
            routed_queries, routed_embedding, eraf_outputs = (
                eraf_module.route_oracle(
                    base_goal_queries=base_goal_queries.detach(),
                    base_goal_embedding=base_goal_embedding.detach(),
                    outputs=eraf_outputs,
                    oracle=policy_guard_eraf_oracle,
                )
            )
        action_grounding_metrics: dict[str, torch.Tensor] = {}
        if self.policy_guard_eraf_grounding_objective_version >= 15:
            eraf_outputs = dict(eraf_outputs)
            eraf_outputs["pre_action_grounding_goal_queries"] = routed_queries
            bypass_action_grounding = bool(
                torch.as_tensor(
                    eraf_outputs.get("audit_bypass_bridge", False)
                ).all()
            )
            if not bypass_action_grounding:
                routed_queries, action_grounding_metrics = (
                    self.policy_guard_modules[
                        "eraf_action_grounding_bridge"
                    ](
                        goal_queries=routed_queries,
                        eraf_outputs=eraf_outputs,
                    )
                )
        # ``prepare_trainable_parameters`` intentionally leaves the frozen root
        # model in eval mode and toggles only the active sidecar module.  Root
        # ``self.training`` is therefore false during every PGC stage; use the
        # autograd context to distinguish verifier fitting from rollout.
        if (
            self.policy_guard_eraf_training_stage == "verifier"
            and torch.is_grad_enabled()
        ):
            for negative_kind in ("entity", "relation"):
                negative_queries, negative_embedding = (
                    eraf_module.negative_goal_queries(
                        base_goal_queries=base_goal_queries.detach(),
                        base_goal_embedding=base_goal_embedding.detach(),
                        outputs=eraf_outputs,
                        kind=negative_kind,
                    )
                )
                eraf_outputs[f"wrong_{negative_kind}_goal_queries"] = (
                    negative_queries
                )
                eraf_outputs[f"wrong_{negative_kind}_goal_embedding"] = (
                    negative_embedding
                )
        metrics = dict(base_metrics)
        metrics.update(eraf_metrics)
        metrics.update(action_grounding_metrics)
        metrics.update(
            {
                "pgc_v9_predicate_id_mean": eraf_outputs[
                    "predicate_logits"
                ].argmax(dim=-1).float().mean(),
                "pgc_v9_subject_anchor_norm": eraf_outputs[
                    "subject_position"
                ].float().norm(dim=-1).mean(),
                "pgc_v9_grasp_anchor_norm": eraf_outputs[
                    "grasp_anchor"
                ].float().norm(dim=-1).mean(),
                "pgc_v9_goal_anchor_norm": eraf_outputs[
                    "goal_anchor"
                ].float().norm(dim=-1).mean(),
                "pgc_v9_phase_id_mean": eraf_outputs[
                    "phase_logits"
                ].argmax(dim=-1).float().mean(),
            }
        )
        if not torch.is_grad_enabled():
            self._policy_guard_last_eraf_outputs = dict(eraf_outputs)
            diagnostic_names = (
                "active_logits",
                "predicate_logits",
                "subject_attention",
                "reference_attention",
                "subject_base_attention",
                "reference_base_attention",
                "subject_position",
                "reference_position",
                "subject_view_visibility_logits",
                "reference_view_visibility_logits",
                "subject_view_centers",
                "reference_view_centers",
                "subject_view_attention_mass",
                "reference_view_attention_mass",
                "subject_base_view_attention_mass",
                "reference_base_view_attention_mass",
                "subject_view_gate_residual_logits",
                "reference_view_gate_residual_logits",
                "subject_view_attention_delta",
                "reference_view_attention_delta",
                "grasp_anchor",
                "goal_anchor",
                "interaction_anchor",
                "predicate_truth_logits",
                "phase_logits",
                "clause_execution_logits",
                "clause_execution_probability",
                "clause_routing_residual",
                "clause_routing_multiplier",
                "view_scheduler_enabled",
                "pre_rebinding_subject_attention",
                "pre_rebinding_reference_attention",
                "pre_rebinding_subject_position",
                "pre_rebinding_reference_position",
                "pre_rebinding_subject_view_attention_mass",
                "pre_rebinding_reference_view_attention_mass",
                "pre_rebinding_goal_anchor",
                "pre_rebinding_predicate_truth_logits",
                "pre_rebinding_phase_logits",
                "pre_rebinding_clause_execution_probability",
                "pre_rebinding_clause_routing_residual",
                "closed_loop_rebinding_enabled",
                "phase_safe_memory_enabled",
                "phase_safe_memory_state_logits",
                "phase_safe_memory_state_probability",
                "phase_safe_memory_previous_state_ids",
                "phase_safe_memory_previous_state_valid",
                "phase_safe_memory_next_state_ids",
                "phase_safe_memory_next_state_valid",
                "phase_safe_memory_routing_residual",
                "phase_safe_memory_completed_sticky",
                "phase_safe_memory_released_unsatisfied_retry",
                "pre_memory_subject_attention",
                "pre_memory_reference_attention",
                "pre_memory_subject_position",
                "pre_memory_reference_position",
                "pre_memory_goal_anchor",
                "spatial_coordinates",
                "camera_ids",
            )
            if "oracle_eraf_enabled" in eraf_outputs:
                diagnostic_names = diagnostic_names + (
                    "oracle_eraf_enabled",
                    "oracle_selected_clause",
                    "oracle_clause_valid",
                )
            self._policy_guard_last_eraf_diagnostics = {
                name: eraf_outputs[name].detach().float().cpu()
                if eraf_outputs[name].is_floating_point()
                else eraf_outputs[name].detach().cpu()
                for name in diagnostic_names
            }
        return routed_queries, routed_embedding, eraf_outputs, metrics

    def _encode_policy_guard_target_binding(
        self,
        *,
        current_visual_hidden: torch.Tensor,
        video_tokens_per_frame: int,
        language_hidden: torch.Tensor,
        language_mask: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        if (
            not self.policy_guard_enabled
            or self.policy_guard_version not in {6, 7}
        ):
            raise RuntimeError("Visual target binding requires PGC v6/v7.")
        current_token_count = min(
            int(video_tokens_per_frame), int(current_visual_hidden.shape[1])
        )
        if current_token_count <= 0:
            raise ValueError("PGC v6/v7 requires current-frame visual patches.")
        seed_module = self.policy_guard_modules["goal_query_seeds"]
        base_queries = seed_module.weight.unsqueeze(0).expand(
            current_visual_hidden.shape[0], -1, -1
        )
        return self.policy_guard_modules["target_binder"](
            base_queries=base_queries,
            language_hidden=language_hidden,
            language_mask=language_mask,
            current_video_hidden=(
                current_visual_hidden[:, :current_token_count].detach()
            ),
        )

    def _apply_policy_guard_v3_velocity_residual(
        self,
        *,
        base_action_hidden: torch.Tensor,
        base_action_velocity: torch.Tensor,
        routed_goal_queries: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if not self.policy_guard_enabled or self.policy_guard_version != 3:
            raise RuntimeError("PGC bounded velocity residual requires version=3.")
        residual, metrics = self.policy_guard_modules[
            "action_velocity_residual"
        ](
            base_action_hidden.detach(),
            routed_goal_queries,
        )
        if residual.shape != base_action_velocity.shape:
            raise ValueError(
                "PGC v3 velocity residual and Base velocity must share shape: "
                f"{tuple(residual.shape)} vs "
                f"{tuple(base_action_velocity.shape)}."
            )
        counterfactual_velocity = base_action_velocity.detach() + residual
        return counterfactual_velocity, residual, metrics

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
        if self.policy_guard_version >= 3:
            raise RuntimeError(
                "PGC v3+ do not use an independent action-token branch."
            )
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
        checkpoint_frozen_action_expert: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if (
            self.policy_guard_version == 9
            and self.policy_guard_eraf_grounding_objective_version >= 25
        ):
            (
                injected_context,
                injected_context_mask,
                injection_metrics,
            ) = self.policy_guard_modules["eraf_action_context_injector"](
                context=context,
                context_mask=full_context_mask,
                goal_queries=routed_goal_queries,
            )
            action_pre = self.action_expert.pre_dit(
                action_tokens=action_tokens,
                timestep=timestep_action,
                context=injected_context,
                context_mask=injected_context_mask,
                use_queries=False,
            )
            attention_mask = self._build_mot_attention_mask(
                video_seq_len=video_seq_len,
                action_seq_len=int(action_pre["tokens"].shape[1]),
                video_tokens_per_frame=video_tokens_per_frame,
                device=action_pre["tokens"].device,
                num_queries=0,
                action_reads_raw_video=True,
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
                action_expert=self.action_expert,
                training_override=checkpoint_frozen_action_expert,
            )
            output = self.action_expert.post_dit(output_tokens, action_pre)
            if return_metrics:
                return output, injection_metrics
            return output
        if self.policy_guard_version == 3:
            action_pre = self.action_expert.pre_dit(
                action_tokens=action_tokens,
                timestep=timestep_action,
                context=context,
                context_mask=full_context_mask,
                use_queries=False,
            )
            attention_mask = self._build_mot_attention_mask(
                video_seq_len=video_seq_len,
                action_seq_len=int(action_pre["tokens"].shape[1]),
                video_tokens_per_frame=video_tokens_per_frame,
                device=action_pre["tokens"].device,
                num_queries=0,
                action_reads_raw_video=True,
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
                action_expert=self.action_expert,
            )
            base_velocity = self.action_expert.post_dit(
                output_tokens, action_pre
            )
            output, _, metrics = (
                self._apply_policy_guard_v3_velocity_residual(
                    base_action_hidden=output_tokens,
                    base_action_velocity=base_velocity,
                    routed_goal_queries=routed_goal_queries,
                )
            )
            if return_metrics:
                return output, metrics
            return output
        if self.policy_guard_version >= 4:
            raise RuntimeError(
                "PGC v4 applies its proposal after the frozen Base sampler, "
                "not inside a diffusion step (except the V9.25 internal "
                "ERAF context-injection path)."
            )
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

    def _compute_policy_guard_v3_action_losses(
        self,
        *,
        predicted_residual: torch.Tensor,
        base_action_teacher: torch.Tensor,
        target_action: torch.Tensor,
        action_weight: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
        is_counterfactual: torch.Tensor,
        direct_action_valid: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        """Supervise a bounded correction while keeping native residual zero."""
        if self.policy_guard_version != 3:
            raise RuntimeError("PGC v3 residual losses require version=3.")
        if predicted_residual.shape != target_action.shape or (
            predicted_residual.shape != base_action_teacher.shape
        ):
            raise ValueError(
                "PGC v3 residual, Base velocity, and target velocity must "
                "share shape."
            )
        is_counterfactual = is_counterfactual.to(
            device=predicted_residual.device, dtype=torch.bool
        )
        direct_action_valid = direct_action_valid.to(
            device=predicted_residual.device, dtype=torch.bool
        )
        if is_counterfactual.shape != direct_action_valid.shape or (
            is_counterfactual.ndim != 1
        ):
            raise ValueError("PGC v3 sample masks must share [B] shape.")
        batch_size = int(is_counterfactual.shape[0])
        if action_weight.ndim == 0:
            action_weight = action_weight.expand(batch_size)
        elif action_weight.numel() == batch_size:
            action_weight = action_weight.reshape(batch_size)
        else:
            raise ValueError(
                "PGC v3 action weights must contain exactly one value per "
                f"sample, got shape {tuple(action_weight.shape)} for "
                f"batch size {batch_size}."
            )
        action_weight = action_weight.to(
            device=predicted_residual.device, dtype=torch.float32
        )

        native_valid = direct_action_valid & ~is_counterfactual
        counterfactual_valid = direct_action_valid & is_counterfactual
        target_residual = target_action - base_action_teacher.detach()
        counterfactual_error = self._compute_action_loss_per_sample(
            pred_action=predicted_residual,
            target_action=target_residual,
            action_is_pad=action_is_pad,
        )
        native_zero_error = self._compute_action_loss_per_sample(
            pred_action=predicted_residual,
            target_action=torch.zeros_like(predicted_residual),
            action_is_pad=action_is_pad,
        )
        counterfactual_action_loss = self._masked_policy_guard_mean(
            counterfactual_error * action_weight,
            counterfactual_valid,
        )
        native_zero_loss = self._masked_policy_guard_mean(
            native_zero_error * action_weight,
            native_valid,
        )

        def _per_sample_step_mean(values: torch.Tensor) -> torch.Tensor:
            if action_is_pad is None:
                return values.mean(dim=1)
            valid_steps = (~action_is_pad).to(
                device=values.device, dtype=values.dtype
            )
            return (values * valid_steps).sum(dim=1) / valid_steps.sum(
                dim=1
            ).clamp_min(1.0)

        residual_step_magnitude = predicted_residual.float().square().mean(
            dim=-1
        )
        residual_magnitude = _per_sample_step_mean(
            residual_step_magnitude
        )
        residual_regularization_loss = self._masked_policy_guard_mean(
            residual_magnitude,
            counterfactual_valid,
        )

        if predicted_residual.shape[1] > 1:
            residual_delta = (
                predicted_residual[:, 1:].float()
                - predicted_residual[:, :-1].float()
            ).square().mean(dim=-1)
            if action_is_pad is None:
                residual_smoothness = residual_delta.mean(dim=1)
            else:
                pair_valid = (
                    ~action_is_pad[:, 1:]
                    & ~action_is_pad[:, :-1]
                ).to(device=residual_delta.device, dtype=residual_delta.dtype)
                residual_smoothness = (
                    residual_delta * pair_valid
                ).sum(dim=1) / pair_valid.sum(dim=1).clamp_min(1.0)
        else:
            residual_smoothness = residual_magnitude * 0.0
        residual_smoothness_loss = self._masked_policy_guard_mean(
            residual_smoothness,
            counterfactual_valid,
        )

        cap = self.policy_guard_modules[
            "action_velocity_residual"
        ].max_abs.to(
            device=target_residual.device, dtype=target_residual.dtype
        )
        outside_cap_step = (
            target_residual.detach().abs() > cap
        ).float().mean(dim=-1)
        outside_cap = _per_sample_step_mean(outside_cap_step)
        target_norm = _per_sample_step_mean(
            target_residual.detach().float().norm(dim=-1)
        )
        metrics = {
            "pgc_native_fraction": (~is_counterfactual).float().mean(),
            "pgc_counterfactual_fraction": is_counterfactual.float().mean(),
            "pgc_native_valid_fraction": native_valid.float().mean(),
            "pgc_counterfactual_valid_fraction": (
                counterfactual_valid.float().mean()
            ),
            "pgc_native_residual_zero_mse": (
                self._masked_policy_guard_mean(
                    native_zero_error, native_valid
                ).detach()
            ),
            "pgc_counterfactual_residual_target_mse": (
                self._masked_policy_guard_mean(
                    counterfactual_error, counterfactual_valid
                ).detach()
            ),
            "pgc_counterfactual_target_residual_norm": (
                self._masked_policy_guard_mean(
                    target_norm, counterfactual_valid
                ).detach()
            ),
            "pgc_counterfactual_target_outside_cap_fraction": (
                self._masked_policy_guard_mean(
                    outside_cap, counterfactual_valid
                ).detach()
            ),
            "pgc_residual_regularization": (
                residual_regularization_loss.detach()
            ),
            "pgc_residual_temporal_smoothness": (
                residual_smoothness_loss.detach()
            ),
        }
        return (
            counterfactual_action_loss,
            native_zero_loss,
            residual_regularization_loss,
            residual_smoothness_loss,
            metrics,
        )

    def _policy_guard_v4_dimension_weight(
        self, reference: torch.Tensor
    ) -> torch.Tensor:
        dimension_weight = torch.ones(
            reference.shape[-1],
            device=reference.device,
            dtype=torch.float32,
        )
        if dimension_weight.numel() > 0:
            dimension_weight[-1] = self.policy_guard_action_gripper_weight
        return dimension_weight / dimension_weight.mean().clamp_min(1.0e-6)

    def _compute_policy_guard_v4_weighted_action_mse_per_sample(
        self,
        *,
        prediction: torch.Tensor,
        target: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if prediction.shape != target.shape or prediction.ndim != 3:
            raise ValueError("PGC v4 action-MSE inputs must share [B,T,A] shape.")
        dimension_weight = self._policy_guard_v4_dimension_weight(prediction)
        per_step = (
            (prediction.float() - target.float()).square() * dimension_weight
        ).mean(dim=-1)
        if action_is_pad is None:
            return per_step.mean(dim=1)
        valid_steps = (~action_is_pad).to(
            device=per_step.device, dtype=per_step.dtype
        )
        return (per_step * valid_steps).sum(dim=1) / valid_steps.sum(
            dim=1
        ).clamp_min(1.0)

    def _compute_policy_guard_v4_action_losses(
        self,
        *,
        proposed_action: torch.Tensor,
        predicted_residual: torch.Tensor,
        base_action: torch.Tensor,
        target_action: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
        is_counterfactual: torch.Tensor,
        direct_action_valid: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        """Directly supervise the final action chunk deployed by PGC v4."""
        if self.policy_guard_version != 4:
            raise RuntimeError("PGC v4 final-action losses require version=4.")
        if not (
            proposed_action.shape
            == predicted_residual.shape
            == base_action.shape
            == target_action.shape
        ):
            raise ValueError(
                "PGC v4 proposal, residual, Base, and target actions must "
                "share shape."
            )
        is_counterfactual = is_counterfactual.to(
            device=proposed_action.device, dtype=torch.bool
        )
        direct_action_valid = direct_action_valid.to(
            device=proposed_action.device, dtype=torch.bool
        )
        if is_counterfactual.shape != direct_action_valid.shape or (
            is_counterfactual.ndim != 1
        ):
            raise ValueError("PGC v4 provenance masks must share [B] shape.")
        native_valid = direct_action_valid & ~is_counterfactual
        counterfactual_valid = direct_action_valid & is_counterfactual

        dimension_weight = self._policy_guard_v4_dimension_weight(
            proposed_action
        )

        def _per_sample_huber(
            prediction: torch.Tensor, target: torch.Tensor
        ) -> torch.Tensor:
            per_dimension = F.smooth_l1_loss(
                prediction.float(), target.float(), reduction="none"
            )
            per_step = (per_dimension * dimension_weight).mean(dim=-1)
            if action_is_pad is None:
                return per_step.mean(dim=1)
            valid_steps = (~action_is_pad).to(
                device=per_step.device, dtype=per_step.dtype
            )
            return (per_step * valid_steps).sum(dim=1) / valid_steps.sum(
                dim=1
            ).clamp_min(1.0)

        proposal_error = _per_sample_huber(proposed_action, target_action)
        native_zero_error = _per_sample_huber(
            predicted_residual, torch.zeros_like(predicted_residual)
        )
        counterfactual_action_loss = self._masked_policy_guard_mean(
            proposal_error, counterfactual_valid
        )
        native_zero_loss = self._masked_policy_guard_mean(
            native_zero_error, native_valid
        )

        residual_step_magnitude = predicted_residual.float().square().mean(
            dim=-1
        )
        if action_is_pad is None:
            residual_magnitude = residual_step_magnitude.mean(dim=1)
        else:
            valid_steps = (~action_is_pad).to(
                device=residual_step_magnitude.device,
                dtype=residual_step_magnitude.dtype,
            )
            residual_magnitude = (
                residual_step_magnitude * valid_steps
            ).sum(dim=1) / valid_steps.sum(dim=1).clamp_min(1.0)
        residual_regularization_loss = self._masked_policy_guard_mean(
            residual_magnitude, counterfactual_valid
        )

        if predicted_residual.shape[1] > 1:
            residual_delta = (
                predicted_residual[:, 1:].float()
                - predicted_residual[:, :-1].float()
            ).square().mean(dim=-1)
            if action_is_pad is None:
                residual_smoothness = residual_delta.mean(dim=1)
            else:
                pair_valid = (
                    ~action_is_pad[:, 1:] & ~action_is_pad[:, :-1]
                ).to(device=residual_delta.device, dtype=residual_delta.dtype)
                residual_smoothness = (
                    residual_delta * pair_valid
                ).sum(dim=1) / pair_valid.sum(dim=1).clamp_min(1.0)
        else:
            residual_smoothness = residual_magnitude * 0.0
        residual_smoothness_loss = self._masked_policy_guard_mean(
            residual_smoothness, counterfactual_valid
        )

        base_mse = self._compute_policy_guard_v4_weighted_action_mse_per_sample(
            prediction=base_action.detach(),
            target=target_action,
            action_is_pad=action_is_pad,
        )
        proposal_mse = self._compute_policy_guard_v4_weighted_action_mse_per_sample(
            prediction=proposed_action,
            target=target_action,
            action_is_pad=action_is_pad,
        )
        target_residual = target_action - base_action.detach()
        cap = self.policy_guard_modules["action_chunk_proposal"].max_abs.to(
            device=target_residual.device, dtype=target_residual.dtype
        )
        outside_cap = (target_residual.detach().abs() > cap).float().mean(
            dim=(1, 2)
        )
        metrics = {
            "pgc_native_fraction": (~is_counterfactual).float().mean(),
            "pgc_counterfactual_fraction": is_counterfactual.float().mean(),
            "pgc_native_valid_fraction": native_valid.float().mean(),
            "pgc_counterfactual_valid_fraction": counterfactual_valid.float().mean(),
            "pgc_v4_final_action_mse_base": self._masked_policy_guard_mean(
                base_mse, counterfactual_valid
            ).detach(),
            "pgc_v4_final_action_mse_proposal": self._masked_policy_guard_mean(
                proposal_mse, counterfactual_valid
            ).detach(),
            "pgc_v4_final_action_mse_improvement": self._masked_policy_guard_mean(
                base_mse - proposal_mse, counterfactual_valid
            ).detach(),
            "pgc_v4_native_final_action_gap": self._masked_policy_guard_mean(
                predicted_residual.float().square().mean(dim=(1, 2)),
                native_valid,
            ).detach(),
            "pgc_v4_target_outside_cap_fraction": self._masked_policy_guard_mean(
                outside_cap, counterfactual_valid
            ).detach(),
            "pgc_v4_residual_regularization": residual_regularization_loss.detach(),
            "pgc_v4_residual_temporal_smoothness": residual_smoothness_loss.detach(),
        }
        return (
            counterfactual_action_loss,
            native_zero_loss,
            residual_regularization_loss,
            residual_smoothness_loss,
            metrics,
        )

    def _compute_policy_guard_v4_verifier_loss(
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
        """Train an FP32 raw-logit advantage on deployed action candidates."""
        if self.policy_guard_version < 4:
            raise RuntimeError("PGC pairwise verifier loss requires version>=4.")
        verifier = self.policy_guard_modules["verifier"]
        (
            advantage,
            base_value,
            counterfactual_value,
            goal_state,
            _,
            counterfactual_embedding,
        ) = verifier(
            current_video_hidden=current_video_hidden.detach(),
            # Keep the frozen visual tokens detached, but allow the delayed
            # verifier/alignment objective to improve the Goal Graph language
            # representation once the action proposal has stabilized.
            goal_embedding=goal_embedding,
            base_action=base_candidate_action.detach(),
            counterfactual_action=counterfactual_candidate_action.detach(),
            action_is_pad=action_is_pad,
        )
        demonstrated_embedding = verifier.encode_action(
            demonstrated_action.detach(), action_is_pad
        )

        valid = direct_action_valid.to(
            device=advantage.device, dtype=torch.bool
        )
        is_counterfactual = is_counterfactual.to(
            device=advantage.device, dtype=torch.bool
        )
        counterfactual_valid = valid & is_counterfactual
        base_error = self._compute_policy_guard_v4_weighted_action_mse_per_sample(
            prediction=base_candidate_action.detach(),
            target=demonstrated_action.detach(),
            action_is_pad=action_is_pad,
        )
        proposal_error = self._compute_policy_guard_v4_weighted_action_mse_per_sample(
            prediction=counterfactual_candidate_action.detach(),
            target=demonstrated_action.detach(),
            action_is_pad=action_is_pad,
        )
        target_advantage = (
            (base_error - proposal_error)
            / self.policy_guard_advantage_temperature
        ).clamp(
            min=-self.policy_guard_advantage_clip,
            max=self.policy_guard_advantage_clip,
        )
        # Equal native candidates are an abstention target. A positive gate
        # threshold then deterministically falls back to Base.
        target_advantage = torch.where(
            counterfactual_valid,
            target_advantage,
            torch.zeros_like(target_advantage),
        )
        regression = F.smooth_l1_loss(
            advantage.float(), target_advantage.float(), reduction="none"
        )
        loss_regression = self._masked_policy_guard_mean(regression, valid)

        ranking_valid = counterfactual_valid & (
            target_advantage.abs() >= self.policy_guard_gate_threshold
        )
        preference = torch.sign(target_advantage).clamp(min=-1.0, max=1.0)
        ranking = F.softplus(-preference * advantage.float())
        loss_ranking = self._masked_policy_guard_mean(ranking, ranking_valid)
        # On native/equal candidates any positive margin is a false override.
        loss_native_fallback = self._masked_policy_guard_mean(
            torch.relu(advantage.float()), valid & ~is_counterfactual
        )
        verifier_loss = loss_regression + loss_ranking + loss_native_fallback

        if self.policy_guard_alignment_loss is None:
            raise RuntimeError("PGC alignment loss was not initialized.")
        alignment_loss, alignment_metrics = self.policy_guard_alignment_loss(
            goal_state,
            demonstrated_embedding,
            group_ids=goal_ids,
        )
        predicted_override = advantage >= self.policy_guard_gate_threshold
        metrics = {
            "loss_pgc_verifier_advantage_regression": loss_regression.detach(),
            "loss_pgc_verifier_advantage_ranking": loss_ranking.detach(),
            "loss_pgc_verifier_native_fallback": loss_native_fallback.detach(),
            "pgc_v4_verifier_base_value": base_value.detach().mean(),
            "pgc_v4_verifier_counterfactual_value": counterfactual_value.detach().mean(),
            "pgc_v4_verifier_advantage": advantage.detach().mean(),
            "pgc_v4_verifier_target_advantage": target_advantage.detach().mean(),
            "pgc_v4_verifier_ranking_valid_fraction": ranking_valid.float().mean(),
            "pgc_direct_counterfactual_fraction": is_counterfactual.float().mean(),
            "pgc_direct_action_valid_fraction": valid.float().mean(),
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

    def _policy_guard_v5_temporal_weight(
        self,
        reference: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Weight the action prefix that LIBERO actually executes before replanning."""
        if reference.ndim != 3:
            raise ValueError("PGC v5 temporal weighting requires [B,T,A].")
        prefix = min(
            self.policy_guard_execution_prefix_steps,
            int(reference.shape[1]),
        )
        weight = torch.full(
            reference.shape[:2],
            self.policy_guard_suffix_loss_weight,
            device=reference.device,
            dtype=torch.float32,
        )
        weight[:, :prefix] = 1.0
        if action_is_pad is not None:
            if action_is_pad.shape != reference.shape[:2]:
                raise ValueError("PGC v5 action padding mask must be [B,T].")
            weight = weight * (~action_is_pad).to(
                device=weight.device, dtype=weight.dtype
            )
        return weight

    @staticmethod
    def _policy_guard_weighted_temporal_mean(
        per_step: torch.Tensor,
        temporal_weight: torch.Tensor,
    ) -> torch.Tensor:
        if per_step.shape != temporal_weight.shape or per_step.ndim != 2:
            raise ValueError("PGC temporal values/weights must share [B,T].")
        return (per_step * temporal_weight).sum(dim=1) / temporal_weight.sum(
            dim=1
        ).clamp_min(1.0e-6)

    def _compute_policy_guard_v5_weighted_action_mse_per_sample(
        self,
        *,
        prediction: torch.Tensor,
        target: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if prediction.shape != target.shape or prediction.ndim != 3:
            raise ValueError("PGC v5 action-MSE inputs must share [B,T,A].")
        dimension_weight = self._policy_guard_v4_dimension_weight(prediction)
        per_step = (
            (prediction.float() - target.float()).square() * dimension_weight
        ).mean(dim=-1)
        return self._policy_guard_weighted_temporal_mean(
            per_step,
            self._policy_guard_v5_temporal_weight(prediction, action_is_pad),
        )

    def _compute_policy_guard_v5_action_losses(
        self,
        *,
        proposed_action: torch.Tensor,
        predicted_residual: torch.Tensor,
        source_predicted_residual: torch.Tensor,
        base_action: torch.Tensor,
        target_action: torch.Tensor,
        counterfactual_goal_embedding: torch.Tensor,
        source_goal_embedding: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
        is_counterfactual: torch.Tensor,
        direct_action_valid: torch.Tensor,
        paired_language_valid: torch.Tensor,
        is_closed_loop_corrective: Optional[torch.Tensor] = None,
        completion_phase: Optional[torch.Tensor] = None,
        completion_phase_valid: Optional[torch.Tensor] = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        """PGC v5 paired-language and execution-prefix proposal objectives."""
        if self.policy_guard_version not in {5, 6, 7, 8, 9}:
            raise RuntimeError(
                "PGC paired action losses require version 5, 6, 7, 8, or 9."
            )
        if not (
            proposed_action.shape
            == predicted_residual.shape
            == source_predicted_residual.shape
            == base_action.shape
            == target_action.shape
        ):
            raise ValueError("PGC v5 action tensors must share [B,T,A] shape.")
        if counterfactual_goal_embedding.shape != source_goal_embedding.shape:
            raise ValueError("PGC v5 paired goal embeddings must share shape.")

        is_counterfactual = is_counterfactual.to(
            device=proposed_action.device, dtype=torch.bool
        )
        direct_action_valid = direct_action_valid.to(
            device=proposed_action.device, dtype=torch.bool
        )
        paired_language_valid = paired_language_valid.to(
            device=proposed_action.device, dtype=torch.bool
        )
        if not (
            is_counterfactual.shape
            == direct_action_valid.shape
            == paired_language_valid.shape
        ) or is_counterfactual.ndim != 1:
            raise ValueError("PGC v5 provenance masks must share [B] shape.")
        native_valid = direct_action_valid & ~is_counterfactual
        counterfactual_valid = direct_action_valid & is_counterfactual
        paired_valid = counterfactual_valid & paired_language_valid
        closed_loop_valid = torch.zeros_like(counterfactual_valid)
        offline_counterfactual_valid = counterfactual_valid
        acquisition_sample_weight = torch.ones_like(
            proposed_action[:, 0, 0], dtype=torch.float32
        )
        if self.policy_guard_version == 8:
            if is_closed_loop_corrective is None:
                raise ValueError(
                    "PGC v8 action loss requires closed-loop provenance."
                )
            is_closed_loop_corrective = is_closed_loop_corrective.to(
                device=proposed_action.device, dtype=torch.bool
            )
            if is_closed_loop_corrective.shape != is_counterfactual.shape:
                raise ValueError(
                    "PGC v8 closed-loop provenance must share [B] shape."
                )
            if bool((is_closed_loop_corrective & ~is_counterfactual).any()):
                raise ValueError(
                    "PGC v8 corrective samples must be counterfactual samples."
                )
            closed_loop_valid = (
                counterfactual_valid & is_closed_loop_corrective
            )
            offline_counterfactual_valid = (
                counterfactual_valid & ~is_closed_loop_corrective
            )
            acquisition_sample_weight = torch.where(
                is_closed_loop_corrective,
                acquisition_sample_weight.new_tensor(
                    self.policy_guard_closed_loop_corrective_weight
                ),
                acquisition_sample_weight.new_tensor(
                    self.policy_guard_offline_acquisition_weight
                ),
            )

        dimension_weight = self._policy_guard_v4_dimension_weight(
            proposed_action
        )
        temporal_weight = self._policy_guard_v5_temporal_weight(
            proposed_action, action_is_pad
        )

        def _per_sample_huber(
            prediction: torch.Tensor, target: torch.Tensor
        ) -> torch.Tensor:
            per_dimension = F.smooth_l1_loss(
                prediction.float(), target.float(), reduction="none"
            )
            per_step = (per_dimension * dimension_weight).mean(dim=-1)
            return self._policy_guard_weighted_temporal_mean(
                per_step, temporal_weight
            )

        proposal_error = _per_sample_huber(proposed_action, target_action)
        current_zero_error = _per_sample_huber(
            predicted_residual, torch.zeros_like(predicted_residual)
        )
        source_zero_error = _per_sample_huber(
            source_predicted_residual,
            torch.zeros_like(source_predicted_residual),
        )
        completion_sample_weight = torch.ones_like(
            proposal_error, dtype=torch.float32
        )
        completion_transport = torch.zeros_like(counterfactual_valid)
        completion_release = torch.zeros_like(counterfactual_valid)
        if self.policy_guard_completion_phase_enabled:
            if completion_phase is None or completion_phase_valid is None:
                raise ValueError(
                    "PGC V5-completion action loss requires phase/value mask."
                )
            completion_phase = completion_phase.to(
                device=proposed_action.device, dtype=torch.long
            )
            completion_phase_valid = completion_phase_valid.to(
                device=proposed_action.device, dtype=torch.bool
            )
            if completion_phase.shape != is_counterfactual.shape or (
                completion_phase_valid.shape != is_counterfactual.shape
            ):
                raise ValueError(
                    "PGC V5-completion phase/value mask must share [B] shape."
                )
            completion_transport = completion_phase_valid & (completion_phase == 1)
            completion_release = completion_phase_valid & (completion_phase == 2)
            completion_sample_weight = torch.where(
                completion_transport,
                completion_sample_weight.new_tensor(
                    self.policy_guard_completion_transport_weight
                ),
                completion_sample_weight,
            )
            completion_sample_weight = torch.where(
                completion_release,
                completion_sample_weight.new_tensor(
                    self.policy_guard_completion_release_weight
                ),
                completion_sample_weight,
            )
        # Do not renormalize by the sum of phase weights. With micro-batch one,
        # that would cancel the intended stronger gradient on rare completion
        # states. Native and pre-grasp samples retain their original scale.
        counterfactual_action_loss = self._masked_policy_guard_mean(
            proposal_error
            * completion_sample_weight
            * acquisition_sample_weight,
            counterfactual_valid,
        )
        native_zero_loss = self._masked_policy_guard_mean(
            current_zero_error, native_valid
        )
        same_state_source_zero_loss = self._masked_policy_guard_mean(
            source_zero_error, paired_valid
        )

        goal_cosine_distance = 1.0 - F.cosine_similarity(
            counterfactual_goal_embedding.float(),
            source_goal_embedding.float(),
            dim=-1,
        )
        goal_separation_per_sample = torch.relu(
            self.policy_guard_goal_separation_margin - goal_cosine_distance
        )
        goal_separation_loss = self._masked_policy_guard_mean(
            goal_separation_per_sample, paired_valid
        )

        residual_difference_per_step = (
            predicted_residual.float() - source_predicted_residual.float()
        ).square().mean(dim=-1)
        residual_pair_distance = self._policy_guard_weighted_temporal_mean(
            residual_difference_per_step, temporal_weight
        ).clamp_min(1.0e-12).sqrt()
        target_residual_per_step = (
            target_action.float() - base_action.detach().float()
        ).square().mean(dim=-1)
        target_residual_distance = self._policy_guard_weighted_temporal_mean(
            target_residual_per_step, temporal_weight
        ).clamp_min(1.0e-12).sqrt()
        required_residual_distance = torch.minimum(
            target_residual_distance.detach(),
            target_residual_distance.new_full(
                target_residual_distance.shape,
                self.policy_guard_residual_separation_margin,
            ),
        )
        residual_separation_loss = self._masked_policy_guard_mean(
            torch.relu(required_residual_distance - residual_pair_distance),
            paired_valid,
        )

        residual_magnitude = self._policy_guard_weighted_temporal_mean(
            predicted_residual.float().square().mean(dim=-1),
            temporal_weight,
        )
        residual_regularization_loss = self._masked_policy_guard_mean(
            residual_magnitude, counterfactual_valid
        )
        if predicted_residual.shape[1] > 1:
            residual_delta = (
                predicted_residual[:, 1:].float()
                - predicted_residual[:, :-1].float()
            ).square().mean(dim=-1)
            pair_weight = torch.minimum(
                temporal_weight[:, 1:], temporal_weight[:, :-1]
            )
            residual_smoothness = self._policy_guard_weighted_temporal_mean(
                residual_delta, pair_weight
            )
        else:
            residual_smoothness = residual_magnitude * 0.0
        residual_smoothness_loss = self._masked_policy_guard_mean(
            residual_smoothness, counterfactual_valid
        )

        prefix = min(
            self.policy_guard_execution_prefix_steps,
            int(proposed_action.shape[1]),
        )
        prefix_pad = (
            None if action_is_pad is None else action_is_pad[:, :prefix]
        )
        prefix_base_mse = self._compute_policy_guard_v4_weighted_action_mse_per_sample(
            prediction=base_action.detach()[:, :prefix],
            target=target_action[:, :prefix],
            action_is_pad=prefix_pad,
        )
        prefix_proposal_mse = self._compute_policy_guard_v4_weighted_action_mse_per_sample(
            prediction=proposed_action[:, :prefix],
            target=target_action[:, :prefix],
            action_is_pad=prefix_pad,
        )
        full_base_mse = self._compute_policy_guard_v5_weighted_action_mse_per_sample(
            prediction=base_action.detach(),
            target=target_action,
            action_is_pad=action_is_pad,
        )
        full_proposal_mse = self._compute_policy_guard_v5_weighted_action_mse_per_sample(
            prediction=proposed_action,
            target=target_action,
            action_is_pad=action_is_pad,
        )
        metrics = {
            "pgc_native_fraction": (~is_counterfactual).float().mean(),
            "pgc_counterfactual_fraction": is_counterfactual.float().mean(),
            "pgc_v5_paired_language_valid_fraction": paired_valid.float().mean(),
            "pgc_v5_prefix_final_action_mse_base": self._masked_policy_guard_mean(
                prefix_base_mse, counterfactual_valid
            ).detach(),
            "pgc_v5_prefix_final_action_mse_proposal": self._masked_policy_guard_mean(
                prefix_proposal_mse, counterfactual_valid
            ).detach(),
            "pgc_v5_prefix_final_action_mse_improvement": self._masked_policy_guard_mean(
                prefix_base_mse - prefix_proposal_mse,
                counterfactual_valid,
            ).detach(),
            "pgc_v5_weighted_final_action_mse_base": self._masked_policy_guard_mean(
                full_base_mse, counterfactual_valid
            ).detach(),
            "pgc_v5_weighted_final_action_mse_proposal": self._masked_policy_guard_mean(
                full_proposal_mse, counterfactual_valid
            ).detach(),
            "pgc_v5_same_state_source_residual_mse": self._masked_policy_guard_mean(
                source_predicted_residual.float().square().mean(dim=(1, 2)),
                paired_valid,
            ).detach(),
            "pgc_v5_goal_cosine_distance": self._masked_policy_guard_mean(
                goal_cosine_distance, paired_valid
            ).detach(),
            "pgc_v5_residual_pair_distance": self._masked_policy_guard_mean(
                residual_pair_distance, paired_valid
            ).detach(),
            "pgc_v5_required_residual_pair_distance": self._masked_policy_guard_mean(
                required_residual_distance, paired_valid
            ).detach(),
            "pgc_v5_completion_transport_fraction": self._masked_policy_guard_mean(
                completion_transport.float(), counterfactual_valid
            ).detach(),
            "pgc_v5_completion_release_fraction": self._masked_policy_guard_mean(
                completion_release.float(), counterfactual_valid
            ).detach(),
            "pgc_v5_completion_sample_weight": self._masked_policy_guard_mean(
                completion_sample_weight, counterfactual_valid
            ).detach(),
            "pgc_v8_closed_loop_fraction": closed_loop_valid.float().mean(),
            "pgc_v8_offline_counterfactual_fraction": (
                offline_counterfactual_valid.float().mean()
            ),
            "pgc_v8_closed_loop_action_loss": self._masked_policy_guard_mean(
                proposal_error, closed_loop_valid
            ).detach(),
            "pgc_v8_offline_action_loss": self._masked_policy_guard_mean(
                proposal_error, offline_counterfactual_valid
            ).detach(),
            "pgc_v8_closed_loop_prefix_mse_improvement": (
                self._masked_policy_guard_mean(
                    prefix_base_mse - prefix_proposal_mse,
                    closed_loop_valid,
                ).detach()
            ),
            "pgc_v8_offline_prefix_mse_improvement": (
                self._masked_policy_guard_mean(
                    prefix_base_mse - prefix_proposal_mse,
                    offline_counterfactual_valid,
                ).detach()
            ),
            "pgc_v8_acquisition_sample_weight": (
                self._masked_policy_guard_mean(
                    acquisition_sample_weight, counterfactual_valid
                ).detach()
            ),
        }
        return (
            counterfactual_action_loss,
            native_zero_loss,
            same_state_source_zero_loss,
            goal_separation_loss,
            residual_separation_loss,
            residual_regularization_loss,
            residual_smoothness_loss,
            metrics,
        )

    def _compute_policy_guard_v5_verifier_loss(
        self,
        *,
        current_video_hidden: torch.Tensor,
        goal_embedding: torch.Tensor,
        demonstrated_action: torch.Tensor,
        base_candidate_action: torch.Tensor,
        counterfactual_candidate_action: torch.Tensor,
        source_candidate_action: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
        is_counterfactual: torch.Tensor,
        direct_action_valid: torch.Tensor,
        paired_language_valid: torch.Tensor,
        goal_ids: Optional[torch.Tensor],
        wrong_entity_candidate_action: Optional[torch.Tensor] = None,
        wrong_relation_candidate_action: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Train v5 on the deployed proposal plus wrong-language/bad negatives."""
        if self.policy_guard_version not in {5, 6, 7, 8, 9}:
            raise RuntimeError(
                "PGC paired verifier loss requires version 5, 6, 7, 8, or 9."
            )
        prefix = min(
            self.policy_guard_execution_prefix_steps,
            int(demonstrated_action.shape[1]),
        )
        prefix_pad = (
            None if action_is_pad is None else action_is_pad[:, :prefix]
        )
        primary_loss, alignment_loss, metrics = (
            self._compute_policy_guard_v4_verifier_loss(
                current_video_hidden=current_video_hidden,
                goal_embedding=goal_embedding,
                demonstrated_action=demonstrated_action[:, :prefix],
                base_candidate_action=base_candidate_action[:, :prefix],
                counterfactual_candidate_action=(
                    counterfactual_candidate_action[:, :prefix]
                ),
                action_is_pad=prefix_pad,
                is_counterfactual=is_counterfactual,
                direct_action_valid=direct_action_valid,
                goal_ids=goal_ids,
            )
        )
        verifier = self.policy_guard_modules["verifier"]
        valid = (
            direct_action_valid.to(device=goal_embedding.device, dtype=torch.bool)
            & is_counterfactual.to(
                device=goal_embedding.device, dtype=torch.bool
            )
            & paired_language_valid.to(
                device=goal_embedding.device, dtype=torch.bool
            )
        )
        base_prefix = base_candidate_action[:, :prefix].detach()
        target_prefix = demonstrated_action[:, :prefix].detach()

        def _auxiliary_candidate_loss(
            candidate: torch.Tensor,
            metric_prefix: str,
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            candidate = candidate[:, :prefix].detach()
            advantage, _, _, _, _, _ = verifier(
                current_video_hidden=current_video_hidden.detach(),
                goal_embedding=goal_embedding,
                base_action=base_prefix,
                counterfactual_action=candidate,
                action_is_pad=prefix_pad,
            )
            base_error = self._compute_policy_guard_v4_weighted_action_mse_per_sample(
                prediction=base_prefix,
                target=target_prefix,
                action_is_pad=prefix_pad,
            )
            candidate_error = self._compute_policy_guard_v4_weighted_action_mse_per_sample(
                prediction=candidate,
                target=target_prefix,
                action_is_pad=prefix_pad,
            )
            target_advantage = (
                (base_error - candidate_error)
                / self.policy_guard_advantage_temperature
            ).clamp(
                min=-self.policy_guard_advantage_clip,
                max=self.policy_guard_advantage_clip,
            )
            regression = F.smooth_l1_loss(
                advantage.float(), target_advantage.float(), reduction="none"
            )
            regression_loss = self._masked_policy_guard_mean(regression, valid)
            rank_valid = valid & (
                target_advantage.abs() >= self.policy_guard_gate_threshold
            )
            preference = torch.sign(target_advantage)
            ranking_loss = self._masked_policy_guard_mean(
                F.softplus(-preference * advantage.float()), rank_valid
            )
            auxiliary_loss = regression_loss + ranking_loss
            return auxiliary_loss, {
                f"loss_{metric_prefix}_regression": regression_loss.detach(),
                f"loss_{metric_prefix}_ranking": ranking_loss.detach(),
                f"{metric_prefix}_advantage": self._masked_policy_guard_mean(
                    advantage, valid
                ).detach(),
                f"{metric_prefix}_target_advantage": self._masked_policy_guard_mean(
                    target_advantage, valid
                ).detach(),
                f"{metric_prefix}_false_accept_rate": self._masked_policy_guard_mean(
                    (advantage >= self.policy_guard_gate_threshold).float(),
                    valid & (target_advantage < self.policy_guard_gate_threshold),
                ).detach(),
            }

        wrong_language_loss, wrong_metrics = _auxiliary_candidate_loss(
            source_candidate_action,
            "pgc_v5_wrong_language",
        )
        mirrored_bad_candidate = (
            2.0 * base_candidate_action.detach()
            - counterfactual_candidate_action.detach()
        )
        bad_candidate_loss, bad_metrics = _auxiliary_candidate_loss(
            mirrored_bad_candidate,
            "pgc_v5_bad_candidate",
        )
        verifier_loss = (
            primary_loss
            + self.policy_guard_verifier_wrong_language_weight
            * wrong_language_loss
            + self.policy_guard_verifier_bad_candidate_weight
            * bad_candidate_loss
        )
        if (
            self.policy_guard_version == 9
            and self.policy_guard_eraf_training_stage == "verifier"
        ):
            if wrong_entity_candidate_action is None or (
                wrong_relation_candidate_action is None
            ):
                raise ValueError(
                    "PGC v9 verifier calibration requires explicit wrong-entity "
                    "and wrong-relation action candidates."
                )
            wrong_entity_loss, wrong_entity_metrics = _auxiliary_candidate_loss(
                wrong_entity_candidate_action,
                "pgc_v9_wrong_entity",
            )
            wrong_relation_loss, wrong_relation_metrics = (
                _auxiliary_candidate_loss(
                    wrong_relation_candidate_action,
                    "pgc_v9_wrong_relation",
                )
            )
            verifier_loss = (
                verifier_loss
                + self.policy_guard_verifier_wrong_entity_weight
                * wrong_entity_loss
                + self.policy_guard_verifier_wrong_relation_weight
                * wrong_relation_loss
            )
            metrics.update(wrong_entity_metrics)
            metrics.update(wrong_relation_metrics)
        metrics.update(wrong_metrics)
        metrics.update(bad_metrics)
        metrics["loss_pgc_v5_verifier_primary"] = primary_loss.detach()
        return verifier_loss, alignment_loss, metrics

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
        candidate_supported: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if base_action.shape != counterfactual_action.shape:
            raise ValueError("PGC action candidates must share shape.")
        if base_score.shape != counterfactual_score.shape or base_score.ndim != 1:
            raise ValueError("PGC candidate scores must share [B] shape.")
        if self.policy_guard_gate_mode == "base":
            selected = torch.zeros_like(base_score, dtype=torch.bool)
        elif self.policy_guard_gate_mode == "counterfactual":
            selected = torch.ones_like(base_score, dtype=torch.bool)
        elif self.policy_guard_version >= 4:
            selected = (
                counterfactual_score - base_score
                >= self.policy_guard_gate_threshold
            )
            if candidate_supported is not None:
                if candidate_supported.shape != selected.shape:
                    raise ValueError(
                        "PGC v4 candidate support mask must match [B] scores."
                    )
                selected = selected & candidate_supported.to(
                    device=selected.device, dtype=torch.bool
                )
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

    @torch.no_grad()
    def _rollout_policy_guard_base_action(
        self,
        *,
        first_frame_latents: torch.Tensor,
        initial_action_noise: torch.Tensor,
        context: torch.Tensor,
        full_context_mask: torch.Tensor,
        state_only_context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        num_inference_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Run the exact frozen Base action sampler used by v4+ deployment."""
        if self.policy_guard_version < 4:
            raise RuntimeError("Rollout-aligned Base sampling requires PGC v4+.")
        if num_inference_steps <= 0:
            raise ValueError("PGC v4 rollout inference steps must be positive.")
        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            device=first_frame_latents.device,
            dtype=first_frame_latents.dtype,
        )
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=full_context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        video_tokens_per_frame = int(video_pre["meta"]["tokens_per_frame"])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=int(initial_action_noise.shape[1]),
            video_tokens_per_frame=video_tokens_per_frame,
            device=video_pre["tokens"].device,
            num_queries=0,
            action_reads_raw_video=True,
        )
        prefill_result = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
            return_final_hidden=True,
        )
        if not isinstance(prefill_result, tuple):
            raise RuntimeError("PGC v4 Base prefill did not return hidden tokens.")
        video_kv_cache, final_video_hidden = prefill_result
        action = initial_action_noise.detach().clone()
        timesteps, deltas = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=action.device,
            dtype=action.dtype,
        )
        for timestep, delta in zip(timesteps, deltas):
            timestep_action = timestep.expand(action.shape[0]).to(
                device=action.device, dtype=action.dtype
            )
            velocity = self._predict_action_noise_with_cache(
                latents_action=action,
                timestep_action=timestep_action,
                context=context,
                context_mask=full_context_mask,
                state_only_context_mask=state_only_context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
            action = self.infer_action_scheduler.step(velocity, delta, action)
        return (
            action.detach(),
            final_video_hidden.detach(),
            video_pre["tokens"].detach(),
            video_tokens_per_frame,
        )

    @torch.no_grad()
    def _rollout_policy_guard_eraf_context_action(
        self,
        *,
        initial_action_noise: torch.Tensor,
        context: torch.Tensor,
        full_context_mask: torch.Tensor,
        state_only_context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        video_seq_len: int,
        video_tokens_per_frame: int,
        routed_goal_queries: torch.Tensor,
        num_inference_steps: int,
        sigma_shift: Optional[float] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Denoise one candidate through internal ERAF context injection."""
        if not (
            self.policy_guard_enabled
            and self.policy_guard_version == 9
            and self.policy_guard_eraf_grounding_objective_version >= 25
        ):
            raise RuntimeError(
                "Internal ERAF action rollout requires PGC v9 objective 25+."
            )
        action = initial_action_noise.detach().clone()
        timesteps, deltas = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=action.device,
            dtype=action.dtype,
            shift_override=sigma_shift,
        )
        last_metrics: dict[str, torch.Tensor] = {}
        for timestep, delta in zip(timesteps, deltas):
            timestep_action = timestep.expand(action.shape[0]).to(
                device=action.device, dtype=action.dtype
            )
            velocity, last_metrics = (
                self._forward_policy_guard_action_from_cache(
                    action_tokens=action,
                    timestep_action=timestep_action,
                    context=context,
                    full_context_mask=full_context_mask,
                    state_only_context_mask=state_only_context_mask,
                    video_kv_cache=video_kv_cache,
                    video_seq_len=video_seq_len,
                    video_tokens_per_frame=video_tokens_per_frame,
                    routed_goal_queries=routed_goal_queries,
                    return_metrics=True,
                )
            )
            action = self.infer_action_scheduler.step(velocity, delta, action)
        return action, last_metrics

    def _training_loss_policy_guard_v4(
        self,
        *,
        inputs: dict[str, Any],
        full_context_mask: torch.Tensor,
        state_only_context_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Train PGC v4 on the same final action chunks used at inference."""
        if self.policy_guard_version != 4:
            raise RuntimeError("Rollout-aligned training requires PGC v4.")
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        is_counterfactual = inputs["pgc_is_counterfactual"]
        direct_action_valid = inputs["pgc_direct_action_valid"]
        if is_counterfactual is None or direct_action_valid is None:
            raise ValueError(
                "PGC v4 requires direct-action provenance for every sample."
            )
        initial_action_noise = torch.randn_like(action)
        (
            base_action,
            final_video_hidden,
            _,
            video_tokens_per_frame,
        ) = (
            self._rollout_policy_guard_base_action(
                first_frame_latents=inputs["input_latents"][:, :, 0:1],
                initial_action_noise=initial_action_noise,
                context=inputs["context"],
                full_context_mask=full_context_mask,
                state_only_context_mask=state_only_context_mask,
                fuse_vae_embedding_in_latents=inputs[
                    "fuse_vae_embedding_in_latents"
                ],
                num_inference_steps=(
                    self.policy_guard_rollout_num_inference_steps
                ),
            )
        )
        goal_queries, goal_embedding, goal_metrics = (
            self._encode_policy_guard_goal(
                final_video_hidden=final_video_hidden,
                video_tokens_per_frame=video_tokens_per_frame,
                context=inputs["context"],
                context_mask=full_context_mask,
            )
        )
        proposal_action, residual, proposal_metrics = self.policy_guard_modules[
            "action_chunk_proposal"
        ](
            base_action=base_action,
            goal_queries=goal_queries,
            action_is_pad=action_is_pad,
        )
        (
            counterfactual_action_loss,
            native_zero_loss,
            residual_regularization_loss,
            residual_smoothness_loss,
            action_metrics,
        ) = self._compute_policy_guard_v4_action_losses(
            proposed_action=proposal_action,
            predicted_residual=residual,
            base_action=base_action,
            target_action=action,
            action_is_pad=action_is_pad,
            is_counterfactual=is_counterfactual,
            direct_action_valid=direct_action_valid,
        )
        proposal_cap = self.policy_guard_modules[
            "action_chunk_proposal"
        ].max_abs.to(device=residual.device, dtype=residual.dtype)
        proposal_delta_mse_per_step = residual.float().square().mean(dim=-1)
        proposal_saturation_per_step = (
            residual.float().abs() >= proposal_cap.float() * 0.95
        ).float().mean(dim=-1)
        if action_is_pad is None:
            proposal_delta_rms = proposal_delta_mse_per_step.mean(dim=1).sqrt()
            proposal_saturation = proposal_saturation_per_step.mean(dim=1)
        else:
            proposal_valid = (~action_is_pad).to(
                device=residual.device, dtype=torch.float32
            )
            proposal_valid_count = proposal_valid.sum(dim=1).clamp_min(1.0)
            proposal_delta_rms = (
                (proposal_delta_mse_per_step * proposal_valid).sum(dim=1)
                / proposal_valid_count
            ).sqrt()
            proposal_saturation = (
                proposal_saturation_per_step * proposal_valid
            ).sum(dim=1) / proposal_valid_count
        proposal_supported = (
            proposal_delta_rms <= self.policy_guard_candidate_max_delta_rms
        ) & (
            proposal_saturation
            <= self.policy_guard_candidate_max_saturation_fraction
        )
        action_metrics.update(
            {
                "pgc_v4_candidate_supported_rate": (
                    proposal_supported.float().mean().detach()
                ),
                "pgc_v4_candidate_delta_rms_max": (
                    proposal_delta_rms.max().detach()
                ),
                "pgc_v4_candidate_saturation_fraction_max": (
                    proposal_saturation.max().detach()
                ),
            }
        )
        proposal_objective = (
            self.policy_guard_action_weight * counterfactual_action_loss
            + self.policy_guard_native_distillation_weight * native_zero_loss
            + self.policy_guard_residual_regularization_weight
            * residual_regularization_loss
            + self.policy_guard_residual_smoothness_weight
            * residual_smoothness_loss
        )

        current_video_hidden = final_video_hidden[:, :video_tokens_per_frame]
        verifier_scale = self._policy_guard_verifier_scale()
        verifier_args = {
            "current_video_hidden": current_video_hidden,
            "goal_embedding": goal_embedding,
            "demonstrated_action": action,
            "base_candidate_action": base_action,
            "counterfactual_candidate_action": proposal_action,
            "action_is_pad": action_is_pad,
            "is_counterfactual": is_counterfactual,
            "direct_action_valid": direct_action_valid,
            "goal_ids": inputs["pgc_goal_id"],
        }
        if verifier_scale > 0.0:
            verifier_loss, alignment_loss, verifier_metrics = (
                self._compute_policy_guard_v4_verifier_loss(**verifier_args)
            )
        else:
            with torch.no_grad():
                verifier_loss, alignment_loss, verifier_metrics = (
                    self._compute_policy_guard_v4_verifier_loss(**verifier_args)
                )
        loss_total = (
            proposal_objective
            + verifier_scale * self.policy_guard_verifier_weight * verifier_loss
            + verifier_scale * self.policy_guard_alignment_weight * alignment_loss
        )

        loss_dict: dict[str, float] = {
            "loss_action": float(proposal_objective.detach().item()),
            "loss_pgc_action": float(counterfactual_action_loss.detach().item()),
            "loss_pgc_native_policy_distillation": float(
                native_zero_loss.detach().item()
            ),
            "loss_pgc_native_residual_zero": float(native_zero_loss.detach().item()),
            "loss_pgc_residual_regularization": float(
                residual_regularization_loss.detach().item()
            ),
            "loss_pgc_residual_smoothness": float(
                residual_smoothness_loss.detach().item()
            ),
            "loss_pgc_verifier": float(verifier_loss.detach().item()),
            "loss_pgc_goal_action_alignment": float(
                alignment_loss.detach().item()
            ),
            "pgc_action_effective_weight": self.policy_guard_action_weight,
            "pgc_native_distillation_effective_weight": (
                self.policy_guard_native_distillation_weight
            ),
            "pgc_verifier_effective_weight": (
                verifier_scale * self.policy_guard_verifier_weight
            ),
            "pgc_alignment_effective_weight": (
                verifier_scale * self.policy_guard_alignment_weight
            ),
            "pgc_residual_regularization_effective_weight": (
                self.policy_guard_residual_regularization_weight
            ),
            "pgc_residual_smoothness_effective_weight": (
                self.policy_guard_residual_smoothness_weight
            ),
            "pgc_verifier_training_scale": verifier_scale,
            "pgc_v4_rollout_num_inference_steps": float(
                self.policy_guard_rollout_num_inference_steps
            ),
            "pgc_base_policy_frozen": 1.0,
            "pgc_video_loss_optimization_weight": 0.0,
        }
        loss_dict.update(detached_policy_guard_metrics(goal_metrics))
        loss_dict.update(detached_policy_guard_metrics(proposal_metrics))
        loss_dict.update(detached_policy_guard_metrics(action_metrics))
        loss_dict.update(detached_policy_guard_metrics(verifier_metrics))
        return loss_total, loss_dict

    def _compute_policy_guard_v6_target_binding_losses(
        self,
        *,
        target_attention: torch.Tensor,
        source_attention: torch.Tensor,
        interaction_teacher: torch.Tensor,
        interaction_valid: torch.Tensor,
        target_prototype_attention: torch.Tensor,
        target_prototype_valid: torch.Tensor,
        source_prototype_attention: torch.Tensor,
        source_prototype_valid: torch.Tensor,
        direct_action_valid: torch.Tensor,
        paired_language_valid: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        """Supervise language-selected visual patches with state hard negatives."""
        if self.policy_guard_version != 6:
            raise RuntimeError("Target-binding losses require PGC v6.")
        attention_tensors = (
            source_attention,
            interaction_teacher,
            target_prototype_attention,
            source_prototype_attention,
        )
        if target_attention.ndim != 2 or any(
            tensor.shape != target_attention.shape
            for tensor in attention_tensors
        ):
            raise ValueError(
                "PGC v6 target/source/teacher distributions must share [B,N]."
            )
        batch_size = int(target_attention.shape[0])
        masks = (
            interaction_valid,
            target_prototype_valid,
            source_prototype_valid,
            direct_action_valid,
            paired_language_valid,
        )
        if any(mask.shape != (batch_size,) for mask in masks):
            raise ValueError("PGC v6 target-binding masks must be [B].")

        target_log = target_attention.float().clamp_min(1.0e-8).log()
        source_log = source_attention.float().clamp_min(1.0e-8).log()

        def _cross_entropy(
            prediction_log: torch.Tensor, target: torch.Tensor
        ) -> torch.Tensor:
            # Normalize by the uniform-attention entropy so this auxiliary
            # objective stays O(1) across different camera/token resolutions.
            return -(target.float() * prediction_log).sum(dim=-1) / math.log(
                max(2, prediction_log.shape[-1])
            )

        direct_valid = direct_action_valid.to(
            device=target_attention.device, dtype=torch.bool
        )
        paired_valid = paired_language_valid.to(
            device=target_attention.device, dtype=torch.bool
        )
        interaction_mask = direct_valid & interaction_valid.to(
            device=target_attention.device, dtype=torch.bool
        )
        target_prototype_mask = direct_valid & target_prototype_valid.to(
            device=target_attention.device, dtype=torch.bool
        )
        source_prototype_mask = paired_valid & source_prototype_valid.to(
            device=target_attention.device, dtype=torch.bool
        )

        target_interaction_ce = _cross_entropy(
            target_log, interaction_teacher
        )
        source_interaction_ce = _cross_entropy(
            source_log, interaction_teacher
        )
        interaction_loss = self._masked_policy_guard_mean(
            target_interaction_ce, interaction_mask
        )
        target_prototype_loss = self._masked_policy_guard_mean(
            _cross_entropy(target_log, target_prototype_attention),
            target_prototype_mask,
        )
        source_prototype_loss = self._masked_policy_guard_mean(
            _cross_entropy(source_log, source_prototype_attention),
            source_prototype_mask,
        )

        hard_negative_mask = paired_valid & interaction_mask
        hard_negative_loss = self._masked_policy_guard_mean(
            torch.relu(
                self.policy_guard_target_binding_hard_negative_margin
                + target_interaction_ce
                - source_interaction_ce
            ),
            hard_negative_mask,
        )
        attention_distance = 0.5 * (
            target_attention.float() - source_attention.float()
        ).abs().sum(dim=-1)
        separation_loss = self._masked_policy_guard_mean(
            torch.relu(
                self.policy_guard_target_binding_separation_margin
                - attention_distance
            ),
            paired_valid,
        )

        metrics = {
            "pgc_v6_interaction_teacher_valid_fraction": (
                interaction_mask.float().mean()
            ),
            "pgc_v6_target_prototype_valid_fraction": (
                target_prototype_mask.float().mean()
            ),
            "pgc_v6_source_prototype_valid_fraction": (
                source_prototype_mask.float().mean()
            ),
            "pgc_v6_same_state_attention_distance": (
                self._masked_policy_guard_mean(
                    attention_distance, paired_valid
                ).detach()
            ),
            "pgc_v6_target_teacher_log_likelihood": (
                self._masked_policy_guard_mean(
                    -target_interaction_ce, interaction_mask
                ).detach()
            ),
            "pgc_v6_source_on_target_teacher_log_likelihood": (
                self._masked_policy_guard_mean(
                    -source_interaction_ce, hard_negative_mask
                ).detach()
            ),
            "pgc_v6_target_teacher_top1_agreement": (
                self._masked_policy_guard_mean(
                    (
                        target_attention.argmax(dim=-1)
                        == interaction_teacher.argmax(dim=-1)
                    ).float(),
                    interaction_mask,
                ).detach()
            ),
        }
        return (
            interaction_loss,
            target_prototype_loss,
            source_prototype_loss,
            hard_negative_loss,
            separation_loss,
            metrics,
        )

    def _compute_policy_guard_v7_target_mask_losses(
        self,
        *,
        target_attention: torch.Tensor,
        source_attention: torch.Tensor,
        aux_attention: torch.Tensor,
        target_teacher: torch.Tensor,
        source_teacher: torch.Tensor,
        aux_teacher: torch.Tensor,
        target_mask_valid: torch.Tensor,
        source_mask_valid: torch.Tensor,
        aux_mask_valid: torch.Tensor,
        direct_action_valid: torch.Tensor,
        paired_language_valid: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        """Train V7 language binding against explicit current-state masks."""
        if self.policy_guard_version != 7:
            raise RuntimeError("Explicit object-mask losses require PGC v7.")
        distributions = (
            source_attention,
            aux_attention,
            target_teacher,
            source_teacher,
            aux_teacher,
        )
        if target_attention.ndim != 2 or any(
            value.shape != target_attention.shape for value in distributions
        ):
            raise ValueError(
                "PGC v7 attention and mask distributions must share [B,N]."
            )
        batch_size = int(target_attention.shape[0])
        validity = (
            target_mask_valid,
            source_mask_valid,
            aux_mask_valid,
            direct_action_valid,
            paired_language_valid,
        )
        if any(value.shape != (batch_size,) for value in validity):
            raise ValueError("PGC v7 target-mask validity tensors must be [B].")

        target_valid = direct_action_valid.bool() & target_mask_valid.bool()
        source_valid = paired_language_valid.bool() & source_mask_valid.bool()
        aux_valid = paired_language_valid.bool() & aux_mask_valid.bool()
        log_normalizer = math.log(max(2, int(target_attention.shape[-1])))

        def _normalized_ce(
            prediction: torch.Tensor, teacher: torch.Tensor
        ) -> torch.Tensor:
            return -(
                teacher.float()
                * prediction.float().clamp_min(1.0e-8).log()
            ).sum(dim=-1) / log_normalizer

        target_loss = self._masked_policy_guard_mean(
            _normalized_ce(target_attention, target_teacher), target_valid
        )
        source_loss = self._masked_policy_guard_mean(
            _normalized_ce(source_attention, source_teacher), source_valid
        )
        aux_loss = self._masked_policy_guard_mean(
            _normalized_ce(aux_attention, aux_teacher), aux_valid
        )

        def _support_mass(
            prediction: torch.Tensor, teacher: torch.Tensor
        ) -> torch.Tensor:
            support = teacher.float() > 0
            return (
                prediction.float() * support.to(prediction.dtype)
            ).sum(dim=-1)

        target_own_mass = _support_mass(target_attention, target_teacher)
        source_own_mass = _support_mass(source_attention, source_teacher)
        aux_own_mass = _support_mass(aux_attention, aux_teacher)
        target_on_source = _support_mass(target_attention, source_teacher)
        source_on_target = _support_mass(source_attention, target_teacher)
        aux_on_target = _support_mass(aux_attention, target_teacher)
        aux_on_source = _support_mass(aux_attention, source_teacher)

        mass_terms = torch.cat(
            (
                -target_own_mass.clamp_min(1.0e-8).log()[target_valid],
                -source_own_mass.clamp_min(1.0e-8).log()[source_valid],
                -aux_own_mass.clamp_min(1.0e-8).log()[aux_valid],
            )
        )
        if mass_terms.numel() == 0:
            mass_loss = target_attention.sum() * 0.0
        else:
            mass_loss = mass_terms.mean()

        target_source_valid = target_valid & source_valid
        aux_target_valid = aux_valid & target_valid
        aux_source_valid = aux_valid & source_valid
        cross_terms = torch.cat(
            (
                torch.relu(
                    self.policy_guard_cross_object_margin
                    - target_own_mass
                    + target_on_source
                )[target_source_valid],
                torch.relu(
                    self.policy_guard_cross_object_margin
                    - source_own_mass
                    + source_on_target
                )[target_source_valid],
                torch.relu(
                    self.policy_guard_cross_object_margin
                    - aux_own_mass
                    + aux_on_target
                )[aux_target_valid],
                torch.relu(
                    self.policy_guard_cross_object_margin
                    - aux_own_mass
                    + aux_on_source
                )[aux_source_valid],
            )
        )
        if cross_terms.numel() == 0:
            cross_object_loss = target_attention.sum() * 0.0
        else:
            cross_object_loss = cross_terms.mean()

        target_source_distance = 0.5 * (
            target_attention.float() - source_attention.float()
        ).abs().sum(dim=-1)
        metrics = {
            "pgc_v7_target_mask_valid_fraction": target_valid.float().mean(),
            "pgc_v7_source_mask_valid_fraction": source_valid.float().mean(),
            "pgc_v7_aux_mask_valid_fraction": aux_valid.float().mean(),
            "pgc_v7_target_mask_mass": self._masked_policy_guard_mean(
                target_own_mass, target_valid
            ).detach(),
            "pgc_v7_source_mask_mass": self._masked_policy_guard_mean(
                source_own_mass, source_valid
            ).detach(),
            "pgc_v7_aux_mask_mass": self._masked_policy_guard_mean(
                aux_own_mass, aux_valid
            ).detach(),
            "pgc_v7_target_on_source_mask_mass": (
                self._masked_policy_guard_mean(
                    target_on_source, target_source_valid
                ).detach()
            ),
            "pgc_v7_source_on_target_mask_mass": (
                self._masked_policy_guard_mean(
                    source_on_target, target_source_valid
                ).detach()
            ),
            "pgc_v7_same_state_attention_distance": (
                self._masked_policy_guard_mean(
                    target_source_distance, target_source_valid
                ).detach()
            ),
            "pgc_v7_target_mask_top1_agreement": (
                self._masked_policy_guard_mean(
                    (
                        target_attention.argmax(dim=-1)
                        == target_teacher.argmax(dim=-1)
                    ).float(),
                    target_valid,
                ).detach()
            ),
        }
        return (
            target_loss,
            source_loss,
            aux_loss,
            mass_loss,
            cross_object_loss,
            metrics,
        )

    def _policy_guard_v9_labels(
        self,
        inputs: dict[str, Any],
        *,
        prefix: str = "",
    ) -> dict[str, torch.Tensor]:
        if self.policy_guard_version != 9:
            raise RuntimeError("ERAF labels are only defined for PGC v9.")
        labels: dict[str, torch.Tensor] = {}
        missing: list[str] = []
        for name in PGC_ENTITY_RELATION_ARRAY_NAMES:
            input_name = f"pgc_eraf_{prefix}{name}"
            value = inputs.get(input_name)
            if value is None:
                missing.append(input_name)
            else:
                labels[name] = value
        if self.policy_guard_eraf_grounding_objective_version >= 14:
            for name in PGC_PHASE_SAFE_MEMORY_LABEL_NAMES:
                input_name = f"pgc_eraf_{prefix}{name}"
                value = inputs.get(input_name)
                if value is None:
                    missing.append(input_name)
                else:
                    labels[name] = value
        if missing:
            raise ValueError(
                "PGC v9 requires audited entity-relation labels: "
                f"{missing}."
            )
        return labels

    @staticmethod
    def _policy_guard_v915_negative_eraf_outputs(
        *,
        target_outputs: Mapping[str, torch.Tensor],
        source_outputs: Mapping[str, torch.Tensor],
        kind: str,
        target_labels: Optional[Mapping[str, torch.Tensor]] = None,
        source_labels: Optional[Mapping[str, torch.Tensor]] = None,
        reference_subject_fallback: bool = False,
        anchor_mirror_fallback: bool = False,
        preserve_clause_route: bool = False,
        propagate_reference_to_anchor: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Build one isolated same-state ERAF intervention for V9.15+.

        V9.16 makes reference corruptions semantic: it uses a genuinely
        different source reference when available and otherwise substitutes
        the target subject as a visible, same-state invalid reference.
        """
        negative = dict(target_outputs)
        source_fields = {
            "subject": ("subject_token", "subject_position", "grasp_anchor"),
            "reference": ("reference_token", "reference_position"),
            "anchor": ("goal_anchor", "interaction_anchor"),
        }
        if kind == "anchor" and anchor_mirror_fallback:
            reference_position = target_outputs.get("reference_position")
            if reference_position is None:
                raise ValueError(
                    "V9.17 anchor intervention requires reference_position."
                )
            for name in source_fields["anchor"]:
                target_value = target_outputs[name]
                source_value = source_outputs[name]
                if not (
                    target_value.shape
                    == source_value.shape
                    == reference_position.shape
                ):
                    raise ValueError(
                        f"V9.17 anchor intervention shape mismatch for {name}."
                    )
                source_changed = (
                    source_value.float() - target_value.float()
                ).norm(dim=-1, keepdim=True) > 1.0e-4
                mirrored = 2.0 * reference_position - target_value
                mirror_changed = (
                    mirrored.float() - target_value.float()
                ).norm(dim=-1, keepdim=True) > 1.0e-4
                offset = target_value.new_tensor((0.25, -0.25, 0.125))
                fallback = torch.where(
                    mirror_changed,
                    mirrored,
                    target_value + offset,
                ).clamp(-1.0, 1.0)
                negative[name] = torch.where(
                    source_changed,
                    source_value,
                    fallback,
                )
            return negative
        if kind == "reference" and reference_subject_fallback:
            if target_labels is None or source_labels is None:
                raise ValueError(
                    "V9.16 reference intervention requires target/source labels."
                )
            clause_valid = target_labels["clause_valid"].bool()
            source_clause_valid = source_labels["clause_valid"].bool()
            target_subject_ids = target_labels["subject_entity_ids"].long()
            target_reference_ids = target_labels["reference_entity_ids"].long()
            source_reference_ids = source_labels["reference_entity_ids"].long()
            genuine_source_swap = (
                clause_valid
                & source_clause_valid
                & (target_reference_ids >= 0)
                & (source_reference_ids >= 0)
                & (target_reference_ids != source_reference_ids)
            )
            subject_fallback = (
                clause_valid
                & ~genuine_source_swap
                & (target_subject_ids >= 0)
                & (target_reference_ids >= 0)
                & (target_subject_ids != target_reference_ids)
            )
            subject_fields = {
                "reference_token": "subject_token",
                "reference_position": "subject_position",
            }
            for name in source_fields["reference"]:
                target_value = target_outputs[name]
                source_value = source_outputs[name]
                subject_value = target_outputs[subject_fields[name]]
                if not (
                    target_value.shape
                    == source_value.shape
                    == subject_value.shape
                ):
                    raise ValueError(
                        f"V9.16 reference intervention shape mismatch for {name}."
                    )
                broadcast_shape = (
                    *genuine_source_swap.shape,
                    *([1] * (target_value.ndim - 2)),
                )
                source_mask = genuine_source_swap.reshape(broadcast_shape)
                fallback_mask = subject_fallback.reshape(broadcast_shape)
                negative[name] = torch.where(
                    source_mask,
                    source_value,
                    torch.where(fallback_mask, subject_value, target_value),
                )
            if propagate_reference_to_anchor:
                reference_delta = (
                    negative["reference_position"]
                    - target_outputs["reference_position"]
                )
                for name in ("goal_anchor", "interaction_anchor"):
                    negative[name] = (
                        target_outputs[name] + reference_delta
                    ).clamp(-1.0, 1.0)
            return negative
        if kind in source_fields:
            for name in source_fields[kind]:
                if name not in target_outputs or name not in source_outputs:
                    raise ValueError(
                        f"V9.15 {kind} intervention requires ERAF field {name!r}."
                    )
                if target_outputs[name].shape != source_outputs[name].shape:
                    raise ValueError(
                        f"V9.15 {kind} intervention shape mismatch for {name}."
                    )
                negative[name] = source_outputs[name]
            return negative
        if kind != "clause":
            raise ValueError(f"Unknown V9.15 ERAF intervention kind: {kind!r}.")

        clause_fields = (
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
            "subject_visibility_logits",
            "reference_visibility_logits",
        )
        for name in clause_fields:
            value = target_outputs.get(name)
            if value is None:
                raise ValueError(
                    f"V9.15 clause intervention requires ERAF field {name!r}."
                )
            if value.ndim < 2 or value.shape[1] < 2:
                raise ValueError(
                    "V9.15 clause intervention requires at least two clauses "
                    f"for {name}."
                )
            index = torch.arange(value.shape[1], device=value.device)
            index = index.clone()
            index[0], index[1] = index[1].clone(), index[0].clone()
            negative[name] = value.index_select(1, index)
        if preserve_clause_route:
            # V9.19 audits the causal meaning of the selected slot.  Swapping
            # both its semantics and its router score is a pure permutation
            # and can never change any permutation-invariant action path.
            for name in ("active_logits", "clause_execution_probability"):
                negative[name] = target_outputs[name]
        return negative

    def _compute_policy_guard_v915_causal_action_loss(
        self,
        *,
        correct_action: torch.Tensor,
        negative_actions: Mapping[str, torch.Tensor],
        target_action: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
        target_labels: Mapping[str, torch.Tensor],
        source_labels: Mapping[str, torch.Tensor],
        is_counterfactual: torch.Tensor,
        direct_action_valid: torch.Tensor,
        paired_language_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Prefer the correct ERAF intervention at the deployed action level."""
        prefix = min(
            self.policy_guard_execution_prefix_steps,
            int(correct_action.shape[1]),
        )
        valid_step = torch.ones(
            correct_action.shape[:2],
            device=correct_action.device,
            dtype=torch.float32,
        )
        if action_is_pad is not None:
            valid_step = (~action_is_pad.bool()).float()
        valid_step = valid_step[:, :prefix]
        valid_step_count = valid_step.sum(dim=-1).clamp_min(1.0)

        def per_sample_error(candidate: torch.Tensor) -> torch.Tensor:
            error = (
                candidate[:, :prefix].float()
                - target_action[:, :prefix].float()
            ).square().mean(dim=-1)
            return (error * valid_step).sum(dim=-1) / valid_step_count

        correct_error = per_sample_error(correct_action)
        clause_valid = target_labels["clause_valid"].bool()
        shared_valid = (
            is_counterfactual.bool()
            & direct_action_valid.bool()
            & paired_language_valid.bool()
            & clause_valid.any(dim=-1)
        )
        subject_changed = (
            target_labels["subject_entity_ids"].long()
            != source_labels["subject_entity_ids"].long()
        ) & clause_valid & source_labels["clause_valid"].bool()
        reference_changed = (
            target_labels["reference_entity_ids"].long()
            != source_labels["reference_entity_ids"].long()
        ) & clause_valid & source_labels["clause_valid"].bool()
        if self.policy_guard_eraf_grounding_objective_version >= 16:
            target_subject_ids = target_labels["subject_entity_ids"].long()
            source_subject_ids = source_labels["subject_entity_ids"].long()
            target_reference_ids = target_labels["reference_entity_ids"].long()
            source_reference_ids = source_labels["reference_entity_ids"].long()
            subject_changed = subject_changed & (target_subject_ids >= 0) & (
                source_subject_ids >= 0
            )
            reference_changed = reference_changed & (
                target_reference_ids >= 0
            ) & (source_reference_ids >= 0)
            reference_subject_fallback = (
                clause_valid
                & ~reference_changed
                & (target_subject_ids >= 0)
                & (target_reference_ids >= 0)
                & (target_subject_ids != target_reference_ids)
            )
            reference_eligible = (
                reference_changed | reference_subject_fallback
            ).any(dim=-1)
        else:
            reference_subject_fallback = torch.zeros_like(reference_changed)
            reference_eligible = reference_changed.any(dim=-1)
        anchor_valid = (
            target_labels["goal_anchor_valid"].bool()
            & clause_valid
        )
        if self.policy_guard_eraf_grounding_objective_version < 17:
            anchor_valid = (
                anchor_valid
                & source_labels["goal_anchor_valid"].bool()
                & source_labels["clause_valid"].bool()
            )
        anchor_changed = (
            target_labels["goal_anchors"].float()
            - source_labels["goal_anchors"].float()
        ).norm(dim=-1) > 1.0e-4
        eligibility = {
            "subject": shared_valid & subject_changed.any(dim=-1),
            "reference": shared_valid & reference_eligible,
            "anchor": shared_valid
            & (
                anchor_valid.any(dim=-1)
                if self.policy_guard_eraf_grounding_objective_version >= 17
                else (anchor_valid & anchor_changed).any(dim=-1)
            ),
            "clause": shared_valid & (clause_valid.sum(dim=-1) > 1),
        }

        total = correct_error.sum() * 0.0
        active_kind_count = correct_error.sum() * 0.0
        metrics: dict[str, torch.Tensor] = {}
        margin = float(self.policy_guard_eraf_action_causal_margin)
        for kind, candidate in negative_actions.items():
            negative_error = per_sample_error(candidate)
            valid = eligibility[kind]
            valid_weight = valid.float()
            valid_count = valid_weight.sum().clamp_min(1.0)
            ranking = torch.relu(margin + correct_error - negative_error)
            kind_loss = (ranking * valid_weight).sum() / valid_count
            active_kind_count = active_kind_count + valid.any().float()
            total = total + kind_loss
            correct_win = (
                (correct_error < negative_error).float() * valid_weight
            ).sum() / valid_count
            action_effect_per_sample = (
                candidate[:, :prefix].float()
                - correct_action[:, :prefix].float()
            ).square().mean(dim=(1, 2)).sqrt()
            action_effect = (
                action_effect_per_sample * valid_weight
            ).sum() / valid_count
            metrics.update(
                {
                    f"loss_pgc_v915_{kind}_action_ranking": kind_loss.detach(),
                    f"pgc_v915_{kind}_eligible_rate": valid.float().mean().detach(),
                    f"pgc_v915_{kind}_correct_action_win_rate": (
                        correct_win.detach()
                    ),
                    f"pgc_v915_{kind}_action_delta_rms": action_effect.detach(),
                }
            )
        total = total / active_kind_count.clamp_min(1.0)
        metrics["loss_pgc_v915_action_causal_ranking"] = total.detach()
        metrics["pgc_v915_action_causal_active_kinds"] = (
            active_kind_count.detach()
        )
        if self.policy_guard_eraf_grounding_objective_version >= 16:
            metrics["pgc_v916_reference_subject_fallback_rate"] = (
                (shared_valid & reference_subject_fallback.any(dim=-1))
                .float()
                .mean()
                .detach()
            )
        return total, metrics

    def _compute_policy_guard_v918_phase_residual_loss(
        self,
        *,
        geometry_residual: torch.Tensor,
        pre_geometry_action: torch.Tensor,
        target_action: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
        target_labels: Mapping[str, torch.Tensor],
        eraf_outputs: Mapping[str, torch.Tensor],
        is_counterfactual: torch.Tensor,
        direct_action_valid: torch.Tensor,
        paired_language_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Fit the bounded V9.17 residual to phase-specific expert corrections.

        The upstream Proposal is deliberately detached when constructing the
        target.  V9.18 therefore learns only the missing geometry-to-control
        correction instead of allowing the loss to be satisfied by changing
        the already audited ERAF, semantic bridge, or Proposal.
        """
        if geometry_residual.shape != pre_geometry_action.shape or (
            target_action.shape != pre_geometry_action.shape
        ):
            raise ValueError(
                "PGC V9.18 residual, candidate, and expert actions must share "
                "the same [B,T,A] shape."
            )
        clause_valid = target_labels["clause_valid"].bool()
        phase_valid = target_labels["phase_valid"].bool() & clause_valid
        phase_ids = target_labels["phase_ids"].long()
        execution_probability = eraf_outputs[
            "clause_execution_probability"
        ].float()
        if execution_probability.shape != clause_valid.shape:
            raise ValueError(
                "PGC V9.18 clause execution probabilities do not match labels."
            )
        selected_clause = execution_probability.detach().masked_fill(
            ~phase_valid, -1.0
        ).argmax(dim=-1)
        selected_phase = phase_ids.gather(
            1, selected_clause.unsqueeze(-1)
        ).squeeze(-1)
        selected_valid = phase_valid.gather(
            1, selected_clause.unsqueeze(-1)
        ).squeeze(-1)
        predicate_truth = target_labels["predicate_truth"].float().gather(
            1, selected_clause.unsqueeze(-1)
        ).squeeze(-1)
        predicate_truth_valid = target_labels[
            "predicate_truth_valid"
        ].bool().gather(1, selected_clause.unsqueeze(-1)).squeeze(-1)
        # The observation immediately before a successful release can still
        # carry phase=1 (holding).  If the spatial predicate is already true
        # while held, treat that frame as release-ready rather than generic
        # transport.  Phase=2 remains the post-release/completed case.
        control_phase = torch.where(
            (selected_phase == 1)
            & predicate_truth_valid
            & (predicate_truth >= 0.5),
            torch.full_like(selected_phase, 2),
            selected_phase,
        )
        sample_valid = (
            is_counterfactual.bool()
            & direct_action_valid.bool()
            & paired_language_valid.bool()
            & selected_valid
        )

        max_abs = float(self.policy_guard_eraf_action_geometry_residual_max_abs)
        target_residual = (
            target_action.float() - pre_geometry_action.detach().float()
        ).clamp(min=-max_abs, max=max_abs)
        predicted_residual = geometry_residual.float()
        batch, horizon, _ = predicted_residual.shape
        prefix = min(self.policy_guard_execution_prefix_steps, horizon)
        temporal_weight = predicted_residual.new_full(
            (batch, horizon), float(self.policy_guard_suffix_loss_weight)
        )
        temporal_weight[:, :prefix] = 1.0
        if action_is_pad is not None:
            temporal_weight = temporal_weight * (~action_is_pad.bool()).float()

        phase_weight_values = predicted_residual.new_tensor(
            (
                self.policy_guard_eraf_action_phase_approach_weight,
                self.policy_guard_eraf_action_phase_transport_weight,
                self.policy_guard_eraf_action_phase_release_weight,
            )
        )
        phase_weight = phase_weight_values[control_phase.clamp(0, 2)]
        sample_weight = sample_valid.float() * phase_weight
        weighted_step = temporal_weight * sample_weight.unsqueeze(-1)
        weighted_count = weighted_step.sum().clamp_min(1.0)

        regression_per_step = F.smooth_l1_loss(
            predicted_residual,
            target_residual,
            reduction="none",
            beta=0.01,
        ).mean(dim=-1)
        imitation_loss = (
            regression_per_step * weighted_step
        ).sum() / weighted_count

        translation_dim = min(3, int(predicted_residual.shape[-1]))
        predicted_translation = predicted_residual[..., :translation_dim]
        target_translation = target_residual[..., :translation_dim]
        target_translation_norm = target_translation.norm(dim=-1)
        direction_valid = (
            target_translation_norm
            >= self.policy_guard_eraf_action_phase_direction_min_norm
        )
        direction_weight = weighted_step * direction_valid.float()
        direction_count = direction_weight.sum().clamp_min(1.0)
        direction_cosine = F.cosine_similarity(
            predicted_translation,
            target_translation,
            dim=-1,
            eps=1.0e-6,
        )
        direction_loss = (
            (1.0 - direction_cosine) * direction_weight
        ).sum() / direction_count
        total = (
            self.policy_guard_eraf_action_phase_residual_imitation_weight
            * imitation_loss
            + self.policy_guard_eraf_action_phase_direction_weight
            * direction_loss
        )

        prefix_valid = temporal_weight[:, :prefix] > 0
        prefix_valid_float = prefix_valid.float()
        prefix_count = prefix_valid_float.sum(dim=-1).clamp_min(1.0)
        pre_error = (
            (
                pre_geometry_action[:, :prefix].float()
                - target_action[:, :prefix].float()
            )
            .square()
            .mean(dim=-1)
            * prefix_valid_float
        ).sum(dim=-1) / prefix_count
        post_error = (
            (
                pre_geometry_action[:, :prefix].float()
                + predicted_residual[:, :prefix]
                - target_action[:, :prefix].float()
            )
            .square()
            .mean(dim=-1)
            * prefix_valid_float
        ).sum(dim=-1) / prefix_count
        valid_count = sample_valid.float().sum().clamp_min(1.0)
        metrics: dict[str, torch.Tensor] = {
            "loss_pgc_v918_phase_residual_imitation": imitation_loss.detach(),
            "loss_pgc_v918_translation_direction": direction_loss.detach(),
            "pgc_v918_phase_valid_fraction": sample_valid.float().mean().detach(),
            "pgc_v918_target_residual_rms": (
                (
                    target_residual.square().mean(dim=(1, 2)).sqrt()
                    * sample_valid.float()
                ).sum()
                / valid_count
            ).detach(),
            "pgc_v918_prefix_mse_improvement": (
                ((pre_error - post_error) * sample_valid.float()).sum()
                / valid_count
            ).detach(),
            "pgc_v918_translation_direction_cosine": (
                (direction_cosine * direction_weight).sum() / direction_count
            ).detach(),
            "pgc_v918_translation_direction_positive_rate": (
                ((direction_cosine > 0).float() * direction_weight).sum()
                / direction_count
            ).detach(),
        }
        for phase_id, phase_name in enumerate(
            ("approach", "transport", "release")
        ):
            phase_sample = sample_valid & (control_phase == phase_id)
            phase_count = phase_sample.float().sum().clamp_min(1.0)
            metrics[f"pgc_v918_{phase_name}_sample_fraction"] = (
                phase_sample.float().mean().detach()
            )
            metrics[f"pgc_v918_{phase_name}_prefix_mse_improvement"] = (
                ((pre_error - post_error) * phase_sample.float()).sum()
                / phase_count
            ).detach()
        return total, metrics

    def _compute_policy_guard_v920_waypoint_loss(
        self,
        *,
        geometry_residual: torch.Tensor,
        pre_geometry_action: torch.Tensor,
        target_action: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
        waypoint_metrics: Mapping[str, torch.Tensor],
        is_counterfactual: torch.Tensor,
        direct_action_valid: torch.Tensor,
        paired_language_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Supervise local waypoints only on anchor-compatible expert steps."""
        required = {
            "pgc_v919_selected_control_phase",
            "pgc_v919_desired_direction",
            "pgc_v920_compatibility_logits",
            "pgc_v920_local_direction",
            "pgc_v920_servo_translation",
            "pgc_v920_effective_servo_residual",
        }
        missing = sorted(required - set(waypoint_metrics))
        if missing:
            raise ValueError(f"PGC V9.20 waypoint metrics are missing: {missing}.")
        target_residual = (
            target_action.float() - pre_geometry_action.detach().float()
        ).clamp(
            min=-self.policy_guard_eraf_action_geometry_residual_max_abs,
            max=self.policy_guard_eraf_action_geometry_residual_max_abs,
        )
        predicted_residual = geometry_residual.float()
        batch, horizon, _ = predicted_residual.shape
        prefix = min(self.policy_guard_execution_prefix_steps, horizon)
        temporal_weight = predicted_residual.new_full(
            (batch, horizon), float(self.policy_guard_suffix_loss_weight)
        )
        temporal_weight[:, :prefix] = 1.0
        if action_is_pad is not None:
            temporal_weight = temporal_weight * (~action_is_pad.bool()).float()

        sample_valid = (
            is_counterfactual.bool()
            & direct_action_valid.bool()
            & paired_language_valid.bool()
        )
        phase = waypoint_metrics["pgc_v919_selected_control_phase"].long()
        phase_weights = predicted_residual.new_tensor(
            (
                self.policy_guard_eraf_action_phase_approach_weight,
                self.policy_guard_eraf_action_phase_transport_weight,
                self.policy_guard_eraf_action_phase_release_weight,
            )
        )[phase.clamp(0, 2)]
        valid_weight = (
            temporal_weight
            * sample_valid.float().unsqueeze(-1)
            * phase_weights.unsqueeze(-1)
        )

        target_translation = target_residual[..., :3]
        target_norm = target_translation.norm(dim=-1)
        anchor_direction = F.normalize(
            waypoint_metrics["pgc_v919_desired_direction"].float(),
            dim=-1,
            eps=1.0e-6,
        )
        anchor_cosine = F.cosine_similarity(
            anchor_direction.unsqueeze(1),
            target_translation,
            dim=-1,
            eps=1.0e-6,
        )
        compatible = (
            (target_norm >= self.policy_guard_eraf_action_phase_direction_min_norm)
            & (anchor_cosine >= self.policy_guard_eraf_action_waypoint_min_cosine)
        )
        compatibility_target = compatible.float()
        compatibility_logits = waypoint_metrics[
            "pgc_v920_compatibility_logits"
        ].float()
        compatibility_loss_step = F.binary_cross_entropy_with_logits(
            compatibility_logits, compatibility_target, reduction="none"
        )
        valid_count = valid_weight.sum().clamp_min(1.0)
        compatibility_loss = (
            compatibility_loss_step * valid_weight
        ).sum() / valid_count

        compatible_weight = valid_weight * compatibility_target
        compatible_count = compatible_weight.sum().clamp_min(1.0)
        imitation_step = F.smooth_l1_loss(
            predicted_residual, target_residual, reduction="none", beta=0.01
        ).mean(dim=-1)
        imitation_loss = (
            imitation_step * compatible_weight
        ).sum() / compatible_count
        local_direction = waypoint_metrics["pgc_v920_local_direction"].float()
        local_cosine = F.cosine_similarity(
            local_direction, target_translation, dim=-1, eps=1.0e-6
        )
        direction_loss = (
            (1.0 - local_cosine) * compatible_weight
        ).sum() / compatible_count

        effective_servo_residual = waypoint_metrics[
            "pgc_v920_effective_servo_residual"
        ].float()
        incompatible_weight = valid_weight * (~compatible).float()
        native_weight = temporal_weight * (~is_counterfactual.bool()).float().unsqueeze(-1)
        zero_weight = incompatible_weight + native_weight
        zero_count = zero_weight.sum().clamp_min(1.0)
        zero_loss = (
            effective_servo_residual.square().mean(dim=-1) * zero_weight
        ).sum() / zero_count
        total = (
            self.policy_guard_eraf_action_waypoint_compatibility_weight
            * compatibility_loss
            + self.policy_guard_eraf_action_waypoint_imitation_weight
            * imitation_loss
            + self.policy_guard_eraf_action_waypoint_direction_weight
            * direction_loss
            + self.policy_guard_eraf_action_waypoint_zero_weight * zero_loss
        )
        predicted_compatible = compatibility_logits >= 0
        audit_weight = (valid_weight > 0).float()
        audit_count = audit_weight.sum().clamp_min(1.0)
        metrics = {
            "loss_pgc_v920_compatibility": compatibility_loss.detach(),
            "loss_pgc_v920_waypoint_imitation": imitation_loss.detach(),
            "loss_pgc_v920_waypoint_direction": direction_loss.detach(),
            "loss_pgc_v920_incompatible_zero": zero_loss.detach(),
            "pgc_v920_compatible_step_rate": (
                (compatibility_target * audit_weight).sum() / audit_count
            ).detach(),
            "pgc_v920_compatibility_accuracy": (
                ((predicted_compatible == compatible).float() * audit_weight).sum()
                / audit_count
            ).detach(),
            "pgc_v920_anchor_expert_cosine": (
                (anchor_cosine * compatible_weight).sum() / compatible_count
            ).detach(),
            "pgc_v920_local_waypoint_expert_cosine": (
                (local_cosine * compatible_weight).sum() / compatible_count
            ).detach(),
        }
        return total, metrics

    def _compute_policy_guard_v921_expert_alignment_loss(
        self,
        *,
        candidate_action: torch.Tensor,
        pre_alignment_action: torch.Tensor,
        pre_alignment_residual: torch.Tensor,
        deployed_metrics: Mapping[str, torch.Tensor],
        waypoint_metrics: Mapping[str, torch.Tensor],
        target_labels: Mapping[str, torch.Tensor],
        target_action: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
        is_counterfactual: torch.Tensor,
        direct_action_valid: torch.Tensor,
        paired_language_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Align privileged phase geometry with expert action prefixes.

        The deployed branch consumes learned ERAF geometry.  A second training-
        only call uses audited anchors from the sidecar with the same frozen
        action stack.  Both are fitted to ``expert - current_candidate`` while
        deployed corrections are distilled toward the privileged branch.
        """
        required_waypoint = {
            "pgc_v919_selected_clause",
            "pgc_v919_selected_control_phase",
            "pgc_v919_route_confidence_per_sample",
            "pgc_v920_compatibility_probability_per_step",
            "pgc_v919_calibrated_eef_position",
        }
        required_deployed = {"pgc_v921_effective_expert_correction"}
        missing = sorted(required_waypoint - set(waypoint_metrics))
        missing += sorted(required_deployed - set(deployed_metrics))
        if missing:
            raise ValueError(
                f"PGC V9.21 expert-alignment metrics are missing: {missing}."
            )
        batch, horizon, _ = candidate_action.shape
        eef_position = waypoint_metrics[
            "pgc_v919_calibrated_eef_position"
        ].float()
        if eef_position.shape != (batch, 3):
            raise ValueError("PGC V9.21 canonical EEF position must be [B,3].")

        selected_clause = waypoint_metrics["pgc_v919_selected_clause"].long()

        def gather_clause(value: torch.Tensor) -> torch.Tensor:
            index = selected_clause.view(batch, 1, *([1] * (value.ndim - 2)))
            index = index.expand(batch, 1, *value.shape[2:])
            return value.gather(1, index).squeeze(1)

        clause_valid = gather_clause(target_labels["clause_valid"].bool())
        deployed_phase = waypoint_metrics[
            "pgc_v919_selected_control_phase"
        ].long().clamp(0, 2)
        phase_valid = gather_clause(target_labels["phase_valid"].bool())
        teacher_phase = gather_clause(target_labels["phase_ids"].long()).clamp(
            0, 2
        )
        predicate_truth = gather_clause(
            target_labels["predicate_truth"].float()
        )
        predicate_truth_valid = gather_clause(
            target_labels["predicate_truth_valid"].bool()
        )
        # Use the audited phase only in the privileged training branch.  As in
        # the phase-residual teacher, a spatial predicate that is already true
        # while the object is held is release-ready rather than transport.
        privileged_phase = torch.where(
            (teacher_phase == 1)
            & predicate_truth_valid
            & (predicate_truth >= 0.5),
            torch.full_like(teacher_phase, 2),
            teacher_phase,
        )
        grasp = gather_clause(target_labels["grasp_anchors"].float())
        goal = gather_clause(target_labels["goal_anchors"].float())
        interaction = gather_clause(target_labels["interaction_anchors"].float())
        grasp_valid = gather_clause(target_labels["grasp_anchor_valid"].bool())
        goal_valid = gather_clause(target_labels["goal_anchor_valid"].bool())
        interaction_valid = gather_clause(
            target_labels["interaction_anchor_valid"].bool()
        )
        release_anchor = torch.where(
            interaction_valid.unsqueeze(-1), interaction, goal
        )
        privileged_anchor = torch.where(
            (privileged_phase == 0).unsqueeze(-1),
            grasp,
            torch.where(
                (privileged_phase == 1).unsqueeze(-1), goal, release_anchor
            ),
        )
        anchor_valid = torch.where(
            privileged_phase == 0,
            grasp_valid,
            torch.where(
                privileged_phase == 1,
                goal_valid,
                interaction_valid | goal_valid,
            ),
        )
        privileged_vector = privileged_anchor - eef_position.float()
        privileged_distance = privileged_vector.norm(dim=-1)
        privileged_direction = F.normalize(
            privileged_vector, dim=-1, eps=1.0e-6
        )

        expert_adapter = self.policy_guard_modules[
            "eraf_phase_expert_residual_adapter"
        ]
        _, _, privileged_metrics = expert_adapter(
            candidate_action=candidate_action,
            current_residual=pre_alignment_residual,
            desired_direction=privileged_direction,
            desired_distance=privileged_distance,
            control_phase=privileged_phase,
            route_confidence=torch.ones_like(privileged_distance),
            waypoint_compatibility=torch.ones(
                (batch, horizon),
                device=candidate_action.device,
                dtype=candidate_action.dtype,
            ),
            action_is_pad=action_is_pad,
        )
        privileged_correction = privileged_metrics[
            "pgc_v921_effective_expert_correction"
        ].float()
        deployed_correction = deployed_metrics[
            "pgc_v921_effective_expert_correction"
        ].float()
        target_correction = (
            target_action.float() - pre_alignment_action.detach().float()
        ).clamp(
            min=-self.policy_guard_eraf_action_geometry_residual_max_abs,
            max=self.policy_guard_eraf_action_geometry_residual_max_abs,
        )

        prefix = min(self.policy_guard_execution_prefix_steps, horizon)
        temporal = target_correction.new_full(
            (batch, horizon), float(self.policy_guard_suffix_loss_weight)
        )
        temporal[:, :prefix] = 1.0
        if action_is_pad is not None:
            temporal = temporal * (~action_is_pad.bool()).float()
        sample_valid = (
            is_counterfactual.bool()
            & direct_action_valid.bool()
            & paired_language_valid.bool()
            & clause_valid
            & phase_valid
            & anchor_valid
        )
        phase_weights = target_correction.new_tensor(
            (
                self.policy_guard_eraf_action_phase_approach_weight,
                self.policy_guard_eraf_action_phase_transport_weight,
                self.policy_guard_eraf_action_phase_release_weight,
            )
        )[privileged_phase]
        valid_weight = (
            temporal
            * sample_valid.float().unsqueeze(-1)
            * phase_weights.unsqueeze(-1)
        )
        valid_count = valid_weight.sum().clamp_min(1.0)

        def imitation(prediction: torch.Tensor) -> torch.Tensor:
            step = F.smooth_l1_loss(
                prediction,
                target_correction,
                reduction="none",
                beta=0.01,
            ).mean(dim=-1)
            return (step * valid_weight).sum() / valid_count

        privileged_imitation = imitation(privileged_correction)
        deployed_imitation = imitation(deployed_correction)
        target_translation = target_correction[..., :3]
        translation_valid = (
            target_translation.norm(dim=-1)
            >= self.policy_guard_eraf_action_phase_direction_min_norm
        )
        direction_weight = valid_weight * translation_valid.float()
        direction_count = direction_weight.sum().clamp_min(1.0)
        direction_cosine = F.cosine_similarity(
            privileged_correction[..., :3],
            target_translation,
            dim=-1,
            eps=1.0e-6,
        )
        direction_loss = (
            (1.0 - direction_cosine) * direction_weight
        ).sum() / direction_count
        distillation_step = (
            deployed_correction - privileged_correction.detach()
        ).square().mean(dim=-1)
        distillation_loss = (
            distillation_step * valid_weight
        ).sum() / valid_count
        native_weight = temporal * (~is_counterfactual.bool()).float().unsqueeze(-1)
        native_count = native_weight.sum().clamp_min(1.0)
        native_zero = (
            deployed_correction.square().mean(dim=-1) * native_weight
        ).sum() / native_count

        total = (
            self.policy_guard_eraf_action_expert_imitation_weight
            * privileged_imitation
            + self.policy_guard_eraf_action_expert_deployed_weight
            * deployed_imitation
            + self.policy_guard_eraf_action_expert_direction_weight
            * direction_loss
            + self.policy_guard_eraf_action_expert_distillation_weight
            * distillation_loss
            + self.policy_guard_eraf_action_expert_native_zero_weight
            * native_zero
        )

        prefix_valid = temporal[:, :prefix] > 0
        prefix_weight = prefix_valid.float()
        prefix_count = prefix_weight.sum(dim=-1).clamp_min(1.0)
        pre_error = (
            (
                pre_alignment_action[:, :prefix].float()
                - target_action[:, :prefix].float()
            ).square().mean(dim=-1)
            * prefix_weight
        ).sum(dim=-1) / prefix_count
        privileged_error = (
            (
                pre_alignment_action[:, :prefix].float()
                + privileged_correction[:, :prefix]
                - target_action[:, :prefix].float()
            ).square().mean(dim=-1)
            * prefix_weight
        ).sum(dim=-1) / prefix_count
        deployed_error = (
            (
                pre_alignment_action[:, :prefix].float()
                + deployed_correction[:, :prefix]
                - target_action[:, :prefix].float()
            ).square().mean(dim=-1)
            * prefix_weight
        ).sum(dim=-1) / prefix_count
        sample_count = sample_valid.float().sum().clamp_min(1.0)
        metrics: dict[str, torch.Tensor] = {
            "loss_pgc_v921_privileged_prefix_imitation": (
                privileged_imitation.detach()
            ),
            "loss_pgc_v921_deployed_prefix_imitation": deployed_imitation.detach(),
            "loss_pgc_v921_privileged_direction": direction_loss.detach(),
            "loss_pgc_v921_privileged_deployed_distillation": (
                distillation_loss.detach()
            ),
            "loss_pgc_v921_native_correction_zero": native_zero.detach(),
            "pgc_v921_valid_fraction": sample_valid.float().mean().detach(),
            "pgc_v921_privileged_prefix_mse_improvement": (
                ((pre_error - privileged_error) * sample_valid.float()).sum()
                / sample_count
            ).detach(),
            "pgc_v921_deployed_prefix_mse_improvement": (
                ((pre_error - deployed_error) * sample_valid.float()).sum()
                / sample_count
            ).detach(),
            "pgc_v921_privileged_direction_cosine": (
                (direction_cosine * direction_weight).sum() / direction_count
            ).detach(),
            "pgc_v921_privileged_anchor_distance": (
                (privileged_distance * sample_valid.float()).sum() / sample_count
            ).detach(),
            "pgc_v921_deployed_phase_accuracy": (
                (
                    (deployed_phase == privileged_phase).float()
                    * sample_valid.float()
                ).sum()
                / sample_count
            ).detach(),
        }
        for phase_id, phase_name in enumerate(("approach", "transport", "release")):
            phase_sample = sample_valid & (privileged_phase == phase_id)
            phase_count = phase_sample.float().sum().clamp_min(1.0)
            metrics[f"pgc_v921_{phase_name}_sample_fraction"] = (
                phase_sample.float().mean().detach()
            )
            metrics[f"pgc_v921_{phase_name}_privileged_mse_improvement"] = (
                ((pre_error - privileged_error) * phase_sample.float()).sum()
                / phase_count
            ).detach()
            metrics[f"pgc_v921_{phase_name}_deployed_mse_improvement"] = (
                ((pre_error - deployed_error) * phase_sample.float()).sum()
                / phase_count
            ).detach()
        return total, metrics

    def _compute_policy_guard_v922_clause_action_ranking_loss(
        self,
        *,
        correct_action: torch.Tensor,
        wrong_clause_action: torch.Tensor,
        target_action: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
        target_labels: Mapping[str, torch.Tensor],
        waypoint_metrics: Mapping[str, torch.Tensor],
        is_counterfactual: torch.Tensor,
        direct_action_valid: torch.Tensor,
        paired_language_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Rank coherent wrong-clause actions below the routed expert prefix.

        Earlier causal training averages clause swaps with entity, reference,
        and anchor interventions.  With small batches, multi-clause examples
        therefore contribute too rarely to establish a directional ordering.
        This objective operates only on valid multi-clause counterfactual
        prefixes and averages active approach/transport/release groups equally.
        The comparison is made at the final deployed action output.
        """
        if wrong_clause_action.shape != correct_action.shape or (
            target_action.shape != correct_action.shape
        ):
            raise ValueError(
                "PGC V9.22 correct, wrong-clause, and expert actions must "
                "share [B,T,A]."
            )
        required = {
            "pgc_v919_selected_clause",
        }
        missing = sorted(required - set(waypoint_metrics))
        if missing:
            raise ValueError(
                f"PGC V9.22 clause ranking metrics are missing: {missing}."
            )
        batch, horizon, _ = correct_action.shape
        prefix = min(self.policy_guard_execution_prefix_steps, horizon)
        valid_step = correct_action.new_ones((batch, prefix), dtype=torch.float32)
        if action_is_pad is not None:
            valid_step = (~action_is_pad[:, :prefix].bool()).float()
        step_count = valid_step.sum(dim=-1).clamp_min(1.0)

        def prefix_error(candidate: torch.Tensor) -> torch.Tensor:
            error = (
                candidate[:, :prefix].float()
                - target_action[:, :prefix].float()
            ).square().mean(dim=-1)
            return (error * valid_step).sum(dim=-1) / step_count

        correct_error = prefix_error(correct_action)
        wrong_error = prefix_error(wrong_clause_action)
        clause_valid = target_labels["clause_valid"].bool()
        selected_clause = waypoint_metrics["pgc_v919_selected_clause"].long()
        if selected_clause.shape != (batch,):
            raise ValueError("PGC V9.22 selected clause must be [B].")
        selected_clause = selected_clause.clamp(0, clause_valid.shape[1] - 1)

        def gather_clause(value: torch.Tensor) -> torch.Tensor:
            index = selected_clause.view(batch, 1, *([1] * (value.ndim - 2)))
            index = index.expand(batch, 1, *value.shape[2:])
            return value.gather(1, index).squeeze(1)

        selected_valid = gather_clause(clause_valid)
        phase_valid = gather_clause(target_labels["phase_valid"].bool())
        phase = gather_clause(target_labels["phase_ids"].long()).clamp(0, 2)
        truth = gather_clause(target_labels["predicate_truth"].float())
        truth_valid = gather_clause(
            target_labels["predicate_truth_valid"].bool()
        )
        phase = torch.where(
            (phase == 1) & truth_valid & (truth >= 0.5),
            torch.full_like(phase, 2),
            phase,
        )
        eligible = (
            is_counterfactual.bool()
            & direct_action_valid.bool()
            & paired_language_valid.bool()
            & (clause_valid.sum(dim=-1) > 1)
            & selected_valid
            & phase_valid
            & (valid_step.sum(dim=-1) > 0)
        )
        margin = float(self.policy_guard_eraf_action_clause_ranking_margin)
        ranking = torch.relu(margin + correct_error - wrong_error)
        active_phase_count = ranking.sum() * 0.0
        balanced_loss = ranking.sum() * 0.0
        metrics: dict[str, torch.Tensor] = {}
        for phase_id, phase_name in enumerate(("approach", "transport", "release")):
            phase_mask = eligible & (phase == phase_id)
            weight = phase_mask.float()
            count = weight.sum().clamp_min(1.0)
            phase_loss = (ranking * weight).sum() / count
            phase_win = (
                (correct_error < wrong_error).float() * weight
            ).sum() / count
            balanced_loss = balanced_loss + phase_loss
            active_phase_count = active_phase_count + phase_mask.any().float()
            metrics[f"loss_pgc_v922_{phase_name}_clause_ranking"] = (
                phase_loss.detach()
            )
            metrics[f"pgc_v922_{phase_name}_clause_win_rate"] = (
                phase_win.detach()
            )
            metrics[f"pgc_v922_{phase_name}_eligible_fraction"] = (
                phase_mask.float().mean().detach()
            )
        balanced_loss = balanced_loss / active_phase_count.clamp_min(1.0)
        weight = eligible.float()
        count = weight.sum().clamp_min(1.0)
        action_delta = (
            wrong_clause_action[:, :prefix].float()
            - correct_action[:, :prefix].float()
        ).square().mean(dim=(1, 2)).sqrt()
        metrics.update(
            {
                "loss_pgc_v922_clause_action_ranking": balanced_loss.detach(),
                "pgc_v922_clause_eligible_fraction": (
                    eligible.float().mean().detach()
                ),
                "pgc_v922_clause_correct_action_win_rate": (
                    ((correct_error < wrong_error).float() * weight).sum()
                    / count
                ).detach(),
                "pgc_v922_clause_margin_satisfied_rate": (
                    ((wrong_error - correct_error >= margin).float() * weight).sum()
                    / count
                ).detach(),
                "pgc_v922_clause_action_delta_rms": (
                    (action_delta * weight).sum() / count
                ).detach(),
                "pgc_v922_active_phase_groups": active_phase_count.detach(),
            }
        )
        return (
            self.policy_guard_eraf_action_clause_ranking_weight * balanced_loss,
            metrics,
        )

    def _compute_policy_guard_v923_alignment_preserving_clause_loss(
        self,
        *,
        correct_action: torch.Tensor,
        wrong_clause_action: torch.Tensor,
        teacher_action: torch.Tensor,
        target_action: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
        target_labels: Mapping[str, torch.Tensor],
        waypoint_metrics: Mapping[str, torch.Tensor],
        is_counterfactual: torch.Tensor,
        direct_action_valid: torch.Tensor,
        paired_language_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Improve wrong-clause ranking without sacrificing V9.21 alignment.

        The admitted V9.21 output is a frozen teacher.  Clause ranking detaches
        the correct branch, so its hinge can only push the wrong clause away
        from the expert prefix.  Teacher preservation and an explicit
        no-regression hinge constrain collateral movement of the shared
        deployed adapter.  All terms use the same balanced phase groups.
        """
        if not (
            correct_action.shape
            == wrong_clause_action.shape
            == teacher_action.shape
            == target_action.shape
        ):
            raise ValueError(
                "PGC V9.23 correct, wrong-clause, teacher, and expert actions "
                "must share [B,T,A]."
            )
        if "pgc_v919_selected_clause" not in waypoint_metrics:
            raise ValueError(
                "PGC V9.23 requires pgc_v919_selected_clause routing metrics."
            )
        batch, horizon, _ = correct_action.shape
        prefix = min(self.policy_guard_execution_prefix_steps, horizon)
        valid_step = correct_action.new_ones((batch, prefix), dtype=torch.float32)
        if action_is_pad is not None:
            valid_step = (~action_is_pad[:, :prefix].bool()).float()
        step_count = valid_step.sum(dim=-1).clamp_min(1.0)

        def prefix_error(candidate: torch.Tensor) -> torch.Tensor:
            error = (
                candidate[:, :prefix].float()
                - target_action[:, :prefix].float()
            ).square().mean(dim=-1)
            return (error * valid_step).sum(dim=-1) / step_count

        correct_error = prefix_error(correct_action)
        wrong_error = prefix_error(wrong_clause_action)
        teacher_error = prefix_error(teacher_action).detach()
        teacher_distance = (
            correct_action[:, :prefix].float()
            - teacher_action[:, :prefix].detach().float()
        ).square().mean(dim=-1)
        teacher_distance = (teacher_distance * valid_step).sum(dim=-1) / step_count

        clause_valid = target_labels["clause_valid"].bool()
        selected_clause = waypoint_metrics["pgc_v919_selected_clause"].long()
        if selected_clause.shape != (batch,):
            raise ValueError("PGC V9.23 selected clause must be [B].")
        selected_clause = selected_clause.clamp(0, clause_valid.shape[1] - 1)

        def gather_clause(value: torch.Tensor) -> torch.Tensor:
            index = selected_clause.view(batch, 1, *([1] * (value.ndim - 2)))
            index = index.expand(batch, 1, *value.shape[2:])
            return value.gather(1, index).squeeze(1)

        selected_valid = gather_clause(clause_valid)
        phase_valid = gather_clause(target_labels["phase_valid"].bool())
        phase = gather_clause(target_labels["phase_ids"].long()).clamp(0, 2)
        truth = gather_clause(target_labels["predicate_truth"].float())
        truth_valid = gather_clause(
            target_labels["predicate_truth_valid"].bool()
        )
        phase = torch.where(
            (phase == 1) & truth_valid & (truth >= 0.5),
            torch.full_like(phase, 2),
            phase,
        )
        eligible = (
            is_counterfactual.bool()
            & direct_action_valid.bool()
            & paired_language_valid.bool()
            & (clause_valid.sum(dim=-1) > 1)
            & selected_valid
            & phase_valid
            & (valid_step.sum(dim=-1) > 0)
        )

        margin = float(self.policy_guard_eraf_action_clause_ranking_margin)
        # The detached correct error is deliberate: ranking can degrade only
        # the negative route, while the teacher and expert losses own the
        # correct route.
        ranking = torch.relu(margin + correct_error.detach() - wrong_error)
        alignment_guard = torch.relu(correct_error - teacher_error)
        active_phase_count = ranking.sum() * 0.0
        ranking_balanced = ranking.sum() * 0.0
        teacher_balanced = teacher_distance.sum() * 0.0
        guard_balanced = alignment_guard.sum() * 0.0
        metrics: dict[str, torch.Tensor] = {}
        for phase_id, phase_name in enumerate(("approach", "transport", "release")):
            phase_mask = eligible & (phase == phase_id)
            phase_weight = phase_mask.float()
            phase_count = phase_weight.sum().clamp_min(1.0)
            phase_ranking = (ranking * phase_weight).sum() / phase_count
            phase_teacher = (teacher_distance * phase_weight).sum() / phase_count
            phase_guard = (alignment_guard * phase_weight).sum() / phase_count
            ranking_balanced = ranking_balanced + phase_ranking
            teacher_balanced = teacher_balanced + phase_teacher
            guard_balanced = guard_balanced + phase_guard
            active_phase_count = active_phase_count + phase_mask.any().float()
            metrics[f"loss_pgc_v923_{phase_name}_clause_ranking"] = (
                phase_ranking.detach()
            )
            metrics[f"loss_pgc_v923_{phase_name}_teacher_preservation"] = (
                phase_teacher.detach()
            )
            metrics[f"loss_pgc_v923_{phase_name}_alignment_guard"] = (
                phase_guard.detach()
            )
        divisor = active_phase_count.clamp_min(1.0)
        ranking_balanced = ranking_balanced / divisor
        teacher_balanced = teacher_balanced / divisor
        guard_balanced = guard_balanced / divisor
        total = (
            self.policy_guard_eraf_action_clause_ranking_weight * ranking_balanced
            + self.policy_guard_eraf_action_clause_teacher_weight * teacher_balanced
            + self.policy_guard_eraf_action_clause_alignment_guard_weight
            * guard_balanced
        )
        weight = eligible.float()
        count = weight.sum().clamp_min(1.0)
        metrics.update(
            {
                "loss_pgc_v923_clause_action_ranking": ranking_balanced.detach(),
                "loss_pgc_v923_teacher_preservation": teacher_balanced.detach(),
                "loss_pgc_v923_alignment_guard": guard_balanced.detach(),
                "pgc_v923_clause_eligible_fraction": weight.mean().detach(),
                "pgc_v923_clause_correct_action_win_rate": (
                    ((correct_error < wrong_error).float() * weight).sum() / count
                ).detach(),
                "pgc_v923_alignment_nonregression_rate": (
                    ((correct_error <= teacher_error).float() * weight).sum() / count
                ).detach(),
                "pgc_v923_correct_minus_teacher_mse": (
                    ((correct_error - teacher_error) * weight).sum() / count
                ).detach(),
                "pgc_v923_active_phase_groups": active_phase_count.detach(),
            }
        )
        return total, metrics

    def _compute_policy_guard_v924_isolated_clause_residual_loss(
        self,
        *,
        correct_action: torch.Tensor,
        wrong_clause_action: torch.Tensor,
        teacher_action: torch.Tensor,
        base_action: torch.Tensor,
        target_action: torch.Tensor,
        correct_route_metrics: Mapping[str, torch.Tensor],
        wrong_route_metrics: Mapping[str, torch.Tensor],
        action_is_pad: Optional[torch.Tensor],
        target_labels: Mapping[str, torch.Tensor],
        waypoint_metrics: Mapping[str, torch.Tensor],
        is_counterfactual: torch.Tensor,
        direct_action_valid: torch.Tensor,
        paired_language_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Separate correct-action preservation from wrong-clause rejection.

        The V9.21 teacher and its action adapter are immutable.  The new clause
        residual is identity initialized and can only retain the teacher or
        move toward Base.  Correct-route preservation covers every valid
        action sample; only coherent multi-clause counterfactual swaps receive
        negative suppression and ranking gradients.
        """
        if not (
            correct_action.shape
            == wrong_clause_action.shape
            == teacher_action.shape
            == base_action.shape
            == target_action.shape
        ):
            raise ValueError(
                "PGC V9.24 correct/wrong/teacher/Base/expert actions must "
                "share [B,T,A]."
            )
        required_route = {"pgc_v924_clause_suppression_training"}
        missing = sorted(required_route - set(correct_route_metrics))
        missing += sorted(required_route - set(wrong_route_metrics))
        if missing:
            raise ValueError(
                f"PGC V9.24 clause-retention metrics are missing: {missing}."
            )
        if "pgc_v919_selected_clause" not in waypoint_metrics:
            raise ValueError(
                "PGC V9.24 requires pgc_v919_selected_clause routing metrics."
            )

        batch, horizon, _ = correct_action.shape
        prefix = min(self.policy_guard_execution_prefix_steps, horizon)
        valid_step = correct_action.new_ones((batch, prefix), dtype=torch.float32)
        if action_is_pad is not None:
            valid_step = (~action_is_pad[:, :prefix].bool()).float()
        step_count = valid_step.sum(dim=-1).clamp_min(1.0)

        def prefix_error(candidate: torch.Tensor) -> torch.Tensor:
            error = (
                candidate[:, :prefix].float()
                - target_action[:, :prefix].float()
            ).square().mean(dim=-1)
            return (error * valid_step).sum(dim=-1) / step_count

        correct_error = prefix_error(correct_action)
        wrong_error = prefix_error(wrong_clause_action)
        teacher_error = prefix_error(teacher_action).detach()
        base_error = prefix_error(base_action).detach()
        teacher_distance = (
            correct_action[:, :prefix].float()
            - teacher_action[:, :prefix].detach().float()
        ).square().mean(dim=-1)
        teacher_distance = (teacher_distance * valid_step).sum(dim=-1) / step_count
        correct_suppression = correct_route_metrics[
            "pgc_v924_clause_suppression_training"
        ][:, :prefix].float()
        wrong_suppression = wrong_route_metrics[
            "pgc_v924_clause_suppression_training"
        ][:, :prefix].float()

        clause_valid = target_labels["clause_valid"].bool()
        selected_clause = waypoint_metrics["pgc_v919_selected_clause"].long()
        if selected_clause.shape != (batch,):
            raise ValueError("PGC V9.24 selected clause must be [B].")
        selected_clause = selected_clause.clamp(0, clause_valid.shape[1] - 1)

        def gather_clause(value: torch.Tensor) -> torch.Tensor:
            index = selected_clause.view(batch, 1, *([1] * (value.ndim - 2)))
            index = index.expand(batch, 1, *value.shape[2:])
            return value.gather(1, index).squeeze(1)

        selected_valid = gather_clause(clause_valid)
        phase_valid = gather_clause(target_labels["phase_valid"].bool())
        phase = gather_clause(target_labels["phase_ids"].long()).clamp(0, 2)
        truth = gather_clause(target_labels["predicate_truth"].float())
        truth_valid = gather_clause(
            target_labels["predicate_truth_valid"].bool()
        )
        phase = torch.where(
            (phase == 1) & truth_valid & (truth >= 0.5),
            torch.full_like(phase, 2),
            phase,
        )
        correct_valid = valid_step.sum(dim=-1) > 0
        expert_valid = (
            is_counterfactual.bool()
            & direct_action_valid.bool()
            & paired_language_valid.bool()
            & selected_valid
            & phase_valid
            & correct_valid
        )
        negative_valid = expert_valid & (clause_valid.sum(dim=-1) > 1)

        correct_step_count = valid_step.sum().clamp_min(1.0)
        correct_zero = (
            correct_suppression.square() * valid_step
        ).sum() / correct_step_count
        teacher_preservation = teacher_distance.mean()
        admitted_teacher = expert_valid & (teacher_error < base_error)
        admitted_count = admitted_teacher.float().sum().clamp_min(1.0)
        alignment_guard = (
            torch.relu(correct_error - teacher_error)
            * admitted_teacher.float()
        ).sum() / admitted_count

        margin = float(self.policy_guard_eraf_action_clause_ranking_margin)
        ranking = torch.relu(margin + correct_error.detach() - wrong_error)
        active_phase_count = ranking.sum() * 0.0
        ranking_balanced = ranking.sum() * 0.0
        wrong_suppression_balanced = ranking.sum() * 0.0
        metrics: dict[str, torch.Tensor] = {}
        wrong_step_target = (1.0 - wrong_suppression).square()
        wrong_per_sample = (
            wrong_step_target * valid_step
        ).sum(dim=-1) / step_count
        for phase_id, phase_name in enumerate(("approach", "transport", "release")):
            phase_mask = negative_valid & (phase == phase_id)
            weight = phase_mask.float()
            count = weight.sum().clamp_min(1.0)
            phase_ranking = (ranking * weight).sum() / count
            phase_wrong = (wrong_per_sample * weight).sum() / count
            ranking_balanced = ranking_balanced + phase_ranking
            wrong_suppression_balanced = (
                wrong_suppression_balanced + phase_wrong
            )
            active_phase_count = active_phase_count + phase_mask.any().float()
            metrics[f"loss_pgc_v924_{phase_name}_clause_ranking"] = (
                phase_ranking.detach()
            )
            metrics[f"loss_pgc_v924_{phase_name}_wrong_suppression"] = (
                phase_wrong.detach()
            )
        divisor = active_phase_count.clamp_min(1.0)
        ranking_balanced = ranking_balanced / divisor
        wrong_suppression_balanced = wrong_suppression_balanced / divisor
        total = (
            self.policy_guard_eraf_action_clause_ranking_weight
            * ranking_balanced
            + self.policy_guard_eraf_action_clause_teacher_weight
            * (teacher_preservation + correct_zero)
            + self.policy_guard_eraf_action_clause_alignment_guard_weight
            * alignment_guard
            + self.policy_guard_eraf_action_clause_wrong_suppression_weight
            * wrong_suppression_balanced
        )
        negative_weight = negative_valid.float()
        negative_count = negative_weight.sum().clamp_min(1.0)
        metrics.update(
            {
                "loss_pgc_v924_clause_action_ranking": ranking_balanced.detach(),
                "loss_pgc_v924_teacher_preservation": (
                    teacher_preservation.detach()
                ),
                "loss_pgc_v924_correct_suppression_zero": correct_zero.detach(),
                "loss_pgc_v924_alignment_guard": alignment_guard.detach(),
                "loss_pgc_v924_wrong_suppression": (
                    wrong_suppression_balanced.detach()
                ),
                "pgc_v924_clause_eligible_fraction": (
                    negative_valid.float().mean().detach()
                ),
                "pgc_v924_clause_correct_action_win_rate": (
                    (
                        (correct_error < wrong_error).float()
                        * negative_weight
                    ).sum()
                    / negative_count
                ).detach(),
                "pgc_v924_alignment_nonregression_rate": (
                    (
                        (correct_error <= teacher_error).float()
                        * admitted_teacher.float()
                    ).sum()
                    / admitted_count
                ).detach(),
                "pgc_v924_correct_minus_teacher_mse": (
                    (
                        (correct_error - teacher_error)
                        * admitted_teacher.float()
                    ).sum()
                    / admitted_count
                ).detach(),
                "pgc_v924_teacher_improves_base_rate": (
                    (teacher_error < base_error).float() * expert_valid.float()
                ).sum()
                / expert_valid.float().sum().clamp_min(1.0),
                "pgc_v924_active_phase_groups": active_phase_count.detach(),
            }
        )
        return total, metrics

    def _training_loss_policy_guard_v925_action_context(
        self,
        *,
        inputs: dict[str, Any],
        full_context_mask: torch.Tensor,
        state_only_context_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Train internal ERAF conditioning with no action-space residual.

        The released Action Expert, Video Expert, ERAF, memory, and gate are
        frozen. Only the ERAF-to-context projection is optimized. Supervision
        is restricted to direct counterfactual rows; native/correct rollout
        continues to use the untouched Base candidate at deployment.
        """
        if not (
            self.policy_guard_enabled
            and self.policy_guard_version == 9
            and self.policy_guard_eraf_grounding_objective_version >= 25
            and self.policy_guard_eraf_training_stage == "action"
        ):
            raise RuntimeError(
                "Internal ERAF action-context training requires PGC v9 "
                "objective 25+ in the action stage."
            )
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        is_counterfactual = inputs["pgc_is_counterfactual"]
        direct_action_valid = inputs["pgc_direct_action_valid"]
        paired_language_valid = inputs["pgc_paired_language_valid"]
        source_context = inputs["pgc_source_context"]
        source_context_mask = inputs["pgc_source_context_mask"]
        if any(
            value is None
            for value in (
                is_counterfactual,
                direct_action_valid,
                paired_language_valid,
                source_context,
                source_context_mask,
            )
        ):
            raise ValueError(
                "Internal ERAF action-context training requires direct "
                "counterfactual and paired-language provenance."
            )

        batch_size = int(action.shape[0])
        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=action.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(
            action, noise_action, timestep_action
        )
        target_velocity = self.train_action_scheduler.training_target(
            action, noise_action, timestep_action
        )

        with torch.no_grad():
            first_frame_latents = inputs["input_latents"][:, :, 0:1]
            timestep_video = torch.zeros(
                (batch_size,),
                device=first_frame_latents.device,
                dtype=first_frame_latents.dtype,
            )
            video_pre = self.video_expert.pre_dit(
                x=first_frame_latents,
                timestep=timestep_video,
                context=inputs["context"],
                context_mask=full_context_mask,
                action=None,
                fuse_vae_embedding_in_latents=inputs[
                    "fuse_vae_embedding_in_latents"
                ],
            )
            video_seq_len = int(video_pre["tokens"].shape[1])
            video_tokens_per_frame = int(
                video_pre["meta"]["tokens_per_frame"]
            )
            attention_mask = self._build_mot_attention_mask(
                video_seq_len=video_seq_len,
                action_seq_len=int(action.shape[1]),
                video_tokens_per_frame=video_tokens_per_frame,
                device=video_pre["tokens"].device,
                num_queries=0,
                action_reads_raw_video=True,
            )
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
                return_final_hidden=True,
            )
            if not isinstance(prefill_result, tuple):
                raise RuntimeError(
                    "Internal ERAF context training requires Video hidden state."
                )
            video_kv_cache, final_video_hidden = prefill_result
            language_context_len = int(inputs["language_context_len"])
            target_labels = self._policy_guard_v9_labels(inputs)
            source_labels = self._policy_guard_v9_labels(
                inputs, prefix="source_"
            )
            target_state = {
                "phase_safe_memory_state_ids": target_labels[
                    "phase_safe_memory_previous_state_ids"
                ],
                "phase_safe_memory_valid": target_labels[
                    "phase_safe_memory_state_valid"
                ],
            }
            source_state = {
                "phase_safe_memory_state_ids": source_labels[
                    "phase_safe_memory_previous_state_ids"
                ],
                "phase_safe_memory_valid": source_labels[
                    "phase_safe_memory_state_valid"
                ],
            }
            target_queries, _, _, _ = self._encode_policy_guard_eraf(
                final_video_hidden=final_video_hidden,
                current_visual_hidden=video_pre["tokens"],
                video_tokens_per_frame=video_tokens_per_frame,
                context=inputs["context"],
                context_mask=full_context_mask,
                language_context_len=language_context_len,
                policy_guard_state=target_state,
                proprio=inputs.get("proprio_current"),
            )
            source_queries, _, _, _ = self._encode_policy_guard_eraf(
                final_video_hidden=final_video_hidden,
                current_visual_hidden=video_pre["tokens"],
                video_tokens_per_frame=video_tokens_per_frame,
                context=source_context,
                context_mask=source_context_mask,
                language_context_len=language_context_len,
                policy_guard_state=source_state,
                proprio=inputs.get("proprio_current"),
            )
            base_velocity = self._predict_action_noise_with_cache(
                latents_action=noisy_action,
                timestep_action=timestep_action,
                context=inputs["context"],
                context_mask=full_context_mask,
                state_only_context_mask=state_only_context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )

        correct_velocity, injection_metrics = (
            self._forward_policy_guard_action_from_cache(
                action_tokens=noisy_action,
                timestep_action=timestep_action,
                context=inputs["context"],
                full_context_mask=full_context_mask,
                state_only_context_mask=state_only_context_mask,
                video_kv_cache=video_kv_cache,
                video_seq_len=video_seq_len,
                video_tokens_per_frame=video_tokens_per_frame,
                routed_goal_queries=target_queries.detach(),
                return_metrics=True,
                checkpoint_frozen_action_expert=True,
            )
        )
        wrong_velocity, _ = self._forward_policy_guard_action_from_cache(
            action_tokens=noisy_action,
            timestep_action=timestep_action,
            context=inputs["context"],
            full_context_mask=full_context_mask,
            state_only_context_mask=state_only_context_mask,
            video_kv_cache=video_kv_cache,
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            routed_goal_queries=source_queries.detach(),
            return_metrics=True,
            checkpoint_frozen_action_expert=True,
        )

        correct_error = self._compute_action_loss_per_sample(
            pred_action=correct_velocity,
            target_action=target_velocity,
            action_is_pad=action_is_pad,
        )
        wrong_error = self._compute_action_loss_per_sample(
            pred_action=wrong_velocity,
            target_action=target_velocity,
            action_is_pad=action_is_pad,
        )
        base_error = self._compute_action_loss_per_sample(
            pred_action=base_velocity,
            target_action=target_velocity,
            action_is_pad=action_is_pad,
        )
        action_weight = self.train_action_scheduler.training_weight(
            timestep_action
        ).to(device=correct_error.device, dtype=correct_error.dtype)
        if action_weight.ndim == 0:
            action_weight = action_weight.expand(batch_size)
        else:
            action_weight = action_weight.reshape(batch_size)
        counterfactual_valid = direct_action_valid.to(
            device=action.device, dtype=torch.bool
        ) & is_counterfactual.to(device=action.device, dtype=torch.bool)
        semantic_valid = counterfactual_valid & paired_language_valid.to(
            device=action.device, dtype=torch.bool
        )
        imitation_loss = self._masked_policy_guard_mean(
            correct_error * action_weight, counterfactual_valid
        )
        ranking_per_sample = torch.relu(
            float(self.policy_guard_eraf_action_causal_margin)
            + correct_error.detach()
            - wrong_error
        )
        ranking_loss = self._masked_policy_guard_mean(
            ranking_per_sample, semantic_valid
        )
        total = (
            self.loss_lambda_action * imitation_loss
            + self.policy_guard_eraf_action_causal_ranking_weight
            * ranking_loss
        )
        valid_count = counterfactual_valid.float().sum().clamp_min(1.0)
        semantic_count = semantic_valid.float().sum().clamp_min(1.0)
        metrics: dict[str, torch.Tensor] = {
            "loss_pgc_v925_internal_action_flow": imitation_loss.detach(),
            "loss_pgc_v925_wrong_semantic_ranking": ranking_loss.detach(),
            "pgc_v925_counterfactual_valid_fraction": (
                counterfactual_valid.float().mean()
            ),
            "pgc_v925_semantic_valid_fraction": semantic_valid.float().mean(),
            "pgc_v925_internal_velocity_delta_rms": (
                (correct_velocity - base_velocity)
                .float()
                .square()
                .mean()
                .sqrt()
            ),
            "pgc_v925_expert_improvement_rate": (
                ((correct_error < base_error).float() * counterfactual_valid.float())
                .sum()
                / valid_count
            ),
            "pgc_v925_wrong_semantics_worse_rate": (
                ((wrong_error > correct_error).float() * semantic_valid.float())
                .sum()
                / semantic_count
            ),
            "pgc_v925_native_action_supervision_enabled": (
                correct_error.new_zeros(())
            ),
            "pgc_v925_post_action_residual_enabled": (
                correct_error.new_zeros(())
            ),
            "pgc_base_policy_frozen": correct_error.new_ones(()),
            "pgc_v9_stage_action": correct_error.new_ones(()),
            "pgc_v9_stage_grounding": correct_error.new_zeros(()),
            "pgc_v9_stage_verifier": correct_error.new_zeros(()),
        }
        metrics.update(injection_metrics)
        detached = detached_policy_guard_metrics(metrics)
        detached.update(
            {
                "loss_video": 0.0,
                "loss_action": float(total.detach().float().item()),
                "loss_pgc_action": float(imitation_loss.detach().float().item()),
                "loss_pgc_v9_eraf": 0.0,
                "pgc_video_loss_optimization_weight": 0.0,
                "pgc_action_effective_weight": 1.0,
            }
        )
        return total, detached

    def _training_loss_policy_guard_v926_eraf_expert_lora(
        self,
        *,
        inputs: dict[str, Any],
        full_context_mask: torch.Tensor,
        state_only_context_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Jointly adapt the one ERAF policy path with shared Expert LoRA.

        ERAF, completion memory, and all legacy post-action residuals remain
        frozen. Future-video flow and paired wrong-language ranking update the
        Video LoRA; native/counterfactual action flow and semantic ranking
        update the Action LoRA and ERAF context injector. The privileged ERAF
        objective is retained as a backbone-preservation constraint.
        """
        if not (
            self.policy_guard_enabled
            and self.policy_guard_version == 9
            and self.policy_guard_eraf_grounding_objective_version >= 26
            and self.policy_guard_eraf_training_stage == "action"
            and self.lora_enabled
        ):
            raise RuntimeError(
                "ERAF-only shared Expert LoRA training requires PGC v9 "
                "objective 26+ in the action stage with LoRA enabled."
            )
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]
        is_counterfactual = inputs["pgc_is_counterfactual"]
        direct_action_valid = inputs["pgc_direct_action_valid"]
        paired_language_valid = inputs["pgc_paired_language_valid"]
        source_context = inputs["pgc_source_context"]
        source_context_mask = inputs["pgc_source_context_mask"]
        if any(
            value is None
            for value in (
                is_counterfactual,
                direct_action_valid,
                paired_language_valid,
                source_context,
                source_context_mask,
            )
        ):
            raise ValueError(
                "PGC V9.26 requires native/counterfactual provenance and "
                "same-state paired language."
            )

        batch_size = int(action.shape[0])
        input_latents = inputs["input_latents"]
        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=input_latents.device,
            dtype=input_latents.dtype,
        )
        noisy_video = self.train_video_scheduler.add_noise(
            input_latents, noise_video, timestep_video
        )
        target_video = self.train_video_scheduler.training_target(
            input_latents, noise_video, timestep_video
        )
        if inputs["first_frame_latents"] is not None:
            noisy_video[:, :, 0:1] = inputs["first_frame_latents"]

        video_pre = self.video_expert.pre_dit(
            x=noisy_video,
            timestep=timestep_video,
            context=inputs["context"],
            context_mask=full_context_mask,
            action=action,
            fuse_vae_embedding_in_latents=inputs[
                "fuse_vae_embedding_in_latents"
            ],
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        video_tokens_per_frame = int(
            video_pre["meta"]["tokens_per_frame"]
        )
        video_attention_mask = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=video_pre["tokens"].device,
        )
        prefill_result = self.mot.prefill_video_cache(
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
        if not isinstance(prefill_result, tuple):
            raise RuntimeError("PGC V9.26 Video prefill did not return hidden state.")
        video_kv_cache, final_video_hidden = prefill_result
        pred_video = self.video_expert.post_dit(final_video_hidden, video_pre)

        include_initial_video_step = inputs["first_frame_latents"] is None
        target_video_for_loss = target_video
        if not include_initial_video_step:
            pred_video = pred_video[:, :, 1:]
            target_video_for_loss = target_video_for_loss[:, :, 1:]
        correct_video_error = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video_for_loss,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
        )
        video_weight = self.train_video_scheduler.training_weight(
            timestep_video
        ).to(device=correct_video_error.device, dtype=correct_video_error.dtype)
        if video_weight.ndim == 0:
            video_weight = video_weight.expand(batch_size)
        else:
            video_weight = video_weight.reshape(batch_size)
        world_flow_loss = (correct_video_error * video_weight).mean()

        wrong_video_pre = self.video_expert.pre_dit(
            x=noisy_video,
            timestep=timestep_video,
            context=source_context,
            context_mask=source_context_mask,
            action=action,
            fuse_vae_embedding_in_latents=inputs[
                "fuse_vae_embedding_in_latents"
            ],
        )
        wrong_video_hidden = self.mot.prefill_video_cache(
            video_tokens=wrong_video_pre["tokens"],
            video_freqs=wrong_video_pre["freqs"],
            video_t_mod=wrong_video_pre["t_mod"],
            video_context_payload={
                "context": wrong_video_pre["context"],
                "mask": wrong_video_pre["context_mask"],
            },
            video_attention_mask=video_attention_mask,
            return_final_hidden=True,
        )
        if not isinstance(wrong_video_hidden, tuple):
            raise RuntimeError(
                "PGC V9.26 wrong-language Video prefill did not return hidden."
            )
        _, wrong_video_hidden = wrong_video_hidden
        wrong_pred_video = self.video_expert.post_dit(
            wrong_video_hidden, wrong_video_pre
        )
        if not include_initial_video_step:
            wrong_pred_video = wrong_pred_video[:, :, 1:]
        wrong_video_error = self._compute_video_loss_per_sample(
            pred_video=wrong_pred_video,
            target_video=target_video_for_loss,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
        )

        direct_action_valid = direct_action_valid.to(
            device=action.device, dtype=torch.bool
        )
        is_counterfactual = is_counterfactual.to(
            device=action.device, dtype=torch.bool
        )
        paired_language_valid = paired_language_valid.to(
            device=action.device, dtype=torch.bool
        )
        native_valid = direct_action_valid & ~is_counterfactual
        counterfactual_valid = direct_action_valid & is_counterfactual
        semantic_valid = counterfactual_valid & paired_language_valid
        world_language_ranking = self._masked_policy_guard_mean(
            torch.relu(
                self.policy_guard_eraf_expert_lora_world_language_margin
                + correct_video_error.detach()
                - wrong_video_error
            ),
            semantic_valid,
        )

        language_context_len = int(inputs["language_context_len"])
        target_labels = self._policy_guard_v9_labels(inputs)
        source_labels = self._policy_guard_v9_labels(inputs, prefix="source_")
        target_state = {
            "phase_safe_memory_state_ids": target_labels[
                "phase_safe_memory_previous_state_ids"
            ],
            "phase_safe_memory_valid": target_labels[
                "phase_safe_memory_state_valid"
            ],
        }
        source_state = {
            "phase_safe_memory_state_ids": source_labels[
                "phase_safe_memory_previous_state_ids"
            ],
            "phase_safe_memory_valid": source_labels[
                "phase_safe_memory_state_valid"
            ],
        }
        target_queries, _, target_eraf_outputs, target_goal_metrics = (
            self._encode_policy_guard_eraf(
                final_video_hidden=final_video_hidden,
                current_visual_hidden=video_pre["tokens"],
                video_tokens_per_frame=video_tokens_per_frame,
                context=inputs["context"],
                context_mask=full_context_mask,
                language_context_len=language_context_len,
                policy_guard_state=target_state,
                proprio=inputs.get("proprio_current"),
            )
        )
        source_queries, _, source_eraf_outputs, source_goal_metrics = (
            self._encode_policy_guard_eraf(
                final_video_hidden=wrong_video_hidden,
                current_visual_hidden=wrong_video_pre["tokens"],
                video_tokens_per_frame=video_tokens_per_frame,
                context=source_context,
                context_mask=source_context_mask,
                language_context_len=language_context_len,
                policy_guard_state=source_state,
                proprio=inputs.get("proprio_current"),
            )
        )
        target_eraf_loss, target_eraf_metrics = entity_relation_affordance_loss(
            target_eraf_outputs,
            target_labels,
            weights=self.policy_guard_eraf_loss_weights,
        )
        source_eraf_loss, source_eraf_metrics = entity_relation_affordance_loss(
            source_eraf_outputs,
            source_labels,
            weights=self.policy_guard_eraf_loss_weights,
        )
        eraf_preservation_loss = target_eraf_loss + 0.5 * source_eraf_loss

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=action.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(
            action, noise_action, timestep_action
        )
        target_action = self.train_action_scheduler.training_target(
            action, noise_action, timestep_action
        )
        correct_action, injection_metrics = (
            self._forward_policy_guard_action_from_cache(
                action_tokens=noisy_action,
                timestep_action=timestep_action,
                context=inputs["context"],
                full_context_mask=full_context_mask,
                state_only_context_mask=state_only_context_mask,
                video_kv_cache=video_kv_cache,
                video_seq_len=video_seq_len,
                video_tokens_per_frame=video_tokens_per_frame,
                routed_goal_queries=target_queries,
                return_metrics=True,
            )
        )
        wrong_action = self._forward_policy_guard_action_from_cache(
            action_tokens=noisy_action,
            timestep_action=timestep_action,
            context=inputs["context"],
            full_context_mask=full_context_mask,
            state_only_context_mask=state_only_context_mask,
            video_kv_cache=video_kv_cache,
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            routed_goal_queries=source_queries,
        )
        correct_action_error = self._compute_action_loss_per_sample(
            pred_action=correct_action,
            target_action=target_action,
            action_is_pad=action_is_pad,
        )
        wrong_action_error = self._compute_action_loss_per_sample(
            pred_action=wrong_action,
            target_action=target_action,
            action_is_pad=action_is_pad,
        )
        action_weight = self.train_action_scheduler.training_weight(
            timestep_action
        ).to(device=action.device, dtype=correct_action_error.dtype)
        if action_weight.ndim == 0:
            action_weight = action_weight.expand(batch_size)
        else:
            action_weight = action_weight.reshape(batch_size)
        native_action_loss = self._masked_policy_guard_mean(
            correct_action_error * action_weight, native_valid
        )
        counterfactual_action_loss = self._masked_policy_guard_mean(
            correct_action_error * action_weight, counterfactual_valid
        )
        wrong_semantic_ranking = self._masked_policy_guard_mean(
            torch.relu(
                self.policy_guard_eraf_action_causal_margin
                + correct_action_error.detach()
                - wrong_action_error
            ),
            semantic_valid,
        )

        lora_terms = [
            parameter.float().square().mean()
            for name, parameter in self.mot.named_parameters()
            if name.endswith(".lora_B")
        ]
        if not lora_terms:
            raise RuntimeError("PGC V9.26 found no LoRA-B parameters.")
        lora_regularization = torch.stack(lora_terms).mean()
        action_objective = self.loss_lambda_action * (
            self.policy_guard_eraf_expert_lora_native_action_weight
            * native_action_loss
            + self.policy_guard_eraf_expert_lora_counterfactual_action_weight
            * counterfactual_action_loss
        )
        total = (
            self.loss_lambda_video * world_flow_loss
            + self.policy_guard_eraf_expert_lora_world_language_weight
            * world_language_ranking
            + action_objective
            + self.policy_guard_eraf_action_causal_ranking_weight
            * wrong_semantic_ranking
            + self.policy_guard_eraf_grounding_aux_weight
            * eraf_preservation_loss
            + self.policy_guard_eraf_expert_lora_regularization_weight
            * lora_regularization
        )

        semantic_count = semantic_valid.float().sum().clamp_min(1.0)
        metrics: dict[str, torch.Tensor] = {
            "loss_pgc_v926_world_flow": world_flow_loss.detach(),
            "loss_pgc_v926_world_language_ranking": (
                world_language_ranking.detach()
            ),
            "loss_pgc_v926_native_action_flow": native_action_loss.detach(),
            "loss_pgc_v926_counterfactual_action_flow": (
                counterfactual_action_loss.detach()
            ),
            "loss_pgc_v926_wrong_semantic_ranking": (
                wrong_semantic_ranking.detach()
            ),
            "loss_pgc_v926_eraf_preservation": eraf_preservation_loss.detach(),
            "loss_pgc_v926_lora_regularization": lora_regularization.detach(),
            "pgc_v926_native_fraction": native_valid.float().mean(),
            "pgc_v926_counterfactual_fraction": (
                counterfactual_valid.float().mean()
            ),
            "pgc_v926_semantic_fraction": semantic_valid.float().mean(),
            "pgc_v926_world_wrong_language_worse_rate": (
                ((wrong_video_error > correct_video_error).float()
                 * semantic_valid.float()).sum()
                / semantic_count
            ),
            "pgc_v926_action_wrong_semantics_worse_rate": (
                ((wrong_action_error > correct_action_error).float()
                 * semantic_valid.float()).sum()
                / semantic_count
            ),
            "pgc_v926_action_supervised_fraction": (
                direct_action_valid.float().mean()
            ),
            "pgc_v926_eraf_frozen": correct_action_error.new_ones(()),
            "pgc_v926_single_eraf_path": correct_action_error.new_ones(()),
            "pgc_v926_post_action_residual_enabled": (
                correct_action_error.new_zeros(())
            ),
            "pgc_base_policy_frozen": correct_action_error.new_ones(()),
            "pgc_v9_stage_action": correct_action_error.new_ones(()),
            "pgc_v9_stage_grounding": correct_action_error.new_zeros(()),
            "pgc_v9_stage_verifier": correct_action_error.new_zeros(()),
        }
        metrics.update(injection_metrics)
        metrics.update(target_goal_metrics)
        metrics.update(
            {
                f"pgc_v9_source_{name.removeprefix('pgc_v9_')}": value
                for name, value in source_goal_metrics.items()
            }
        )
        metrics.update(target_eraf_metrics)
        metrics.update(
            {
                f"pgc_v9_source_{name.removeprefix('pgc_v9_')}": value
                for name, value in source_eraf_metrics.items()
            }
        )
        detached = detached_policy_guard_metrics(metrics)
        detached.update(
            {
                "loss_video": float(
                    (self.loss_lambda_video * world_flow_loss)
                    .detach()
                    .float()
                    .item()
                ),
                "loss_action": float(action_objective.detach().float().item()),
                "loss_pgc_action": float(
                    counterfactual_action_loss.detach().float().item()
                ),
                "loss_pgc_v9_eraf": float(
                    eraf_preservation_loss.detach().float().item()
                ),
                "pgc_video_loss_optimization_weight": float(
                    self.loss_lambda_video
                ),
                "pgc_action_effective_weight": float(self.loss_lambda_action),
                "pgc_v9_grounding_effective_weight": float(
                    self.policy_guard_eraf_grounding_aux_weight
                ),
            }
        )
        return total, detached

    def _training_loss_policy_guard_v5(
        self,
        *,
        inputs: dict[str, Any],
        full_context_mask: torch.Tensor,
        state_only_context_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Train a language-identifiable, prefix-aligned protected proposal."""
        if self.policy_guard_version not in {5, 6, 7, 8, 9}:
            raise RuntimeError(
                "Paired-language training requires PGC v5/v6/v7/v8/v9."
            )
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        is_counterfactual = inputs["pgc_is_counterfactual"]
        direct_action_valid = inputs["pgc_direct_action_valid"]
        paired_language_valid = inputs["pgc_paired_language_valid"]
        source_context = inputs["pgc_source_context"]
        source_context_mask = inputs["pgc_source_context_mask"]
        if any(
            value is None
            for value in (
                is_counterfactual,
                direct_action_valid,
                paired_language_valid,
                source_context,
                source_context_mask,
            )
        ):
            raise ValueError(
                "PGC v5 requires direct-action and paired-language provenance."
            )

        initial_action_noise = torch.randn_like(action)
        (
            base_action,
            final_video_hidden,
            neutral_visual_hidden,
            video_tokens_per_frame,
        ) = (
            self._rollout_policy_guard_base_action(
                first_frame_latents=inputs["input_latents"][:, :, 0:1],
                initial_action_noise=initial_action_noise,
                context=inputs["context"],
                full_context_mask=full_context_mask,
                state_only_context_mask=state_only_context_mask,
                fuse_vae_embedding_in_latents=inputs[
                    "fuse_vae_embedding_in_latents"
                ],
                num_inference_steps=(
                    self.policy_guard_rollout_num_inference_steps
                ),
            )
        )
        binding_metrics: dict[str, torch.Tensor] = {}
        prototype_metrics: dict[str, torch.Tensor] = {}
        eraf_loss_metrics: dict[str, torch.Tensor] = {}
        eraf_binding_loss = base_action.sum() * 0.0
        if self.policy_guard_version == 9:
            language_context_len = int(inputs["language_context_len"])
            target_labels = self._policy_guard_v9_labels(inputs)
            source_labels = self._policy_guard_v9_labels(
                inputs, prefix="source_"
            )
            target_policy_state = None
            source_policy_state = None
            if self.policy_guard_eraf_grounding_objective_version >= 14:
                target_policy_state = {
                    "phase_safe_memory_state_ids": target_labels[
                        "phase_safe_memory_previous_state_ids"
                    ],
                    "phase_safe_memory_valid": target_labels[
                        "phase_safe_memory_state_valid"
                    ],
                }
                source_policy_state = {
                    "phase_safe_memory_state_ids": source_labels[
                        "phase_safe_memory_previous_state_ids"
                    ],
                    "phase_safe_memory_valid": source_labels[
                        "phase_safe_memory_state_valid"
                    ],
                }
            (
                goal_queries,
                goal_embedding,
                eraf_outputs,
                goal_metrics,
            ) = self._encode_policy_guard_eraf(
                final_video_hidden=final_video_hidden,
                current_visual_hidden=neutral_visual_hidden,
                video_tokens_per_frame=video_tokens_per_frame,
                context=inputs["context"],
                context_mask=full_context_mask,
                language_context_len=language_context_len,
                policy_guard_state=target_policy_state,
                proprio=inputs.get("proprio_current"),
            )
            (
                source_goal_queries,
                source_goal_embedding,
                source_eraf_outputs,
                source_goal_metrics,
            ) = self._encode_policy_guard_eraf(
                final_video_hidden=final_video_hidden,
                current_visual_hidden=neutral_visual_hidden,
                video_tokens_per_frame=video_tokens_per_frame,
                context=source_context,
                context_mask=source_context_mask,
                language_context_len=language_context_len,
                policy_guard_state=source_policy_state,
                proprio=inputs.get("proprio_current"),
            )
            if (
                self.policy_guard_eraf_training_stage == "action"
                and self.policy_guard_eraf_grounding_aux_weight <= 0.0
            ):
                # V9.14 consumes the frozen ERAF representation but does not
                # optimize or even materialize the large privileged grounding
                # objective graph. Gradients still flow through the four
                # trainable ERAF-to-action bridge modules above.
                eraf_binding_loss = goal_embedding.sum() * 0.0
            else:
                target_eraf_loss, target_eraf_metrics = (
                    entity_relation_affordance_loss(
                        eraf_outputs,
                        target_labels,
                        weights=self.policy_guard_eraf_loss_weights,
                    )
                )
                source_eraf_loss, source_eraf_metrics = (
                    entity_relation_affordance_loss(
                        source_eraf_outputs,
                        source_labels,
                        weights=self.policy_guard_eraf_loss_weights,
                    )
                )
                eraf_binding_loss = target_eraf_loss + 0.5 * source_eraf_loss
                eraf_loss_metrics.update(target_eraf_metrics)
                for name, value in source_eraf_metrics.items():
                    eraf_loss_metrics[
                        f"pgc_v9_source_{name.removeprefix('pgc_v9_')}"
                    ] = value
            zero = goal_embedding.sum() * 0.0
            binding_interaction_loss = zero
            binding_prototype_loss = zero
            binding_source_loss = zero
            binding_hard_negative_loss = zero
            binding_separation_loss = zero
        elif self.policy_guard_version == 6:
            language_context_len = int(inputs["language_context_len"])
            (
                goal_queries,
                goal_embedding,
                target_attention,
                target_visual_features,
                goal_metrics,
            ) = self._encode_policy_guard_target_binding(
                current_visual_hidden=neutral_visual_hidden,
                video_tokens_per_frame=video_tokens_per_frame,
                language_hidden=inputs["context"][:, :language_context_len],
                language_mask=full_context_mask[:, :language_context_len],
            )
            (
                source_goal_queries,
                source_goal_embedding,
                source_attention,
                _,
                source_goal_metrics,
            ) = self._encode_policy_guard_target_binding(
                current_visual_hidden=neutral_visual_hidden,
                video_tokens_per_frame=video_tokens_per_frame,
                language_hidden=source_context[:, :language_context_len],
                language_mask=source_context_mask[:, :language_context_len],
            )
            teacher_latents = inputs.get("pgc_grounding_teacher_latents")
            if teacher_latents is None:
                raise ValueError(
                    "PGC v6 requires detached future latents for its "
                    "interaction-location teacher."
                )
            (
                interaction_teacher,
                interaction_valid,
                interaction_metrics,
            ) = interaction_patch_distribution(
                teacher_latents,
                tokens_per_frame=video_tokens_per_frame,
                topk_fraction=(
                    self.policy_guard_target_binding_teacher_topk
                ),
                temperature=(
                    self.policy_guard_target_binding_teacher_temperature
                ),
            )
            prototype_bank = self.policy_guard_target_prototype_bank
            if prototype_bank is None:
                raise RuntimeError("PGC v6 target prototype bank is unavailable.")
            prototype_bank.update(
                task_ids=inputs["pgc_goal_id"],
                visual_features=target_visual_features,
                teacher_attention=interaction_teacher,
                valid_mask=(direct_action_valid & interaction_valid),
            )
            (
                target_prototype_attention,
                target_prototype_valid,
                target_prototype_metrics,
            ) = prototype_bank.target_distribution(
                task_ids=inputs["pgc_goal_id"],
                visual_features=target_visual_features,
                valid_mask=direct_action_valid,
            )
            (
                source_prototype_attention,
                source_prototype_valid,
                source_prototype_metrics,
            ) = prototype_bank.target_distribution(
                task_ids=inputs["pgc_source_goal_id"],
                visual_features=target_visual_features,
                valid_mask=paired_language_valid,
            )
            (
                target_identity_loss,
                target_identity_metrics,
            ) = prototype_bank.classification_loss(
                task_ids=inputs["pgc_goal_id"],
                visual_features=target_visual_features,
                attention=target_attention,
                valid_mask=target_prototype_valid,
            )
            (
                source_identity_loss,
                source_identity_metrics,
            ) = prototype_bank.classification_loss(
                task_ids=inputs["pgc_source_goal_id"],
                visual_features=target_visual_features,
                attention=source_attention,
                valid_mask=source_prototype_valid,
            )
            (
                binding_interaction_loss,
                binding_prototype_loss,
                binding_source_loss,
                binding_hard_negative_loss,
                binding_separation_loss,
                binding_metrics,
            ) = self._compute_policy_guard_v6_target_binding_losses(
                target_attention=target_attention,
                source_attention=source_attention,
                interaction_teacher=interaction_teacher,
                interaction_valid=interaction_valid,
                target_prototype_attention=target_prototype_attention,
                target_prototype_valid=target_prototype_valid,
                source_prototype_attention=source_prototype_attention,
                source_prototype_valid=source_prototype_valid,
                direct_action_valid=direct_action_valid,
                paired_language_valid=paired_language_valid,
            )
            binding_prototype_loss = (
                binding_prototype_loss + target_identity_loss
            )
            binding_source_loss = binding_source_loss + source_identity_loss
            prototype_metrics.update(interaction_metrics)
            for name, value in target_prototype_metrics.items():
                prototype_metrics[f"pgc_v6_target_{name}"] = value
            for name, value in source_prototype_metrics.items():
                prototype_metrics[f"pgc_v6_source_{name}"] = value
            for name, value in target_identity_metrics.items():
                prototype_metrics[f"pgc_v6_target_{name}"] = value
            for name, value in source_identity_metrics.items():
                prototype_metrics[f"pgc_v6_source_{name}"] = value
            prototype_metrics["loss_pgc_v6_target_identity"] = (
                target_identity_loss.detach()
            )
            prototype_metrics["loss_pgc_v6_source_identity"] = (
                source_identity_loss.detach()
            )
            prototype_metrics["pgc_v6_target_prototype_retrieval_acc"] = (
                prototype_bank.retrieval_accuracy(
                    task_ids=inputs["pgc_goal_id"],
                    visual_features=target_visual_features,
                    attention=target_attention,
                    valid_mask=target_prototype_valid,
                ).detach()
            )
        elif self.policy_guard_version == 7:
            language_context_len = int(inputs["language_context_len"])
            (
                goal_queries,
                goal_embedding,
                target_attention,
                _,
                goal_metrics,
            ) = self._encode_policy_guard_target_binding(
                current_visual_hidden=neutral_visual_hidden,
                video_tokens_per_frame=video_tokens_per_frame,
                language_hidden=inputs["context"][:, :language_context_len],
                language_mask=full_context_mask[:, :language_context_len],
            )
            (
                source_goal_queries,
                source_goal_embedding,
                source_attention,
                _,
                source_goal_metrics,
            ) = self._encode_policy_guard_target_binding(
                current_visual_hidden=neutral_visual_hidden,
                video_tokens_per_frame=video_tokens_per_frame,
                language_hidden=source_context[:, :language_context_len],
                language_mask=source_context_mask[:, :language_context_len],
            )
            (
                _,
                _,
                aux_attention,
                _,
                aux_goal_metrics,
            ) = self._encode_policy_guard_target_binding(
                current_visual_hidden=neutral_visual_hidden,
                video_tokens_per_frame=video_tokens_per_frame,
                language_hidden=inputs["pgc_aux_context"][
                    :, :language_context_len
                ],
                language_mask=inputs["pgc_aux_context_mask"][
                    :, :language_context_len
                ],
            )
            (
                target_teacher,
                target_teacher_valid,
                target_mask_metrics,
            ) = spatial_mask_to_patch_distribution(
                inputs["pgc_target_object_mask"],
                token_count=int(target_attention.shape[-1]),
            )
            (
                source_teacher,
                source_teacher_valid,
                source_mask_metrics,
            ) = spatial_mask_to_patch_distribution(
                inputs["pgc_source_object_mask"],
                token_count=int(source_attention.shape[-1]),
            )
            (
                aux_teacher,
                aux_teacher_valid,
                aux_mask_metrics,
            ) = spatial_mask_to_patch_distribution(
                inputs["pgc_aux_object_mask"],
                token_count=int(aux_attention.shape[-1]),
            )
            target_mask_valid = (
                inputs["pgc_target_mask_valid"] & target_teacher_valid
            )
            source_mask_valid = (
                inputs["pgc_source_mask_valid"] & source_teacher_valid
            )
            aux_mask_valid = inputs["pgc_aux_mask_valid"] & aux_teacher_valid
            (
                binding_interaction_loss,
                binding_prototype_loss,
                binding_source_loss,
                binding_hard_negative_loss,
                binding_separation_loss,
                binding_metrics,
            ) = self._compute_policy_guard_v7_target_mask_losses(
                target_attention=target_attention,
                source_attention=source_attention,
                aux_attention=aux_attention,
                target_teacher=target_teacher,
                source_teacher=source_teacher,
                aux_teacher=aux_teacher,
                target_mask_valid=target_mask_valid,
                source_mask_valid=source_mask_valid,
                aux_mask_valid=aux_mask_valid,
                direct_action_valid=direct_action_valid,
                paired_language_valid=paired_language_valid,
            )
            for prefix, metrics in (
                ("target", target_mask_metrics),
                ("source", source_mask_metrics),
                ("aux", aux_mask_metrics),
            ):
                for name, value in metrics.items():
                    prototype_metrics[
                        f"pgc_v7_{prefix}_{name.removeprefix('pgc_v7_')}"
                    ] = value
            for name, value in aux_goal_metrics.items():
                prototype_metrics[
                    f"pgc_v7_aux_{name.removeprefix('pgc_v7_')}"
                ] = value
        else:
            goal_queries, goal_embedding, goal_metrics = (
                self._encode_policy_guard_goal(
                    final_video_hidden=final_video_hidden,
                    video_tokens_per_frame=video_tokens_per_frame,
                    context=inputs["context"],
                    context_mask=full_context_mask,
                )
            )
            source_goal_queries, source_goal_embedding, source_goal_metrics = (
                self._encode_policy_guard_goal(
                    # Deliberately reuse the identical frozen visual tokens.
                    # The only changed input is language.
                    final_video_hidden=final_video_hidden,
                    video_tokens_per_frame=video_tokens_per_frame,
                    context=source_context,
                    context_mask=source_context_mask,
                )
            )
            zero = goal_embedding.sum() * 0.0
            binding_interaction_loss = zero
            binding_prototype_loss = zero
            binding_source_loss = zero
            binding_hard_negative_loss = zero
            binding_separation_loss = zero
        proposal_module = self.policy_guard_modules["action_chunk_proposal"]
        proposal_action, residual, proposal_metrics = proposal_module(
            base_action=base_action,
            goal_queries=goal_queries,
            action_is_pad=action_is_pad,
        )
        source_proposal_action, source_residual, source_proposal_metrics = (
            proposal_module(
                base_action=base_action,
                goal_queries=source_goal_queries,
                action_is_pad=action_is_pad,
            )
        )
        pre_geometry_proposal_action = proposal_action
        pre_geometry_source_proposal_action = source_proposal_action
        geometry_residual = proposal_action * 0.0
        clause_alignment_teacher_action: Optional[torch.Tensor] = None
        clause_retention_teacher_action: Optional[torch.Tensor] = None
        if (
            self.policy_guard_version == 9
            and self.policy_guard_eraf_grounding_objective_version >= 17
        ):
            proprio_current = inputs.get("proprio_current")
            if proprio_current is None:
                raise ValueError(
                    "PGC V9.17 action training requires proprio_current."
                )
            geometry_adapter = self.policy_guard_modules[
                "eraf_geometry_action_adapter"
            ]
            (
                proposal_action,
                geometry_residual,
                geometry_metrics,
            ) = geometry_adapter(
                candidate_action=proposal_action,
                eraf_outputs=eraf_outputs,
                proprio=proprio_current,
                action_is_pad=action_is_pad,
            )
            (
                source_proposal_action,
                source_geometry_residual,
                source_geometry_metrics,
            ) = geometry_adapter(
                candidate_action=source_proposal_action,
                eraf_outputs=source_eraf_outputs,
                proprio=proprio_current,
                action_is_pad=action_is_pad,
            )
            if self.policy_guard_eraf_grounding_objective_version >= 19:
                phase_servo = self.policy_guard_modules[
                    "eraf_hard_routed_phase_servo"
                ]
                (
                    proposal_action,
                    geometry_residual,
                    phase_servo_metrics,
                ) = phase_servo(
                    candidate_action=pre_geometry_proposal_action,
                    legacy_residual=geometry_residual,
                    eraf_outputs=eraf_outputs,
                    proprio=proprio_current,
                    eef_position=inputs.get("eraf_eef_position_current"),
                    action_is_pad=action_is_pad,
                )
                (
                    source_proposal_action,
                    source_geometry_residual,
                    source_phase_servo_metrics,
                ) = phase_servo(
                    candidate_action=pre_geometry_source_proposal_action,
                    legacy_residual=source_geometry_residual,
                    eraf_outputs=source_eraf_outputs,
                    proprio=proprio_current,
                    eef_position=inputs.get("eraf_eef_position_current"),
                    action_is_pad=action_is_pad,
                )
                if self.policy_guard_eraf_grounding_objective_version >= 20:
                    waypoint_adapter = self.policy_guard_modules[
                        "eraf_phase_compatible_waypoint_adapter"
                    ]
                    (
                        proposal_action,
                        geometry_residual,
                        waypoint_metrics,
                    ) = waypoint_adapter(
                        candidate_action=pre_geometry_proposal_action,
                        legacy_residual=phase_servo_metrics[
                            "pgc_v919_retained_legacy_residual"
                        ],
                        inherited_servo_residual=phase_servo_metrics[
                            "pgc_v919_servo_residual"
                        ],
                        desired_direction=phase_servo_metrics[
                            "pgc_v919_desired_direction"
                        ],
                        control_phase=phase_servo_metrics[
                            "pgc_v919_selected_control_phase"
                        ],
                        route_confidence=phase_servo_metrics[
                            "pgc_v919_route_confidence_per_sample"
                        ],
                        action_is_pad=action_is_pad,
                    )
                    (
                        source_proposal_action,
                        source_geometry_residual,
                        source_waypoint_metrics,
                    ) = waypoint_adapter(
                        candidate_action=pre_geometry_source_proposal_action,
                        legacy_residual=source_phase_servo_metrics[
                            "pgc_v919_retained_legacy_residual"
                        ],
                        inherited_servo_residual=source_phase_servo_metrics[
                            "pgc_v919_servo_residual"
                        ],
                        desired_direction=source_phase_servo_metrics[
                            "pgc_v919_desired_direction"
                        ],
                        control_phase=source_phase_servo_metrics[
                            "pgc_v919_selected_control_phase"
                        ],
                        route_confidence=source_phase_servo_metrics[
                            "pgc_v919_route_confidence_per_sample"
                        ],
                        action_is_pad=action_is_pad,
                    )
                    phase_servo_metrics = dict(phase_servo_metrics)
                    phase_servo_metrics.update(waypoint_metrics)
                    source_phase_servo_metrics = dict(source_phase_servo_metrics)
                    source_phase_servo_metrics.update(source_waypoint_metrics)
                    if self.policy_guard_eraf_grounding_objective_version >= 21:
                        pre_expert_alignment_action = proposal_action
                        pre_expert_alignment_residual = geometry_residual
                        if self.policy_guard_eraf_grounding_objective_version == 23:
                            teacher_adapter = self.policy_guard_modules[
                                "eraf_phase_expert_residual_teacher"
                            ]
                            with torch.no_grad():
                                (
                                    clause_alignment_teacher_action,
                                    _,
                                    _,
                                ) = teacher_adapter(
                                    candidate_action=pre_geometry_proposal_action,
                                    current_residual=geometry_residual.detach(),
                                    desired_direction=phase_servo_metrics[
                                        "pgc_v919_desired_direction"
                                    ].detach(),
                                    desired_distance=phase_servo_metrics[
                                        "pgc_v919_desired_distance_per_sample"
                                    ].detach(),
                                    control_phase=phase_servo_metrics[
                                        "pgc_v919_selected_control_phase"
                                    ].detach(),
                                    route_confidence=phase_servo_metrics[
                                        "pgc_v919_route_confidence_per_sample"
                                    ].detach(),
                                    waypoint_compatibility=phase_servo_metrics[
                                        "pgc_v920_compatibility_probability_per_step"
                                    ].detach(),
                                    action_is_pad=action_is_pad,
                                )
                        expert_adapter = self.policy_guard_modules[
                            "eraf_phase_expert_residual_adapter"
                        ]
                        (
                            proposal_action,
                            geometry_residual,
                            expert_alignment_metrics,
                        ) = expert_adapter(
                            candidate_action=pre_geometry_proposal_action,
                            current_residual=geometry_residual,
                            desired_direction=phase_servo_metrics[
                                "pgc_v919_desired_direction"
                            ],
                            desired_distance=phase_servo_metrics[
                                "pgc_v919_desired_distance_per_sample"
                            ],
                            control_phase=phase_servo_metrics[
                                "pgc_v919_selected_control_phase"
                            ],
                            route_confidence=phase_servo_metrics[
                                "pgc_v919_route_confidence_per_sample"
                            ],
                            waypoint_compatibility=phase_servo_metrics[
                                "pgc_v920_compatibility_probability_per_step"
                            ],
                            action_is_pad=action_is_pad,
                        )
                        (
                            source_proposal_action,
                            source_geometry_residual,
                            source_expert_alignment_metrics,
                        ) = expert_adapter(
                            candidate_action=pre_geometry_source_proposal_action,
                            current_residual=source_geometry_residual,
                            desired_direction=source_phase_servo_metrics[
                                "pgc_v919_desired_direction"
                            ],
                            desired_distance=source_phase_servo_metrics[
                                "pgc_v919_desired_distance_per_sample"
                            ],
                            control_phase=source_phase_servo_metrics[
                                "pgc_v919_selected_control_phase"
                            ],
                            route_confidence=source_phase_servo_metrics[
                                "pgc_v919_route_confidence_per_sample"
                            ],
                            waypoint_compatibility=source_phase_servo_metrics[
                                "pgc_v920_compatibility_probability_per_step"
                            ],
                            action_is_pad=action_is_pad,
                        )
                        phase_servo_metrics.update(expert_alignment_metrics)
                        source_phase_servo_metrics.update(
                            source_expert_alignment_metrics
                        )
                        if self.policy_guard_eraf_grounding_objective_version >= 24:
                            clause_retention_teacher_action = proposal_action.detach()
                            retention_module = self.policy_guard_modules[
                                "eraf_clause_semantic_retention_residual"
                            ]
                            (
                                proposal_action,
                                _,
                                retention_metrics,
                            ) = retention_module(
                                base_action=base_action,
                                aligned_action=proposal_action,
                                goal_queries=goal_queries,
                                selected_clause=phase_servo_metrics[
                                    "pgc_v919_selected_clause"
                                ],
                                control_phase=phase_servo_metrics[
                                    "pgc_v919_selected_control_phase"
                                ],
                                route_confidence=phase_servo_metrics[
                                    "pgc_v919_route_confidence_per_sample"
                                ],
                                action_is_pad=action_is_pad,
                            )
                            (
                                source_proposal_action,
                                _,
                                source_retention_metrics,
                            ) = retention_module(
                                base_action=base_action,
                                aligned_action=source_proposal_action,
                                goal_queries=source_goal_queries,
                                selected_clause=source_phase_servo_metrics[
                                    "pgc_v919_selected_clause"
                                ],
                                control_phase=source_phase_servo_metrics[
                                    "pgc_v919_selected_control_phase"
                                ],
                                route_confidence=source_phase_servo_metrics[
                                    "pgc_v919_route_confidence_per_sample"
                                ],
                                action_is_pad=action_is_pad,
                            )
                            geometry_residual = (
                                proposal_action - pre_geometry_proposal_action
                            )
                            source_geometry_residual = (
                                source_proposal_action
                                - pre_geometry_source_proposal_action
                            )
                            phase_servo_metrics.update(retention_metrics)
                            source_phase_servo_metrics.update(
                                source_retention_metrics
                            )
                geometry_metrics = dict(geometry_metrics)
                geometry_metrics.update(phase_servo_metrics)
                source_geometry_metrics = dict(source_geometry_metrics)
                source_geometry_metrics.update(source_phase_servo_metrics)
            residual = residual + geometry_residual
            source_residual = source_residual + source_geometry_residual
            proposal_metrics = dict(proposal_metrics)
            proposal_metrics.update(geometry_metrics)
            for metric_name, metric_value in source_geometry_metrics.items():
                source_proposal_metrics[
                    f"source_{metric_name}"
                ] = metric_value
        v915_negative_actions: dict[str, torch.Tensor] = {}
        v924_negative_route_metrics: dict[str, Mapping[str, torch.Tensor]] = {}
        if (
            self.policy_guard_version == 9
            and self.policy_guard_eraf_training_stage == "action"
            and self.policy_guard_eraf_grounding_objective_version >= 15
        ):
            pre_action_queries = eraf_outputs.get(
                "pre_action_grounding_goal_queries"
            )
            if pre_action_queries is None:
                raise RuntimeError(
                    "V9.15 action training requires frozen V9.14 queries."
                )
            action_grounding_bridge = self.policy_guard_modules[
                "eraf_action_grounding_bridge"
            ]
            for negative_kind in (
                "subject",
                "reference",
                "anchor",
                "clause",
            ):
                negative_eraf_outputs = (
                    self._policy_guard_v915_negative_eraf_outputs(
                        target_outputs=eraf_outputs,
                        source_outputs=source_eraf_outputs,
                        kind=negative_kind,
                        target_labels=target_labels,
                        source_labels=source_labels,
                        reference_subject_fallback=(
                            self.policy_guard_eraf_grounding_objective_version
                            >= 16
                        ),
                        anchor_mirror_fallback=(
                            self.policy_guard_eraf_grounding_objective_version
                            >= 17
                        ),
                        preserve_clause_route=(
                            self.policy_guard_eraf_grounding_objective_version
                            >= 19
                        ),
                        propagate_reference_to_anchor=(
                            self.policy_guard_eraf_grounding_objective_version
                            >= 19
                        ),
                    )
                )
                negative_queries, _ = action_grounding_bridge(
                    goal_queries=pre_action_queries,
                    eraf_outputs=negative_eraf_outputs,
                )
                negative_action, _, _ = proposal_module(
                    base_action=base_action,
                    goal_queries=negative_queries,
                    action_is_pad=action_is_pad,
                )
                if self.policy_guard_eraf_grounding_objective_version >= 17:
                    pre_geometry_negative_action = negative_action
                    negative_action, negative_geometry_residual, _ = self.policy_guard_modules[
                        "eraf_geometry_action_adapter"
                    ](
                        candidate_action=negative_action,
                        eraf_outputs=negative_eraf_outputs,
                        proprio=inputs["proprio_current"],
                        action_is_pad=action_is_pad,
                    )
                    if self.policy_guard_eraf_grounding_objective_version >= 19:
                        negative_action, negative_servo_residual, negative_servo_metrics = self.policy_guard_modules[
                            "eraf_hard_routed_phase_servo"
                        ](
                            candidate_action=pre_geometry_negative_action,
                            legacy_residual=negative_geometry_residual,
                            eraf_outputs=negative_eraf_outputs,
                            proprio=inputs["proprio_current"],
                            eef_position=inputs.get("eraf_eef_position_current"),
                            action_is_pad=action_is_pad,
                        )
                        if self.policy_guard_eraf_grounding_objective_version >= 20:
                            (
                                negative_action,
                                negative_waypoint_residual,
                                negative_waypoint_metrics,
                            ) = self.policy_guard_modules[
                                "eraf_phase_compatible_waypoint_adapter"
                            ](
                                candidate_action=pre_geometry_negative_action,
                                legacy_residual=negative_servo_metrics[
                                    "pgc_v919_retained_legacy_residual"
                                ],
                                inherited_servo_residual=negative_servo_metrics[
                                    "pgc_v919_servo_residual"
                                ],
                                desired_direction=negative_servo_metrics[
                                    "pgc_v919_desired_direction"
                                ],
                                control_phase=negative_servo_metrics[
                                    "pgc_v919_selected_control_phase"
                                ],
                                route_confidence=negative_servo_metrics[
                                    "pgc_v919_route_confidence_per_sample"
                                ],
                                action_is_pad=action_is_pad,
                            )
                            if self.policy_guard_eraf_grounding_objective_version >= 21:
                                negative_action, _, _ = self.policy_guard_modules[
                                    "eraf_phase_expert_residual_adapter"
                                ](
                                    candidate_action=pre_geometry_negative_action,
                                    current_residual=negative_waypoint_residual,
                                    desired_direction=negative_servo_metrics[
                                        "pgc_v919_desired_direction"
                                    ],
                                    desired_distance=negative_servo_metrics[
                                        "pgc_v919_desired_distance_per_sample"
                                    ],
                                    control_phase=negative_servo_metrics[
                                        "pgc_v919_selected_control_phase"
                                    ],
                                    route_confidence=negative_servo_metrics[
                                        "pgc_v919_route_confidence_per_sample"
                                    ],
                                    waypoint_compatibility=negative_waypoint_metrics[
                                        "pgc_v920_compatibility_probability_per_step"
                                    ],
                                    action_is_pad=action_is_pad,
                                )
                                if (
                                    self.policy_guard_eraf_grounding_objective_version
                                    >= 24
                                ):
                                    (
                                        negative_action,
                                        _,
                                        negative_retention_metrics,
                                    ) = self.policy_guard_modules[
                                        "eraf_clause_semantic_retention_residual"
                                    ](
                                        base_action=base_action,
                                        aligned_action=negative_action,
                                        goal_queries=negative_queries,
                                        selected_clause=negative_servo_metrics[
                                            "pgc_v919_selected_clause"
                                        ],
                                        control_phase=negative_servo_metrics[
                                            "pgc_v919_selected_control_phase"
                                        ],
                                        route_confidence=negative_servo_metrics[
                                            "pgc_v919_route_confidence_per_sample"
                                        ],
                                        action_is_pad=action_is_pad,
                                    )
                                    v924_negative_route_metrics[
                                        negative_kind
                                    ] = negative_retention_metrics
                v915_negative_actions[negative_kind] = negative_action
        wrong_entity_candidate_action = None
        wrong_relation_candidate_action = None
        if (
            self.policy_guard_version == 9
            and self.policy_guard_eraf_training_stage == "verifier"
        ):
            wrong_entity_candidate_action, _, _ = proposal_module(
                base_action=base_action,
                goal_queries=eraf_outputs["wrong_entity_goal_queries"],
                action_is_pad=action_is_pad,
            )
            wrong_relation_candidate_action, _, _ = proposal_module(
                base_action=base_action,
                goal_queries=eraf_outputs["wrong_relation_goal_queries"],
                action_is_pad=action_is_pad,
            )
        (
            counterfactual_action_loss,
            native_zero_loss,
            same_state_source_zero_loss,
            goal_separation_loss,
            residual_separation_loss,
            residual_regularization_loss,
            residual_smoothness_loss,
            action_metrics,
        ) = self._compute_policy_guard_v5_action_losses(
            proposed_action=proposal_action,
            predicted_residual=residual,
            source_predicted_residual=source_residual,
            base_action=base_action,
            target_action=action,
            counterfactual_goal_embedding=goal_embedding,
            source_goal_embedding=source_goal_embedding,
            action_is_pad=action_is_pad,
            is_counterfactual=is_counterfactual,
            direct_action_valid=direct_action_valid,
            paired_language_valid=paired_language_valid,
            is_closed_loop_corrective=inputs.get(
                "pgc_is_closed_loop_corrective"
            ),
            completion_phase=inputs.get("pgc_completion_phase"),
            completion_phase_valid=inputs.get("pgc_completion_phase_valid"),
        )

        prefix = min(
            self.policy_guard_execution_prefix_steps,
            int(residual.shape[1]),
        )
        prefix_residual = residual[:, :prefix]
        prefix_pad = (
            None if action_is_pad is None else action_is_pad[:, :prefix]
        )
        proposal_cap = proposal_module.max_abs.to(
            device=residual.device, dtype=residual.dtype
        )
        delta_per_step = prefix_residual.float().square().mean(dim=-1)
        saturation_per_step = (
            prefix_residual.float().abs() >= proposal_cap.float() * 0.95
        ).float().mean(dim=-1)
        if prefix_pad is None:
            prefix_valid = torch.ones_like(delta_per_step)
        else:
            prefix_valid = (~prefix_pad).to(
                device=residual.device, dtype=torch.float32
            )
        prefix_valid_count = prefix_valid.sum(dim=1).clamp_min(1.0)
        proposal_delta_rms = (
            (delta_per_step * prefix_valid).sum(dim=1) / prefix_valid_count
        ).sqrt()
        proposal_saturation = (
            (saturation_per_step * prefix_valid).sum(dim=1)
            / prefix_valid_count
        )
        proposal_supported = (
            proposal_delta_rms <= self.policy_guard_candidate_max_delta_rms
        ) & (
            proposal_saturation
            <= self.policy_guard_candidate_max_saturation_fraction
        )
        action_metrics.update(
            {
                "pgc_v5_candidate_supported_rate": (
                    proposal_supported.float().mean().detach()
                ),
                "pgc_v5_prefix_candidate_delta_rms": (
                    proposal_delta_rms.mean().detach()
                ),
                "pgc_v5_prefix_candidate_delta_rms_max": (
                    proposal_delta_rms.max().detach()
                ),
                "pgc_v5_prefix_candidate_saturation_fraction_max": (
                    proposal_saturation.max().detach()
                ),
            }
        )

        action_training_scale = (
            self._policy_guard_target_binding_action_scale()
        )
        action_objective = (
            self.policy_guard_action_weight * counterfactual_action_loss
            + self.policy_guard_native_distillation_weight
            * (
                self.policy_guard_native_guard_weight
                if self.policy_guard_version in {8, 9}
                else 1.0
            )
            * native_zero_loss
            + self.policy_guard_same_state_source_zero_weight
            * same_state_source_zero_loss
            + self.policy_guard_goal_separation_weight * goal_separation_loss
            + self.policy_guard_residual_separation_weight
            * residual_separation_loss
            + self.policy_guard_residual_regularization_weight
            * residual_regularization_loss
            + self.policy_guard_residual_smoothness_weight
            * residual_smoothness_loss
        )
        v918_phase_residual_loss = action_objective.sum() * 0.0
        v918_phase_residual_metrics: dict[str, torch.Tensor] = {}
        if (
            self.policy_guard_version == 9
            and 18 <= self.policy_guard_eraf_grounding_objective_version < 20
        ):
            (
                v918_phase_residual_loss,
                v918_phase_residual_metrics,
            ) = self._compute_policy_guard_v918_phase_residual_loss(
                geometry_residual=geometry_residual,
                pre_geometry_action=pre_geometry_proposal_action,
                target_action=action,
                action_is_pad=action_is_pad,
                target_labels=target_labels,
                eraf_outputs=eraf_outputs,
                is_counterfactual=is_counterfactual,
                direct_action_valid=direct_action_valid,
                paired_language_valid=paired_language_valid,
            )
            action_objective = action_objective + v918_phase_residual_loss
        v919_frame_loss = action_objective.sum() * 0.0
        if (
            self.policy_guard_version == 9
            and self.policy_guard_eraf_grounding_objective_version == 19
        ):
            frame = self.policy_guard_modules[
                "eraf_hard_routed_phase_servo"
            ]
            workspace_to_action = frame.workspace_to_action.float()
            identity = torch.eye(
                3,
                device=workspace_to_action.device,
                dtype=workspace_to_action.dtype,
            )
            v919_frame_loss = (
                (
                    workspace_to_action.T @ workspace_to_action
                    - identity
                ).square().mean()
                + (torch.linalg.det(workspace_to_action) - 1.0).square()
                + (
                    frame.eef_scale.float()
                    - frame.initial_eef_scale.float()
                ).square().mean()
                + (
                    frame.eef_bias.float()
                    - frame.initial_eef_bias.float()
                ).square().mean()
            )
            action_objective = (
                action_objective
                + self.policy_guard_eraf_action_servo_frame_weight
                * v919_frame_loss
            )
        v920_waypoint_loss = action_objective.sum() * 0.0
        v920_waypoint_metrics: dict[str, torch.Tensor] = {}
        if (
            self.policy_guard_version == 9
            and self.policy_guard_eraf_grounding_objective_version == 20
        ):
            action_objective = action_objective.detach() * 0.0
            (
                v920_waypoint_loss,
                v920_waypoint_metrics,
            ) = self._compute_policy_guard_v920_waypoint_loss(
                geometry_residual=geometry_residual,
                pre_geometry_action=pre_geometry_proposal_action,
                target_action=action,
                action_is_pad=action_is_pad,
                waypoint_metrics=phase_servo_metrics,
                is_counterfactual=is_counterfactual,
                direct_action_valid=direct_action_valid,
                paired_language_valid=paired_language_valid,
            )
            action_objective = action_objective + v920_waypoint_loss
        v921_expert_alignment_loss = action_objective.sum() * 0.0
        v921_expert_alignment_metrics: dict[str, torch.Tensor] = {}
        if (
            self.policy_guard_version == 9
            and self.policy_guard_eraf_grounding_objective_version >= 21
        ):
            # Every older action path is frozen.  Optimize only the zero-init
            # expert residual adapter against learned and privileged geometry.
            action_objective = action_objective.detach() * 0.0
            (
                v921_expert_alignment_loss,
                v921_expert_alignment_metrics,
            ) = self._compute_policy_guard_v921_expert_alignment_loss(
                candidate_action=pre_geometry_proposal_action,
                pre_alignment_action=pre_expert_alignment_action,
                pre_alignment_residual=pre_expert_alignment_residual,
                deployed_metrics=expert_alignment_metrics,
                waypoint_metrics=phase_servo_metrics,
                target_labels=target_labels,
                target_action=action,
                action_is_pad=action_is_pad,
                is_counterfactual=is_counterfactual,
                direct_action_valid=direct_action_valid,
                paired_language_valid=paired_language_valid,
            )
            action_objective = (
                action_objective + v921_expert_alignment_loss
            )
        v915_causal_loss = action_objective.sum() * 0.0
        v915_causal_metrics: dict[str, torch.Tensor] = {}
        if (
            v915_negative_actions
            and (
                self.policy_guard_eraf_grounding_objective_version < 20
                or self.policy_guard_eraf_grounding_objective_version == 21
            )
        ):
            v915_causal_loss, v915_causal_metrics = (
                self._compute_policy_guard_v915_causal_action_loss(
                    correct_action=proposal_action,
                    negative_actions=v915_negative_actions,
                    target_action=action,
                    action_is_pad=action_is_pad,
                    target_labels=target_labels,
                    source_labels=source_labels,
                    is_counterfactual=is_counterfactual,
                    direct_action_valid=direct_action_valid,
                    paired_language_valid=paired_language_valid,
                )
            )
            action_objective = (
                action_objective
                + self.policy_guard_eraf_action_causal_ranking_weight
                * v915_causal_loss
            )
        v922_clause_ranking_loss = action_objective.sum() * 0.0
        v922_clause_ranking_metrics: dict[str, torch.Tensor] = {}
        if (
            self.policy_guard_version == 9
            and self.policy_guard_eraf_grounding_objective_version == 22
        ):
            wrong_clause_action = v915_negative_actions.get("clause")
            if wrong_clause_action is None:
                raise RuntimeError(
                    "PGC V9.22 requires the coherent final wrong-clause action."
                )
            (
                v922_clause_ranking_loss,
                v922_clause_ranking_metrics,
            ) = self._compute_policy_guard_v922_clause_action_ranking_loss(
                correct_action=proposal_action,
                wrong_clause_action=wrong_clause_action,
                target_action=action,
                action_is_pad=action_is_pad,
                target_labels=target_labels,
                waypoint_metrics=phase_servo_metrics,
                is_counterfactual=is_counterfactual,
                direct_action_valid=direct_action_valid,
                paired_language_valid=paired_language_valid,
            )
            action_objective = action_objective + v922_clause_ranking_loss
        v923_clause_ranking_loss = action_objective.sum() * 0.0
        v923_clause_ranking_metrics: dict[str, torch.Tensor] = {}
        if (
            self.policy_guard_version == 9
            and self.policy_guard_eraf_grounding_objective_version == 23
        ):
            wrong_clause_action = v915_negative_actions.get("clause")
            if wrong_clause_action is None:
                raise RuntimeError(
                    "PGC V9.23 requires the coherent final wrong-clause action."
                )
            if clause_alignment_teacher_action is None:
                raise RuntimeError(
                    "PGC V9.23 requires its frozen V9.21 teacher action."
                )
            (
                v923_clause_ranking_loss,
                v923_clause_ranking_metrics,
            ) = self._compute_policy_guard_v923_alignment_preserving_clause_loss(
                correct_action=proposal_action,
                wrong_clause_action=wrong_clause_action,
                teacher_action=clause_alignment_teacher_action,
                target_action=action,
                action_is_pad=action_is_pad,
                target_labels=target_labels,
                waypoint_metrics=phase_servo_metrics,
                is_counterfactual=is_counterfactual,
                direct_action_valid=direct_action_valid,
                paired_language_valid=paired_language_valid,
            )
            action_objective = action_objective + v923_clause_ranking_loss
        v924_clause_residual_loss = action_objective.sum() * 0.0
        v924_clause_residual_metrics: dict[str, torch.Tensor] = {}
        if (
            self.policy_guard_version == 9
            and self.policy_guard_eraf_grounding_objective_version >= 24
        ):
            wrong_clause_action = v915_negative_actions.get("clause")
            wrong_clause_metrics = v924_negative_route_metrics.get("clause")
            if wrong_clause_action is None or wrong_clause_metrics is None:
                raise RuntimeError(
                    "PGC V9.24 requires its gated coherent wrong-clause action."
                )
            if clause_retention_teacher_action is None:
                raise RuntimeError(
                    "PGC V9.24 requires the immutable V9.21 teacher action."
                )
            (
                v924_clause_residual_loss,
                v924_clause_residual_metrics,
            ) = self._compute_policy_guard_v924_isolated_clause_residual_loss(
                correct_action=proposal_action,
                wrong_clause_action=wrong_clause_action,
                teacher_action=clause_retention_teacher_action,
                base_action=base_action,
                target_action=action,
                correct_route_metrics=phase_servo_metrics,
                wrong_route_metrics=wrong_clause_metrics,
                action_is_pad=action_is_pad,
                target_labels=target_labels,
                waypoint_metrics=phase_servo_metrics,
                is_counterfactual=is_counterfactual,
                direct_action_valid=direct_action_valid,
                paired_language_valid=paired_language_valid,
            )
            action_objective = (
                action_objective.detach() * 0.0 + v924_clause_residual_loss
            )
        if self.policy_guard_version == 9:
            grounding_scale = {
                "grounding": 1.0,
                "action": self.policy_guard_eraf_grounding_aux_weight,
                "verifier": 0.0,
            }[self.policy_guard_eraf_training_stage]
            binding_objective = (
                eraf_binding_loss.detach() * 0.0
                if grounding_scale <= 0.0
                else grounding_scale * eraf_binding_loss
            )
        elif self.policy_guard_version == 7:
            binding_objective = (
                self.policy_guard_target_mask_weight
                * binding_interaction_loss
                + self.policy_guard_source_mask_weight
                * binding_prototype_loss
                + self.policy_guard_aux_mask_weight
                * binding_source_loss
                + self.policy_guard_mask_mass_weight
                * binding_hard_negative_loss
                + self.policy_guard_cross_object_weight
                * binding_separation_loss
            )
        else:
            binding_objective = (
                self.policy_guard_target_binding_interaction_weight
                * binding_interaction_loss
                + self.policy_guard_target_binding_prototype_weight
                * binding_prototype_loss
                + self.policy_guard_target_binding_source_weight
                * binding_source_loss
                + self.policy_guard_target_binding_hard_negative_weight
                * binding_hard_negative_loss
                + self.policy_guard_target_binding_separation_weight
                * binding_separation_loss
            )
        scaled_action_objective = (
            action_objective.detach() * 0.0
            if action_training_scale <= 0.0
            else action_training_scale * action_objective
        )
        proposal_objective = scaled_action_objective + binding_objective

        current_video_hidden = final_video_hidden[:, :video_tokens_per_frame]
        verifier_scale = self._policy_guard_verifier_scale()
        verifier_args = {
            "current_video_hidden": current_video_hidden,
            "goal_embedding": goal_embedding,
            "demonstrated_action": action,
            "base_candidate_action": base_action,
            "counterfactual_candidate_action": proposal_action,
            "source_candidate_action": source_proposal_action,
            "action_is_pad": action_is_pad,
            "is_counterfactual": is_counterfactual,
            "direct_action_valid": direct_action_valid,
            "paired_language_valid": paired_language_valid,
            "goal_ids": inputs["pgc_goal_id"],
            "wrong_entity_candidate_action": wrong_entity_candidate_action,
            "wrong_relation_candidate_action": wrong_relation_candidate_action,
        }
        if verifier_scale > 0.0:
            verifier_loss, alignment_loss, verifier_metrics = (
                self._compute_policy_guard_v5_verifier_loss(**verifier_args)
            )
        else:
            with torch.no_grad():
                verifier_loss, alignment_loss, verifier_metrics = (
                    self._compute_policy_guard_v5_verifier_loss(
                        **verifier_args
                    )
                )
        loss_total = (
            proposal_objective
            + verifier_scale * self.policy_guard_verifier_weight * verifier_loss
            + verifier_scale * self.policy_guard_alignment_weight * alignment_loss
        )

        loss_dict: dict[str, float] = {
            "loss_action": float(scaled_action_objective.detach().item()),
            "loss_pgc_binding_objective": float(
                binding_objective.detach().item()
            ),
            "loss_pgc_action": float(counterfactual_action_loss.detach().item()),
            "loss_pgc_native_residual_zero": float(native_zero_loss.detach().item()),
            "loss_pgc_v5_same_state_source_zero": float(
                same_state_source_zero_loss.detach().item()
            ),
            "loss_pgc_v5_goal_separation": float(
                goal_separation_loss.detach().item()
            ),
            "loss_pgc_v5_residual_separation": float(
                residual_separation_loss.detach().item()
            ),
            "loss_pgc_residual_regularization": float(
                residual_regularization_loss.detach().item()
            ),
            "loss_pgc_residual_smoothness": float(
                residual_smoothness_loss.detach().item()
            ),
            "loss_pgc_v915_action_causal_ranking": float(
                v915_causal_loss.detach().item()
            ),
            "loss_pgc_v918_phase_residual": float(
                v918_phase_residual_loss.detach().item()
            ),
            "loss_pgc_v919_servo_frame": float(
                v919_frame_loss.detach().item()
            ),
            "loss_pgc_v920_waypoint": float(
                v920_waypoint_loss.detach().item()
            ),
            "loss_pgc_v921_expert_alignment": float(
                v921_expert_alignment_loss.detach().item()
            ),
            "loss_pgc_v922_clause_action_ranking": float(
                v922_clause_ranking_loss.detach().item()
            ),
            "loss_pgc_v923_alignment_preserving_clause": float(
                v923_clause_ranking_loss.detach().item()
            ),
            "loss_pgc_v924_isolated_clause_residual": float(
                v924_clause_residual_loss.detach().item()
            ),
            "loss_pgc_verifier": float(verifier_loss.detach().item()),
            "loss_pgc_goal_action_alignment": float(
                alignment_loss.detach().item()
            ),
            "pgc_action_effective_weight": (
                action_training_scale * self.policy_guard_action_weight
            ),
            "pgc_native_distillation_effective_weight": (
                action_training_scale
                * self.policy_guard_native_distillation_weight
                * (
                    self.policy_guard_native_guard_weight
                    if self.policy_guard_version in {8, 9}
                    else 1.0
                )
            ),
            "pgc_v5_source_zero_effective_weight": (
                action_training_scale
                * self.policy_guard_same_state_source_zero_weight
            ),
            "pgc_v5_goal_separation_effective_weight": (
                action_training_scale
                * self.policy_guard_goal_separation_weight
            ),
            "pgc_v5_residual_separation_effective_weight": (
                action_training_scale
                * self.policy_guard_residual_separation_weight
            ),
            "pgc_residual_regularization_effective_weight": (
                action_training_scale
                * self.policy_guard_residual_regularization_weight
            ),
            "pgc_residual_smoothness_effective_weight": (
                action_training_scale
                * self.policy_guard_residual_smoothness_weight
            ),
            "pgc_v915_action_causal_effective_weight": (
                action_training_scale
                * self.policy_guard_eraf_action_causal_ranking_weight
                if self.policy_guard_version == 9
                and self.policy_guard_eraf_grounding_objective_version >= 15
                else 0.0
            ),
            "pgc_v918_phase_residual_effective_weight": (
                action_training_scale
                if self.policy_guard_version == 9
                and self.policy_guard_eraf_grounding_objective_version >= 18
                else 0.0
            ),
            "pgc_v919_servo_frame_effective_weight": (
                action_training_scale
                * self.policy_guard_eraf_action_servo_frame_weight
                if self.policy_guard_version == 9
                and self.policy_guard_eraf_grounding_objective_version == 19
                else 0.0
            ),
            "pgc_v920_waypoint_effective_weight": (
                action_training_scale
                if self.policy_guard_version == 9
                and self.policy_guard_eraf_grounding_objective_version == 20
                else 0.0
            ),
            "pgc_v921_expert_alignment_effective_weight": (
                action_training_scale
                if self.policy_guard_version == 9
                and 21
                <= self.policy_guard_eraf_grounding_objective_version
                < 24
                else 0.0
            ),
            "pgc_v922_clause_ranking_effective_weight": (
                action_training_scale
                * self.policy_guard_eraf_action_clause_ranking_weight
                if self.policy_guard_version == 9
                and self.policy_guard_eraf_grounding_objective_version == 22
                else 0.0
            ),
            "pgc_v923_alignment_preserving_clause_effective_weight": (
                action_training_scale
                if self.policy_guard_version == 9
                and self.policy_guard_eraf_grounding_objective_version == 23
                else 0.0
            ),
            "pgc_v924_isolated_clause_residual_effective_weight": (
                action_training_scale
                if self.policy_guard_version == 9
                and self.policy_guard_eraf_grounding_objective_version >= 24
                else 0.0
            ),
            "pgc_v6_target_interaction_effective_weight": (
                self.policy_guard_target_binding_interaction_weight
                if self.policy_guard_version == 6
                else 0.0
            ),
            "pgc_v6_target_prototype_effective_weight": (
                self.policy_guard_target_binding_prototype_weight
                if self.policy_guard_version == 6
                else 0.0
            ),
            "pgc_v6_source_prototype_effective_weight": (
                self.policy_guard_target_binding_source_weight
                if self.policy_guard_version == 6
                else 0.0
            ),
            "pgc_v6_hard_negative_effective_weight": (
                self.policy_guard_target_binding_hard_negative_weight
                if self.policy_guard_version == 6
                else 0.0
            ),
            "pgc_v6_attention_separation_effective_weight": (
                self.policy_guard_target_binding_separation_weight
                if self.policy_guard_version == 6
                else 0.0
            ),
            "pgc_verifier_effective_weight": (
                verifier_scale * self.policy_guard_verifier_weight
            ),
            "pgc_alignment_effective_weight": (
                verifier_scale * self.policy_guard_alignment_weight
            ),
            "pgc_verifier_training_scale": verifier_scale,
            "pgc_target_binding_action_training_scale": action_training_scale,
            "pgc_v5_execution_prefix_steps": float(prefix),
            "pgc_v5_suffix_loss_weight": self.policy_guard_suffix_loss_weight,
            "pgc_v5_completion_enabled": float(
                self.policy_guard_completion_phase_enabled
            ),
            "pgc_v5_completion_transport_weight": (
                self.policy_guard_completion_transport_weight
                if self.policy_guard_completion_phase_enabled
                else 1.0
            ),
            "pgc_v5_completion_release_weight": (
                self.policy_guard_completion_release_weight
                if self.policy_guard_completion_phase_enabled
                else 1.0
            ),
            "pgc_v5_completion_proposal_only": float(
                self.policy_guard_completion_phase_enabled
                and self.policy_guard_completion_train_proposal_only
            ),
            "pgc_v5_rollout_num_inference_steps": float(
                self.policy_guard_rollout_num_inference_steps
            ),
            "pgc_v8_enabled": float(self.policy_guard_version == 8),
            "pgc_v8_closed_loop_corrective_weight": (
                self.policy_guard_closed_loop_corrective_weight
                if self.policy_guard_version == 8
                else 0.0
            ),
            "pgc_v8_offline_acquisition_weight": (
                self.policy_guard_offline_acquisition_weight
                if self.policy_guard_version == 8
                else 0.0
            ),
            "pgc_v8_native_guard_multiplier": (
                self.policy_guard_native_guard_weight
                if self.policy_guard_version == 8
                else 1.0
            ),
            "pgc_v8_proposal_only": float(
                self.policy_guard_version == 8
                and self.policy_guard_closed_loop_train_proposal_only
            ),
            "pgc_base_policy_frozen": 1.0,
            "pgc_video_loss_optimization_weight": 0.0,
        }
        if self.policy_guard_version == 9:
            loss_dict.update(
                {
                    "loss_pgc_v9_eraf": float(
                        eraf_binding_loss.detach().item()
                    ),
                    "pgc_v9_grounding_effective_weight": float(
                        {
                            "grounding": 1.0,
                            "action": self.policy_guard_eraf_grounding_aux_weight,
                            "verifier": 0.0,
                        }[self.policy_guard_eraf_training_stage]
                    ),
                    "pgc_v9_stage_grounding": float(
                        self.policy_guard_eraf_training_stage == "grounding"
                    ),
                    "pgc_v9_stage_action": float(
                        self.policy_guard_eraf_training_stage == "action"
                    ),
                    "pgc_v9_stage_verifier": float(
                        self.policy_guard_eraf_training_stage == "verifier"
                    ),
                    "pgc_v914_completion_only_memory": float(
                        self.policy_guard_eraf_completion_only_memory
                    ),
                    "pgc_v914_eraf_proposal_joint": float(
                        self.policy_guard_eraf_action_joint_training
                    ),
                    "pgc_v914_frozen_grounding_core": float(
                        self.policy_guard_eraf_action_joint_training
                    ),
                }
            )
        if self.policy_guard_version == 6:
            loss_dict.update(
                {
                    "loss_pgc_v6_binding_objective": float(
                        binding_objective.detach().item()
                    ),
                    "loss_pgc_v6_target_interaction": float(
                        binding_interaction_loss.detach().item()
                    ),
                    "loss_pgc_v6_target_prototype": float(
                        binding_prototype_loss.detach().item()
                    ),
                    "loss_pgc_v6_source_prototype": float(
                        binding_source_loss.detach().item()
                    ),
                    "loss_pgc_v6_same_state_hard_negative": float(
                        binding_hard_negative_loss.detach().item()
                    ),
                    "loss_pgc_v6_attention_separation": float(
                        binding_separation_loss.detach().item()
                    ),
                    "pgc_v6_action_training_scale": action_training_scale,
                }
            )
        elif self.policy_guard_version == 7:
            loss_dict.update(
                {
                    "loss_pgc_v7_mask_binding_objective": float(
                        binding_objective.detach().item()
                    ),
                    "loss_pgc_v7_target_mask": float(
                        binding_interaction_loss.detach().item()
                    ),
                    "loss_pgc_v7_source_mask": float(
                        binding_prototype_loss.detach().item()
                    ),
                    "loss_pgc_v7_aux_mask": float(
                        binding_source_loss.detach().item()
                    ),
                    "loss_pgc_v7_mask_mass": float(
                        binding_hard_negative_loss.detach().item()
                    ),
                    "loss_pgc_v7_cross_object": float(
                        binding_separation_loss.detach().item()
                    ),
                    "pgc_v7_target_mask_effective_weight": (
                        self.policy_guard_target_mask_weight
                    ),
                    "pgc_v7_source_mask_effective_weight": (
                        self.policy_guard_source_mask_weight
                    ),
                    "pgc_v7_aux_mask_effective_weight": (
                        self.policy_guard_aux_mask_weight
                    ),
                    "pgc_v7_mask_mass_effective_weight": (
                        self.policy_guard_mask_mass_weight
                    ),
                    "pgc_v7_cross_object_effective_weight": (
                        self.policy_guard_cross_object_weight
                    ),
                    "pgc_v7_action_training_scale": action_training_scale,
                }
            )
        loss_dict.update(detached_policy_guard_metrics(goal_metrics))
        loss_dict.update(detached_policy_guard_metrics(proposal_metrics))
        loss_dict.update(detached_policy_guard_metrics(action_metrics))
        loss_dict.update(detached_policy_guard_metrics(verifier_metrics))
        loss_dict.update(detached_policy_guard_metrics(binding_metrics))
        loss_dict.update(detached_policy_guard_metrics(prototype_metrics))
        loss_dict.update(detached_policy_guard_metrics(eraf_loss_metrics))
        loss_dict.update(detached_policy_guard_metrics(v915_causal_metrics))
        loss_dict.update(
            detached_policy_guard_metrics(v918_phase_residual_metrics)
        )
        loss_dict.update(detached_policy_guard_metrics(v920_waypoint_metrics))
        loss_dict.update(
            detached_policy_guard_metrics(v921_expert_alignment_metrics)
        )
        loss_dict.update(
            detached_policy_guard_metrics(v922_clause_ranking_metrics)
        )
        loss_dict.update(
            detached_policy_guard_metrics(v923_clause_ranking_metrics)
        )
        loss_dict.update(
            detached_policy_guard_metrics(v924_clause_residual_metrics)
        )
        for name, value in detached_policy_guard_metrics(
            source_goal_metrics
        ).items():
            loss_dict[f"pgc_v5_source_{name.removeprefix('pgc_')}"] = value
        for name, value in detached_policy_guard_metrics(
            source_proposal_metrics
        ).items():
            loss_dict[f"pgc_v5_source_{name.removeprefix('pgc_v4_')}"] = value
        return loss_total, loss_dict

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
        pgc_is_closed_loop_corrective = sample.get(
            "pgc_is_closed_loop_corrective"
        )
        pgc_direct_action_valid = sample.get("pgc_direct_action_valid")
        pgc_goal_id = sample.get("pgc_goal_id")
        pgc_source_context = sample.get("pgc_source_context")
        pgc_source_context_mask = sample.get("pgc_source_context_mask")
        pgc_source_goal_id = sample.get("pgc_source_goal_id")
        pgc_paired_language_valid = sample.get("pgc_paired_language_valid")
        pgc_completion_phase = sample.get("pgc_completion_phase")
        pgc_completion_phase_valid = sample.get("pgc_completion_phase_valid")
        pgc_target_object_mask = sample.get("pgc_target_object_mask")
        pgc_source_object_mask = sample.get("pgc_source_object_mask")
        pgc_aux_object_mask = sample.get("pgc_aux_object_mask")
        pgc_target_mask_valid = sample.get("pgc_target_mask_valid")
        pgc_source_mask_valid = sample.get("pgc_source_mask_valid")
        pgc_aux_mask_valid = sample.get("pgc_aux_mask_valid")
        pgc_aux_context = sample.get("pgc_aux_context")
        pgc_aux_context_mask = sample.get("pgc_aux_context_mask")
        pgc_aux_goal_id = sample.get("pgc_aux_goal_id")
        pgc_eraf_labels = {
            f"pgc_eraf_{prefix}{name}": sample.get(
                f"pgc_eraf_{prefix}{name}"
            )
            for prefix in ("", "source_")
            for name in (
                *PGC_ENTITY_RELATION_ARRAY_NAMES,
                *(
                    PGC_PHASE_SAFE_MEMORY_LABEL_NAMES
                    if self.policy_guard_enabled
                    and self.policy_guard_version == 9
                    and self.policy_guard_eraf_grounding_objective_version >= 14
                    else ()
                ),
            )
        }
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
            if self.policy_guard_version >= 5:
                missing_v5 = [
                    name
                    for name, value in (
                        ("pgc_source_context", pgc_source_context),
                        ("pgc_source_context_mask", pgc_source_context_mask),
                        ("pgc_source_goal_id", pgc_source_goal_id),
                        (
                            "pgc_paired_language_valid",
                            pgc_paired_language_valid,
                        ),
                    )
                    if value is None
                ]
                if missing_v5:
                    raise ValueError(
                        "PGC v5+ requires same-state paired-language fields: "
                        f"{missing_v5}. Recreate the dataset loader after "
                        "updating LF-FastWAM."
                    )
            if (
                self.policy_guard_version == 8
                and pgc_is_closed_loop_corrective is None
            ):
                raise ValueError(
                    "PGC v8 requires `pgc_is_closed_loop_corrective` from "
                    "the audited mixed dataset."
                )
            if self.policy_guard_completion_phase_enabled:
                missing_completion = [
                    name
                    for name, value in (
                        ("pgc_completion_phase", pgc_completion_phase),
                        (
                            "pgc_completion_phase_valid",
                            pgc_completion_phase_valid,
                        ),
                    )
                    if value is None
                ]
                if missing_completion:
                    raise ValueError(
                        "PGC V5-completion requires audited phase fields: "
                        f"{missing_completion}. Build the completion sidecar "
                        "and enable its dataset contract."
                    )
            if self.policy_guard_version == 7:
                missing_v7 = [
                    name
                    for name, value in (
                        ("pgc_target_object_mask", pgc_target_object_mask),
                        ("pgc_source_object_mask", pgc_source_object_mask),
                        ("pgc_aux_object_mask", pgc_aux_object_mask),
                        ("pgc_target_mask_valid", pgc_target_mask_valid),
                        ("pgc_source_mask_valid", pgc_source_mask_valid),
                        ("pgc_aux_mask_valid", pgc_aux_mask_valid),
                        ("pgc_aux_context", pgc_aux_context),
                        ("pgc_aux_context_mask", pgc_aux_context_mask),
                        ("pgc_aux_goal_id", pgc_aux_goal_id),
                    )
                    if value is None
                ]
                if missing_v7:
                    raise ValueError(
                        "PGC v7 requires explicit current-state object masks "
                        f"and auxiliary language fields: {missing_v7}. Build "
                        "the audited mask sidecar before training."
                    )
            if self.policy_guard_version == 9:
                missing_v9 = [
                    name
                    for name, value in pgc_eraf_labels.items()
                    if value is None
                ]
                if missing_v9:
                    raise ValueError(
                        "PGC v9 requires the audited entity-relation sidecar: "
                        f"{missing_v9}."
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
        # Legacy PGC v4-v25 is deployed from a single observed frame and has no
        # trainable world-model objective. Encode only that frame so its frozen
        # Base rollout cannot receive future demonstration frames through the
        # VAE temporal path. V9.26 is different by contract: its Video Expert
        # LoRA is explicitly supervised with future-video flow, while the
        # Action Expert mask below still exposes only current-frame Video K/V.
        pgc_grounding_teacher_latents = None
        legacy_single_frame_policy_guard = bool(
            self.policy_guard_enabled
            and self.policy_guard_version >= 4
            and not (
                self.policy_guard_version == 9
                and self.policy_guard_eraf_grounding_objective_version >= 26
            )
        )
        if legacy_single_frame_policy_guard:
            input_latents = torch.cat(
                [
                    self._encode_input_image_latents_tensor(
                        input_video[index, :, 0], tiled=tiled
                    )
                    for index in range(batch_size)
                ],
                dim=0,
            )
            if self.policy_guard_version == 6:
                # Future frames are used only to construct a detached
                # interaction-location teacher. They never enter the frozen
                # Base rollout or the deployed Proposal path.
                with torch.no_grad():
                    pgc_grounding_teacher_latents = self._encode_video_latents(
                        input_video, tiled=tiled
                    ).detach()
        else:
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
        if self.policy_guard_version >= 5:
            if (
                pgc_source_context.ndim != 3
                or pgc_source_context_mask.ndim != 2
            ):
                raise ValueError(
                    "PGC v5 source context/mask must be [B,L,D]/[B,L]."
                )
            if pgc_source_context.shape != context.shape or (
                pgc_source_context_mask.shape != context_mask.shape
            ):
                raise ValueError(
                    "PGC v5 current/source text-cache tensor shapes must match."
                )
        if self.policy_guard_version == 7:
            if pgc_aux_context.ndim != 3 or pgc_aux_context_mask.ndim != 2:
                raise ValueError(
                    "PGC v7 auxiliary context/mask must be [B,L,D]/[B,L]."
                )
            if pgc_aux_context.shape != context.shape or (
                pgc_aux_context_mask.shape != context_mask.shape
            ):
                raise ValueError(
                    "PGC v7 target/source/auxiliary text-cache shapes must match."
                )
            mask_tensors = (
                pgc_target_object_mask,
                pgc_source_object_mask,
                pgc_aux_object_mask,
            )
            if any(mask.ndim != 3 for mask in mask_tensors):
                raise ValueError("PGC v7 object masks must be [B,H,W].")
            if any(mask.shape != mask_tensors[0].shape for mask in mask_tensors):
                raise ValueError("PGC v7 object-mask tensor shapes must match.")
            if int(mask_tensors[0].shape[0]) != batch_size:
                raise ValueError("PGC v7 object-mask batch size mismatch.")
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
        if pgc_source_context is not None:
            pgc_source_context = pgc_source_context.to(
                device=self.device,
                dtype=self.torch_dtype,
                non_blocking=True,
            )
            pgc_source_context_mask = pgc_source_context_mask.to(
                device=self.device,
                dtype=torch.bool,
                non_blocking=True,
            )
        if pgc_aux_context is not None:
            pgc_aux_context = pgc_aux_context.to(
                device=self.device,
                dtype=self.torch_dtype,
                non_blocking=True,
            )
            pgc_aux_context_mask = pgc_aux_context_mask.to(
                device=self.device,
                dtype=torch.bool,
                non_blocking=True,
            )
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
        if pgc_is_closed_loop_corrective is not None:
            pgc_is_closed_loop_corrective = torch.as_tensor(
                pgc_is_closed_loop_corrective,
                device=self.device,
                dtype=torch.bool,
            )
            if pgc_is_closed_loop_corrective.ndim == 0:
                pgc_is_closed_loop_corrective = (
                    pgc_is_closed_loop_corrective.expand(batch_size)
                )
            if pgc_is_closed_loop_corrective.shape != (batch_size,):
                raise ValueError(
                    "`pgc_is_closed_loop_corrective` must be [B]."
                )
            if pgc_is_counterfactual is not None and bool(
                (pgc_is_closed_loop_corrective & ~pgc_is_counterfactual).any()
            ):
                raise ValueError(
                    "PGC V8 corrective rows must also be counterfactual rows."
                )
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
        if pgc_source_goal_id is not None:
            pgc_source_goal_id = torch.as_tensor(
                pgc_source_goal_id, device=self.device, dtype=torch.long
            )
            if pgc_source_goal_id.ndim == 0:
                pgc_source_goal_id = pgc_source_goal_id.expand(batch_size)
            if pgc_source_goal_id.shape != (batch_size,):
                raise ValueError("`pgc_source_goal_id` must be [B].")
        if pgc_paired_language_valid is not None:
            pgc_paired_language_valid = torch.as_tensor(
                pgc_paired_language_valid,
                device=self.device,
                dtype=torch.bool,
            )
            if pgc_paired_language_valid.ndim == 0:
                pgc_paired_language_valid = pgc_paired_language_valid.expand(
                    batch_size
                )
            if pgc_paired_language_valid.shape != (batch_size,):
                raise ValueError("`pgc_paired_language_valid` must be [B].")
        if pgc_completion_phase is not None:
            pgc_completion_phase = torch.as_tensor(
                pgc_completion_phase,
                device=self.device,
                dtype=torch.long,
            )
            pgc_completion_phase_valid = torch.as_tensor(
                pgc_completion_phase_valid,
                device=self.device,
                dtype=torch.bool,
            )
            if pgc_completion_phase.ndim == 0:
                pgc_completion_phase = pgc_completion_phase.expand(batch_size)
            if pgc_completion_phase_valid.ndim == 0:
                pgc_completion_phase_valid = pgc_completion_phase_valid.expand(
                    batch_size
                )
            if pgc_completion_phase.shape != (batch_size,) or (
                pgc_completion_phase_valid.shape != (batch_size,)
            ):
                raise ValueError(
                    "PGC completion phase/value mask must share [B] shape."
                )
            if bool(((pgc_completion_phase < 0) | (pgc_completion_phase > 2)).any()):
                raise ValueError("PGC completion phase must be 0, 1, or 2.")
        if self.policy_guard_version == 7:
            pgc_target_object_mask = pgc_target_object_mask.to(
                device=self.device, dtype=torch.float32, non_blocking=True
            )
            pgc_source_object_mask = pgc_source_object_mask.to(
                device=self.device, dtype=torch.float32, non_blocking=True
            )
            pgc_aux_object_mask = pgc_aux_object_mask.to(
                device=self.device, dtype=torch.float32, non_blocking=True
            )
            for name, value in (
                ("pgc_target_mask_valid", pgc_target_mask_valid),
                ("pgc_source_mask_valid", pgc_source_mask_valid),
                ("pgc_aux_mask_valid", pgc_aux_mask_valid),
            ):
                converted = torch.as_tensor(
                    value, device=self.device, dtype=torch.bool
                )
                if converted.ndim == 0:
                    converted = converted.expand(batch_size)
                if converted.shape != (batch_size,):
                    raise ValueError(f"`{name}` must be [B].")
                if name == "pgc_target_mask_valid":
                    pgc_target_mask_valid = converted
                elif name == "pgc_source_mask_valid":
                    pgc_source_mask_valid = converted
                else:
                    pgc_aux_mask_valid = converted
            pgc_aux_goal_id = torch.as_tensor(
                pgc_aux_goal_id, device=self.device, dtype=torch.long
            )
            if pgc_aux_goal_id.ndim == 0:
                pgc_aux_goal_id = pgc_aux_goal_id.expand(batch_size)
            if pgc_aux_goal_id.shape != (batch_size,):
                raise ValueError("`pgc_aux_goal_id` must be [B].")
        if self.policy_guard_version == 9:
            bool_suffixes = (
                "clause_valid",
                "subject_mask_valid",
                "reference_mask_valid",
                "subject_view_visible",
                "reference_view_visible",
                "subject_position_valid",
                "reference_position_valid",
                "grasp_anchor_valid",
                "goal_anchor_valid",
                "interaction_anchor_valid",
                "predicate_truth_valid",
                "phase_valid",
            )
            long_suffixes = (
                "predicate_ids",
                "phase_ids",
                "subject_entity_ids",
                "reference_entity_ids",
            )
            for name, value in tuple(pgc_eraf_labels.items()):
                if name.endswith(PGC_PHASE_SAFE_MEMORY_SAMPLE_LABEL_NAMES):
                    if name.endswith(
                        (
                            "phase_safe_memory_execution_valid",
                            "phase_safe_memory_stage_valid",
                        )
                    ):
                        dtype = torch.bool
                    else:
                        dtype = torch.long
                    converted = torch.as_tensor(
                        value,
                        device=self.device,
                        dtype=dtype,
                    )
                    if converted.ndim == 0:
                        converted = converted.expand(batch_size)
                    if converted.shape != (batch_size,):
                        raise ValueError(f"`{name}` must be [B].")
                    pgc_eraf_labels[name] = converted
                    continue
                if name.endswith(bool_suffixes):
                    dtype = torch.bool
                elif name.endswith("phase_safe_memory_state_valid"):
                    dtype = torch.bool
                elif name.endswith(long_suffixes):
                    dtype = torch.long
                elif name.endswith(PGC_PHASE_SAFE_MEMORY_CLAUSE_LABEL_NAMES):
                    dtype = torch.long
                else:
                    dtype = torch.float32
                converted = torch.as_tensor(
                    value,
                    device=self.device,
                    dtype=dtype,
                )
                if converted.ndim < 2 or converted.shape[:2] != (
                    batch_size,
                    self.policy_guard_eraf_max_clauses,
                ):
                    raise ValueError(
                        f"`{name}` must start with [B,max_clauses]="
                        f"[{batch_size},{self.policy_guard_eraf_max_clauses}], "
                        f"got {tuple(converted.shape)}."
                    )
                pgc_eraf_labels[name] = converted
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
            if pgc_source_context is not None:
                pgc_source_context, pgc_source_context_mask = (
                    self._append_proprio_to_context(
                        context=pgc_source_context,
                        context_mask=pgc_source_context_mask,
                        proprio=proprio_current,
                    )
                )
            if pgc_aux_context is not None:
                pgc_aux_context, pgc_aux_context_mask = (
                    self._append_proprio_to_context(
                        context=pgc_aux_context,
                        context_mask=pgc_aux_context_mask,
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
            "pgc_is_closed_loop_corrective": (
                pgc_is_closed_loop_corrective
            ),
            "pgc_direct_action_valid": pgc_direct_action_valid,
            "pgc_goal_id": pgc_goal_id,
            "pgc_source_context": pgc_source_context,
            "pgc_source_context_mask": pgc_source_context_mask,
            "pgc_source_goal_id": pgc_source_goal_id,
            "pgc_paired_language_valid": pgc_paired_language_valid,
            "pgc_completion_phase": pgc_completion_phase,
            "pgc_completion_phase_valid": pgc_completion_phase_valid,
            "pgc_target_object_mask": pgc_target_object_mask,
            "pgc_source_object_mask": pgc_source_object_mask,
            "pgc_aux_object_mask": pgc_aux_object_mask,
            "pgc_target_mask_valid": pgc_target_mask_valid,
            "pgc_source_mask_valid": pgc_source_mask_valid,
            "pgc_aux_mask_valid": pgc_aux_mask_valid,
            "pgc_aux_context": pgc_aux_context,
            "pgc_aux_context_mask": pgc_aux_context_mask,
            "pgc_aux_goal_id": pgc_aux_goal_id,
            "language_context_len": language_context_len,
            "has_proprio": has_proprio,
            "proprio_current": proprio_current,
            "input_latents": input_latents,
            "pgc_grounding_teacher_latents": (
                pgc_grounding_teacher_latents
            ),
            "first_frame_latents": first_frame_latents,
            "fuse_vae_embedding_in_latents": fuse_flag,
            "action": action,
            "action_is_pad": action_is_pad,
            "image_is_pad": image_is_pad,
            **pgc_eraf_labels,
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

        if (
            self.policy_guard_enabled
            and self.policy_guard_version == 9
            and self.policy_guard_eraf_grounding_objective_version >= 26
        ):
            return self._training_loss_policy_guard_v926_eraf_expert_lora(
                inputs=inputs,
                full_context_mask=full_context_mask,
                state_only_context_mask=state_only_context_mask,
            )
        if (
            self.policy_guard_enabled
            and self.policy_guard_version == 9
            and self.policy_guard_eraf_grounding_objective_version >= 25
        ):
            return self._training_loss_policy_guard_v925_action_context(
                inputs=inputs,
                full_context_mask=full_context_mask,
                state_only_context_mask=state_only_context_mask,
            )
        if self.policy_guard_enabled and self.policy_guard_version >= 5:
            return self._training_loss_policy_guard_v5(
                inputs=inputs,
                full_context_mask=full_context_mask,
                state_only_context_mask=state_only_context_mask,
            )
        if self.policy_guard_enabled and self.policy_guard_version == 4:
            return self._training_loss_policy_guard_v4(
                inputs=inputs,
                full_context_mask=full_context_mask,
                state_only_context_mask=state_only_context_mask,
            )

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
        policy_guard_velocity_residual = None
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
            if self.policy_guard_version >= 3:
                (
                    pred_action_post,
                    policy_guard_velocity_residual,
                    policy_guard_residual_metrics,
                ) = self._apply_policy_guard_v3_velocity_residual(
                    base_action_hidden=base_tokens_out["action"],
                    base_action_velocity=pred_action_base,
                    routed_goal_queries=routed_goal_queries,
                )
            else:
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
            verifier_scale = self._policy_guard_verifier_scale()
            if verifier_scale > 0.0:
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
            else:
                # Keep diagnostics visible during the action-only phase while
                # ensuring Verifier/alignment parameters receive no gradients.
                with torch.no_grad():
                    verifier_loss, alignment_loss, verifier_metrics = (
                        self._compute_policy_guard_verifier_loss(
                            current_video_hidden=(
                                policy_guard_current_video_hidden
                            ),
                            goal_embedding=policy_guard_goal_embedding,
                            demonstrated_action=action,
                            base_candidate_action=base_clean_action,
                            counterfactual_candidate_action=(
                                counterfactual_clean_action
                            ),
                            action_is_pad=action_is_pad,
                            is_counterfactual=(
                                inputs["pgc_is_counterfactual"]
                            ),
                            direct_action_valid=(
                                inputs["pgc_direct_action_valid"]
                            ),
                            goal_ids=inputs["pgc_goal_id"],
                        )
                    )
            policy_action_metrics: dict[str, torch.Tensor] = {}
            residual_regularization_loss = loss_action_post.detach() * 0.0
            residual_smoothness_loss = loss_action_post.detach() * 0.0
            if self.policy_guard_version >= 3:
                if policy_guard_velocity_residual is None:
                    raise RuntimeError(
                        "PGC v3 training did not produce a velocity residual."
                    )
                (
                    counterfactual_action_loss,
                    native_distillation_loss,
                    residual_regularization_loss,
                    residual_smoothness_loss,
                    policy_action_metrics,
                ) = self._compute_policy_guard_v3_action_losses(
                    predicted_residual=policy_guard_velocity_residual,
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
                    + self.policy_guard_residual_regularization_weight
                    * residual_regularization_loss
                    + self.policy_guard_residual_smoothness_weight
                    * residual_smoothness_loss
                )
            elif self.policy_guard_version >= 2:
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
                + verifier_scale
                * self.policy_guard_verifier_weight
                * verifier_loss
                + verifier_scale
                * self.policy_guard_alignment_weight
                * alignment_loss
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
                    "loss_pgc_native_residual_zero": float(
                        native_distillation_loss.detach().item()
                    ) if self.policy_guard_version >= 3 else 0.0,
                    "loss_pgc_residual_regularization": float(
                        residual_regularization_loss.detach().item()
                    ),
                    "loss_pgc_residual_smoothness": float(
                        residual_smoothness_loss.detach().item()
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
                        verifier_scale * self.policy_guard_verifier_weight
                    ),
                    "pgc_alignment_effective_weight": float(
                        verifier_scale * self.policy_guard_alignment_weight
                    ),
                    "pgc_residual_regularization_effective_weight": float(
                        self.policy_guard_residual_regularization_weight
                    ),
                    "pgc_residual_smoothness_effective_weight": float(
                        self.policy_guard_residual_smoothness_weight
                    ),
                    "pgc_verifier_training_scale": float(verifier_scale),
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
        policy_guard_state: Optional[Mapping[str, torch.Tensor]] = None,
        policy_guard_eraf_oracle: Optional[Mapping[str, torch.Tensor]] = None,
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
                policy_guard_state=policy_guard_state,
                policy_guard_eraf_oracle=policy_guard_eraf_oracle,
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
                "policy_guard_score_space",
                "policy_guard_candidate_supported",
                "policy_guard_candidate_delta_rms",
                "policy_guard_candidate_saturation_fraction",
                "policy_guard_gate_mode",
                "policy_guard_base_action",
                "policy_guard_counterfactual_action",
                "policy_guard_eraf_diagnostics",
                "policy_guard_state",
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
        policy_guard_state: Optional[Mapping[str, torch.Tensor]] = None,
        policy_guard_eraf_oracle: Optional[Mapping[str, torch.Tensor]] = None,
        policy_guard_eraf_audit_variants: Optional[
            Mapping[str, Optional[Mapping[str, Any]]]
        ] = None,
    ) -> dict[str, Any]:
        self.eval()
        if policy_guard_eraf_oracle is not None and not (
            self.policy_guard_enabled and self.policy_guard_version == 9
        ):
            raise ValueError("Oracle ERAF input requires PGC v9.")
        if policy_guard_eraf_audit_variants is not None and not (
            self.policy_guard_enabled and self.policy_guard_version == 9
        ):
            raise ValueError("ERAF causal variants require PGC v9.")
        if (
            policy_guard_eraf_audit_variants is not None
            and policy_guard_eraf_oracle is not None
        ):
            raise ValueError(
                "Ordinary Oracle ERAF routing and causal variants are mutually "
                "exclusive."
            )
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
        policy_guard_initial_action_noise = latents_action.clone()
        eraf_only_action_path = bool(
            self.policy_guard_enabled
            and self.policy_guard_version == 9
            and self.policy_guard_eraf_grounding_objective_version >= 26
        )
        use_internal_eraf_action_context = bool(
            self.policy_guard_enabled
            and self.policy_guard_version == 9
            and self.policy_guard_eraf_grounding_objective_version >= 25
        )
        policy_guard_latents_action = (
            latents_action.clone()
            if self.policy_guard_enabled
            and not eraf_only_action_path
            and (
                self.policy_guard_version <= 3
                or use_internal_eraf_action_context
            )
            else None
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
        policy_guard_goal_metrics: dict[str, torch.Tensor] = {}
        policy_guard_eraf_audit_goals: dict[
            str,
            tuple[
                torch.Tensor,
                torch.Tensor,
                Optional[dict[str, torch.Tensor]],
                Optional[dict[str, torch.Tensor]],
            ],
        ] = {}
        if self.policy_guard_enabled and self.policy_guard_version == 9:
            self._policy_guard_last_eraf_diagnostics = None
            self._policy_guard_last_eraf_outputs = None
        if self.policy_guard_enabled:
            if final_video_hidden is None:
                raise RuntimeError("PGC inference requires final Video hidden tokens.")
            (
                policy_guard_goal_queries,
                policy_guard_goal_embedding,
                policy_guard_goal_metrics,
            ) = self._encode_policy_guard_goal(
                final_video_hidden=final_video_hidden,
                video_tokens_per_frame=int(
                    video_pre["meta"]["tokens_per_frame"]
                ),
                context=context,
                context_mask=context_mask,
                language_context_len=language_context_len,
                current_visual_hidden=video_pre["tokens"],
                policy_guard_state=policy_guard_state,
                policy_guard_eraf_oracle=policy_guard_eraf_oracle,
                proprio=proprio,
            )
            policy_guard_current_video_hidden = final_video_hidden[:, : int(
                video_pre["meta"]["tokens_per_frame"]
            )].detach()
            if policy_guard_eraf_audit_variants is not None:
                learned_diagnostics = self._policy_guard_last_eraf_diagnostics
                learned_eraf_outputs = self._policy_guard_last_eraf_outputs
                for raw_name, oracle_variant in (
                    policy_guard_eraf_audit_variants.items()
                ):
                    name = str(raw_name).strip()
                    if not name:
                        raise ValueError("ERAF causal variant names must be non-empty.")
                    if name in policy_guard_eraf_audit_goals:
                        raise ValueError(f"Duplicate ERAF causal variant {name!r}.")
                    if oracle_variant is None:
                        variant_queries = policy_guard_goal_queries
                        variant_embedding = policy_guard_goal_embedding
                        variant_diagnostics = learned_diagnostics
                        variant_eraf_outputs = learned_eraf_outputs
                    else:
                        (
                            variant_queries,
                            variant_embedding,
                            _,
                        ) = self._encode_policy_guard_goal(
                            final_video_hidden=final_video_hidden,
                            video_tokens_per_frame=int(
                                video_pre["meta"]["tokens_per_frame"]
                            ),
                            context=context,
                            context_mask=context_mask,
                            language_context_len=language_context_len,
                            current_visual_hidden=video_pre["tokens"],
                            policy_guard_state=policy_guard_state,
                            policy_guard_eraf_oracle=oracle_variant,
                            proprio=proprio,
                        )
                        variant_diagnostics = (
                            self._policy_guard_last_eraf_diagnostics
                        )
                        variant_eraf_outputs = self._policy_guard_last_eraf_outputs
                    policy_guard_eraf_audit_goals[name] = (
                        variant_queries,
                        variant_embedding,
                        variant_diagnostics,
                        variant_eraf_outputs,
                    )
                # The deployed result and recurrent completion state must remain
                # those of the learned ERAF path, independent of audit order.
                self._policy_guard_last_eraf_diagnostics = learned_diagnostics
                self._policy_guard_last_eraf_outputs = learned_eraf_outputs

        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        eraf_only_injection_metrics: Mapping[str, torch.Tensor] = {}
        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            if eraf_only_action_path:
                if policy_guard_goal_queries is None:
                    raise RuntimeError(
                        "ERAF-only inference did not initialize goal queries."
                    )
                pred_action_posi, eraf_only_injection_metrics = (
                    self._forward_policy_guard_action_from_cache(
                        action_tokens=latents_action,
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
                        return_metrics=True,
                    )
                )
            else:
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

            if self.policy_guard_enabled and not eraf_only_action_path and (
                self.policy_guard_version <= 3
                or use_internal_eraf_action_context
            ):
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

        if eraf_only_action_path:
            result: dict[str, Any] = {
                "action": latents_action[0].detach().to(
                    device="cpu", dtype=torch.float32
                ),
                "policy_guard_selected_counterfactual": True,
                "policy_guard_gate_mode": "eraf_only",
                "policy_guard_candidate_supported": True,
                "policy_guard_eraf_single_path": True,
            }
            for metric_name, metric_value in eraf_only_injection_metrics.items():
                if metric_value.numel() == 1:
                    result[metric_name] = float(
                        metric_value.detach().float().item()
                    )

            if policy_guard_eraf_audit_goals:
                diagnostic_base = policy_guard_initial_action_noise.clone()
                for step_t_action, step_delta_action in zip(
                    infer_timesteps_action, infer_deltas_action
                ):
                    timestep_action = step_t_action.unsqueeze(0).to(
                        dtype=diagnostic_base.dtype, device=self.device
                    )
                    diagnostic_noise = self._predict_action_noise_with_cache(
                        latents_action=diagnostic_base,
                        timestep_action=timestep_action,
                        context=context,
                        context_mask=context_mask,
                        state_only_context_mask=state_only_context_mask,
                        video_kv_cache=video_kv_cache,
                        attention_mask=attention_mask,
                        video_seq_len=video_seq_len,
                        routed_transition_tokens=routed_transition_tokens,
                    )
                    diagnostic_base = self.infer_action_scheduler.step(
                        diagnostic_noise,
                        step_delta_action,
                        diagnostic_base,
                    )
                result["policy_guard_base_action"] = (
                    diagnostic_base[0]
                    .detach()
                    .to(device="cpu", dtype=torch.float32)
                )
                result["policy_guard_counterfactual_action"] = (
                    latents_action[0]
                    .detach()
                    .to(device="cpu", dtype=torch.float32)
                )
                causal_variants: dict[str, dict[str, Any]] = {}
                learned_queries = policy_guard_goal_queries
                learned_embedding = policy_guard_goal_embedding
                if learned_queries is None or learned_embedding is None:
                    raise RuntimeError(
                        "ERAF-only causal audit did not retain learned routing."
                    )
                for name, (
                    variant_queries,
                    variant_embedding,
                    _,
                    variant_eraf_outputs,
                ) in policy_guard_eraf_audit_goals.items():
                    bypass = bool(
                        variant_eraf_outputs is not None
                        and torch.as_tensor(
                            variant_eraf_outputs.get(
                                "audit_bypass_bridge", False
                            )
                        ).all()
                    )
                    if (
                        name == "learned"
                        and variant_queries is policy_guard_goal_queries
                    ):
                        variant_action = latents_action
                        variant_metrics = eraf_only_injection_metrics
                    elif bypass:
                        variant_action = diagnostic_base
                        variant_metrics = {}
                    else:
                        variant_action, variant_metrics = (
                            self._rollout_policy_guard_eraf_context_action(
                                initial_action_noise=(
                                    policy_guard_initial_action_noise
                                ),
                                context=context,
                                full_context_mask=context_mask,
                                state_only_context_mask=(
                                    state_only_context_mask
                                ),
                                video_kv_cache=video_kv_cache,
                                video_seq_len=video_seq_len,
                                video_tokens_per_frame=int(
                                    video_pre["meta"]["tokens_per_frame"]
                                ),
                                routed_goal_queries=variant_queries,
                                num_inference_steps=num_inference_steps,
                                sigma_shift=sigma_shift,
                            )
                        )
                    variant_residual = variant_action - diagnostic_base
                    causal_variants[name] = {
                        "goal_queries": variant_queries.detach().to(
                            device="cpu", dtype=torch.float32
                        ),
                        "action": variant_action[0].detach().to(
                            device="cpu", dtype=torch.float32
                        ),
                        "residual": variant_residual[0].detach().to(
                            device="cpu", dtype=torch.float32
                        ),
                        "residual_rms": float(
                            variant_residual.float().square().mean().sqrt().item()
                        ),
                        "goal_query_rms": float(
                            variant_queries.float().square().mean().sqrt().item()
                        ),
                        "goal_query_delta_from_learned_rms": float(
                            (variant_queries - learned_queries)
                            .float()
                            .square()
                            .mean()
                            .sqrt()
                            .item()
                        ),
                        "goal_embedding_delta_from_learned_rms": float(
                            (variant_embedding - learned_embedding)
                            .float()
                            .square()
                            .mean()
                            .sqrt()
                            .item()
                        ),
                        "proposal_attention_entropy": None,
                        "v919_selected_clause": None,
                        "v919_selected_control_phase": None,
                    }
                result["policy_guard_eraf_causal_variants"] = causal_variants

            if self._policy_guard_last_eraf_diagnostics is None:
                raise RuntimeError(
                    "ERAF-only inference did not retain ERAF diagnostics."
                )
            result["policy_guard_eraf_diagnostics"] = (
                self._policy_guard_last_eraf_diagnostics
            )
            next_policy_state = {
                "phase_safe_memory_state_ids": (
                    self._policy_guard_last_eraf_diagnostics[
                        "phase_safe_memory_next_state_ids"
                    ].clone()
                ),
                "phase_safe_memory_valid": (
                    self._policy_guard_last_eraf_diagnostics[
                        "phase_safe_memory_next_state_valid"
                    ].clone()
                ),
            }
            result["policy_guard_state"] = (
                self._policy_guard_completion_only_state(
                    next_policy_state,
                    previous_state=policy_guard_state,
                )
            )
            return result

        if self.policy_guard_version >= 4:
            if (
                policy_guard_goal_queries is None
                or policy_guard_goal_embedding is None
                or policy_guard_current_video_hidden is None
            ):
                raise RuntimeError("PGC v4 inference did not encode its goal state.")
            if use_internal_eraf_action_context:
                if policy_guard_latents_action is None:
                    raise RuntimeError(
                        "Internal ERAF action candidate was not denoised."
                    )
                action_residual = policy_guard_latents_action - latents_action
                _, _, policy_guard_proposal_metrics = (
                    self.policy_guard_modules["eraf_action_context_injector"](
                        context=context,
                        context_mask=context_mask,
                        goal_queries=policy_guard_goal_queries,
                    )
                )
            else:
                (
                    policy_guard_latents_action,
                    action_residual,
                    policy_guard_proposal_metrics,
                ) = self.policy_guard_modules["action_chunk_proposal"](
                    base_action=latents_action,
                    goal_queries=policy_guard_goal_queries,
                    action_is_pad=None,
                )
            pre_geometry_policy_guard_action = policy_guard_latents_action
            if (
                not use_internal_eraf_action_context
                and
                self.policy_guard_version == 9
                and self.policy_guard_eraf_grounding_objective_version >= 17
            ):
                if self._policy_guard_last_eraf_outputs is None or proprio is None:
                    raise RuntimeError(
                        "PGC V9.17 inference requires live ERAF outputs and proprio."
                    )
                (
                    policy_guard_latents_action,
                    geometry_residual,
                    geometry_metrics,
                ) = self.policy_guard_modules[
                    "eraf_geometry_action_adapter"
                ](
                    candidate_action=policy_guard_latents_action,
                    eraf_outputs=self._policy_guard_last_eraf_outputs,
                    proprio=proprio,
                    action_is_pad=None,
                )
                if self.policy_guard_eraf_grounding_objective_version >= 19:
                    (
                        policy_guard_latents_action,
                        geometry_residual,
                        phase_servo_metrics,
                    ) = self.policy_guard_modules[
                        "eraf_hard_routed_phase_servo"
                    ](
                        candidate_action=pre_geometry_policy_guard_action,
                        legacy_residual=geometry_residual,
                        eraf_outputs=self._policy_guard_last_eraf_outputs,
                        proprio=proprio,
                        action_is_pad=None,
                    )
                    if self.policy_guard_eraf_grounding_objective_version >= 20:
                        (
                            policy_guard_latents_action,
                            geometry_residual,
                            waypoint_metrics,
                        ) = self.policy_guard_modules[
                            "eraf_phase_compatible_waypoint_adapter"
                        ](
                            candidate_action=pre_geometry_policy_guard_action,
                            legacy_residual=phase_servo_metrics[
                                "pgc_v919_retained_legacy_residual"
                            ],
                            inherited_servo_residual=phase_servo_metrics[
                                "pgc_v919_servo_residual"
                            ],
                            desired_direction=phase_servo_metrics[
                                "pgc_v919_desired_direction"
                            ],
                            control_phase=phase_servo_metrics[
                                "pgc_v919_selected_control_phase"
                            ],
                            route_confidence=phase_servo_metrics[
                                "pgc_v919_route_confidence_per_sample"
                            ],
                            action_is_pad=None,
                        )
                        phase_servo_metrics = dict(phase_servo_metrics)
                        phase_servo_metrics.update(waypoint_metrics)
                        if self.policy_guard_eraf_grounding_objective_version >= 21:
                            (
                                policy_guard_latents_action,
                                geometry_residual,
                                expert_alignment_metrics,
                            ) = self.policy_guard_modules[
                                "eraf_phase_expert_residual_adapter"
                            ](
                                candidate_action=pre_geometry_policy_guard_action,
                                current_residual=geometry_residual,
                                desired_direction=phase_servo_metrics[
                                    "pgc_v919_desired_direction"
                                ],
                                desired_distance=phase_servo_metrics[
                                    "pgc_v919_desired_distance_per_sample"
                                ],
                                control_phase=phase_servo_metrics[
                                    "pgc_v919_selected_control_phase"
                                ],
                                route_confidence=phase_servo_metrics[
                                    "pgc_v919_route_confidence_per_sample"
                                ],
                                waypoint_compatibility=waypoint_metrics[
                                    "pgc_v920_compatibility_probability_per_step"
                                ],
                                action_is_pad=None,
                            )
                            phase_servo_metrics.update(expert_alignment_metrics)
                            if (
                                self.policy_guard_eraf_grounding_objective_version
                                >= 24
                            ):
                                (
                                    policy_guard_latents_action,
                                    _,
                                    clause_retention_metrics,
                                ) = self.policy_guard_modules[
                                    "eraf_clause_semantic_retention_residual"
                                ](
                                    base_action=latents_action,
                                    aligned_action=policy_guard_latents_action,
                                    goal_queries=policy_guard_goal_queries,
                                    selected_clause=phase_servo_metrics[
                                        "pgc_v919_selected_clause"
                                    ],
                                    control_phase=phase_servo_metrics[
                                        "pgc_v919_selected_control_phase"
                                    ],
                                    route_confidence=phase_servo_metrics[
                                        "pgc_v919_route_confidence_per_sample"
                                    ],
                                    action_is_pad=None,
                                )
                                geometry_residual = (
                                    policy_guard_latents_action
                                    - pre_geometry_policy_guard_action
                                )
                                phase_servo_metrics.update(
                                    clause_retention_metrics
                                )
                    geometry_metrics = dict(geometry_metrics)
                    geometry_metrics.update(phase_servo_metrics)
                action_residual = action_residual + geometry_residual
                policy_guard_proposal_metrics = dict(
                    policy_guard_proposal_metrics
                )
                policy_guard_proposal_metrics.update(geometry_metrics)
            verifier = self.policy_guard_modules["verifier"]
            verifier_horizon = (
                min(
                    self.policy_guard_execution_prefix_steps,
                    int(latents_action.shape[1]),
                )
                if self.policy_guard_version >= 5
                else int(latents_action.shape[1])
            )
            (
                advantage,
                base_score,
                counterfactual_score,
                _,
                _,
                _,
            ) = verifier(
                current_video_hidden=policy_guard_current_video_hidden,
                goal_embedding=policy_guard_goal_embedding,
                base_action=latents_action[:, :verifier_horizon],
                counterfactual_action=(
                    policy_guard_latents_action[:, :verifier_horizon]
                ),
                action_is_pad=None,
            )
            support_residual = action_residual[:, :verifier_horizon]
            residual_rms = support_residual.float().square().mean(
                dim=(1, 2)
            ).sqrt()
            if use_internal_eraf_action_context:
                # This is a fully denoised second candidate, not a clipped
                # action residual. Retain delta RMS as a conservative guard,
                # while reporting zero residual saturation by construction.
                saturation = torch.zeros_like(residual_rms)
                candidate_supported = (
                    residual_rms <= self.policy_guard_candidate_max_delta_rms
                )
            else:
                cap = self.policy_guard_modules[
                    "action_chunk_proposal"
                ].max_abs.to(
                    device=action_residual.device,
                    dtype=action_residual.dtype,
                )
                saturation = (
                    support_residual.float().abs() >= cap.float() * 0.95
                ).float().mean(dim=(1, 2))
                candidate_supported = (
                    residual_rms <= self.policy_guard_candidate_max_delta_rms
                ) & (
                    saturation
                    <= self.policy_guard_candidate_max_saturation_fraction
                )
            selected_action, selected_counterfactual = (
                self._select_policy_guard_action(
                    base_action=latents_action,
                    counterfactual_action=policy_guard_latents_action,
                    base_score=base_score,
                    counterfactual_score=counterfactual_score,
                    candidate_supported=candidate_supported,
                )
            )
            result = {
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
                "policy_guard_score_margin": float(advantage[0].item()),
                "policy_guard_score_space": "fp32_raw_advantage",
                "policy_guard_verifier_horizon": verifier_horizon,
                "policy_guard_candidate_supported": bool(
                    candidate_supported[0].item()
                ),
                "policy_guard_candidate_delta_rms": float(
                    residual_rms[0].item()
                ),
                "policy_guard_candidate_saturation_fraction": float(
                    saturation[0].item()
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
            for metric_name in (
                "pgc_v917_geometry_action_residual_rms",
                "pgc_v917_geometry_action_residual_max_abs",
                "pgc_v917_geometry_route_confidence",
                "pgc_v917_geometry_eef_position_norm",
                "pgc_v917_geometry_goal_relative_norm",
                "pgc_v919_hard_clause_route_max",
                "pgc_v919_selected_clause_mean",
                "pgc_v919_selected_phase_mean",
                "pgc_v919_route_confidence",
                "pgc_v919_canonical_eef_from_state",
                "pgc_v919_eef_scale_min",
                "pgc_v919_eef_bias_max_abs",
                "pgc_v919_desired_distance",
                "pgc_v919_translation_gain",
                "pgc_v919_servo_residual_rms",
                "pgc_v919_total_residual_rms",
                "pgc_v919_legacy_suppression",
                "pgc_v919_workspace_to_action_orthogonality_error",
                "pgc_v919_workspace_to_action_determinant",
                "pgc_v920_compatibility_probability",
                "pgc_v920_inherited_servo_retention",
                "pgc_v920_waypoint_tangent_rms",
                "pgc_v920_translation_gain",
                "pgc_v920_servo_residual_rms",
                "pgc_v920_total_residual_rms",
                "pgc_v921_expert_correction_rms",
                "pgc_v921_total_residual_rms",
                "pgc_v924_clause_suppression_mean",
                "pgc_v924_clause_retention_mean",
                "pgc_v924_clause_semantic_residual_rms",
                "pgc_v925_action_context_token_rms",
                "pgc_v925_action_context_scale",
                "pgc_v925_action_context_token_count",
                "pgc_v925_post_action_residual_enabled",
            ):
                metric_value = policy_guard_proposal_metrics.get(metric_name)
                if metric_value is not None:
                    result[metric_name] = float(
                        metric_value.detach().float().item()
                    )
            if policy_guard_eraf_audit_goals:
                learned_queries = policy_guard_goal_queries
                learned_embedding = policy_guard_goal_embedding
                causal_variants: dict[str, dict[str, Any]] = {}
                for name, (
                    variant_queries,
                    variant_embedding,
                    _,
                    variant_eraf_outputs,
                ) in policy_guard_eraf_audit_goals.items():
                    if (
                        name == "learned"
                        and variant_queries is policy_guard_goal_queries
                    ):
                        variant_action = policy_guard_latents_action
                        variant_residual = action_residual
                        variant_metrics: Mapping[str, torch.Tensor] = (
                            policy_guard_proposal_metrics
                        )
                    elif use_internal_eraf_action_context:
                        bypass_internal_injection = bool(
                            variant_eraf_outputs is not None
                            and torch.as_tensor(
                                variant_eraf_outputs.get(
                                    "audit_bypass_bridge", False
                                )
                            ).all()
                        )
                        if bypass_internal_injection:
                            # The bypass counterfactual is the exact Base
                            # sampler result from the same initial noise. It
                            # must not traverse any legacy Proposal/residual
                            # module.
                            variant_action = latents_action
                            variant_residual = torch.zeros_like(latents_action)
                            variant_metrics = {}
                        else:
                            variant_action, variant_metrics = (
                                self._rollout_policy_guard_eraf_context_action(
                                    initial_action_noise=(
                                        policy_guard_initial_action_noise
                                    ),
                                    context=context,
                                    full_context_mask=context_mask,
                                    state_only_context_mask=(
                                        state_only_context_mask
                                    ),
                                    video_kv_cache=video_kv_cache,
                                    video_seq_len=video_seq_len,
                                    video_tokens_per_frame=int(
                                        video_pre["meta"]["tokens_per_frame"]
                                    ),
                                    routed_goal_queries=variant_queries,
                                    num_inference_steps=num_inference_steps,
                                    sigma_shift=sigma_shift,
                                )
                            )
                            variant_residual = variant_action - latents_action
                    else:
                        (
                            variant_action,
                            variant_residual,
                            variant_metrics,
                        ) = self.policy_guard_modules["action_chunk_proposal"](
                            base_action=latents_action,
                            goal_queries=variant_queries,
                            action_is_pad=None,
                        )
                        pre_geometry_variant_action = variant_action
                        if (
                            self.policy_guard_version == 9
                            and self.policy_guard_eraf_grounding_objective_version
                            >= 17
                        ):
                            if variant_eraf_outputs is None or proprio is None:
                                raise RuntimeError(
                                    "PGC V9.17 causal variants require ERAF outputs "
                                    "and proprio."
                                )
                            (
                                variant_action,
                                variant_geometry_residual,
                                variant_geometry_metrics,
                            ) = self.policy_guard_modules[
                                "eraf_geometry_action_adapter"
                            ](
                                candidate_action=variant_action,
                                eraf_outputs=variant_eraf_outputs,
                                proprio=proprio,
                                action_is_pad=None,
                            )
                            if (
                                self.policy_guard_eraf_grounding_objective_version
                                >= 19
                            ):
                                (
                                    variant_action,
                                    variant_geometry_residual,
                                    variant_phase_servo_metrics,
                                ) = self.policy_guard_modules[
                                    "eraf_hard_routed_phase_servo"
                                ](
                                    candidate_action=pre_geometry_variant_action,
                                    legacy_residual=variant_geometry_residual,
                                    eraf_outputs=variant_eraf_outputs,
                                    proprio=proprio,
                                    action_is_pad=None,
                                )
                                variant_geometry_metrics = dict(
                                    variant_geometry_metrics
                                )
                                variant_geometry_metrics.update(
                                    variant_phase_servo_metrics
                                )
                                if (
                                    self.policy_guard_eraf_grounding_objective_version
                                    >= 20
                                ):
                                    (
                                        variant_action,
                                        variant_geometry_residual,
                                        variant_waypoint_metrics,
                                    ) = self.policy_guard_modules[
                                        "eraf_phase_compatible_waypoint_adapter"
                                    ](
                                        candidate_action=pre_geometry_variant_action,
                                        legacy_residual=variant_phase_servo_metrics[
                                            "pgc_v919_retained_legacy_residual"
                                        ],
                                        inherited_servo_residual=variant_phase_servo_metrics[
                                            "pgc_v919_servo_residual"
                                        ],
                                        desired_direction=variant_phase_servo_metrics[
                                            "pgc_v919_desired_direction"
                                        ],
                                        control_phase=variant_phase_servo_metrics[
                                            "pgc_v919_selected_control_phase"
                                        ],
                                        route_confidence=variant_phase_servo_metrics[
                                            "pgc_v919_route_confidence_per_sample"
                                        ],
                                        action_is_pad=None,
                                    )
                                    variant_geometry_metrics.update(
                                        variant_waypoint_metrics
                                    )
                                    if (
                                        self.policy_guard_eraf_grounding_objective_version
                                        >= 21
                                    ):
                                        (
                                            variant_action,
                                            variant_geometry_residual,
                                            variant_expert_metrics,
                                        ) = self.policy_guard_modules[
                                            "eraf_phase_expert_residual_adapter"
                                        ](
                                            candidate_action=(
                                                pre_geometry_variant_action
                                            ),
                                            current_residual=(
                                                variant_geometry_residual
                                            ),
                                            desired_direction=(
                                                variant_phase_servo_metrics[
                                                    "pgc_v919_desired_direction"
                                                ]
                                            ),
                                            desired_distance=(
                                                variant_phase_servo_metrics[
                                                    "pgc_v919_desired_distance_per_sample"
                                                ]
                                            ),
                                            control_phase=(
                                                variant_phase_servo_metrics[
                                                    "pgc_v919_selected_control_phase"
                                                ]
                                            ),
                                            route_confidence=(
                                                variant_phase_servo_metrics[
                                                    "pgc_v919_route_confidence_per_sample"
                                                ]
                                            ),
                                            waypoint_compatibility=(
                                                variant_waypoint_metrics[
                                                    "pgc_v920_compatibility_probability_per_step"
                                                ]
                                            ),
                                            action_is_pad=None,
                                        )
                                        variant_geometry_metrics.update(
                                            variant_expert_metrics
                                        )
                                        if (
                                            self.policy_guard_eraf_grounding_objective_version
                                            >= 24
                                        ):
                                            (
                                                variant_action,
                                                _,
                                                variant_retention_metrics,
                                            ) = self.policy_guard_modules[
                                                "eraf_clause_semantic_retention_residual"
                                            ](
                                                base_action=latents_action,
                                                aligned_action=variant_action,
                                                goal_queries=variant_queries,
                                                selected_clause=(
                                                    variant_phase_servo_metrics[
                                                        "pgc_v919_selected_clause"
                                                    ]
                                                ),
                                                control_phase=(
                                                    variant_phase_servo_metrics[
                                                        "pgc_v919_selected_control_phase"
                                                    ]
                                                ),
                                                route_confidence=(
                                                    variant_phase_servo_metrics[
                                                        "pgc_v919_route_confidence_per_sample"
                                                    ]
                                                ),
                                                action_is_pad=None,
                                            )
                                            variant_geometry_residual = (
                                                variant_action
                                                - pre_geometry_variant_action
                                            )
                                            variant_geometry_metrics.update(
                                                variant_retention_metrics
                                            )
                            variant_residual = (
                                variant_residual + variant_geometry_residual
                            )
                            variant_metrics = dict(variant_metrics)
                            variant_metrics.update(variant_geometry_metrics)
                    causal_variants[name] = {
                        # Retained only for the explicit causal-audit API.  The
                        # standalone audit replays the small Proposal with this
                        # tensor to measure the expert-loss gradient with
                        # respect to ERAF-routed queries; ordinary rollout
                        # results never expose it.
                        "goal_queries": variant_queries.detach().to(
                            device="cpu", dtype=torch.float32
                        ),
                        "action": variant_action[0].detach().to(
                            device="cpu", dtype=torch.float32
                        ),
                        "residual": variant_residual[0].detach().to(
                            device="cpu", dtype=torch.float32
                        ),
                        "residual_rms": float(
                            variant_residual.float().square().mean().sqrt().item()
                        ),
                        "goal_query_rms": float(
                            variant_queries.float().square().mean().sqrt().item()
                        ),
                        "goal_query_delta_from_learned_rms": float(
                            (variant_queries - learned_queries)
                            .float()
                            .square()
                            .mean()
                            .sqrt()
                            .item()
                        ),
                        "goal_embedding_delta_from_learned_rms": float(
                            (variant_embedding - learned_embedding)
                            .float()
                            .square()
                            .mean()
                            .sqrt()
                            .item()
                        ),
                        "proposal_attention_entropy": (
                            None
                            if "pgc_v4_action_residual_attention_entropy"
                            not in variant_metrics
                            else float(
                                variant_metrics[
                                    "pgc_v4_action_residual_attention_entropy"
                                ]
                                .detach()
                                .float()
                                .item()
                            )
                        ),
                        "v919_selected_clause": (
                            None
                            if "pgc_v919_selected_clause" not in variant_metrics
                            else int(
                                variant_metrics["pgc_v919_selected_clause"]
                                .detach()
                                .reshape(-1)[0]
                                .item()
                            )
                        ),
                        "v919_selected_control_phase": (
                            None
                            if "pgc_v919_selected_control_phase"
                            not in variant_metrics
                            else int(
                                variant_metrics[
                                    "pgc_v919_selected_control_phase"
                                ]
                                .detach()
                                .reshape(-1)[0]
                                .item()
                            )
                        ),
                    }
                result["policy_guard_eraf_causal_variants"] = causal_variants
            if self.policy_guard_version in {6, 7}:
                metric_prefix = (
                    "pgc_v7" if self.policy_guard_version == 7 else "pgc_v6"
                )
                metric_names = (
                    (
                        f"{metric_prefix}_target_attention_top1_mass",
                        "policy_guard_target_binding_top1_mass",
                    ),
                    (
                        f"{metric_prefix}_target_attention_entropy",
                        "policy_guard_target_binding_entropy",
                    ),
                    (
                        f"{metric_prefix}_target_similarity_max",
                        "policy_guard_target_binding_similarity_max",
                    ),
                )
                for metric_name, output_name in metric_names:
                    value = policy_guard_goal_metrics.get(metric_name)
                    if value is not None:
                        result[output_name] = float(
                            value.detach().float().item()
                        )
            if self.policy_guard_version == 9:
                if self._policy_guard_last_eraf_diagnostics is None:
                    raise RuntimeError(
                        "PGC v9 inference did not retain ERAF diagnostics."
                    )
                result["policy_guard_eraf_diagnostics"] = (
                    self._policy_guard_last_eraf_diagnostics
                )
                if self.policy_guard_eraf_grounding_objective_version >= 14:
                    next_policy_state = {
                        "phase_safe_memory_state_ids": (
                            self._policy_guard_last_eraf_diagnostics[
                                "phase_safe_memory_next_state_ids"
                            ].clone()
                        ),
                        "phase_safe_memory_valid": (
                            self._policy_guard_last_eraf_diagnostics[
                                "phase_safe_memory_next_state_valid"
                            ].clone()
                        ),
                    }
                    result["policy_guard_state"] = (
                        self._policy_guard_completion_only_state(
                            next_policy_state,
                            previous_state=policy_guard_state,
                        )
                    )
            return result

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
        if not self.policy_guard_enabled:
            raise RuntimeError("PGC metadata requested while policy guard is disabled.")
        is_v2 = self.policy_guard_version == 2
        is_v3 = self.policy_guard_version == 3
        is_v4 = self.policy_guard_version >= 4
        is_v5 = self.policy_guard_version >= 5
        is_v6 = self.policy_guard_version == 6
        is_v7 = self.policy_guard_version == 7
        is_v8 = self.policy_guard_version == 8
        is_v9 = self.policy_guard_version == 9
        is_v926 = bool(
            is_v9
            and self.policy_guard_eraf_grounding_objective_version >= 26
        )
        uses_target_binder = is_v6 or is_v7
        residual_cap = None
        if is_v3:
            residual_cap = (
                self.policy_guard_modules["action_velocity_residual"]
                .max_abs.detach()
                .to(device="cpu", dtype=torch.float32)
                .reshape(-1)
                .tolist()
            )
        action_chunk_cap = None
        if is_v4:
            action_chunk_cap = (
                self.policy_guard_modules["action_chunk_proposal"]
                .max_abs.detach()
                .to(device="cpu", dtype=torch.float32)
                .reshape(-1)
                .tolist()
            )
        proposal_module = (
            self.policy_guard_modules["action_chunk_proposal"]
            if is_v4
            else None
        )
        verifier_module = self.policy_guard_modules["verifier"]
        eraf_role_adapter_trainable_scope = None
        if is_v9 and self.policy_guard_eraf_grounding_objective_version >= 4:
            objective_version = self.policy_guard_eraf_grounding_objective_version
            if objective_version >= 14:
                eraf_role_adapter_trainable_scope = (
                    (
                        (
                            "shared_video_action_lora_plus_eraf_action_context_"
                            "injector"
                            if objective_version >= 26
                            else "eraf_action_context_injector_only"
                            if objective_version >= 25
                            else "clause_semantic_retention_residual_only"
                            if objective_version >= 24
                            else "phase_specific_privileged_expert_residual_adapter_only"
                            if objective_version >= 21
                            else "phase_compatible_local_waypoint_adapter_only"
                            if objective_version >= 20
                            else "hard_routed_phase_servo_only"
                            if objective_version >= 19
                            else (
                                "phase_conditioned_geometry_adapter_only_with_"
                                "phase_balanced_residual_imitation"
                            )
                            if objective_version >= 18
                            else "phase_conditioned_relative_geometry_action_"
                            "adapter_only"
                        )
                        if objective_version >= 17
                        else (
                            "semantic_causal_action_grounding_bridge_only"
                            if objective_version >= 16
                            else "frozen_eraf_perception_action_bridge_plus_proposal"
                        )
                    )
                    if self.policy_guard_eraf_action_joint_training
                    else "phase_safe_temporal_clause_memory_only"
                )
            elif objective_version >= 13:
                eraf_role_adapter_trainable_scope = (
                    "closed_loop_phase_rebinding_adapter_only"
                )
            elif objective_version >= 12:
                eraf_role_adapter_trainable_scope = (
                    "audited_hard_clause_tuple_balanced_visual_"
                    "role_binding_adapter_only"
                )
            elif objective_version >= 11:
                eraf_role_adapter_trainable_scope = (
                    "exclusive_all_entity_balanced_visual_role_"
                    "binding_adapter_only"
                )
            elif objective_version >= 10:
                eraf_role_adapter_trainable_scope = (
                    "clause_activation_plus_balanced_role_plus_"
                    "visibility_gated_view_fusion_plus_unfinished_"
                    "clause_scheduler"
                )
            elif objective_version >= 9:
                eraf_role_adapter_trainable_scope = (
                    "clause_activation_calibration_adapter_only"
                )
            elif objective_version >= 8:
                eraf_role_adapter_trainable_scope = (
                    "exclusive_evidence_global_hard_curriculum_"
                    "balanced_visual_role_binding_adapter_only"
                )
            elif objective_version >= 7:
                eraf_role_adapter_trainable_scope = (
                    "global_hard_curriculum_balanced_visual_"
                    "role_binding_adapter_only"
                )
            elif objective_version >= 6:
                eraf_role_adapter_trainable_scope = (
                    "balanced_visual_role_binding_adapter_only"
                )
            elif objective_version >= 5:
                eraf_role_adapter_trainable_scope = (
                    "structured_role_assignment_adapter_only"
                )
            else:
                eraf_role_adapter_trainable_scope = (
                    "role_assignment_adapter_only"
                )
        return {
            "architecture": "pgc_fastwam",
            "policy_guard_version": self.policy_guard_version,
            "grounding": (
                "predicate_entity_relation_affordance_field"
                if is_v9
                else None
            ),
            "privileged_supervision": "training_only" if is_v9 else None,
            "deployment_inputs": (
                "rgb_language_proprio_completed_clause_bitset"
                if is_v9 and self.policy_guard_eraf_completion_only_memory
                else (
                    "rgb_language_proprio_previous_policy_state"
                    if is_v9
                    and self.policy_guard_eraf_grounding_objective_version >= 14
                    else ("rgb_language_proprio" if is_v9 else None)
                )
            ),
            "base_policy": (
                "frozen_released_fastwam_with_shared_expert_lora"
                if is_v926
                else "frozen_released_fastwam"
            ),
            "base_action_interface": "query_free_joint_mot",
            "counterfactual_policy": (
                "single_eraf_path_shared_video_action_expert_lora"
                if is_v926
                else "shared_frozen_action_expert_with_internal_eraf_context_injection"
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 25
                else "frozen_base_plus_rollout_aligned_action_chunk_proposal"
                if is_v4
                else (
                    "frozen_base_plus_bounded_velocity_residual"
                    if is_v3
                    else (
                        "base_equivalent_visual_residual_action_expert_lora"
                        if is_v2
                        else "independent_action_expert_lora"
                    )
                )
            ),
            "counterfactual_tuning": (
                "native_and_counterfactual_world_action_joint_lora"
                if is_v926
                else "counterfactual_only_internal_action_expert_context_conditioning"
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 25
                else "entity_relation_affordance_grounded_paired_action_residual"
                if is_v9
                else (
                    "closed_loop_replay_verified_target_acquisition_residual"
                    if is_v8
                    else (
                        "object_token_mask_grounded_paired_action_residual"
                        if is_v7
                        else (
                            "visual_target_bottleneck_paired_action_residual"
                            if is_v6
                            else (
                                "paired_language_prefix_aligned_action_residual"
                                if is_v5
                                else (
                                    "rollout_aligned_final_action_residual"
                                    if is_v4
                                    else (
                                        "bounded_velocity_residual"
                                        if is_v3
                                        else "lora"
                                    )
                                )
                            )
                        )
                    )
                )
            ),
            "counterfactual_action_interface": (
                "single_eraf_conditioned_action_denoising_path"
                if is_v926
                else "same_noise_full_denoising_with_eraf_context_tokens"
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 25
                else "shared_frozen_base_final_action_chunk"
                if is_v4
                else (
                    "shared_frozen_base_raw_current_visual"
                    if is_v3
                    else (
                        "query_free_raw_current_visual"
                        if is_v2
                        else "latent_query_goal_bottleneck"
                    )
                )
            ),
            "goal_injection": (
                "eraf_tokens_inside_shared_action_expert_denoising"
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 25
                else "eraf_clause_tokens_cross_attention_to_v5_proposal"
                if is_v9
                else (
                    "per_query_spatial_object_tokens_to_action_chunk_residual"
                    if is_v7
                    else (
                        "language_selected_visual_target_to_action_chunk_residual"
                        if is_v6
                        else (
                            "post_sampler_temporal_action_chunk_residual"
                            if is_v4
                            else (
                                "post_dit_bounded_velocity_residual"
                                if is_v3
                                else (
                                    "zero_initialized_action_token_residual"
                                    if is_v2
                                    else "latent_action_query_replacement"
                                )
                            )
                        )
                    )
                )
            ),
            "native_policy_teacher": (
                "frozen_base_fully_denoised_action_same_initial_noise"
                if is_v4
                else (
                    "frozen_base_velocity_same_noise_timestep"
                    if (is_v2 or is_v3)
                    else None
                )
            ),
            "native_distillation_weight": (
                self.policy_guard_native_distillation_weight
            ),
            "goal_residual_scale": self.policy_guard_goal_residual_scale,
            "lora_rank": (
                int(self.lora_config["rank"])
                if is_v926 or not (is_v3 or is_v4)
                else None
            ),
            "lora_alpha": (
                float(self.lora_config["alpha"])
                if is_v926 or not (is_v3 or is_v4)
                else None
            ),
            "lora_dropout": (
                float(self.lora_config["dropout"])
                if is_v926 or not (is_v3 or is_v4)
                else None
            ),
            "lora_target_modules": (
                list(self.lora_config["target_modules"])
                if is_v926 or not (is_v3 or is_v4)
                else []
            ),
            "num_action_queries": int(self.policy_guard_num_action_queries),
            "query_rope_offset": (
                int(self.policy_guard_query_rope_offset)
                if not (is_v2 or is_v3 or is_v4)
                else None
            ),
            "goal_graph_tokens": int(
                self.policy_guard_modules["goal_graph"].num_goal_tokens
            ),
            "gate_mode": "eraf_only" if is_v926 else self.policy_guard_gate_mode,
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
            "policy_protection": (
                "single_eraf_path_no_candidate_gate"
                if is_v926
                else "single_immutable_base_plus_conservative_hard_gate"
                if (is_v3 or is_v4)
                else "immutable_base_plus_conservative_hard_gate"
            ),
            "representation_supervision": (
                (
                    (
                        (
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    "immutable_base_closed_loop_"
                                                    "phase_rebinding_with_offline_"
                                                    "grounding_guards"
                                                    if self.policy_guard_eraf_grounding_objective_version
                                                    >= 13
                                                    else (
                                                        "exclusive_clause_tuple_subject_"
                                                        "predicate_reference_assignment_"
                                                        "with_audited_hard_curriculum"
                                                    )
                                                )
                                                if self.policy_guard_eraf_grounding_objective_version
                                                >= 12
                                                else (
                                                    "exclusive_same_state_all_entity_"
                                                    "bipartite_role_assignment_with_"
                                                    "cross_clause_hard_negatives"
                                                )
                                            )
                                            if self.policy_guard_eraf_grounding_objective_version
                                            >= 11
                                            else (
                                                "exclusive_evidence_subject_reference_"
                                                "assignment_with_full_mask_localization"
                                            )
                                        )
                                        if self.policy_guard_eraf_grounding_objective_version
                                        >= 8
                                        else (
                                            "frozen_v9_3_native_hard_curriculum_global_"
                                            "ddp_bipartite_binding_with_all_clause_"
                                            "geometry_preservation"
                                        )
                                    )
                                )
                                if self.policy_guard_eraf_grounding_objective_version
                                >= 7
                                else (
                                    "frozen_v9_3_visual_candidate_bipartite_role_"
                                    "binding_with_balanced_hard_easy_gradients"
                                )
                            )
                            if self.policy_guard_eraf_grounding_objective_version >= 6
                            else (
                                "frozen_v9_3_cross_clause_structured_role_"
                                "assignment_with_same_state_entity_negatives"
                            )
                        )
                        if self.policy_guard_eraf_grounding_objective_version
                        >= 5
                        else (
                            "frozen_v9_1_role_residual_assignment_with_"
                            "teacher_preserved_attention_relation_and_anchors"
                        )
                    )
                    if self.policy_guard_eraf_grounding_objective_version >= 4
                    else (
                        (
                            "bidirectional_role_assignment_hard_mining_"
                            "gate_aligned_attention_multiview_masks_3d_anchors_"
                            "state_and_phase"
                        )
                        if self.policy_guard_eraf_grounding_objective_version >= 3
                        else (
                            "gate_aligned_attention_mass_role_swap_multiview_"
                            "masks_3d_anchors_state_and_phase"
                            if self.policy_guard_eraf_grounding_objective_version
                            >= 2
                            else (
                                "predicate_roles_multiview_masks_3d_anchors_"
                                "state_and_phase"
                            )
                        )
                    )
                )
                if is_v9
                else (
                    "frozen_v5_language_plus_exact_closed_loop_state_corrective_actions"
                    if is_v8
                    else (
                        "explicit_current_state_element_masks_and_cross_object_negatives"
                        if is_v7
                        else (
                            "interaction_teacher_task_prototypes_same_state_hard_negatives"
                            if is_v6
                            else (
                                "same_state_paired_language_plus_prefix_action_and_hard_negatives"
                                if is_v5
                                else (
                                    "rollout_aligned_direct_action_plus_goal_action_alignment"
                                    if is_v4
                                    else (
                                        "direct_residual_action_plus_goal_action_alignment"
                                        if is_v3
                                        else "direct_goal_action_alignment"
                                    )
                                )
                            )
                        )
                    )
                )
            ),
            "verifier_margin_space": (
                "raw_fp32_pairwise_advantage"
                if is_v4
                else ("probability" if (is_v2 or is_v3) else "logit")
            ),
            "verifier_deployment_role": (
                "diagnostic_only_no_action_selection" if is_v926 else None
            ),
            "world_model_supervision": (
                "future_video_flow_plus_paired_wrong_language_ranking"
                if is_v926
                else None
            ),
            "velocity_residual_max_abs": residual_cap,
            "action_chunk_residual_max_abs": action_chunk_cap,
            "rollout_num_inference_steps": (
                self.policy_guard_rollout_num_inference_steps if is_v4 else None
            ),
            "proposal_hidden_dim": (
                int(proposal_module.hidden_dim) if proposal_module is not None else None
            ),
            "proposal_num_heads": (
                int(proposal_module.num_heads) if proposal_module is not None else None
            ),
            "proposal_num_layers": (
                int(proposal_module.num_layers) if proposal_module is not None else None
            ),
            "verifier_hidden_dim": int(verifier_module.hidden_dim),
            "verifier_num_heads": (
                int(verifier_module.num_heads) if is_v4 else None
            ),
            "verifier_num_layers": (
                int(verifier_module.num_layers) if is_v4 else None
            ),
            "action_gripper_weight": (
                self.policy_guard_action_gripper_weight if is_v4 else None
            ),
            "advantage_temperature": (
                self.policy_guard_advantage_temperature if is_v4 else None
            ),
            "advantage_clip": (
                self.policy_guard_advantage_clip if is_v4 else None
            ),
            "candidate_max_saturation_fraction": (
                self.policy_guard_candidate_max_saturation_fraction
                if is_v4
                else None
            ),
            "candidate_max_delta_rms": (
                self.policy_guard_candidate_max_delta_rms if is_v4 else None
            ),
            "residual_regularization_weight": (
                self.policy_guard_residual_regularization_weight
            ),
            "residual_smoothness_weight": (
                self.policy_guard_residual_smoothness_weight
            ),
            "verifier_start_step": self.policy_guard_verifier_start_step,
            "verifier_ramp_steps": self.policy_guard_verifier_ramp_steps,
            "execution_prefix_steps": (
                self.policy_guard_execution_prefix_steps if is_v5 else None
            ),
            "suffix_loss_weight": (
                self.policy_guard_suffix_loss_weight if is_v5 else None
            ),
            "completion_phase_enabled": (
                self.policy_guard_completion_phase_enabled if is_v5 else None
            ),
            "completion_phase_source": (
                "audited_executed_gripper_transition_sidecar"
                if is_v5 and self.policy_guard_completion_phase_enabled
                else None
            ),
            "completion_transport_weight": (
                self.policy_guard_completion_transport_weight
                if is_v5 and self.policy_guard_completion_phase_enabled
                else None
            ),
            "completion_release_weight": (
                self.policy_guard_completion_release_weight
                if is_v5 and self.policy_guard_completion_phase_enabled
                else None
            ),
            "completion_trainable_scope": (
                "action_chunk_proposal_only"
                if is_v5
                and self.policy_guard_completion_phase_enabled
                and self.policy_guard_completion_train_proposal_only
                else None
            ),
            "same_state_source_zero_weight": (
                self.policy_guard_same_state_source_zero_weight if is_v5 else None
            ),
            "goal_separation_weight": (
                self.policy_guard_goal_separation_weight if is_v5 else None
            ),
            "goal_separation_margin": (
                self.policy_guard_goal_separation_margin if is_v5 else None
            ),
            "residual_separation_weight": (
                self.policy_guard_residual_separation_weight if is_v5 else None
            ),
            "residual_separation_margin": (
                self.policy_guard_residual_separation_margin if is_v5 else None
            ),
            "verifier_wrong_language_weight": (
                self.policy_guard_verifier_wrong_language_weight if is_v5 else None
            ),
            "verifier_bad_candidate_weight": (
                self.policy_guard_verifier_bad_candidate_weight if is_v5 else None
            ),
            "target_binding_bottleneck": (
                (
                    "spatial_object_tokens_no_direct_language_residual"
                    if is_v7
                    else "visual_only_no_direct_language_residual"
                )
                if uses_target_binder
                else None
            ),
            "target_binding_visual_source": (
                "pre_dit_language_neutral_current_frame"
                if uses_target_binder
                else None
            ),
            "target_prototype_bank_persisted": True if is_v6 else None,
            "target_binding_hidden_dim": (
                self.policy_guard_target_binding_hidden_dim
                if uses_target_binder
                else None
            ),
            "target_binding_num_heads": (
                int(self.policy_guard_modules["target_binder"].num_heads)
                if uses_target_binder
                else None
            ),
            "target_binding_projection_dim": (
                int(self.policy_guard_modules["target_binder"].projection_dim)
                if uses_target_binder
                else None
            ),
            "target_binding_temperature": (
                self.policy_guard_target_binding_temperature
                if uses_target_binder
                else None
            ),
            "target_binding_teacher_topk": (
                self.policy_guard_target_binding_teacher_topk if is_v6 else None
            ),
            "target_binding_teacher_temperature": (
                self.policy_guard_target_binding_teacher_temperature
                if is_v6
                else None
            ),
            "target_binding_prototype_slots": (
                self.policy_guard_target_binding_prototype_slots if is_v6 else None
            ),
            "target_binding_prototype_momentum": (
                self.policy_guard_target_binding_prototype_momentum
                if is_v6
                else None
            ),
            "target_binding_prototype_temperature": (
                self.policy_guard_target_binding_prototype_temperature
                if is_v6
                else None
            ),
            "target_binding_prototype_topk": (
                self.policy_guard_target_binding_prototype_topk if is_v6 else None
            ),
            "target_binding_action_start_step": (
                self.policy_guard_target_binding_action_start_step
                if uses_target_binder
                else None
            ),
            "target_binding_action_ramp_steps": (
                self.policy_guard_target_binding_action_ramp_steps
                if uses_target_binder
                else None
            ),
            "target_binding_interaction_weight": (
                self.policy_guard_target_binding_interaction_weight if is_v6 else None
            ),
            "target_binding_prototype_weight": (
                self.policy_guard_target_binding_prototype_weight if is_v6 else None
            ),
            "target_binding_source_weight": (
                self.policy_guard_target_binding_source_weight if is_v6 else None
            ),
            "target_binding_hard_negative_weight": (
                self.policy_guard_target_binding_hard_negative_weight
                if is_v6
                else None
            ),
            "target_binding_separation_weight": (
                self.policy_guard_target_binding_separation_weight if is_v6 else None
            ),
            "target_binding_hard_negative_margin": (
                self.policy_guard_target_binding_hard_negative_margin
                if is_v6
                else None
            ),
            "target_binding_separation_margin": (
                self.policy_guard_target_binding_separation_margin if is_v6 else None
            ),
            "target_mask_supervision": (
                "robosuite_element_current_frame_training_only"
                if is_v7
                else None
            ),
            "target_binding_num_object_tokens": (
                self.policy_guard_target_binding_num_object_tokens
                if is_v7
                else None
            ),
            "target_binding_camera_count": (
                self.policy_guard_target_binding_camera_count if is_v7 else None
            ),
            "target_binding_visual_aspect_ratio": (
                float(
                    self.policy_guard_modules[
                        "target_binder"
                    ].visual_aspect_ratio
                )
                if is_v7
                else None
            ),
            "target_mask_weight": (
                self.policy_guard_target_mask_weight if is_v7 else None
            ),
            "source_mask_weight": (
                self.policy_guard_source_mask_weight if is_v7 else None
            ),
            "aux_mask_weight": (
                self.policy_guard_aux_mask_weight if is_v7 else None
            ),
            "mask_mass_weight": (
                self.policy_guard_mask_mass_weight if is_v7 else None
            ),
            "cross_object_weight": (
                self.policy_guard_cross_object_weight if is_v7 else None
            ),
            "cross_object_margin": (
                self.policy_guard_cross_object_margin if is_v7 else None
            ),
            "closed_loop_corrective_enabled": (
                self.policy_guard_closed_loop_corrective_enabled
                if is_v8
                else None
            ),
            "closed_loop_corrective_format": (
                "pgc_libero_closed_loop_corrective_v1" if is_v8 else None
            ),
            "closed_loop_corrective_weight": (
                self.policy_guard_closed_loop_corrective_weight
                if is_v8
                else None
            ),
            "offline_acquisition_weight": (
                self.policy_guard_offline_acquisition_weight
                if is_v8
                else None
            ),
            "native_guard_weight": (
                self.policy_guard_native_guard_weight if (is_v8 or is_v9) else None
            ),
            "acquisition_only": (
                self.policy_guard_acquisition_only if is_v8 else None
            ),
            "closed_loop_trainable_scope": (
                "action_chunk_proposal_only"
                if is_v8
                and self.policy_guard_closed_loop_train_proposal_only
                else None
            ),
            "warm_start_contract": (
                (
                    self.policy_guard_eraf_initialization_contract
                    if is_v9
                    else "exact_pgc_v5_sidecars"
                )
                if (is_v8 or is_v9)
                else None
            ),
            "eraf_training_stage": (
                self.policy_guard_eraf_training_stage if is_v9 else None
            ),
            "eraf_hidden_dim": (
                self.policy_guard_eraf_hidden_dim if is_v9 else None
            ),
            "eraf_num_heads": (
                self.policy_guard_eraf_num_heads if is_v9 else None
            ),
            "eraf_max_clauses": (
                self.policy_guard_eraf_max_clauses if is_v9 else None
            ),
            "eraf_camera_count": (
                self.policy_guard_eraf_camera_count if is_v9 else None
            ),
            "eraf_visual_aspect_ratio": (
                self.policy_guard_eraf_visual_aspect_ratio if is_v9 else None
            ),
            "eraf_temperature": (
                self.policy_guard_eraf_temperature if is_v9 else None
            ),
            "eraf_learning_rate": (
                self.policy_guard_eraf_learning_rate if is_v9 else None
            ),
            "eraf_grounding_aux_weight": (
                self.policy_guard_eraf_grounding_aux_weight if is_v9 else None
            ),
            "eraf_grounding_objective_version": (
                self.policy_guard_eraf_grounding_objective_version
                if is_v9
                else None
            ),
            "eraf_completion_only_memory": (
                self.policy_guard_eraf_completion_only_memory
                if is_v9
                else None
            ),
            "eraf_action_joint_training": (
                self.policy_guard_eraf_action_joint_training
                if is_v9
                else None
            ),
            "eraf_action_joint_contract": (
                (
                    (
                        "frozen_eraf_completion_memory_plus_shared_video_action_"
                        "expert_lora_and_internal_context_injector_single_path"
                        if self.policy_guard_eraf_grounding_objective_version >= 26
                        else "frozen_eraf_and_shared_action_expert_plus_internal_"
                        "context_injector_no_post_action_residual"
                        if self.policy_guard_eraf_grounding_objective_version >= 25
                        else "frozen_v921_teacher_plus_alignment_preserving_"
                        "negative_focused_final_action_clause_ranking"
                        if self.policy_guard_eraf_grounding_objective_version == 23
                        else "frozen_v921_expert_adapter_plus_isolated_clause_"
                        "semantic_retention_residual"
                        if self.policy_guard_eraf_grounding_objective_version >= 24
                        else "frozen_v920_stack_plus_phase_specific_expert_adapter_"
                        "with_balanced_final_action_clause_ranking"
                        if self.policy_guard_eraf_grounding_objective_version >= 22
                        else "frozen_v920_stack_plus_phase_specific_privileged_"
                        "expert_prefix_residual_alignment"
                        if self.policy_guard_eraf_grounding_objective_version >= 21
                        else "frozen_v919_stack_plus_phase_compatible_local_"
                        "waypoint_vector_field"
                        if self.policy_guard_eraf_grounding_objective_version >= 20
                        else "frozen_v918_stack_plus_hard_clause_phase_"
                        "direction_preserving_servo"
                        if self.policy_guard_eraf_grounding_objective_version >= 19
                        else (
                            "frozen_eraf_v917_stack_plus_phase_balanced_direct_"
                            "geometry_residual_imitation"
                        )
                        if self.policy_guard_eraf_grounding_objective_version >= 18
                        else "frozen_eraf_v916_bridge_and_proposal_plus_direct_"
                        "eef_relative_geometry_action_adapter"
                    )
                    if self.policy_guard_eraf_grounding_objective_version >= 17
                    else (
                        "frozen_eraf_perception_proposal_and_legacy_bridge_plus_"
                        "semantic_causal_action_grounding_bridge"
                        if self.policy_guard_eraf_grounding_objective_version >= 16
                        else (
                            "frozen_eraf_perception_plus_phase_conditioned_geometry_"
                            "bridge_legacy_bridge_and_proposal"
                            if self.policy_guard_eraf_grounding_objective_version >= 15
                            else "frozen_eraf_perception_plus_action_bridge_and_proposal"
                        )
                    )
                )
                if is_v9 and self.policy_guard_eraf_action_joint_training
                else None
            ),
            "eraf_action_trainable_scope": (
                (
                    (
                        "shared_video_action_lora_plus_eraf_action_context_"
                        "injector"
                        if self.policy_guard_eraf_grounding_objective_version >= 26
                        else "eraf_action_context_injector_only"
                        if self.policy_guard_eraf_grounding_objective_version >= 25
                        else "clause_semantic_retention_residual_only"
                        if self.policy_guard_eraf_grounding_objective_version >= 24
                        else "phase_specific_privileged_expert_residual_adapter_only"
                        if self.policy_guard_eraf_grounding_objective_version >= 21
                        else "phase_compatible_local_waypoint_adapter_only"
                        if self.policy_guard_eraf_grounding_objective_version >= 20
                        else "hard_routed_phase_servo_only"
                        if self.policy_guard_eraf_grounding_objective_version >= 19
                        else (
                            "phase_conditioned_geometry_adapter_only_with_phase_"
                            "balanced_residual_imitation"
                        )
                        if self.policy_guard_eraf_grounding_objective_version >= 18
                        else "phase_conditioned_relative_geometry_action_adapter_only"
                    )
                    if self.policy_guard_eraf_grounding_objective_version >= 17
                    else (
                        "semantic_causal_action_grounding_bridge_only"
                        if self.policy_guard_eraf_grounding_objective_version >= 16
                        else (
                            "phase_conditioned_subject_reference_anchor_action_bridge_"
                            "plus_legacy_bridge_and_action_chunk_proposal"
                            if self.policy_guard_eraf_grounding_objective_version >= 15
                            else "base_query_projection_relation_attention_query_"
                            "embedding_delta_plus_action_chunk_proposal"
                        )
                    )
                )
                if is_v9 and self.policy_guard_eraf_action_joint_training
                else None
            ),
            "eraf_action_context_injection_contract": (
                "append_bounded_eraf_tokens_to_shared_action_expert_context_"
                "at_every_denoising_step_no_post_action_residual"
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 25
                else None
            ),
            "eraf_action_context_hidden_dim": (
                self.policy_guard_eraf_action_geometry_hidden_dim
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 25
                else None
            ),
            "eraf_post_action_residual_active": (
                False
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 25
                else None
            ),
            "eraf_single_path": True if is_v926 else None,
            "eraf_shared_expert_lora_config": (
                dict(self.lora_config) if is_v926 else None
            ),
            "eraf_expert_lora_training_contract": (
                {
                    "future_video_flow": self.loss_lambda_video,
                    "paired_wrong_language_world_ranking": (
                        self.policy_guard_eraf_expert_lora_world_language_weight
                    ),
                    "paired_wrong_language_world_margin": (
                        self.policy_guard_eraf_expert_lora_world_language_margin
                    ),
                    "native_action_flow": (
                        self.policy_guard_eraf_expert_lora_native_action_weight
                    ),
                    "counterfactual_action_flow": (
                        self.policy_guard_eraf_expert_lora_counterfactual_action_weight
                    ),
                    "eraf_preservation": (
                        self.policy_guard_eraf_grounding_aux_weight
                    ),
                }
                if is_v926
                else None
            ),
            "eraf_expert_lora_world_language_weight": (
                self.policy_guard_eraf_expert_lora_world_language_weight
                if is_v926
                else None
            ),
            "eraf_expert_lora_world_language_margin": (
                self.policy_guard_eraf_expert_lora_world_language_margin
                if is_v926
                else None
            ),
            "eraf_expert_lora_native_action_weight": (
                self.policy_guard_eraf_expert_lora_native_action_weight
                if is_v926
                else None
            ),
            "eraf_expert_lora_counterfactual_action_weight": (
                self.policy_guard_eraf_expert_lora_counterfactual_action_weight
                if is_v926
                else None
            ),
            "eraf_expert_lora_regularization_weight": (
                self.policy_guard_eraf_expert_lora_regularization_weight
                if is_v926
                else None
            ),
            "eraf_action_grounding_contract": (
                "separate_subject_reference_relation_grasp_goal_interaction_"
                "displacement_tokens_zero_init_v9_14_exact"
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 15
                else None
            ),
            "eraf_action_grounding_hidden_dim": (
                self.policy_guard_eraf_action_grounding_hidden_dim
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 15
                else None
            ),
            "eraf_action_grounding_num_heads": (
                self.policy_guard_eraf_action_grounding_num_heads
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 15
                else None
            ),
            "eraf_action_grounding_learning_rate": (
                self.policy_guard_eraf_action_grounding_learning_rate
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 15
                else None
            ),
            "eraf_action_causal_ranking_weight": (
                self.policy_guard_eraf_action_causal_ranking_weight
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 15
                else None
            ),
            "eraf_action_causal_margin": (
                self.policy_guard_eraf_action_causal_margin
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 15
                else None
            ),
            "eraf_action_semantic_negative_contract": (
                "joint_valid_entity_id_swap_with_same_state_subject_as_"
                "reference_fallback_plus_clause_swap"
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 16
                else None
            ),
            "eraf_action_geometry_contract": (
                "phase_selected_eef_relative_grasp_goal_interaction_relation_"
                "direct_bounded_action_residual_zero_init_v9_16_exact"
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 17
                else None
            ),
            "eraf_action_geometry_hidden_dim": (
                self.policy_guard_eraf_action_geometry_hidden_dim
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 17
                else None
            ),
            "eraf_action_geometry_learning_rate": (
                self.policy_guard_eraf_action_geometry_learning_rate
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 17
                else None
            ),
            "eraf_action_geometry_residual_max_abs": (
                self.policy_guard_eraf_action_geometry_residual_max_abs
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 17
                else None
            ),
            "eraf_action_anchor_negative_contract": (
                "same_state_source_anchor_or_reference_mirror_with_offset_fallback"
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 17
                else None
            ),
            "eraf_action_phase_residual_contract": (
                "phase_balanced_bounded_expert_minus_frozen_v9_17_candidate_"
                "prefix_residual_imitation"
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 18
                else None
            ),
            "eraf_action_phase_servo_contract": (
                "hard_single_clause_explicit_affine_eef_phase_specific_positive_"
                "cartesian_gain_with_legacy_suppression"
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 19
                else None
            ),
            "eraf_action_waypoint_contract": (
                "hard_clause_phase_compatible_positive_progress_local_tangent_"
                "waypoint_with_privileged_training_only_compatibility_labels"
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 20
                else None
            ),
            "eraf_action_waypoint_loss_weights": (
                [
                    self.policy_guard_eraf_action_waypoint_compatibility_weight,
                    self.policy_guard_eraf_action_waypoint_imitation_weight,
                    self.policy_guard_eraf_action_waypoint_direction_weight,
                    self.policy_guard_eraf_action_waypoint_zero_weight,
                ]
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 20
                else None
            ),
            "eraf_action_waypoint_min_cosine": (
                self.policy_guard_eraf_action_waypoint_min_cosine
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 20
                else None
            ),
            "eraf_action_waypoint_tangent_max_ratio": (
                self.policy_guard_eraf_action_waypoint_tangent_max_ratio
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 20
                else None
            ),
            "eraf_action_expert_alignment_contract": (
                "training_only_privileged_phase_anchor_teacher_plus_deployed_"
                "full_action_prefix_residual_and_semantic_causal_ranking"
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 21
                else None
            ),
            "eraf_action_expert_alignment_loss_weights": (
                [
                    self.policy_guard_eraf_action_expert_imitation_weight,
                    self.policy_guard_eraf_action_expert_direction_weight,
                    self.policy_guard_eraf_action_expert_deployed_weight,
                    self.policy_guard_eraf_action_expert_distillation_weight,
                    self.policy_guard_eraf_action_expert_native_zero_weight,
                ]
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 21
                else None
            ),
            "eraf_action_expert_imitation_weight": (
                self.policy_guard_eraf_action_expert_imitation_weight
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 21
                else None
            ),
            "eraf_action_expert_direction_weight": (
                self.policy_guard_eraf_action_expert_direction_weight
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 21
                else None
            ),
            "eraf_action_expert_deployed_weight": (
                self.policy_guard_eraf_action_expert_deployed_weight
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 21
                else None
            ),
            "eraf_action_expert_distillation_weight": (
                self.policy_guard_eraf_action_expert_distillation_weight
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 21
                else None
            ),
            "eraf_action_expert_native_zero_weight": (
                self.policy_guard_eraf_action_expert_native_zero_weight
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 21
                else None
            ),
            "eraf_action_clause_ranking_contract": (
                "frozen_v921_teacher_correct_output_preservation_plus_"
                "expert_nonregression_and_detached_correct_wrong_clause_"
                "ranking_balanced_over_approach_transport_release"
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version == 23
                else "frozen_v921_correct_route_identity_plus_isolated_wrong_"
                "clause_base_fallback_ranking"
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 24
                else "coherent_same_state_clause_swap_final_expert_prefix_mse_"
                "ranking_balanced_over_approach_transport_release"
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 22
                else None
            ),
            "eraf_action_clause_ranking_weight": (
                self.policy_guard_eraf_action_clause_ranking_weight
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 22
                else None
            ),
            "eraf_action_clause_ranking_margin": (
                self.policy_guard_eraf_action_clause_ranking_margin
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 22
                else None
            ),
            "eraf_action_clause_teacher_contract": (
                "training_only_frozen_exact_v921_expert_residual_adapter_"
                "excluded_from_rollout_and_optimizer"
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version == 23
                else None
            ),
            "eraf_action_clause_teacher_weight": (
                self.policy_guard_eraf_action_clause_teacher_weight
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version in {23, 24}
                else None
            ),
            "eraf_action_clause_alignment_guard_weight": (
                self.policy_guard_eraf_action_clause_alignment_guard_weight
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version in {23, 24}
                else None
            ),
            "eraf_action_clause_residual_contract": (
                "frozen_v921_positive_action_plus_identity_initialized_clause_"
                "conditioned_base_fallback"
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 24
                else None
            ),
            "eraf_action_clause_wrong_suppression_weight": (
                self.policy_guard_eraf_action_clause_wrong_suppression_weight
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 24
                else None
            ),
            "eraf_action_servo_frame_weight": (
                self.policy_guard_eraf_action_servo_frame_weight
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 19
                else None
            ),
            "eraf_action_eef_initial_scale": (
                list(self.policy_guard_eraf_action_eef_scale)
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 19
                else None
            ),
            "eraf_action_eef_initial_bias": (
                list(self.policy_guard_eraf_action_eef_bias)
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 19
                else None
            ),
            "eraf_action_phase_residual_imitation_weight": (
                self.policy_guard_eraf_action_phase_residual_imitation_weight
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 18
                else None
            ),
            "eraf_action_phase_direction_weight": (
                self.policy_guard_eraf_action_phase_direction_weight
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 18
                else None
            ),
            "eraf_action_phase_weights": (
                [
                    self.policy_guard_eraf_action_phase_approach_weight,
                    self.policy_guard_eraf_action_phase_transport_weight,
                    self.policy_guard_eraf_action_phase_release_weight,
                ]
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 18
                else None
            ),
            "eraf_action_phase_approach_weight": (
                self.policy_guard_eraf_action_phase_approach_weight
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 18
                else None
            ),
            "eraf_action_phase_transport_weight": (
                self.policy_guard_eraf_action_phase_transport_weight
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 18
                else None
            ),
            "eraf_action_phase_release_weight": (
                self.policy_guard_eraf_action_phase_release_weight
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 18
                else None
            ),
            "eraf_action_phase_direction_min_norm": (
                self.policy_guard_eraf_action_phase_direction_min_norm
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 18
                else None
            ),
            "eraf_hard_role_curriculum": (
                (
                    "v9_10_audited_clause_tuple_native_hard_easy_1_1"
                    if self.policy_guard_eraf_grounding_objective_version >= 12
                    else "v9_3_audited_native_hard_easy_1_1"
                )
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 7
                else None
            ),
            "eraf_ddp_group_balance": (
                "global_count_exact"
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 7
                else None
            ),
            "eraf_geometry_preservation_scope": (
                "all_active_clauses"
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 7
                else None
            ),
            "eraf_role_evidence": (
                "exclusive_subject_reference_support"
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 8
                else None
            ),
            "eraf_role_gate": (
                "exclusive_accuracy_with_full_mask_localization"
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 8
                else None
            ),
            "eraf_exclusive_role_coverage_min": (
                0.5
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 8
                else None
            ),
            "eraf_clause_activation_contract": (
                "zero_init_cross_clause_active_logit_residual"
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 9
                else None
            ),
            "eraf_clause_cardinality_supervision": (
                "balanced_active_bce_plus_count_ce_plus_multi_worst_slot"
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 9
                else None
            ),
            "eraf_clause_gate": (
                "multi_clause_exact_at_least_80pct"
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 9
                else None
            ),
            "eraf_view_fusion_contract": (
                "per_view_local_attention_visibility_gated_zero_init_residual"
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 10
                else None
            ),
            "eraf_clause_scheduler_contract": (
                "first_active_unfinished_predicate_zero_init_residual_route"
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 10
                else None
            ),
            "eraf_all_entity_role_contract": (
                "exclusive_evidence_same_state_all_entity_bipartite_assignment"
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 11
                else None
            ),
            "eraf_multi_clause_gate_contract": (
                "semantic_exact_with_exclusive_role_evidence"
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 11
                else None
            ),
            "eraf_clause_tuple_contract": (
                "exclusive_same_state_subject_predicate_reference_assignment"
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 12
                else None
            ),
            "eraf_clause_tuple_curriculum_contract": (
                "v9_10_audit_native_hard_easy_plus_historical_strict_1_1_1_1"
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 12
                else None
            ),
            "eraf_closed_loop_rebinding_contract": (
                "zero_init_second_pass_role_truth_phase_and_clause_route"
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version == 13
                else None
            ),
            "eraf_closed_loop_state_contract": (
                "immutable_base_correct_replan_exact_simulator_state"
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 13
                else None
            ),
            "eraf_closed_loop_curriculum_contract": (
                "offline_native_closed_loop_native_historical_strict_1_1_1_1"
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 13
                else None
            ),
            "eraf_phase_safe_memory_contract": (
                "explicit_cross_replan_pending_holding_retry_completed"
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 14
                else None
            ),
            "eraf_geometry_protection_contract": (
                "frozen_v9_11_no_query_token_anchor_or_heatmap_residual"
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 14
                else None
            ),
            "eraf_release_transition_contract": (
                "release_true_advance_release_false_retry"
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 14
                else None
            ),
            "eraf_policy_state_contract": (
                "monotonic_completed_bitset_no_pending_holding_retry_recurrence"
                if is_v9 and self.policy_guard_eraf_completion_only_memory
                else (
                    "explicit_caller_owned_reset_per_episode"
                    if is_v9
                    and self.policy_guard_eraf_grounding_objective_version >= 14
                    else None
                )
            ),
            "eraf_phase_safe_memory_warm_start": (
                "exact_v9_11_geometry"
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 14
                else None
            ),
            "eraf_attention_mask_weight": (
                self.policy_guard_eraf_loss_weights.attention_mask
                if is_v9
                else None
            ),
            "eraf_role_swap_weight": (
                self.policy_guard_eraf_loss_weights.role_swap
                if is_v9
                else None
            ),
            "eraf_role_overlap_weight": (
                self.policy_guard_eraf_loss_weights.role_overlap
                if is_v9
                else None
            ),
            "eraf_role_swap_margin": (
                self.policy_guard_eraf_loss_weights.role_swap_margin
                if is_v9
                else None
            ),
            "eraf_role_assignment_weight": (
                self.policy_guard_eraf_loss_weights.role_assignment
                if is_v9
                else None
            ),
            "eraf_role_assignment_temperature": (
                self.policy_guard_eraf_loss_weights.role_assignment_temperature
                if is_v9
                else None
            ),
            "eraf_role_assignment_hard_weight": (
                self.policy_guard_eraf_loss_weights.role_assignment_hard_weight
                if is_v9
                else None
            ),
            "eraf_structured_assignment_weight": (
                self.policy_guard_eraf_loss_weights.structured_assignment
                if is_v9
                else None
            ),
            "eraf_structured_assignment_temperature": (
                self.policy_guard_eraf_loss_weights.structured_assignment_temperature
                if is_v9
                else None
            ),
            "eraf_structured_assignment_hard_weight": (
                self.policy_guard_eraf_loss_weights.structured_assignment_hard_weight
                if is_v9
                else None
            ),
            "eraf_multi_clause_consistency_weight": (
                self.policy_guard_eraf_loss_weights.multi_clause_consistency
                if is_v9
                else None
            ),
            "eraf_clause_tuple_assignment_weight": (
                self.policy_guard_eraf_loss_weights.clause_tuple_assignment
                if is_v9
                else None
            ),
            "eraf_clause_tuple_temperature": (
                self.policy_guard_eraf_loss_weights.clause_tuple_temperature
                if is_v9
                else None
            ),
            "eraf_clause_tuple_hard_weight": (
                self.policy_guard_eraf_loss_weights.clause_tuple_hard_weight
                if is_v9
                else None
            ),
            "eraf_clause_tuple_multi_consistency_weight": (
                self.policy_guard_eraf_loss_weights.clause_tuple_multi_consistency
                if is_v9
                else None
            ),
            "eraf_clause_activation_balance_weight": (
                self.policy_guard_eraf_loss_weights.clause_activation_balance
                if is_v9
                else None
            ),
            "eraf_clause_cardinality_weight": (
                self.policy_guard_eraf_loss_weights.clause_cardinality
                if is_v9
                else None
            ),
            "eraf_clause_worst_slot_weight": (
                self.policy_guard_eraf_loss_weights.clause_worst_slot
                if is_v9
                else None
            ),
            "eraf_clause_multi_group_weight": (
                self.policy_guard_eraf_loss_weights.clause_multi_group_weight
                if is_v9
                else None
            ),
            "eraf_clause_adapter_energy_weight": (
                self.policy_guard_eraf_loss_weights.clause_adapter_energy
                if is_v9
                else None
            ),
            "eraf_view_fusion_weight": (
                self.policy_guard_eraf_loss_weights.view_fusion
                if is_v9
                else None
            ),
            "eraf_view_fusion_energy_weight": (
                self.policy_guard_eraf_loss_weights.view_fusion_energy
                if is_v9
                else None
            ),
            "eraf_clause_scheduler_weight": (
                self.policy_guard_eraf_loss_weights.clause_scheduler
                if is_v9
                else None
            ),
            "eraf_clause_scheduler_energy_weight": (
                self.policy_guard_eraf_loss_weights.clause_scheduler_energy
                if is_v9
                else None
            ),
            "eraf_phase_rebinding_energy_weight": (
                self.policy_guard_eraf_loss_weights.phase_rebinding_energy
                if is_v9
                else None
            ),
            "eraf_phase_safe_memory_state_weight": (
                self.policy_guard_eraf_loss_weights.phase_safe_memory_state
                if is_v9
                else None
            ),
            "eraf_phase_safe_memory_scheduler_weight": (
                self.policy_guard_eraf_loss_weights.phase_safe_memory_scheduler
                if is_v9
                else None
            ),
            "eraf_phase_safe_memory_energy_weight": (
                self.policy_guard_eraf_loss_weights.phase_safe_memory_energy
                if is_v9
                else None
            ),
            "eraf_role_adapter_hidden_dim": (
                self.policy_guard_eraf_role_adapter_hidden_dim
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 4
                else None
            ),
            "eraf_role_adapter_trainable_scope": (
                eraf_role_adapter_trainable_scope
            ),
            "eraf_structured_role_adapter_hidden_dim": (
                self.policy_guard_eraf_structured_role_adapter_hidden_dim
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 5
                else None
            ),
            "eraf_balanced_role_adapter_hidden_dim": (
                self.policy_guard_eraf_balanced_role_adapter_hidden_dim
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 6
                else None
            ),
            "eraf_clause_activation_adapter_hidden_dim": (
                self.policy_guard_eraf_clause_activation_adapter_hidden_dim
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 9
                else None
            ),
            "eraf_clause_activation_residual_max_abs": (
                self.policy_guard_eraf_clause_activation_residual_max_abs
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 9
                else None
            ),
            "eraf_view_fusion_adapter_hidden_dim": (
                self.policy_guard_eraf_view_fusion_adapter_hidden_dim
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 10
                else None
            ),
            "eraf_view_fusion_residual_max_abs": (
                self.policy_guard_eraf_view_fusion_residual_max_abs
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 10
                else None
            ),
            "eraf_clause_scheduler_hidden_dim": (
                self.policy_guard_eraf_clause_scheduler_hidden_dim
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 10
                else None
            ),
            "eraf_clause_scheduler_residual_max_abs": (
                self.policy_guard_eraf_clause_scheduler_residual_max_abs
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 10
                else None
            ),
            "eraf_closed_loop_rebinding_hidden_dim": (
                self.policy_guard_eraf_closed_loop_rebinding_hidden_dim
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version == 13
                else None
            ),
            "eraf_closed_loop_query_residual_max_abs": (
                self.policy_guard_eraf_closed_loop_query_residual_max_abs
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version == 13
                else None
            ),
            "eraf_closed_loop_state_residual_max_abs": (
                self.policy_guard_eraf_closed_loop_state_residual_max_abs
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version == 13
                else None
            ),
            "eraf_phase_safe_memory_hidden_dim": (
                self.policy_guard_eraf_phase_safe_memory_hidden_dim
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 14
                else None
            ),
            "eraf_phase_safe_memory_state_count": (
                self.policy_guard_eraf_phase_safe_memory_state_count
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 14
                else None
            ),
            "eraf_phase_safe_memory_routing_residual_max_abs": (
                self.policy_guard_eraf_phase_safe_memory_routing_residual_max_abs
                if is_v9
                and self.policy_guard_eraf_grounding_objective_version >= 14
                else None
            ),
            "eraf_role_attention_preservation_weight": (
                self.policy_guard_eraf_loss_weights.role_attention_preservation
                if is_v9
                else None
            ),
            "eraf_role_position_preservation_weight": (
                self.policy_guard_eraf_loss_weights.role_position_preservation
                if is_v9
                else None
            ),
            "eraf_role_anchor_preservation_weight": (
                self.policy_guard_eraf_loss_weights.role_anchor_preservation
                if is_v9
                else None
            ),
            "eraf_role_relation_preservation_weight": (
                self.policy_guard_eraf_loss_weights.role_relation_preservation
                if is_v9
                else None
            ),
            "eraf_role_adapter_energy_weight": (
                self.policy_guard_eraf_loss_weights.role_adapter_energy
                if is_v9
                else None
            ),
            "eraf_entity_only": (
                self.policy_guard_eraf_entity_only if is_v9 else None
            ),
            "eraf_use_anchors": (
                self.policy_guard_eraf_use_anchors if is_v9 else None
            ),
        }

    @staticmethod
    def _cpu_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
        return {
            name: value.detach().to(device="cpu")
            for name, value in module.state_dict().items()
        }

    def _policy_guard_target_prototype_state_dict(
        self,
    ) -> dict[str, torch.Tensor]:
        """Persist V6's online teacher state for exact weight-only resume."""
        bank = self.policy_guard_target_prototype_bank
        if self.policy_guard_version != 6 or bank is None:
            raise RuntimeError("PGC v6 target prototype bank is unavailable.")
        return {
            "task_ids": bank.task_ids.detach().to(device="cpu", dtype=torch.long),
            "counts": bank.counts.detach().to(device="cpu", dtype=torch.long),
            "prototypes": bank.prototypes.detach().to(
                device="cpu", dtype=torch.float32
            ),
        }

    def _load_policy_guard_target_prototype_state(
        self, state: Any
    ) -> None:
        """Restore V6's training-only prototype bank without deployment drift."""
        bank = self.policy_guard_target_prototype_bank
        if self.policy_guard_version != 6 or bank is None:
            raise RuntimeError("PGC v6 target prototype bank is unavailable.")
        if not isinstance(state, dict):
            raise ValueError(
                "PGC v6 checkpoint is missing its persisted target prototype bank."
            )
        expected = {
            "task_ids": ((bank.num_slots,), torch.long),
            "counts": ((bank.num_slots,), torch.long),
            "prototypes": (
                (bank.num_slots, bank.feature_dim),
                torch.float32,
            ),
        }
        if set(state) != set(expected):
            raise ValueError(
                "PGC v6 target prototype bank keys are invalid: "
                f"{sorted(state)}."
            )
        normalized: dict[str, torch.Tensor] = {}
        for name, (shape, dtype) in expected.items():
            value = state[name]
            if not isinstance(value, torch.Tensor):
                raise ValueError(
                    f"PGC v6 target prototype bank {name!r} is not a tensor."
                )
            if tuple(value.shape) != tuple(shape) or value.dtype != dtype:
                raise ValueError(
                    f"PGC v6 target prototype bank {name!r} mismatch: "
                    f"shape={tuple(value.shape)} dtype={value.dtype}, "
                    f"expected_shape={tuple(shape)} expected_dtype={dtype}."
                )
            normalized[name] = value
        task_ids = normalized["task_ids"]
        counts = normalized["counts"]
        if bool((counts < 0).any()) or bool(((task_ids < 0) != (counts == 0)).any()):
            raise ValueError(
                "PGC v6 target prototype bank has inconsistent task IDs/counts."
            )
        active_ids = task_ids[task_ids >= 0]
        if active_ids.numel() != torch.unique(active_ids).numel():
            raise ValueError("PGC v6 target prototype bank contains duplicate task IDs.")
        with torch.no_grad():
            bank.task_ids.copy_(
                task_ids.to(device=bank.task_ids.device, dtype=bank.task_ids.dtype)
            )
            bank.counts.copy_(
                counts.to(device=bank.counts.device, dtype=bank.counts.dtype)
            )
            bank.prototypes.copy_(
                normalized["prototypes"].to(
                    device=bank.prototypes.device,
                    dtype=bank.prototypes.dtype,
                )
            )

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
            if not self.policy_guard_base_checkpoint:
                raise ValueError(
                    "PGC checkpoint cannot be saved before a protected base "
                    "checkpoint has been loaded."
                )
            payload = {
                "format": f"fastwam_policy_guard_v{self.policy_guard_version}",
                "base_checkpoint": self.policy_guard_base_checkpoint,
                "policy_guard": self._cpu_state_dict(
                    self.policy_guard_modules
                ),
                "architecture_metadata": self._policy_guard_metadata(),
                "step": step,
                "torch_dtype": str(self.torch_dtype),
            }
            if self.policy_guard_version == 6:
                payload["target_prototype_bank"] = (
                    self._policy_guard_target_prototype_state_dict()
                )
            if self.policy_guard_version <= 2:
                if self.policy_guard_action_expert is None:
                    raise RuntimeError(
                        "PGC Action Expert is unavailable for saving."
                    )
                if not self.lora_enabled:
                    raise ValueError(
                        "PGC v1/v2 checkpoints require action-only LoRA; "
                        "full Action-Expert checkpoints are disabled."
                    )
                if self.policy_guard_legacy_full_loaded:
                    raise ValueError(
                        "Cannot convert a legacy full-PGC checkpoint into a "
                        "partial LoRA checkpoint."
                    )
                action_adapter = (
                    self._policy_guard_action_adapter_state_dict()
                )
                if not action_adapter:
                    raise ValueError(
                        "PGC checkpoint has no Action-Expert adapter tensors."
                    )
                payload["counterfactual_action_adapter"] = action_adapter
                payload["counterfactual_lora_config"] = dict(
                    self.lora_config
                )
            elif (
                self.policy_guard_version == 9
                and self.policy_guard_eraf_grounding_objective_version >= 26
            ):
                if not self.lora_enabled:
                    raise ValueError(
                        "PGC V9.26 checkpoints require shared Video/Action LoRA."
                    )
                shared_adapter = self._lora_adapter_state_dict()
                if not shared_adapter:
                    raise ValueError(
                        "PGC V9.26 checkpoint has no shared Expert LoRA tensors."
                    )
                payload["eraf_shared_expert_lora"] = shared_adapter
                payload["eraf_shared_expert_lora_config"] = dict(
                    self.lora_config
                )
            elif self.lora_enabled or self.policy_guard_action_expert is not None:
                raise ValueError(
                    "PGC v3+ checkpoints must contain no Action-Expert copy or LoRA."
                )
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
        if not self.policy_guard_enabled:
            raise ValueError(
                "This checkpoint contains PGC weights; enable the matching "
                "`policy_guard` model config before loading it."
            )
        metadata = payload.get("architecture_metadata") or {}
        saved_policy_guard_version = int(
            metadata.get("policy_guard_version", -1)
        )
        saved_eraf_grounding_objective = (
            int(metadata.get("eraf_grounding_objective_version", 1))
            if saved_policy_guard_version == 9
            else None
        )
        saved_eraf_completion_only_memory = bool(
            metadata.get("eraf_completion_only_memory", False)
        )
        migrate_v5_to_target_binder = (
            saved_policy_guard_version == 5
            and int(self.policy_guard_version) in {6, 7}
        )
        migrate_v5_to_v8 = (
            saved_policy_guard_version == 5
            and int(self.policy_guard_version) == 8
        )
        migrate_v5_to_v9 = (
            saved_policy_guard_version == 5
            and int(self.policy_guard_version) == 9
        )
        if (
            migrate_v5_to_v9
            and self.policy_guard_eraf_initialization_contract
            != "exact_pgc_v5_sidecars"
        ):
            raise ValueError(
                "A released-base fresh ERAF model cannot import PGC-V5 "
                "sidecars. Load the released FastWAM checkpoint directly."
            )
        migrate_v9_to_role_adapter = (
            saved_policy_guard_version == 9
            and int(self.policy_guard_version) == 9
            and saved_eraf_grounding_objective == 2
            and self.policy_guard_eraf_grounding_objective_version == 4
            and self.policy_guard_eraf_training_stage == "grounding"
            and metadata.get("eraf_training_stage") == "grounding"
        )
        migrate_v9_to_structured_role_adapter = (
            saved_policy_guard_version == 9
            and int(self.policy_guard_version) == 9
            and saved_eraf_grounding_objective == 4
            and self.policy_guard_eraf_grounding_objective_version == 5
            and self.policy_guard_eraf_training_stage == "grounding"
            and metadata.get("eraf_training_stage") == "grounding"
        )
        migrate_v9_to_balanced_role_adapter = (
            saved_policy_guard_version == 9
            and int(self.policy_guard_version) == 9
            and saved_eraf_grounding_objective == 4
            and self.policy_guard_eraf_grounding_objective_version in {6, 7}
            and self.policy_guard_eraf_training_stage == "grounding"
            and metadata.get("eraf_training_stage") == "grounding"
        )
        migrate_v9_to_clause_activation_adapter = (
            saved_policy_guard_version == 9
            and int(self.policy_guard_version) == 9
            and saved_eraf_grounding_objective == 8
            and self.policy_guard_eraf_grounding_objective_version == 9
            and self.policy_guard_eraf_training_stage == "grounding"
            and metadata.get("eraf_training_stage") == "grounding"
        )
        migrate_v9_to_view_scheduler = (
            saved_policy_guard_version == 9
            and int(self.policy_guard_version) == 9
            and saved_eraf_grounding_objective == 9
            and self.policy_guard_eraf_grounding_objective_version == 10
            and self.policy_guard_eraf_training_stage == "grounding"
            and metadata.get("eraf_training_stage") == "grounding"
        )
        migrate_v9_to_exclusive_all_entity = (
            saved_policy_guard_version == 9
            and int(self.policy_guard_version) == 9
            and saved_eraf_grounding_objective == 10
            and self.policy_guard_eraf_grounding_objective_version == 11
            and self.policy_guard_eraf_training_stage == "grounding"
            and metadata.get("eraf_training_stage") == "grounding"
        )
        migrate_v9_to_clause_tuple = (
            saved_policy_guard_version == 9
            and int(self.policy_guard_version) == 9
            and saved_eraf_grounding_objective == 11
            and self.policy_guard_eraf_grounding_objective_version == 12
            and self.policy_guard_eraf_training_stage == "grounding"
            and metadata.get("eraf_training_stage") == "grounding"
        )
        migrate_v9_to_phase_rebinding = (
            saved_policy_guard_version == 9
            and int(self.policy_guard_version) == 9
            and saved_eraf_grounding_objective == 12
            and self.policy_guard_eraf_grounding_objective_version == 13
            and self.policy_guard_eraf_training_stage == "grounding"
            and metadata.get("eraf_training_stage") == "grounding"
        )
        migrate_v9_to_phase_safe_memory = (
            saved_policy_guard_version == 9
            and int(self.policy_guard_version) == 9
            and saved_eraf_grounding_objective == 12
            and self.policy_guard_eraf_grounding_objective_version == 14
            and self.policy_guard_eraf_training_stage == "grounding"
            and metadata.get("eraf_training_stage") == "grounding"
        )
        migrate_v914_to_v915_action_grounding = (
            saved_policy_guard_version == 9
            and int(self.policy_guard_version) == 9
            and saved_eraf_grounding_objective == 14
            and self.policy_guard_eraf_grounding_objective_version == 15
            and self.policy_guard_eraf_training_stage == "action"
            and metadata.get("eraf_training_stage") == "action"
            and saved_eraf_completion_only_memory
            and self.policy_guard_eraf_completion_only_memory
            and bool(metadata.get("eraf_action_joint_training", False))
            and self.policy_guard_eraf_action_joint_training
        )
        migrate_v915_to_v916_semantic_causal = (
            saved_policy_guard_version == 9
            and int(self.policy_guard_version) == 9
            and saved_eraf_grounding_objective == 15
            and self.policy_guard_eraf_grounding_objective_version == 16
            and self.policy_guard_eraf_training_stage == "action"
            and metadata.get("eraf_training_stage") == "action"
            and saved_eraf_completion_only_memory
            and self.policy_guard_eraf_completion_only_memory
            and bool(metadata.get("eraf_action_joint_training", False))
            and self.policy_guard_eraf_action_joint_training
        )
        migrate_v916_to_v917_direct_geometry = (
            saved_policy_guard_version == 9
            and int(self.policy_guard_version) == 9
            and saved_eraf_grounding_objective == 16
            and self.policy_guard_eraf_grounding_objective_version == 17
            and self.policy_guard_eraf_training_stage == "action"
            and metadata.get("eraf_training_stage") == "action"
            and saved_eraf_completion_only_memory
            and self.policy_guard_eraf_completion_only_memory
            and bool(metadata.get("eraf_action_joint_training", False))
            and self.policy_guard_eraf_action_joint_training
        )
        migrate_v917_to_v918_phase_residual = (
            saved_policy_guard_version == 9
            and int(self.policy_guard_version) == 9
            and saved_eraf_grounding_objective == 17
            and self.policy_guard_eraf_grounding_objective_version == 18
            and self.policy_guard_eraf_training_stage == "action"
            and metadata.get("eraf_training_stage") == "action"
            and saved_eraf_completion_only_memory
            and self.policy_guard_eraf_completion_only_memory
            and bool(metadata.get("eraf_action_joint_training", False))
            and self.policy_guard_eraf_action_joint_training
        )
        migrate_v918_to_v919_phase_servo = (
            saved_policy_guard_version == 9
            and int(self.policy_guard_version) == 9
            and saved_eraf_grounding_objective == 18
            and self.policy_guard_eraf_grounding_objective_version == 19
            and self.policy_guard_eraf_training_stage == "action"
            and metadata.get("eraf_training_stage") == "action"
            and saved_eraf_completion_only_memory
            and self.policy_guard_eraf_completion_only_memory
            and bool(metadata.get("eraf_action_joint_training", False))
            and self.policy_guard_eraf_action_joint_training
        )
        migrate_v919_to_v920_waypoint = (
            saved_policy_guard_version == 9
            and int(self.policy_guard_version) == 9
            and saved_eraf_grounding_objective == 19
            and self.policy_guard_eraf_grounding_objective_version == 20
            and self.policy_guard_eraf_training_stage == "action"
            and metadata.get("eraf_training_stage") == "action"
            and saved_eraf_completion_only_memory
            and self.policy_guard_eraf_completion_only_memory
            and bool(metadata.get("eraf_action_joint_training", False))
            and self.policy_guard_eraf_action_joint_training
        )
        migrate_v920_to_v921_expert_alignment = (
            saved_policy_guard_version == 9
            and int(self.policy_guard_version) == 9
            and saved_eraf_grounding_objective == 20
            and self.policy_guard_eraf_grounding_objective_version == 21
            and self.policy_guard_eraf_training_stage == "action"
            and metadata.get("eraf_training_stage") == "action"
            and saved_eraf_completion_only_memory
            and self.policy_guard_eraf_completion_only_memory
            and bool(metadata.get("eraf_action_joint_training", False))
            and self.policy_guard_eraf_action_joint_training
        )
        migrate_v921_to_v922_clause_ranking = (
            saved_policy_guard_version == 9
            and int(self.policy_guard_version) == 9
            and saved_eraf_grounding_objective == 21
            and self.policy_guard_eraf_grounding_objective_version == 22
            and self.policy_guard_eraf_training_stage == "action"
            and metadata.get("eraf_training_stage") == "action"
            and saved_eraf_completion_only_memory
            and self.policy_guard_eraf_completion_only_memory
            and bool(metadata.get("eraf_action_joint_training", False))
            and self.policy_guard_eraf_action_joint_training
        )
        migrate_v921_to_v923_alignment_preserving_clause = (
            saved_policy_guard_version == 9
            and int(self.policy_guard_version) == 9
            and saved_eraf_grounding_objective == 21
            and self.policy_guard_eraf_grounding_objective_version == 23
            and self.policy_guard_eraf_training_stage == "action"
            and metadata.get("eraf_training_stage") == "action"
            and saved_eraf_completion_only_memory
            and self.policy_guard_eraf_completion_only_memory
            and bool(metadata.get("eraf_action_joint_training", False))
            and self.policy_guard_eraf_action_joint_training
        )
        migrate_v921_to_v924_isolated_clause_residual = (
            saved_policy_guard_version == 9
            and int(self.policy_guard_version) == 9
            and saved_eraf_grounding_objective == 21
            and self.policy_guard_eraf_grounding_objective_version == 24
            and self.policy_guard_eraf_training_stage == "action"
            and metadata.get("eraf_training_stage") == "action"
            and saved_eraf_completion_only_memory
            and self.policy_guard_eraf_completion_only_memory
            and bool(metadata.get("eraf_action_joint_training", False))
            and self.policy_guard_eraf_action_joint_training
        )
        migrate_v924_to_v925_action_context = (
            saved_policy_guard_version == 9
            and int(self.policy_guard_version) == 9
            and saved_eraf_grounding_objective == 24
            and self.policy_guard_eraf_grounding_objective_version == 25
            and self.policy_guard_eraf_training_stage == "action"
            and metadata.get("eraf_training_stage") == "action"
            and saved_eraf_completion_only_memory
            and self.policy_guard_eraf_completion_only_memory
            and bool(metadata.get("eraf_action_joint_training", False))
            and self.policy_guard_eraf_action_joint_training
        )
        migrate_v925_to_v926_shared_expert_lora = (
            saved_policy_guard_version == 9
            and int(self.policy_guard_version) == 9
            and saved_eraf_grounding_objective == 25
            and self.policy_guard_eraf_grounding_objective_version == 26
            and self.policy_guard_eraf_training_stage == "action"
            and metadata.get("eraf_training_stage") == "action"
            and saved_eraf_completion_only_memory
            and self.policy_guard_eraf_completion_only_memory
            and bool(metadata.get("eraf_action_joint_training", False))
            and self.policy_guard_eraf_action_joint_training
        )
        if migrate_v914_to_v915_action_grounding:
            expected_v914_warm_start = {
                "eraf_action_joint_contract": (
                    "frozen_eraf_perception_plus_action_bridge_and_proposal"
                ),
                "eraf_action_trainable_scope": (
                    "base_query_projection_relation_attention_query_"
                    "embedding_delta_plus_action_chunk_proposal"
                ),
                "eraf_role_adapter_trainable_scope": (
                    "frozen_eraf_perception_action_bridge_plus_proposal"
                ),
                "eraf_policy_state_contract": (
                    "monotonic_completed_bitset_no_pending_holding_retry_"
                    "recurrence"
                ),
            }
            for name, expected in expected_v914_warm_start.items():
                if metadata.get(name) != expected:
                    raise ValueError(
                        "PGC V9.15 requires the completed V9.14 action "
                        f"checkpoint contract: {name}={metadata.get(name)!r}, "
                        f"expected={expected!r}."
                    )
        if migrate_v915_to_v916_semantic_causal:
            expected_v915_warm_start = {
                "eraf_action_joint_contract": (
                    "frozen_eraf_perception_plus_phase_conditioned_geometry_"
                    "bridge_legacy_bridge_and_proposal"
                ),
                "eraf_action_trainable_scope": (
                    "phase_conditioned_subject_reference_anchor_action_bridge_"
                    "plus_legacy_bridge_and_action_chunk_proposal"
                ),
                "eraf_role_adapter_trainable_scope": (
                    "frozen_eraf_perception_action_bridge_plus_proposal"
                ),
                "eraf_policy_state_contract": (
                    "monotonic_completed_bitset_no_pending_holding_retry_"
                    "recurrence"
                ),
            }
            for name, expected in expected_v915_warm_start.items():
                if metadata.get(name) != expected:
                    raise ValueError(
                        "PGC V9.16 requires an exact completed V9.15 action "
                        f"checkpoint: {name}={metadata.get(name)!r}, "
                        f"expected={expected!r}."
                    )
        if migrate_v916_to_v917_direct_geometry:
            expected_v916_warm_start = {
                "eraf_action_joint_contract": (
                    "frozen_eraf_perception_proposal_and_legacy_bridge_plus_"
                    "semantic_causal_action_grounding_bridge"
                ),
                "eraf_action_trainable_scope": (
                    "semantic_causal_action_grounding_bridge_only"
                ),
                "eraf_role_adapter_trainable_scope": (
                    "semantic_causal_action_grounding_bridge_only"
                ),
                "eraf_policy_state_contract": (
                    "monotonic_completed_bitset_no_pending_holding_retry_"
                    "recurrence"
                ),
            }
            for name, expected in expected_v916_warm_start.items():
                if metadata.get(name) != expected:
                    raise ValueError(
                        "PGC V9.17 requires an exact completed V9.16 action "
                        f"checkpoint: {name}={metadata.get(name)!r}, "
                        f"expected={expected!r}."
                    )
        if migrate_v917_to_v918_phase_residual:
            expected_v917_warm_start = {
                "eraf_action_joint_contract": (
                    "frozen_eraf_v916_bridge_and_proposal_plus_direct_"
                    "eef_relative_geometry_action_adapter"
                ),
                "eraf_action_trainable_scope": (
                    "phase_conditioned_relative_geometry_action_adapter_only"
                ),
                "eraf_role_adapter_trainable_scope": (
                    "phase_conditioned_relative_geometry_action_adapter_only"
                ),
                "eraf_action_geometry_contract": (
                    "phase_selected_eef_relative_grasp_goal_interaction_"
                    "relation_direct_bounded_action_residual_zero_init_v9_16_exact"
                ),
                "eraf_policy_state_contract": (
                    "monotonic_completed_bitset_no_pending_holding_retry_"
                    "recurrence"
                ),
            }
            for name, expected in expected_v917_warm_start.items():
                if metadata.get(name) != expected:
                    raise ValueError(
                        "PGC V9.18 requires an exact completed V9.17 action "
                        f"checkpoint: {name}={metadata.get(name)!r}, "
                        f"expected={expected!r}."
                    )
        if migrate_v918_to_v919_phase_servo:
            expected_v918_warm_start = {
                "eraf_action_joint_contract": (
                    "frozen_eraf_v917_stack_plus_phase_balanced_direct_"
                    "geometry_residual_imitation"
                ),
                "eraf_action_trainable_scope": (
                    "phase_conditioned_geometry_adapter_only_with_phase_"
                    "balanced_residual_imitation"
                ),
                "eraf_role_adapter_trainable_scope": (
                    "phase_conditioned_geometry_adapter_only_with_phase_"
                    "balanced_residual_imitation"
                ),
                "eraf_action_phase_residual_contract": (
                    "phase_balanced_bounded_expert_minus_frozen_v9_17_candidate_"
                    "prefix_residual_imitation"
                ),
                "eraf_policy_state_contract": (
                    "monotonic_completed_bitset_no_pending_holding_retry_"
                    "recurrence"
                ),
            }
            for name, expected in expected_v918_warm_start.items():
                if metadata.get(name) != expected:
                    raise ValueError(
                        "PGC V9.19 requires an exact completed V9.18 action "
                        f"checkpoint: {name}={metadata.get(name)!r}, "
                        f"expected={expected!r}."
                    )
        if migrate_v919_to_v920_waypoint:
            expected_v919_warm_start = {
                "eraf_action_joint_contract": (
                    "frozen_v918_stack_plus_hard_clause_phase_"
                    "direction_preserving_servo"
                ),
                "eraf_action_trainable_scope": "hard_routed_phase_servo_only",
                "eraf_role_adapter_trainable_scope": "hard_routed_phase_servo_only",
                "eraf_action_phase_servo_contract": (
                    "hard_single_clause_explicit_affine_eef_phase_specific_"
                    "positive_cartesian_gain_with_legacy_suppression"
                ),
                "eraf_policy_state_contract": (
                    "monotonic_completed_bitset_no_pending_holding_retry_recurrence"
                ),
            }
            for name, expected in expected_v919_warm_start.items():
                if metadata.get(name) != expected:
                    raise ValueError(
                        "PGC V9.20 requires an exact completed V9.19 action "
                        f"checkpoint: {name}={metadata.get(name)!r}, "
                        f"expected={expected!r}."
                    )
        if migrate_v920_to_v921_expert_alignment:
            expected_v920_warm_start = {
                "eraf_action_joint_contract": (
                    "frozen_v919_stack_plus_phase_compatible_local_"
                    "waypoint_vector_field"
                ),
                "eraf_action_trainable_scope": (
                    "phase_compatible_local_waypoint_adapter_only"
                ),
                "eraf_role_adapter_trainable_scope": (
                    "phase_compatible_local_waypoint_adapter_only"
                ),
                "eraf_action_waypoint_contract": (
                    "hard_clause_phase_compatible_positive_progress_local_"
                    "tangent_waypoint_with_privileged_training_only_"
                    "compatibility_labels"
                ),
                "eraf_policy_state_contract": (
                    "monotonic_completed_bitset_no_pending_holding_retry_"
                    "recurrence"
                ),
            }
            for name, expected in expected_v920_warm_start.items():
                if metadata.get(name) != expected:
                    raise ValueError(
                        "PGC V9.21 requires an exact completed V9.20 action "
                        f"checkpoint: {name}={metadata.get(name)!r}, "
                        f"expected={expected!r}."
                    )
        if migrate_v921_to_v922_clause_ranking:
            expected_v921_warm_start = {
                "eraf_action_joint_contract": (
                    "frozen_v920_stack_plus_phase_specific_privileged_"
                    "expert_prefix_residual_alignment"
                ),
                "eraf_action_trainable_scope": (
                    "phase_specific_privileged_expert_residual_adapter_only"
                ),
                "eraf_role_adapter_trainable_scope": (
                    "phase_specific_privileged_expert_residual_adapter_only"
                ),
                "eraf_action_expert_alignment_contract": (
                    "training_only_privileged_phase_anchor_teacher_plus_"
                    "deployed_full_action_prefix_residual_and_semantic_"
                    "causal_ranking"
                ),
                "eraf_policy_state_contract": (
                    "monotonic_completed_bitset_no_pending_holding_retry_"
                    "recurrence"
                ),
            }
            for name, expected in expected_v921_warm_start.items():
                if metadata.get(name) != expected:
                    raise ValueError(
                        "PGC V9.22 requires an exact completed V9.21 action "
                        f"checkpoint: {name}={metadata.get(name)!r}, "
                        f"expected={expected!r}."
                    )
        if migrate_v921_to_v923_alignment_preserving_clause:
            expected_v921_warm_start = {
                "eraf_action_joint_contract": (
                    "frozen_v920_stack_plus_phase_specific_privileged_"
                    "expert_prefix_residual_alignment"
                ),
                "eraf_action_trainable_scope": (
                    "phase_specific_privileged_expert_residual_adapter_only"
                ),
                "eraf_role_adapter_trainable_scope": (
                    "phase_specific_privileged_expert_residual_adapter_only"
                ),
                "eraf_action_expert_alignment_contract": (
                    "training_only_privileged_phase_anchor_teacher_plus_"
                    "deployed_full_action_prefix_residual_and_semantic_"
                    "causal_ranking"
                ),
                "eraf_policy_state_contract": (
                    "monotonic_completed_bitset_no_pending_holding_retry_"
                    "recurrence"
                ),
            }
            for name, expected in expected_v921_warm_start.items():
                if metadata.get(name) != expected:
                    raise ValueError(
                        "PGC V9.23 requires an exact completed V9.21 action "
                        f"checkpoint: {name}={metadata.get(name)!r}, "
                        f"expected={expected!r}."
                    )
        if migrate_v921_to_v924_isolated_clause_residual:
            expected_v921_warm_start = {
                "eraf_action_joint_contract": (
                    "frozen_v920_stack_plus_phase_specific_privileged_"
                    "expert_prefix_residual_alignment"
                ),
                "eraf_action_trainable_scope": (
                    "phase_specific_privileged_expert_residual_adapter_only"
                ),
                "eraf_role_adapter_trainable_scope": (
                    "phase_specific_privileged_expert_residual_adapter_only"
                ),
                "eraf_action_expert_alignment_contract": (
                    "training_only_privileged_phase_anchor_teacher_plus_"
                    "deployed_full_action_prefix_residual_and_semantic_"
                    "causal_ranking"
                ),
                "eraf_policy_state_contract": (
                    "monotonic_completed_bitset_no_pending_holding_retry_"
                    "recurrence"
                ),
            }
            for name, expected in expected_v921_warm_start.items():
                if metadata.get(name) != expected:
                    raise ValueError(
                        "PGC V9.24 requires an exact completed V9.21 action "
                        f"checkpoint: {name}={metadata.get(name)!r}, "
                        f"expected={expected!r}."
                    )
        if migrate_v924_to_v925_action_context:
            expected_v924_warm_start = {
                "eraf_action_joint_contract": (
                    "frozen_v921_expert_adapter_plus_isolated_clause_"
                    "semantic_retention_residual"
                ),
                "eraf_action_trainable_scope": (
                    "clause_semantic_retention_residual_only"
                ),
                "eraf_role_adapter_trainable_scope": (
                    "clause_semantic_retention_residual_only"
                ),
                "eraf_policy_state_contract": (
                    "monotonic_completed_bitset_no_pending_holding_retry_"
                    "recurrence"
                ),
            }
            for name, expected in expected_v924_warm_start.items():
                if metadata.get(name) != expected:
                    raise ValueError(
                        "PGC V9.25 requires an exact completed V9.24 action "
                        f"checkpoint: {name}={metadata.get(name)!r}, "
                        f"expected={expected!r}."
                    )
        if migrate_v925_to_v926_shared_expert_lora:
            expected_v925_warm_start = {
                "eraf_action_joint_contract": (
                    "frozen_eraf_and_shared_action_expert_plus_internal_"
                    "context_injector_no_post_action_residual"
                ),
                "eraf_action_trainable_scope": (
                    "eraf_action_context_injector_only"
                ),
                "eraf_role_adapter_trainable_scope": (
                    "eraf_action_context_injector_only"
                ),
                "eraf_action_context_injection_contract": (
                    "append_bounded_eraf_tokens_to_shared_action_expert_"
                    "context_at_every_denoising_step_no_post_action_residual"
                ),
                "eraf_post_action_residual_active": False,
                "eraf_policy_state_contract": (
                    "monotonic_completed_bitset_no_pending_holding_retry_"
                    "recurrence"
                ),
            }
            for name, expected in expected_v925_warm_start.items():
                if metadata.get(name) != expected:
                    raise ValueError(
                        "PGC V9.26 requires an exact completed V9.25 action "
                        f"checkpoint: {name}={metadata.get(name)!r}, "
                        f"expected={expected!r}."
                    )
        if migrate_v9_to_exclusive_all_entity:
            expected_v99_warm_start = {
                "eraf_view_fusion_contract": (
                    "per_view_local_attention_visibility_gated_zero_init_residual"
                ),
                "eraf_clause_scheduler_contract": (
                    "first_active_unfinished_predicate_zero_init_residual_route"
                ),
                "eraf_role_adapter_trainable_scope": (
                    "clause_activation_plus_balanced_role_plus_visibility_gated_"
                    "view_fusion_plus_unfinished_clause_scheduler"
                ),
            }
            for name, expected in expected_v99_warm_start.items():
                if metadata.get(name) != expected:
                    raise ValueError(
                        "PGC V9.10 requires the completed V9.9 checkpoint "
                        f"contract: {name}={metadata.get(name)!r}, "
                        f"expected={expected!r}."
                    )
        if migrate_v9_to_clause_tuple:
            expected_v910_warm_start = {
                "eraf_all_entity_role_contract": (
                    "exclusive_evidence_same_state_all_entity_"
                    "bipartite_assignment"
                ),
                "eraf_multi_clause_gate_contract": (
                    "semantic_exact_with_exclusive_role_evidence"
                ),
                "eraf_role_adapter_trainable_scope": (
                    "exclusive_all_entity_balanced_visual_role_"
                    "binding_adapter_only"
                ),
            }
            for name, expected in expected_v910_warm_start.items():
                if metadata.get(name) != expected:
                    raise ValueError(
                        "PGC V9.11 requires the completed V9.10 checkpoint "
                        f"contract: {name}={metadata.get(name)!r}, "
                        f"expected={expected!r}."
                    )
        if migrate_v9_to_phase_rebinding:
            expected_v911_warm_start = {
                "eraf_clause_tuple_contract": (
                    "exclusive_same_state_subject_predicate_reference_assignment"
                ),
                "eraf_role_adapter_trainable_scope": (
                    "audited_hard_clause_tuple_balanced_visual_"
                    "role_binding_adapter_only"
                ),
            }
            for name, expected in expected_v911_warm_start.items():
                if metadata.get(name) != expected:
                    raise ValueError(
                        "PGC V9.12 requires the completed V9.11 checkpoint "
                        f"contract: {name}={metadata.get(name)!r}, "
                        f"expected={expected!r}."
                    )
        if migrate_v9_to_phase_safe_memory:
            expected_v911_warm_start = {
                "eraf_clause_tuple_contract": (
                    "exclusive_same_state_subject_predicate_reference_assignment"
                ),
                "eraf_role_adapter_trainable_scope": (
                    "audited_hard_clause_tuple_balanced_visual_"
                    "role_binding_adapter_only"
                ),
            }
            for name, expected in expected_v911_warm_start.items():
                if metadata.get(name) != expected:
                    raise ValueError(
                        "PGC V9.13 requires the completed V9.11 checkpoint "
                        f"contract: {name}={metadata.get(name)!r}, "
                        f"expected={expected!r}."
                    )
        if (
            saved_policy_guard_version != int(self.policy_guard_version)
            and not migrate_v5_to_target_binder
            and not migrate_v5_to_v8
            and not migrate_v5_to_v9
        ):
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
        shared_expert_lora = payload.get("eraf_shared_expert_lora")
        shared_expert_lora_config = payload.get(
            "eraf_shared_expert_lora_config"
        )
        if int(saved_eraf_grounding_objective or 1) >= 26:
            if not isinstance(shared_expert_lora_config, dict):
                raise ValueError(
                    "PGC V9.26 checkpoint is missing its shared Expert LoRA "
                    "configuration."
                )
            if not isinstance(shared_expert_lora, dict) or not shared_expert_lora:
                raise ValueError(
                    "PGC V9.26 checkpoint has no shared Expert LoRA tensors."
                )
            self.configure_lora(shared_expert_lora_config)
        elif shared_expert_lora is not None or shared_expert_lora_config is not None:
            raise ValueError(
                "A pre-V9.26 PGC checkpoint unexpectedly contains shared "
                "Expert LoRA."
            )
        guard_state = payload.get("policy_guard")
        if not isinstance(guard_state, dict) or not guard_state:
            raise ValueError("PGC checkpoint has no Goal-Graph/Verifier state.")
        target_prototype_state = payload.get("target_prototype_bank")
        if saved_policy_guard_version == 6:
            if metadata.get("target_prototype_bank_persisted") is not True:
                raise ValueError(
                    "PGC v6 checkpoint does not declare a persisted target "
                    "prototype bank."
                )
            if not isinstance(target_prototype_state, dict):
                raise ValueError(
                    "PGC v6 checkpoint is missing its target prototype bank."
                )
        elif target_prototype_state is not None:
            raise ValueError(
                "A non-v6 PGC checkpoint unexpectedly contains a target "
                "prototype bank."
            )

        if saved_policy_guard_version >= 3:
            expected_tuning = (
                "native_and_counterfactual_world_action_joint_lora"
                if saved_policy_guard_version == 9
                and int(saved_eraf_grounding_objective or 1) >= 26
                else "counterfactual_only_internal_action_expert_context_conditioning"
                if saved_policy_guard_version == 9
                and int(saved_eraf_grounding_objective or 1) >= 25
                else "entity_relation_affordance_grounded_paired_action_residual"
                if saved_policy_guard_version == 9
                else (
                    "closed_loop_replay_verified_target_acquisition_residual"
                    if saved_policy_guard_version == 8
                    else (
                        "object_token_mask_grounded_paired_action_residual"
                        if saved_policy_guard_version == 7
                        else (
                            "visual_target_bottleneck_paired_action_residual"
                            if saved_policy_guard_version == 6
                            else (
                                "paired_language_prefix_aligned_action_residual"
                                if saved_policy_guard_version >= 5
                                else (
                                    "rollout_aligned_final_action_residual"
                                    if saved_policy_guard_version >= 4
                                    else "bounded_velocity_residual"
                                )
                            )
                        )
                    )
                )
            )
            expected_protection = (
                "single_eraf_path_no_candidate_gate"
                if saved_policy_guard_version == 9
                and int(saved_eraf_grounding_objective or 1) >= 26
                else "single_immutable_base_plus_conservative_hard_gate"
            )
            if metadata.get("counterfactual_tuning") != expected_tuning or (
                metadata.get("policy_protection") != expected_protection
            ):
                raise ValueError(
                    f"PGC v{saved_policy_guard_version} checkpoint does not "
                    "declare its single-Base protection/tuning contract."
                )
            if self.policy_guard_action_expert is not None or (
                self.lora_enabled
                and int(saved_eraf_grounding_objective or 1) < 26
                and not migrate_v925_to_v926_shared_expert_lora
            ):
                raise ValueError(
                    "PGC v3+ loading requires an adapter-free model with no "
                    "independent Action Expert, except the V9.26 shared "
                    "Expert-LoRA path."
                )
            if (
                action_adapter is not None
                or legacy_action_state is not None
                or payload.get("counterfactual_lora_config") is not None
            ):
                raise ValueError(
                    "PGC v3+ checkpoints must not contain counterfactual "
                    "Action-Expert or LoRA tensors."
                )
            if saved_policy_guard_version >= 4:
                proposal_module = self.policy_guard_modules[
                    "action_chunk_proposal"
                ]
                verifier_module = self.policy_guard_modules["verifier"]
                architecture_fields = {
                    "proposal_hidden_dim": int(proposal_module.hidden_dim),
                    "proposal_num_heads": int(proposal_module.num_heads),
                    "proposal_num_layers": int(proposal_module.num_layers),
                    "verifier_hidden_dim": int(verifier_module.hidden_dim),
                    "verifier_num_heads": int(verifier_module.num_heads),
                    "verifier_num_layers": int(verifier_module.num_layers),
                }
                if saved_policy_guard_version in {6, 7}:
                    target_binder = self.policy_guard_modules["target_binder"]
                    architecture_fields.update(
                        {
                            "target_binding_hidden_dim": int(
                                target_binder.hidden_dim
                            ),
                            "target_binding_num_heads": int(
                                target_binder.num_heads
                            ),
                            "target_binding_projection_dim": int(
                                target_binder.projection_dim
                            ),
                        }
                    )
                    if saved_policy_guard_version == 6:
                        architecture_fields[
                            "target_binding_prototype_slots"
                        ] = int(
                            self.policy_guard_target_binding_prototype_slots
                        )
                    else:
                        architecture_fields.update(
                            {
                                "target_binding_num_object_tokens": int(
                                    target_binder.num_object_tokens
                                ),
                                "target_binding_camera_count": int(
                                    target_binder.camera_count
                                ),
                            }
                        )
                elif saved_policy_guard_version == 9:
                    eraf = self.policy_guard_modules[
                        "entity_relation_affordance"
                    ]
                    architecture_fields.update(
                        {
                            "eraf_hidden_dim": int(eraf.hidden_dim),
                            "eraf_num_heads": int(eraf.num_heads),
                            "eraf_max_clauses": int(eraf.max_clauses),
                            "eraf_camera_count": int(eraf.camera_count),
                        }
                    )
                    if int(saved_eraf_grounding_objective or 1) >= 4:
                        architecture_fields["eraf_role_adapter_hidden_dim"] = int(
                            eraf.role_adapter_hidden_dim
                        )
                for metadata_name, expected_value in architecture_fields.items():
                    try:
                        saved_value = int(metadata[metadata_name])
                    except (KeyError, TypeError, ValueError) as exc:
                        raise ValueError(
                            "PGC v4 checkpoint is missing valid architecture "
                            f"value {metadata_name!r}."
                        ) from exc
                    if saved_value != expected_value:
                        raise ValueError(
                            f"PGC v4 {metadata_name} mismatch: "
                            f"checkpoint={saved_value}, model={expected_value}."
                        )
                try:
                    action_chunk_caps = [
                        float(value)
                        for value in metadata["action_chunk_residual_max_abs"]
                    ]
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        "PGC v4 checkpoint has invalid action residual caps."
                    ) from exc
                if (
                    len(action_chunk_caps) != int(proposal_module.action_dim)
                    or any(value <= 0 for value in action_chunk_caps)
                ):
                    raise ValueError(
                        "PGC v4 checkpoint action residual caps must be positive "
                        "and match action_dim."
                    )
                scalar_fields = {
                    "action_gripper_weight": (
                        "policy_guard_action_gripper_weight",
                        float,
                    ),
                    "advantage_temperature": (
                        "policy_guard_advantage_temperature",
                        float,
                    ),
                    "advantage_clip": (
                        "policy_guard_advantage_clip",
                        float,
                    ),
                    "candidate_max_saturation_fraction": (
                        "policy_guard_candidate_max_saturation_fraction",
                        float,
                    ),
                    "candidate_max_delta_rms": (
                        "policy_guard_candidate_max_delta_rms",
                        float,
                    ),
                    "rollout_num_inference_steps": (
                        "policy_guard_rollout_num_inference_steps",
                        int,
                    ),
                }
                for metadata_name, (attribute_name, cast) in scalar_fields.items():
                    value = metadata.get(metadata_name)
                    if value is None:
                        raise ValueError(
                            "PGC v4 checkpoint is missing architecture value "
                            f"{metadata_name!r}."
                        )
                    setattr(self, attribute_name, cast(value))
                if self.policy_guard_rollout_num_inference_steps <= 0:
                    raise ValueError("PGC v4 checkpoint has invalid rollout steps.")
                if min(
                    self.policy_guard_action_gripper_weight,
                    self.policy_guard_advantage_temperature,
                    self.policy_guard_advantage_clip,
                    self.policy_guard_candidate_max_delta_rms,
                ) <= 0:
                    raise ValueError(
                        "PGC v4 checkpoint has non-positive scalar settings."
                    )
                if not (
                    0.0
                    <= self.policy_guard_candidate_max_saturation_fraction
                    <= 1.0
                ):
                    raise ValueError(
                        "PGC v4 checkpoint has invalid candidate saturation guard."
                    )
                if saved_policy_guard_version >= 5:
                    v5_scalar_fields = {
                        "execution_prefix_steps": (
                            "policy_guard_execution_prefix_steps",
                            int,
                        ),
                        "suffix_loss_weight": (
                            "policy_guard_suffix_loss_weight",
                            float,
                        ),
                        "same_state_source_zero_weight": (
                            "policy_guard_same_state_source_zero_weight",
                            float,
                        ),
                        "goal_separation_weight": (
                            "policy_guard_goal_separation_weight",
                            float,
                        ),
                        "goal_separation_margin": (
                            "policy_guard_goal_separation_margin",
                            float,
                        ),
                        "residual_separation_weight": (
                            "policy_guard_residual_separation_weight",
                            float,
                        ),
                        "residual_separation_margin": (
                            "policy_guard_residual_separation_margin",
                            float,
                        ),
                        "verifier_wrong_language_weight": (
                            "policy_guard_verifier_wrong_language_weight",
                            float,
                        ),
                        "verifier_bad_candidate_weight": (
                            "policy_guard_verifier_bad_candidate_weight",
                            float,
                        ),
                    }
                    for metadata_name, (
                        attribute_name,
                        cast,
                    ) in v5_scalar_fields.items():
                        value = metadata.get(metadata_name)
                        if value is None:
                            raise ValueError(
                                "PGC v5 checkpoint is missing training/deployment "
                                f"value {metadata_name!r}."
                            )
                        setattr(self, attribute_name, cast(value))
                    if self.policy_guard_execution_prefix_steps <= 0:
                        raise ValueError(
                            "PGC v5 checkpoint has invalid execution prefix."
                        )
                    if not 0.0 <= self.policy_guard_suffix_loss_weight <= 1.0:
                        raise ValueError(
                            "PGC v5 checkpoint has invalid suffix loss weight."
                        )
                    if min(
                        self.policy_guard_same_state_source_zero_weight,
                        self.policy_guard_goal_separation_weight,
                        self.policy_guard_goal_separation_margin,
                        self.policy_guard_residual_separation_weight,
                        self.policy_guard_residual_separation_margin,
                        self.policy_guard_verifier_wrong_language_weight,
                        self.policy_guard_verifier_bad_candidate_weight,
                    ) < 0:
                        raise ValueError(
                            "PGC v5 checkpoint has negative paired-language weights."
                        )
                    if saved_policy_guard_version == 6:
                        v6_scalar_fields = {
                            "target_binding_temperature": (
                                "policy_guard_target_binding_temperature",
                                float,
                            ),
                            "target_binding_teacher_topk": (
                                "policy_guard_target_binding_teacher_topk",
                                float,
                            ),
                            "target_binding_teacher_temperature": (
                                "policy_guard_target_binding_teacher_temperature",
                                float,
                            ),
                            "target_binding_prototype_momentum": (
                                "policy_guard_target_binding_prototype_momentum",
                                float,
                            ),
                            "target_binding_prototype_temperature": (
                                "policy_guard_target_binding_prototype_temperature",
                                float,
                            ),
                            "target_binding_prototype_topk": (
                                "policy_guard_target_binding_prototype_topk",
                                float,
                            ),
                            "target_binding_action_start_step": (
                                "policy_guard_target_binding_action_start_step",
                                int,
                            ),
                            "target_binding_action_ramp_steps": (
                                "policy_guard_target_binding_action_ramp_steps",
                                int,
                            ),
                            "target_binding_interaction_weight": (
                                "policy_guard_target_binding_interaction_weight",
                                float,
                            ),
                            "target_binding_prototype_weight": (
                                "policy_guard_target_binding_prototype_weight",
                                float,
                            ),
                            "target_binding_source_weight": (
                                "policy_guard_target_binding_source_weight",
                                float,
                            ),
                            "target_binding_hard_negative_weight": (
                                "policy_guard_target_binding_hard_negative_weight",
                                float,
                            ),
                            "target_binding_separation_weight": (
                                "policy_guard_target_binding_separation_weight",
                                float,
                            ),
                            "target_binding_hard_negative_margin": (
                                "policy_guard_target_binding_hard_negative_margin",
                                float,
                            ),
                            "target_binding_separation_margin": (
                                "policy_guard_target_binding_separation_margin",
                                float,
                            ),
                        }
                        if (
                            metadata.get("target_binding_bottleneck")
                            != "visual_only_no_direct_language_residual"
                        ):
                            raise ValueError(
                                "PGC v6 checkpoint does not declare its visual-only "
                                "target bottleneck."
                            )
                        if (
                            metadata.get("target_binding_visual_source")
                            != "pre_dit_language_neutral_current_frame"
                        ):
                            raise ValueError(
                                "PGC v6 checkpoint does not use language-neutral "
                                "pre-DiT visual patches."
                            )
                        for metadata_name, (
                            attribute_name,
                            cast,
                        ) in v6_scalar_fields.items():
                            value = metadata.get(metadata_name)
                            if value is None:
                                raise ValueError(
                                    "PGC v6 checkpoint is missing target-binding "
                                    f"value {metadata_name!r}."
                                )
                            setattr(self, attribute_name, cast(value))
                        if min(
                            self.policy_guard_target_binding_temperature,
                            self.policy_guard_target_binding_teacher_temperature,
                            self.policy_guard_target_binding_prototype_temperature,
                        ) <= 0:
                            raise ValueError(
                                "PGC v6 checkpoint has non-positive target-binding "
                                "temperatures."
                            )
                        if not (
                            0.0
                            < self.policy_guard_target_binding_teacher_topk
                            <= 1.0
                            and 0.0
                            < self.policy_guard_target_binding_prototype_topk
                            <= 1.0
                        ):
                            raise ValueError(
                                "PGC v6 checkpoint has invalid target-binding top-k."
                            )
                        if not (
                            0.0
                            <= self.policy_guard_target_binding_prototype_momentum
                            < 1.0
                        ):
                            raise ValueError(
                                "PGC v6 checkpoint has invalid prototype momentum."
                            )
                        if min(
                            self.policy_guard_target_binding_interaction_weight,
                            self.policy_guard_target_binding_prototype_weight,
                            self.policy_guard_target_binding_source_weight,
                            self.policy_guard_target_binding_hard_negative_weight,
                            self.policy_guard_target_binding_separation_weight,
                            self.policy_guard_target_binding_hard_negative_margin,
                            self.policy_guard_target_binding_separation_margin,
                        ) < 0:
                            raise ValueError(
                                "PGC v6 checkpoint has negative target-binding "
                                "weights or margins."
                            )
                        if (
                            self.policy_guard_target_binding_action_start_step < 0
                            or self.policy_guard_target_binding_action_ramp_steps < 0
                        ):
                            raise ValueError(
                                "PGC v6 checkpoint has invalid target-binding "
                                "action schedule."
                            )
                        target_binder = self.policy_guard_modules[
                            "target_binder"
                        ]
                        target_binder.temperature = (
                            self.policy_guard_target_binding_temperature
                        )
                        prototype_bank = self.policy_guard_target_prototype_bank
                        if prototype_bank is None:
                            raise RuntimeError(
                                "PGC v6 target prototype bank is unavailable."
                            )
                        prototype_bank.momentum = (
                            self.policy_guard_target_binding_prototype_momentum
                        )
                        prototype_bank.temperature = (
                            self.policy_guard_target_binding_prototype_temperature
                        )
                        prototype_bank.topk_fraction = (
                            self.policy_guard_target_binding_prototype_topk
                        )
                    elif saved_policy_guard_version == 7:
                        if (
                            metadata.get("target_binding_bottleneck")
                            != "spatial_object_tokens_no_direct_language_residual"
                        ):
                            raise ValueError(
                                "PGC v7 checkpoint does not declare its spatial "
                                "object-token bottleneck."
                            )
                        if (
                            metadata.get("target_binding_visual_source")
                            != "pre_dit_language_neutral_current_frame"
                        ):
                            raise ValueError(
                                "PGC v7 checkpoint does not use language-neutral "
                                "current-frame visual patches."
                            )
                        if (
                            metadata.get("target_mask_supervision")
                            != "robosuite_element_current_frame_training_only"
                        ):
                            raise ValueError(
                                "PGC v7 checkpoint does not declare explicit "
                                "training-only element-mask supervision."
                            )
                        v7_scalar_fields = {
                            "target_binding_temperature": (
                                "policy_guard_target_binding_temperature",
                                float,
                            ),
                            "target_binding_action_start_step": (
                                "policy_guard_target_binding_action_start_step",
                                int,
                            ),
                            "target_binding_action_ramp_steps": (
                                "policy_guard_target_binding_action_ramp_steps",
                                int,
                            ),
                            "target_mask_weight": (
                                "policy_guard_target_mask_weight",
                                float,
                            ),
                            "source_mask_weight": (
                                "policy_guard_source_mask_weight",
                                float,
                            ),
                            "aux_mask_weight": (
                                "policy_guard_aux_mask_weight",
                                float,
                            ),
                            "mask_mass_weight": (
                                "policy_guard_mask_mass_weight",
                                float,
                            ),
                            "cross_object_weight": (
                                "policy_guard_cross_object_weight",
                                float,
                            ),
                            "cross_object_margin": (
                                "policy_guard_cross_object_margin",
                                float,
                            ),
                        }
                        for metadata_name, (
                            attribute_name,
                            cast,
                        ) in v7_scalar_fields.items():
                            value = metadata.get(metadata_name)
                            if value is None:
                                raise ValueError(
                                    "PGC v7 checkpoint is missing object-mask "
                                    f"binding value {metadata_name!r}."
                                )
                            setattr(self, attribute_name, cast(value))
                        if self.policy_guard_target_binding_temperature <= 0:
                            raise ValueError(
                                "PGC v7 checkpoint has non-positive binding "
                                "temperature."
                            )
                        if min(
                            self.policy_guard_target_mask_weight,
                            self.policy_guard_source_mask_weight,
                            self.policy_guard_aux_mask_weight,
                            self.policy_guard_mask_mass_weight,
                            self.policy_guard_cross_object_weight,
                            self.policy_guard_cross_object_margin,
                            self.policy_guard_target_binding_action_start_step,
                            self.policy_guard_target_binding_action_ramp_steps,
                        ) < 0:
                            raise ValueError(
                                "PGC v7 checkpoint has negative mask weights, "
                                "margin, or schedule values."
                            )
                        target_binder = self.policy_guard_modules[
                            "target_binder"
                        ]
                        try:
                            saved_aspect_ratio = float(
                                metadata["target_binding_visual_aspect_ratio"]
                            )
                        except (KeyError, TypeError, ValueError) as exc:
                            raise ValueError(
                                "PGC v7 checkpoint has invalid visual aspect ratio."
                            ) from exc
                        if not math.isclose(
                            saved_aspect_ratio,
                            float(target_binder.visual_aspect_ratio),
                            rel_tol=0.0,
                            abs_tol=1.0e-9,
                        ):
                            raise ValueError(
                                "PGC v7 target-binding visual aspect ratio "
                                f"mismatch: checkpoint={saved_aspect_ratio}, "
                                f"model={target_binder.visual_aspect_ratio}."
                            )
                        target_binder.temperature = (
                            self.policy_guard_target_binding_temperature
                        )
                    elif saved_policy_guard_version == 8:
                        if (
                            metadata.get("closed_loop_corrective_format")
                            != "pgc_libero_closed_loop_corrective_v1"
                            or metadata.get("closed_loop_corrective_enabled")
                            is not True
                            or metadata.get("acquisition_only") is not True
                            or metadata.get("closed_loop_trainable_scope")
                            != "action_chunk_proposal_only"
                        ):
                            raise ValueError(
                                "PGC v8 checkpoint does not declare its audited "
                                "closed-loop acquisition-only contract."
                            )
                        v8_scalar_fields = {
                            "closed_loop_corrective_weight": (
                                "policy_guard_closed_loop_corrective_weight",
                                float,
                            ),
                            "offline_acquisition_weight": (
                                "policy_guard_offline_acquisition_weight",
                                float,
                            ),
                            "native_guard_weight": (
                                "policy_guard_native_guard_weight",
                                float,
                            ),
                        }
                        for metadata_name, (
                            attribute_name,
                            cast,
                        ) in v8_scalar_fields.items():
                            value = metadata.get(metadata_name)
                            if value is None:
                                raise ValueError(
                                    "PGC v8 checkpoint is missing corrective "
                                    f"value {metadata_name!r}."
                                )
                            setattr(self, attribute_name, cast(value))
                        if min(
                            self.policy_guard_closed_loop_corrective_weight,
                            self.policy_guard_offline_acquisition_weight,
                            self.policy_guard_native_guard_weight,
                        ) < 0:
                            raise ValueError(
                                "PGC v8 checkpoint has negative corrective weights."
                            )
                    elif saved_policy_guard_version == 9:
                        saved_initialization_contract = str(
                            metadata.get("warm_start_contract", "")
                        )
                        expected_deployment_inputs = (
                            "rgb_language_proprio_completed_clause_bitset"
                            if saved_eraf_completion_only_memory
                            else (
                                "rgb_language_proprio_previous_policy_state"
                                if int(saved_eraf_grounding_objective or 1) >= 14
                                else "rgb_language_proprio"
                            )
                        )
                        if (
                            saved_initialization_contract
                            not in {
                                "exact_pgc_v5_sidecars",
                                "released_base_fresh_eraf",
                            }
                            or metadata.get("grounding")
                            != "predicate_entity_relation_affordance_field"
                            or metadata.get("privileged_supervision")
                            != "training_only"
                            or metadata.get("deployment_inputs")
                            != expected_deployment_inputs
                        ):
                            raise ValueError(
                                "PGC v9 checkpoint does not declare its ERAF "
                                "deployment and supervision contract."
                            )
                        # This field records checkpoint provenance rather than
                        # a tensor-shape choice. Inherit it on every grounding
                        # objective upgrade so clean Base restarts cannot be
                        # mislabeled as historical PGC-V5 warm starts.
                        self.policy_guard_eraf_initialization_contract = (
                            saved_initialization_contract
                        )
                        try:
                            saved_grounding_objective = int(
                                metadata.get(
                                    "eraf_grounding_objective_version", 1
                                )
                            )
                        except (TypeError, ValueError) as exc:
                            raise ValueError(
                                "PGC v9 checkpoint has an invalid ERAF "
                                "grounding objective version."
                            ) from exc
                        objective_upgrade = (
                            (
                                saved_grounding_objective == 2
                                and self.policy_guard_eraf_grounding_objective_version
                                in {3, 4}
                            )
                            or (
                                saved_grounding_objective == 4
                                and self.policy_guard_eraf_grounding_objective_version
                                in {5, 6, 7}
                            )
                            or (
                                saved_grounding_objective == 7
                                and self.policy_guard_eraf_grounding_objective_version
                                == 8
                            )
                            or (
                                saved_grounding_objective == 8
                                and self.policy_guard_eraf_grounding_objective_version
                                == 9
                            )
                            or (
                                saved_grounding_objective == 9
                                and self.policy_guard_eraf_grounding_objective_version
                                == 10
                            )
                            or (
                                saved_grounding_objective == 10
                                and self.policy_guard_eraf_grounding_objective_version
                                == 11
                            )
                            or (
                                saved_grounding_objective == 11
                                and self.policy_guard_eraf_grounding_objective_version
                                == 12
                            )
                            or (
                                saved_grounding_objective == 12
                                and self.policy_guard_eraf_grounding_objective_version
                                == 13
                            )
                            or (
                                saved_grounding_objective == 12
                                and self.policy_guard_eraf_grounding_objective_version
                                == 14
                            )
                            or (
                                saved_grounding_objective == 14
                                and self.policy_guard_eraf_grounding_objective_version
                                == 15
                            )
                            or (
                                saved_grounding_objective == 15
                                and self.policy_guard_eraf_grounding_objective_version
                                == 16
                            )
                            or (
                                saved_grounding_objective == 16
                                and self.policy_guard_eraf_grounding_objective_version
                                == 17
                            )
                            or (
                                saved_grounding_objective == 17
                                and self.policy_guard_eraf_grounding_objective_version
                                == 18
                            )
                            or (
                                saved_grounding_objective == 18
                                and self.policy_guard_eraf_grounding_objective_version
                                == 19
                            )
                            or (
                                saved_grounding_objective == 19
                                and self.policy_guard_eraf_grounding_objective_version
                                == 20
                            )
                            or (
                                saved_grounding_objective == 20
                                and self.policy_guard_eraf_grounding_objective_version
                                == 21
                            )
                            or (
                                saved_grounding_objective == 21
                                and self.policy_guard_eraf_grounding_objective_version
                                == 22
                            )
                            or (
                                saved_grounding_objective == 21
                                and self.policy_guard_eraf_grounding_objective_version
                                == 23
                            )
                            or (
                                saved_grounding_objective == 21
                                and self.policy_guard_eraf_grounding_objective_version
                                == 24
                            )
                            or (
                                saved_grounding_objective == 24
                                and self.policy_guard_eraf_grounding_objective_version
                                == 25
                            )
                            or (
                                saved_grounding_objective == 25
                                and self.policy_guard_eraf_grounding_objective_version
                                == 26
                            )
                        )
                        objective_upgrade = objective_upgrade and (
                            (
                                self.policy_guard_eraf_training_stage == "grounding"
                                and metadata.get("eraf_training_stage")
                                == "grounding"
                            )
                            or migrate_v914_to_v915_action_grounding
                            or migrate_v915_to_v916_semantic_causal
                            or migrate_v916_to_v917_direct_geometry
                            or migrate_v917_to_v918_phase_residual
                            or migrate_v918_to_v919_phase_servo
                            or migrate_v919_to_v920_waypoint
                            or migrate_v920_to_v921_expert_alignment
                            or migrate_v921_to_v922_clause_ranking
                            or migrate_v921_to_v923_alignment_preserving_clause
                            or migrate_v921_to_v924_isolated_clause_residual
                            or migrate_v924_to_v925_action_context
                            or migrate_v925_to_v926_shared_expert_lora
                        )
                        if (
                            saved_grounding_objective
                            != self.policy_guard_eraf_grounding_objective_version
                            and not objective_upgrade
                        ):
                            raise ValueError(
                                "PGC v9 ERAF grounding objective mismatch: "
                                f"checkpoint={saved_grounding_objective}, "
                                "model="
                                f"{self.policy_guard_eraf_grounding_objective_version}."
                            )
                        if (
                            saved_grounding_objective >= 15
                            and not objective_upgrade
                        ):
                            expected_v915 = {
                                "eraf_action_grounding_contract": (
                                    "separate_subject_reference_relation_grasp_"
                                    "goal_interaction_displacement_tokens_zero_"
                                    "init_v9_14_exact"
                                ),
                                "eraf_action_grounding_hidden_dim": (
                                    self.policy_guard_eraf_action_grounding_hidden_dim
                                ),
                                "eraf_action_grounding_num_heads": (
                                    self.policy_guard_eraf_action_grounding_num_heads
                                ),
                            }
                            for name, expected in expected_v915.items():
                                if metadata.get(name) != expected:
                                    raise ValueError(
                                        "PGC V9.15 action-grounding checkpoint "
                                        f"mismatch: {name}={metadata.get(name)!r}, "
                                        f"expected={expected!r}."
                                    )
                            if (
                                saved_grounding_objective >= 16
                                and metadata.get(
                                    "eraf_action_semantic_negative_contract"
                                )
                                != "joint_valid_entity_id_swap_with_same_state_"
                                "subject_as_reference_fallback_plus_clause_swap"
                            ):
                                raise ValueError(
                                    "PGC V9.16 checkpoint lacks its audited "
                                    "semantic-negative contract."
                                )
                            if saved_grounding_objective >= 17:
                                expected_v917 = {
                                    "eraf_action_geometry_contract": (
                                        "phase_selected_eef_relative_grasp_goal_"
                                        "interaction_relation_direct_bounded_action_"
                                        "residual_zero_init_v9_16_exact"
                                    ),
                                    "eraf_action_geometry_hidden_dim": (
                                        self.policy_guard_eraf_action_geometry_hidden_dim
                                    ),
                                    "eraf_action_geometry_residual_max_abs": (
                                        self.policy_guard_eraf_action_geometry_residual_max_abs
                                    ),
                                    "eraf_action_anchor_negative_contract": (
                                        "same_state_source_anchor_or_reference_mirror_"
                                        "with_offset_fallback"
                                    ),
                                }
                                for name, expected in expected_v917.items():
                                    value = metadata.get(name)
                                    if isinstance(expected, float):
                                        matches = math.isclose(
                                            float(value), expected,
                                            rel_tol=0.0,
                                            abs_tol=1.0e-9,
                                        )
                                    else:
                                        matches = value == expected
                                    if not matches:
                                        raise ValueError(
                                            "PGC V9.17 geometry-action checkpoint "
                                            f"mismatch: {name}={value!r}, "
                                            f"expected={expected!r}."
                                        )
                                if saved_grounding_objective >= 18:
                                    expected_v918 = {
                                        "eraf_action_phase_residual_contract": (
                                            "phase_balanced_bounded_expert_minus_"
                                            "frozen_v9_17_candidate_prefix_residual_"
                                            "imitation"
                                        ),
                                        "eraf_action_phase_residual_imitation_weight": (
                                            self.policy_guard_eraf_action_phase_residual_imitation_weight
                                        ),
                                        "eraf_action_phase_direction_weight": (
                                            self.policy_guard_eraf_action_phase_direction_weight
                                        ),
                                        "eraf_action_phase_direction_min_norm": (
                                            self.policy_guard_eraf_action_phase_direction_min_norm
                                        ),
                                    }
                                    for name, expected in expected_v918.items():
                                        value = metadata.get(name)
                                        matches = (
                                            value == expected
                                            if isinstance(expected, str)
                                            else math.isclose(
                                                float(value),
                                                float(expected),
                                                rel_tol=0.0,
                                                abs_tol=1.0e-9,
                                            )
                                        )
                                        if not matches:
                                            raise ValueError(
                                                "PGC V9.18 phase-residual checkpoint "
                                                f"mismatch: {name}={value!r}, "
                                                f"expected={expected!r}."
                                            )
                                    saved_phase_weights = metadata.get(
                                        "eraf_action_phase_weights"
                                    )
                                    expected_phase_weights = [
                                        self.policy_guard_eraf_action_phase_approach_weight,
                                        self.policy_guard_eraf_action_phase_transport_weight,
                                        self.policy_guard_eraf_action_phase_release_weight,
                                    ]
                                    if not isinstance(saved_phase_weights, list) or (
                                        len(saved_phase_weights) != 3
                                    ) or any(
                                        not math.isclose(
                                            float(saved),
                                            float(expected),
                                            rel_tol=0.0,
                                            abs_tol=1.0e-9,
                                        )
                                        for saved, expected in zip(
                                            saved_phase_weights,
                                            expected_phase_weights,
                                        )
                                    ):
                                        raise ValueError(
                                            "PGC V9.18 phase-weight contract mismatch."
                                        )
                                if saved_grounding_objective >= 19:
                                    expected_v919 = {
                                        "eraf_action_phase_servo_contract": (
                                            "hard_single_clause_explicit_affine_eef_"
                                            "phase_specific_positive_cartesian_gain_"
                                            "with_legacy_suppression"
                                        ),
                                        "eraf_action_servo_frame_weight": (
                                            self.policy_guard_eraf_action_servo_frame_weight
                                        ),
                                    }
                                    for name, expected in expected_v919.items():
                                        value = metadata.get(name)
                                        matches = (
                                            value == expected
                                            if isinstance(expected, str)
                                            else math.isclose(
                                                float(value), float(expected),
                                                rel_tol=0.0, abs_tol=1.0e-9,
                                            )
                                        )
                                        if not matches:
                                            raise ValueError(
                                                "PGC V9.19 phase-servo checkpoint "
                                                f"mismatch: {name}={value!r}, "
                                                f"expected={expected!r}."
                                            )
                                    for name, expected in {
                                        "eraf_action_eef_initial_scale": (
                                            self.policy_guard_eraf_action_eef_scale
                                        ),
                                        "eraf_action_eef_initial_bias": (
                                            self.policy_guard_eraf_action_eef_bias
                                        ),
                                    }.items():
                                        saved = metadata.get(name)
                                        if (
                                            not isinstance(saved, list)
                                            or len(saved) != 3
                                            or any(
                                                not math.isclose(
                                                    float(left), float(right),
                                                    rel_tol=0.0, abs_tol=1.0e-9,
                                                )
                                                for left, right in zip(
                                                    saved, expected
                                                )
                                            )
                                        ):
                                            raise ValueError(
                                                "PGC V9.19 EEF affine checkpoint "
                                                f"mismatch: {name}={saved!r}, "
                                                f"expected={list(expected)!r}."
                                            )
                                if saved_grounding_objective >= 20:
                                    if metadata.get("eraf_action_waypoint_contract") != (
                                        "hard_clause_phase_compatible_positive_progress_"
                                        "local_tangent_waypoint_with_privileged_training_"
                                        "only_compatibility_labels"
                                    ):
                                        raise ValueError(
                                            "PGC V9.20 checkpoint lacks its local-waypoint contract."
                                        )
                                    for name, expected in {
                                        "eraf_action_waypoint_min_cosine": self.policy_guard_eraf_action_waypoint_min_cosine,
                                        "eraf_action_waypoint_tangent_max_ratio": self.policy_guard_eraf_action_waypoint_tangent_max_ratio,
                                    }.items():
                                        if not math.isclose(
                                            float(metadata.get(name)), float(expected),
                                            rel_tol=0.0, abs_tol=1.0e-9,
                                        ):
                                            raise ValueError(
                                                f"PGC V9.20 waypoint mismatch: {name}."
                                            )
                                if saved_grounding_objective >= 21:
                                    if metadata.get(
                                        "eraf_action_expert_alignment_contract"
                                    ) != (
                                        "training_only_privileged_phase_anchor_teacher_"
                                        "plus_deployed_full_action_prefix_residual_and_"
                                        "semantic_causal_ranking"
                                    ):
                                        raise ValueError(
                                            "PGC V9.21 checkpoint lacks its privileged "
                                            "expert-prefix alignment contract."
                                        )
                                    expected_weights = [
                                        self.policy_guard_eraf_action_expert_imitation_weight,
                                        self.policy_guard_eraf_action_expert_direction_weight,
                                        self.policy_guard_eraf_action_expert_deployed_weight,
                                        self.policy_guard_eraf_action_expert_distillation_weight,
                                        self.policy_guard_eraf_action_expert_native_zero_weight,
                                    ]
                                    saved_weights = metadata.get(
                                        "eraf_action_expert_alignment_loss_weights"
                                    )
                                    if (
                                        not isinstance(saved_weights, list)
                                        or len(saved_weights) != len(expected_weights)
                                        or any(
                                            not math.isclose(
                                                float(left),
                                                float(right),
                                                rel_tol=0.0,
                                                abs_tol=1.0e-9,
                                            )
                                            for left, right in zip(
                                                saved_weights, expected_weights
                                            )
                                        )
                                    ):
                                        raise ValueError(
                                            "PGC V9.21 expert-alignment loss weights "
                                            "do not match the model configuration."
                                        )
                                if saved_grounding_objective >= 22:
                                    expected_clause_contract = (
                                        "frozen_v921_correct_route_identity_plus_"
                                        "isolated_wrong_clause_base_fallback_ranking"
                                        if saved_grounding_objective >= 24
                                        else
                                        "frozen_v921_teacher_correct_output_"
                                        "preservation_plus_expert_nonregression_"
                                        "and_detached_correct_wrong_clause_ranking_"
                                        "balanced_over_approach_transport_release"
                                        if saved_grounding_objective == 23
                                        else "coherent_same_state_clause_swap_final_"
                                        "expert_prefix_mse_ranking_balanced_over_"
                                        "approach_transport_release"
                                    )
                                    if metadata.get(
                                        "eraf_action_clause_ranking_contract"
                                    ) != expected_clause_contract:
                                        raise ValueError(
                                            "PGC V9.22 checkpoint lacks its balanced "
                                            "final-action clause-ranking contract."
                                        )
                                    for name, expected in {
                                        "eraf_action_clause_ranking_weight": (
                                            self.policy_guard_eraf_action_clause_ranking_weight
                                        ),
                                        "eraf_action_clause_ranking_margin": (
                                            self.policy_guard_eraf_action_clause_ranking_margin
                                        ),
                                    }.items():
                                        if not math.isclose(
                                            float(metadata.get(name)),
                                            float(expected),
                                            rel_tol=0.0,
                                            abs_tol=1.0e-9,
                                        ):
                                            raise ValueError(
                                                f"PGC V9.22 clause-ranking mismatch: {name}."
                                            )
                                if saved_grounding_objective == 23:
                                    if metadata.get(
                                        "eraf_action_clause_teacher_contract"
                                    ) != (
                                        "training_only_frozen_exact_v921_expert_"
                                        "residual_adapter_excluded_from_rollout_"
                                        "and_optimizer"
                                    ):
                                        raise ValueError(
                                            "PGC V9.23 checkpoint lacks its frozen "
                                            "V9.21 teacher contract."
                                        )
                                    for name, expected in {
                                        "eraf_action_clause_teacher_weight": (
                                            self.policy_guard_eraf_action_clause_teacher_weight
                                        ),
                                        "eraf_action_clause_alignment_guard_weight": (
                                            self.policy_guard_eraf_action_clause_alignment_guard_weight
                                        ),
                                    }.items():
                                        if not math.isclose(
                                            float(metadata.get(name)),
                                            float(expected),
                                            rel_tol=0.0,
                                            abs_tol=1.0e-9,
                                        ):
                                            raise ValueError(
                                                f"PGC V9.23 teacher mismatch: {name}."
                                            )
                                if saved_grounding_objective >= 24:
                                    if metadata.get(
                                        "eraf_action_clause_residual_contract"
                                    ) != (
                                        "frozen_v921_positive_action_plus_identity_"
                                        "initialized_clause_conditioned_base_fallback"
                                    ):
                                        raise ValueError(
                                            "PGC V9.24 checkpoint lacks its isolated "
                                            "clause semantic-residual contract."
                                        )
                                    if not math.isclose(
                                        float(
                                            metadata.get(
                                                "eraf_action_clause_wrong_"
                                                "suppression_weight"
                                            )
                                        ),
                                        float(
                                            self.policy_guard_eraf_action_clause_wrong_suppression_weight
                                        ),
                                        rel_tol=0.0,
                                        abs_tol=1.0e-9,
                                    ):
                                        raise ValueError(
                                            "PGC V9.24 wrong-clause suppression "
                                            "weight does not match the model configuration."
                                        )
                        # V9.13 intentionally freezes the complete V9.11 ERAF
                        # geometry and disables every legacy grounding loss.
                        # Those loss weights are training-time hyperparameters,
                        # not part of the restored geometry contract, so a
                        # V9.11 -> V9.13 warm start must not compare them with
                        # the zeroed V9.13 values.  Structural V9.11 metadata
                        # and exact sidecar tensors are validated separately.
                        if (
                            saved_grounding_objective >= 2
                            and not migrate_v9_to_phase_safe_memory
                            and not objective_upgrade
                        ):
                            for metadata_name, expected_value in {
                                "eraf_attention_mask_weight": (
                                    self.policy_guard_eraf_loss_weights.attention_mask
                                ),
                                "eraf_role_swap_weight": (
                                    self.policy_guard_eraf_loss_weights.role_swap
                                ),
                                "eraf_role_overlap_weight": (
                                    self.policy_guard_eraf_loss_weights.role_overlap
                                ),
                                "eraf_role_swap_margin": (
                                    self.policy_guard_eraf_loss_weights.role_swap_margin
                                ),
                            }.items():
                                try:
                                    saved_value = float(metadata[metadata_name])
                                except (KeyError, TypeError, ValueError) as exc:
                                    raise ValueError(
                                        "PGC v9 checkpoint is missing valid "
                                        f"grounding value {metadata_name!r}."
                                    ) from exc
                                if not math.isclose(
                                    saved_value,
                                    float(expected_value),
                                    rel_tol=0.0,
                                    abs_tol=1.0e-9,
                                ):
                                    raise ValueError(
                                        f"PGC v9 {metadata_name} mismatch: "
                                        f"checkpoint={saved_value}, "
                                        f"model={expected_value}."
                                    )
                        if (
                            saved_grounding_objective >= 3
                            and not migrate_v9_to_phase_safe_memory
                            and not objective_upgrade
                        ):
                            for metadata_name, expected_value in {
                                "eraf_role_assignment_weight": (
                                    self.policy_guard_eraf_loss_weights.role_assignment
                                ),
                                "eraf_role_assignment_temperature": (
                                    self.policy_guard_eraf_loss_weights.role_assignment_temperature
                                ),
                                "eraf_role_assignment_hard_weight": (
                                    self.policy_guard_eraf_loss_weights.role_assignment_hard_weight
                                ),
                            }.items():
                                try:
                                    saved_value = float(metadata[metadata_name])
                                except (KeyError, TypeError, ValueError) as exc:
                                    raise ValueError(
                                        "PGC v9 checkpoint is missing valid "
                                        f"grounding value {metadata_name!r}."
                                    ) from exc
                                if not math.isclose(
                                    saved_value,
                                    float(expected_value),
                                    rel_tol=0.0,
                                    abs_tol=1.0e-9,
                                ):
                                    raise ValueError(
                                        f"PGC v9 {metadata_name} mismatch: "
                                        f"checkpoint={saved_value}, "
                                        f"model={expected_value}."
                                    )
                        if saved_grounding_objective >= 4 and not objective_upgrade:
                            if saved_grounding_objective >= 14:
                                saved_action_joint_training = bool(
                                    metadata.get(
                                        "eraf_action_joint_training", False
                                    )
                                )
                                saved_action_stage = (
                                    metadata.get("eraf_training_stage") == "action"
                                )
                                expected_scope = (
                                    (
                                        "shared_video_action_lora_plus_eraf_action_"
                                        "context_injector"
                                        if saved_grounding_objective >= 26
                                        else "eraf_action_context_injector_only"
                                        if saved_grounding_objective >= 25
                                        else "clause_semantic_retention_residual_only"
                                        if saved_grounding_objective >= 24
                                        else "phase_specific_privileged_expert_residual_adapter_only"
                                        if saved_grounding_objective >= 21
                                        else "phase_compatible_local_waypoint_adapter_only"
                                        if saved_grounding_objective >= 20
                                        else "hard_routed_phase_servo_only"
                                        if saved_grounding_objective >= 19
                                        else (
                                            "phase_conditioned_geometry_adapter_only_"
                                            "with_phase_balanced_residual_imitation"
                                        )
                                        if saved_grounding_objective >= 18
                                        else "phase_conditioned_relative_geometry_"
                                        "action_adapter_only"
                                    )
                                    if (
                                        saved_grounding_objective >= 17
                                        and saved_action_joint_training
                                        and saved_action_stage
                                    )
                                    else (
                                        "semantic_causal_action_grounding_bridge_only"
                                        if (
                                            saved_grounding_objective >= 16
                                            and saved_action_joint_training
                                            and saved_action_stage
                                        )
                                        else (
                                            "frozen_eraf_perception_action_bridge_"
                                            "plus_proposal"
                                            if (
                                                saved_action_joint_training
                                                and saved_action_stage
                                            )
                                            else (
                                                "phase_safe_temporal_clause_memory_only"
                                            )
                                        )
                                    )
                                )
                            elif saved_grounding_objective == 13:
                                expected_scope = (
                                    "closed_loop_phase_rebinding_adapter_only"
                                )
                            elif saved_grounding_objective >= 12:
                                expected_scope = (
                                    "audited_hard_clause_tuple_balanced_"
                                    "visual_role_binding_adapter_only"
                                )
                            elif saved_grounding_objective >= 11:
                                expected_scope = (
                                    "exclusive_all_entity_balanced_"
                                    "visual_role_binding_adapter_only"
                                )
                            elif saved_grounding_objective >= 10:
                                expected_scope = (
                                    "clause_activation_plus_balanced_role_plus_"
                                    "visibility_gated_view_fusion_plus_unfinished_"
                                    "clause_scheduler"
                                )
                            elif saved_grounding_objective >= 9:
                                expected_scope = (
                                    "clause_activation_calibration_adapter_only"
                                )
                            elif saved_grounding_objective >= 8:
                                expected_scope = (
                                    "exclusive_evidence_global_hard_curriculum_"
                                    "balanced_visual_role_binding_adapter_only"
                                )
                            elif saved_grounding_objective >= 7:
                                expected_scope = (
                                    "global_hard_curriculum_balanced_visual_"
                                    "role_binding_adapter_only"
                                )
                            elif saved_grounding_objective >= 6:
                                expected_scope = (
                                    "balanced_visual_role_binding_adapter_only"
                                )
                            elif saved_grounding_objective >= 5:
                                expected_scope = (
                                    "structured_role_assignment_adapter_only"
                                )
                            else:
                                expected_scope = "role_assignment_adapter_only"
                            if (
                                metadata.get("eraf_role_adapter_trainable_scope")
                                != expected_scope
                            ):
                                raise ValueError(
                                    "PGC v9 role-adapter checkpoint does not "
                                    "declare its expected trainable scope: "
                                    f"{expected_scope!r}."
                                )
                            for metadata_name, expected_value in {
                                "eraf_role_attention_preservation_weight": (
                                    self.policy_guard_eraf_loss_weights.role_attention_preservation
                                ),
                                "eraf_role_position_preservation_weight": (
                                    self.policy_guard_eraf_loss_weights.role_position_preservation
                                ),
                                "eraf_role_anchor_preservation_weight": (
                                    self.policy_guard_eraf_loss_weights.role_anchor_preservation
                                ),
                                "eraf_role_relation_preservation_weight": (
                                    self.policy_guard_eraf_loss_weights.role_relation_preservation
                                ),
                                "eraf_role_adapter_energy_weight": (
                                    self.policy_guard_eraf_loss_weights.role_adapter_energy
                                ),
                            }.items():
                                try:
                                    saved_value = float(metadata[metadata_name])
                                except (KeyError, TypeError, ValueError) as exc:
                                    raise ValueError(
                                        "PGC v9.3 checkpoint is missing valid "
                                        f"preservation value {metadata_name!r}."
                                    ) from exc
                                if not math.isclose(
                                    saved_value,
                                    float(expected_value),
                                    rel_tol=0.0,
                                    abs_tol=1.0e-9,
                                ):
                                    raise ValueError(
                                        f"PGC v9.3 {metadata_name} mismatch: "
                                        f"checkpoint={saved_value}, "
                                        f"model={expected_value}."
                                    )
                        if saved_grounding_objective >= 5 and not objective_upgrade:
                            for metadata_name, expected_value in {
                                "eraf_structured_assignment_weight": (
                                    self.policy_guard_eraf_loss_weights.structured_assignment
                                ),
                                "eraf_structured_assignment_temperature": (
                                    self.policy_guard_eraf_loss_weights.structured_assignment_temperature
                                ),
                                "eraf_structured_assignment_hard_weight": (
                                    self.policy_guard_eraf_loss_weights.structured_assignment_hard_weight
                                ),
                                "eraf_multi_clause_consistency_weight": (
                                    self.policy_guard_eraf_loss_weights.multi_clause_consistency
                                ),
                            }.items():
                                try:
                                    saved_value = float(metadata[metadata_name])
                                except (KeyError, TypeError, ValueError) as exc:
                                    raise ValueError(
                                        "PGC v9.4 checkpoint is missing valid "
                                        f"structured assignment value {metadata_name!r}."
                                    ) from exc
                                if not math.isclose(
                                    saved_value,
                                    float(expected_value),
                                    rel_tol=0.0,
                                    abs_tol=1.0e-9,
                                ):
                                    raise ValueError(
                                        f"PGC v9.4 {metadata_name} mismatch: "
                                        f"checkpoint={saved_value}, "
                                        f"model={expected_value}."
                                    )
                            try:
                                saved_structured_hidden_dim = int(
                                    metadata[
                                        "eraf_structured_role_adapter_hidden_dim"
                                    ]
                                )
                            except (KeyError, TypeError, ValueError) as exc:
                                raise ValueError(
                                    "PGC v9.4 checkpoint is missing its structured "
                                    "role-adapter hidden dimension."
                                ) from exc
                            if (
                                saved_structured_hidden_dim
                                != self.policy_guard_eraf_structured_role_adapter_hidden_dim
                            ):
                                raise ValueError(
                                    "PGC v9.4 structured role-adapter hidden "
                                    "dimension mismatch: checkpoint="
                                    f"{saved_structured_hidden_dim}, model="
                                    f"{self.policy_guard_eraf_structured_role_adapter_hidden_dim}."
                                )
                        if saved_grounding_objective >= 6 and not objective_upgrade:
                            try:
                                saved_balanced_hidden_dim = int(
                                    metadata["eraf_balanced_role_adapter_hidden_dim"]
                                )
                            except (KeyError, TypeError, ValueError) as exc:
                                raise ValueError(
                                    "PGC v9.5 checkpoint is missing its balanced "
                                    "role-adapter hidden dimension."
                                ) from exc
                            if (
                                saved_balanced_hidden_dim
                                != self.policy_guard_eraf_balanced_role_adapter_hidden_dim
                            ):
                                raise ValueError(
                                    "PGC v9.5 balanced role-adapter hidden "
                                    "dimension mismatch: checkpoint="
                                    f"{saved_balanced_hidden_dim}, model="
                                    f"{self.policy_guard_eraf_balanced_role_adapter_hidden_dim}."
                                )
                        if saved_grounding_objective >= 7 and not objective_upgrade:
                            expected_v96_contract = {
                                "eraf_hard_role_curriculum": (
                                    "v9_10_audited_clause_tuple_native_hard_easy_1_1"
                                    if saved_grounding_objective >= 12
                                    else "v9_3_audited_native_hard_easy_1_1"
                                ),
                                "eraf_ddp_group_balance": "global_count_exact",
                                "eraf_geometry_preservation_scope": (
                                    "all_active_clauses"
                                ),
                            }
                            for name, expected in expected_v96_contract.items():
                                if metadata.get(name) != expected:
                                    raise ValueError(
                                        "PGC V9.6 checkpoint contract mismatch: "
                                        f"{name}={metadata.get(name)!r}, "
                                        f"expected={expected!r}."
                                    )
                        if saved_grounding_objective >= 8 and not objective_upgrade:
                            expected_v97_contract = {
                                "eraf_role_evidence": (
                                    "exclusive_subject_reference_support"
                                ),
                                "eraf_role_gate": (
                                    "exclusive_accuracy_with_full_mask_localization"
                                ),
                                "eraf_exclusive_role_coverage_min": 0.5,
                            }
                            for name, expected in expected_v97_contract.items():
                                if metadata.get(name) != expected:
                                    raise ValueError(
                                        "PGC V9.7 checkpoint contract mismatch: "
                                        f"{name}={metadata.get(name)!r}, "
                                        f"expected={expected!r}."
                                    )
                        if saved_grounding_objective >= 9 and not objective_upgrade:
                            expected_v98_contract = {
                                "eraf_clause_activation_contract": (
                                    "zero_init_cross_clause_active_logit_residual"
                                ),
                                "eraf_clause_cardinality_supervision": (
                                    "balanced_active_bce_plus_count_ce_plus_"
                                    "multi_worst_slot"
                                ),
                                "eraf_clause_gate": (
                                    "multi_clause_exact_at_least_80pct"
                                ),
                            }
                            for name, expected in expected_v98_contract.items():
                                if metadata.get(name) != expected:
                                    raise ValueError(
                                        "PGC V9.8 checkpoint contract mismatch: "
                                        f"{name}={metadata.get(name)!r}, "
                                        f"expected={expected!r}."
                                    )
                            for metadata_name, expected_value in {
                                "eraf_clause_activation_balance_weight": (
                                    self.policy_guard_eraf_loss_weights.clause_activation_balance
                                ),
                                "eraf_clause_cardinality_weight": (
                                    self.policy_guard_eraf_loss_weights.clause_cardinality
                                ),
                                "eraf_clause_worst_slot_weight": (
                                    self.policy_guard_eraf_loss_weights.clause_worst_slot
                                ),
                                "eraf_clause_multi_group_weight": (
                                    self.policy_guard_eraf_loss_weights.clause_multi_group_weight
                                ),
                                "eraf_clause_adapter_energy_weight": (
                                    self.policy_guard_eraf_loss_weights.clause_adapter_energy
                                ),
                                "eraf_clause_activation_residual_max_abs": (
                                    self.policy_guard_eraf_clause_activation_residual_max_abs
                                ),
                            }.items():
                                try:
                                    saved_value = float(metadata[metadata_name])
                                except (KeyError, TypeError, ValueError) as exc:
                                    raise ValueError(
                                        "PGC V9.8 checkpoint is missing valid "
                                        f"clause calibration value {metadata_name!r}."
                                    ) from exc
                                if not math.isclose(
                                    saved_value,
                                    float(expected_value),
                                    rel_tol=0.0,
                                    abs_tol=1.0e-9,
                                ):
                                    raise ValueError(
                                        f"PGC V9.8 {metadata_name} mismatch: "
                                        f"checkpoint={saved_value}, "
                                        f"model={expected_value}."
                                    )
                            try:
                                saved_clause_hidden_dim = int(
                                    metadata[
                                        "eraf_clause_activation_adapter_hidden_dim"
                                    ]
                                )
                            except (KeyError, TypeError, ValueError) as exc:
                                raise ValueError(
                                    "PGC V9.8 checkpoint is missing its clause "
                                    "activation adapter hidden dimension."
                                ) from exc
                            if (
                                saved_clause_hidden_dim
                                != self.policy_guard_eraf_clause_activation_adapter_hidden_dim
                            ):
                                raise ValueError(
                                    "PGC V9.8 clause activation adapter hidden "
                                    "dimension mismatch: checkpoint="
                                    f"{saved_clause_hidden_dim}, model="
                                    f"{self.policy_guard_eraf_clause_activation_adapter_hidden_dim}."
                                )
                        if saved_grounding_objective >= 10 and not objective_upgrade:
                            expected_v99_contract = {
                                "eraf_view_fusion_contract": (
                                    "per_view_local_attention_visibility_gated_"
                                    "zero_init_residual"
                                ),
                                "eraf_clause_scheduler_contract": (
                                    "first_active_unfinished_predicate_zero_init_"
                                    "residual_route"
                                ),
                            }
                            for name, expected in expected_v99_contract.items():
                                if metadata.get(name) != expected:
                                    raise ValueError(
                                        "PGC V9.9 checkpoint contract mismatch: "
                                        f"{name}={metadata.get(name)!r}, "
                                        f"expected={expected!r}."
                                    )
                            for metadata_name, expected_value in {
                                "eraf_view_fusion_weight": (
                                    self.policy_guard_eraf_loss_weights.view_fusion
                                ),
                                "eraf_view_fusion_energy_weight": (
                                    self.policy_guard_eraf_loss_weights.view_fusion_energy
                                ),
                                "eraf_clause_scheduler_weight": (
                                    self.policy_guard_eraf_loss_weights.clause_scheduler
                                ),
                                "eraf_clause_scheduler_energy_weight": (
                                    self.policy_guard_eraf_loss_weights.clause_scheduler_energy
                                ),
                                "eraf_view_fusion_residual_max_abs": (
                                    self.policy_guard_eraf_view_fusion_residual_max_abs
                                ),
                                "eraf_clause_scheduler_residual_max_abs": (
                                    self.policy_guard_eraf_clause_scheduler_residual_max_abs
                                ),
                            }.items():
                                try:
                                    saved_value = float(metadata[metadata_name])
                                except (KeyError, TypeError, ValueError) as exc:
                                    raise ValueError(
                                        "PGC V9.9 checkpoint is missing valid "
                                        f"calibration value {metadata_name!r}."
                                    ) from exc
                                if not math.isclose(
                                    saved_value,
                                    float(expected_value),
                                    rel_tol=0.0,
                                    abs_tol=1.0e-9,
                                ):
                                    raise ValueError(
                                        f"PGC V9.9 {metadata_name} mismatch: "
                                        f"checkpoint={saved_value}, "
                                        f"model={expected_value}."
                                    )
                            for metadata_name, expected_value in {
                                "eraf_view_fusion_adapter_hidden_dim": (
                                    self.policy_guard_eraf_view_fusion_adapter_hidden_dim
                                ),
                                "eraf_clause_scheduler_hidden_dim": (
                                    self.policy_guard_eraf_clause_scheduler_hidden_dim
                                ),
                            }.items():
                                try:
                                    saved_value = int(metadata[metadata_name])
                                except (KeyError, TypeError, ValueError) as exc:
                                    raise ValueError(
                                        "PGC V9.9 checkpoint is missing valid "
                                        f"dimension {metadata_name!r}."
                                    ) from exc
                                if saved_value != int(expected_value):
                                    raise ValueError(
                                        f"PGC V9.9 {metadata_name} mismatch: "
                                        f"checkpoint={saved_value}, "
                                        f"model={expected_value}."
                                    )
                        if saved_grounding_objective >= 11 and not objective_upgrade:
                            expected_v910_contract = {
                                "eraf_all_entity_role_contract": (
                                    "exclusive_evidence_same_state_all_entity_"
                                    "bipartite_assignment"
                                ),
                                "eraf_multi_clause_gate_contract": (
                                    "semantic_exact_with_exclusive_role_evidence"
                                ),
                            }
                            for name, expected in expected_v910_contract.items():
                                if metadata.get(name) != expected:
                                    raise ValueError(
                                        "PGC V9.10 checkpoint contract mismatch: "
                                        f"{name}={metadata.get(name)!r}, "
                                        f"expected={expected!r}."
                                    )
                        if saved_grounding_objective >= 12 and not objective_upgrade:
                            expected_v911_contract = {
                                "eraf_clause_tuple_contract": (
                                    "exclusive_same_state_subject_predicate_"
                                    "reference_assignment"
                                ),
                                "eraf_clause_tuple_curriculum_contract": (
                                    "v9_10_audit_native_hard_easy_plus_"
                                    "historical_strict_1_1_1_1"
                                ),
                            }
                            for name, expected in expected_v911_contract.items():
                                if metadata.get(name) != expected:
                                    raise ValueError(
                                        "PGC V9.11 checkpoint contract mismatch: "
                                        f"{name}={metadata.get(name)!r}, "
                                        f"expected={expected!r}."
                                    )
                            for metadata_name, expected_value in {
                                "eraf_clause_tuple_assignment_weight": (
                                    self.policy_guard_eraf_loss_weights.clause_tuple_assignment
                                ),
                                "eraf_clause_tuple_temperature": (
                                    self.policy_guard_eraf_loss_weights.clause_tuple_temperature
                                ),
                                "eraf_clause_tuple_hard_weight": (
                                    self.policy_guard_eraf_loss_weights.clause_tuple_hard_weight
                                ),
                                "eraf_clause_tuple_multi_consistency_weight": (
                                    self.policy_guard_eraf_loss_weights.clause_tuple_multi_consistency
                                ),
                            }.items():
                                try:
                                    saved_value = float(metadata[metadata_name])
                                except (KeyError, TypeError, ValueError) as exc:
                                    raise ValueError(
                                        "PGC V9.11 checkpoint is missing valid "
                                        f"clause-tuple value {metadata_name!r}."
                                    ) from exc
                                if not math.isclose(
                                    saved_value,
                                    float(expected_value),
                                    rel_tol=0.0,
                                    abs_tol=1.0e-9,
                                ):
                                    raise ValueError(
                                        f"PGC V9.11 {metadata_name} mismatch: "
                                        f"checkpoint={saved_value}, "
                                        f"model={expected_value}."
                                    )
                        if saved_grounding_objective == 13 and not objective_upgrade:
                            expected_v912_contract = {
                                "eraf_closed_loop_rebinding_contract": (
                                    "zero_init_second_pass_role_truth_phase_and_"
                                    "clause_route"
                                ),
                                "eraf_closed_loop_state_contract": (
                                    "immutable_base_correct_replan_exact_simulator_"
                                    "state"
                                ),
                                "eraf_closed_loop_curriculum_contract": (
                                    "offline_native_closed_loop_native_historical_"
                                    "strict_1_1_1_1"
                                ),
                                "eraf_role_adapter_trainable_scope": (
                                    "closed_loop_phase_rebinding_adapter_only"
                                ),
                            }
                            for name, expected in expected_v912_contract.items():
                                if metadata.get(name) != expected:
                                    raise ValueError(
                                        "PGC V9.12 checkpoint contract mismatch: "
                                        f"{name}={metadata.get(name)!r}, "
                                        f"expected={expected!r}."
                                    )
                            for metadata_name, expected_value in {
                                "eraf_phase_rebinding_energy_weight": (
                                    self.policy_guard_eraf_loss_weights.phase_rebinding_energy
                                ),
                                "eraf_closed_loop_query_residual_max_abs": (
                                    self.policy_guard_eraf_closed_loop_query_residual_max_abs
                                ),
                                "eraf_closed_loop_state_residual_max_abs": (
                                    self.policy_guard_eraf_closed_loop_state_residual_max_abs
                                ),
                            }.items():
                                try:
                                    saved_value = float(metadata[metadata_name])
                                except (KeyError, TypeError, ValueError) as exc:
                                    raise ValueError(
                                        "PGC V9.12 checkpoint is missing valid "
                                        f"rebinding value {metadata_name!r}."
                                    ) from exc
                                if not math.isclose(
                                    saved_value,
                                    float(expected_value),
                                    rel_tol=0.0,
                                    abs_tol=1.0e-9,
                                ):
                                    raise ValueError(
                                        f"PGC V9.12 {metadata_name} mismatch: "
                                        f"checkpoint={saved_value}, "
                                        f"model={expected_value}."
                                    )
                            try:
                                saved_hidden_dim = int(
                                    metadata[
                                        "eraf_closed_loop_rebinding_hidden_dim"
                                    ]
                                )
                            except (KeyError, TypeError, ValueError) as exc:
                                raise ValueError(
                                    "PGC V9.12 checkpoint is missing its "
                                    "rebinding hidden dimension."
                                ) from exc
                            if saved_hidden_dim != int(
                                self.policy_guard_eraf_closed_loop_rebinding_hidden_dim
                            ):
                                raise ValueError(
                                    "PGC V9.12 rebinding hidden dimension "
                                    f"mismatch: checkpoint={saved_hidden_dim}, "
                                    "model="
                                    f"{self.policy_guard_eraf_closed_loop_rebinding_hidden_dim}."
                                )
                        if saved_grounding_objective >= 14 and not objective_upgrade:
                            expected_policy_state_contract = (
                                "monotonic_completed_bitset_no_pending_holding_"
                                "retry_recurrence"
                                if saved_eraf_completion_only_memory
                                else "explicit_caller_owned_reset_per_episode"
                            )
                            expected_v913_contract = {
                                "eraf_closed_loop_state_contract": (
                                    "immutable_base_correct_replan_exact_simulator_"
                                    "state"
                                ),
                                "eraf_closed_loop_curriculum_contract": (
                                    "offline_native_closed_loop_native_historical_"
                                    "strict_1_1_1_1"
                                ),
                                "eraf_phase_safe_memory_contract": (
                                    "explicit_cross_replan_pending_holding_retry_"
                                    "completed"
                                ),
                                "eraf_geometry_protection_contract": (
                                    "frozen_v9_11_no_query_token_anchor_or_heatmap_"
                                    "residual"
                                ),
                                "eraf_release_transition_contract": (
                                    "release_true_advance_release_false_retry"
                                ),
                                "eraf_policy_state_contract": (
                                    expected_policy_state_contract
                                ),
                                "eraf_phase_safe_memory_warm_start": (
                                    "exact_v9_11_geometry"
                                ),
                            }
                            for name, expected in expected_v913_contract.items():
                                if metadata.get(name) != expected:
                                    raise ValueError(
                                        "PGC V9.13 checkpoint contract mismatch: "
                                        f"{name}={metadata.get(name)!r}, "
                                        f"expected={expected!r}."
                                    )
                            if saved_eraf_completion_only_memory:
                                if not self.policy_guard_eraf_completion_only_memory:
                                    raise ValueError(
                                        "PGC V9.14 checkpoint requires the model's "
                                        "completion-only memory contract."
                                    )
                                expected_action_joint_contract = (
                                    (
                                        "frozen_eraf_completion_memory_plus_shared_"
                                        "video_action_expert_lora_and_internal_"
                                        "context_injector_single_path"
                                        if saved_grounding_objective >= 26
                                        else "frozen_eraf_and_shared_action_expert_plus_"
                                        "internal_context_injector_no_post_action_"
                                        "residual"
                                        if saved_grounding_objective >= 25
                                        else "frozen_v921_expert_adapter_plus_isolated_"
                                        "clause_semantic_retention_residual"
                                        if saved_grounding_objective >= 24
                                        else
                                        "frozen_v921_teacher_plus_alignment_"
                                        "preserving_negative_focused_final_action_"
                                        "clause_ranking"
                                        if saved_grounding_objective == 23
                                        else "frozen_v920_stack_plus_phase_specific_"
                                        "expert_adapter_with_balanced_final_action_"
                                        "clause_ranking"
                                        if saved_grounding_objective >= 22
                                        else "frozen_v920_stack_plus_phase_specific_"
                                        "privileged_expert_prefix_residual_alignment"
                                        if saved_grounding_objective >= 21
                                        else "frozen_v919_stack_plus_phase_compatible_"
                                        "local_waypoint_vector_field"
                                        if saved_grounding_objective >= 20
                                        else "frozen_v918_stack_plus_hard_clause_"
                                        "phase_direction_preserving_servo"
                                        if saved_grounding_objective >= 19
                                        else (
                                            "frozen_eraf_v917_stack_plus_phase_balanced_"
                                            "direct_geometry_residual_imitation"
                                        )
                                        if saved_grounding_objective >= 18
                                        else "frozen_eraf_v916_bridge_and_proposal_"
                                        "plus_direct_eef_relative_geometry_action_"
                                        "adapter"
                                    )
                                    if saved_grounding_objective >= 17
                                    else (
                                        "frozen_eraf_perception_proposal_and_legacy_"
                                        "bridge_plus_semantic_causal_action_"
                                        "grounding_bridge"
                                        if saved_grounding_objective >= 16
                                        else (
                                            "frozen_eraf_perception_plus_phase_"
                                            "conditioned_geometry_bridge_legacy_bridge_"
                                            "and_proposal"
                                            if saved_grounding_objective >= 15
                                            else "frozen_eraf_perception_plus_action_"
                                            "bridge_and_proposal"
                                        )
                                    )
                                )
                                expected_action_trainable_scope = (
                                    (
                                        "shared_video_action_lora_plus_eraf_action_"
                                        "context_injector"
                                        if saved_grounding_objective >= 26
                                        else "eraf_action_context_injector_only"
                                        if saved_grounding_objective >= 25
                                        else "clause_semantic_retention_residual_only"
                                        if saved_grounding_objective >= 24
                                        else "phase_specific_privileged_expert_residual_adapter_only"
                                        if saved_grounding_objective >= 21
                                        else "phase_compatible_local_waypoint_adapter_only"
                                        if saved_grounding_objective >= 20
                                        else "hard_routed_phase_servo_only"
                                        if saved_grounding_objective >= 19
                                        else (
                                            "phase_conditioned_geometry_adapter_only_"
                                            "with_phase_balanced_residual_imitation"
                                        )
                                        if saved_grounding_objective >= 18
                                        else "phase_conditioned_relative_geometry_"
                                        "action_adapter_only"
                                    )
                                    if saved_grounding_objective >= 17
                                    else (
                                        "semantic_causal_action_grounding_bridge_only"
                                        if saved_grounding_objective >= 16
                                        else (
                                            "phase_conditioned_subject_reference_anchor_"
                                            "action_bridge_plus_legacy_bridge_and_action_"
                                            "chunk_proposal"
                                            if saved_grounding_objective >= 15
                                            else "base_query_projection_relation_"
                                            "attention_query_embedding_delta_plus_action_"
                                            "chunk_proposal"
                                        )
                                    )
                                )
                                expected_v914_contract = {
                                    "eraf_training_stage": "action",
                                    "eraf_action_joint_training": True,
                                    "eraf_action_joint_contract": (
                                        expected_action_joint_contract
                                    ),
                                    "eraf_action_trainable_scope": (
                                        expected_action_trainable_scope
                                    ),
                                    "eraf_role_adapter_trainable_scope": (
                                        (
                                            "shared_video_action_lora_plus_eraf_"
                                            "action_context_injector"
                                            if saved_grounding_objective >= 26
                                            else "eraf_action_context_injector_only"
                                            if saved_grounding_objective >= 25
                                            else "clause_semantic_retention_residual_only"
                                            if saved_grounding_objective >= 24
                                            else "phase_specific_privileged_expert_residual_adapter_only"
                                            if saved_grounding_objective >= 21
                                            else "phase_compatible_local_waypoint_adapter_only"
                                            if saved_grounding_objective >= 20
                                            else "hard_routed_phase_servo_only"
                                            if saved_grounding_objective >= 19
                                            else (
                                                "phase_conditioned_geometry_adapter_"
                                                "only_with_phase_balanced_residual_"
                                                "imitation"
                                            )
                                            if saved_grounding_objective >= 18
                                            else "phase_conditioned_relative_"
                                            "geometry_action_adapter_only"
                                        )
                                        if saved_grounding_objective >= 17
                                        else (
                                            "semantic_causal_action_grounding_bridge_only"
                                            if saved_grounding_objective >= 16
                                            else "frozen_eraf_perception_action_bridge_"
                                            "plus_proposal"
                                        )
                                    ),
                                }
                                if saved_grounding_objective >= 25:
                                    expected_v914_contract.update(
                                        {
                                            "eraf_action_context_injection_contract": (
                                                "append_bounded_eraf_tokens_to_shared_"
                                                "action_expert_context_at_every_"
                                                "denoising_step_no_post_action_residual"
                                            ),
                                            "eraf_post_action_residual_active": False,
                                        }
                                    )
                                if saved_grounding_objective >= 26:
                                    expected_v914_contract.update(
                                        {
                                            "eraf_single_path": True,
                                            "gate_mode": "eraf_only",
                                        }
                                    )
                                for name, expected in expected_v914_contract.items():
                                    if metadata.get(name) != expected:
                                        raise ValueError(
                                            "PGC V9.14 checkpoint contract mismatch: "
                                            f"{name}={metadata.get(name)!r}, "
                                            f"expected={expected!r}."
                                        )
                            for metadata_name, expected_value in {
                                "eraf_phase_safe_memory_state_weight": (
                                    self.policy_guard_eraf_loss_weights.phase_safe_memory_state
                                ),
                                "eraf_phase_safe_memory_scheduler_weight": (
                                    self.policy_guard_eraf_loss_weights.phase_safe_memory_scheduler
                                ),
                                "eraf_phase_safe_memory_energy_weight": (
                                    self.policy_guard_eraf_loss_weights.phase_safe_memory_energy
                                ),
                                "eraf_phase_safe_memory_routing_residual_max_abs": (
                                    self.policy_guard_eraf_phase_safe_memory_routing_residual_max_abs
                                ),
                            }.items():
                                try:
                                    saved_value = float(metadata[metadata_name])
                                except (KeyError, TypeError, ValueError) as exc:
                                    raise ValueError(
                                        "PGC V9.13 checkpoint is missing valid "
                                        f"memory value {metadata_name!r}."
                                    ) from exc
                                if not math.isclose(
                                    saved_value,
                                    float(expected_value),
                                    rel_tol=0.0,
                                    abs_tol=1.0e-9,
                                ):
                                    raise ValueError(
                                        f"PGC V9.13 {metadata_name} mismatch: "
                                        f"checkpoint={saved_value}, "
                                        f"model={expected_value}."
                                    )
                            for metadata_name, expected_value in {
                                "eraf_phase_safe_memory_hidden_dim": (
                                    self.policy_guard_eraf_phase_safe_memory_hidden_dim
                                ),
                                "eraf_phase_safe_memory_state_count": (
                                    self.policy_guard_eraf_phase_safe_memory_state_count
                                ),
                            }.items():
                                try:
                                    saved_value = int(metadata[metadata_name])
                                except (KeyError, TypeError, ValueError) as exc:
                                    raise ValueError(
                                        "PGC V9.13 checkpoint is missing valid "
                                        f"memory dimension {metadata_name!r}."
                                    ) from exc
                                if saved_value != int(expected_value):
                                    raise ValueError(
                                        f"PGC V9.13 {metadata_name} mismatch: "
                                        f"checkpoint={saved_value}, "
                                        f"model={expected_value}."
                                    )
                        if (
                            bool(metadata.get("eraf_entity_only", False))
                            != self.policy_guard_eraf_entity_only
                            or bool(metadata.get("eraf_use_anchors", True))
                            != self.policy_guard_eraf_use_anchors
                        ):
                            raise ValueError(
                                "PGC v9 ERAF ablation configuration does not "
                                "match the checkpoint."
                            )
                        saved_stage = str(metadata.get("eraf_training_stage", ""))
                        if saved_stage not in {"grounding", "action", "verifier"}:
                            raise ValueError(
                                "PGC v9 checkpoint has an invalid ERAF training stage: "
                                f"{saved_stage!r}."
                            )
                        eraf = self.policy_guard_modules[
                            "entity_relation_affordance"
                        ]
                        for metadata_name, expected_value in {
                            "eraf_visual_aspect_ratio": float(
                                eraf.entity_grounder.visual_aspect_ratio
                            ),
                            "eraf_temperature": float(
                                eraf.entity_grounder.temperature
                            ),
                        }.items():
                            try:
                                saved_value = float(metadata[metadata_name])
                            except (KeyError, TypeError, ValueError) as exc:
                                raise ValueError(
                                    "PGC v9 checkpoint is missing valid ERAF value "
                                    f"{metadata_name!r}."
                                ) from exc
                            if not math.isclose(
                                saved_value,
                                expected_value,
                                rel_tol=0.0,
                                abs_tol=1.0e-9,
                            ):
                                raise ValueError(
                                    f"PGC v9 {metadata_name} mismatch: "
                                    f"checkpoint={saved_value}, "
                                    f"model={expected_value}."
                                )
            base_payload = self.load_checkpoint(resolved_base, optimizer=None)
            if base_payload.get("format") in {
                "fastwam_policy_guard_v1",
                "fastwam_policy_guard_v2",
                "fastwam_policy_guard_v3",
                "fastwam_policy_guard_v4",
                "fastwam_policy_guard_v5",
                "fastwam_policy_guard_v6",
                "fastwam_policy_guard_v7",
                "fastwam_policy_guard_v8",
                "fastwam_policy_guard_v9",
            }:
                raise ValueError(
                    "Nested PGC checkpoints are not supported as bases."
                )
            if int(saved_eraf_grounding_objective or 1) >= 26:
                current_adapter = self._lora_adapter_state_dict()
                expected_keys = set(current_adapter)
                saved_keys = set(shared_expert_lora)
                missing_adapter = sorted(expected_keys - saved_keys)
                unexpected_adapter = sorted(saved_keys - expected_keys)
                if missing_adapter or unexpected_adapter:
                    raise ValueError(
                        "PGC V9.26 shared Expert LoRA key mismatch: "
                        f"missing={missing_adapter[:20]}, "
                        f"unexpected={unexpected_adapter[:20]}."
                    )
                shape_mismatches = {
                    name: (
                        tuple(shared_expert_lora[name].shape),
                        tuple(current_adapter[name].shape),
                    )
                    for name in expected_keys
                    if tuple(shared_expert_lora[name].shape)
                    != tuple(current_adapter[name].shape)
                }
                if shape_mismatches:
                    raise ValueError(
                        "PGC V9.26 shared Expert LoRA shape mismatch: "
                        f"{shape_mismatches}."
                    )
                incompatible_lora = self.mot.load_state_dict(
                    shared_expert_lora, strict=False
                )
                if incompatible_lora.unexpected_keys:
                    raise ValueError(
                        "PGC V9.26 shared Expert LoRA contains unexpected "
                        f"tensors: {list(incompatible_lora.unexpected_keys)}."
                    )
            migrate_with_new_modules = (
                migrate_v5_to_target_binder
                or migrate_v5_to_v9
                or migrate_v9_to_role_adapter
                or migrate_v9_to_structured_role_adapter
                or migrate_v9_to_balanced_role_adapter
                or migrate_v9_to_clause_activation_adapter
                or migrate_v9_to_view_scheduler
                or migrate_v9_to_phase_rebinding
                or migrate_v9_to_phase_safe_memory
                or migrate_v914_to_v915_action_grounding
                or migrate_v916_to_v917_direct_geometry
                or migrate_v917_to_v918_phase_residual
                or migrate_v918_to_v919_phase_servo
                or migrate_v919_to_v920_waypoint
                or migrate_v920_to_v921_expert_alignment
                or migrate_v921_to_v923_alignment_preserving_clause
                or migrate_v921_to_v924_isolated_clause_residual
                or migrate_v924_to_v925_action_context
            )
            incompatible = self.policy_guard_modules.load_state_dict(
                guard_state, strict=not migrate_with_new_modules
            )
            if migrate_v5_to_target_binder:
                missing = list(incompatible.missing_keys)
                unexpected = list(incompatible.unexpected_keys)
                disallowed_missing = [
                    key
                    for key in missing
                    if not key.startswith("target_binder.")
                ]
                if disallowed_missing or unexpected or not missing:
                    raise ValueError(
                        "PGC v5 target-binder warm start has incompatible sidecars: "
                        f"missing={missing}, unexpected={unexpected}."
                    )
            elif migrate_v5_to_v9:
                missing = list(incompatible.missing_keys)
                unexpected = list(incompatible.unexpected_keys)
                disallowed_missing = [
                    key
                    for key in missing
                    if not key.startswith("entity_relation_affordance.")
                ]
                if disallowed_missing or unexpected or not missing:
                    raise ValueError(
                        "PGC v5 -> v9 warm start has incompatible sidecars: "
                        f"missing={missing}, unexpected={unexpected}."
                    )
            elif migrate_v9_to_role_adapter:
                missing = list(incompatible.missing_keys)
                unexpected = list(incompatible.unexpected_keys)
                disallowed_missing = [
                    key
                    for key in missing
                    if not key.startswith(
                        "entity_relation_affordance.role_assignment_adapter."
                    )
                ]
                if disallowed_missing or unexpected or not missing:
                    raise ValueError(
                        "PGC v9.1 -> v9.3 role-adapter warm start has "
                        f"incompatible sidecars: missing={missing}, "
                        f"unexpected={unexpected}."
                    )
            elif migrate_v9_to_structured_role_adapter:
                missing = list(incompatible.missing_keys)
                unexpected = list(incompatible.unexpected_keys)
                disallowed_missing = [
                    key
                    for key in missing
                    if not key.startswith(
                        "entity_relation_affordance."
                        "structured_role_assignment_adapter."
                    )
                ]
                if disallowed_missing or unexpected or not missing:
                    raise ValueError(
                        "PGC v9.3 -> v9.4 structured role-adapter warm start "
                        f"has incompatible sidecars: missing={missing}, "
                        f"unexpected={unexpected}."
                    )
            elif migrate_v9_to_balanced_role_adapter:
                missing = list(incompatible.missing_keys)
                unexpected = list(incompatible.unexpected_keys)
                disallowed_missing = [
                    key
                    for key in missing
                    if not key.startswith(
                        "entity_relation_affordance."
                        "balanced_role_binding_adapter."
                    )
                ]
                if disallowed_missing or unexpected or not missing:
                    raise ValueError(
                        "PGC v9.3 -> v9.5 balanced role-binding warm start "
                        f"has incompatible sidecars: missing={missing}, "
                        f"unexpected={unexpected}."
                    )
            elif migrate_v9_to_clause_activation_adapter:
                missing = list(incompatible.missing_keys)
                unexpected = list(incompatible.unexpected_keys)
                disallowed_missing = [
                    key
                    for key in missing
                    if not key.startswith(
                        "entity_relation_affordance."
                        "clause_activation_adapter."
                    )
                ]
                if disallowed_missing or unexpected or not missing:
                    raise ValueError(
                        "PGC v9.7 -> v9.8 clause-activation warm start has "
                        f"incompatible sidecars: missing={missing}, "
                        f"unexpected={unexpected}."
                    )
            elif migrate_v9_to_view_scheduler:
                missing = list(incompatible.missing_keys)
                unexpected = list(incompatible.unexpected_keys)
                allowed_prefixes = (
                    "entity_relation_affordance.entity_grounder."
                    "view_fusion_adapter.",
                    "entity_relation_affordance.clause_execution_scheduler.",
                )
                disallowed_missing = [
                    key
                    for key in missing
                    if not key.startswith(allowed_prefixes)
                ]
                if disallowed_missing or unexpected or not missing:
                    raise ValueError(
                        "PGC v9.8 -> v9.9 view/scheduler warm start has "
                        f"incompatible sidecars: missing={missing}, "
                        f"unexpected={unexpected}."
                    )
            elif migrate_v9_to_phase_rebinding:
                missing = list(incompatible.missing_keys)
                unexpected = list(incompatible.unexpected_keys)
                allowed_prefix = (
                    "entity_relation_affordance."
                    "closed_loop_phase_rebinding_adapter."
                )
                disallowed_missing = [
                    key for key in missing if not key.startswith(allowed_prefix)
                ]
                if disallowed_missing or unexpected or not missing:
                    raise ValueError(
                        "PGC v9.11 -> v9.12 phase-rebinding warm start has "
                        f"incompatible sidecars: missing={missing}, "
                        f"unexpected={unexpected}."
                    )
            elif migrate_v9_to_phase_safe_memory:
                missing = list(incompatible.missing_keys)
                unexpected = list(incompatible.unexpected_keys)
                allowed_prefix = (
                    "entity_relation_affordance.phase_safe_clause_memory."
                )
                disallowed_missing = [
                    key for key in missing if not key.startswith(allowed_prefix)
                ]
                if disallowed_missing or unexpected or not missing:
                    raise ValueError(
                        "PGC v9.11 -> v9.13 phase-safe-memory warm start has "
                        f"incompatible sidecars: missing={missing}, "
                        f"unexpected={unexpected}."
                    )
            elif migrate_v914_to_v915_action_grounding:
                missing = list(incompatible.missing_keys)
                unexpected = list(incompatible.unexpected_keys)
                allowed_prefix = "eraf_action_grounding_bridge."
                disallowed_missing = [
                    key for key in missing if not key.startswith(allowed_prefix)
                ]
                if disallowed_missing or unexpected or not missing:
                    raise ValueError(
                        "PGC V9.14 -> V9.15 action-grounding warm start has "
                        f"incompatible sidecars: missing={missing}, "
                        f"unexpected={unexpected}."
                    )
            elif migrate_v916_to_v917_direct_geometry:
                missing = list(incompatible.missing_keys)
                unexpected = list(incompatible.unexpected_keys)
                allowed_prefix = "eraf_geometry_action_adapter."
                disallowed_missing = [
                    key for key in missing if not key.startswith(allowed_prefix)
                ]
                if disallowed_missing or unexpected or not missing:
                    raise ValueError(
                        "PGC V9.16 -> V9.17 direct geometry-action warm start "
                        f"has incompatible sidecars: missing={missing}, "
                        f"unexpected={unexpected}."
                    )
            elif migrate_v917_to_v918_phase_residual:
                missing = list(incompatible.missing_keys)
                unexpected = list(incompatible.unexpected_keys)
                if missing or unexpected:
                    raise ValueError(
                        "PGC V9.17 -> V9.18 phase-residual warm start must "
                        "restore every existing tensor exactly: "
                        f"missing={missing}, unexpected={unexpected}."
                    )
            elif migrate_v918_to_v919_phase_servo:
                missing = list(incompatible.missing_keys)
                unexpected = list(incompatible.unexpected_keys)
                allowed_prefix = "eraf_hard_routed_phase_servo."
                disallowed_missing = [
                    key for key in missing if not key.startswith(allowed_prefix)
                ]
                if disallowed_missing or unexpected or not missing:
                    raise ValueError(
                        "PGC V9.18 -> V9.19 hard-routed phase-servo warm start "
                        f"has incompatible sidecars: missing={missing}, "
                        f"unexpected={unexpected}."
                    )
            elif migrate_v919_to_v920_waypoint:
                missing = list(incompatible.missing_keys)
                unexpected = list(incompatible.unexpected_keys)
                allowed_prefix = "eraf_phase_compatible_waypoint_adapter."
                disallowed_missing = [
                    key for key in missing if not key.startswith(allowed_prefix)
                ]
                if disallowed_missing or unexpected or not missing:
                    raise ValueError(
                        "PGC V9.19 -> V9.20 waypoint warm start has "
                        f"incompatible sidecars: missing={missing}, "
                        f"unexpected={unexpected}."
                    )
            elif migrate_v920_to_v921_expert_alignment:
                missing = list(incompatible.missing_keys)
                unexpected = list(incompatible.unexpected_keys)
                allowed_prefix = "eraf_phase_expert_residual_adapter."
                disallowed_missing = [
                    key for key in missing if not key.startswith(allowed_prefix)
                ]
                if disallowed_missing or unexpected or not missing:
                    raise ValueError(
                        "PGC V9.20 -> V9.21 expert-alignment warm start has "
                        f"incompatible sidecars: missing={missing}, "
                        f"unexpected={unexpected}."
                    )
            elif migrate_v921_to_v923_alignment_preserving_clause:
                missing = list(incompatible.missing_keys)
                unexpected = list(incompatible.unexpected_keys)
                allowed_prefix = "eraf_phase_expert_residual_teacher."
                disallowed_missing = [
                    key for key in missing if not key.startswith(allowed_prefix)
                ]
                if disallowed_missing or unexpected or not missing:
                    raise ValueError(
                        "PGC V9.21 -> V9.23 alignment-preserving clause warm "
                        f"start has incompatible sidecars: missing={missing}, "
                        f"unexpected={unexpected}."
                    )
                teacher = self.policy_guard_modules[
                    "eraf_phase_expert_residual_teacher"
                ]
                teacher.load_state_dict(
                    self.policy_guard_modules[
                        "eraf_phase_expert_residual_adapter"
                    ].state_dict(),
                    strict=True,
                )
                teacher.eval()
                teacher.requires_grad_(False)
            elif migrate_v921_to_v924_isolated_clause_residual:
                missing = list(incompatible.missing_keys)
                unexpected = list(incompatible.unexpected_keys)
                allowed_prefix = "eraf_clause_semantic_retention_residual."
                disallowed_missing = [
                    key for key in missing if not key.startswith(allowed_prefix)
                ]
                if disallowed_missing or unexpected or not missing:
                    raise ValueError(
                        "PGC V9.21 -> V9.24 isolated clause-residual warm start "
                        f"has incompatible sidecars: missing={missing}, "
                        f"unexpected={unexpected}."
                    )
            elif migrate_v924_to_v925_action_context:
                missing = list(incompatible.missing_keys)
                unexpected = list(incompatible.unexpected_keys)
                allowed_prefix = "eraf_action_context_injector."
                disallowed_missing = [
                    key for key in missing if not key.startswith(allowed_prefix)
                ]
                if disallowed_missing or unexpected or not missing:
                    raise ValueError(
                        "PGC V9.24 -> V9.25 internal action-context warm start "
                        f"has incompatible sidecars: missing={missing}, "
                        f"unexpected={unexpected}."
                    )
            elif saved_policy_guard_version == 6:
                self._load_policy_guard_target_prototype_state(
                    target_prototype_state
                )
            self.policy_guard_base_checkpoint = resolved_base
            self.policy_guard_legacy_full_loaded = False
            if (
                optimizer is not None
                and "optimizer" in payload
                and not migrate_v5_to_target_binder
                and not migrate_v5_to_v8
                and not migrate_v5_to_v9
                and not migrate_v9_to_role_adapter
                and not migrate_v9_to_structured_role_adapter
                and not migrate_v9_to_balanced_role_adapter
                and not migrate_v9_to_clause_activation_adapter
                and not migrate_v9_to_view_scheduler
                and not migrate_v9_to_exclusive_all_entity
                and not migrate_v9_to_clause_tuple
                and not migrate_v9_to_phase_rebinding
                and not migrate_v9_to_phase_safe_memory
                and not migrate_v914_to_v915_action_grounding
                and not migrate_v915_to_v916_semantic_causal
                and not migrate_v916_to_v917_direct_geometry
                and not migrate_v917_to_v918_phase_residual
                and not migrate_v918_to_v919_phase_servo
                and not migrate_v919_to_v920_waypoint
                and not migrate_v920_to_v921_expert_alignment
                and not migrate_v921_to_v922_clause_ranking
                and not migrate_v921_to_v923_alignment_preserving_clause
                and not migrate_v921_to_v924_isolated_clause_residual
                and not migrate_v924_to_v925_action_context
                and not migrate_v925_to_v926_shared_expert_lora
            ):
                optimizer.load_state_dict(payload["optimizer"])
            if migrate_v5_to_target_binder:
                logger.info(
                    "Warm-started PGC v%d from validated PGC v5 sidecars at %s "
                    "(base=%s restored=%d new_target_binder=%d).",
                    self.policy_guard_version,
                    path,
                    resolved_base,
                    len(guard_state),
                    len(incompatible.missing_keys),
                )
            elif migrate_v5_to_v8:
                logger.info(
                    "Warm-started PGC v8 from exact validated PGC v5 "
                    "sidecars at %s (base=%s restored=%d new_tensors=0).",
                    path,
                    resolved_base,
                    len(guard_state),
                )
            elif migrate_v5_to_v9:
                logger.info(
                    "Warm-started PGC v9 from exact validated PGC v5 "
                    "sidecars at %s (base=%s restored=%d new_eraf=%d).",
                    path,
                    resolved_base,
                    len(guard_state),
                    len(incompatible.missing_keys),
                )
            elif migrate_v9_to_role_adapter:
                logger.info(
                    "Warm-started PGC v9.3 from frozen V9.1 ERAF at %s "
                    "(base=%s restored=%d new_role_adapter=%d).",
                    path,
                    resolved_base,
                    len(guard_state),
                    len(incompatible.missing_keys),
                )
            elif migrate_v9_to_structured_role_adapter:
                logger.info(
                    "Warm-started PGC v9.4 from frozen V9.3 ERAF at %s "
                    "(base=%s restored=%d new_structured_role_adapter=%d).",
                    path,
                    resolved_base,
                    len(guard_state),
                    len(incompatible.missing_keys),
                )
            elif migrate_v9_to_balanced_role_adapter:
                logger.info(
                    "Warm-started PGC v9.5 from frozen V9.3 ERAF at %s "
                    "(base=%s restored=%d new_balanced_role_adapter=%d).",
                    path,
                    resolved_base,
                    len(guard_state),
                    len(incompatible.missing_keys),
                )
            elif migrate_v9_to_clause_activation_adapter:
                logger.info(
                    "Warm-started PGC v9.8 from frozen V9.7 ERAF at %s "
                    "(base=%s restored=%d new_clause_adapter=%d).",
                    path,
                    resolved_base,
                    len(guard_state),
                    len(incompatible.missing_keys),
                )
            elif migrate_v9_to_view_scheduler:
                logger.info(
                    "Warm-started PGC v9.9 from frozen V9.8 ERAF at %s "
                    "(base=%s restored=%d new_view_scheduler=%d).",
                    path,
                    resolved_base,
                    len(guard_state),
                    len(incompatible.missing_keys),
                )
            elif migrate_v9_to_exclusive_all_entity:
                logger.info(
                    "Warm-started PGC v9.10 from frozen V9.9 ERAF at %s "
                    "(base=%s restored=%d new_tensors=0).",
                    path,
                    resolved_base,
                    len(guard_state),
                )
            elif migrate_v9_to_clause_tuple:
                logger.info(
                    "Warm-started PGC v9.11 from frozen V9.10 ERAF at %s "
                    "(base=%s restored=%d new_tensors=0).",
                    path,
                    resolved_base,
                    len(guard_state),
                )
            elif migrate_v9_to_phase_rebinding:
                logger.info(
                    "Warm-started PGC v9.12 from frozen V9.11 ERAF at %s "
                    "(base=%s restored=%d new_phase_rebinding=%d).",
                    path,
                    resolved_base,
                    len(guard_state),
                    len(incompatible.missing_keys),
                )
            elif migrate_v9_to_phase_safe_memory:
                logger.info(
                    "Warm-started PGC v9.13 from frozen V9.11 ERAF at %s "
                    "(base=%s restored=%d new_phase_safe_memory=%d).",
                    path,
                    resolved_base,
                    len(guard_state),
                    len(incompatible.missing_keys),
                )
            elif migrate_v914_to_v915_action_grounding:
                logger.info(
                    "Warm-started PGC v9.15 from exact V9.14 joint action "
                    "sidecars at %s (base=%s restored=%d "
                    "new_action_grounding=%d).",
                    path,
                    resolved_base,
                    len(guard_state),
                    len(incompatible.missing_keys),
                )
            elif migrate_v915_to_v916_semantic_causal:
                logger.info(
                    "Warm-started PGC v9.16 semantic causal calibration from "
                    "exact V9.15 action sidecars at %s (base=%s restored=%d "
                    "new_tensors=0).",
                    path,
                    resolved_base,
                    len(guard_state),
                )
            elif migrate_v916_to_v917_direct_geometry:
                logger.info(
                    "Warm-started PGC v9.17 direct geometry-action adapter from "
                    "exact V9.16 action sidecars at %s (base=%s restored=%d "
                    "new_geometry_tensors=%d).",
                    path,
                    resolved_base,
                    len(guard_state),
                    len(incompatible.missing_keys),
                )
            elif migrate_v917_to_v918_phase_residual:
                logger.info(
                    "Warm-started PGC v9.18 phase-residual calibration from "
                    "exact V9.17 action sidecars at %s (base=%s restored=%d "
                    "new_tensors=0).",
                    path,
                    resolved_base,
                    len(guard_state),
                )
            elif migrate_v918_to_v919_phase_servo:
                logger.info(
                    "Warm-started PGC v9.19 hard-routed phase servo from "
                    "exact V9.18 action sidecars at %s (base=%s restored=%d "
                    "new_servo_tensors=%d).",
                    path,
                    resolved_base,
                    len(guard_state),
                    len(incompatible.missing_keys),
                )
            elif migrate_v919_to_v920_waypoint:
                logger.info(
                    "Warm-started PGC v9.20 phase-compatible waypoint field "
                    "from exact V9.19 action sidecars at %s (base=%s "
                    "restored=%d new_waypoint_tensors=%d).",
                    path,
                    resolved_base,
                    len(guard_state),
                    len(incompatible.missing_keys),
                )
            elif migrate_v920_to_v921_expert_alignment:
                logger.info(
                    "Warm-started PGC v9.21 phase-specific expert residual "
                    "alignment from exact V9.20 action sidecars at %s "
                    "(base=%s restored=%d new_expert_tensors=%d).",
                    path,
                    resolved_base,
                    len(guard_state),
                    len(incompatible.missing_keys),
                )
            elif migrate_v921_to_v922_clause_ranking:
                logger.info(
                    "Warm-started PGC v9.22 balanced final-action clause "
                    "ranking from exact V9.21 action sidecars at %s "
                    "(base=%s restored=%d new_tensors=0).",
                    path,
                    resolved_base,
                    len(guard_state),
                )
            elif migrate_v921_to_v923_alignment_preserving_clause:
                logger.info(
                    "Warm-started PGC v9.23 alignment-preserving clause "
                    "ranking from exact V9.21 action sidecars at %s "
                    "(base=%s restored=%d teacher_tensors=%d).",
                    path,
                    resolved_base,
                    len(guard_state),
                    len(incompatible.missing_keys),
                )
            elif migrate_v921_to_v924_isolated_clause_residual:
                logger.info(
                    "Warm-started PGC v9.24 isolated clause semantic residual "
                    "from exact V9.21 action sidecars at %s "
                    "(base=%s restored=%d new_clause_residual_tensors=%d).",
                    path,
                    resolved_base,
                    len(guard_state),
                    len(incompatible.missing_keys),
                )
            elif migrate_v924_to_v925_action_context:
                logger.info(
                    "Warm-started PGC v9.25 internal ERAF Action-Expert "
                    "context injection from exact V9.24 action sidecars at %s "
                    "(base=%s restored=%d new_context_tensors=%d).",
                    path,
                    resolved_base,
                    len(guard_state),
                    len(incompatible.missing_keys),
                )
            elif migrate_v925_to_v926_shared_expert_lora:
                logger.info(
                    "Warm-started PGC v9.26 single ERAF path with zero-init "
                    "shared Video/Action Expert LoRA from exact V9.25 action "
                    "sidecars at %s (base=%s restored=%d lora_tensors=%d).",
                    path,
                    resolved_base,
                    len(guard_state),
                    len(self._lora_adapter_state_dict()),
                )
            else:
                logger.info(
                    "Loaded PGC v%d checkpoint from %s (base=%s "
                    "sidecar_and_guard_tensors=%d).",
                    saved_policy_guard_version,
                    path,
                    resolved_base,
                    len(guard_state),
                )
            return payload

        if self.policy_guard_action_expert is None:
            raise ValueError(
                "PGC v1/v2 checkpoint loading requires its independent "
                "Action Expert."
            )
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
            "fastwam_policy_guard_v3",
            "fastwam_policy_guard_v4",
            "fastwam_policy_guard_v5",
            "fastwam_policy_guard_v6",
            "fastwam_policy_guard_v7",
            "fastwam_policy_guard_v8",
            "fastwam_policy_guard_v9",
        }:
            raise ValueError("Nested PGC checkpoints are not supported as bases.")

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
            "fastwam_policy_guard_v3",
            "fastwam_policy_guard_v4",
            "fastwam_policy_guard_v5",
            "fastwam_policy_guard_v6",
            "fastwam_policy_guard_v7",
            "fastwam_policy_guard_v8",
            "fastwam_policy_guard_v9",
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
