import logging
import json
import inspect
import os
import re
from math import ceil
from pathlib import Path
import time

import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import DictConfig
from PIL import Image
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from .utils.fs import ensure_dir
from .utils.logging_config import get_logger, setup_logging
from .utils.pytorch_utils import set_global_seed
from .utils.samplers import ResumableEpochSampler
from .training_progress import optimizer_step_to_sampler_position
from .utils.video_io import save_mp4
from .utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim

logger = get_logger(__name__)


SAFE_GAIN_FULL_POLICY_RESUME_OBJECTIVES = frozenset({29, 30, 31, 32})


def _is_safe_gain_full_policy_resume(model, resume) -> bool:
    """Return whether ``resume`` is the complete safe-gain policy warm start."""

    return bool(resume) and (
        int(getattr(model, "policy_guard_version", 0)) == 9
        and int(
            getattr(
                model,
                "policy_guard_eraf_grounding_objective_version",
                0,
            )
        )
        in SAFE_GAIN_FULL_POLICY_RESUME_OBJECTIVES
        and bool(
            getattr(
                model,
                "policy_guard_eraf_safe_gain_training",
                False,
            )
        )
    )


class Wan22Trainer:
    def __init__(self, model, train_dataset, val_dataset=None, *, cfg: DictConfig):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = cfg
        self.output_dir = str(cfg.output_dir)
        self.learning_rate = float(cfg.learning_rate)
        self.weight_decay = float(cfg.weight_decay)
        self.batch_size = int(cfg.batch_size)
        self.num_workers = int(cfg.num_workers)
        self.num_epochs = int(cfg.num_epochs)
        max_steps = cfg.max_steps
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.log_every = int(cfg.log_every)
        self.save_every = int(cfg.save_every)
        self.eval_every = int(cfg.eval_every)
        self.eval_num_inference_steps = int(cfg.eval_num_inference_steps)
        self.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
        self.max_grad_norm = float(cfg.max_grad_norm)
        self.seed = int(cfg.seed)
        self.save_training_state = bool(cfg.get("save_training_state", True))
        
        self.resume = cfg.resume
        raw_weight_only_start_step = cfg.get("weight_only_start_step", None)
        self.weight_only_start_step = (
            None
            if raw_weight_only_start_step in (None, "", "null")
            else int(raw_weight_only_start_step)
        )
        if (
            self.weight_only_start_step is not None
            and self.weight_only_start_step < 0
        ):
            raise ValueError("`weight_only_start_step` must be non-negative.")
        if self.weight_only_start_step is not None and not self.resume:
            raise ValueError(
                "`weight_only_start_step` requires a weight checkpoint in "
                "`resume`."
            )
        self.mixed_precision = str(cfg.mixed_precision).strip().lower()
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(
                f"Unsupported mixed_precision: {cfg.mixed_precision}. "
                "Expected one of: ['no', 'fp16', 'bf16']."
            )
        self.wandb_enabled = bool(cfg.wandb.enabled)

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            step_scheduler_with_optimizer=False,
        )
        
        logger.info(
            "Accelerate training: distributed_type=%s zero_stage=%s world_size=%d process_index=%d cfg_mixed_precision=%s accelerator_mixed_precision=%s grad_accum=%d grad_clip=%.4f",
            self.accelerator.distributed_type,
            self.accelerator.state.deepspeed_plugin.deepspeed_config.get("zero_optimization", {}).get("stage", "unknown"),
            self.accelerator.num_processes,
            self.accelerator.process_index,
            self.mixed_precision,
            self.accelerator.mixed_precision,
            self.gradient_accumulation_steps,
            self.max_grad_norm,
        )
        logger.info("using accelerator.device=%s", self.accelerator.device)
        worker_init_fn = set_global_seed(self.seed, get_worker_init_fn=True)
        self._assert_dataset_length_consistent(self.train_dataset, "train_dataset")
        if self.val_dataset is not None:
            self._assert_dataset_length_consistent(self.val_dataset, "val_dataset")
        if bool(
            getattr(
                self.model,
                "lora_paired_language_control_enabled",
                False,
            )
        ):
            required_dataset_contracts = {
                "pgc_has_counterfactual_data": True,
                "pgc_balance_native_counterfactual": True,
                "pgc_entity_relation_supervision_required": True,
                "pgc_v9_balanced_sampling": True,
                "pgc_v9_phase_safe_memory": True,
            }
            if bool(
                getattr(
                    self.model,
                    "lora_paired_language_control_config",
                    {},
                ).get("bidirectional_supervision", False)
            ):
                required_dataset_contracts[
                    "pgc_bidirectional_language_supervision_required"
                ] = True
            mismatches = {
                name: bool(getattr(self.train_dataset, name, False))
                for name, expected in required_dataset_contracts.items()
                if bool(getattr(self.train_dataset, name, False)) != expected
            }
            closed_loop_count = int(
                getattr(
                    self.train_dataset,
                    "pgc_v9_closed_loop_native_dataset_count",
                    0,
                )
            )
            if mismatches or closed_loop_count != 1:
                raise ValueError(
                    "The no-ERAF LoRA control requires the exact V9.26 "
                    "offline-native/closed-loop-native/historical-CF/strict-CF "
                    "1:1:1:1 dataset contract; "
                    f"mismatches={mismatches}, "
                    f"closed_loop_native_count={closed_loop_count}."
                )
        if bool(
            getattr(
                self.model,
                "policy_guard_eraf_end_to_end_joint_training",
                False,
            )
        ):
            required_dataset_contracts = {
                "pgc_has_counterfactual_data": True,
                "pgc_balance_native_counterfactual": True,
                "pgc_entity_relation_supervision_required": True,
                "pgc_bidirectional_language_supervision_required": True,
                "pgc_v9_balanced_sampling": True,
                "pgc_v9_phase_safe_memory": True,
            }
            mismatches = {
                name: bool(getattr(self.train_dataset, name, False))
                for name, expected in required_dataset_contracts.items()
                if bool(getattr(self.train_dataset, name, False)) != expected
            }
            closed_loop_count = int(
                getattr(
                    self.train_dataset,
                    "pgc_v9_closed_loop_native_dataset_count",
                    0,
                )
            )
            if mismatches or closed_loop_count != 1:
                raise ValueError(
                    "End-to-end ERAF joint training requires the exact no-ERAF "
                    "offline-native/closed-loop-native/historical-CF/strict-CF "
                    "1:1:1:1 dataset and bidirectional supervision contract; "
                    f"mismatches={mismatches}, "
                    f"closed_loop_native_count={closed_loop_count}."
                )
        safe_gain_full_policy_resume = _is_safe_gain_full_policy_resume(
            self.model,
            self.resume,
        )
        if (
            bool(
                getattr(
                    self.model,
                    "policy_guard_eraf_pretrained_joint_training",
                    False,
                )
            )
            and not getattr(
                self.model, "policy_guard_eraf_pretrained_checkpoint", None
            )
            and not safe_gain_full_policy_resume
        ):
            raise ValueError(
                "Pretrained ERAF joint training requires "
                "entity_relation_grounding.pretrained_checkpoint unless "
                "objective-29+ safe-gain training restores a validated full "
                "PGC policy through resume."
            )
        if bool(getattr(self.model, "policy_guard_enabled", False)) and bool(
            getattr(
                self.model,
                "policy_guard_require_direct_counterfactual_actions",
                False,
            )
        ):
            if not bool(
                getattr(
                    self.train_dataset, "pgc_has_counterfactual_data", False
                )
            ):
                raise ValueError(
                    "PGC requires at least one state-aligned direct "
                    "counterfactual LeRobot dataset. Set "
                    "`data.train.pgc_counterfactual_dataset_dirs`."
                )
            if int(getattr(self.model, "policy_guard_version", 1)) >= 2 and (
                not bool(
                    getattr(
                        self.train_dataset,
                        "pgc_balance_native_counterfactual",
                        False,
                    )
                )
            ):
                raise ValueError(
                    "PGC v2/v3 requires explicit 1:1 native/counterfactual "
                    "sampling. Set "
                    "`data.train.pgc_balance_native_counterfactual=true`."
                )
            if bool(
                getattr(
                    self.model,
                    "policy_guard_completion_phase_enabled",
                    False,
                )
            ) and not bool(
                getattr(
                    self.train_dataset,
                    "pgc_completion_phase_supervision_required",
                    False,
                )
            ):
                raise ValueError(
                    "PGC V5-completion requires the audited phase sidecar. "
                    "Set data.train.pgc_completion_phase_supervision_required=true."
                )
            if int(getattr(self.model, "policy_guard_version", 1)) == 8:
                if not bool(
                    getattr(
                        self.train_dataset,
                        "pgc_has_closed_loop_corrective_data",
                        False,
                    )
                ):
                    raise ValueError(
                        "PGC v8 requires a replay-verified closed-loop "
                        "corrective LeRobot dataset. Set "
                        "data.train.pgc_closed_loop_corrective_dataset_dirs."
                    )
                if not bool(
                    getattr(
                        self.model,
                        "policy_guard_closed_loop_train_proposal_only",
                        False,
                    )
                ):
                    raise ValueError(
                        "PGC v8 must preserve V5 language/gating sidecars and "
                        "train only the ActionChunkProposal."
                    )
            if int(getattr(self.model, "policy_guard_version", 1)) == 9:
                if not bool(
                    getattr(
                        self.train_dataset,
                        "pgc_entity_relation_supervision_required",
                        False,
                    )
                ):
                    raise ValueError(
                        "PGC v9 requires audited entity-relation supervision. "
                        "Set data.train.pgc_entity_relation_supervision_required=true "
                        "and provide one sidecar per dataset."
                    )
                objective_version = int(
                    getattr(
                        self.model,
                        "policy_guard_eraf_grounding_objective_version",
                        1,
                    )
                )
                hard_curriculum = bool(
                    getattr(
                        self.train_dataset,
                        "pgc_v9_hard_role_curriculum",
                        False,
                    )
                )
                training_stage = str(
                    getattr(
                        self.model,
                        "policy_guard_eraf_training_stage",
                        "",
                    )
                )
                if (
                    objective_version in {7, 8, 12}
                    and training_stage == "grounding"
                    and not hard_curriculum
                ):
                    raise ValueError(
                        "PGC V9.6/V9.7/V9.11 grounding requires its audited native hard/easy "
                        "curriculum."
                    )
                if hard_curriculum and not (
                    objective_version in {7, 8, 12}
                    and training_stage == "grounding"
                ):
                    raise ValueError(
                        "PGC hard-role curriculum is valid only for "
                        "objective-v7/v8/v12 grounding repair."
                    )
                closed_loop_rebinding = bool(
                    getattr(
                        self.train_dataset,
                        "pgc_v9_closed_loop_rebinding",
                        False,
                    )
                )
                phase_safe_memory = bool(
                    getattr(
                        self.train_dataset,
                        "pgc_v9_phase_safe_memory",
                        False,
                    )
                )
                completion_only_memory = bool(
                    getattr(
                        self.model,
                        "policy_guard_eraf_completion_only_memory",
                        False,
                    )
                )
                action_joint_training = bool(
                    getattr(
                        self.model,
                        "policy_guard_eraf_action_joint_training",
                        False,
                    )
                )
                safe_gain_counterfactual_replay = bool(
                    getattr(
                        self.train_dataset,
                        "pgc_v9_safe_gain_counterfactual_replay",
                        False,
                    )
                )
                if objective_version >= 29 and not safe_gain_counterfactual_replay:
                    raise ValueError(
                        "PGC V9.29 requires the replay-verified closed-loop "
                        "counterfactual sampling contract."
                    )
                if safe_gain_counterfactual_replay and objective_version < 29:
                    raise ValueError(
                        "PGC V9.29 replay sampling is invalid for earlier "
                        "grounding objectives."
                    )
                if (
                    objective_version == 13
                    and training_stage == "grounding"
                    and not closed_loop_rebinding
                ):
                    raise ValueError(
                        "PGC V9.12 grounding requires the audited four-way "
                        "closed-loop rebinding curriculum."
                    )
                if closed_loop_rebinding and not (
                    objective_version == 13 and training_stage == "grounding"
                ):
                    raise ValueError(
                        "PGC V9.12 closed-loop rebinding data is valid only for "
                        "objective-v13 grounding repair."
                    )
                if (
                    objective_version == 14
                    and training_stage == "grounding"
                    and not phase_safe_memory
                ):
                    raise ValueError(
                        "PGC V9.13 grounding requires the audited four-way "
                        "phase-safe memory curriculum."
                    )
                valid_phase_safe_stage = objective_version >= 14 and (
                    training_stage == "grounding"
                    or (
                        training_stage == "action"
                        and completion_only_memory
                        and action_joint_training
                    )
                )
                if phase_safe_memory and not valid_phase_safe_stage:
                    raise ValueError(
                        "PGC phase-memory data is valid only for objective-v14 "
                        "V9.13 grounding or objective-v14+ completion-only "
                        "joint action training."
                    )
                if (
                    objective_version >= 14
                    and training_stage == "action"
                    and action_joint_training
                    and not phase_safe_memory
                ):
                    raise ValueError(
                        "PGC V9.14+ joint action training requires the audited "
                        "four-way phase-memory curriculum."
                    )

        # Freeze non-trainable modules before optimizer/deepspeed initialization.
        # In LoRA mode only adapters plus explicitly selected small modules are
        # exposed to the optimizer; full fine-tuning retains the old behavior.
        self._apply_dit_only_train_mode(self.model)
        transition_modules = getattr(
            self.model, "transition_contract_modules", None
        )
        transition_contract_enabled = bool(
            getattr(self.model, "transition_contract_enabled", False)
        )
        transition_parameter_ids = (
            {
                id(parameter)
                for parameter in transition_modules.parameters()
            }
            if transition_modules is not None
            else set()
        )
        transition_params = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
            and id(parameter) in transition_parameter_ids
        ]
        policy_params = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
            and id(parameter) not in transition_parameter_ids
        ]
        trainable_params = policy_params + transition_params
        if not trainable_params:
            raise ValueError("No trainable parameters were selected for optimization.")
        trainable_count = sum(parameter.numel() for parameter in trainable_params)
        total_count = sum(parameter.numel() for parameter in self.model.parameters())
        logger.info(
            "Optimizer parameters: trainable=%d total=%d fraction=%.6f",
            trainable_count,
            total_count,
            trainable_count / max(total_count, 1),
        )
        optimizer_groups = []
        policy_guard_group_builder = getattr(
            self.model, "policy_guard_optimizer_groups", None
        )
        policy_guard_groups = (
            policy_guard_group_builder(self.learning_rate)
            if callable(policy_guard_group_builder)
            else None
        )
        if policy_guard_groups is not None:
            optimizer_groups.extend(policy_guard_groups)
            logger.info(
                "PGC v9 optimizer groups: %s",
                [
                    {
                        "name": group.get("pgc_v9_group"),
                        "lr": float(group["lr"]),
                        "tensors": len(group["params"]),
                    }
                    for group in optimizer_groups
                ],
            )
        elif transition_contract_enabled:
            if policy_params:
                optimizer_groups.append(
                    {
                        "params": policy_params,
                        "tc_recovery_group": "policy",
                    }
                )
            if transition_params:
                optimizer_groups.append(
                    {
                        "params": transition_params,
                        "tc_recovery_group": "router",
                    }
                )
        else:
            optimizer_groups.append({"params": trainable_params})
        self._tc_optimizer_group_kinds = [
            group.get("tc_recovery_group") for group in optimizer_groups
        ]
        self._tc_recovery_base_lrs = [
            float(group.get("lr", self.learning_rate))
            for group in optimizer_groups
        ]
        self.optimizer = torch.optim.AdamW(
            optimizer_groups,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )
        
        self.train_loader = self._build_loader(self.train_dataset, worker_init_fn=worker_init_fn)
        total_train_steps = self._estimate_total_train_steps()
        self.max_steps = total_train_steps
        scheduler_train_steps = total_train_steps
        if self.weight_only_start_step is not None:
            if self.weight_only_start_step >= total_train_steps:
                raise ValueError(
                    "Weight-only continuation must end after its start: "
                    f"start={self.weight_only_start_step}, "
                    f"max_steps={total_train_steps}."
                )
            scheduler_train_steps -= self.weight_only_start_step
            logger.info(
                "Weight-only continuation schedule: absolute_step=%d->%d "
                "fresh_optimizer_steps=%d.",
                self.weight_only_start_step,
                total_train_steps,
                scheduler_train_steps,
            )
        warmup_steps = int(scheduler_train_steps * 0.05)
        self.scheduler = self._build_scheduler(
            scheduler_type=cfg.lr_scheduler_type,
            total_train_steps=scheduler_train_steps,
            warmup_steps=warmup_steps,
        )
        self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0

        self.checkpoint_root = os.path.join(self.output_dir, "checkpoints")
        self.weights_dir = os.path.join(self.checkpoint_root, "weights")
        self.state_dir = os.path.join(self.checkpoint_root, "state")
        self.eval_dir = os.path.join(self.output_dir, "eval")

        ensure_dir(self.output_dir)
        ensure_dir(self.checkpoint_root)
        ensure_dir(self.weights_dir)
        ensure_dir(self.state_dir)
        ensure_dir(self.eval_dir)

        self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.scheduler
        )
        if len(self.optimizer.param_groups) != len(self._tc_optimizer_group_kinds):
            raise RuntimeError(
                "Accelerate changed the number of optimizer parameter groups; "
                "TC-C recovery cannot safely mask the restored M1 policy."
            )
        for group, group_kind in zip(
            self.optimizer.param_groups,
            self._tc_optimizer_group_kinds,
        ):
            if group_kind is not None:
                group["tc_recovery_group"] = group_kind
        self._tc_recovery_base_lrs = [
            float(group.get("lr", self.learning_rate))
            for group in self.optimizer.param_groups
        ]
        scheduler_last_lrs = getattr(self.scheduler, "_last_lr", None)
        if isinstance(scheduler_last_lrs, list) and len(scheduler_last_lrs) == len(
            self._tc_recovery_base_lrs
        ):
            scheduler_last_lrs[:] = self._tc_recovery_base_lrs
        self.optimizer.zero_grad(set_to_none=True)
        self.wandb_run = None
        self._init_wandb()
        self._resume_or_load_checkpoint()

        val_size = len(self.val_dataset) if self.val_dataset is not None else len(self.train_dataset)
        logger.info("Train/val dataset size: %d/%d", len(self.train_dataset), val_size)

    def _init_wandb(self):
        if not self.wandb_enabled or not self.accelerator.is_main_process:
            return
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb logging is enabled in config (`wandb.enabled=true`) but wandb is not installed."
            ) from e

        self.wandb_run = wandb.init(
            entity=self.cfg.wandb.workspace,
            project=self.cfg.wandb.project,
            name=self.cfg.wandb.name,
            group=None if self.cfg.wandb.group in (None, "null", "") else str(self.cfg.wandb.group),
            mode=self.cfg.wandb.mode,
            dir=self.output_dir,
        )
        logger.info(
            "Initialized wandb run: workspace=%s project=%s name=%s",
            self.cfg.wandb.workspace,
            self.cfg.wandb.project,
            self.cfg.wandb.name,
        )

    def _wandb_log(self, payload: dict):
        if self.wandb_run is None:
            return
        self.wandb_run.log(payload, step=self.global_step)

    def _finish_wandb(self):
        if self.wandb_run is None:
            return
        self.wandb_run.finish()
        self.wandb_run = None

    def _build_loader(self, dataset, worker_init_fn=None):
        self.train_sampler = ResumableEpochSampler(
            dataset=dataset,
            seed=self.seed,
            batch_size=self.batch_size,
            num_processes=self.accelerator.num_processes,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn,
            drop_last=bool(
                (
                    getattr(self.model, "transition_contract_enabled", False)
                    or getattr(self.model, "policy_guard_enabled", False)
                )
                and self.accelerator.num_processes > 1
            ),
        )

    def _assert_dataset_length_consistent(self, dataset, dataset_name: str):
        if not hasattr(dataset, "__len__"):
            raise TypeError(f"`{dataset_name}` must implement __len__ for rank consistency checks.")

        local_length = len(dataset)
        gathered_lengths = self.accelerator.gather(
            torch.tensor([local_length], device=self.accelerator.device, dtype=torch.int64)
        ).reshape(-1)
        if torch.all(gathered_lengths == gathered_lengths[0]):
            return

        if self.accelerator.is_main_process:
            print(f"[dataset-check] {dataset_name} length mismatch across ranks after initialization:")
            for rank, rank_length in enumerate(gathered_lengths.cpu().tolist()):
                print(f"rank {rank}: {rank_length}")
        self.accelerator.wait_for_everyone()
        raise RuntimeError(
            f"{dataset_name} length mismatch across ranks: {gathered_lengths.cpu().tolist()}"
        )

    def _estimate_total_train_steps(self) -> int:
        if self.max_steps is not None:
            return max(int(self.max_steps), 1)

        if not hasattr(self.train_dataset, "__len__"):
            raise TypeError("`train_dataset` must implement __len__ when `max_steps` is None.")

        num_processes = max(int(self.accelerator.num_processes), 1)
        global_batch_size = max(self.batch_size * num_processes, 1)
        micro_steps_per_epoch = max(ceil(len(self.train_dataset) / global_batch_size), 1)
        opt_steps_per_epoch = max(
            ceil(micro_steps_per_epoch / self.gradient_accumulation_steps),
            1,
        )
        return max(opt_steps_per_epoch * self.num_epochs, 1)

    def _build_scheduler(self, scheduler_type, total_train_steps: int, warmup_steps: int = 0):
        scheduler_type = str(scheduler_type).strip().lower()
        total_train_steps = max(int(total_train_steps), 1)
        warmup_steps = min(max(int(warmup_steps), 0), total_train_steps - 1)

        remaining_steps = max(total_train_steps - warmup_steps, 1)
        if scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=remaining_steps,
                eta_min=self.learning_rate * 0.01,
            )
        elif scheduler_type == "constant":
            main_scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=remaining_steps)
        else:
            raise ValueError(
                f"Unsupported lr_scheduler_type: {scheduler_type}. "
                "Expected one of: ['cosine', 'constant']."
            )

        if warmup_steps <= 0:
            return main_scheduler

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )
    
    def _estimate_eta(self):
        elapsed = max(time.perf_counter() - self.run_start_time, 1e-6)
        done_steps = max(self.global_step - self.run_start_step, 1)
        steps_per_sec = done_steps / elapsed
        remaining_steps = max(self.max_steps - self.global_step, 0)
        eta_seconds = int(remaining_steps / max(steps_per_sec, 1e-9))
        eta_h, eta_rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(eta_rem, 60)
        return f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}", steps_per_sec

    def _resume_or_load_checkpoint(self):
        resume = self.resume
        if not resume:
            return
        resume_path = Path(str(resume))
        if resume_path.is_dir():
            if self.weight_only_start_step is not None:
                raise ValueError(
                    "`weight_only_start_step` is only valid for a weight "
                    "checkpoint file, not a full training-state directory."
                )
            logger.info("Resuming full training state from directory: %s", resume)
            self.load_training_state(str(resume_path))
            return
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        logger.info("Loading weight checkpoint only: %s", resume)
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        payload = unwrapped_model.load_checkpoint(
            str(resume_path), optimizer=None
        )
        pretrained_eraf = getattr(
            unwrapped_model,
            "policy_guard_eraf_pretrained_checkpoint",
            None,
        )
        if pretrained_eraf is not None:
            if str(payload.get("format", "")).startswith(
                "fastwam_policy_guard_"
            ):
                raise ValueError(
                    "Pretrained ERAF joint training must load the released "
                    "FastWAM Base in resume and import ERAF through the "
                    "separate pretrained_checkpoint field."
                )
            unwrapped_model.load_pretrained_eraf_checkpoint(pretrained_eraf)
        self._sync_optimizer_recovery_parameter_groups()
        if self.weight_only_start_step is None:
            logger.warning(
                "Loaded .pt weights only; optimizer/scheduler/step were not "
                "restored under ZeRO2."
            )
            return

        saved_step = payload.get("step") if isinstance(payload, dict) else None
        try:
            saved_step = int(saved_step)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Weight-only continuation requires a checkpoint with an "
                f"integer `step`, got {saved_step!r}."
            ) from exc
        if saved_step != self.weight_only_start_step:
            raise ValueError(
                "Weight-only continuation step mismatch: "
                f"checkpoint={saved_step}, "
                f"configured={self.weight_only_start_step}."
            )
        self._restore_weight_only_progress(saved_step)
        logger.warning(
            "Continued weights from absolute step=%d with a fresh optimizer "
            "and a fresh remaining-segment scheduler; Adam moments were not "
            "available under ZeRO2.",
            saved_step,
        )

    def _restore_weight_only_progress(self, start_step: int) -> None:
        """Restore absolute step and deterministic dataloader position."""
        position = optimizer_step_to_sampler_position(
            dataset_size=len(self.train_dataset),
            batch_size=self.batch_size,
            num_processes=self.accelerator.num_processes,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            optimizer_step=start_step,
        )
        self.global_step = int(start_step)
        self.epoch = position["epoch"]
        self.batch_in_epoch = position["batch_in_epoch"]
        self.train_sampler.set_epoch_offset(self.epoch)
        self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
        logger.info(
            "Restored weight-only progress: global_step=%d epoch=%d "
            "optimizer_step_in_epoch=%d batch_in_epoch=%d sample_offset=%d "
            "optimizer_steps_per_epoch=%d.",
            self.global_step,
            self.epoch,
            position["optimizer_step_in_epoch"],
            self.batch_in_epoch,
            self.batch_in_epoch
            * self.batch_size
            * self.accelerator.num_processes,
            position["optimizer_steps_per_epoch"],
        )

    def _sync_optimizer_recovery_parameter_groups(self):
        """Initialize post-checkpoint learning-rate masks for recovery.

        TC-C v2 initializes from an M1 adapter after Accelerate/DeepSpeed has
        already built the optimizer. Its initial recovery window freezes those
        restored policy tensors by setting the policy optimizer group LR to
        zero until the Router ramp begins.
        """
        policy_parameter_entries = 0
        router_parameter_entries = 0
        for group in self.optimizer.param_groups:
            group_kind = group.get("tc_recovery_group")
            if group_kind is None:
                continue
            parameters = list(group.get("params", []))
            if group_kind == "router":
                router_parameter_entries += len(parameters)
            else:
                policy_parameter_entries += len(parameters)
        unwrapped = self.accelerator.unwrap_model(self.model)
        if not bool(getattr(unwrapped, "transition_contract_enabled", False)):
            return
        logger.info(
            "Initialized TC recovery optimizer groups: policy_tensors=%d "
            "router_tensors=%d.",
            policy_parameter_entries,
            router_parameter_entries,
        )
        freeze_m1_policy = bool(
            getattr(unwrapped, "transition_freeze_m1_policy", False)
        )
        if router_parameter_entries <= 0:
            raise RuntimeError(
                "TC-C requires a non-empty Router optimizer group."
            )
        if freeze_m1_policy:
            if policy_parameter_entries != 0:
                raise RuntimeError(
                    "Protected TC v3+ requires zero trainable M1 "
                    "policy tensors."
                )
            logger.info(
                "Protected TC policy active: frozen M1 teacher, "
                "transition-module-only optimizer."
            )
        elif policy_parameter_entries <= 0:
            raise RuntimeError(
                "TC-C v2 requires separate non-empty policy and Router "
                "optimizer groups."
            )

    def _apply_transition_recovery_learning_rates(self):
        """Freeze/unfreeze restored M1 optimizer groups without rebuilding ZeRO."""
        unwrapped = self.accelerator.unwrap_model(self.model)
        scale_fn = getattr(unwrapped, "_transition_router_scale", None)
        if not callable(scale_fn) or not bool(
            getattr(unwrapped, "transition_freeze_m1_during_recovery", False)
        ):
            return
        router_scale = float(scale_fn())
        for index, group in enumerate(self.optimizer.param_groups):
            group_kind = group.get("tc_recovery_group")
            if group_kind is None:
                continue
            base_lr = self._tc_recovery_base_lrs[index]
            masked_lr = (
                base_lr
                if group_kind == "router" or router_scale > 0.0
                else 0.0
            )
            group["lr"] = masked_lr

    def _restore_transition_recovery_learning_rates(self):
        """Restore scheduled LRs after the masked optimizer update."""
        for index, group in enumerate(self.optimizer.param_groups):
            if group.get("tc_recovery_group") is not None:
                group["lr"] = self._tc_recovery_base_lrs[index]

    def _clear_frozen_policy_gradients(self):
        """Keep frozen M1 gradients out of clipping and Adam moments."""
        unwrapped = self.accelerator.unwrap_model(self.model)
        scale_fn = getattr(unwrapped, "_transition_router_scale", None)
        if not callable(scale_fn) or not bool(
            getattr(unwrapped, "transition_freeze_m1_during_recovery", False)
        ):
            return
        if float(scale_fn()) > 0.0:
            return
        for group in self.optimizer.param_groups:
            if group.get("tc_recovery_group") != "policy":
                continue
            for parameter in group.get("params", []):
                parameter.grad = None

    def _capture_scheduled_learning_rates(self):
        """Record scheduler output before the next recovery LR mask is applied."""
        for index, group in enumerate(self.optimizer.param_groups):
            if group.get("tc_recovery_group") is not None:
                self._tc_recovery_base_lrs[index] = float(group.get("lr", 0.0))

    def _set_dit_only_train_mode(self):
        # FastWAM decides whether the DiT policy or only TC modules should train.
        logger.info("Preparing model trainability and module modes.")
        model = self.accelerator.unwrap_model(self.model)
        self._apply_dit_only_train_mode(model)

    @staticmethod
    def _apply_dit_only_train_mode(model):
        if hasattr(model, "prepare_trainable_parameters"):
            report = model.prepare_trainable_parameters()
            logger.info(
                "Prepared model trainability: trainable=%d total=%d",
                report["trainable"],
                report["total"],
            )
            return
        model.eval()
        model.requires_grad_(False)
        model.dit.train()
        model.dit.requires_grad_(True)
        proprio_encoder = getattr(model, "proprio_encoder", None)
        if proprio_encoder is not None:
            proprio_encoder.train()
            proprio_encoder.requires_grad_(True)

    @staticmethod
    def _to_batched_eval_sample(sample):
        video = sample["video"]
        prompt = sample["prompt"]
        action = sample.get("action", None)
        proprio = sample.get("proprio", None)
        context = sample.get("context", None)
        context_mask = sample.get("context_mask", None)

        if not isinstance(video, torch.Tensor):
            raise TypeError(
                f"Expected tensor video for evaluation, got {type(video)}. "
                "Evaluation now expects `video` with shape [3,T,H,W] or [B,3,T,H,W]."
            )
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5:
            raise ValueError(f"Expected video shape [3,T,H,W] or [B,3,T,H,W], got {tuple(video.shape)}")
        num_video_frames = video.shape[2]
        if num_video_frames <= 1:
            raise ValueError(f"`sample['video']` must have at least 2 frames for action evaluation, got {num_video_frames}")

        if isinstance(prompt, str):
            prompt = [prompt]
        elif isinstance(prompt, tuple):
            prompt = list(prompt)
        elif not isinstance(prompt, list):
            raise TypeError(f"Expected prompt type str/list[str], got {type(prompt)}")
        if len(prompt) != video.shape[0]:
            raise ValueError(f"Prompt batch mismatch: len(prompt)={len(prompt)} vs video batch={video.shape[0]}")
        
        action_horizon = None
        action = None
        if "action" in sample:
            action = sample["action"]
            if not isinstance(action, torch.Tensor):
                raise TypeError(
                    f"`sample['action']` must be a torch.Tensor, got {type(action)}"
                )
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3:
                raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
            if action.shape[1] % (num_video_frames - 1) != 0:
                raise ValueError(f"`sample['action']` temporal dimension must be divisible by video frames-1={num_video_frames - 1}, got {action.shape[1]}")
            action_horizon = int(action.shape[1])

        proprio = None
        if "proprio" in sample:
            proprio = sample["proprio"]
            if not isinstance(proprio, torch.Tensor):
                raise TypeError(f"`sample['proprio']` must be a torch.Tensor, got {type(proprio)}")
            if proprio.ndim == 2:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")

        if context is not None or context_mask is not None:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must both exist in eval sample.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )

        return {
            "video": video,
            "prompt": prompt,
            "action": action,
            "proprio": proprio,
            "context": context,
            "context_mask": context_mask,
            "action_horizon": action_horizon,
        }

    @torch.no_grad()
    def evaluate(self):
        if self.val_dataset is None:
            return None

        model = self.accelerator.unwrap_model(self.model)
        was_dit_training = model.dit.training
        transition_modules = getattr(model, "transition_contract_modules", None)
        was_transition_training = bool(
            transition_modules is not None and transition_modules.training
        )
        policy_guard_action = getattr(
            model, "policy_guard_action_expert", None
        )
        policy_guard_modules = getattr(model, "policy_guard_modules", None)
        was_policy_guard_training = bool(
            (policy_guard_action is not None and policy_guard_action.training)
            or (
                policy_guard_modules is not None
                and policy_guard_modules.training
            )
        )
        model.eval()

        # eval_index = (self.global_step + self.accelerator.process_index) % len(self.val_dataset)
        rng = torch.Generator(device="cpu").manual_seed(self.global_step + self.accelerator.process_index)
        eval_index = torch.randint(0, len(self.val_dataset), (1,), generator=rng).item()
        sample = self._to_batched_eval_sample(self.val_dataset[eval_index])

        # 1. training loss
        with self.accelerator.autocast():
            val_loss, _ = model.training_loss(sample)
            val_loss = val_loss.float().item()
        
        prompt = sample["prompt"][0]
        video0 = sample["video"][0] # Tensor [3, T, H, W] in (-1, 1)
        action = sample["action"][0] if "action" in sample and sample["action"] is not None else None
        proprio = sample["proprio"][0, 0] if "proprio" in sample and sample["proprio"] is not None else None # from [1, T, d] to [d]
        input_image = video0[:, 0].unsqueeze(0)
        _, num_frames, _, _ = video0.shape

        # 2. inference and video saving
        infer_kwargs = {
            "input_image": input_image,
            "num_frames": num_frames,
            "action": action,
            "action_horizon": sample['action_horizon'],
            "proprio": proprio,
            "text_cfg_scale": 1.0,
            "action_cfg_scale": 1.0,
            "num_inference_steps": self.eval_num_inference_steps,
            "seed": 42,
            "tiled": False,
        }
        if sample["context"] is not None:
            infer_kwargs["prompt"] = None
            infer_kwargs["context"] = sample["context"][0]
            infer_kwargs["context_mask"] = sample["context_mask"][0]
        else:
            infer_kwargs["prompt"] = prompt

        pred = model.infer(
            **infer_kwargs,
        )
        
        pred_video = pred["video"]
        pred_action = pred.get("action", None)

        # 3. inference metrics against GT video
        pred_video_tensor = pil_frames_to_video_tensor(pred_video)
        gt_video_tensor = ((video0.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()

        assert pred_video_tensor.shape == gt_video_tensor.shape, (
            "Eval infer prediction/GT shape mismatch: "
            f"pred={tuple(pred_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_rollout_vs_gt = video_psnr(pred=pred_video_tensor, target=gt_video_tensor)
        ssim_rollout_vs_gt = video_ssim(pred=pred_video_tensor, target=gt_video_tensor)

        action_l1 = None
        action_l2 = None
        if action is not None and pred_action is not None:
            if sample["proprio"] is None:
                raise ValueError("Eval sample must contain `proprio` for action denormalization.")
            proprio = sample["proprio"].detach().to(device="cpu", dtype=torch.float32)
            
            processor = self.val_dataset.lerobot_dataset.processor

            denorm_actions = {}
            action_meta = processor.shape_meta["action"]
            state_meta = processor.shape_meta["state"]
            for action_name, raw_action in (("pred", pred_action), ("gt", action)):
                if not isinstance(raw_action, torch.Tensor):
                    raise TypeError(f"{action_name} action must be a torch.Tensor, got {type(raw_action)}")
                if raw_action.ndim == 2:
                    action_btd = raw_action.unsqueeze(0)
                elif raw_action.ndim == 3 and raw_action.shape[0] == 1:
                    action_btd = raw_action
                else:
                    raise ValueError(
                        f"{action_name} action must have shape [T, D] or [1, T, D], got {tuple(raw_action.shape)}"
                    )
                action_btd = action_btd.detach().to(device="cpu", dtype=torch.float32)

                batch = {
                    "action": action_btd,
                    "state": proprio,
                }
                batch = processor.action_state_merger.backward(batch)
                batch = processor.normalizer.backward(batch)
                merged_batch = {
                    "action": {meta["key"]: batch["action"][meta["key"]].squeeze(0) for meta in action_meta},
                    "state": {meta["key"]: batch["state"][meta["key"]].squeeze(0) for meta in state_meta},
                }
                merged_batch = processor.action_state_merger.forward(merged_batch)
                denorm_action = merged_batch["action"].unsqueeze(0)
                if denorm_action.ndim != 3 or denorm_action.shape[0] != 1:
                    raise ValueError(
                        f"Denormalized {action_name} action must have shape [1, T, D], got {tuple(denorm_action.shape)}"
                    )
                denorm_actions[action_name] = denorm_action

            pred_action_denorm = denorm_actions["pred"]
            gt_action_denorm = denorm_actions["gt"]

            if pred_action_denorm.shape != gt_action_denorm.shape:
                raise ValueError(
                    "Predicted action/GT action shape mismatch after denormalization: "
                    f"pred={tuple(pred_action_denorm.shape)} vs gt={tuple(gt_action_denorm.shape)}"
                )
            action_diff = pred_action_denorm - gt_action_denorm
            action_l1 = action_diff.abs().mean().item()
            action_l2 = action_diff.pow(2).mean().item()

        # 4. VAE reconstruction metrics against GT video
        gt_video_batch = video0.unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
        vae_latents = model._encode_video_latents(gt_video_batch, tiled=False)
        vae_recon_video = model._decode_latents(vae_latents, tiled=False)
        vae_video_tensor = pil_frames_to_video_tensor(vae_recon_video)

        assert vae_video_tensor.shape == gt_video_tensor.shape, (
            "Eval VAE reconstruction/GT shape mismatch: "
            f"vae={tuple(vae_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_decode_vs_gt = video_psnr(pred=vae_video_tensor, target=gt_video_tensor)
        ssim_decode_vs_gt = video_ssim(pred=vae_video_tensor, target=gt_video_tensor)

        psnr_rollout_vs_decode = video_psnr(pred=pred_video_tensor, target=vae_video_tensor)
        ssim_rollout_vs_decode = video_ssim(pred=pred_video_tensor, target=vae_video_tensor)

        stitched_video_tensor = torch.cat(
            [pred_video_tensor, vae_video_tensor, gt_video_tensor],
            dim=2,
        ).contiguous()
        stitched_frames = []
        for t in range(stitched_video_tensor.shape[1]):
            frame = (stitched_video_tensor[:, t].permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
            stitched_frames.append(Image.fromarray(frame))

        video_path = os.path.join(
            self.eval_dir,
            f"step_{self.global_step:06d}_rank_{self.accelerator.process_index:03d}.mp4",
        )
        save_mp4(stitched_frames, video_path, fps=8)

        local_metrics = torch.tensor(
            [
                float(val_loss),
                float(psnr_rollout_vs_gt),
                float(ssim_rollout_vs_gt),
                float(psnr_rollout_vs_decode),
                float(ssim_rollout_vs_decode),
                float(psnr_decode_vs_gt),
                float(ssim_decode_vs_gt),
                float(action_l2) if action_l2 is not None else -1.0,
                float(action_l1) if action_l1 is not None else -1.0,
            ],
            device=self.accelerator.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        gathered_metrics = self.accelerator.gather_for_metrics(local_metrics)
        mean_metrics = gathered_metrics[:, :7].mean(dim=0)
        action_l2_mean = gathered_metrics[:, 7].mean().item() if action_l2 is not None else None
        action_l1_mean = gathered_metrics[:, 8].mean().item() if action_l1 is not None else None

        if (
            was_dit_training
            or was_transition_training
            or was_policy_guard_training
        ):
            self._set_dit_only_train_mode()

        result = {
            "val_loss": float(mean_metrics[0].item()),
            "psnr_rg": float(mean_metrics[1].item()),
            "ssim_rg": float(mean_metrics[2].item()),
            "psnr_rd": float(mean_metrics[3].item()),
            "ssim_rd": float(mean_metrics[4].item()),
            "psnr_dg": float(mean_metrics[5].item()),
            "ssim_dg": float(mean_metrics[6].item()),
            "video_path": video_path,
        }
        if action_l2_mean is not None:
            result["action_l2"] = float(action_l2_mean)
        if action_l1_mean is not None:
            result["action_l1"] = float(action_l1_mean)
        return result

    def _save_weights_checkpoint(self, step_tag: str):
        model = self.accelerator.unwrap_model(self.model)
        ckpt_path = os.path.join(self.weights_dir, f"{step_tag}.pt")
        model.save_checkpoint(ckpt_path, optimizer=None, step=self.global_step)
        return ckpt_path

    def _save_trainer_state(self, state_path: str):
        state_file = os.path.join(state_path, "trainer_state.json")
        payload = {
            "global_step": int(self.global_step),
            "epoch": int(self.epoch),
            "batch_in_epoch": int(self.batch_in_epoch),
        }
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    def save_checkpoint(self):
        step_tag = f"step_{self.global_step:06d}"

        self.accelerator.wait_for_everyone()
        ckpt_path = None
        if self.accelerator.is_main_process:
            ckpt_path = self._save_weights_checkpoint(step_tag=step_tag)
        self.accelerator.wait_for_everyone()

        state_path = None
        if self.save_training_state:
            state_path = os.path.join(self.state_dir, step_tag)
            ensure_dir(state_path)
            self.accelerator.save_state(output_dir=state_path)
            if self.accelerator.is_main_process:
                self._save_trainer_state(state_path)
            self.accelerator.wait_for_everyone()

        return {"weights_path": ckpt_path, "state_path": state_path}

    def load_training_state(self, state_dir: str):
        self.accelerator.load_state(input_dir=state_dir)
        state_file = Path(state_dir) / "trainer_state.json"
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.global_step = int(payload["global_step"])

            if "epoch" in payload and "batch_in_epoch" in payload:
                self.epoch = int(payload["epoch"])
                self.batch_in_epoch = int(payload["batch_in_epoch"])
                self.train_sampler.set_epoch_offset(self.epoch)
                self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
                logger.info(
                    "Restored dataloader progress: epoch=%d batch_in_epoch=%d sample_offset=%d",
                    self.epoch,
                    self.batch_in_epoch,
                    self.batch_in_epoch * self.batch_size * self.accelerator.num_processes,
                )
            else:
                self.epoch = 0
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                logger.warning(
                    "State file does not contain `epoch`/`batch_in_epoch`; "
                    "optimizer/scheduler were restored, but dataloader progress resume is skipped."
                )
            self.accelerator.wait_for_everyone()
            return

        match = re.search(r"step[_-](\d+)$", str(state_dir).rstrip("/"))
        if match:
            self.global_step = int(match.group(1))
        else:
            self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.train_sampler.clear_resume_batch_offset()
        self.accelerator.wait_for_everyone()
        logger.info("Loaded accelerate training state from %s at step=%d", state_dir, self.global_step)
        logger.warning(
            "State file `%s` is missing; dataloader progress resume is skipped.",
            state_file,
        )

    def train(self):
        self._set_dit_only_train_mode()

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        if self.max_steps is None:
            raise ValueError("`max_steps` must be set before entering the while-step training loop.")

        logger.info("Starting training with max_steps=%d.", self.max_steps)
        data_iter = iter(self.train_loader)
        self.run_start_step = self.global_step
        self.run_start_time = time.perf_counter()

        while self.global_step < self.max_steps:
            try:
                sample = next(data_iter)
                self.batch_in_epoch += 1
            except StopIteration:
                self.epoch += 1
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                data_iter = iter(self.train_loader)
                continue

            with self.accelerator.accumulate(self.model):
                train_model = self.model if hasattr(self.model, "training_loss") else self.accelerator.unwrap_model(self.model)
                if hasattr(unwrapped_model, "set_training_progress"):
                    unwrapped_model.set_training_progress(
                        self.global_step, self.max_steps
                    )
                self._apply_transition_recovery_learning_rates()

                with self.accelerator.autocast():
                    loss, loss_dict = train_model.training_loss(sample)
                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    self._clear_frozen_policy_gradients()
                    grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self._restore_transition_recovery_learning_rates()
                    if not self.accelerator.optimizer_step_was_skipped:
                        self.scheduler.step()
                        self._capture_scheduled_learning_rates()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    global_loss = float(
                        self.accelerator.gather(loss.detach().float().reshape(1)).mean().item()
                    )
                    global_loss_metrics = {}
                    for key, value in loss_dict.items():
                        metric_tensor = torch.tensor(float(value), device=loss.device, dtype=torch.float32).reshape(1)
                        global_loss_metrics[key] = float(
                            self.accelerator.gather(metric_tensor).mean().item()
                        )
                    grad_norm_tensor = torch.tensor(grad_norm, device=loss.device, dtype=torch.float32)
                    global_grad_norm = float(self.accelerator.gather(grad_norm_tensor).mean().item())

                    current_lr = max(
                        float(group["lr"])
                        for group in self.optimizer.param_groups
                    )

                    if self.log_every > 0 and self.global_step % self.log_every == 0 and self.accelerator.is_main_process:
                        eta_str, steps_per_sec = self._estimate_eta()
                        description = "[train] epoch=%d step=%d/%d loss=%.4f " % (
                            self.epoch,
                            self.global_step,
                            self.max_steps,
                            global_loss,
                        )
                        if global_loss_metrics:
                            detail_str = " ".join([f"{k}={v:.4f}" for k, v in sorted(global_loss_metrics.items())])
                            description += detail_str + " "
                        description += "lr=%.2e speed=%.2f step/s, %.2f samples/s eta=%s" % (
                            current_lr,
                            steps_per_sec,
                            steps_per_sec * self.batch_size * self.accelerator.num_processes,
                            eta_str,
                        )
                        logger.info(description)

                        wandb_payload = {
                            "train/loss": global_loss,
                            "train/grad_norm": global_grad_norm,
                            "train/lr": current_lr,
                            "performance/steps_per_sec": steps_per_sec,
                            "performance/samples_per_sec": steps_per_sec * self.batch_size * self.accelerator.num_processes,
                        }
                        for key, value in global_loss_metrics.items():
                            wandb_payload[f"train/{key}"] = value
                        self._wandb_log(wandb_payload)

                    if (
                        self.eval_every > 0
                        and self.val_dataset is not None
                        and self.global_step % self.eval_every == 0
                    ):
                        metrics = self.evaluate()
                        self.accelerator.wait_for_everyone()
                        if metrics is not None and self.accelerator.is_main_process:
                            description = "[eval] step=%d val_loss=%.4f infer_psnr=%.4f infer_ssim=%.4f" % (
                                self.global_step,
                                metrics["val_loss"],
                                metrics["psnr_rd"],
                                metrics["ssim_rd"],
                            )
                            if "action_l2" in metrics:
                                description += " action_l2=%.4f" % metrics["action_l2"]
                            if "action_l1" in metrics:
                                description += " action_l1=%.4f" % metrics["action_l1"]
                            logger.info(description)
                            eval_payload = {
                                "eval/val_loss": float(metrics["val_loss"]),
                                "eval/psnr_rg": float(metrics["psnr_rg"]),
                                "eval/ssim_rg": float(metrics["ssim_rg"]),
                                "eval/psnr_rd": float(metrics["psnr_rd"]),
                                "eval/ssim_rd": float(metrics["ssim_rd"]),
                                "eval/psnr_dg": float(metrics["psnr_dg"]),
                                "eval/ssim_dg": float(metrics["ssim_dg"]),
                            }
                            if "action_l2" in metrics:
                                eval_payload["eval/action_l2"] = float(metrics["action_l2"])
                            if "action_l1" in metrics:
                                eval_payload["eval/action_l1"] = float(metrics["action_l1"])
                            self._wandb_log(eval_payload)

                    if self.save_every > 0 and self.global_step % self.save_every == 0:
                        ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[ckpt] step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )

                    if self.global_step >= self.max_steps:
                        ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[done] max_steps reached step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )
                        return

        ckpt_info = self.save_checkpoint()
        if self.accelerator.is_main_process:
            logger.info(
                "[done] training finished step=%d weights=%s state=%s",
                self.global_step,
                ckpt_info["weights_path"],
                ckpt_info["state_path"],
            )
        
