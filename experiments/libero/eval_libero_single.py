import json
import inspect
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import hydra
import numpy as np
import torch
from accelerate import PartialState
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw
from tqdm import tqdm

# try:
#     import rootutils

#     rootutils.setup_root(__file__, indicator=".python-version", pythonpath=True)
# except ModuleNotFoundError:
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.libero.libero_utils import (
    LIBERO_ENV_RESOLUTION,
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    invert_gripper_action,
    quat2axisangle,
    save_prediction_video,
    save_rollout_video,
)
from experiments.libero.init_state_utils import load_libero_task_init_states
from experiments.libero.counterfactual_diagnostics import (
    CounterfactualEpisodeTracker,
    empty_behavior_counts,
)
from experiments.libero.eraf_shadow_audit import (
    ERAFOracleProvider,
    ERAFShadowAuditor,
    ERAFShadowContract,
    summarize_eraf_shadow_records,
    verify_shadow_action_integrity,
)
from experiments.libero.language_condition import normalize_instruction_condition
from experiments.libero.language_ood import (
    resolve_paraphrase_instruction,
    sha256_file,
    validate_language_ood_record,
)
from experiments.libero.oracle_phase_servo import (
    OraclePhaseServoConfig,
    apply_oracle_phase_servo,
    summarize_oracle_phase_servo,
)
from experiments.libero.language_interventions import (
    load_language_intervention_manifest,
    select_language_intervention_record,
    validate_counterfactual_problem,
)
from experiments.libero.policy_guard_state import PolicyGuardStateController
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.pgc_libero import state_sha256 as _canonical_state_sha256
from fastwam.models.wan22.entity_relation_affordance import ERAF_PREDICATES
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.utils.pytorch_utils import set_global_seed
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from libero.libero import benchmark, get_libero_path
from action_ensembler import ActionEnsembler

OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("max", lambda x: max(x))
OmegaConf.register_new_resolver("split", lambda s, idx: s.split("/")[int(idx)])

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _save_eraf_diagnostics(
    *,
    cfg: DictConfig,
    images: dict[str, np.ndarray],
    diagnostics: dict[str, Any],
    episode_idx: int,
    replan_idx: int,
) -> dict[str, Any]:
    """Persist V9 heatmaps and a two-camera subject/reference overlay."""
    output_value = cfg.EVALUATION.get("entity_relation_overlay_dir")
    if output_value in (None, "", "null"):
        raise ValueError(
            "EVALUATION.entity_relation_diagnostics=true requires "
            "EVALUATION.entity_relation_overlay_dir."
        )
    output_dir = (
        Path(str(output_value)).expanduser().resolve()
        / str(cfg.EVALUATION.task_suite_name)
        / f"task_{int(cfg.EVALUATION.task_id):02d}"
        / f"trial_{int(episode_idx):03d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"replan_{int(replan_idx):04d}"
    arrays = {name: np.asarray(value) for name, value in diagnostics.items()}
    npz_path = output_dir / f"{stem}.npz"
    np.savez_compressed(npz_path, **arrays)

    subject = np.asarray(arrays["subject_attention"])[0]
    reference = np.asarray(arrays["reference_attention"])[0]
    coordinates = np.asarray(arrays["spatial_coordinates"])
    grid_height = len(np.unique(np.round(coordinates[:, 1], decimals=6)))
    grid_width = len(np.unique(np.round(coordinates[:, 0], decimals=6)))
    if grid_height * grid_width != subject.shape[-1] or grid_width % 2:
        raise ValueError(
            "PGC v9 ERAF diagnostics do not form a two-camera spatial grid."
        )
    predicate_ids = np.asarray(arrays["predicate_logits"])[0].argmax(axis=-1)
    active_probability = 1.0 / (1.0 + np.exp(-np.asarray(arrays["active_logits"])[0]))
    phase_ids = np.asarray(arrays["phase_logits"])[0].argmax(axis=-1)
    active_indices = np.flatnonzero(active_probability >= 0.5)
    if active_indices.size == 0:
        active_indices = np.asarray([int(active_probability.argmax())])

    camera_images = (
        np.asarray(images["image"], dtype=np.uint8),
        np.asarray(images["wrist_image"], dtype=np.uint8),
    )

    def _overlay(base: np.ndarray, heat: np.ndarray) -> Image.Image:
        base_image = Image.fromarray(base).convert("RGB").resize((224, 224))
        heat_image = Image.fromarray(
            np.uint8(255.0 * heat / max(1.0e-8, float(heat.max())))
        ).resize((224, 224), resample=Image.Resampling.BILINEAR)
        heat_array = np.asarray(heat_image, dtype=np.float32) / 255.0
        base_array = np.asarray(base_image, dtype=np.float32)
        color = np.zeros_like(base_array)
        color[..., 0] = 255.0
        alpha = (0.65 * heat_array)[..., None]
        blended = base_array * (1.0 - alpha) + color * alpha
        return Image.fromarray(np.uint8(np.clip(blended, 0, 255)))

    clause_rows: list[Image.Image] = []
    half_width = grid_width // 2
    for clause_index in active_indices.tolist():
        role_rows = []
        for role_name, role_attention in (
            ("subject", subject),
            ("reference", reference),
        ):
            grid = role_attention[clause_index].reshape(grid_height, grid_width)
            view_overlays = [
                _overlay(camera_images[0], grid[:, :half_width]),
                _overlay(camera_images[1], grid[:, half_width:]),
            ]
            row = Image.new("RGB", (448, 246), "white")
            row.paste(view_overlays[0], (0, 22))
            row.paste(view_overlays[1], (224, 22))
            ImageDraw.Draw(row).text(
                (4, 4),
                f"clause {clause_index} {role_name}",
                fill="black",
            )
            role_rows.append(row)
        predicate_id = int(predicate_ids[clause_index])
        predicate = (
            ERAF_PREDICATES[predicate_id]
            if 0 <= predicate_id < len(ERAF_PREDICATES)
            else f"unknown:{predicate_id}"
        )
        header = Image.new("RGB", (448, 42), "white")
        grasp_anchor = np.asarray(arrays["grasp_anchor"])[0, clause_index]
        goal_anchor = np.asarray(arrays["goal_anchor"])[0, clause_index]
        interaction_anchor = np.asarray(arrays["interaction_anchor"])[0, clause_index]
        header_draw = ImageDraw.Draw(header)
        header_draw.text(
            (4, 3),
            f"predicate={predicate} phase={int(phase_ids[clause_index])}",
            fill="black",
        )
        header_draw.text(
            (4, 20),
            f"grasp={np.round(grasp_anchor, 2).tolist()} "
            f"goal={np.round(goal_anchor, 2).tolist()} "
            f"op={np.round(interaction_anchor, 2).tolist()}",
            fill="black",
        )
        clause = Image.new("RGB", (448, 534), "white")
        clause.paste(header, (0, 0))
        clause.paste(role_rows[0], (0, 42))
        clause.paste(role_rows[1], (0, 288))
        clause_rows.append(clause)
    canvas = Image.new("RGB", (448, 534 * len(clause_rows)), "white")
    for index, row in enumerate(clause_rows):
        canvas.paste(row, (0, 534 * index))
    png_path = output_dir / f"{stem}.png"
    canvas.save(png_path)
    return {
        "predicate_ids": predicate_ids.tolist(),
        "active_probability": active_probability.tolist(),
        "phase_ids": phase_ids.tolist(),
        "subject_positions": np.asarray(arrays["subject_position"])[0].tolist(),
        "reference_positions": np.asarray(arrays["reference_position"])[0].tolist(),
        "subject_view_visibility": (
            1.0
            / (1.0 + np.exp(-np.asarray(arrays["subject_view_visibility_logits"])[0]))
        ).tolist(),
        "reference_view_visibility": (
            1.0
            / (1.0 + np.exp(-np.asarray(arrays["reference_view_visibility_logits"])[0]))
        ).tolist(),
        "subject_view_centers": np.asarray(arrays["subject_view_centers"])[0].tolist(),
        "reference_view_centers": np.asarray(arrays["reference_view_centers"])[
            0
        ].tolist(),
        "grasp_anchors": np.asarray(arrays["grasp_anchor"])[0].tolist(),
        "goal_anchors": np.asarray(arrays["goal_anchor"])[0].tolist(),
        "interaction_anchors": np.asarray(arrays["interaction_anchor"])[0].tolist(),
        "npz": str(npz_path),
        "overlay": str(png_path),
    }


def _normalize_mixed_precision(mixed_precision: str) -> str:
    key = str(mixed_precision).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _resolve_eval_device(cfg: DictConfig) -> str:
    eval_device = cfg.EVALUATION.get("device")
    if eval_device is not None:
        return str(eval_device)
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_dataset_stats_path(cfg: DictConfig) -> Path:
    explicit = cfg.EVALUATION.get("dataset_stats_path")
    candidates: list[Path] = []

    if explicit is not None:
        candidates.append(Path(os.path.expanduser(os.path.expandvars(str(explicit)))))

    ckpt = Path(os.path.expanduser(os.path.expandvars(str(cfg.ckpt))))
    for parent in list(ckpt.parents)[:4]:
        candidates.append(parent / "dataset_stats.json")

    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved

    msg = (
        "Failed to locate dataset_stats.json. Tried explicit "
        "EVALUATION.dataset_stats_path and checkpoint parent directories. "
        "Please pass EVALUATION.dataset_stats_path=/path/to/dataset_stats.json."
    )
    raise FileNotFoundError(msg)


def _load_model_checkpoint(model: torch.nn.Module, ckpt: str) -> None:
    model.load_checkpoint(ckpt)
    logging.info("Loaded checkpoint via model.load_checkpoint: %s", ckpt)
    return

    # deprecated legacy checkpoint loading
    payload = torch.load(ckpt, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(
            f"Legacy checkpoint payload must be dict, got: {type(payload)}"
        )

    if "mot" in payload and hasattr(model, "mot"):
        missing, unexpected = model.mot.load_state_dict(payload["mot"], strict=False)
        logging.warning(
            "Loaded fallback `mot` state_dict with strict=False. Missing=%d Unexpected=%d",
            len(missing),
            len(unexpected),
        )
        return

    state_dict = None
    for key in ("model_state_dict", "state_dict", "model"):
        value = payload.get(key)
        if isinstance(value, dict):
            state_dict = value
            break
    if state_dict is None and all(torch.is_tensor(v) for v in payload.values()):
        state_dict = payload
    if state_dict is None:
        raise ValueError(f"Cannot parse legacy checkpoint keys from: {ckpt}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logging.warning(
        "Loaded fallback model state_dict with strict=False. Missing=%d Unexpected=%d",
        len(missing),
        len(unexpected),
    )


def _center_crop_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    pil_image = Image.fromarray(image)
    src_w, src_h = pil_image.size
    scale = max(width / src_w, height / src_h)
    resized = pil_image.resize(
        (round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR
    )
    rw, rh = resized.size
    left = max((rw - width) // 2, 0)
    top = max((rh - height) // 2, 0)
    cropped = resized.crop((left, top, left + width, top + height))
    return np.asarray(cropped, dtype=np.uint8)


def _normalize_proprio(
    proprio: np.ndarray,
    processor: FastWAMProcessor,
) -> torch.Tensor:
    state_meta = processor.shape_meta["state"]
    if len(state_meta) != 1:
        raise ValueError(
            "LIBERO eval currently expects a single merged state key in shape_meta['state']."
        )
    state_key = state_meta[0]["key"]

    state_batch = {
        "state": {state_key: torch.as_tensor(proprio, dtype=torch.float32).unsqueeze(0)}
    }
    state_batch = processor.action_state_transform(state_batch)
    state_batch = processor.normalizer.forward(state_batch)
    return state_batch["state"][state_key]


def _obs_to_model_input(
    obs: dict,
    cfg: DictConfig,
    processor: FastWAMProcessor,
    width: int,
    height: int,
    device: str,
    dtype: torch.dtype,
):
    imgs = get_libero_image(obs)
    image_meta = processor.shape_meta["images"]
    if len(image_meta) < int(processor.num_output_cameras):
        raise ValueError(
            f"shape_meta.images has {len(image_meta)} entries, "
            f"but num_output_cameras={processor.num_output_cameras}."
        )

    def _meta_to_hw(meta: dict, camera_idx: int) -> tuple[int, int]:
        shape = meta["shape"]
        if len(shape) != 3:
            raise ValueError(
                f"shape_meta.images[{camera_idx}].shape must be [C,H,W], got {shape}"
            )
        return int(shape[1]), int(shape[2])

    concatenation = cfg.data.train.get("concat_multi_camera", "horizontal")
    num_cameras = processor.num_output_cameras
    if num_cameras == 1:
        primary_h, primary_w = _meta_to_hw(image_meta[0], camera_idx=0)
        rgb = _center_crop_resize(imgs["image"], width=primary_w, height=primary_h)
    elif num_cameras == 2:
        primary_h, primary_w = _meta_to_hw(image_meta[0], camera_idx=0)
        wrist_h, wrist_w = _meta_to_hw(image_meta[1], camera_idx=1)
        primary = _center_crop_resize(imgs["image"], width=primary_w, height=primary_h)
        wrist = _center_crop_resize(imgs["wrist_image"], width=wrist_w, height=wrist_h)
        if concatenation == "horizontal":
            rgb = np.concatenate([primary, wrist], axis=1)
        elif concatenation == "vertical":
            rgb = np.concatenate([primary, wrist], axis=0)
        else:
            raise ValueError(f"Invalid concat_multi_camera: {concatenation}")
    else:
        raise ValueError(
            f"LIBERO eval currently supports num_output_cameras in [1, 2], got {num_cameras}."
        )

    actual_h, actual_w = int(rgb.shape[0]), int(rgb.shape[1])
    expected_h, expected_w = int(height), int(width)
    image_shapes = [meta["shape"] for meta in image_meta]
    assert actual_h == expected_h and actual_w == expected_w, (
        "Input image size mismatch after per-camera resize + concat: "
        f"got (H,W)=({actual_h},{actual_w}), expected (H,W)=({expected_h},{expected_w}) "
        f"from data.train.video_size={[expected_h, expected_w]}; "
        f"shape_meta.images={image_shapes}, concat_multi_camera={concatenation}."
    )

    x = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    x = x * (2.0 / 255.0) - 1.0

    proprio = _normalize_proprio(_extract_sim_state(obs), processor)

    return x, proprio, imgs


def _extract_sim_state(obs: dict) -> np.ndarray:
    """Build simulator state from current observation.

    This is used as proprio input for model inference.
    """
    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    ).astype(np.float32)
    return state


def _denormalize_action(
    action: torch.Tensor, processor: FastWAMProcessor
) -> np.ndarray:
    if action.ndim == 2:
        action = action.unsqueeze(0)
    if action.ndim != 3:
        raise ValueError(f"Expected action tensor [B, T, D], got {tuple(action.shape)}")

    action_meta = processor.shape_meta["action"]
    if len(action_meta) != 1:
        raise ValueError(
            "LIBERO eval currently expects a single merged action key in shape_meta['action']."
        )

    action_key = action_meta[0]["key"]
    normalizer = processor.normalizer.normalizers["action"][action_key]
    action = action.to(dtype=torch.float32, device="cpu")
    denorm = normalizer.backward(action)
    return denorm.numpy()


def _get_num_video_frames(cfg: DictConfig) -> int:
    return (int(cfg.data.train.num_frames) - 1) // int(
        cfg.data.train.action_video_freq_ratio
    ) + 1


def _validate_visualize_future_video_cfg(cfg: DictConfig) -> None:
    if not bool(cfg.EVALUATION.get("visualize_future_video", False)):
        return

    action_conditioned = cfg.model.video_dit_config.get("action_conditioned", None)
    if action_conditioned is not False:
        raise ValueError(
            "EVALUATION.visualize_future_video=true requires "
            "model.video_dit_config.action_conditioned=false."
        )


def _select_predicted_future_frames(
    pred_video: list[Image.Image], cfg: DictConfig
) -> list[Image.Image]:
    if len(pred_video) == 0:
        raise ValueError("`infer_joint` returned an empty predicted video.")

    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    action_video_freq_ratio = int(cfg.data.train.action_video_freq_ratio)
    num_future_frames = replan_steps // action_video_freq_ratio
    keep_frames = 1 + num_future_frames
    return list(pred_video[:keep_frames])


def _get_future_frame_capture_steps(cfg: DictConfig) -> list[int]:
    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    action_video_freq_ratio = int(cfg.data.train.action_video_freq_ratio)
    num_future_frames = replan_steps // action_video_freq_ratio
    return [
        step_idx * action_video_freq_ratio for step_idx in range(num_future_frames + 1)
    ]


def _frame_to_rgb_array(frame: Any) -> np.ndarray:
    if isinstance(frame, dict):
        images = []
        for value in frame.values():
            value_array = (
                np.array(value)
                if isinstance(value, Image.Image)
                else np.array(value, copy=True)
            )
            images.append(value_array)
        return np.concatenate(images, axis=1)
    if isinstance(frame, Image.Image):
        return np.array(frame.convert("RGB"))
    return np.array(frame, copy=True)


def _compute_clip_mean_psnr(
    gt_frames: list[Any],
    pred_frames: list[Any],
    eps: float = 1e-8,
) -> Optional[float]:
    if len(gt_frames) == 0 or len(pred_frames) == 0:
        return None
    assert len(gt_frames) == len(pred_frames), (
        "GT/pred frame count mismatch for PSNR: "
        f"len(gt_frames)={len(gt_frames)} len(pred_frames)={len(pred_frames)}. "
        "This indicates temporal misalignment in future-video capture."
    )
    num_frames = len(gt_frames)

    frame_psnr_values = []
    for gt_frame, pred_frame in zip(gt_frames[:num_frames], pred_frames[:num_frames]):
        gt_image = _frame_to_rgb_array(gt_frame)
        pred_image = _frame_to_rgb_array(pred_frame)
        target_h, target_w = pred_image.shape[:2]
        if gt_image.shape[:2] != (target_h, target_w):
            gt_image = np.array(
                Image.fromarray(gt_image).resize(
                    (target_w, target_h), resample=Image.BILINEAR
                )
            )

        gt_f32 = gt_image.astype(np.float32)
        pred_f32 = pred_image.astype(np.float32)
        mse = float(np.mean((pred_f32 - gt_f32) ** 2))
        psnr = 10.0 * np.log10((255.0 * 255.0) / max(mse, eps))
        frame_psnr_values.append(float(psnr))

    if len(frame_psnr_values) == 0:
        return None
    return float(np.mean(frame_psnr_values))


def _resolve_language_intervention(
    task_description: str,
    cfg: DictConfig,
) -> tuple[str, str, bool, Optional[dict[str, Any]]]:
    """Resolve policy language and its optional paired manifest record."""
    condition = normalize_instruction_condition(
        cfg.EVALUATION.get("instruction_condition", "correct")
    )
    if condition == "correct":
        return condition, task_description, False, None
    if condition == "null":
        # Keep a normal encoder input but mask every language position inside
        # FastWAM. Proprio/state tokens remain visible.
        return condition, task_description, True, None
    if condition not in {"shuffled", "counterfactual", "paraphrase"}:
        raise ValueError(
            "EVALUATION.instruction_condition must be one of "
            "correct/null/shuffled/counterfactual/paraphrase, got "
            f"{condition!r}."
        )

    override = cfg.EVALUATION.get("instruction_override")
    if condition == "shuffled" and override is not None and str(override).strip():
        return condition, str(override).strip(), False, None

    manifest_path_cfg = cfg.EVALUATION.get("language_intervention_manifest")
    if manifest_path_cfg is None:
        raise ValueError(
            f"{condition} evaluation requires "
            "EVALUATION.instruction_override or "
            "EVALUATION.language_intervention_manifest."
        )
    manifest_path = Path(os.path.expanduser(os.path.expandvars(str(manifest_path_cfg))))
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Language-intervention manifest not found: {manifest_path}"
        )

    record = select_language_intervention_record(
        load_language_intervention_manifest(manifest_path),
        suite_name=str(cfg.EVALUATION.task_suite_name),
        task_id=int(cfg.EVALUATION.task_id),
        task_description=task_description,
    )
    if condition == "paraphrase":
        validate_language_ood_record(record)
        variant = cfg.EVALUATION.get("language_ood_variant")
        if variant in (None, "", "null"):
            raise ValueError(
                "Paraphrase evaluation requires EVALUATION.language_ood_variant."
            )
        instruction = resolve_paraphrase_instruction(record, str(variant))
        field = f"paraphrases.{str(variant).strip().casefold()}"
    else:
        field = (
            "shuffled_instruction"
            if condition == "shuffled"
            else "counterfactual_instruction"
        )
        instruction = str(record.get(field, "")).strip()
    if not instruction:
        raise ValueError(
            f"Manifest line {record.get('_line_number', '?')} has no non-empty "
            f"`{field}`."
        )
    if (
        condition == "counterfactual"
        and record.get("counterfactual_is_executable") is not True
    ):
        raise ValueError(
            f"Manifest line {record.get('_line_number', '?')} must mark "
            "counterfactual_is_executable=true."
        )
    return condition, instruction, False, record


def _resolve_counterfactual_task(
    record: dict[str, Any],
) -> tuple[str, int, Any]:
    suite_name = str(
        record.get(
            "counterfactual_task_suite_name",
            record.get("task_suite_name", ""),
        )
    ).strip()
    if not suite_name:
        raise ValueError("Counterfactual manifest record has no task suite selector.")
    benchmark_dict = benchmark.get_benchmark_dict()
    if suite_name not in benchmark_dict:
        raise ValueError(f"Unknown counterfactual task suite: {suite_name!r}.")
    task_suite = benchmark_dict[suite_name]()

    raw_task_id = record.get("counterfactual_task_id")
    if raw_task_id is not None:
        task_id = int(raw_task_id)
        return suite_name, task_id, task_suite.get_task(task_id)

    expected_name = str(record.get("counterfactual_task_name", "")).strip()
    matches = [
        task_id
        for task_id in range(int(task_suite.n_tasks))
        if task_suite.get_task(task_id).language.strip().casefold()
        == expected_name.casefold()
    ]
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one counterfactual task matching "
            f"{suite_name}/{expected_name!r}, found {len(matches)}."
        )
    task_id = matches[0]
    return suite_name, task_id, task_suite.get_task(task_id)


def _activate_counterfactual_goal(
    env,
    record: dict[str, Any],
    policy_instruction: str,
) -> dict[str, Any]:
    """Replace only the success goal while preserving source scene and state."""
    from libero.libero.envs import bddl_utils as BDDLUtils

    suite_name, task_id, task = _resolve_counterfactual_task(record)
    task_instruction = str(task.language).strip()
    if task_instruction.casefold() != policy_instruction.strip().casefold():
        raise ValueError(
            "Counterfactual manifest instruction does not match the selected "
            f"LIBERO task: {policy_instruction!r} != {task_instruction!r}."
        )

    bddl_path = (
        Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    )
    counterfactual_problem = BDDLUtils.robosuite_parse_problem(str(bddl_path))
    inner_env = getattr(env, "env", None)
    if inner_env is None or not hasattr(inner_env, "parsed_problem"):
        raise TypeError(
            "Counterfactual evaluation requires a LIBERO ControlEnv wrapper "
            "with an inner parsed_problem."
        )
    source_problem = inner_env.parsed_problem
    counterfactual_goal = validate_counterfactual_problem(
        source_problem,
        counterfactual_problem,
    )

    runtime_entities = set(getattr(inner_env, "object_states_dict", {}))
    runtime_missing = sorted(
        {str(entity) for predicate in counterfactual_goal for entity in predicate[1:]}
        - runtime_entities
    )
    if runtime_missing:
        raise ValueError(
            "Counterfactual predicate entities are absent from the instantiated "
            f"source environment: {runtime_missing}."
        )

    source_goal = [list(predicate) for predicate in source_problem["goal_state"]]
    inner_env.parsed_problem["goal_state"] = counterfactual_goal
    logging.info(
        "Activated paired counterfactual predicate for %s/%s: %s -> %s",
        suite_name,
        task_id,
        source_goal,
        counterfactual_goal,
    )
    return {
        "pair_id": record.get("pair_id"),
        "counterfactual_task_suite_name": suite_name,
        "counterfactual_task_id": task_id,
        "counterfactual_task_name": task_instruction,
        "counterfactual_bddl_file": str(bddl_path),
        "source_goal_state": source_goal,
        "counterfactual_goal_state": counterfactual_goal,
    }


def _predict_action_chunk(
    obs: dict,
    policy_instruction: str,
    mask_language: bool,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
    policy_guard_state: Optional[dict[str, Any]] = None,
    policy_guard_eraf_oracle: Optional[dict[str, Any]] = None,
) -> tuple[
    np.ndarray,
    dict,
    Optional[list[Image.Image]],
    float,
    Optional[dict[str, Any]],
    Optional[dict[str, Any]],
]:
    num_inference_steps_cfg = cfg.EVALUATION.get("num_inference_steps", None)
    if num_inference_steps_cfg is None:
        num_inference_steps = int(cfg.get("eval_num_inference_steps", 20))
    else:
        num_inference_steps = int(num_inference_steps_cfg)
    prompt_template = DEFAULT_PROMPT
    prompt = prompt_template.format(task=policy_instruction)

    image, proprio, imgs = _obs_to_model_input(
        obs,
        cfg=cfg,
        processor=processor,
        width=input_w,
        height=input_h,
        device=model_device,
        dtype=model.torch_dtype,
    )

    infer_kwargs = {
        "prompt": prompt,
        "input_image": image,
        "action_horizon": action_horizon,
        "negative_prompt": str(cfg.EVALUATION.get("negative_prompt", "")),
        "text_cfg_scale": float(cfg.EVALUATION.get("text_cfg_scale", 1.0)),
        "num_inference_steps": num_inference_steps,
        "proprio": proprio,
        "sigma_shift": (
            None
            if cfg.EVALUATION.get("sigma_shift") is None
            else float(cfg.EVALUATION.get("sigma_shift"))
        ),
        "seed": None if cfg.get("seed") is None else int(cfg.seed),
        "rand_device": str(cfg.EVALUATION.get("rand_device", "cpu")),
        "tiled": bool(cfg.EVALUATION.get("tiled", False)),
    }
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    predicted_future_frames = None
    if visualize_future_video:
        infer_kwargs["num_video_frames"] = _get_num_video_frames(cfg)
    elif "num_video_frames" in inspect.signature(model.infer_action).parameters:
        infer_kwargs["num_video_frames"] = _get_num_video_frames(cfg)

    inference_method = (
        model.infer_joint if visualize_future_video else model.infer_action
    )
    if mask_language:
        if "mask_language" not in inspect.signature(inference_method).parameters:
            raise ValueError(
                "Null-language evaluation requires an inference method with a "
                "`mask_language` argument."
            )
        infer_kwargs["mask_language"] = True
    if (
        policy_guard_state is not None
        and "policy_guard_state" in inspect.signature(inference_method).parameters
    ):
        infer_kwargs["policy_guard_state"] = policy_guard_state
    if policy_guard_eraf_oracle is not None:
        if "policy_guard_eraf_oracle" not in inspect.signature(
            inference_method
        ).parameters:
            raise ValueError(
                "Oracle ERAF evaluation requires an inference method with a "
                "`policy_guard_eraf_oracle` argument."
            )
        infer_kwargs["policy_guard_eraf_oracle"] = policy_guard_eraf_oracle

    with torch.no_grad():
        if str(model_device).startswith("cuda"):
            torch.cuda.synchronize()
        inference_start = time.perf_counter()
        if visualize_future_video:
            pred = model.infer_joint(**infer_kwargs)
            predicted_future_frames = _select_predicted_future_frames(
                pred["video"], cfg
            )
        else:
            pred = model.infer_action(**infer_kwargs)
        if str(model_device).startswith("cuda"):
            torch.cuda.synchronize()
        inference_latency_ms = (time.perf_counter() - inference_start) * 1000.0
    action = pred["action"]  # [T, D]
    policy_guard_diagnostics = None
    if "policy_guard_selected_counterfactual" in pred:
        policy_guard_diagnostics = {
            "selected_counterfactual": bool(
                pred["policy_guard_selected_counterfactual"]
            ),
            "base_score": float(pred["policy_guard_base_score"]),
            "counterfactual_score": float(pred["policy_guard_counterfactual_score"]),
            "score_margin": float(pred["policy_guard_score_margin"]),
            "gate_mode": str(pred["policy_guard_gate_mode"]),
        }
        for source_key, output_key, converter in (
            ("policy_guard_score_space", "score_space", str),
            (
                "policy_guard_candidate_supported",
                "candidate_supported",
                bool,
            ),
            (
                "policy_guard_candidate_delta_rms",
                "candidate_delta_rms",
                float,
            ),
            (
                "policy_guard_candidate_saturation_fraction",
                "candidate_saturation_fraction",
                float,
            ),
            (
                "policy_guard_target_binding_top1_mass",
                "target_binding_top1_mass",
                float,
            ),
            (
                "policy_guard_target_binding_entropy",
                "target_binding_entropy",
                float,
            ),
            (
                "policy_guard_target_binding_similarity_max",
                "target_binding_similarity_max",
                float,
            ),
        ):
            if source_key in pred:
                policy_guard_diagnostics[output_key] = converter(pred[source_key])
        eraf_shadow_enabled = bool(
            cfg.EVALUATION.get("entity_relation_shadow_audit", False)
        )
        eraf_oracle_enabled = bool(
            cfg.EVALUATION.get("entity_relation_oracle", False)
        )
        if bool(cfg.EVALUATION.get("entity_relation_diagnostics", False)) or (
            eraf_shadow_enabled
        ) or eraf_oracle_enabled:
            eraf = pred.get("policy_guard_eraf_diagnostics")
            if eraf is None:
                raise RuntimeError(
                    "ERAF diagnostics were requested from a non-V9 checkpoint."
                )
            policy_guard_diagnostics["_entity_relation_raw"] = {
                name: (
                    value.detach().cpu().numpy()
                    if torch.is_tensor(value)
                    else np.asarray(value)
                )
                for name, value in eraf.items()
            }
        if eraf_shadow_enabled:
            if "policy_guard_base_action" not in pred:
                raise RuntimeError(
                    "ERAF shadow audit requires the explicit immutable Base "
                    "action returned by PGC v4+."
                )
            policy_guard_diagnostics["shadow_action_integrity"] = (
                verify_shadow_action_integrity(
                    pred["action"],
                    pred["policy_guard_base_action"],
                    gate_mode=str(pred["policy_guard_gate_mode"]),
                )
            )

    action = _denormalize_action(action, processor)[0]  # [T, D]

    # The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    action[..., -1] = action[..., -1] * 2 - 1
    action = invert_gripper_action(action)
    if bool(cfg.EVALUATION.get("binarize_gripper", False)):
        action[..., -1] = np.sign(action[..., -1])
    return (
        action,
        imgs,
        predicted_future_frames,
        inference_latency_ms,
        policy_guard_diagnostics,
        pred.get("policy_guard_state"),
    )


def _get_max_steps(task_suite_name: str) -> int:
    suite_steps = {
        "libero_spatial": 400,
        "libero_object": 400,
        "libero_goal": 400,
        "libero_10": 700,
        "libero_90": 700,
    }
    if task_suite_name not in suite_steps:
        raise ValueError(f"Unknown task suite: {task_suite_name}")
    return suite_steps[task_suite_name]


def _resolve_max_steps(cfg: DictConfig) -> int:
    max_steps_cfg = cfg.EVALUATION.get("max_steps", None)
    max_steps = (
        _get_max_steps(cfg.EVALUATION.task_suite_name)
        if max_steps_cfg is None
        else int(max_steps_cfg)
    )
    if max_steps <= 0:
        raise ValueError(f"EVALUATION.max_steps must be positive, got {max_steps}.")
    return max_steps


def _oracle_phase_servo_config(cfg: DictConfig) -> OraclePhaseServoConfig:
    evaluation = cfg.EVALUATION
    config = OraclePhaseServoConfig(
        enabled=bool(evaluation.get("entity_relation_oracle_phase_servo", False)),
        scope=str(
            evaluation.get("entity_relation_oracle_servo_scope", "full")
        ),
        approach_gain=float(
            evaluation.get("entity_relation_oracle_servo_approach_gain", 4.0)
        ),
        transport_gain=float(
            evaluation.get("entity_relation_oracle_servo_transport_gain", 4.0)
        ),
        max_translation_action=float(
            evaluation.get(
                "entity_relation_oracle_servo_max_translation_action", 0.20
            )
        ),
        approach_height_m=float(
            evaluation.get("entity_relation_oracle_servo_approach_height_m", 0.08)
        ),
        transport_height_m=float(
            evaluation.get("entity_relation_oracle_servo_transport_height_m", 0.10)
        ),
        grasp_offset_m=float(
            evaluation.get("entity_relation_oracle_servo_grasp_offset_m", 0.01)
        ),
        release_height_m=float(
            evaluation.get("entity_relation_oracle_servo_release_height_m", 0.04)
        ),
        horizontal_tolerance_m=float(
            evaluation.get(
                "entity_relation_oracle_servo_horizontal_tolerance_m", 0.035
            )
        ),
        grasp_distance_m=float(
            evaluation.get("entity_relation_oracle_servo_grasp_distance_m", 0.035)
        ),
        release_distance_m=float(
            evaluation.get("entity_relation_oracle_servo_release_distance_m", 0.05)
        ),
        interaction_distance_m=float(
            evaluation.get(
                "entity_relation_oracle_servo_interaction_distance_m", 0.045
            )
        ),
    )
    if config.enabled:
        config.validate()
    return config


def _capture_libero_sim_state(env: Any) -> np.ndarray:
    """Return the exact flattened simulator state used by LIBERO reset APIs."""
    inner = getattr(env, "env", None)
    sim = None if inner is None else getattr(inner, "sim", None)
    if sim is None or not hasattr(sim, "get_state"):
        raise RuntimeError("PGC closed-loop capture requires env.env.sim.get_state().")
    state = sim.get_state()
    if hasattr(state, "flatten"):
        state = state.flatten()
    return np.asarray(state).copy()


def _state_sha256(state: np.ndarray) -> str:
    return _canonical_state_sha256(state)


def _write_closed_loop_capture_records(
    *,
    cfg: DictConfig,
    episode_idx: int,
    initial_state: np.ndarray,
    task_description: str,
    policy_instruction: str,
    counterfactual_metadata: dict[str, Any],
    counterfactual_diagnostics: dict[str, Any],
    captured_states: list[dict[str, Any]],
) -> int:
    """Persist failed pre-grasp rollout states without cross-worker writes."""
    capture_root_value = cfg.EVALUATION.get("closed_loop_capture_dir")
    if capture_root_value in (None, "", "null") or not captured_states:
        return 0
    target_objects = set(counterfactual_diagnostics["counterfactual_target_objects"])
    lifted_objects = set(counterfactual_diagnostics["lifted_objects"])
    if target_objects & lifted_objects:
        # V8 first repairs target acquisition. States from episodes that already
        # lifted the requested target belong to a later completion stage.
        return 0

    suite_name = str(cfg.EVALUATION.task_suite_name)
    task_id = int(cfg.EVALUATION.task_id)
    trial_dir = (
        Path(str(capture_root_value)).expanduser().resolve()
        / suite_name
        / f"task_{task_id:02d}"
        / f"trial_{episode_idx:03d}"
    )
    trial_dir.mkdir(parents=True, exist_ok=True)
    initial_state = np.asarray(initial_state).copy()
    written = 0
    for item in captured_states:
        replan_index = int(item["replan_index"])
        capture_id = (
            f"{suite_name}_task{task_id:02d}_trial{episode_idx:03d}_"
            f"replan{replan_index:04d}"
        )
        state = np.asarray(item["state"]).copy()
        state_path = trial_dir / f"{capture_id}.npz"
        np.savez_compressed(
            state_path,
            simulator_state=state,
            source_initial_state=initial_state,
        )
        record = {
            "format": "pgc_libero_closed_loop_capture_v1",
            "capture_id": capture_id,
            "state_file": state_path.name,
            "capture_state_sha256": _state_sha256(state),
            "source_initial_state_sha256": _state_sha256(initial_state),
            "task_suite_name": suite_name,
            "task_id": task_id,
            "trial_index": int(episode_idx),
            "replan_index": replan_index,
            "policy_step": int(item["policy_step"]),
            "pair_id": str(counterfactual_metadata.get("pair_id", "")),
            "correct_instruction": str(task_description),
            "counterfactual_instruction": str(policy_instruction),
            "source_goal_state": counterfactual_metadata["source_goal_state"],
            "counterfactual_goal_state": counterfactual_metadata[
                "counterfactual_goal_state"
            ],
            "episode_category": str(counterfactual_diagnostics["category"]),
            "target_objects": sorted(target_objects),
            "grasped_objects": counterfactual_diagnostics["grasped_objects"],
            "lifted_objects": counterfactual_diagnostics["lifted_objects"],
            "checkpoint": str(cfg.ckpt),
            "seed": None if cfg.get("seed") is None else int(cfg.seed),
        }
        record_path = trial_dir / f"{capture_id}.json"
        temporary_path = record_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(record, indent=2, cls=NumpyEncoder) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, record_path)
        written += 1
    return written


def _write_eraf_closed_loop_capture_records(
    *,
    cfg: DictConfig,
    episode_idx: int,
    initial_state: np.ndarray,
    task_description: str,
    policy_instruction: str,
    success: bool,
    captured_states: list[dict[str, Any]],
) -> int:
    """Persist phase-labelled Base-rollout states for V9.12 rebinding.

    This path is deliberately separate from the V8 acquisition capture above.
    V9.12 captures Correct, immutable-Base rollouts after the privileged shadow
    observer has assigned a state-derived phase.  The labels are never exposed
    to inference; a later offline builder restores the exact simulator states
    and constructs the ordinary training-only ERAF sidecar.
    """
    capture_root_value = cfg.EVALUATION.get(
        "entity_relation_closed_loop_capture_dir"
    )
    if capture_root_value in (None, "", "null") or not captured_states:
        return 0

    suite_name = str(cfg.EVALUATION.task_suite_name)
    task_id = int(cfg.EVALUATION.task_id)
    trial_dir = (
        Path(str(capture_root_value)).expanduser().resolve()
        / suite_name
        / f"task_{task_id:02d}"
        / f"trial_{episode_idx:03d}"
    )
    trial_dir.mkdir(parents=True, exist_ok=True)
    initial_state = np.asarray(initial_state).copy()
    written = 0
    for item in captured_states:
        replan_index = int(item["replan_index"])
        stage = str(item["online_stage_v2"])
        capture_id = (
            f"{suite_name}_task{task_id:02d}_trial{episode_idx:03d}_"
            f"replan{replan_index:04d}_{stage}"
        )
        state = np.asarray(item["state"]).copy()
        state_path = trial_dir / f"{capture_id}.npz"
        np.savez_compressed(
            state_path,
            simulator_state=state,
            source_initial_state=initial_state,
        )
        record = {
            "format": "pgc_v9_eraf_closed_loop_capture_v1",
            "capture_id": capture_id,
            "state_file": state_path.name,
            "capture_state_sha256": _state_sha256(state),
            "source_initial_state_sha256": _state_sha256(initial_state),
            "task_suite_name": suite_name,
            "task_id": task_id,
            "trial_index": int(episode_idx),
            "replan_index": replan_index,
            "policy_step": int(item["policy_step"]),
            "online_stage_v2": stage,
            "clause_statuses": list(item["clause_statuses"]),
            "phase_targets": [int(value) for value in item["phase_targets"]],
            "predicate_truth": [bool(value) for value in item["predicate_truth"]],
            "subject_grasped": [bool(value) for value in item["subject_grasped"]],
            "subject_ever_grasped": [
                bool(value) for value in item["subject_ever_grasped"]
            ],
            "correct_instruction": str(task_description),
            "policy_instruction": str(policy_instruction),
            "episode_success": bool(success),
            "rollout_policy": "immutable_base",
            "action_integrity": "selected_equals_immutable_base_exact",
            "checkpoint": str(cfg.ckpt),
            "seed": None if cfg.get("seed") is None else int(cfg.seed),
            "privileged_supervision": "training_only",
            "deployment_inputs": "rgb_language_proprio",
        }
        record_path = trial_dir / f"{capture_id}.json"
        temporary_path = record_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(record, indent=2, cls=NumpyEncoder) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, record_path)
        written += 1
    return written


def run_single_episode(
    env,
    initial_state,
    task_description: str,
    policy_instruction: str,
    mask_language: bool,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    episode_idx: int,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
    counterfactual_metadata: Optional[dict[str, Any]] = None,
    eraf_shadow_auditor: Optional[ERAFShadowAuditor] = None,
    eraf_oracle_provider: Optional[ERAFOracleProvider] = None,
) -> tuple[
    bool,
    list,
    list[dict[str, Any]],
    Optional[float],
    list[float],
    int,
    bool,
    Optional[dict[str, Any]],
    list[dict[str, Any]],
]:
    max_steps = _resolve_max_steps(cfg)
    oracle_phase_servo = _oracle_phase_servo_config(cfg)
    if oracle_phase_servo.enabled and eraf_oracle_provider is None:
        raise ValueError(
            "Oracle phase servo requires the live ERAF oracle provider."
        )
    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    num_steps_wait = int(cfg.EVALUATION.get("num_steps_wait", 5))
    use_action_ensembler = bool(cfg.EVALUATION.get("use_action_ensembler", False))
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    capture_steps = set(_get_future_frame_capture_steps(cfg)[1:])

    env.reset()
    obs = env.set_init_state(initial_state)
    counterfactual_tracker = None
    if counterfactual_metadata is not None:
        counterfactual_tracker = CounterfactualEpisodeTracker(
            env,
            source_goal_state=counterfactual_metadata["source_goal_state"],
            counterfactual_goal_state=counterfactual_metadata[
                "counterfactual_goal_state"
            ],
            lift_threshold_m=float(
                cfg.EVALUATION.get(
                    "counterfactual_lift_threshold_m",
                    0.04,
                )
            ),
        )
        counterfactual_tracker.observe(policy_step=0)
    if use_action_ensembler:
        ensembler = ActionEnsembler()
        ensembler.reset()

    replay_images = []
    predicted_future_video_clips: list[dict[str, Any]] = []
    episode_future_clip_psnr: list[float] = []
    pending_actions: list[list[float]] = []
    current_predicted_future_clip: Optional[dict[str, Any]] = None
    current_replan_step = 0
    current_replan_idx = -1
    inference_latencies_ms: list[float] = []
    policy_guard_decisions: list[dict[str, Any]] = []
    deployed_completion_only_memory = bool(
        getattr(
            model,
            "policy_guard_eraf_completion_only_memory",
            False,
        )
    )
    # Explicit per-episode state: never retained on the model, shared between
    # environments, or carried across LIBERO trials.  Evaluation can either
    # cut the recurrent PSCM channel completely or preserve only a monotonic
    # completed-clause bitset; neither mode changes checkpoint weights or the
    # ERAF/scheduler forward path.
    policy_guard_state = PolicyGuardStateController(
        reset_each_replan=bool(
            cfg.EVALUATION.get(
                "entity_relation_stateless_replan_ablation", False
            )
        ),
        completion_only=bool(
            not cfg.EVALUATION.get(
                "entity_relation_stateless_replan_ablation", False
            )
            and (
                deployed_completion_only_memory
                or cfg.EVALUATION.get(
                    "entity_relation_completion_only_memory_ablation", False
                )
            )
        ),
    )
    closed_loop_capture_enabled = cfg.EVALUATION.get("closed_loop_capture_dir") not in (
        None,
        "",
        "null",
    )
    capture_stride_replans = int(
        cfg.EVALUATION.get("closed_loop_capture_stride_replans", 1)
    )
    capture_max_states = int(
        cfg.EVALUATION.get("closed_loop_capture_max_states_per_episode", 12)
    )
    if capture_stride_replans <= 0 or capture_max_states <= 0:
        raise ValueError("PGC closed-loop capture stride/max states must be positive.")
    if closed_loop_capture_enabled and counterfactual_tracker is None:
        raise ValueError("PGC closed-loop capture requires counterfactual diagnostics.")
    captured_states: list[dict[str, Any]] = []
    eraf_capture_enabled = cfg.EVALUATION.get(
        "entity_relation_closed_loop_capture_dir"
    ) not in (None, "", "null")
    eraf_capture_stride = int(
        cfg.EVALUATION.get(
            "entity_relation_closed_loop_capture_stride_replans", 1
        )
    )
    eraf_capture_max_states = int(
        cfg.EVALUATION.get(
            "entity_relation_closed_loop_capture_max_states_per_episode", 48
        )
    )
    eraf_capture_stages = {
        value.strip()
        for value in str(
            cfg.EVALUATION.get(
                "entity_relation_closed_loop_capture_stages",
                "initial_search,holding,released_unfinished,next_clause_search",
            )
        ).split(",")
        if value.strip()
    }
    allowed_eraf_capture_stages = {
        "initial_search",
        "holding",
        "released_unfinished",
        "next_clause_search",
    }
    if eraf_capture_enabled:
        if eraf_shadow_auditor is None:
            raise ValueError(
                "ERAF closed-loop capture requires the passive shadow auditor."
            )
        if eraf_capture_stride <= 0 or eraf_capture_max_states <= 0:
            raise ValueError(
                "ERAF closed-loop capture stride/max states must be positive."
            )
        unknown_stages = eraf_capture_stages - allowed_eraf_capture_stages
        if unknown_stages:
            raise ValueError(
                "Unsupported ERAF closed-loop capture stages: "
                f"{sorted(unknown_stages)}."
            )
    eraf_captured_states: list[dict[str, Any]] = []
    inference_replan_index = -1

    t = 0
    policy_steps_executed = 0
    done = False
    pbar = tqdm(total=max_steps + num_steps_wait, desc=f"Episode {episode_idx + 1}")
    while t < max_steps + num_steps_wait:
        pbar.update(1)
        if t < num_steps_wait:
            obs, _, done, _ = env.step(get_libero_dummy_action())
            t += 1
            continue

        if len(pending_actions) == 0:
            inference_replan_index += 1
            oracle_policy_input = (
                eraf_oracle_provider.policy_input(
                    obs=obs,
                    episode_idx=episode_idx,
                )
                if eraf_oracle_provider is not None
                else None
            )
            eraf_capture_candidate = None
            if (
                eraf_capture_enabled
                and inference_replan_index % eraf_capture_stride == 0
                and len(eraf_captured_states) < eraf_capture_max_states
            ):
                eraf_capture_candidate = {
                    "replan_index": inference_replan_index,
                    "policy_step": policy_steps_executed,
                    "state": _capture_libero_sim_state(env),
                }
            capture_before_interaction = True
            if counterfactual_tracker is not None:
                targets = counterfactual_tracker.counterfactual_target_objects
                capture_before_interaction = not bool(
                    targets
                    & (
                        counterfactual_tracker.grasped_objects
                        | counterfactual_tracker.lifted_objects
                    )
                )
            if (
                closed_loop_capture_enabled
                and capture_before_interaction
                and inference_replan_index % capture_stride_replans == 0
                and len(captured_states) < capture_max_states
            ):
                captured_states.append(
                    {
                        "replan_index": inference_replan_index,
                        "policy_step": policy_steps_executed,
                        "state": _capture_libero_sim_state(env),
                    }
                )
            (
                action_chunk,
                imgs,
                predicted_future_frames,
                inference_latency_ms,
                policy_guard_diagnostics,
                next_policy_guard_state,
            ) = _predict_action_chunk(
                obs=obs,
                policy_instruction=policy_instruction,
                mask_language=mask_language,
                model=model,
                processor=processor,
                cfg=cfg,
                action_horizon=action_horizon,
                input_w=input_w,
                input_h=input_h,
                model_device=model_device,
                policy_guard_state=policy_guard_state.state_for_replan(),
                policy_guard_eraf_oracle=oracle_policy_input,
            )
            if oracle_phase_servo.enabled:
                if oracle_policy_input is None or eraf_oracle_provider is None:
                    raise RuntimeError(
                        "Oracle phase servo lost its live privileged input."
                    )
                action_chunk, servo_diagnostics = apply_oracle_phase_servo(
                    action_chunk,
                    obs=obs,
                    oracle=oracle_policy_input,
                    workspace_min=eraf_oracle_provider.contract.workspace_min,
                    workspace_max=eraf_oracle_provider.contract.workspace_max,
                    config=oracle_phase_servo,
                )
                if policy_guard_diagnostics is None:
                    raise RuntimeError(
                        "Oracle phase servo requires policy-guard diagnostics."
                    )
                servo_diagnostics["replan_index"] = inference_replan_index
                servo_diagnostics["policy_step"] = policy_steps_executed
                policy_guard_diagnostics["entity_relation_oracle_phase_servo"] = (
                    servo_diagnostics
                )
            policy_guard_state.accept_model_state(next_policy_guard_state)
            inference_latencies_ms.append(inference_latency_ms)
            if policy_guard_diagnostics is not None:
                eraf_raw = policy_guard_diagnostics.pop("_entity_relation_raw", None)
                if eraf_raw is not None:
                    if eraf_oracle_provider is not None:
                        oracle_enabled = bool(
                            np.asarray(
                                eraf_raw.get("oracle_eraf_enabled", False)
                            ).reshape(-1)[0]
                        )
                        if not oracle_enabled:
                            raise RuntimeError(
                                "Oracle ERAF was requested but the model did "
                                "not acknowledge the privileged intervention."
                            )
                        selected_clause = int(
                            np.asarray(
                                eraf_raw["oracle_selected_clause"]
                            ).reshape(-1)[0]
                        )
                        policy_guard_diagnostics["entity_relation_oracle"] = {
                            "enabled": True,
                            "selected_clause": selected_clause,
                        }
                    if eraf_shadow_auditor is not None:
                        shadow_record = eraf_shadow_auditor.observe(
                            obs=obs,
                            diagnostics=eraf_raw,
                            episode_idx=episode_idx,
                            replan_idx=inference_replan_index,
                            policy_step=policy_steps_executed,
                        )
                        policy_guard_diagnostics[
                            "entity_relation_shadow"
                        ] = shadow_record
                        if (
                            eraf_capture_candidate is not None
                            and shadow_record["online_stage_v2"]
                            in eraf_capture_stages
                        ):
                            eraf_captured_states.append(
                                {
                                    **eraf_capture_candidate,
                                    "online_stage_v2": shadow_record[
                                        "online_stage_v2"
                                    ],
                                    "clause_statuses": shadow_record[
                                        "clause_statuses"
                                    ],
                                    "phase_targets": shadow_record[
                                        "phase_targets"
                                    ],
                                    "predicate_truth": shadow_record[
                                        "predicate_truth"
                                    ],
                                    "subject_grasped": shadow_record[
                                        "subject_grasped"
                                    ],
                                    "subject_ever_grasped": shadow_record[
                                        "subject_ever_grasped"
                                    ],
                                }
                            )
                    if bool(cfg.EVALUATION.get("entity_relation_diagnostics", False)):
                        policy_guard_diagnostics["entity_relation"] = (
                            _save_eraf_diagnostics(
                                cfg=cfg,
                                images=imgs,
                                diagnostics=eraf_raw,
                                episode_idx=episode_idx,
                                replan_idx=inference_replan_index,
                            )
                        )
                policy_guard_decisions.append(policy_guard_diagnostics)
            if predicted_future_frames is not None:
                current_replan_idx += 1
                current_predicted_future_clip = {
                    "replan_idx": current_replan_idx,
                    "gt_frames": [imgs.copy()],
                    "pred_frames": predicted_future_frames,
                }
            else:
                current_predicted_future_clip = None
            current_replan_step = 0
            if use_action_ensembler:
                ensembler.add_actions(action_chunk, t)
                pending_actions = [
                    ensembler.get_action(ts).tolist()
                    for ts in range(t, t + replan_steps)
                ]
            else:
                pending_actions = action_chunk[:replan_steps].tolist()
            replay_images.append(imgs.copy())
        else:
            imgs = get_libero_image(obs)
            replay_images.append(imgs.copy())

        obs, _, done, _ = env.step(pending_actions.pop(0))
        policy_steps_executed += 1
        if counterfactual_tracker is not None:
            counterfactual_tracker.observe(policy_step=policy_steps_executed)
        if visualize_future_video and current_predicted_future_clip is not None:
            current_replan_step += 1
            if current_replan_step in capture_steps:
                current_predicted_future_clip["gt_frames"].append(get_libero_image(obs))
            if done or len(pending_actions) == 0:
                expected_frame_count = 1 + sum(
                    1
                    for capture_step in capture_steps
                    if capture_step <= current_replan_step
                )
                gt_len = len(current_predicted_future_clip["gt_frames"])
                pred_len = len(current_predicted_future_clip["pred_frames"])
                assert gt_len == expected_frame_count, (
                    "GT future frames do not match expected capture count: "
                    f"gt_len={gt_len} expected={expected_frame_count} "
                    f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']} "
                    f"current_replan_step={current_replan_step} capture_steps={sorted(capture_steps)}."
                )
                assert pred_len >= expected_frame_count, (
                    "Predicted future frames shorter than expected capture count: "
                    f"pred_len={pred_len} expected={expected_frame_count} "
                    f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']}."
                )
                if pred_len != expected_frame_count:
                    logging.info(
                        "Align predicted clip length to executed steps: "
                        "episode=%s replan=%s done=%s expected=%s pred_full=%s",
                        episode_idx,
                        current_predicted_future_clip["replan_idx"],
                        done,
                        expected_frame_count,
                        pred_len,
                    )
                current_predicted_future_clip["pred_frames"] = (
                    current_predicted_future_clip["pred_frames"][:expected_frame_count]
                )
                assert len(current_predicted_future_clip["gt_frames"]) == len(
                    current_predicted_future_clip["pred_frames"]
                ), (
                    "GT/pred frame count mismatch after alignment: "
                    f"len(gt_frames)={len(current_predicted_future_clip['gt_frames'])} "
                    f"len(pred_frames)={len(current_predicted_future_clip['pred_frames'])} "
                    f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']}."
                )
                clip_psnr = _compute_clip_mean_psnr(
                    current_predicted_future_clip["gt_frames"],
                    current_predicted_future_clip["pred_frames"],
                )
                if clip_psnr is not None:
                    episode_future_clip_psnr.append(clip_psnr)
                predicted_future_video_clips.append(current_predicted_future_clip)
                current_predicted_future_clip = None
        if done:
            break
        t += 1
    pbar.close()

    episode_mean_psnr = (
        float(np.mean(episode_future_clip_psnr))
        if len(episode_future_clip_psnr) > 0
        else None
    )
    counterfactual_diagnostics = (
        None
        if counterfactual_tracker is None
        else counterfactual_tracker.result(episode_idx=episode_idx)
    )
    if closed_loop_capture_enabled:
        if counterfactual_diagnostics is None or counterfactual_metadata is None:
            raise RuntimeError(
                "PGC closed-loop capture finished without counterfactual metadata."
            )
        capture_count = _write_closed_loop_capture_records(
            cfg=cfg,
            episode_idx=episode_idx,
            initial_state=np.asarray(initial_state),
            task_description=task_description,
            policy_instruction=policy_instruction,
            counterfactual_metadata=counterfactual_metadata,
            counterfactual_diagnostics=counterfactual_diagnostics,
            captured_states=captured_states,
        )
        counterfactual_diagnostics["closed_loop_capture_count"] = int(capture_count)
    if eraf_capture_enabled:
        capture_count = _write_eraf_closed_loop_capture_records(
            cfg=cfg,
            episode_idx=episode_idx,
            initial_state=np.asarray(initial_state),
            task_description=task_description,
            policy_instruction=policy_instruction,
            success=bool(done),
            captured_states=eraf_captured_states,
        )
        logging.info(
            "Wrote %d V9.12 phase-labelled ERAF states for episode %d.",
            capture_count,
            episode_idx,
        )
    return (
        bool(done),
        replay_images,
        predicted_future_video_clips,
        episode_mean_psnr,
        inference_latencies_ms,
        int(policy_steps_executed),
        bool(not done and policy_steps_executed >= max_steps),
        counterfactual_diagnostics,
        policy_guard_decisions,
    )


def run_single_task(
    task,
    initial_states,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    video_dir: Path,
    predicted_video_dir: Path,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
) -> dict:
    eraf_shadow_enabled = bool(
        cfg.EVALUATION.get("entity_relation_shadow_audit", False)
    )
    eraf_oracle_enabled = bool(
        cfg.EVALUATION.get("entity_relation_oracle", False)
    )
    oracle_phase_servo = _oracle_phase_servo_config(cfg)
    if oracle_phase_servo.enabled and not eraf_oracle_enabled:
        raise ValueError(
            "EVALUATION.entity_relation_oracle_phase_servo requires "
            "EVALUATION.entity_relation_oracle=true."
        )
    stateless_replan_ablation = bool(
        cfg.EVALUATION.get(
            "entity_relation_stateless_replan_ablation", False
        )
    )
    completion_only_ablation = bool(
        cfg.EVALUATION.get(
            "entity_relation_completion_only_memory_ablation", False
        )
    )
    deployed_completion_only_memory = bool(
        getattr(
            model,
            "policy_guard_eraf_completion_only_memory",
            False,
        )
    )
    if stateless_replan_ablation and completion_only_ablation:
        raise ValueError(
            "Stateless and completion-only policy-state ablations are mutually "
            "exclusive."
        )
    if (stateless_replan_ablation or completion_only_ablation) and not (
        eraf_shadow_enabled
    ):
        raise ValueError(
            "Policy-state ablations require the passive ERAF shadow "
            "audit so the cut state channel and immutable Base action are "
            "independently verified."
        )
    env, task_description = get_libero_env(
        task,
        LIBERO_ENV_RESOLUTION,
        cfg.get("seed"),
        camera_segmentations=(
            "element" if eraf_shadow_enabled or eraf_oracle_enabled else None
        ),
    )
    (
        instruction_condition,
        policy_instruction,
        mask_language,
        intervention_record,
    ) = _resolve_language_intervention(task_description, cfg)
    counterfactual_metadata = None
    if instruction_condition == "counterfactual":
        if intervention_record is None:
            raise ValueError(
                "Counterfactual evaluation requires a paired manifest record."
            )
        counterfactual_metadata = _activate_counterfactual_goal(
            env,
            intervention_record,
            policy_instruction,
        )
    eraf_shadow_auditor = None
    if eraf_shadow_enabled:
        if instruction_condition not in {"correct", "counterfactual"}:
            raise ValueError(
                "EVALUATION.entity_relation_shadow_audit supports only "
                "correct or counterfactual instructions."
            )
        if (
            not bool(getattr(model, "policy_guard_enabled", False))
            or int(getattr(model, "policy_guard_version", -1)) != 9
        ):
            raise ValueError("ERAF shadow audit requires a PGC v9 checkpoint.")
        if bool(model.training):
            raise ValueError(
                "ERAF shadow audit requires model.eval() so dropout cannot "
                "perturb the Base rollout."
            )
        if (stateless_replan_ablation or completion_only_ablation) and int(
            getattr(
                model,
                "policy_guard_eraf_grounding_objective_version",
                0,
            )
        ) < 14:
            raise ValueError(
                "Policy-state ablations require a V9.13+ "
                "phase-memory checkpoint."
            )
        if str(getattr(model, "policy_guard_gate_mode", "")) != "base":
            raise ValueError(
                "ERAF shadow audit is passive and requires "
                "model.policy_guard.gate_mode=base."
            )
        sidecar_value = cfg.EVALUATION.get("entity_relation_shadow_sidecar_dir")
        if sidecar_value in (None, "", "null"):
            raise ValueError(
                "ERAF shadow audit requires "
                "EVALUATION.entity_relation_shadow_sidecar_dir."
            )
        eraf_shadow_auditor = ERAFShadowAuditor(
            env=env,
            policy_instruction=policy_instruction,
            instruction_condition=instruction_condition,
            contract=ERAFShadowContract.load(str(sidecar_value)),
            counterfactual_metadata=counterfactual_metadata,
            all_entity_role_gate=(
                int(
                    getattr(
                        model,
                        "policy_guard_eraf_grounding_objective_version",
                        0,
                    )
                )
                >= 11
            ),
        )
    eraf_oracle_provider = None
    if eraf_oracle_enabled:
        if eraf_shadow_enabled:
            raise ValueError(
                "Oracle ERAF and passive ERAF shadow audit are mutually exclusive."
            )
        if instruction_condition not in {"correct", "counterfactual"}:
            raise ValueError(
                "EVALUATION.entity_relation_oracle supports only correct or "
                "counterfactual instructions."
            )
        if (
            not bool(getattr(model, "policy_guard_enabled", False))
            or int(getattr(model, "policy_guard_version", -1)) != 9
        ):
            raise ValueError("Oracle ERAF requires a PGC v9 checkpoint.")
        if bool(model.training):
            raise ValueError("Oracle ERAF requires model.eval().")
        if str(getattr(model, "policy_guard_gate_mode", "")) != "counterfactual":
            raise ValueError(
                "Oracle ERAF must execute the Proposal and therefore requires "
                "model.policy_guard.gate_mode=counterfactual."
            )
        if bool(cfg.EVALUATION.get("visualize_future_video", False)):
            raise ValueError(
                "Oracle ERAF does not support future-video visualization."
            )
        sidecar_value = cfg.EVALUATION.get("entity_relation_oracle_sidecar_dir")
        if sidecar_value in (None, "", "null"):
            raise ValueError(
                "Oracle ERAF requires "
                "EVALUATION.entity_relation_oracle_sidecar_dir."
            )
        eraf_oracle_provider = ERAFOracleProvider(
            env=env,
            policy_instruction=policy_instruction,
            instruction_condition=instruction_condition,
            contract=ERAFShadowContract.load(str(sidecar_value)),
            counterfactual_metadata=counterfactual_metadata,
            all_entity_role_gate=True,
        )
    counterfactual_diagnostics_enabled = bool(
        cfg.EVALUATION.get("counterfactual_diagnostics", False)
    )
    if counterfactual_diagnostics_enabled and counterfactual_metadata is None:
        raise ValueError(
            "EVALUATION.counterfactual_diagnostics=true requires "
            "instruction_condition=counterfactual."
        )
    closed_loop_capture_enabled = cfg.EVALUATION.get("closed_loop_capture_dir") not in (
        None,
        "",
        "null",
    )
    if closed_loop_capture_enabled and not counterfactual_diagnostics_enabled:
        raise ValueError(
            "EVALUATION.closed_loop_capture_dir requires "
            "counterfactual_diagnostics=true."
        )
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    results = {
        "successes": 0,
        "failure_episodes": [],
        "success_episodes": [],
        "episode_policy_steps": [],
        "horizon_timeout_episodes": [],
        "max_policy_steps": _resolve_max_steps(cfg),
        "task_description": task_description,
        "instruction_condition": instruction_condition,
        "policy_instruction": policy_instruction,
        "mask_language": mask_language,
        "inference_latencies_ms": [],
        "latency_p50_ms": None,
        "latency_p95_ms": None,
        "success_predicate": (
            "counterfactual" if instruction_condition == "counterfactual" else "source"
        ),
        "policy_guard_episode_diagnostics": [],
        "closed_loop_capture_count": 0,
    }
    if intervention_record is not None:
        manifest_value = cfg.EVALUATION.get("language_intervention_manifest")
        manifest_path = Path(
            os.path.expanduser(os.path.expandvars(str(manifest_value)))
        ).resolve()
        results.update(
            {
                "language_intervention_pair_id": intervention_record.get("pair_id"),
                "language_intervention_manifest": str(manifest_path),
                "language_intervention_manifest_sha256": sha256_file(manifest_path),
            }
        )
    if instruction_condition == "paraphrase":
        results.update(
            {
                "language_ood_variant": str(
                    cfg.EVALUATION.get("language_ood_variant")
                )
                .strip()
                .casefold(),
                "language_ood_canonical_instruction": task_description,
                "language_ood_policy_training_exact_match": False,
                "language_ood_source_goal_unchanged": True,
                "language_ood_prompt_wrapper_unchanged": True,
            }
        )
    if eraf_shadow_enabled:
        results["eraf_shadow_audit"] = {
            "enabled": True,
            "observer_only": True,
            "executed_policy": "immutable_base",
            "privileged_labels": "evaluation_only",
            "policy_state_mode": (
                "reset_each_replan"
                if stateless_replan_ablation
                else (
                    "completion_only"
                    if (
                        completion_only_ablation
                        or deployed_completion_only_memory
                    )
                    else "recurrent"
                )
            ),
            "stateless_replan_ablation": stateless_replan_ablation,
            "completion_only_memory_ablation": completion_only_ablation,
            "completion_only_memory_deployed": (
                deployed_completion_only_memory
            ),
            "records": [],
        }
    if eraf_oracle_enabled:
        results["eraf_oracle"] = {
            "enabled": True,
            "causal_intervention": True,
            "privileged_labels": "evaluation_only_mujoco_bddl",
            "executed_policy": "forced_proposal",
            "intervention_scope": (
                "clause_predicate_entity_geometry_truth_phase_scheduler"
            ),
            "learned_components_unchanged": (
                "eraf_relation_reasoner_bridge_and_action_proposal"
            ),
        }
    if oracle_phase_servo.enabled:
        results["eraf_oracle_phase_servo"] = {
            "enabled": True,
            "privileged_evaluation_only": True,
            "deployment_eligible": False,
            "translation": "live_subject_goal_phase_cartesian_servo",
            "orientation": "learned_proposal_passthrough",
            "placement_gripper": "explicit_phase_command",
            "interaction_near_fixture": "learned_proposal_passthrough",
            "config": {
                name: getattr(oracle_phase_servo, name)
                for name in oracle_phase_servo.__dataclass_fields__
            },
        }
    if intervention_record is not None:
        results["pair_id"] = intervention_record.get("pair_id")
    if counterfactual_metadata is not None:
        results.update(counterfactual_metadata)
    if counterfactual_diagnostics_enabled:
        results["counterfactual_episode_diagnostics"] = []
        results["counterfactual_behavior_counts"] = empty_behavior_counts()
    if visualize_future_video:
        results["episode_future_video_psnr"] = []
        results["future_video_psnr_mean"] = None

    for trial_idx in range(int(cfg.EVALUATION.num_trials)):
        (
            success,
            replay_images,
            predicted_future_video_clips,
            episode_mean_psnr,
            inference_latencies_ms,
            policy_steps_executed,
            horizon_timeout,
            counterfactual_diagnostics,
            policy_guard_decisions,
        ) = run_single_episode(
            env=env,
            initial_state=initial_states[trial_idx],
            task_description=task_description,
            policy_instruction=policy_instruction,
            mask_language=mask_language,
            model=model,
            processor=processor,
            cfg=cfg,
            episode_idx=trial_idx,
            action_horizon=action_horizon,
            input_w=input_w,
            input_h=input_h,
            model_device=model_device,
            counterfactual_metadata=(
                counterfactual_metadata if counterfactual_diagnostics_enabled else None
            ),
            eraf_shadow_auditor=eraf_shadow_auditor,
            eraf_oracle_provider=eraf_oracle_provider,
        )
        results["inference_latencies_ms"].extend(inference_latencies_ms)
        results["episode_policy_steps"].append(policy_steps_executed)
        results["policy_guard_episode_diagnostics"].append(
            {
                "episode": trial_idx,
                "decision_count": len(policy_guard_decisions),
                "override_count": sum(
                    int(item["selected_counterfactual"])
                    for item in policy_guard_decisions
                ),
                "decisions": policy_guard_decisions,
            }
        )
        if eraf_shadow_enabled:
            results["eraf_shadow_audit"]["records"].extend(
                item["entity_relation_shadow"]
                for item in policy_guard_decisions
                if "entity_relation_shadow" in item
            )
        if horizon_timeout:
            results["horizon_timeout_episodes"].append(trial_idx)
        if success:
            results["successes"] += 1
            results["success_episodes"].append(trial_idx)
        else:
            results["failure_episodes"].append(trial_idx)
        if counterfactual_diagnostics_enabled:
            if counterfactual_diagnostics is None:
                raise RuntimeError(
                    "Counterfactual diagnostics were enabled but no episode "
                    "diagnostics were returned."
                )
            counterfactual_diagnostics["policy_steps"] = int(policy_steps_executed)
            counterfactual_diagnostics["horizon_timeout"] = bool(horizon_timeout)
            results["counterfactual_episode_diagnostics"].append(
                counterfactual_diagnostics
            )
            results["closed_loop_capture_count"] += int(
                counterfactual_diagnostics.get("closed_loop_capture_count", 0)
            )
            category = str(counterfactual_diagnostics["category"])
            results["counterfactual_behavior_counts"][category] += 1
        if visualize_future_video:
            results["episode_future_video_psnr"].append(episode_mean_psnr)

        save_rollout_video(
            video_dir,
            replay_images,
            f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
            success=success,
            task_description=task_description,
        )
        if visualize_future_video:
            if len(predicted_future_video_clips) == 0:
                logging.warning(
                    "No predicted future frames collected for task %s trial %s.",
                    cfg.EVALUATION.task_id,
                    trial_idx,
                )
            else:
                all_gt_frames = []
                all_pred_frames = []
                for clip in predicted_future_video_clips:
                    all_gt_frames.extend(clip["gt_frames"])
                    all_pred_frames.extend(clip["pred_frames"])
                    save_prediction_video(
                        predicted_video_dir,
                        clip["gt_frames"],
                        clip["pred_frames"],
                        f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
                        clip["replan_idx"],
                        success=success,
                        task_description=task_description,
                    )
                save_prediction_video(
                    predicted_video_dir,
                    all_gt_frames,
                    all_pred_frames,
                    f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
                    "all",
                    success=success,
                    task_description=task_description,
                )

    if visualize_future_video:
        valid_episode_psnr = [
            x for x in results["episode_future_video_psnr"] if x is not None
        ]
        if len(valid_episode_psnr) > 0:
            results["future_video_psnr_mean"] = float(np.mean(valid_episode_psnr))
    if results["inference_latencies_ms"]:
        results["latency_p50_ms"] = float(
            np.percentile(results["inference_latencies_ms"], 50)
        )
        results["latency_p95_ms"] = float(
            np.percentile(results["inference_latencies_ms"], 95)
        )
    policy_guard_decisions = [
        decision
        for episode in results["policy_guard_episode_diagnostics"]
        for decision in episode["decisions"]
    ]
    results["policy_guard_decision_count"] = len(policy_guard_decisions)
    results["policy_guard_override_count"] = sum(
        int(item["selected_counterfactual"]) for item in policy_guard_decisions
    )
    results["policy_guard_override_rate"] = float(
        results["policy_guard_override_count"]
    ) / max(1, results["policy_guard_decision_count"])
    if oracle_phase_servo.enabled:
        episode_servo_records = [
            [
                decision["entity_relation_oracle_phase_servo"]
                for decision in episode["decisions"]
                if "entity_relation_oracle_phase_servo" in decision
            ]
            for episode in results["policy_guard_episode_diagnostics"]
        ]
        results["eraf_oracle_phase_servo"]["summary"] = (
            summarize_oracle_phase_servo(episode_servo_records)
        )
    if policy_guard_decisions:
        results["policy_guard_base_score_mean"] = float(
            np.mean([item["base_score"] for item in policy_guard_decisions])
        )
        results["policy_guard_counterfactual_score_mean"] = float(
            np.mean([item["counterfactual_score"] for item in policy_guard_decisions])
        )
        results["policy_guard_score_margin_mean"] = float(
            np.mean([item["score_margin"] for item in policy_guard_decisions])
        )
        score_spaces = sorted(
            {
                str(item["score_space"])
                for item in policy_guard_decisions
                if "score_space" in item
            }
        )
        if score_spaces:
            results["policy_guard_score_spaces"] = score_spaces
        supported = [
            bool(item["candidate_supported"])
            for item in policy_guard_decisions
            if "candidate_supported" in item
        ]
        if supported:
            results["policy_guard_candidate_supported_count"] = sum(supported)
            results["policy_guard_candidate_supported_rate"] = float(np.mean(supported))
        delta_rms = [
            float(item["candidate_delta_rms"])
            for item in policy_guard_decisions
            if "candidate_delta_rms" in item
        ]
        if delta_rms:
            results["policy_guard_candidate_delta_rms_mean"] = float(np.mean(delta_rms))
            results["policy_guard_candidate_delta_rms_max"] = float(np.max(delta_rms))
        saturation = [
            float(item["candidate_saturation_fraction"])
            for item in policy_guard_decisions
            if "candidate_saturation_fraction" in item
        ]
        if saturation:
            results["policy_guard_candidate_saturation_fraction_mean"] = float(
                np.mean(saturation)
            )
            results["policy_guard_candidate_saturation_fraction_max"] = float(
                np.max(saturation)
            )
        for decision_key, result_key in (
            ("target_binding_top1_mass", "policy_guard_target_binding_top1_mass_mean"),
            ("target_binding_entropy", "policy_guard_target_binding_entropy_mean"),
            (
                "target_binding_similarity_max",
                "policy_guard_target_binding_similarity_max_mean",
            ),
        ):
            values = [
                float(item[decision_key])
                for item in policy_guard_decisions
                if decision_key in item
            ]
            if values:
                results[result_key] = float(np.mean(values))
    if eraf_shadow_enabled:
        shadow_records = results["eraf_shadow_audit"]["records"]
        action_integrity = [
            item["shadow_action_integrity"]
            for item in policy_guard_decisions
            if "shadow_action_integrity" in item
        ]
        if len(shadow_records) != len(policy_guard_decisions):
            raise RuntimeError(
                "ERAF shadow audit did not produce one privileged record per "
                f"decision: records={len(shadow_records)} "
                f"decisions={len(policy_guard_decisions)}."
            )
        results["eraf_shadow_audit"]["summary"] = summarize_eraf_shadow_records(
            shadow_records,
            action_integrity=action_integrity,
        )
    if instruction_condition == "shuffled":
        # The simulator success predicate still represents the original task.
        results["default_task_successes"] = results["successes"]
    elif instruction_condition == "counterfactual":
        # The simulator success predicate was replaced before reset/rollout.
        results["counterfactual_successes"] = results["successes"]
    try:
        env.close()
    except Exception:
        # Evaluation results are already complete at this point. Keep a
        # renderer-cleanup failure from discarding them, but retain the full
        # traceback in the task log for diagnosis.
        logging.warning("Failed to close LIBERO environment cleanly.", exc_info=True)
    return results


@hydra.main(
    version_base="1.3", config_path="../../configs", config_name="sim_libero.yaml"
)
def eval_single_process(cfg: DictConfig):
    start_time = time.time()
    partial_state = PartialState()
    partial_state.config = cfg

    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)

    if cfg.ckpt is None:
        raise ValueError("cfg.ckpt must not be None.")
    _validate_visualize_future_video_cfg(cfg)

    env_num = int(cfg.EVALUATION.get("env_num", 1))
    if env_num != 1:
        raise ValueError(
            "Only env_num=1 is supported in eval_libero_single.py. "
            "Use run_libero_manager/run_libero_parallel_test.sh for multi-GPU task parallelism."
        )

    model_device = _resolve_eval_device(cfg)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    _load_model_checkpoint(model, str(cfg.ckpt))
    model = model.to(model_device).eval()

    dataset_stats_path = _resolve_dataset_stats_path(cfg)
    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)
    logging.info("Using dataset stats: %s", dataset_stats_path)

    action_horizon_cfg = cfg.EVALUATION.get("action_horizon", None)
    if action_horizon_cfg is None:
        action_horizon = int(cfg.data.train.num_frames) - 1
    else:
        action_horizon = int(action_horizon_cfg)
    if action_horizon <= 0:
        raise ValueError(
            f"EVALUATION.action_horizon must be positive, got {action_horizon}"
        )

    video_size = cfg.data.train.get("video_size", [224, 224])
    if len(video_size) != 2:
        raise ValueError(f"data.train.video_size must be [H, W], got {video_size}")
    input_h = int(video_size[0])
    input_w = int(video_size[1])
    concat_multi_camera = cfg.data.train.get("concat_multi_camera", None)
    shape_meta_images = [meta["shape"] for meta in processor.shape_meta["images"]]

    local_log_dir = Path(cfg.EVALUATION.output_dir)
    local_log_dir.mkdir(parents=True, exist_ok=True)
    video_dir = local_log_dir / cfg.EVALUATION.task_suite_name / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    predicted_video_dir = (
        local_log_dir / cfg.EVALUATION.task_suite_name / "predicted_videos"
    )
    if bool(cfg.EVALUATION.get("visualize_future_video", False)):
        predicted_video_dir.mkdir(parents=True, exist_ok=True)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.EVALUATION.task_suite_name]()
    task = task_suite.get_task(cfg.EVALUATION.task_id)
    initial_states = load_libero_task_init_states(
        task_suite,
        int(cfg.EVALUATION.task_id),
        get_libero_path("init_states"),
    )

    while len(initial_states) < int(cfg.EVALUATION.num_trials):
        initial_states.extend(
            initial_states[: (int(cfg.EVALUATION.num_trials) - len(initial_states))]
        )

    results = {
        "task_suite": cfg.EVALUATION.task_suite_name,
        "task_id": cfg.EVALUATION.task_id,
        "task_description": None,
        "successes": 0,
        "total_episodes": int(cfg.EVALUATION.num_trials),
        "gpu_id": int(cfg.gpu_id),
        "success_episodes": [],
        "failure_episodes": [],
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration": 0,
    }

    logging.info("Running LIBERO evaluation with env_num=1")
    task_results = run_single_task(
        task=task,
        initial_states=initial_states,
        model=model,
        processor=processor,
        cfg=cfg,
        video_dir=video_dir,
        predicted_video_dir=predicted_video_dir,
        action_horizon=action_horizon,
        input_w=input_w,
        input_h=input_h,
        model_device=model_device,
    )
    results.update(task_results)

    results["duration"] = time.time() - start_time
    output_dir = Path(cfg.EVALUATION.output_dir) / cfg.EVALUATION.task_suite_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = (
        output_dir / f"gpu{cfg.gpu_id}_task{cfg.EVALUATION.task_id}_results.json"
    )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)

    print(
        f"Task {cfg.EVALUATION.task_id} completed: "
        f"{results['successes']}/{cfg.EVALUATION.num_trials} successes"
    )
    if results.get("future_video_psnr_mean") is not None:
        print(
            f"Task {cfg.EVALUATION.task_id} future-video PSNR mean: {results['future_video_psnr_mean']:.4f}"
        )
    print(f"Time taken: {results['duration']:.2f} seconds")
    return results


if __name__ == "__main__":
    eval_single_process()
