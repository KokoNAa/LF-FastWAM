"""Training-only CF loss masks; never filter samples or select an action path."""

from enum import IntEnum


class CorrectiveVerification(IntEnum):
    NONE = 0
    TARGET_LIFT = 1
    COUNTERFACTUAL_GOAL = 2


MODES = (
    "none",
    "mask_lift_corrective",
    "mask_corrective_ranking",
    "mask_lift_ranking",
)
CONTRACT = "cf_loss_numerator_masks_v1_same_samples_forwards_and_denominators"
POSITIVE_ONLY_RANKING_CONTRACT = (
    "counterfactual_goal_correct_error_gradient_with_detached_wrong_error_threshold"
)
UNIVERSAL_POSITIVE_ONLY_RANKING_CONTRACT = (
    "all_paired_language_correct_error_gradient_with_detached_wrong_error_threshold"
)
PAIRED_SEMANTIC_CONTRAST_CONTRACT = (
    "same_state_bidirectional_injected_context_to_frozen_language_hinge_v1"
)
ACTION_VIOLATION_GATED_SEMANTIC_CONTRAST_CONTRACT = (
    "same_state_bidirectional_injected_context_hinge_on_detached_used_action_"
    "ranking_violation_v1"
)


def validate_mode(mode):
    if not isinstance(mode, str) or mode not in MODES:
        raise ValueError(f"Unsupported CF ablation {mode!r}; expected one of {MODES}.")
    return mode


def verification_code(kind):
    """Encode an already audited corrective record, without guessing its goal."""
    codes = {
        "target_lift": CorrectiveVerification.TARGET_LIFT,
        "counterfactual_goal": CorrectiveVerification.COUNTERFACTUAL_GOAL,
    }
    if kind not in codes:
        raise ValueError(f"Missing/unknown corrective verification_kind: {kind!r}.")
    return int(codes[kind])


def checkpoint_mode(metadata):
    """Old V9.30 checkpoints lack both fields and mean the unmasked control."""
    if "eraf_cf_ablation" not in metadata:
        if metadata.get("eraf_cf_ablation_contract") is not None:
            raise ValueError("CF ablation contract has no recorded mode.")
        return "none"
    mode = validate_mode(metadata["eraf_cf_ablation"])
    if metadata.get("eraf_cf_ablation_contract") != CONTRACT:
        raise ValueError("CF ablation checkpoint contract mismatch.")
    return mode


def causal_ranking_per_sample(margin, correct_error, wrong_error, *, positive_only):
    """Return a causal margin with exactly one prediction side trainable.

    Historical ranking raises the wrong-language error around a detached
    correct-language anchor. Positive-only ranking instead lowers the correct
    error around a detached wrong-language threshold, avoiding destructive
    gradients through a shared wrong-language interface.
    """

    import torch

    if correct_error.shape != wrong_error.shape:
        raise ValueError("Correct/wrong ranking errors must share shape [B].")
    margin = torch.as_tensor(
        margin,
        device=correct_error.device,
        dtype=correct_error.dtype,
    )
    if positive_only:
        return torch.relu(margin + correct_error - wrong_error.detach())
    return torch.relu(margin + correct_error.detach() - wrong_error)


def routed_causal_ranking_per_sample(
    margin,
    correct_error,
    wrong_error,
    *,
    objective,
    corrective,
):
    """Route the one-sided ranking gradient without changing sample validity.

    V9.33 protects only closed-loop corrective rows and preserves the historical
    negative-side gradient elsewhere. V9.34 removes that remaining conflict:
    every semantic source/target pair lowers its correct-language action error,
    and the wrong-language error is a detached threshold. The caller still owns
    semantic validity and target-lift numerator masks.
    """

    import torch

    corrective = torch.as_tensor(corrective, device=correct_error.device).bool()
    if corrective.shape != correct_error.shape:
        raise ValueError("Corrective ranking mask must share error shape [B].")
    if objective in {34, 35, 36}:
        return causal_ranking_per_sample(
            margin,
            correct_error,
            wrong_error,
            positive_only=True,
        )
    legacy = causal_ranking_per_sample(
        margin,
        correct_error,
        wrong_error,
        positive_only=False,
    )
    if objective != 33:
        return legacy
    positive = causal_ranking_per_sample(
        margin,
        correct_error,
        wrong_error,
        positive_only=True,
    )
    return torch.where(corrective, positive, legacy)


def paired_semantic_contrast_per_sample(
    injected_context,
    correct_language,
    correct_language_mask,
    wrong_language,
    wrong_language_mask,
    *,
    margin,
):
    """Contrast the deployed ERAF context against frozen language anchors.

    Only ``injected_context`` remains trainable. Correct/wrong text embeddings
    are detached anchors, so the loss cannot update either language encoding or
    the wrong action path. Masks use the FastWAM convention ``True == valid``.
    """

    import torch
    import torch.nn.functional as F

    if injected_context.ndim != 3 or injected_context.shape[1] == 0:
        raise ValueError("Injected ERAF context must be non-empty [B,T,D].")
    if (
        correct_language.ndim != 3
        or wrong_language.ndim != 3
        or correct_language.shape[0] != injected_context.shape[0]
        or wrong_language.shape[0] != injected_context.shape[0]
        or correct_language.shape[2] != injected_context.shape[2]
        or wrong_language.shape[2] != injected_context.shape[2]
    ):
        raise ValueError("Injected context and language anchors must share [B,D].")
    for name, language, mask in (
        ("correct", correct_language, correct_language_mask),
        ("wrong", wrong_language, wrong_language_mask),
    ):
        if mask.shape != language.shape[:2] or mask.dtype != torch.bool:
            raise ValueError(f"{name} language mask must be boolean [B,L].")
        if not bool(mask.any(dim=1).all()):
            raise ValueError(f"{name} language anchor cannot be empty.")
    margin = float(margin)
    if not 0.0 < margin < 2.0:
        raise ValueError("Paired semantic contrast margin must lie in (0,2).")

    def masked_mean(language, mask):
        weights = mask.to(device=language.device, dtype=torch.float32).unsqueeze(-1)
        return (
            (language.float() * weights).sum(dim=1)
            / weights.sum(dim=1).clamp_min(1.0)
        )

    deployed = F.normalize(injected_context.float().mean(dim=1), dim=-1)
    correct_anchor = F.normalize(
        masked_mean(correct_language, correct_language_mask).detach(), dim=-1
    )
    wrong_anchor = F.normalize(
        masked_mean(wrong_language, wrong_language_mask).detach(), dim=-1
    )
    correct_similarity = (deployed * correct_anchor).sum(dim=-1)
    wrong_similarity = (deployed * wrong_anchor).sum(dim=-1)
    loss = torch.relu(
        deployed.new_tensor(margin) + wrong_similarity - correct_similarity
    )
    return loss, correct_similarity, wrong_similarity


def action_violation_semantic_mask(
    semantic_valid,
    ranking_multiplier,
    ranking_per_sample,
):
    """Select detached, routed action-ranking violations for V9.36."""

    import torch

    semantic_valid = torch.as_tensor(
        semantic_valid, device=ranking_per_sample.device
    ).bool()
    ranking_multiplier = torch.as_tensor(
        ranking_multiplier, device=ranking_per_sample.device
    ).bool()
    if (
        semantic_valid.shape != ranking_per_sample.shape
        or ranking_multiplier.shape != ranking_per_sample.shape
    ):
        raise ValueError(
            "Semantic validity, ranking route and ranking loss must share [B]."
        )
    return (
        semantic_valid
        & ranking_multiplier
        & (ranking_per_sample.detach() > 0)
    )


def loss_multipliers(mode, corrective, verification_kind=None):
    """Return boolean numerator multipliers and audited kind IDs, all [B].

    The caller must retain its ORIGINAL validity masks/mean denominators and
    execute all the original forwards/noise draws. A zero contribution is not
    a promise of a zero optimizer update (momentum/weight decay still apply).
    """
    import torch

    validate_mode(mode)
    corrective = torch.as_tensor(corrective).bool()
    if corrective.ndim != 1:
        raise ValueError("CF corrective mask must have shape [B].")
    if verification_kind is None:
        if mode != "none":
            raise ValueError("CF ablations require pgc_corrective_verification_kind.")
        # Legacy control batches remain supported. -1 means unknown for logs,
        # never an inferred full-goal or target-lift label.
        kind = torch.full_like(corrective, -1, dtype=torch.long)
    else:
        kind = torch.as_tensor(verification_kind, device=corrective.device)
        if kind.shape != corrective.shape or kind.dtype not in (
            torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8
        ):
            raise ValueError("pgc_corrective_verification_kind must be integer [B].")
        known = (kind == 0) | (kind == 1) | (kind == 2)
        if not bool(known.all()) or not torch.equal(kind != 0, corrective):
            raise ValueError("Corrective flag/verification_kind mismatch.")
    action = torch.ones_like(corrective)
    ranking = torch.ones_like(corrective)
    if mode == "mask_lift_corrective":
        action = kind != CorrectiveVerification.TARGET_LIFT
        ranking = action
    elif mode == "mask_corrective_ranking":
        ranking = ~corrective
    elif mode == "mask_lift_ranking":
        # Partial target-lift verification supports action acquisition but is
        # not evidence that the full source/target goal ordering is correct.
        # Full counterfactual-goal rows retain both action and ranking losses.
        ranking = kind != CorrectiveVerification.TARGET_LIFT
    return action, ranking, kind
