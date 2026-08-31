"""Training-only CF loss masks; never filter samples or select an action path."""

from enum import IntEnum


class CorrectiveVerification(IntEnum):
    NONE = 0
    TARGET_LIFT = 1
    COUNTERFACTUAL_GOAL = 2


MODES = ("none", "mask_lift_corrective", "mask_corrective_ranking")
CONTRACT = "cf_loss_numerator_masks_v1_same_samples_forwards_and_denominators"


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
    return action, ranking, kind
