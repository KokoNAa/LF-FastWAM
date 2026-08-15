from math import ceil


def optimizer_step_to_sampler_position(
    *,
    dataset_size: int,
    batch_size: int,
    num_processes: int,
    gradient_accumulation_steps: int,
    optimizer_step: int,
) -> dict[str, int]:
    """Map an absolute optimizer step to deterministic sampler progress.

    Weight-only continuation cannot restore an Accelerator dataloader state.
    This helper reconstructs the epoch and local micro-batch offset using the
    same global-batch arithmetic as ``Wan22Trainer``. The resumable sampler
    then skips the already-consumed global samples before the first continued
    batch is yielded.
    """
    values = {
        "dataset_size": dataset_size,
        "batch_size": batch_size,
        "num_processes": num_processes,
        "gradient_accumulation_steps": gradient_accumulation_steps,
    }
    for name, value in values.items():
        if int(value) <= 0:
            raise ValueError(f"`{name}` must be positive, got {value}.")
    if int(optimizer_step) < 0:
        raise ValueError(
            f"`optimizer_step` must be non-negative, got {optimizer_step}."
        )

    global_batch_size = int(batch_size) * int(num_processes)
    micro_steps_per_epoch = max(
        ceil(int(dataset_size) / global_batch_size),
        1,
    )
    optimizer_steps_per_epoch = max(
        ceil(micro_steps_per_epoch / int(gradient_accumulation_steps)),
        1,
    )
    epoch, optimizer_step_in_epoch = divmod(
        int(optimizer_step), optimizer_steps_per_epoch
    )
    batch_in_epoch = min(
        optimizer_step_in_epoch * int(gradient_accumulation_steps),
        micro_steps_per_epoch,
    )
    return {
        "epoch": epoch,
        "optimizer_step_in_epoch": optimizer_step_in_epoch,
        "batch_in_epoch": batch_in_epoch,
        "micro_steps_per_epoch": micro_steps_per_epoch,
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
    }
