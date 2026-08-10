"""Compatibility helpers for loading trusted LIBERO benchmark artifacts."""

from pathlib import Path
from typing import Any

import torch


def load_libero_task_init_states(
    task_suite: Any,
    task_id: int,
    init_states_root: str | Path,
) -> Any:
    """Load a trusted LIBERO ``.pruned_init`` file.

    LIBERO 1.4 calls ``torch.load(path)`` without specifying ``weights_only``.
    PyTorch 2.6 changed that default to ``True``, which rejects LIBERO's
    NumPy-backed init-state files. The path is constrained to the configured
    LIBERO init-state root before legacy pickle loading is enabled.
    """
    trusted_root = Path(init_states_root).expanduser().resolve()
    task = task_suite.get_task(int(task_id))
    init_states_path = (
        trusted_root / task.problem_folder / task.init_states_file
    ).resolve()

    if not init_states_path.is_relative_to(trusted_root):
        raise ValueError(
            "Resolved LIBERO init-state path escapes the configured root: "
            f"{init_states_path} (root={trusted_root})"
        )
    if init_states_path.suffix != ".pruned_init":
        raise ValueError(
            "Refusing to deserialize an unexpected LIBERO init-state file: "
            f"{init_states_path}"
        )
    if not init_states_path.is_file():
        raise FileNotFoundError(
            f"LIBERO init-state file not found: {init_states_path}"
        )

    try:
        return torch.load(
            init_states_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        # Compatibility with torch versions predating the weights_only kwarg.
        return torch.load(init_states_path, map_location="cpu")
