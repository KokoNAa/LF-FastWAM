"""Audited counterfactual-instruction manifest helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def stable_instruction_id(instruction: str) -> int:
    """Return a deterministic signed-int64-safe ID for a task instruction."""
    normalized = str(instruction).strip().casefold()
    if not normalized:
        raise ValueError("Cannot build a task ID from an empty instruction.")
    return int.from_bytes(
        hashlib.sha256(normalized.encode("utf-8")).digest()[:8],
        byteorder="big",
        signed=False,
    ) & ((1 << 63) - 1)


def load_counterfactual_instruction_map(path: str | os.PathLike) -> dict[str, str]:
    """Load an executable source-to-negative instruction map from JSONL."""
    manifest_path = Path(path).expanduser()
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Counterfactual training manifest not found: {manifest_path}"
        )
    mapping: dict[str, str] = {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            source = str(
                record.get("correct_instruction") or record.get("task_name") or ""
            ).strip()
            negative = str(record.get("counterfactual_instruction") or "").strip()
            if not source or not negative:
                raise ValueError(
                    f"Manifest line {line_number} requires positive and negative instructions."
                )
            if source.casefold() == negative.casefold():
                raise ValueError(
                    f"Manifest line {line_number} uses the positive instruction as its negative."
                )
            if record.get("counterfactual_is_executable") is not True:
                raise ValueError(
                    f"Manifest line {line_number} is not marked executable."
                )
            key = source.casefold()
            if key in mapping and mapping[key] != negative:
                raise ValueError(
                    f"Manifest defines conflicting negatives for {source!r}."
                )
            mapping[key] = negative
    if not mapping:
        raise ValueError(f"Counterfactual manifest is empty: {manifest_path}")
    return mapping
