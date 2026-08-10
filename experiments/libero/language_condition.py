"""Helpers for parsing LIBERO language-intervention conditions."""

from typing import Any


def normalize_instruction_condition(value: Any) -> str:
    """Normalize Hydra values used by instruction-condition overrides.

    Hydra parses an unquoted ``key=null`` override as Python ``None``. The LF
    evaluation CLI intentionally uses ``null`` as the name of the masked-
    language condition, so preserve that user-facing spelling here.
    """
    if value is None:
        return "null"
    return str(value).strip().lower()
