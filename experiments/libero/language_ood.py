"""Contracts for sentence-level LIBERO language-OOD evaluation."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping


MANIFEST_FORMAT = "libero_language_ood_v1"
PROVENANCE_FORMAT = "libero_language_ood_provenance_v1"
PARAPHRASE_VARIANTS = ("near", "sequence", "goal")
OBJECT_CANONICAL_PATTERN = re.compile(
    r"^pick up the (?P<object>.+?) and place it in the basket[.!]?$",
    flags=re.IGNORECASE,
)


def normalize_instruction(value: Any) -> str:
    """Normalize an instruction for exact policy-training-corpus audits."""
    return " ".join(str(value).strip().casefold().split())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_object_phrase(canonical_instruction: str) -> str:
    match = OBJECT_CANONICAL_PATTERN.fullmatch(canonical_instruction.strip())
    if match is None:
        raise ValueError(
            "LIBERO-Object canonical instruction does not match the audited "
            f"single-object template: {canonical_instruction!r}."
        )
    return match.group("object").strip()


def build_paraphrases(canonical_instruction: str) -> dict[str, str]:
    """Create controlled equivalent descriptions while preserving entity nouns."""
    object_phrase = extract_object_phrase(canonical_instruction)
    return {
        "near": f"put the {object_phrase} into the basket",
        "sequence": (
            f"grasp the {object_phrase}, carry it to the basket, "
            "and release it inside"
        ),
        "goal": f"make sure the {object_phrase} ends up inside the basket",
    }


def load_training_instructions(
    dataset_dirs: Iterable[Path],
) -> tuple[set[str], list[dict[str, Any]]]:
    """Read the authoritative policy-training instructions and corpus hashes."""
    normalized: set[str] = set()
    audits: list[dict[str, Any]] = []
    for raw_dataset_dir in dataset_dirs:
        dataset_dir = Path(raw_dataset_dir).expanduser().resolve()
        tasks_path = dataset_dir / "meta" / "tasks.jsonl"
        if not tasks_path.is_file():
            raise FileNotFoundError(f"Training tasks metadata not found: {tasks_path}")
        task_count = 0
        with tasks_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if (
                    not isinstance(record, dict)
                    or not str(record.get("task", "")).strip()
                ):
                    raise ValueError(
                        f"Training task metadata requires a non-empty `task`: "
                        f"{tasks_path}:{line_number}"
                    )
                normalized.add(normalize_instruction(record["task"]))
                task_count += 1
        audits.append(
            {
                "dataset_dir": str(dataset_dir),
                "tasks_path": str(tasks_path),
                "tasks_sha256": sha256_file(tasks_path),
                "task_rows": task_count,
            }
        )
    return normalized, audits


def resolve_paraphrase_instruction(record: Mapping[str, Any], variant: str) -> str:
    normalized_variant = str(variant).strip().casefold()
    if normalized_variant not in PARAPHRASE_VARIANTS:
        raise ValueError(
            "EVALUATION.language_ood_variant must be one of "
            f"{list(PARAPHRASE_VARIANTS)}, got {variant!r}."
        )
    paraphrases = record.get("paraphrases")
    if not isinstance(paraphrases, Mapping):
        raise ValueError("Language-OOD manifest record has no `paraphrases` mapping.")
    instruction = str(paraphrases.get(normalized_variant, "")).strip()
    if not instruction:
        raise ValueError(
            f"Language-OOD manifest record has no {normalized_variant!r} paraphrase."
        )
    return instruction


def validate_language_ood_record(
    record: Mapping[str, Any],
    *,
    training_instructions: set[str] | None = None,
) -> None:
    """Validate the local semantic-preservation and unseen-string contract."""
    if record.get("format") != MANIFEST_FORMAT:
        raise ValueError(
            f"Expected language-OOD format {MANIFEST_FORMAT!r}, "
            f"got {record.get('format')!r}."
        )
    if str(record.get("task_suite_name", "")) != "libero_object":
        raise ValueError("Language-OOD v1 is restricted to `libero_object`.")
    canonical = str(record.get("correct_instruction", "")).strip()
    task_name = str(record.get("task_name", "")).strip()
    if not canonical or normalize_instruction(canonical) != normalize_instruction(
        task_name
    ):
        raise ValueError("`correct_instruction` must exactly match `task_name`.")
    object_phrase = extract_object_phrase(canonical)
    if normalize_instruction(
        record.get("preserved_entity_phrase", "")
    ) != normalize_instruction(object_phrase):
        raise ValueError("`preserved_entity_phrase` does not match the canonical task.")
    if record.get("semantic_equivalence_reviewed") is not True:
        raise ValueError("Language-OOD records must be marked semantically reviewed.")
    exact_match_audit = record.get("policy_training_exact_match")
    if not isinstance(exact_match_audit, Mapping):
        raise ValueError(
            "Language-OOD record has no policy-training exact-match audit."
        )

    seen_variants: set[str] = set()
    for variant in PARAPHRASE_VARIANTS:
        instruction = resolve_paraphrase_instruction(record, variant)
        normalized = normalize_instruction(instruction)
        if normalized == normalize_instruction(canonical):
            raise ValueError(f"{variant} paraphrase equals the canonical instruction.")
        if normalized in seen_variants:
            raise ValueError(f"Duplicate paraphrase in one record: {instruction!r}.")
        seen_variants.add(normalized)
        if normalize_instruction(object_phrase) not in normalized:
            raise ValueError(
                f"{variant} paraphrase does not preserve object phrase "
                f"{object_phrase!r}: {instruction!r}."
            )
        if re.search(r"\bbasket\b", normalized) is None:
            raise ValueError(
                f"{variant} paraphrase does not preserve the basket entity noun."
            )
        if len(instruction.split()) > 64:
            raise ValueError(f"{variant} paraphrase exceeds the v1 length bound.")
        if training_instructions is not None and normalized in training_instructions:
            raise ValueError(
                f"{variant} paraphrase appears verbatim in policy training data: "
                f"{instruction!r}."
            )
        if exact_match_audit.get(variant) is not False:
            raise ValueError(
                f"{variant} paraphrase is not frozen as policy-training unseen."
            )


def provenance_path_for_manifest(manifest_path: Path) -> Path:
    return manifest_path.with_suffix(".provenance.json")
