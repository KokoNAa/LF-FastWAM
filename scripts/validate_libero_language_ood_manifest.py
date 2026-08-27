#!/usr/bin/env python3
"""Validate a frozen LIBERO-Object sentence-level language-OOD manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.libero.language_ood import (  # noqa: E402
    PARAPHRASE_VARIANTS,
    PROVENANCE_FORMAT,
    load_training_instructions,
    normalize_instruction,
    provenance_path_for_manifest,
    sha256_file,
    validate_language_ood_record,
)
from experiments.libero.language_interventions import (  # noqa: E402
    load_language_intervention_manifest,
)
from scripts.prepare_libero_object_language_ood import (  # noqa: E402
    DEFAULT_DATASET_DIRS,
)


def validate_manifest(
    manifest_path: Path,
    *,
    dataset_dirs: list[Path],
) -> dict:
    from libero.libero import benchmark

    manifest_path = manifest_path.expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Language-OOD manifest not found: {manifest_path}")
    training_instructions, current_corpus = load_training_instructions(dataset_dirs)
    records = load_language_intervention_manifest(manifest_path)

    suite = benchmark.get_benchmark_dict()["libero_object"]()
    expected_task_count = int(suite.n_tasks)
    if expected_task_count != 10 or len(records) != expected_task_count:
        raise ValueError(
            "Expected exactly 10 records matching LIBERO-Object, got "
            f"benchmark={expected_task_count} manifest={len(records)}."
        )

    seen_task_ids: set[int] = set()
    seen_pair_ids: set[str] = set()
    seen_paraphrases: set[str] = set()
    for record in records:
        validate_language_ood_record(
            record,
            training_instructions=training_instructions,
        )
        task_id = int(record["task_id"])
        if task_id in seen_task_ids:
            raise ValueError(f"Duplicate language-OOD task_id: {task_id}.")
        seen_task_ids.add(task_id)
        pair_id = str(record.get("pair_id", "")).strip()
        if not pair_id or pair_id in seen_pair_ids:
            raise ValueError(f"Missing or duplicate language-OOD pair_id: {pair_id!r}.")
        seen_pair_ids.add(pair_id)

        canonical = str(suite.get_task(task_id).language).strip()
        if normalize_instruction(
            record["correct_instruction"]
        ) != normalize_instruction(canonical):
            raise ValueError(
                f"Task {task_id} canonical instruction drift: "
                f"{record['correct_instruction']!r} != {canonical!r}."
            )
        shuffled = normalize_instruction(record.get("shuffled_instruction", ""))
        if not shuffled or shuffled == normalize_instruction(canonical):
            raise ValueError(
                f"Task {task_id} requires a non-canonical shuffled negative control."
            )
        for variant in PARAPHRASE_VARIANTS:
            normalized = normalize_instruction(record["paraphrases"][variant])
            if normalized in seen_paraphrases:
                raise ValueError(
                    f"Duplicate paraphrase across manifest: {normalized!r}."
                )
            seen_paraphrases.add(normalized)

    if seen_task_ids != set(range(expected_task_count)):
        raise ValueError(f"Manifest task IDs are incomplete: {sorted(seen_task_ids)}.")

    provenance_path = provenance_path_for_manifest(manifest_path)
    if not provenance_path.is_file():
        raise FileNotFoundError(
            f"Language-OOD provenance sidecar not found: {provenance_path}"
        )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("format") != PROVENANCE_FORMAT:
        raise ValueError("Language-OOD provenance format mismatch.")
    manifest_sha256 = sha256_file(manifest_path)
    if provenance.get("manifest_sha256") != manifest_sha256:
        raise ValueError("Language-OOD manifest SHA256 does not match its provenance.")
    recorded_corpus = provenance.get("training_corpus")
    if not isinstance(recorded_corpus, list):
        raise ValueError("Language-OOD provenance has no training-corpus audit.")
    recorded_hashes = {
        str(item.get("tasks_path")): str(item.get("tasks_sha256"))
        for item in recorded_corpus
    }
    current_hashes = {
        str(item["tasks_path"]): str(item["tasks_sha256"]) for item in current_corpus
    }
    if recorded_hashes != current_hashes:
        raise ValueError(
            "Policy-training tasks metadata differs from the frozen OOD provenance. "
            f"recorded={recorded_hashes} current={current_hashes}"
        )

    return {
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "records": len(records),
        "paraphrases": len(seen_paraphrases),
        "training_task_strings": len(training_instructions),
        "source_goal_unchanged": provenance.get("source_goal_unchanged") is True,
        "entity_nouns_preserved": provenance.get("entity_nouns_preserved") is True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--training-dataset-dir",
        type=Path,
        action="append",
        dest="training_dataset_dirs",
    )
    args = parser.parse_args()
    dataset_dirs = [
        path if path.is_absolute() else PROJECT_ROOT / path
        for path in (
            args.training_dataset_dirs
            if args.training_dataset_dirs is not None
            else [Path(value) for value in DEFAULT_DATASET_DIRS]
        )
    ]
    result = validate_manifest(args.manifest, dataset_dirs=dataset_dirs)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(
        "PASS: 10 canonical tasks, 30 semantically reviewed entity-preserving "
        "paraphrases, exact-string absent from frozen FastWAM policy training data."
    )


if __name__ == "__main__":
    main()
