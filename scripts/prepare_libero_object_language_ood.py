#!/usr/bin/env python3
"""Build an audited sentence-level OOD manifest for LIBERO-Object."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.libero.language_ood import (  # noqa: E402
    MANIFEST_FORMAT,
    PARAPHRASE_VARIANTS,
    PROVENANCE_FORMAT,
    build_paraphrases,
    extract_object_phrase,
    load_training_instructions,
    normalize_instruction,
    provenance_path_for_manifest,
    sha256_file,
    validate_language_ood_record,
)


DEFAULT_DATASET_DIRS = (
    "data/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot",
    "data/libero_mujoco3.3.2/libero_object_no_noops_lerobot",
    "data/libero_mujoco3.3.2/libero_goal_no_noops_lerobot",
    "data/libero_mujoco3.3.2/libero_10_no_noops_lerobot",
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def build_manifest_records(
    canonical_instructions: list[str],
    shuffled_instructions: dict[int, str],
    *,
    training_instructions: set[str],
) -> list[dict]:
    records: list[dict] = []
    globally_seen_paraphrases: set[str] = set()
    for task_id, canonical in enumerate(canonical_instructions):
        paraphrases = build_paraphrases(canonical)
        record = {
            "format": MANIFEST_FORMAT,
            "pair_id": f"libero_object_language_ood_{task_id:02d}",
            "task_suite_name": "libero_object",
            "task_id": task_id,
            "task_name": canonical,
            "correct_instruction": canonical,
            "shuffled_instruction": shuffled_instructions[task_id],
            "paraphrases": paraphrases,
            "preserved_entity_phrase": extract_object_phrase(canonical),
            "semantic_equivalence_reviewed": True,
            "policy_training_exact_match": {
                variant: normalize_instruction(instruction) in training_instructions
                for variant, instruction in paraphrases.items()
            },
            "notes": (
                "Only the policy task description changes. The source scene, "
                "initial state, action interface, and source success predicate "
                "remain unchanged. Entity nouns are preserved in OOD v1."
            ),
        }
        validate_language_ood_record(
            record,
            training_instructions=training_instructions,
        )
        for variant, instruction in paraphrases.items():
            normalized = normalize_instruction(instruction)
            if normalized in globally_seen_paraphrases:
                raise ValueError(
                    f"Duplicate global paraphrase ({variant}): {instruction!r}."
                )
            globally_seen_paraphrases.add(normalized)
        records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/eval/libero_object_language_ood_v1.jsonl"),
    )
    parser.add_argument(
        "--training-dataset-dir",
        type=Path,
        action="append",
        dest="training_dataset_dirs",
        help=(
            "Policy-training dataset directory. Repeat for every native LIBERO "
            "dataset. Defaults to the four released-checkpoint datasets."
        ),
    )
    args = parser.parse_args()

    from libero.libero import benchmark

    from scripts.prepare_libero_object_interventions import build_manifest

    dataset_dirs = [
        path if path.is_absolute() else PROJECT_ROOT / path
        for path in (
            args.training_dataset_dirs
            if args.training_dataset_dirs is not None
            else [Path(value) for value in DEFAULT_DATASET_DIRS]
        )
    ]
    training_instructions, training_corpus = load_training_instructions(dataset_dirs)

    suite = benchmark.get_benchmark_dict()["libero_object"]()
    canonical_instructions = [
        str(suite.get_task(task_id).language).strip()
        for task_id in range(int(suite.n_tasks))
    ]
    if len(canonical_instructions) != 10:
        raise ValueError(
            "Language-OOD v1 expects exactly 10 LIBERO-Object tasks, got "
            f"{len(canonical_instructions)}."
        )

    intervention_records = build_manifest("libero_object")
    shuffled_instructions = {
        int(record["task_id"]): str(record["shuffled_instruction"]).strip()
        for record in intervention_records
    }
    if set(shuffled_instructions) != set(range(len(canonical_instructions))):
        raise ValueError("Could not build one same-scene shuffled control per task.")

    records = build_manifest_records(
        canonical_instructions,
        shuffled_instructions,
        training_instructions=training_instructions,
    )
    output = args.output.expanduser().resolve()
    _write_jsonl(output, records)

    provenance = {
        "format": PROVENANCE_FORMAT,
        "manifest": str(output),
        "manifest_sha256": sha256_file(output),
        "task_suite_name": "libero_object",
        "task_count": len(records),
        "paraphrase_variants": list(PARAPHRASE_VARIANTS),
        "paraphrase_count": len(records) * len(PARAPHRASE_VARIANTS),
        "training_ood_definition": (
            "The normalized complete task description is absent from every "
            "listed FastWAM policy-training tasks.jsonl. This does not claim "
            "absence from the pretrained Wan text encoder corpus."
        ),
        "training_corpus": training_corpus,
        "source_goal_unchanged": True,
        "prompt_wrapper_unchanged": True,
        "entity_nouns_preserved": True,
    }
    provenance_path = provenance_path_for_manifest(output)
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {len(records)} LIBERO-Object records: {output}")
    print(
        f"Wrote {len(records) * len(PARAPHRASE_VARIANTS)} policy-training-unseen "
        f"paraphrases and provenance: {provenance_path}"
    )
    print(f"MANIFEST_SHA256={provenance['manifest_sha256']}")


if __name__ == "__main__":
    main()
