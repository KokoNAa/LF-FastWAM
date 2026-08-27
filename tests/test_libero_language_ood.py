import json
import tempfile
import unittest
from pathlib import Path

from experiments.libero.language_ood import (
    MANIFEST_FORMAT,
    PARAPHRASE_VARIANTS,
    build_paraphrases,
    load_training_instructions,
    normalize_instruction,
    resolve_paraphrase_instruction,
    validate_language_ood_record,
)
from scripts.prepare_libero_object_language_ood import build_manifest_records
from scripts.summarize_libero_language_ood import summarize


class LanguageOODManifestTest(unittest.TestCase):
    def _record(self):
        canonical = "pick up the alphabet soup and place it in the basket"
        return {
            "format": MANIFEST_FORMAT,
            "pair_id": "libero_object_language_ood_00",
            "task_suite_name": "libero_object",
            "task_id": 0,
            "task_name": canonical,
            "correct_instruction": canonical,
            "shuffled_instruction": (
                "pick up the cream cheese and place it in the basket"
            ),
            "paraphrases": build_paraphrases(canonical),
            "preserved_entity_phrase": "alphabet soup",
            "semantic_equivalence_reviewed": True,
            "policy_training_exact_match": {
                variant: False for variant in PARAPHRASE_VARIANTS
            },
        }

    def test_three_variants_preserve_entities_and_are_distinct(self):
        record = self._record()
        validate_language_ood_record(record, training_instructions=set())
        self.assertEqual(set(record["paraphrases"]), set(PARAPHRASE_VARIANTS))
        self.assertEqual(
            resolve_paraphrase_instruction(record, " Near "),
            "put the alphabet soup into the basket",
        )
        self.assertEqual(len(set(record["paraphrases"].values())), 3)

    def test_rejects_policy_training_exact_match(self):
        record = self._record()
        training = {normalize_instruction(record["paraphrases"]["near"])}
        with self.assertRaisesRegex(ValueError, "policy training data"):
            validate_language_ood_record(record, training_instructions=training)

    def test_training_metadata_scan_is_exact_and_hashed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = Path(tmpdir) / "dataset"
            tasks_path = dataset / "meta" / "tasks.jsonl"
            tasks_path.parent.mkdir(parents=True)
            tasks_path.write_text(
                json.dumps({"task_index": 0, "task": "Canonical Task"}) + "\n",
                encoding="utf-8",
            )
            instructions, audits = load_training_instructions([dataset])
        self.assertEqual(instructions, {"canonical task"})
        self.assertEqual(audits[0]["task_rows"], 1)
        self.assertEqual(len(audits[0]["tasks_sha256"]), 64)

    def test_build_manifest_records_covers_ten_object_tasks(self):
        canonicals = [
            f"pick up the object {index} and place it in the basket"
            for index in range(10)
        ]
        shuffled = {
            index: canonicals[(index + 1) % len(canonicals)]
            for index in range(len(canonicals))
        }
        records = build_manifest_records(
            canonicals,
            shuffled,
            training_instructions={normalize_instruction(item) for item in canonicals},
        )
        self.assertEqual(len(records), 10)
        self.assertTrue(
            all(
                not any(record["policy_training_exact_match"].values())
                for record in records
            )
        )


class LanguageOODSummaryTest(unittest.TestCase):
    def _write_condition(self, root: Path, label: str, successes: int) -> None:
        condition = "paraphrase" if label.startswith("paraphrase_") else label
        variant = (
            label.removeprefix("paraphrase_") if condition == "paraphrase" else None
        )
        suite_dir = root / label / "libero_object"
        suite_dir.mkdir(parents=True)
        for task_id in range(10):
            payload = {
                "instruction_condition": condition,
                "success_predicate": "source",
                "successes": successes,
                "total_episodes": 5,
                "success_episodes": list(range(successes)),
                "task_description": f"canonical {task_id}",
                "policy_instruction": (
                    f"ood {task_id}"
                    if condition == "paraphrase"
                    else f"canonical {task_id}"
                ),
            }
            if condition == "paraphrase":
                payload.update(
                    {
                        "language_ood_variant": variant,
                        "language_ood_source_goal_unchanged": True,
                        "language_ood_policy_training_exact_match": False,
                    }
                )
            (suite_dir / f"gpu0_task{task_id}_results.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )

    def test_reports_drop_retention_and_paired_conditional_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_condition(root, "correct", 4)
            self._write_condition(root, "paraphrase_near", 3)
            report = summarize(root)
        near = report["conditions"]["paraphrase_near"]
        self.assertAlmostEqual(near["success_rate"], 0.6)
        self.assertAlmostEqual(near["absolute_drop_from_id"], -0.2)
        self.assertAlmostEqual(near["retention_vs_id"], 0.75)
        self.assertAlmostEqual(near["p_success_given_id_success"], 0.75)


class LanguageOODContractTest(unittest.TestCase):
    def test_evaluator_and_launcher_preserve_native_source_goal_contract(self):
        root = Path(__file__).resolve().parents[1]
        evaluator = (root / "experiments/libero/eval_libero_single.py").read_text(
            encoding="utf-8"
        )
        launcher = (
            root / "scripts/eval_fastwam_libero_object_language_ood.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'condition not in {"shuffled", "counterfactual", "paraphrase"}', evaluator
        )
        self.assertIn('instruction_condition == "paraphrase"', evaluator)
        self.assertIn('"language_ood_source_goal_unchanged": True', evaluator)
        self.assertIn("model.langforce_mvp.enabled=false", launcher)
        self.assertIn("model.transition_contract.enabled=false", launcher)
        self.assertIn("model.policy_guard.enabled=false", launcher)
        self.assertIn("model.lora.enabled=false", launcher)
        self.assertIn("EVALUATION.max_steps=400", launcher)


if __name__ == "__main__":
    unittest.main()
