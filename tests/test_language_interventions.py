import tempfile
import unittest
from pathlib import Path

from experiments.libero.language_interventions import (
    load_language_intervention_manifest,
    select_language_intervention_record,
    validate_counterfactual_problem,
)


def _problem(goal, *, objects=None, problem_name="LIBERO_Floor_Manipulation"):
    return {
        "problem_name": problem_name,
        "objects": objects
        or {
            "alphabet_soup": ["alphabet_soup_1"],
            "cream_cheese": ["cream_cheese_1"],
            "basket": ["basket_1"],
        },
        "fixtures": {"floor": ["floor"]},
        "regions": {"basket_1_contain_region": {"target": "basket_1"}},
        "initial_state": [],
        "goal_state": goal,
    }


class LanguageInterventionManifestTest(unittest.TestCase):
    def test_explicit_selector_does_not_fall_back_to_matching_name(self):
        records = [
            {
                "task_suite_name": "libero_object",
                "task_id": 1,
                "task_name": "task zero",
                "correct_instruction": "task zero",
            }
        ]
        with self.assertRaisesRegex(ValueError, "found 0"):
            select_language_intervention_record(
                records,
                suite_name="libero_object",
                task_id=0,
                task_description="task zero",
            )

    def test_manifest_loader_tracks_line_numbers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.jsonl"
            path.write_text('\n{"pair_id": "p0"}\n', encoding="utf-8")
            records = load_language_intervention_manifest(path)
        self.assertEqual(records[0]["_line_number"], 2)


class CounterfactualGoalTest(unittest.TestCase):
    def test_accepts_alternate_goal_whose_entities_exist_in_source_scene(self):
        source = _problem([["in", "alphabet_soup_1", "basket_1_contain_region"]])
        alternate = _problem([["in", "cream_cheese_1", "basket_1_contain_region"]])
        goal = validate_counterfactual_problem(source, alternate)
        self.assertEqual(
            goal,
            [["in", "cream_cheese_1", "basket_1_contain_region"]],
        )

    def test_rejects_missing_counterfactual_object(self):
        source = _problem([["in", "alphabet_soup_1", "basket_1_contain_region"]])
        alternate = _problem(
            [["in", "milk_1", "basket_1_contain_region"]],
            objects={
                "milk": ["milk_1"],
                "basket": ["basket_1"],
            },
        )
        with self.assertRaisesRegex(ValueError, "milk_1"):
            validate_counterfactual_problem(source, alternate)

    def test_rejects_different_environment_class(self):
        source = _problem([["in", "alphabet_soup_1", "basket_1_contain_region"]])
        alternate = _problem(
            [["in", "cream_cheese_1", "basket_1_contain_region"]],
            problem_name="LIBERO_Kitchen_Tabletop_Manipulation",
        )
        with self.assertRaisesRegex(ValueError, "different LIBERO environment"):
            validate_counterfactual_problem(source, alternate)


if __name__ == "__main__":
    unittest.main()
