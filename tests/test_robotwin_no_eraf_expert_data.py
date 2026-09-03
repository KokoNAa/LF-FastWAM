import unittest
from pathlib import Path

from scripts.collect_pgc_robotwin_pairs import collection_contract
from scripts.prepare_pgc_robotwin_no_eraf_expert_data import build_plan


class RoboTwinNoERAFExpertDataTest(unittest.TestCase):
    def test_collection_profiles_bind_training_roles(self):
        offline = collection_contract("no_eraf_historical", "native")
        historical = collection_contract(
            "no_eraf_historical", "counterfactual"
        )
        strict = collection_contract("no_eraf_strict", "counterfactual")
        strict_audit = collection_contract("no_eraf_strict", "native")

        self.assertEqual(offline["artifact_role"], "offline_native")
        self.assertEqual(historical["artifact_role"], "historical_cf")
        self.assertEqual(strict["artifact_role"], "strict_cf")
        self.assertEqual(strict_audit["allowed_training_stages"], [])
        for contract in (offline, historical, strict):
            self.assertIn("no_eraf", contract["allowed_training_stages"])
            self.assertIn("grounding", contract["forbidden_training_stages"])

    def test_historical_plan_has_paired_five_task_capture(self):
        plan = build_plan(
            robotwin_root=Path("/repo/third_party/RoboTwin"),
            work_root=Path("/data/pgc_robotwin_no_eraf_v1/formal"),
            profile="historical",
            task_config="demo_clean",
            episodes=30,
            start_seed=14_000_000,
            fps=10,
            video_codec="h264",
            skip_collection=False,
            python="python",
        )
        self.assertEqual(plan["pair_count"], 5)
        self.assertEqual(plan["raw_trajectory_count"], 300)
        self.assertIn("--collection-profile", plan["commands"]["collect"])
        self.assertIn("no_eraf_historical", plan["commands"]["collect"])
        self.assertEqual(
            plan["profile_root"],
            "/data/pgc_robotwin_no_eraf_v1/formal/expert/historical/demo_clean",
        )

    def test_strict_plan_never_names_full_goal(self):
        plan = build_plan(
            robotwin_root=Path("/repo/third_party/RoboTwin"),
            work_root=Path("/data/pgc_robotwin_no_eraf_v1/smoke"),
            profile="strict",
            task_config="demo_clean",
            episodes=1,
            start_seed=34_000_000,
            fps=10,
            video_codec="h264",
            skip_collection=False,
            python="python",
        )
        self.assertEqual(plan["collector_profile"], "no_eraf_strict")
        self.assertNotIn("full_goal", " ".join(plan["commands"]["collect"]))

    def test_unknown_collection_profile_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            collection_contract("full_goal", "counterfactual")


if __name__ == "__main__":
    unittest.main()
