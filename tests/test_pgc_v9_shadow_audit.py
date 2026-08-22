import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiments.libero.eraf_shadow_audit import (
    ERAFOracleProvider,
    ERAFShadowAuditor,
    ERAFShadowContract,
    _clause_statuses,
    summarize_eraf_shadow_records,
    verify_shadow_action_integrity,
)
from fastwam.datasets.pgc_libero import (
    PGC_ENTITY_RELATION_FORMAT,
    PGC_ENTITY_RELATION_PREDICATES,
)


def _good_record():
    clause = {
        "full_mask_correct": True,
        "exclusive_mask_valid": True,
        "exclusive_mask_correct": True,
        "mask_overlap_iou": 0.0,
        "subject_overlap_fraction": 0.0,
        "reference_overlap_fraction": 0.0,
        "full_subject_margin": 0.5,
        "full_reference_margin": 0.5,
        "exclusive_subject_margin": 0.5,
        "exclusive_reference_margin": 0.5,
        "dataset_kind": "online_closed_loop",
        "predicate": "on",
        "task": "put object on basket",
    }
    return {
        "subject_top1_hits": [True, True],
        "reference_top1_hits": [True, True],
        "role_swap_correct": [True, True],
        "role_audit_clauses": [dict(clause), dict(clause)],
        "relation_targets": [2, 2],
        "relation_predictions": [2, 2],
        "goal_anchor_errors_m": [0.01, 0.02],
        "clause_exact": True,
        "clause_count": 2,
        "online_stage": "pregrasp",
    }


def _extended_clause(status="initial_search", oracle_partition="role_pass"):
    view = {
        "gt_visible": True,
        "predicted_visible": True,
        "visibility_correct": True,
        "top1_hit": True,
        "attention_mass": 0.5,
        "center_error_px": 1.0,
    }
    return {
        "predicate": "on",
        "status": status,
        "active_correct": True,
        "predicate_correct": True,
        "predicate_truth_correct": True,
        "phase_target": 0,
        "phase_prediction": 0,
        "phase_correct": True,
        "subject_top1_hit": True,
        "reference_top1_hit": True,
        "full_role_correct": True,
        "exclusive_role_valid": True,
        "exclusive_role_correct": True,
        "oracle_partition": oracle_partition,
        "subject_position_error_m": 0.01,
        "reference_position_error_m": 0.02,
        "goal_anchor_error_m": 0.03,
        "views": {
            camera: {"subject": dict(view), "reference": dict(view)}
            for camera in ("agentview", "robot0_eye_in_hand")
        },
    }


class _FakeModel:
    instances_to_ids = {
        "object_1": {"geom": [1]},
        "basket_1": {"geom": [2]},
    }
    site_names = []

    @staticmethod
    def body_name2id(name):
        return {"object_1": 0, "basket_1": 1}[name]

    @staticmethod
    def site_name2id(_name):
        raise KeyError


class _FakeInnerEnv:
    def __init__(self):
        self.parsed_problem = {
            "goal_state": [["on", "object_1", "basket_1"]],
            "regions": {},
            "objects": {"object": ["object_1"], "basket": ["basket_1"]},
            "fixtures": {},
        }
        self.model = _FakeModel()
        self.obj_body_id = {"object_1": 0, "basket_1": 1}
        self.sim = SimpleNamespace(
            model=self.model,
            data=SimpleNamespace(
                body_xpos=np.asarray(
                    [[0.0, 0.0, 0.2], [0.2, 0.0, 0.2]], dtype=np.float32
                )
            ),
        )
        self.objects_dict = {"object_1": object(), "basket_1": object()}
        self.robots = [SimpleNamespace(gripper=object())]

    def _check_success(self):
        return False

    def _check_grasp(self, **_kwargs):
        return False


class PGCERAFShadowAuditTest(unittest.TestCase):
    def _contract(self, root: Path) -> ERAFShadowContract:
        payload = {
            "format": PGC_ENTITY_RELATION_FORMAT,
            "privileged_supervision": "training_only",
            "deployment_inputs": "rgb_language_proprio",
            "camera_names": ["agentview", "robot0_eye_in_hand"],
            "predicate_vocabulary": list(PGC_ENTITY_RELATION_PREDICATES),
            "mask_size": [4, 8],
            "workspace_min": [-0.8, -0.8, 0.0],
            "workspace_max": [0.8, 0.8, 1.2],
            "max_clauses": 4,
        }
        (root / "index.json").write_text(json.dumps(payload), encoding="utf-8")
        return ERAFShadowContract.load(root)

    def test_shadow_action_requires_exact_base(self):
        action = torch.randn(4, 7)
        audit = verify_shadow_action_integrity(action, action.clone(), gate_mode="base")
        self.assertTrue(audit["exact"])
        self.assertEqual(audit["max_abs_error"], 0.0)
        with self.assertRaisesRegex(RuntimeError, "changed the deployed Base"):
            verify_shadow_action_integrity(
                action + 1.0e-4,
                action,
                gate_mode="base",
            )
        with self.assertRaisesRegex(RuntimeError, "gate_mode=base"):
            verify_shadow_action_integrity(action, action, gate_mode="guarded")

    def test_contract_and_online_same_state_record(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            contract = self._contract(Path(tmpdir))
        env = SimpleNamespace(env=_FakeInnerEnv())
        auditor = ERAFShadowAuditor(
            env=env,
            policy_instruction="put object on basket",
            instruction_condition="correct",
            contract=contract,
            counterfactual_metadata=None,
            all_entity_role_gate=True,
        )
        segmentation = np.zeros((4, 4, 1), dtype=np.int32)
        segmentation[:2, :2, 0] = 1
        segmentation[2:, 2:, 0] = 2
        obs = {
            "agentview_segmentation_element": segmentation,
            "robot0_eye_in_hand_segmentation_element": segmentation,
        }

        oracle = ERAFOracleProvider(
            env=env,
            policy_instruction="put object on basket",
            instruction_condition="correct",
            contract=contract,
            counterfactual_metadata=None,
        ).policy_input(obs=obs, episode_idx=0)
        self.assertEqual(oracle["clause_valid"].tolist(), [True, False, False, False])
        self.assertEqual(oracle["predicate_ids"][0], 2)
        self.assertTrue(oracle["subject_mask_valid"][0])
        self.assertTrue(oracle["reference_mask_valid"][0])
        self.assertEqual(oracle["phase_ids"][0], 0)
        self.assertFalse(oracle["predicate_truth"][0])
        predicate_logits = np.zeros((1, 4, len(PGC_ENTITY_RELATION_PREDICATES)))
        predicate_logits[0, 0, PGC_ENTITY_RELATION_PREDICATES.index("on")] = 5.0
        subject_attention = np.zeros((1, 4, 8), dtype=np.float32)
        reference_attention = np.zeros_like(subject_attention)
        # The exact patch indices are not part of this test's contract; assign
        # mass to every patch occupied by the corresponding downsampled mask.
        subject_attention[0, 0, [2, 3, 6, 7]] = 0.25
        reference_attention[0, 0, [0, 1, 4, 5]] = 0.25
        goal_anchor = np.zeros((1, 4, 3), dtype=np.float32)
        goal_anchor[0, 0] = (
            2.0
            * (np.asarray([0.2, 0.0, 0.2], dtype=np.float32) - contract.workspace_min)
            / (contract.workspace_max - contract.workspace_min)
            - 1.0
        )
        subject_position = np.zeros((1, 4, 3), dtype=np.float32)
        subject_position[0, 0] = (
            2.0
            * (np.asarray([0.0, 0.0, 0.2], dtype=np.float32) - contract.workspace_min)
            / (contract.workspace_max - contract.workspace_min)
            - 1.0
        )
        reference_position = np.zeros((1, 4, 3), dtype=np.float32)
        reference_position[0, 0] = goal_anchor[0, 0]
        phase_logits = np.zeros((1, 4, 3), dtype=np.float32)
        phase_logits[0, 0, 0] = 5.0
        diagnostics = {
            "active_logits": np.asarray([[5.0, -5.0, -5.0, -5.0]]),
            "predicate_logits": predicate_logits,
            "subject_attention": subject_attention,
            "reference_attention": reference_attention,
            "subject_position": subject_position,
            "reference_position": reference_position,
            "goal_anchor": goal_anchor,
            "predicate_truth_logits": np.asarray(
                [[-5.0, -5.0, -5.0, -5.0]], dtype=np.float32
            ),
            "phase_logits": phase_logits,
            "camera_ids": np.asarray([0, 0, 1, 1, 0, 0, 1, 1]),
            "subject_view_visibility_logits": np.asarray(
                [[[5.0, 5.0], [-5.0, -5.0], [-5.0, -5.0], [-5.0, -5.0]]]
            ),
            "reference_view_visibility_logits": np.asarray(
                [[[5.0, 5.0], [-5.0, -5.0], [-5.0, -5.0], [-5.0, -5.0]]]
            ),
            "subject_view_centers": np.zeros((1, 4, 2, 2), dtype=np.float32),
            "reference_view_centers": np.zeros((1, 4, 2, 2), dtype=np.float32),
            "subject_view_attention_mass": np.full((1, 4, 2), 0.5, dtype=np.float32),
            "reference_view_attention_mass": np.full((1, 4, 2), 0.5, dtype=np.float32),
            "subject_base_view_attention_mass": np.full(
                (1, 4, 2), 0.5, dtype=np.float32
            ),
            "reference_base_view_attention_mass": np.full(
                (1, 4, 2), 0.5, dtype=np.float32
            ),
            "subject_view_gate_residual_logits": np.zeros(
                (1, 4, 2), dtype=np.float32
            ),
            "reference_view_gate_residual_logits": np.zeros(
                (1, 4, 2), dtype=np.float32
            ),
            "clause_execution_probability": np.asarray(
                [[1.0, 0.0, 0.0, 0.0]], dtype=np.float32
            ),
            "clause_routing_residual": np.zeros((1, 4), dtype=np.float32),
            "clause_routing_multiplier": np.ones((1, 4), dtype=np.float32),
            "view_scheduler_enabled": np.asarray([True]),
        }
        diagnostics.update(
            {
                "closed_loop_rebinding_enabled": np.asarray([True]),
                "pre_rebinding_subject_attention": subject_attention.copy(),
                "pre_rebinding_reference_attention": reference_attention.copy(),
                "pre_rebinding_subject_position": subject_position.copy(),
                "pre_rebinding_reference_position": reference_position.copy(),
                "pre_rebinding_subject_view_attention_mass": diagnostics[
                    "subject_view_attention_mass"
                ].copy(),
                "pre_rebinding_reference_view_attention_mass": diagnostics[
                    "reference_view_attention_mass"
                ].copy(),
                "pre_rebinding_goal_anchor": goal_anchor.copy(),
                "pre_rebinding_predicate_truth_logits": diagnostics[
                    "predicate_truth_logits"
                ].copy(),
                "pre_rebinding_phase_logits": phase_logits.copy(),
                "pre_rebinding_clause_execution_probability": diagnostics[
                    "clause_execution_probability"
                ].copy(),
                "pre_rebinding_clause_routing_residual": diagnostics[
                    "clause_routing_residual"
                ].copy(),
            }
        )
        record = auditor.observe(
            obs=obs,
            diagnostics=diagnostics,
            episode_idx=0,
            replan_idx=0,
            policy_step=0,
        )
        self.assertEqual(record["clause_count"], 1)
        self.assertTrue(record["all_entity_role_gate"])
        self.assertEqual(record["online_stage"], "pregrasp")
        self.assertEqual(record["online_stage_v2"], "initial_search")
        self.assertEqual(record["clause_statuses"], ["initial_search"])
        self.assertTrue(record["extended_diagnostics"]["available"])
        self.assertTrue(
            record["extended_diagnostics"]["v99_view_fusion_available"]
        )
        self.assertTrue(
            record["extended_diagnostics"]["v99_clause_scheduler_available"]
        )
        self.assertTrue(record["extended_diagnostics"]["clauses"][0]["phase_correct"])
        self.assertTrue(
            record["extended_diagnostics"]["clauses"][0][
                "execution_selection_correct"
            ]
        )
        self.assertTrue(
            record["extended_diagnostics"]["clauses"][0]["predicate_truth_correct"]
        )
        self.assertEqual(record["relation_predictions"], [2])
        self.assertEqual(len(record["goal_anchor_errors_m"]), 1)
        self.assertAlmostEqual(record["goal_anchor_errors_m"][0], 0.0, places=6)
        self.assertTrue(record["closed_loop_rebinding_enabled"])
        self.assertIsNotNone(record["pre_rebinding_record"])
        self.assertEqual(
            record["pre_rebinding_record"]["clause_exact"],
            record["clause_exact"],
        )

    def test_summary_requires_grounding_and_action_integrity(self):
        records = [_good_record() for _ in range(5)]
        for record in records:
            record["online_stage_v2"] = "initial_search"
            record["extended_diagnostics"] = {
                "available": True,
                "missing_diagnostics": [],
                "clauses": [_extended_clause(), _extended_clause()],
            }
        integrity = [
            {
                "exact": True,
                "max_abs_error": 0.0,
                "rms_error": 0.0,
                "gate_mode": "base",
            }
            for _ in records
        ]
        summary = summarize_eraf_shadow_records(
            records,
            action_integrity=integrity,
        )
        self.assertTrue(summary["passed"])
        self.assertEqual(summary["action_integrity"]["exact_rate"], 1.0)
        self.assertEqual(summary["by_online_stage"]["pregrasp"]["decisions"], 5)
        self.assertTrue(summary["extended_diagnostics"]["available"])
        self.assertEqual(summary["extended_diagnostics"]["overall"]["clauses"], 10)
        self.assertEqual(
            summary["extended_diagnostics"]["overall"][
                "privileged_gt_mask_oracle_partition"
            ]["counts"]["role_pass"],
            10,
        )
        self.assertEqual(
            summary["by_online_stage_v2"]["initial_search"]["decisions"], 5
        )

    def test_summary_reports_same_state_pre_post_rebinding_delta(self):
        records = []
        for _ in range(5):
            record = _good_record()
            record["online_stage_v2"] = "holding"
            record["pre_rebinding_record"] = _good_record()
            record["closed_loop_rebinding_enabled"] = True
            record["extended_diagnostics"] = {
                "available": True,
                "missing_diagnostics": [],
                "clauses": [_extended_clause("holding"), _extended_clause("holding")],
            }
            records.append(record)
        integrity = [
            {
                "exact": True,
                "max_abs_error": 0.0,
                "rms_error": 0.0,
                "gate_mode": "base",
            }
            for _ in records
        ]
        summary = summarize_eraf_shadow_records(
            records, action_integrity=integrity
        )
        same_state = summary["same_state_rebinding"]
        self.assertTrue(same_state["available"])
        self.assertEqual(same_state["record_coverage"], 1.0)
        self.assertEqual(
            same_state["post_minus_pre_rebinding"]["clause_exact_match"], 0.0
        )
        self.assertEqual(
            summary["by_online_stage_v2"]["holding"][
                "post_minus_pre_rebinding"
            ]["visible_goal_anchor_median_error_cm"],
            0.0,
        )

    def test_v913_shadow_admission_requires_exact_action_and_frozen_geometry(self):
        records = []
        for _ in range(3):
            record = _good_record()
            record["online_stage_v2"] = "released_unfinished"
            record["phase_safe_memory_enabled"] = True
            record["phase_safe_memory_geometry_max_abs"] = 0.0
            clauses = []
            for _ in range(2):
                clause = _extended_clause("released_unfinished")
                clause.update(
                    {
                        "phase_safe_memory_state_valid": True,
                        "phase_safe_memory_state_correct": True,
                        "phase_safe_memory_completed_sticky_violation": False,
                        "phase_safe_memory_retry_transition": True,
                        "execution_selection_correct": True,
                    }
                )
                clauses.append(clause)
            record["extended_diagnostics"] = {
                "available": True,
                "missing_diagnostics": [],
                "clauses": clauses,
            }
            records.append(record)
        integrity = [
            {
                "exact": True,
                "max_abs_error": 0.0,
                "rms_error": 0.0,
                "gate_mode": "base",
            }
            for _ in records
        ]
        summary = summarize_eraf_shadow_records(
            records, action_integrity=integrity
        )
        memory = summary["phase_safe_clause_memory"]
        self.assertTrue(summary["phase_safe_memory_admission_passed"])
        self.assertEqual(memory["record_coverage"], 1.0)
        self.assertEqual(memory["state_coverage"], 1.0)
        self.assertEqual(memory["state_accuracy"], 1.0)
        self.assertEqual(memory["released_unfinished_retry_rate"], 1.0)
        self.assertEqual(memory["postgrasp_clause_scheduler_accuracy"], 1.0)
        self.assertEqual(memory["completed_sticky_violation_rate"], 0.0)
        self.assertEqual(memory["geometry_max_abs"], 0.0)

        records[0]["phase_safe_memory_geometry_max_abs"] = 1.0e-6
        failed = summarize_eraf_shadow_records(
            records, action_integrity=integrity
        )
        self.assertFalse(failed["phase_safe_memory_admission_passed"])

        for record in records:
            for clause in record["extended_diagnostics"]["clauses"]:
                clause["phase_safe_memory_retry_transition"] = True
                clause["execution_selection_correct"] = False
        failed = summarize_eraf_shadow_records(
            records, action_integrity=integrity
        )
        self.assertFalse(failed["phase_safe_memory_admission_passed"])

        records[0]["phase_safe_memory_geometry_max_abs"] = 0.0
        for record in records:
            for clause in record["extended_diagnostics"]["clauses"]:
                clause["execution_selection_correct"] = True
        records[0]["extended_diagnostics"]["clauses"][0][
            "phase_safe_memory_state_correct"
        ] = False
        failed = summarize_eraf_shadow_records(
            records, action_integrity=integrity
        )
        self.assertFalse(failed["phase_safe_memory_admission_passed"])

        for record in records:
            for clause in record["extended_diagnostics"]["clauses"]:
                clause["phase_safe_memory_state_correct"] = True
                clause["phase_safe_memory_retry_transition"] = False
        failed = summarize_eraf_shadow_records(
            records, action_integrity=integrity
        )
        self.assertFalse(failed["phase_safe_memory_admission_passed"])

    def test_non_sticky_clause_statuses_separate_second_search_and_release(self):
        statuses, phases, stage = _clause_statuses(
            clause_valid=np.asarray([True, True, False, False]),
            predicate_truth=np.asarray([True, False, False, False]),
            subject_grasped=np.asarray([False, False, False, False]),
            subject_ever_grasped=np.asarray([True, False, False, False]),
        )
        self.assertEqual(statuses, ["completed", "next_clause_search"])
        self.assertEqual(phases.tolist(), [2, 0, 0, 0])
        self.assertEqual(stage, "next_clause_search")

        statuses, phases, stage = _clause_statuses(
            clause_valid=np.asarray([True, False, False, False]),
            predicate_truth=np.asarray([False, False, False, False]),
            subject_grasped=np.asarray([False, False, False, False]),
            subject_ever_grasped=np.asarray([True, False, False, False]),
        )
        self.assertEqual(statuses, ["released_unfinished"])
        self.assertEqual(phases.tolist(), [1, 0, 0, 0])
        self.assertEqual(stage, "released_unfinished")


if __name__ == "__main__":
    unittest.main()
