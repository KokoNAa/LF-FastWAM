import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiments.libero.eraf_shadow_audit import (
    ERAFShadowAuditor,
    ERAFShadowContract,
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
        )
        segmentation = np.zeros((4, 4, 1), dtype=np.int32)
        segmentation[:2, :2, 0] = 1
        segmentation[2:, 2:, 0] = 2
        obs = {
            "agentview_segmentation_element": segmentation,
            "robot0_eye_in_hand_segmentation_element": segmentation,
        }
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
        diagnostics = {
            "active_logits": np.asarray([[5.0, -5.0, -5.0, -5.0]]),
            "predicate_logits": predicate_logits,
            "subject_attention": subject_attention,
            "reference_attention": reference_attention,
            "goal_anchor": goal_anchor,
        }
        record = auditor.observe(
            obs=obs,
            diagnostics=diagnostics,
            episode_idx=0,
            replan_idx=0,
            policy_step=0,
        )
        self.assertEqual(record["clause_count"], 1)
        self.assertEqual(record["online_stage"], "pregrasp")
        self.assertEqual(record["relation_predictions"], [2])
        self.assertEqual(len(record["goal_anchor_errors_m"]), 1)
        self.assertAlmostEqual(record["goal_anchor_errors_m"][0], 0.0, places=6)

    def test_summary_requires_grounding_and_action_integrity(self):
        records = [_good_record() for _ in range(5)]
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


if __name__ == "__main__":
    unittest.main()
