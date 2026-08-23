import unittest

import numpy as np

from scripts.audit_pgc_v9_eraf_action_causality import (
    ORACLE_FIELDS,
    build_causal_variants,
    compute_causal_report,
)


def _sample():
    sample = {}
    clause_valid = np.array([True, True, False, False])
    for source in (False, True):
        prefix = "pgc_eraf_source_" if source else "pgc_eraf_"
        offset = 5 if source else 0
        values = {
            "clause_valid": clause_valid,
            "predicate_ids": np.array([1 + offset, 2, 0, 0]),
            "subject_entity_ids": np.array([10 + offset, 11 + offset, -1, -1]),
            "reference_entity_ids": np.array([20 + offset, 21 + offset, -1, -1]),
            "subject_masks": np.full((4, 8, 16), offset, dtype=np.float32),
            "reference_masks": np.full((4, 8, 16), offset + 1, dtype=np.float32),
            "subject_mask_valid": clause_valid,
            "reference_mask_valid": clause_valid,
            "subject_positions": np.full((4, 3), offset, dtype=np.float32),
            "reference_positions": np.full((4, 3), offset + 1, dtype=np.float32),
            "subject_position_valid": clause_valid,
            "reference_position_valid": clause_valid,
            "goal_anchors": np.full((4, 3), offset + 2, dtype=np.float32),
            "goal_anchor_valid": clause_valid,
            "predicate_truth": np.array([False, False, False, False]),
            "phase_ids": np.zeros(4, dtype=np.int64),
            "phase_valid": clause_valid,
        }
        for name in ORACLE_FIELDS:
            sample[prefix + name] = values[name]
    return sample


class PGCV9CausalAuditTest(unittest.TestCase):
    def test_builds_independent_same_state_interventions(self):
        sample = _sample()
        variants, eligible = build_causal_variants(sample)
        self.assertEqual(
            tuple(variants),
            (
                "learned",
                "oracle",
                "bypass",
                "wrong_subject",
                "wrong_reference",
                "wrong_goal_anchor",
                "clause_swap",
            ),
        )
        self.assertTrue(all(eligible.values()))
        np.testing.assert_array_equal(
            variants["wrong_subject"]["subject_positions"][:2],
            sample["pgc_eraf_source_subject_positions"][:2],
        )
        np.testing.assert_array_equal(
            variants["wrong_subject"]["reference_positions"],
            sample["pgc_eraf_reference_positions"],
        )
        self.assertEqual(variants["clause_swap"]["predicate_ids"][:2].tolist(), [2, 1])
        self.assertIsNone(variants["learned"])
        self.assertTrue(variants["bypass"]["_audit_bypass_bridge"])

    def test_entity_ids_not_mask_drift_define_semantic_eligibility(self):
        sample = _sample()
        sample["pgc_eraf_source_subject_entity_ids"] = sample[
            "pgc_eraf_subject_entity_ids"
        ].copy()
        sample["pgc_eraf_source_reference_entity_ids"] = sample[
            "pgc_eraf_reference_entity_ids"
        ].copy()
        variants, eligible = build_causal_variants(sample)
        self.assertFalse(eligible["wrong_subject"])
        # Shared-reference tasks still receive the documented same-state
        # subject-as-reference negative, never a mask-drift pseudo swap.
        self.assertTrue(eligible["wrong_reference"])
        np.testing.assert_array_equal(
            variants["wrong_subject"]["subject_masks"],
            sample["pgc_eraf_subject_masks"],
        )
        np.testing.assert_array_equal(
            variants["wrong_reference"]["reference_entity_ids"][:2],
            sample["pgc_eraf_subject_entity_ids"][:2],
        )

    def test_v917_anchor_audit_uses_same_state_mirror_fallback(self):
        sample = _sample()
        sample["pgc_eraf_source_goal_anchors"] = sample[
            "pgc_eraf_goal_anchors"
        ].copy()
        variants, eligible = build_causal_variants(
            sample, anchor_mirror_fallback=True
        )
        self.assertTrue(eligible["wrong_goal_anchor"])
        self.assertFalse(
            np.array_equal(
                variants["wrong_goal_anchor"]["goal_anchors"][:2],
                sample["pgc_eraf_goal_anchors"][:2],
            )
        )

    def test_report_distinguishes_bridge_response_and_alignment(self):
        records = []
        for _ in range(4):
            variants = {
                "learned": (0.7, 0.3),
                "oracle": (0.4, 0.6),
                "bypass": (1.0, 0.0),
                "wrong_subject": (0.9, 0.4),
                "wrong_reference": (0.8, 0.4),
                "wrong_goal_anchor": (0.8, 0.4),
                "clause_swap": (0.7, 0.4),
            }
            records.append(
                {
                    "base_expert_mse": 1.0,
                    "eligibility": {
                        "wrong_subject": True,
                        "wrong_reference": True,
                        "wrong_goal_anchor": True,
                        "clause_swap": True,
                    },
                    "variants": {
                        name: {
                            "expert_mse": mse,
                            "base_mse_improvement": 1.0 - mse,
                            "residual_rms": residual,
                            "attention_entropy": 0.5,
                            "expert_loss_goal_query_gradient_rms": 0.1,
                        }
                        for name, (mse, residual) in variants.items()
                    },
                    "pairwise_action_delta_rms": {
                        "oracle__bypass": 0.03,
                        "learned__oracle": 0.01,
                        "wrong_subject__oracle": 0.02,
                        "wrong_reference__oracle": 0.02,
                        "wrong_goal_anchor__oracle": 0.02,
                        "clause_swap__oracle": 0.01,
                    },
                    "pairwise_goal_query_delta_rms": {
                        "oracle__bypass": 0.3,
                        "learned__oracle": 0.1,
                        "wrong_subject__oracle": 0.2,
                        "wrong_reference__oracle": 0.2,
                        "wrong_goal_anchor__oracle": 0.2,
                        "clause_swap__oracle": 0.1,
                    },
                    "pairwise_goal_query_cosine": {
                        "oracle__bypass": 0.8,
                        "learned__oracle": 0.9,
                        "wrong_subject__oracle": 0.8,
                        "wrong_reference__oracle": 0.8,
                        "wrong_goal_anchor__oracle": 0.8,
                        "clause_swap__oracle": 0.9,
                    },
                }
            )
        report = compute_causal_report(records)
        self.assertTrue(report["passed"])
        self.assertEqual(report["diagnosis"], "eraf_action_causal_interface_pass")

    def test_report_calls_out_an_active_but_misaligned_bridge(self):
        record = {
            "base_expert_mse": 1.0,
            "eligibility": {
                "wrong_subject": True,
                "wrong_reference": True,
                "wrong_goal_anchor": True,
                "clause_swap": True,
            },
            "variants": {
                name: {
                    "expert_mse": mse,
                    "base_mse_improvement": 1.0 - mse,
                    "residual_rms": 0.2,
                    "attention_entropy": 0.9,
                    "expert_loss_goal_query_gradient_rms": 0.1,
                }
                for name, mse in {
                    "learned": 1.1,
                    "oracle": 1.2,
                    "bypass": 1.0,
                    "wrong_subject": 1.4,
                    "wrong_reference": 1.4,
                    "wrong_goal_anchor": 1.4,
                    "clause_swap": 1.3,
                }.items()
            },
            "pairwise_action_delta_rms": {
                "oracle__bypass": 0.1,
                "learned__oracle": 0.1,
                "wrong_subject__oracle": 0.1,
                "wrong_reference__oracle": 0.1,
                "wrong_goal_anchor__oracle": 0.1,
                "clause_swap__oracle": 0.1,
            },
            "pairwise_goal_query_delta_rms": {
                "oracle__bypass": 0.3,
                "learned__oracle": 0.1,
                "wrong_subject__oracle": 0.2,
                "wrong_reference__oracle": 0.2,
                "wrong_goal_anchor__oracle": 0.2,
                "clause_swap__oracle": 0.1,
            },
            "pairwise_goal_query_cosine": {
                "oracle__bypass": 0.8,
                "learned__oracle": 0.9,
                "wrong_subject__oracle": 0.8,
                "wrong_reference__oracle": 0.8,
                "wrong_goal_anchor__oracle": 0.8,
                "clause_swap__oracle": 0.9,
            },
        }
        report = compute_causal_report([record])
        self.assertFalse(report["passed"])
        self.assertEqual(
            report["diagnosis"], "eraf_action_bridge_active_but_expert_misaligned"
        )


if __name__ == "__main__":
    unittest.main()
