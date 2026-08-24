import hashlib
import json
import tempfile
import unittest
from collections import Counter, OrderedDict
from dataclasses import replace
from inspect import signature
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from fastwam.datasets.lerobot.robot_video_dataset import (
    RobotVideoDataset,
    build_pgc_v96_sample_plan,
    build_pgc_v912_sample_plan,
    build_pgc_v9_sample_indices,
)
from fastwam.datasets.lerobot.base_lerobot_dataset import BaseLerobotDataset
from fastwam.datasets.pgc_libero import (
    PGC_ACTION_CONVENTION_FASTWAM,
    PGC_ACTION_CONVENTION_LIBERO_ENV,
    PGC_ACTION_REPLAY_FASTWAM_TO_LIBERO_ENV,
    PGC_ENTITY_RELATION_ARRAY_NAMES,
    PGC_ENTITY_RELATION_WORKSPACE_MAX,
    PGC_ENTITY_RELATION_WORKSPACE_MIN,
    array_sha256,
    classify_strict_conflict,
    fastwam_actions_to_libero_env,
    libero_env_actions_to_fastwam,
    load_pgc_entity_relation_index,
    parse_libero_goal_clauses,
    pgc_entity_relation_workspace_bounds,
    provenance_pair,
    state_sha256,
    validate_strict_conflict_audit,
)
from fastwam.models.wan22.entity_relation_affordance import (
    ClauseActivationCalibrationAdapter,
    ERAFLossWeights,
    EntityRelationAffordanceField,
    PhaseSafeClauseMemory,
    _balanced_bipartite_assignment_loss,
    _balanced_clause_tuple_assignment_loss,
    _balanced_exclusive_all_entity_assignment_loss,
    _balanced_exclusive_role_assignment_loss,
    _clause_activation_calibration_loss,
    _structured_all_entity_assignment_loss,
    entity_relation_affordance_loss,
)
from fastwam.models.wan22.fastwam import FastWAM
from fastwam.models.wan22.policy_guard import (
    HardRoutedERAFPhaseServo,
    PhaseCompatibleERAFWaypointAdapter,
    PhaseConditionedERAFActionBridge,
    PhaseConditionedERAFGeometryActionAdapter,
    PhaseSpecificERAFExpertResidualAdapter,
    detached_policy_guard_metrics,
    infer_spatial_patch_grid,
)
import scripts.build_pgc_libero_entity_relations as eraf_builder
import scripts.build_pgc_v912_closed_loop_grounding_data as v912_builder
from scripts.build_pgc_libero_entity_relations import (
    ARRAY_NAMES,
    _match_native_demo,
    _region_anchor,
)
from scripts.eval_pgc_v9_grounding_gate import (
    compute_grounding_gate_report,
    mine_hard_native_rows,
)
from tests.test_policy_guard import tiny_pgc_fastwam
from fastwam.utils.samplers import ResumableEpochSampler


class PGCERAFParsingTest(unittest.TestCase):
    def test_v912_builder_uses_the_shared_eraf_workspace_contract(self):
        with patch(
            "sys.argv",
            [
                "build_pgc_v912_closed_loop_grounding_data.py",
                "--captures",
                "/tmp/captures",
                "--output",
                "/tmp/dataset",
                "--sidecar-output",
                "/tmp/sidecar",
                "--suite",
                "libero_10",
            ],
        ):
            args = v912_builder._parse_args()
        self.assertEqual(
            tuple(args.workspace_min), PGC_ENTITY_RELATION_WORKSPACE_MIN
        )
        self.assertEqual(
            tuple(args.workspace_max), PGC_ENTITY_RELATION_WORKSPACE_MAX
        )

    def test_eraf_workspace_contract_rejects_mixed_coordinate_frames(self):
        canonical = {
            "workspace_min": list(PGC_ENTITY_RELATION_WORKSPACE_MIN),
            "workspace_max": list(PGC_ENTITY_RELATION_WORKSPACE_MAX),
        }
        lower, upper = pgc_entity_relation_workspace_bounds(
            {0: canonical, 1: canonical}
        )
        np.testing.assert_allclose(lower, PGC_ENTITY_RELATION_WORKSPACE_MIN)
        np.testing.assert_allclose(upper, PGC_ENTITY_RELATION_WORKSPACE_MAX)
        incompatible = {
            "workspace_min": [-0.65, -0.60, 0.70],
            "workspace_max": [0.65, 0.60, 1.45],
        }
        with self.assertRaisesRegex(ValueError, "disagree on workspace bounds"):
            pgc_entity_relation_workspace_bounds(
                {0: canonical, 1: incompatible}
            )

    def test_eraf_array_contract_is_shared_by_builder_loader_and_model(self):
        self.assertEqual(ARRAY_NAMES, PGC_ENTITY_RELATION_ARRAY_NAMES)
        for name in (
            "subject_view_visible",
            "reference_view_visible",
            "subject_view_centers",
            "reference_view_centers",
        ):
            self.assertIn(name, PGC_ENTITY_RELATION_ARRAY_NAMES)

    def test_predicate_inventory_and_structural_regions(self):
        regions = {
            "basket_region": {"target": "basket_1"},
            "left_region": {"target": ["plate_1"]},
            "cabinet_middle_region": {"target": "cabinet_1"},
        }
        goals = [
            ["in", "soup_1", "basket_region"],
            ["on", "mug_1", "left_region"],
            ["left", "pudding_1", "plate_1"],
            ["right", "pudding_1", "plate_1"],
            ["front", "plate_1", "stove_1"],
            ["back", "plate_1", "stove_1"],
            ["open", "cabinet_middle_region"],
            ["close", "microwave_1"],
            ["turnon", "stove_1"],
            ["turnoff", "stove_1"],
        ]
        parsed = [
            parse_libero_goal_clauses([goal], regions=regions)[0]
            for goal in goals
        ]
        self.assertEqual(
            [clause["predicate"] for clause in parsed],
            ["in", "on", "left", "right", "front", "back", "open", "close", "turnon", "turnoff"],
        )
        self.assertEqual(parsed[0]["reference"], "basket_1")
        self.assertEqual(parsed[0]["reference_region"], "basket_region")
        self.assertEqual(parsed[1]["reference"], "plate_1")
        self.assertEqual(parsed[1]["reference_region"], "left_region")
        self.assertEqual(parsed[6]["subject"], "cabinet_1")
        self.assertEqual(parsed[6]["reference"], "cabinet_1")
        self.assertEqual(parsed[6]["subject_region"], "cabinet_middle_region")

    def test_multiple_subjects_stay_in_independent_clauses(self):
        clauses = parse_libero_goal_clauses(
            [
                ["in", "soup_1", "basket_1"],
                ["in", "tomato_1", "basket_1"],
            ]
        )
        self.assertEqual(len(clauses), 2)
        self.assertEqual([item["subject"] for item in clauses], ["soup_1", "tomato_1"])

    def test_directional_on_region_uses_structural_target_and_language_relation(self):
        regions = {
            "table_plate_left_region": {
                "target": "living_room_table_1",
                "ranges": [[-0.2, -0.1, 0.0, 0.1]],
            }
        }
        clause = parse_libero_goal_clauses(
            [["on", "pudding_1", "table_plate_left_region"]],
            regions=regions,
            instruction="put the pudding to the left of the plate",
            entity_catalog={
                "chocolate_pudding": ["pudding_1"],
                "plate": ["plate_1"],
                "living_room_table": ["living_room_table_1"],
            },
        )[0]
        self.assertEqual(clause["predicate"], "left")
        self.assertEqual(clause["reference"], "plate_1")
        self.assertEqual(clause["reference_region"], "table_plate_left_region")
        self.assertEqual(clause["reference_region_target"], "living_room_table_1")
        self.assertEqual(clause["raw"][0], "on")

    def test_strict_categories_and_shortcut_rejection(self):
        def clause(predicate="in", subject="a", reference="basket"):
            return {"predicate": predicate, "subject": subject, "reference": reference}

        self.assertEqual(
            classify_strict_conflict([clause(subject="a")], [clause(subject="b")]),
            "entity_swap",
        )
        self.assertEqual(
            classify_strict_conflict([clause("left")], [clause("right")]),
            "direction_swap",
        )
        self.assertEqual(
            classify_strict_conflict(
                [clause("open", "drawer", "drawer")],
                [clause("close", "drawer", "drawer")],
            ),
            "articulated_state",
        )
        shared = clause(subject="shared")
        self.assertIsNone(
            classify_strict_conflict(
                [shared], [shared, clause(subject="new")]
            )
        )
        self.assertEqual(
            classify_strict_conflict(
                [clause(subject="a"), clause(subject="b")],
                [clause(subject="c"), clause(subject="d")],
            ),
            "conjunction",
        )
        region_a = clause()
        region_a["reference_region"] = "plate_left_region"
        region_b = clause()
        region_b["reference_region"] = "plate_right_region"
        self.assertEqual(
            classify_strict_conflict([region_a], [region_b]),
            "relation_swap",
        )
        passing = {
            "source_demo_source_success": 5,
            "target_demo_target_success": 5,
            "target_initially_false": 5,
            "source_initially_false": 5,
            "source_demo_target_success": 0,
            "target_demo_source_success": 0,
        }
        self.assertTrue(validate_strict_conflict_audit(passing)["strict_conflict_passed"])
        failing = dict(passing, source_demo_target_success=1)
        with self.assertRaisesRegex(ValueError, "shortcut|source_demo_target_success"):
            validate_strict_conflict_audit(failing)

    def test_strict_replay_audit_survives_dataset_provenance(self):
        audit = {
            "source_demo_source_success": 5,
            "target_demo_target_success": 5,
            "target_initially_false": 5,
            "source_initially_false": 5,
            "source_demo_target_success": 0,
            "target_demo_source_success": 0,
            "required_demos": 5,
            "strict_conflict_passed": True,
        }
        record = {
            "pair_id": "libero_10_00_to_libero_10_01",
            "task_suite_name": "libero_10",
            "task_id": 0,
            "correct_instruction": "source",
            "counterfactual_instruction": "target",
            "counterfactual_goal_state": [["in", "b", "basket"]],
            "counterfactual_bddl_file": "/tmp/target.bddl",
            "strict_conflict": True,
            "strict_conflict_type": "entity_swap",
            "strict_replay_audit": audit,
        }
        pair = provenance_pair(record)
        self.assertTrue(pair["strict_conflict"])
        self.assertEqual(pair["strict_conflict_type"], "entity_swap")
        self.assertTrue(pair["strict_replay_audit"]["strict_conflict_passed"])


class PGCERAFGeometryTest(unittest.TestCase):
    class _Model:
        def __init__(self, site_names=()):
            self.site_names = list(site_names)

        def site_name2id(self, name):
            if name not in self.site_names:
                raise KeyError(name)
            return self.site_names.index(name)

    @classmethod
    def _env(cls, *, site_names=(), site_xpos=(), workspace_offset=(0, 0, 0)):
        model = cls._Model(site_names)
        data = SimpleNamespace(
            site_xpos=np.asarray(site_xpos, dtype=np.float32).reshape(-1, 3),
            body_xpos=np.zeros((1, 3), dtype=np.float32),
        )
        inner = SimpleNamespace(
            sim=SimpleNamespace(model=model, data=data),
            object_sites_dict={},
            workspace_offset=np.asarray(workspace_offset, dtype=np.float32),
            obj_body_id={},
        )
        return SimpleNamespace(env=inner)

    def test_region_anchor_prefers_runtime_site_world_position(self):
        env = self._env(
            site_names=("fixture_plate_left_region",),
            site_xpos=((0.31, -0.22, 0.87),),
            workspace_offset=(10.0, 20.0, 30.0),
        )
        position, valid = _region_anchor(
            env,
            {
                "subject": "object_1",
                "reference": "plate_1",
                "reference_region": "plate_left_region",
            },
            {
                "regions": {
                    "plate_left_region": {"ranges": [[-0.1, -0.2, 0.1, 0.2]]}
                }
            },
        )
        self.assertTrue(valid)
        np.testing.assert_allclose(position, [0.31, -0.22, 0.87])

    def test_table_region_range_is_local_to_workspace_offset(self):
        env = self._env(workspace_offset=(0.5, -0.25, 0.8))
        position, valid = _region_anchor(
            env,
            {
                "subject": "object_1",
                "reference": "table_1",
                "reference_region": "table_target_region",
            },
            {
                "regions": {
                    "table_target_region": {
                        "target": "table_1",
                        "ranges": [[-0.2, -0.1, 0.4, 0.3]],
                    }
                }
            },
        )
        self.assertTrue(valid)
        np.testing.assert_allclose(position, [0.6, -0.15, 0.8])

    def test_two_camera_patch_grid_keeps_view_identity(self):
        self.assertEqual(
            infer_spatial_patch_grid(392, aspect_ratio=2.0), (14, 28)
        )
        module = EntityRelationAffordanceField(
            text_dim=10,
            video_dim=16,
            action_dim=12,
            projection_dim=8,
            hidden_dim=8,
            num_heads=2,
            camera_count=2,
            visual_aspect_ratio=2.0,
        )
        _, _, camera_ids, view_coordinates = (
            module.entity_grounder._spatial_visual(torch.randn(1, 392, 16))
        )
        camera_grid = camera_ids.reshape(14, 28)
        self.assertTrue(torch.equal(camera_grid[:, :14], torch.zeros(14, 14, dtype=torch.long)))
        self.assertTrue(torch.equal(camera_grid[:, 14:], torch.ones(14, 14, dtype=torch.long)))
        local_x = view_coordinates[:, 0].reshape(14, 28)
        self.assertTrue(torch.equal(local_x[:, :14], local_x[:, 14:]))


class PGCERAFSamplingTest(unittest.TestCase):
    def test_nested_one_to_one_and_relation_balance(self):
        result = build_pgc_v9_sample_indices(
            native_indices=[0, 1, 2, 3],
            original_counterfactual_indices=[4, 5],
            strict_counterfactual_indices=[6, 7, 8],
            strict_relation_categories=["entity", "entity", "relation"],
        )
        counts = Counter(result)
        native_count = sum(counts[index] for index in range(4))
        original_count = counts[4] + counts[5]
        strict_count = counts[6] + counts[7] + counts[8]
        entity_count = counts[6] + counts[7]
        relation_count = counts[8]
        self.assertEqual(native_count, original_count + strict_count)
        self.assertEqual(original_count, strict_count)
        self.assertEqual(entity_count, relation_count)

    def test_ratios_remain_exact_when_native_pool_dominates(self):
        result = build_pgc_v9_sample_indices(
            native_indices=list(range(101)),
            original_counterfactual_indices=[101, 102, 103],
            strict_counterfactual_indices=[104, 105, 106],
            strict_relation_categories=["entity", "entity", "relation"],
        )
        counts = Counter(result)
        native_count = sum(counts[index] for index in range(101))
        original_count = sum(counts[index] for index in (101, 102, 103))
        entity_count = sum(counts[index] for index in (104, 105))
        relation_count = counts[106]
        strict_count = entity_count + relation_count
        self.assertEqual(native_count, original_count + strict_count)
        self.assertEqual(original_count, strict_count)
        self.assertEqual(entity_count, relation_count)

    def test_v94_balances_each_pool_by_task_or_pair(self):
        result = build_pgc_v9_sample_indices(
            native_indices=[0, 1, 2, 3],
            original_counterfactual_indices=[4, 5, 6],
            strict_counterfactual_indices=[7, 8, 9],
            strict_relation_categories=["entity", "entity", "relation"],
            native_role_categories=["native_a", "native_a", "native_a", "native_b"],
            original_role_categories=["original_a", "original_a", "original_b"],
            strict_role_categories=["strict_a", "strict_a", "strict_b"],
        )
        counts = Counter(result)
        native_a = sum(counts[index] for index in (0, 1, 2))
        native_b = counts[3]
        original_a = sum(counts[index] for index in (4, 5))
        original_b = counts[6]
        strict_a = sum(counts[index] for index in (7, 8))
        strict_b = counts[9]
        self.assertEqual(native_a, native_b)
        self.assertEqual(original_a, original_b)
        self.assertEqual(strict_a, strict_b)
        native_count = native_a + native_b
        original_count = original_a + original_b
        strict_count = strict_a + strict_b
        self.assertEqual(native_count, original_count + strict_count)
        self.assertEqual(original_count, strict_count)

    def test_v94_requires_structured_categories_for_every_pool(self):
        with self.assertRaisesRegex(ValueError, "all three"):
            build_pgc_v9_sample_indices(
                native_indices=[0],
                original_counterfactual_indices=[1],
                strict_counterfactual_indices=[2],
                strict_relation_categories=["entity"],
                native_role_categories=["native"],
            )

    def test_v96_builds_equal_four_way_curriculum(self):
        indices, groups = build_pgc_v96_sample_plan(
            native_indices=list(range(8)),
            hard_native_indices=[1, 5],
            original_counterfactual_indices=[8, 9],
            strict_counterfactual_indices=[10, 11],
            strict_relation_categories=["entity", "relation"],
            native_role_categories=["a", "a", "a", "a", "b", "b", "b", "b"],
            original_role_categories=["original_a", "original_b"],
            strict_role_categories=["strict_a", "strict_b"],
        )
        self.assertEqual(len(indices), len(groups))
        counts = Counter(groups)
        self.assertEqual(counts, Counter({0: 6, 1: 6, 2: 6, 3: 6}))
        for index, group in zip(indices, groups, strict=True):
            if group == 0:
                self.assertIn(index, {1, 5})
            elif group == 1:
                self.assertIn(index, {0, 2, 3, 4, 6, 7})
            elif group == 2:
                self.assertIn(index, {8, 9})
            else:
                self.assertIn(index, {10, 11})

    def test_v912_balances_closed_loop_phases_and_four_guard_pools(self):
        indices, groups = build_pgc_v912_sample_plan(
            offline_native_indices=[0, 1, 2, 3, 4],
            closed_loop_native_indices=[5, 6, 7, 8, 9, 10],
            original_counterfactual_indices=[11, 12],
            strict_counterfactual_indices=[13, 14, 15],
            closed_loop_stage_categories=[
                "initial_search",
                "initial_search",
                "initial_search",
                "holding",
                "released_unfinished",
                "next_clause_search",
            ],
            strict_relation_categories=["entity", "entity", "relation"],
        )
        self.assertEqual(len(indices), len(groups))
        counts = Counter(groups)
        self.assertEqual(len(set(counts.values())), 1)
        self.assertEqual(set(counts), {0, 1, 2, 3})
        for position in range(0, len(groups), 4):
            self.assertEqual(groups[position : position + 4], [0, 1, 2, 3])
        sampled = Counter(indices)
        initial_search = sum(sampled[index] for index in (5, 6, 7))
        holding = sampled[8]
        released = sampled[9]
        next_clause = sampled[10]
        self.assertEqual(initial_search, holding)
        self.assertEqual(holding, released)
        self.assertEqual(released, next_clause)
        self.assertEqual(sampled[13] + sampled[14], sampled[15])

    def test_v911_mines_only_multiclause_v910_native_failures(self):
        def record(
            raw_index: int,
            *,
            kind: str,
            clauses: int,
            role_correct: bool,
            localized: bool,
        ) -> dict[str, object]:
            return {
                "_raw_index": raw_index,
                "_dataset_kind": kind,
                "clause_count": clauses,
                "subject_top1_hits": [localized] * clauses,
                "reference_top1_hits": [True] * clauses,
                "role_audit_clauses": [
                    {
                        "all_entity_exclusive_valid": True,
                        "all_entity_exclusive_correct": role_correct,
                    }
                    for _ in range(clauses)
                ],
            }

        mined = mine_hard_native_rows(
            [
                record(
                    0,
                    kind="native",
                    clauses=2,
                    role_correct=True,
                    localized=True,
                ),
                record(
                    1,
                    kind="native",
                    clauses=2,
                    role_correct=False,
                    localized=True,
                ),
                record(
                    2,
                    kind="native",
                    clauses=1,
                    role_correct=False,
                    localized=False,
                ),
                record(
                    99,
                    kind="counterfactual",
                    clauses=2,
                    role_correct=False,
                    localized=False,
                ),
            ],
            objective_version=11,
        )
        self.assertEqual(mined["audited_native_raw_indices"], [0, 1, 2])
        self.assertEqual(mined["hard_native_raw_indices"], [1])
        self.assertEqual(mined["format"], "pgc_v9_hard_role_index_v2")

    def test_v96_sampler_balances_every_global_optimizer_window(self):
        class CurriculumDataset:
            pgc_v9_hard_curriculum_group_ids = [0, 1, 2, 3] * 12

            def __len__(self):
                return len(self.pgc_v9_hard_curriculum_group_ids)

        dataset = CurriculumDataset()
        sampler = ResumableEpochSampler(
            dataset=dataset,
            seed=42,
            batch_size=1,
            num_processes=3,
            gradient_accumulation_steps=4,
        )
        order = list(sampler)
        labels = [dataset.pgc_v9_hard_curriculum_group_ids[index] for index in order]
        for start in range(0, len(labels), 12):
            self.assertEqual(
                Counter(labels[start : start + 12]),
                Counter({0: 3, 1: 3, 2: 3, 3: 3}),
            )
            window = labels[start : start + 12]
            for process in range(3):
                self.assertEqual(
                    Counter(window[process::3]),
                    Counter({0: 1, 1: 1, 2: 1, 3: 1}),
                )

    def test_v912_sampler_balances_every_global_optimizer_window(self):
        class ClosedLoopCurriculumDataset:
            pgc_v9_closed_loop_group_ids = [0, 1, 2, 3] * 12

            def __len__(self):
                return len(self.pgc_v9_closed_loop_group_ids)

        dataset = ClosedLoopCurriculumDataset()
        sampler = ResumableEpochSampler(
            dataset=dataset,
            seed=912,
            batch_size=1,
            num_processes=3,
            gradient_accumulation_steps=4,
        )
        order = list(sampler)
        labels = [dataset.pgc_v9_closed_loop_group_ids[index] for index in order]
        for start in range(0, len(labels), 12):
            window = labels[start : start + 12]
            self.assertEqual(
                Counter(window), Counter({0: 3, 1: 3, 2: 3, 3: 3})
            )
            for process in range(3):
                self.assertEqual(
                    Counter(window[process::3]),
                    Counter({0: 1, 1: 1, 2: 1, 3: 1}),
                )

    def test_rejects_overlapping_v9_curriculum_contracts(self):
        class InvalidCurriculumDataset:
            pgc_v9_hard_curriculum_group_ids = [0, 1, 2, 3]
            pgc_v9_closed_loop_group_ids = [0, 1, 2, 3]

            def __len__(self):
                return 4

        sampler = ResumableEpochSampler(
            dataset=InvalidCurriculumDataset(),
            seed=42,
            batch_size=1,
            num_processes=1,
            gradient_accumulation_steps=4,
        )
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            list(sampler)

    def test_rejects_missing_or_overlapping_pools(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            build_pgc_v9_sample_indices(
                native_indices=[0],
                original_counterfactual_indices=[],
                strict_counterfactual_indices=[2],
                strict_relation_categories=["entity"],
            )
        with self.assertRaisesRegex(ValueError, "disjoint"):
            build_pgc_v9_sample_indices(
                native_indices=[0],
                original_counterfactual_indices=[1],
                strict_counterfactual_indices=[1],
                strict_relation_categories=["entity"],
            )


class PGCERAFDatasetAuditTest(unittest.TestCase):
    def test_native_demo_lookup_uses_source_suite_for_cross_suite_pair(self):
        hdf5_actions = np.asarray(
            [
                [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
                [0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        actions = libero_env_actions_to_fastwam(hdf5_actions)
        record = {
            "pair_id": "libero_10_02_to_libero_90_18",
            "task_suite_name": "libero_10",
            "task_id": 2,
            "correct_instruction": "turn on the stove and put the moka pot on it",
            "counterfactual_task_suite_name": "libero_90",
            "counterfactual_task_id": 18,
            "counterfactual_instruction": "put the frying pan on the stove",
            "counterfactual_bddl_file": "/demo/libero_90/target.bddl",
        }
        demo = SimpleNamespace(
            actions=hdf5_actions,
            initial_state=np.asarray([1.0, 2.0], dtype=np.float64),
            group_name="demo_0",
        )
        captured = {}

        def resolve(_root, lookup):
            captured.update(lookup)
            return Path("/demo/libero_10/source_demo.hdf5")

        with (
            patch.object(
                eraf_builder,
                "_problem_for_task",
                return_value=(None, None, Path("/bddl/libero_10/source.bddl")),
            ),
            patch.object(eraf_builder, "resolve_demo_file", side_effect=resolve),
            patch.object(
                eraf_builder,
                "iter_libero_hdf5_demos",
                return_value=iter([demo]),
            ),
        ):
            state, demo_path, group_name = _match_native_demo(
                record=record,
                actions=actions,
                hdf5_root=Path("/demo"),
                used=set(),
            )

        self.assertEqual(captured["counterfactual_task_suite_name"], "libero_10")
        self.assertEqual(captured["counterfactual_task_id"], 2)
        self.assertEqual(captured["counterfactual_instruction"], record["correct_instruction"])
        self.assertEqual(captured["counterfactual_bddl_file"], "/bddl/libero_10/source.bddl")
        np.testing.assert_array_equal(state, demo.initial_state)
        self.assertEqual(demo_path, "/demo/libero_10/source_demo.hdf5")
        self.assertEqual(group_name, "demo_0")

    def test_libero_and_fastwam_gripper_conventions_round_trip(self):
        env_actions = np.asarray(
            [
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -1.0],
                [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, 1.0],
            ],
            dtype=np.float32,
        )
        fastwam_actions = libero_env_actions_to_fastwam(env_actions)
        np.testing.assert_array_equal(fastwam_actions[:, :6], env_actions[:, :6])
        np.testing.assert_array_equal(fastwam_actions[:, -1], [1.0, 0.0])
        np.testing.assert_array_equal(
            fastwam_actions_to_libero_env(fastwam_actions), env_actions
        )

    def test_base_dataset_aligns_only_authenticated_counterfactual_actions(self):
        dataset = object.__new__(BaseLerobotDataset)
        dataset.multi_dataset = SimpleNamespace(_datasets=[object(), object()])
        dataset.action_conventions_by_dataset_index = {}
        dataset.set_action_conventions_by_dataset_index(
            {
                0: PGC_ACTION_CONVENTION_FASTWAM,
                1: PGC_ACTION_CONVENTION_LIBERO_ENV,
            }
        )
        meta = {"key": "default", "lerobot_key": "action", "raw_shape": 7}
        native = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float32
        )
        counterfactual = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        native_result = dataset._get_action(
            meta, {"action": native, "dataset_index": torch.tensor(0)}
        )
        cf_result = dataset._get_action(
            meta,
            {"action": counterfactual, "dataset_index": torch.tensor(1)},
        )
        self.assertTrue(torch.equal(native_result, native))
        self.assertTrue(torch.equal(cf_result[:, -1], torch.tensor([1.0, 0.0])))
        self.assertTrue(torch.equal(counterfactual[:, -1], torch.tensor([-1.0, 1.0])))

    class _Metadata:
        total_episodes = 1

    class _Dataset:
        meta = None
        episodes = None

        def __init__(self, action):
            self.meta = PGCERAFDatasetAuditTest._Metadata()
            self._action = torch.as_tensor(action)

        def get_episode_data(self, episode_index):
            if int(episode_index) != 0:
                raise IndexError(episode_index)
            return {"action": self._action}

    def _loader(self, record, *, native_dataset_count):
        loader = object.__new__(RobotVideoDataset)
        loader.pgc_entity_relation_indices = {
            0: {
                "episode_count": 1,
                "episodes_by_index": {0: record},
            }
        }
        loader.pgc_native_dataset_count = int(native_dataset_count)
        return loader

    def test_action_and_counterfactual_state_hashes_are_cross_audited(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            state_dir = root / "meta/pgc_initial_states"
            state_dir.mkdir(parents=True)
            state = np.asarray([0.1, 0.2, 0.3], dtype=np.float64)
            state_path = state_dir / "episode_000000.npy"
            np.save(state_path, state, allow_pickle=False)
            action = np.arange(21, dtype=np.float32).reshape(3, 7)
            record = {
                "episode_index": 0,
                "pair_id": "strict_pair",
                "frame_count": 3,
                "action_sha256": array_sha256(action),
                "initial_state_sha256": state_sha256(state),
            }
            audit = {
                "episode_index": 0,
                "pair_id": "strict_pair",
                "initial_state_sha256": state_sha256(state),
                "source_initial_state_catalog": (
                    "meta/pgc_initial_states/episode_000000.npy"
                ),
            }
            (root / "meta/pgc_episodes.jsonl").write_text(
                json.dumps(audit) + "\n", encoding="utf-8"
            )
            loader = self._loader(record, native_dataset_count=0)
            loader._validate_pgc_entity_relation_dataset_audits(
                underlying=[self._Dataset(action)],
                combined_dataset_dirs=[str(root)],
            )

            corrupted = action.copy()
            corrupted[0, 0] += 1.0
            with self.assertRaisesRegex(ValueError, "action hash"):
                loader._validate_pgc_entity_relation_dataset_audits(
                    underlying=[self._Dataset(corrupted)],
                    combined_dataset_dirs=[str(root)],
                )

    def test_sidecar_hash_and_multiview_array_schema_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            episode_dir = root / "episodes"
            episode_dir.mkdir()
            scalar_shape = (1, 4)
            arrays = {}
            for role in ("target", "source"):
                clause_valid = np.asarray([[True, False, False, False]])
                arrays[f"{role}_predicate_ids"] = np.asarray(
                    [[1, 0, 0, 0]], dtype=np.int64
                )
                arrays[f"{role}_clause_valid"] = clause_valid
                for entity_role, entity_id in (
                    ("subject", 11),
                    ("reference", 21),
                ):
                    entity_ids = np.full(scalar_shape, -1, dtype=np.int64)
                    entity_ids[0, 0] = entity_id
                    arrays[f"{role}_{entity_role}_entity_ids"] = entity_ids
                    masks = np.zeros((*scalar_shape, 2, 4), dtype=np.bool_)
                    masks[0, 0, 0, 0] = True
                    arrays[f"{role}_{entity_role}_masks"] = masks
                    arrays[f"{role}_{entity_role}_mask_valid"] = (
                        clause_valid.copy()
                    )
                    visible = np.zeros((*scalar_shape, 2), dtype=np.bool_)
                    visible[0, 0, 0] = True
                    arrays[f"{role}_{entity_role}_view_visible"] = visible
                    centers = np.zeros((*scalar_shape, 2, 2), dtype=np.float32)
                    centers[0, 0, 0] = (-0.5, 0.5)
                    arrays[f"{role}_{entity_role}_view_centers"] = centers
                    arrays[f"{role}_{entity_role}_positions"] = np.zeros(
                        (*scalar_shape, 3), dtype=np.float32
                    )
                    arrays[f"{role}_{entity_role}_position_valid"] = (
                        clause_valid.copy()
                    )
                for anchor in ("grasp", "goal", "interaction"):
                    arrays[f"{role}_{anchor}_anchors"] = np.zeros(
                        (*scalar_shape, 3), dtype=np.float32
                    )
                    arrays[f"{role}_{anchor}_anchor_valid"] = (
                        clause_valid.copy()
                    )
                arrays[f"{role}_predicate_truth"] = np.zeros(
                    scalar_shape, dtype=np.float32
                )
                arrays[f"{role}_predicate_truth_valid"] = clause_valid.copy()
                arrays[f"{role}_phase_ids"] = np.zeros(
                    scalar_shape, dtype=np.int64
                )
                arrays[f"{role}_phase_valid"] = clause_valid.copy()

            episode_path = episode_dir / "episode_000000.npz"
            np.savez_compressed(episode_path, **arrays)
            digest = hashlib.sha256(episode_path.read_bytes()).hexdigest()
            index = {
                "format": "pgc_libero_entity_relation_v1",
                "dataset": str(root / "dataset"),
                "dataset_kind": "native",
                "dataset_action_convention": PGC_ACTION_CONVENTION_FASTWAM,
                "simulator_replay_action_transform": (
                    PGC_ACTION_REPLAY_FASTWAM_TO_LIBERO_ENV
                ),
                "privileged_supervision": "training_only",
                "deployment_inputs": "rgb_language_proprio",
                "max_clauses": 4,
                "predicate_vocabulary": [
                    "pad", "in", "on", "left", "right", "front", "back",
                    "open", "close", "turnon", "turnoff",
                ],
                "entity_id_scheme": "sha256_63bit",
                "entity_vocabulary": {"subject_1": 11, "reference_1": 21},
                "camera_names": ["agentview", "robot0_eye_in_hand"],
                "view_center_coordinate_system": "per_camera_normalized_xy",
                "mask_size": [2, 4],
                "workspace_min": [-1.0, -1.0, -1.0],
                "workspace_max": [1.0, 1.0, 1.0],
                "episode_count": 1,
                "episodes": [
                    {
                        "episode_index": 0,
                        "pair_id": "pair",
                        "file": "episodes/episode_000000.npz",
                        "sha256": digest,
                        "state_sha256": "0" * 64,
                        "action_sha256": "1" * 64,
                        "frame_count": 1,
                    }
                ],
            }
            (root / "index.json").write_text(
                json.dumps(index), encoding="utf-8"
            )
            loaded = load_pgc_entity_relation_index(root)
            loader = object.__new__(RobotVideoDataset)
            loader.pgc_entity_relation_indices = {0: loaded}
            loader._pgc_entity_relation_cache = OrderedDict()
            loader._pgc_entity_relation_cache_size = 2
            payload = loader._load_pgc_entity_relation_episode(
                dataset_index=0, episode_index=0
            )
            self.assertEqual(
                payload["target_subject_view_centers"].shape, (1, 4, 2, 2)
            )
            self.assertTrue(payload["target_subject_view_visible"][0, 0, 0])

            incompatible = dict(index)
            incompatible["dataset_action_convention"] = (
                PGC_ACTION_CONVENTION_LIBERO_ENV
            )
            (root / "index.json").write_text(
                json.dumps(incompatible), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "action convention"):
                load_pgc_entity_relation_index(root)
            (root / "index.json").write_text(
                json.dumps(index), encoding="utf-8"
            )

            with episode_path.open("ab") as handle:
                handle.write(b"corruption")
            with self.assertRaisesRegex(ValueError, "sidecar hash changed"):
                load_pgc_entity_relation_index(root)


class PGCERAFModuleTest(unittest.TestCase):
    def test_training_metric_detach_preserves_only_scalar_diagnostics(self):
        metrics = detached_policy_guard_metrics(
            {
                "scalar_tensor": torch.tensor(2.5),
                "python_scalar": 3,
                "selected_clause": torch.tensor([0, 1, 2]),
                "desired_direction": torch.zeros(3, 3),
            }
        )
        self.assertEqual(metrics, {"scalar_tensor": 2.5, "python_scalar": 3.0})

    def test_v919_hard_route_is_zero_init_exact_and_clause_directional(self):
        servo = HardRoutedERAFPhaseServo(
            action_dim=7,
            proprio_dim=8,
            hidden_dim=8,
            max_clauses=2,
            max_abs=0.25,
        )
        candidate = torch.zeros(1, 3, 7)
        legacy = torch.randn_like(candidate) * 0.01
        outputs = {
            "active_logits": torch.full((1, 2), 10.0),
            "predicate_logits": torch.nn.functional.one_hot(
                torch.tensor([[1, 1]]), num_classes=11
            ).float(),
            "grasp_anchor": torch.zeros(1, 2, 3),
            "goal_anchor": torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]),
            "interaction_anchor": torch.zeros(1, 2, 3),
            "phase_logits": torch.tensor(
                [[[-10.0, 10.0, -10.0], [-10.0, 10.0, -10.0]]]
            ),
            "predicate_truth_logits": torch.full((1, 2), -10.0),
            "clause_execution_probability": torch.tensor([[0.1, 0.9]]),
        }
        action, residual, metrics = servo(
            candidate_action=candidate,
            legacy_residual=legacy,
            eraf_outputs=outputs,
            proprio=torch.zeros(1, 8),
        )
        self.assertTrue(torch.equal(residual, legacy))
        self.assertTrue(torch.equal(action, candidate + legacy))
        self.assertEqual(int(metrics["pgc_v919_selected_clause"].item()), 1)
        torch.testing.assert_close(
            metrics["pgc_v919_calibrated_eef_position"],
            torch.zeros(1, 3),
        )
        target = torch.zeros_like(action)
        target[..., 1] = 0.1
        (action - target).square().mean().backward()
        self.assertGreater(
            float(servo.translation_gain[-1].bias.grad.abs().item()), 0.0
        )
        servo.zero_grad(set_to_none=True)

        with torch.no_grad():
            servo.translation_gain[-1].bias.fill_(0.1)
        _, routed_y, _ = servo(
            candidate_action=candidate,
            legacy_residual=torch.zeros_like(legacy),
            eraf_outputs=outputs,
            proprio=torch.zeros(1, 8),
        )
        self.assertGreater(float(routed_y[..., 1].mean()), 0.05)
        self.assertAlmostEqual(float(routed_y[..., 0].abs().max()), 0.0)

        outputs_x = dict(outputs)
        outputs_x["clause_execution_probability"] = torch.tensor([[0.9, 0.1]])
        _, routed_x, metrics_x = servo(
            candidate_action=candidate,
            legacy_residual=torch.zeros_like(legacy),
            eraf_outputs=outputs_x,
            proprio=torch.zeros(1, 8),
        )
        self.assertGreater(float(routed_x[..., 0].mean()), 0.05)
        self.assertAlmostEqual(float(routed_x[..., 1].abs().max()), 0.0)
        self.assertEqual(int(metrics_x["pgc_v919_selected_clause"].item()), 0)

    def test_v920_waypoint_is_zero_init_exact_and_preserves_anchor_progress(self):
        adapter = PhaseCompatibleERAFWaypointAdapter(
            action_dim=7,
            hidden_dim=16,
            max_abs=0.25,
            tangent_max_ratio=0.75,
        )
        candidate = torch.randn(2, 3, 7) * 0.01
        legacy = torch.randn_like(candidate) * 0.01
        direction = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        action, residual, metrics = adapter(
            candidate_action=candidate,
            legacy_residual=legacy,
            inherited_servo_residual=torch.zeros_like(legacy),
            desired_direction=direction,
            control_phase=torch.tensor([0, 1]),
            route_confidence=torch.tensor([0.9, 0.8]),
        )
        self.assertTrue(torch.equal(residual, legacy))
        self.assertTrue(torch.equal(action, candidate + legacy))
        self.assertEqual(float(metrics["pgc_v920_translation_gain"]), 0.0)

        target = action.detach().clone()
        target[..., :3] += direction[:, None, :] * 0.1
        (action - target).square().mean().backward()
        self.assertGreater(float(adapter.gain_head.bias.grad.abs().item()), 0.0)

        adapter.zero_grad(set_to_none=True)
        with torch.no_grad():
            adapter.compatibility_head.bias.fill_(20.0)
            adapter.gain_head.bias.fill_(0.1)
            adapter.tangent_head.bias.zero_()
        tangent_action, _, _ = adapter(
            candidate_action=torch.zeros_like(candidate),
            legacy_residual=torch.zeros_like(legacy),
            inherited_servo_residual=torch.zeros_like(legacy),
            desired_direction=direction,
            control_phase=torch.tensor([0, 1]),
            route_confidence=torch.tensor([0.9, 0.8]),
        )
        orthogonal_target = torch.zeros_like(tangent_action)
        orthogonal_target[0, :, 1] = 0.1
        orthogonal_target[1, :, 0] = 0.1
        (tangent_action - orthogonal_target).square().mean().backward()
        self.assertGreater(
            float(adapter.tangent_head.bias.grad.norm().item()), 0.0
        )

        with torch.no_grad():
            adapter.gain_head.bias.fill_(0.1)
            adapter.tangent_head.bias.copy_(torch.tensor([0.0, 0.2, 0.1]))
        _, _, routed = adapter(
            candidate_action=candidate,
            legacy_residual=torch.zeros_like(legacy),
            inherited_servo_residual=torch.zeros_like(legacy),
            desired_direction=direction,
            control_phase=torch.tensor([0, 1]),
            route_confidence=torch.tensor([0.9, 0.8]),
        )
        local = routed["pgc_v920_local_direction"]
        progress = (local * direction[:, None, :]).sum(dim=-1)
        self.assertTrue(torch.all(progress > 0.0))
        tangent = local - progress.unsqueeze(-1) * direction[:, None, :]
        self.assertTrue(torch.all(tangent.norm(dim=-1) <= 0.75 + 1.0e-5))

        inherited = torch.zeros_like(legacy)
        inherited[..., 0] = 0.1
        with torch.no_grad():
            adapter.compatibility_head.bias.fill_(-20.0)
            adapter.gain_head.bias.zero_()
            adapter.tangent_head.bias.zero_()
        _, suppressed, suppressed_metrics = adapter(
            candidate_action=candidate,
            legacy_residual=torch.zeros_like(legacy),
            inherited_servo_residual=inherited,
            desired_direction=direction,
            control_phase=torch.tensor([0, 1]),
            route_confidence=torch.tensor([0.9, 0.8]),
        )
        self.assertLess(float(suppressed.abs().max()), 1.0e-6)
        self.assertLess(
            float(suppressed_metrics["pgc_v920_inherited_servo_retention"]),
            1.0e-6,
        )

    def test_v921_expert_residual_is_zero_init_exact_and_phase_specific(self):
        adapter = PhaseSpecificERAFExpertResidualAdapter(
            action_dim=7,
            hidden_dim=16,
            max_abs=0.25,
        )
        candidate = torch.randn(3, 4, 7) * 0.01
        current = torch.randn_like(candidate) * 0.01
        direction = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        action, residual, metrics = adapter(
            candidate_action=candidate,
            current_residual=current,
            desired_direction=direction,
            desired_distance=torch.tensor([0.1, 0.2, 0.3]),
            control_phase=torch.tensor([0, 1, 2]),
            route_confidence=torch.ones(3),
            waypoint_compatibility=torch.ones(3, 4),
        )
        self.assertTrue(torch.equal(residual, current))
        self.assertTrue(torch.equal(action, candidate + current))
        self.assertEqual(float(metrics["pgc_v921_expert_correction_rms"]), 0.0)

        target = action.detach().clone()
        target[..., 6] += 0.1
        (action - target).square().mean().backward()
        self.assertGreater(float(adapter.phase_output.bias.grad.norm()), 0.0)

        adapter.zero_grad(set_to_none=True)
        with torch.no_grad():
            adapter.phase_output.weight.zero_()
            adapter.phase_output.bias.zero_()
            adapter.phase_output.bias[0] = 0.4
            adapter.phase_output.bias[7 + 3] = 0.4
            adapter.phase_output.bias[14 + 6] = 0.4
        routed_action, _, routed_metrics = adapter(
            candidate_action=torch.zeros_like(candidate),
            current_residual=torch.zeros_like(current),
            desired_direction=direction,
            desired_distance=torch.tensor([0.1, 0.2, 0.3]),
            control_phase=torch.tensor([0, 1, 2]),
            route_confidence=torch.ones(3),
            waypoint_compatibility=torch.ones(3, 4),
        )
        self.assertGreater(float(routed_action[0, :, 0].mean()), 0.0)
        self.assertAlmostEqual(float(routed_action[0, :, 3].abs().max()), 0.0)
        self.assertGreater(float(routed_action[1, :, 3].mean()), 0.0)
        self.assertAlmostEqual(float(routed_action[1, :, 6].abs().max()), 0.0)
        self.assertGreater(float(routed_action[2, :, 6].mean()), 0.0)
        self.assertEqual(
            tuple(routed_metrics["pgc_v921_phase_residual_candidates"].shape),
            (3, 4, 3, 7),
        )

    def test_v922_clause_ranking_balances_final_actions_across_phases(self):
        model = tiny_pgc_fastwam(
            version=9,
            v9_stage="action",
            v9_grounding_objective_version=22,
            v9_completion_only_memory=True,
            v9_action_joint_training=True,
        )
        batch, horizon, action_dim = 3, 4, 3
        target = torch.zeros(batch, horizon, action_dim)
        correct = torch.zeros_like(target, requires_grad=True)
        wrong = torch.full_like(target, 0.25)
        labels = {
            "clause_valid": torch.tensor(
                [[True, True, False, False]] * batch
            ),
            "phase_valid": torch.tensor(
                [[True, False, False, False]] * batch
            ),
            "phase_ids": torch.tensor(
                [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]]
            ),
            "predicate_truth": torch.zeros(batch, 4),
            "predicate_truth_valid": torch.ones(
                batch, 4, dtype=torch.bool
            ),
        }
        waypoint = {"pgc_v919_selected_clause": torch.zeros(batch, dtype=torch.long)}
        loss, metrics = model._compute_policy_guard_v922_clause_action_ranking_loss(
            correct_action=correct,
            wrong_clause_action=wrong,
            target_action=target,
            action_is_pad=torch.zeros(batch, horizon, dtype=torch.bool),
            target_labels=labels,
            waypoint_metrics=waypoint,
            is_counterfactual=torch.ones(batch, dtype=torch.bool),
            direct_action_valid=torch.ones(batch, dtype=torch.bool),
            paired_language_valid=torch.ones(batch, dtype=torch.bool),
        )
        self.assertEqual(float(loss), 0.0)
        self.assertEqual(float(metrics["pgc_v922_active_phase_groups"]), 3.0)
        self.assertEqual(
            float(metrics["pgc_v922_clause_correct_action_win_rate"]), 1.0
        )

        bad_correct = torch.full_like(target, 0.2, requires_grad=True)
        bad_wrong = torch.zeros_like(target)
        bad_loss, bad_metrics = (
            model._compute_policy_guard_v922_clause_action_ranking_loss(
                correct_action=bad_correct,
                wrong_clause_action=bad_wrong,
                target_action=target,
                action_is_pad=None,
                target_labels=labels,
                waypoint_metrics=waypoint,
                is_counterfactual=torch.ones(batch, dtype=torch.bool),
                direct_action_valid=torch.ones(batch, dtype=torch.bool),
                paired_language_valid=torch.ones(batch, dtype=torch.bool),
            )
        )
        self.assertGreater(float(bad_loss), 0.0)
        self.assertEqual(
            float(bad_metrics["pgc_v922_clause_correct_action_win_rate"]), 0.0
        )
        bad_loss.backward()
        self.assertGreater(float(bad_correct.grad.abs().sum()), 0.0)

    def test_v918_phase_residual_imitation_prefers_expert_correction(self):
        model = tiny_pgc_fastwam(
            version=9,
            v9_stage="action",
            v9_grounding_objective_version=18,
            v9_completion_only_memory=True,
            v9_action_joint_training=True,
        )
        candidate = torch.zeros(3, 2, 3)
        expert = torch.tensor(
            [
                [[0.10, 0.00, 0.00], [0.08, 0.00, 0.00]],
                [[0.00, 0.12, 0.00], [0.00, 0.10, 0.00]],
                [[0.00, 0.00, 0.14], [0.00, 0.00, 0.12]],
            ]
        )
        labels = {
            "clause_valid": torch.tensor(
                [[True, False, False, False]] * 3
            ),
            "phase_valid": torch.tensor(
                [[True, False, False, False]] * 3
            ),
            "phase_ids": torch.tensor(
                [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]]
            ),
            "predicate_truth": torch.zeros(3, 4),
            "predicate_truth_valid": torch.ones(3, 4, dtype=torch.bool),
        }
        eraf_outputs = {
            "clause_execution_probability": torch.tensor(
                [[1.0, 0.0, 0.0, 0.0]] * 3
            )
        }
        common = {
            "pre_geometry_action": candidate,
            "target_action": expert,
            "action_is_pad": torch.zeros(3, 2, dtype=torch.bool),
            "target_labels": labels,
            "eraf_outputs": eraf_outputs,
            "is_counterfactual": torch.ones(3, dtype=torch.bool),
            "direct_action_valid": torch.ones(3, dtype=torch.bool),
            "paired_language_valid": torch.ones(3, dtype=torch.bool),
        }
        correct = expert.clone().requires_grad_(True)
        correct_loss, metrics = (
            model._compute_policy_guard_v918_phase_residual_loss(
                geometry_residual=correct,
                **common,
            )
        )
        zero_loss, _ = model._compute_policy_guard_v918_phase_residual_loss(
            geometry_residual=torch.zeros_like(expert),
            **common,
        )
        wrong_loss, _ = model._compute_policy_guard_v918_phase_residual_loss(
            geometry_residual=-expert,
            **common,
        )
        self.assertLess(float(correct_loss), float(zero_loss))
        self.assertLess(float(zero_loss), float(wrong_loss))
        self.assertAlmostEqual(
            float(metrics["pgc_v918_prefix_mse_improvement"]),
            float(expert.square().mean()),
            places=6,
        )
        self.assertAlmostEqual(
            float(metrics["pgc_v918_translation_direction_positive_rate"]),
            1.0,
            places=6,
        )
        for phase_name in ("approach", "transport", "release"):
            self.assertAlmostEqual(
                float(metrics[f"pgc_v918_{phase_name}_sample_fraction"]),
                1.0 / 3.0,
                places=6,
            )
        correct_loss.backward()
        self.assertIsNotNone(correct.grad)

    def setUp(self):
        torch.manual_seed(91)
        self.module = EntityRelationAffordanceField(
            text_dim=10,
            video_dim=16,
            action_dim=12,
            projection_dim=8,
            hidden_dim=8,
            num_heads=2,
            max_clauses=4,
            camera_count=2,
            visual_aspect_ratio=2.0,
        )

    def _forward(self):
        base_queries = torch.randn(2, 3, 12)
        base_embedding = torch.randn(2, 8)
        output = self.module(
            base_goal_queries=base_queries,
            base_goal_embedding=base_embedding,
            language_hidden=torch.randn(2, 6, 10),
            language_mask=torch.ones(2, 6, dtype=torch.bool),
            current_video_hidden=torch.randn(2, 8, 16),
        )
        return base_queries, base_embedding, output

    def test_zero_bridge_is_exact_and_grounding_loss_backpropagates(self):
        base_queries, base_embedding, (queries, embedding, outputs, _) = self._forward()
        self.assertTrue(torch.equal(queries, base_queries))
        self.assertTrue(torch.equal(embedding, base_embedding))
        self.assertEqual(outputs["subject_view_centers"].shape, (2, 4, 2, 2))
        self.assertEqual(
            outputs["subject_view_visibility_logits"].shape, (2, 4, 2)
        )
        self.assertTrue(
            torch.allclose(
                outputs["subject_view_attention_mass"].sum(dim=-1),
                torch.ones(2, 4),
            )
        )
        labels = {}
        labels["predicate_ids"] = torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0]])
        labels["clause_valid"] = torch.tensor([[True, False, False, False]] * 2)
        for role, entity_ids in (("subject", [11, 11]), ("reference", [21, 21])):
            masks = torch.zeros(2, 4, 8, 16)
            if role == "subject":
                masks[:, 0, :4, :4] = 1
            else:
                masks[:, 0, 4:, 12:] = 1
            labels[f"{role}_masks"] = masks
            labels[f"{role}_mask_valid"] = labels["clause_valid"].clone()
            labels[f"{role}_view_visible"] = torch.zeros(
                2, 4, 2, dtype=torch.bool
            )
            labels[f"{role}_view_visible"][:, 0] = True
            labels[f"{role}_view_centers"] = torch.zeros(2, 4, 2, 2)
            labels[f"{role}_positions"] = torch.zeros(2, 4, 3)
            labels[f"{role}_position_valid"] = labels["clause_valid"].clone()
            labels[f"{role}_entity_ids"] = torch.tensor(
                [[entity_ids[0], -1, -1, -1], [entity_ids[1], -1, -1, -1]]
            )

        for name in ("grasp", "goal", "interaction"):
            labels[f"{name}_anchors"] = torch.zeros(2, 4, 3)
            labels[f"{name}_anchor_valid"] = labels["clause_valid"].clone()
        labels["predicate_truth"] = torch.zeros(2, 4)
        labels["predicate_truth_valid"] = labels["clause_valid"].clone()
        labels["phase_ids"] = torch.zeros(2, 4, dtype=torch.long)
        labels["phase_valid"] = labels["clause_valid"].clone()
        loss, metrics = entity_relation_affordance_loss(
            outputs, labels, weights=ERAFLossWeights()
        )
        self.assertTrue(torch.isfinite(loss))
        for name in (
            "loss_pgc_v9_attention_mask",
            "loss_pgc_v9_spatial_bce",
            "loss_pgc_v9_spatial_dice",
            "loss_pgc_v9_role_overlap",
            "loss_pgc_v9_role_assignment",
            "pgc_v9_subject_gt_attention_mass",
            "pgc_v9_reference_gt_attention_mass",
            "pgc_v9_role_swap_accuracy",
            "pgc_v9_role_swap_valid_fraction",
            "pgc_v9_role_assignment_accuracy",
            "pgc_v9_role_assignment_hard_fraction",
        ):
            self.assertIn(name, metrics)
        loss.backward()
        self.assertIsNotNone(self.module.role_decoder.predicate_head.weight.grad)
        self.assertIsNotNone(self.module.entity_grounder.visual_projection[1].weight.grad)
        self.assertIsNotNone(
            self.module.entity_grounder.view_visibility_head[-1].weight.grad
        )

        # Unary fixture predicates intentionally use the same entity for both
        # roles.  They have no meaningful subject/reference swap negative.
        unary_labels = dict(labels)
        unary_labels["reference_entity_ids"] = labels[
            "subject_entity_ids"
        ].clone()
        _, unary_metrics = entity_relation_affordance_loss(
            outputs, unary_labels, weights=ERAFLossWeights()
        )
        self.assertEqual(unary_metrics["loss_pgc_v9_role_swap"].item(), 0.0)
        self.assertEqual(unary_metrics["loss_pgc_v9_role_overlap"].item(), 0.0)

    def test_oracle_route_replaces_grounding_and_keeps_learned_bridge(self):
        base_queries, base_embedding, (_, _, outputs, _) = self._forward()
        with torch.no_grad():
            self.module.query_delta_projection.weight.normal_(std=0.02)
            self.module.embedding_delta_projection.weight.normal_(std=0.02)
        clause_valid = torch.tensor(
            [[True, True, False, False], [True, False, False, False]]
        )
        subject_masks = torch.zeros(2, 4, 8, 16)
        reference_masks = torch.zeros_like(subject_masks)
        subject_masks[:, :, :4, :4] = 1
        reference_masks[:, :, 4:, 12:] = 1
        oracle = {
            "clause_valid": clause_valid,
            "predicate_ids": torch.tensor([[1, 2, 0, 0], [2, 0, 0, 0]]),
            "subject_masks": subject_masks,
            "reference_masks": reference_masks,
            "subject_mask_valid": clause_valid,
            "reference_mask_valid": clause_valid,
            "subject_positions": torch.full((2, 4, 3), -0.25),
            "reference_positions": torch.full((2, 4, 3), 0.25),
            "subject_position_valid": clause_valid,
            "reference_position_valid": clause_valid,
            "goal_anchors": torch.full((2, 4, 3), 0.5),
            "goal_anchor_valid": clause_valid,
            "predicate_truth": torch.tensor(
                [[True, False, False, False], [False, False, False, False]]
            ),
            "phase_ids": torch.tensor([[2, 0, 0, 0], [0, 0, 0, 0]]),
            "phase_valid": clause_valid,
        }
        with torch.no_grad():
            routed_queries, routed_embedding, routed = self.module.route_oracle(
                base_goal_queries=base_queries,
                base_goal_embedding=base_embedding,
                outputs=outputs,
                oracle=oracle,
            )
        self.assertEqual(
            routed["predicate_logits"].argmax(-1).tolist(),
            [[1, 2, 0, 0], [2, 0, 0, 0]],
        )
        self.assertEqual(
            routed["phase_logits"].argmax(-1).tolist(),
            [[2, 0, 0, 0], [0, 0, 0, 0]],
        )
        self.assertEqual(routed["oracle_selected_clause"].tolist(), [1, 0])
        self.assertEqual(
            routed["clause_routing_multiplier"].tolist(),
            [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        )
        torch.testing.assert_close(
            routed["subject_position"][clause_valid],
            oracle["subject_positions"][clause_valid],
        )
        torch.testing.assert_close(
            routed["goal_anchor"][clause_valid],
            oracle["goal_anchors"][clause_valid],
        )
        self.assertFalse(torch.equal(routed_queries, base_queries))
        self.assertFalse(torch.equal(routed_embedding, base_embedding))
        self.assertEqual(routed_queries.dtype, base_queries.dtype)
        self.assertEqual(routed_embedding.dtype, base_embedding.dtype)
        with self.assertRaisesRegex(RuntimeError, "evaluation-only"):
            self.module.route_oracle(
                base_goal_queries=base_queries,
                base_goal_embedding=base_embedding,
                outputs=outputs,
                oracle=oracle,
            )

    def test_causal_audit_bypass_is_exact_v5_goalgraph(self):
        base_queries, base_embedding, (_, _, outputs, _) = self._forward()
        with torch.no_grad():
            routed_queries, routed_embedding, routed = self.module.route_oracle(
                base_goal_queries=base_queries,
                base_goal_embedding=base_embedding,
                outputs=outputs,
                oracle={"_audit_bypass_bridge": True},
            )
        self.assertTrue(torch.equal(routed_queries, base_queries))
        self.assertTrue(torch.equal(routed_embedding, base_embedding))
        self.assertTrue(routed["oracle_eraf_enabled"].all())
        self.assertTrue(routed["audit_bypass_bridge"].all())
        self.assertEqual(routed["oracle_selected_clause"].tolist(), [-1, -1])

    def test_v99_view_fusion_and_scheduler_are_zero_init_and_trainable(self):
        torch.manual_seed(919)
        module = EntityRelationAffordanceField(
            text_dim=10,
            video_dim=16,
            action_dim=12,
            projection_dim=8,
            hidden_dim=8,
            num_heads=2,
            max_clauses=4,
            camera_count=2,
            visual_aspect_ratio=2.0,
            role_adapter_enabled=True,
            role_adapter_hidden_dim=8,
            role_adapter_teacher_enabled=True,
            balanced_role_adapter_enabled=True,
            balanced_role_adapter_hidden_dim=8,
            clause_activation_adapter_enabled=True,
            clause_activation_adapter_hidden_dim=8,
            view_fusion_enabled=True,
            view_fusion_adapter_hidden_dim=8,
            clause_scheduler_enabled=True,
            clause_scheduler_hidden_dim=8,
        )
        base_queries = torch.randn(2, 3, 12)
        base_embedding = torch.randn(2, 8)
        queries, embedding, outputs, _ = module(
            base_goal_queries=base_queries,
            base_goal_embedding=base_embedding,
            language_hidden=torch.randn(2, 6, 10),
            language_mask=torch.ones(2, 6, dtype=torch.bool),
            current_video_hidden=torch.randn(2, 8, 16),
        )
        self.assertTrue(torch.equal(queries, base_queries))
        self.assertTrue(torch.equal(embedding, base_embedding))
        self.assertTrue(
            torch.equal(
                outputs["subject_attention"], outputs["subject_base_attention"]
            )
        )
        self.assertTrue(
            torch.equal(
                outputs["reference_attention"],
                outputs["reference_base_attention"],
            )
        )
        self.assertEqual(
            int(outputs["subject_view_gate_residual_logits"].count_nonzero()), 0
        )
        self.assertEqual(int(outputs["clause_execution_logits"].count_nonzero()), 0)
        self.assertTrue(
            torch.equal(
                outputs["clause_routing_multiplier"],
                torch.ones_like(outputs["clause_routing_multiplier"]),
            )
        )
        self.assertTrue(bool(outputs["view_scheduler_enabled"].all()))
        repair_loss = (
            outputs["subject_view_gate_residual_logits"].sum()
            + outputs["reference_view_gate_residual_logits"].sum()
            + outputs["clause_execution_logits"].sum()
        )
        repair_loss.backward()
        self.assertIsNotNone(
            module.entity_grounder.view_fusion_adapter.output.weight.grad
        )
        self.assertIsNotNone(module.clause_execution_scheduler.output.weight.grad)

    def test_v912_zero_init_rebinding_is_exact_and_receives_state_gradients(self):
        torch.manual_seed(912)
        common = dict(
            text_dim=10,
            video_dim=16,
            action_dim=12,
            projection_dim=8,
            hidden_dim=8,
            num_heads=2,
            max_clauses=4,
            camera_count=2,
            visual_aspect_ratio=2.0,
            role_adapter_enabled=True,
            role_adapter_hidden_dim=8,
            role_adapter_teacher_enabled=True,
            balanced_role_adapter_enabled=True,
            balanced_role_adapter_hidden_dim=8,
            clause_activation_adapter_enabled=True,
            clause_activation_adapter_hidden_dim=8,
            view_fusion_enabled=True,
            view_fusion_adapter_hidden_dim=8,
            clause_scheduler_enabled=True,
            clause_scheduler_hidden_dim=8,
        )
        v911 = EntityRelationAffordanceField(**common).eval()
        v912 = EntityRelationAffordanceField(
            **common,
            closed_loop_rebinding_enabled=True,
            closed_loop_rebinding_hidden_dim=8,
        ).eval()
        incompatible = v912.load_state_dict(v911.state_dict(), strict=False)
        self.assertFalse(incompatible.unexpected_keys)
        self.assertTrue(incompatible.missing_keys)
        self.assertTrue(
            all(
                key.startswith("closed_loop_phase_rebinding_adapter.")
                for key in incompatible.missing_keys
            )
        )
        inputs = {
            "base_goal_queries": torch.randn(2, 3, 12),
            "base_goal_embedding": torch.randn(2, 8),
            "language_hidden": torch.randn(2, 6, 10),
            "language_mask": torch.ones(2, 6, dtype=torch.bool),
            "current_video_hidden": torch.randn(2, 8, 16),
        }
        output911 = v911(**inputs)
        output912 = v912(**inputs)
        torch.testing.assert_close(output912[0], output911[0], rtol=0.0, atol=0.0)
        torch.testing.assert_close(output912[1], output911[1], rtol=0.0, atol=0.0)
        for name in (
            "subject_attention",
            "reference_attention",
            "predicate_truth_logits",
            "phase_logits",
            "clause_execution_probability",
            "goal_anchor",
        ):
            torch.testing.assert_close(
                output912[2][name], output911[2][name], rtol=0.0, atol=0.0
            )
        for name in (
            "phase_rebinding_subject_delta",
            "phase_rebinding_reference_delta",
            "phase_rebinding_truth_residual",
            "phase_rebinding_phase_residual",
        ):
            self.assertEqual(int(output912[2][name].count_nonzero()), 0)

        v912.train()
        output = v912(**inputs)[2]
        repair_loss = (
            output["subject_position"].float().square().sum()
            + output["reference_position"].float().square().sum()
            + output["predicate_truth_logits"].float().sum()
            + output["phase_logits"].float().sum()
        )
        repair_loss.backward()
        adapter = v912.closed_loop_phase_rebinding_adapter
        self.assertIsNotNone(adapter)
        self.assertIsNotNone(adapter.subject_output.weight.grad)
        self.assertIsNotNone(adapter.reference_output.weight.grad)
        self.assertIsNotNone(adapter.truth_output.weight.grad)
        self.assertIsNotNone(adapter.phase_output.weight.grad)

    def test_v913_zero_init_memory_preserves_v911_geometry_and_routes_only(self):
        torch.manual_seed(913)
        common = dict(
            text_dim=10,
            video_dim=16,
            action_dim=12,
            projection_dim=8,
            hidden_dim=8,
            num_heads=2,
            max_clauses=4,
            camera_count=2,
            visual_aspect_ratio=2.0,
            role_adapter_enabled=True,
            role_adapter_hidden_dim=8,
            role_adapter_teacher_enabled=True,
            balanced_role_adapter_enabled=True,
            balanced_role_adapter_hidden_dim=8,
            clause_activation_adapter_enabled=True,
            clause_activation_adapter_hidden_dim=8,
            view_fusion_enabled=True,
            view_fusion_adapter_hidden_dim=8,
            clause_scheduler_enabled=True,
            clause_scheduler_hidden_dim=8,
        )
        v911 = EntityRelationAffordanceField(**common).eval()
        v913 = EntityRelationAffordanceField(
            **common,
            phase_safe_memory_enabled=True,
            phase_safe_memory_hidden_dim=8,
            phase_safe_memory_state_count=4,
        ).eval()
        incompatible = v913.load_state_dict(v911.state_dict(), strict=False)
        self.assertFalse(incompatible.unexpected_keys)
        self.assertTrue(incompatible.missing_keys)
        self.assertTrue(
            all(
                key.startswith("phase_safe_clause_memory.")
                for key in incompatible.missing_keys
            )
        )
        inputs = {
            "base_goal_queries": torch.randn(2, 3, 12),
            "base_goal_embedding": torch.randn(2, 8),
            "language_hidden": torch.randn(2, 6, 10),
            "language_mask": torch.ones(2, 6, dtype=torch.bool),
            "current_video_hidden": torch.randn(2, 8, 16),
            "policy_state": {
                "phase_safe_memory_state_ids": torch.zeros(2, 4, dtype=torch.long),
                "phase_safe_memory_valid": torch.ones(2, 4, dtype=torch.bool),
            },
            "proprio": torch.randn(2, 7),
        }
        base_inputs = {
            name: value
            for name, value in inputs.items()
            if name not in {"policy_state", "proprio"}
        }
        output911 = v911(**base_inputs)
        output913 = v913(**inputs)
        torch.testing.assert_close(output913[0], output911[0], rtol=0.0, atol=0.0)
        torch.testing.assert_close(output913[1], output911[1], rtol=0.0, atol=0.0)
        for name in (
            "subject_attention",
            "reference_attention",
            "subject_position",
            "reference_position",
            "predicate_truth_logits",
            "phase_logits",
            "goal_anchor",
        ):
            torch.testing.assert_close(
                output913[2][name], output911[2][name], rtol=0.0, atol=0.0
            )
        self.assertEqual(
            int(output913[2]["phase_safe_memory_routing_residual"].count_nonzero()),
            0,
        )
        self.assertEqual(
            float(output913[3]["pgc_v9_phase_safe_memory_geometry_max_abs"]),
            0.0,
        )

        v913.train()
        trained = v913(**inputs)[2]
        (
            trained["phase_safe_memory_state_logits"].sum()
            + trained["clause_execution_logits"].sum()
        ).backward()
        memory = v913.phase_safe_clause_memory
        self.assertIsNotNone(memory)
        self.assertIsNotNone(memory.state_output.weight.grad)
        self.assertIsNotNone(memory.scheduler_output.weight.grad)

    def test_v913_memory_enforces_completed_sticky_and_release_retry(self):
        memory = PhaseSafeClauseMemory(
            hidden_dim=8,
            adapter_hidden_dim=8,
            num_heads=2,
            max_clauses=2,
            phase_count=3,
            state_count=4,
        ).eval()
        common = {
            "clause_hidden": torch.randn(1, 2, 8),
            "subject_tokens": torch.randn(1, 2, 8),
            "reference_tokens": torch.randn(1, 2, 8),
            "active_logits": torch.full((1, 2), 10.0),
            "predicate_truth_logits": torch.full((1, 2), -10.0),
            "phase_logits": torch.tensor([[[10.0, -10.0, -10.0]] * 2]),
            "base_execution_logits": torch.zeros(1, 2),
            "base_execution_probability": torch.full((1, 2), 0.5),
            "base_routing_multiplier": torch.ones(1, 2),
            "policy_state": {
                "phase_safe_memory_state_ids": torch.tensor(
                    [[memory.HOLDING, memory.COMPLETED]]
                ),
                "phase_safe_memory_valid": torch.ones(1, 2, dtype=torch.bool),
            },
        }
        output = memory(**common)
        self.assertEqual(int(output["next_state_ids"][0, 0]), memory.RETRY)
        self.assertEqual(int(output["next_state_ids"][0, 1]), memory.COMPLETED)
        self.assertTrue(bool(output["released_unsatisfied_retry"][0, 0]))
        self.assertTrue(bool(output["completed_sticky"][0, 1]))

        common["active_logits"] = torch.tensor([[10.0, -10.0]])
        output = memory(**common)
        self.assertEqual(int(output["next_state_ids"][0, 1]), memory.COMPLETED)
        self.assertTrue(bool(output["next_state_valid"][0, 1]))
        self.assertTrue(bool(output["completed_sticky"][0, 1]))
        common["active_logits"] = torch.full((1, 2), 10.0)

        common["predicate_truth_logits"] = torch.tensor([[10.0, -10.0]])
        common["phase_logits"] = torch.tensor(
            [[[-10.0, 10.0, -10.0], [10.0, -10.0, -10.0]]]
        )
        output = memory(**common)
        self.assertNotEqual(
            int(output["next_state_ids"][0, 0]), memory.COMPLETED
        )

        with torch.no_grad():
            memory.state_output.bias[memory.COMPLETED] = 20.0
        output = memory(**common)
        self.assertNotEqual(
            int(output["next_state_ids"][0, 0]), memory.COMPLETED
        )

        common["phase_logits"] = torch.tensor(
            [[[-10.0, -10.0, 10.0], [10.0, -10.0, -10.0]]]
        )
        output = memory(**common)
        self.assertEqual(int(output["next_state_ids"][0, 0]), memory.COMPLETED)

    def test_v913_build_inputs_preserves_temporal_memory_labels(self):
        model = tiny_pgc_fastwam(
            version=9,
            v9_stage="grounding",
            v9_grounding_objective_version=14,
        )
        model._encode_input_image_latents_tensor = (
            lambda *_args, **_kwargs: torch.zeros(1, 2, 1, 16, 16)
        )
        batch_size = 2
        clause_count = 4
        sample = {
            "video": torch.randn(batch_size, 3, 5, 16, 16),
            "action": torch.randn(batch_size, 4, 3),
            "context": torch.randn(batch_size, 6, 10),
            "context_mask": torch.ones(batch_size, 6, dtype=torch.bool),
            "pgc_is_counterfactual": torch.tensor([False, True]),
            "pgc_direct_action_valid": torch.ones(batch_size, dtype=torch.bool),
            "pgc_goal_id": torch.tensor([1, 2]),
            "pgc_source_context": torch.randn(batch_size, 6, 10),
            "pgc_source_context_mask": torch.ones(
                batch_size, 6, dtype=torch.bool
            ),
            "pgc_source_goal_id": torch.tensor([3, 4]),
            "pgc_paired_language_valid": torch.ones(
                batch_size, dtype=torch.bool
            ),
        }
        for prefix in ("", "source_"):
            for name in PGC_ENTITY_RELATION_ARRAY_NAMES:
                sample[f"pgc_eraf_{prefix}{name}"] = torch.zeros(
                    batch_size, clause_count
                )
            sample[
                f"pgc_eraf_{prefix}phase_safe_memory_previous_state_ids"
            ] = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
            sample[
                f"pgc_eraf_{prefix}phase_safe_memory_target_state_ids"
            ] = torch.tensor([[1, 1, 2, 3], [3, 2, 2, 0]])
            sample[
                f"pgc_eraf_{prefix}phase_safe_memory_state_valid"
            ] = torch.ones(batch_size, clause_count, dtype=torch.bool)
            sample[
                f"pgc_eraf_{prefix}phase_safe_memory_execution_target"
            ] = torch.tensor([0, 2])
            sample[
                f"pgc_eraf_{prefix}phase_safe_memory_execution_valid"
            ] = torch.tensor([True, False])
            sample[f"pgc_eraf_{prefix}phase_safe_memory_stage_id"] = (
                torch.tensor([1, 2])
            )
            sample[f"pgc_eraf_{prefix}phase_safe_memory_stage_valid"] = (
                torch.ones(batch_size, dtype=torch.bool)
            )

        inputs = model.build_inputs(sample)
        for prefix in ("", "source_"):
            self.assertEqual(
                inputs[
                    f"pgc_eraf_{prefix}phase_safe_memory_previous_state_ids"
                ].dtype,
                torch.long,
            )
            self.assertEqual(
                tuple(
                    inputs[
                        f"pgc_eraf_{prefix}phase_safe_memory_previous_state_ids"
                    ].shape
                ),
                (batch_size, clause_count),
            )
            self.assertEqual(
                inputs[
                    f"pgc_eraf_{prefix}phase_safe_memory_execution_valid"
                ].dtype,
                torch.bool,
            )
            self.assertEqual(
                tuple(
                    inputs[
                        f"pgc_eraf_{prefix}phase_safe_memory_execution_target"
                    ].shape
                ),
                (batch_size,),
            )

    def test_v99_loss_supervises_camera_fusion_and_first_unfinished_clause(self):
        torch.manual_seed(920)
        module = EntityRelationAffordanceField(
            text_dim=10,
            video_dim=16,
            action_dim=12,
            projection_dim=8,
            hidden_dim=8,
            num_heads=2,
            max_clauses=4,
            camera_count=2,
            visual_aspect_ratio=2.0,
            role_adapter_enabled=True,
            role_adapter_hidden_dim=8,
            role_adapter_teacher_enabled=True,
            balanced_role_adapter_enabled=True,
            balanced_role_adapter_hidden_dim=8,
            clause_activation_adapter_enabled=True,
            clause_activation_adapter_hidden_dim=8,
            view_fusion_enabled=True,
            view_fusion_adapter_hidden_dim=8,
            clause_scheduler_enabled=True,
            clause_scheduler_hidden_dim=8,
        )
        _, _, outputs, _ = module(
            base_goal_queries=torch.randn(2, 3, 12),
            base_goal_embedding=torch.randn(2, 8),
            language_hidden=torch.randn(2, 6, 10),
            language_mask=torch.ones(2, 6, dtype=torch.bool),
            current_video_hidden=torch.randn(2, 8, 16),
        )
        clause_valid = torch.tensor(
            [[True, True, False, False], [True, True, False, False]]
        )
        labels = {
            "predicate_ids": torch.tensor([[1, 2, 0, 0], [1, 2, 0, 0]]),
            "clause_valid": clause_valid,
            "predicate_truth": torch.tensor(
                [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
            ),
            "predicate_truth_valid": clause_valid.clone(),
            "phase_ids": torch.zeros(2, 4, dtype=torch.long),
            "phase_valid": clause_valid.clone(),
        }
        for role, ids in (
            ("subject", (11, 12)),
            ("reference", (21, 22)),
        ):
            masks = torch.zeros(2, 4, 8, 16)
            masks[:, 0, 1:7, 1:7] = 1
            masks[:, 1, 1:7, 9:15] = 1
            labels[f"{role}_masks"] = masks
            labels[f"{role}_mask_valid"] = clause_valid.clone()
            view_visible = torch.zeros(2, 4, 2, dtype=torch.bool)
            view_visible[:, 0, 0] = True
            view_visible[:, 1, 1] = True
            labels[f"{role}_view_visible"] = view_visible
            labels[f"{role}_view_centers"] = torch.zeros(2, 4, 2, 2)
            labels[f"{role}_positions"] = torch.zeros(2, 4, 3)
            labels[f"{role}_position_valid"] = clause_valid.clone()
            labels[f"{role}_entity_ids"] = torch.tensor(
                [[ids[0], ids[1], -1, -1], [ids[0], ids[1], -1, -1]]
            )
        for name in ("grasp", "goal", "interaction"):
            labels[f"{name}_anchors"] = torch.zeros(2, 4, 3)
            labels[f"{name}_anchor_valid"] = clause_valid.clone()
        loss, metrics = entity_relation_affordance_loss(
            outputs,
            labels,
            weights=ERAFLossWeights(
                objective_version=10,
                mask=0.0,
                entity=0.0,
                relation=0.0,
                anchor=0.0,
                position=0.0,
                role_swap=0.0,
                view_fusion=1.0,
                clause_scheduler=1.0,
                phase=0.0,
            ),
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(metrics["loss_pgc_v9_view_fusion"]))
        self.assertTrue(torch.isfinite(metrics["loss_pgc_v9_clause_scheduler"]))
        self.assertEqual(metrics["pgc_v9_unfinished_clause_fraction"].item(), 1.0)
        loss.backward()
        self.assertIsNotNone(
            module.entity_grounder.view_fusion_adapter.output.weight.grad
        )
        self.assertIsNotNone(module.clause_execution_scheduler.output.weight.grad)

    def test_v9r3_zero_init_adapter_matches_teacher_and_isolates_gradients(self):
        torch.manual_seed(92)
        module = EntityRelationAffordanceField(
            text_dim=10,
            video_dim=16,
            action_dim=12,
            projection_dim=8,
            hidden_dim=8,
            num_heads=2,
            max_clauses=4,
            camera_count=2,
            visual_aspect_ratio=2.0,
            role_adapter_enabled=True,
            role_adapter_hidden_dim=8,
            role_adapter_teacher_enabled=True,
        )
        module.eval()
        module.requires_grad_(False)
        adapter = module.role_assignment_adapter
        self.assertIsNotNone(adapter)
        adapter.train()
        adapter.requires_grad_(True)
        _, _, outputs, metrics = module(
            base_goal_queries=torch.randn(2, 3, 12),
            base_goal_embedding=torch.randn(2, 8),
            language_hidden=torch.randn(2, 6, 10),
            language_mask=torch.ones(2, 6, dtype=torch.bool),
            current_video_hidden=torch.randn(2, 8, 16),
        )
        for role in ("subject", "reference"):
            self.assertTrue(
                torch.equal(
                    outputs[f"{role}_attention"],
                    outputs[f"teacher_{role}_attention"],
                )
            )
            self.assertTrue(
                torch.equal(
                    outputs[f"{role}_position"],
                    outputs[f"teacher_{role}_position"],
                )
            )
        for name in ("grasp", "goal", "interaction"):
            self.assertTrue(
                torch.equal(
                    outputs[f"{name}_anchor"],
                    outputs[f"teacher_{name}_anchor"],
                )
            )
        self.assertEqual(
            metrics["pgc_v9_role_adapter_subject_delta_norm"].item(), 0.0
        )
        self.assertEqual(
            metrics["pgc_v9_role_adapter_reference_delta_norm"].item(), 0.0
        )
        repair_loss = (
            outputs["subject_attention"][..., 0].pow(2).mean()
            + outputs["reference_attention"][..., -1].pow(2).mean()
        )
        repair_loss.backward()
        self.assertIsNotNone(adapter.subject_output.weight.grad)
        self.assertIsNotNone(adapter.reference_output.weight.grad)
        self.assertTrue(
            all(
                parameter.grad is None
                for name, parameter in module.named_parameters()
                if not name.startswith("role_assignment_adapter.")
            )
        )

    def test_v94_zero_init_structured_adapter_matches_v93_teacher(self):
        torch.manual_seed(93)
        module = EntityRelationAffordanceField(
            text_dim=10,
            video_dim=16,
            action_dim=12,
            projection_dim=8,
            hidden_dim=8,
            num_heads=2,
            max_clauses=4,
            camera_count=2,
            visual_aspect_ratio=2.0,
            role_adapter_enabled=True,
            role_adapter_hidden_dim=8,
            role_adapter_teacher_enabled=True,
            structured_role_adapter_enabled=True,
            structured_role_adapter_hidden_dim=8,
        )
        module.eval()
        module.requires_grad_(False)
        adapter = module.structured_role_assignment_adapter
        self.assertIsNotNone(adapter)
        adapter.train()
        adapter.requires_grad_(True)
        _, _, outputs, metrics = module(
            base_goal_queries=torch.randn(2, 3, 12),
            base_goal_embedding=torch.randn(2, 8),
            language_hidden=torch.randn(2, 6, 10),
            language_mask=torch.ones(2, 6, dtype=torch.bool),
            current_video_hidden=torch.randn(2, 8, 16),
        )
        for role in ("subject", "reference"):
            self.assertTrue(
                torch.equal(
                    outputs[f"{role}_attention"],
                    outputs[f"teacher_{role}_attention"],
                )
            )
        for name in ("grasp", "goal", "interaction"):
            self.assertTrue(
                torch.equal(
                    outputs[f"{name}_anchor"],
                    outputs[f"teacher_{name}_anchor"],
                )
            )
        self.assertEqual(
            metrics[
                "pgc_v9_structured_role_adapter_subject_delta_norm"
            ].item(),
            0.0,
        )
        repair_loss = outputs["subject_attention"][..., 0].pow(2).mean()
        repair_loss.backward()
        self.assertIsNotNone(adapter.subject_output.weight.grad)
        self.assertTrue(
            all(
                parameter.grad is None
                for name, parameter in module.named_parameters()
                if not name.startswith("structured_role_assignment_adapter.")
            )
        )

    def test_v94_cross_clause_assignment_uses_all_different_entities(self):
        subject_attention = torch.tensor(
            [[[0.40, 0.50, 0.10, 0.00], [0.02, 0.90, 0.08, 0.00]]],
            requires_grad=True,
        )
        reference_attention = torch.tensor(
            [[[0.05, 0.05, 0.90, 0.00], [0.05, 0.05, 0.90, 0.00]]],
            requires_grad=True,
        )
        subject_targets = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]
        )
        # Both clauses legally share the same basket/reference entity.
        reference_targets = torch.tensor(
            [[[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0]]]
        )
        loss, multi_loss, metrics = _structured_all_entity_assignment_loss(
            role_attentions={
                "subject": subject_attention,
                "reference": reference_attention,
            },
            role_targets={
                "subject": subject_targets,
                "reference": reference_targets,
            },
            role_entity_ids={
                "subject": torch.tensor([[1, 2]]),
                "reference": torch.tensor([[3, 3]]),
            },
            role_valid={
                "subject": torch.ones(1, 2, dtype=torch.bool),
                "reference": torch.ones(1, 2, dtype=torch.bool),
            },
            clause_valid=torch.ones(1, 2, dtype=torch.bool),
            temperature=0.10,
            hard_weight=2.0,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(multi_loss))
        self.assertAlmostEqual(metrics["accuracy"].item(), 0.75)
        self.assertEqual(metrics["multi_clause_accuracy"].item(), 0.0)
        # Shared basket slots are positives/exclusions, never false negatives.
        self.assertAlmostEqual(metrics["negative_count"].item(), 2.5)
        (loss + multi_loss).backward()
        self.assertIsNotNone(subject_attention.grad)

    def test_v95_bipartite_assignment_balances_rare_hard_rows(self):
        subject_attention = torch.tensor(
            [[[0.40, 0.50, 0.10, 0.00], [0.02, 0.90, 0.08, 0.00]]],
            requires_grad=True,
        )
        reference_attention = torch.tensor(
            [[[0.05, 0.05, 0.90, 0.00], [0.05, 0.05, 0.90, 0.00]]],
            requires_grad=True,
        )
        subject_targets = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]
        )
        reference_targets = torch.tensor(
            [[[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0]]]
        )
        loss, multi_loss, metrics = _balanced_bipartite_assignment_loss(
            role_attentions={
                "subject": subject_attention,
                "reference": reference_attention,
            },
            role_targets={
                "subject": subject_targets,
                "reference": reference_targets,
            },
            role_entity_ids={
                "subject": torch.tensor([[1, 2]]),
                "reference": torch.tensor([[3, 3]]),
            },
            role_valid={
                "subject": torch.ones(1, 2, dtype=torch.bool),
                "reference": torch.ones(1, 2, dtype=torch.bool),
            },
            clause_valid=torch.ones(1, 2, dtype=torch.bool),
            temperature=0.10,
            hard_group_weight=1.0,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(multi_loss))
        self.assertAlmostEqual(metrics["accuracy"].item(), 0.75)
        self.assertAlmostEqual(metrics["hard_gradient_fraction"].item(), 0.5)
        self.assertEqual(metrics["multi_clause_accuracy"].item(), 0.0)
        (loss + multi_loss).backward()
        self.assertIsNotNone(subject_attention.grad)
        self.assertIsNotNone(reference_attention.grad)

    def test_v96_empty_assignment_group_is_finite_and_differentiable(self):
        subject_attention = torch.tensor(
            [[[0.8, 0.2]]], requires_grad=True
        )
        reference_attention = torch.tensor(
            [[[0.2, 0.8]]], requires_grad=True
        )
        # Unary clauses have no different-entity negative, so both assignment
        # groups are globally empty. V9.5's values.sum()*0 path could inherit
        # NaN from the masked logsumexp; V9.6 must return an exact finite zero.
        target = torch.tensor([[[1.0, 0.0]]])
        loss, multi_loss, metrics = _balanced_bipartite_assignment_loss(
            role_attentions={
                "subject": subject_attention,
                "reference": reference_attention,
            },
            role_targets={"subject": target, "reference": target},
            role_entity_ids={
                "subject": torch.tensor([[1]]),
                "reference": torch.tensor([[1]]),
            },
            role_valid={
                "subject": torch.ones(1, 1, dtype=torch.bool),
                "reference": torch.ones(1, 1, dtype=torch.bool),
            },
            clause_valid=torch.ones(1, 1, dtype=torch.bool),
            temperature=0.10,
            hard_group_weight=1.0,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(multi_loss))
        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(metrics["global_easy_count"].item(), 0.0)
        self.assertEqual(metrics["global_hard_count"].item(), 0.0)
        (loss + multi_loss).backward()
        self.assertIsNotNone(subject_attention.grad)

    def test_v97_exclusive_assignment_ignores_shared_role_support(self):
        subject_target = torch.tensor([[[1.0, 1.0, 0.0, 0.0]]])
        reference_target = torch.tensor([[[0.0, 1.0, 1.0, 0.0]]])
        common = {
            "role_targets": {
                "subject": subject_target,
                "reference": reference_target,
            },
            "role_entity_ids": {
                "subject": torch.tensor([[1]]),
                "reference": torch.tensor([[2]]),
            },
            "role_valid": {
                "subject": torch.ones(1, 1, dtype=torch.bool),
                "reference": torch.ones(1, 1, dtype=torch.bool),
            },
            "clause_valid": torch.ones(1, 1, dtype=torch.bool),
            "temperature": 0.10,
            "hard_group_weight": 1.0,
        }
        good_loss, _, good_metrics = _balanced_exclusive_role_assignment_loss(
            role_attentions={
                "subject": torch.tensor([[[0.40, 0.50, 0.10, 0.0]]]),
                "reference": torch.tensor([[[0.10, 0.50, 0.40, 0.0]]]),
            },
            **common,
        )
        bad_loss, _, bad_metrics = _balanced_exclusive_role_assignment_loss(
            role_attentions={
                "subject": torch.tensor([[[0.10, 0.50, 0.40, 0.0]]]),
                "reference": torch.tensor([[[0.40, 0.50, 0.10, 0.0]]]),
            },
            **common,
        )
        self.assertLess(good_loss.item(), bad_loss.item())
        self.assertEqual(good_metrics["accuracy"].item(), 1.0)
        self.assertEqual(bad_metrics["accuracy"].item(), 0.0)
        self.assertEqual(good_metrics["exclusive_coverage"].item(), 1.0)

        ambiguous_loss, _, ambiguous_metrics = (
            _balanced_exclusive_role_assignment_loss(
                role_attentions={
                    "subject": torch.tensor([[[0.5, 0.5]]], requires_grad=True),
                    "reference": torch.tensor([[[0.5, 0.5]]], requires_grad=True),
                },
                role_targets={
                    "subject": torch.tensor([[[1.0, 1.0]]]),
                    "reference": torch.tensor([[[1.0, 1.0]]]),
                },
                role_entity_ids={
                    "subject": torch.tensor([[1]]),
                    "reference": torch.tensor([[2]]),
                },
                role_valid=common["role_valid"],
                clause_valid=common["clause_valid"],
                temperature=0.10,
                hard_group_weight=1.0,
            )
        )
        self.assertTrue(torch.isfinite(ambiguous_loss))
        self.assertEqual(ambiguous_loss.item(), 0.0)
        self.assertEqual(ambiguous_metrics["exclusive_coverage"].item(), 0.0)

    def test_v910_exclusive_all_entity_rejects_cross_clause_entity_swap(self):
        clause_valid = torch.ones(1, 2, dtype=torch.bool)
        role_targets = {
            "subject": torch.tensor(
                [[[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]]
            ),
            "reference": torch.tensor(
                [[[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]
            ),
        }
        common = {
            "role_targets": role_targets,
            "role_entity_ids": {
                "subject": torch.tensor([[1, 3]]),
                "reference": torch.tensor([[2, 2]]),
            },
            "role_valid": {
                "subject": clause_valid.clone(),
                "reference": clause_valid.clone(),
            },
            "clause_valid": clause_valid,
            "temperature": 0.10,
            "hard_group_weight": 1.0,
        }
        # Clause 0 still beats its own reference (0.20 > 0.10), so the old
        # pairwise objective accepts it despite selecting clause 1's subject.
        cross_clause_swap = {
            "subject": torch.tensor(
                [[[0.20, 0.10, 0.70, 0.0], [0.05, 0.05, 0.90, 0.0]]]
            ),
            "reference": torch.tensor(
                [[[0.05, 0.90, 0.05, 0.0], [0.05, 0.90, 0.05, 0.0]]]
            ),
        }
        pairwise_loss, _, pairwise_metrics = (
            _balanced_exclusive_role_assignment_loss(
                role_attentions=cross_clause_swap,
                **common,
            )
        )
        bad_loss, _, bad_metrics = (
            _balanced_exclusive_all_entity_assignment_loss(
                role_attentions=cross_clause_swap,
                **common,
            )
        )
        corrected = {
            "subject": torch.tensor(
                [[[0.80, 0.05, 0.15, 0.0], [0.05, 0.05, 0.90, 0.0]]]
            ),
            "reference": cross_clause_swap["reference"],
        }
        good_loss, _, good_metrics = (
            _balanced_exclusive_all_entity_assignment_loss(
                role_attentions=corrected,
                **common,
            )
        )
        self.assertTrue(torch.isfinite(pairwise_loss))
        self.assertEqual(pairwise_metrics["accuracy"].item(), 1.0)
        self.assertEqual(bad_metrics["accuracy"].item(), 0.75)
        self.assertEqual(good_metrics["accuracy"].item(), 1.0)
        self.assertLess(good_loss.item(), bad_loss.item())
        self.assertEqual(good_metrics["exclusive_coverage"].item(), 1.0)

        differentiable = {
            role: value.clone().requires_grad_(True)
            for role, value in corrected.items()
        }
        gradient_loss, gradient_multi_loss, _ = (
            _balanced_exclusive_all_entity_assignment_loss(
                role_attentions=differentiable,
                **common,
            )
        )
        (gradient_loss + gradient_multi_loss).backward()
        for attention in differentiable.values():
            self.assertIsNotNone(attention.grad)
            self.assertTrue(torch.isfinite(attention.grad).all())

    def test_v911_clause_tuple_rejects_shared_reference_subject_swap(self):
        clause_valid = torch.ones(1, 3, dtype=torch.bool)
        role_targets = {
            "subject": torch.tensor(
                [
                    [
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    ]
                ]
            ),
            "reference": torch.tensor(
                [
                    [
                        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    ]
                ]
            ),
        }
        common = {
            "role_targets": role_targets,
            "role_entity_ids": {
                "subject": torch.tensor([[1, 3, 4]]),
                "reference": torch.tensor([[2, 2, 5]]),
            },
            "role_valid": {
                "subject": clause_valid.clone(),
                "reference": clause_valid.clone(),
            },
            "predicate_logits": torch.tensor(
                [[[0.0, 8.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]]]
            ),
            "predicate_ids": torch.tensor([[1, 1, 2]]),
            "clause_valid": clause_valid,
            "temperature": 0.10,
            "hard_group_weight": 1.0,
        }
        corrected = {
            "subject": torch.tensor(
                [
                    [
                        [0.90, 0.02, 0.03, 0.03, 0.01, 0.01],
                        [0.03, 0.02, 0.90, 0.03, 0.01, 0.01],
                        [0.03, 0.02, 0.03, 0.90, 0.01, 0.01],
                    ]
                ]
            ),
            "reference": torch.tensor(
                [
                    [
                        [0.02, 0.90, 0.02, 0.02, 0.02, 0.02],
                        [0.02, 0.90, 0.02, 0.02, 0.02, 0.02],
                        [0.02, 0.02, 0.02, 0.02, 0.90, 0.02],
                    ]
                ]
            ),
        }
        swapped = {
            "subject": corrected["subject"].clone(),
            "reference": corrected["reference"],
        }
        swapped["subject"][:, [0, 1]] = swapped["subject"][:, [1, 0]]
        good_loss, _, good_metrics = _balanced_clause_tuple_assignment_loss(
            role_attentions=corrected,
            **common,
        )
        bad_loss, _, bad_metrics = _balanced_clause_tuple_assignment_loss(
            role_attentions=swapped,
            **common,
        )
        self.assertEqual(good_metrics["accuracy"].item(), 1.0)
        self.assertLess(bad_metrics["accuracy"].item(), 1.0)
        self.assertLess(good_loss.item(), bad_loss.item())

        differentiable = {
            role: value.clone().requires_grad_(True)
            for role, value in corrected.items()
        }
        predicate_logits = common["predicate_logits"].clone().requires_grad_(True)
        gradient_loss, gradient_multi_loss, _ = (
            _balanced_clause_tuple_assignment_loss(
                role_attentions=differentiable,
                **{**common, "predicate_logits": predicate_logits},
            )
        )
        (gradient_loss + gradient_multi_loss).backward()
        for value in (*differentiable.values(), predicate_logits):
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all())

    def test_v98_zero_init_clause_adapter_preserves_active_logits(self):
        torch.manual_seed(98)
        adapter = ClauseActivationCalibrationAdapter(
            hidden_dim=8,
            adapter_hidden_dim=8,
            num_heads=2,
            max_clauses=4,
            residual_max_abs=4.0,
        )
        clause_hidden = torch.randn(3, 4, 8)
        active_logits = torch.randn(3, 4)
        outputs = adapter(
            clause_hidden=clause_hidden,
            active_logits=active_logits,
        )
        self.assertTrue(torch.equal(outputs["active_logits"], active_logits))
        self.assertEqual(tuple(outputs["cardinality_logits"].shape), (3, 5))
        self.assertEqual(outputs["active_residual"].abs().max().item(), 0.0)
        (
            outputs["active_logits"].sum()
            + outputs["cardinality_logits"].sum()
        ).backward()
        self.assertIsNotNone(adapter.active_output.weight.grad)
        self.assertIsNotNone(adapter.cardinality_output.weight.grad)

    def test_v98_clause_objective_targets_exact_multi_clause_activity(self):
        clause_valid = torch.tensor(
            [[True, True, False, False], [True, False, False, False]]
        )
        good_active = torch.tensor(
            [[5.0, 5.0, -5.0, -5.0], [5.0, -5.0, -5.0, -5.0]],
            requires_grad=True,
        )
        bad_active = torch.tensor(
            [[5.0, -5.0, 5.0, -5.0], [-5.0, 5.0, -5.0, -5.0]],
            requires_grad=True,
        )
        good_cardinality = torch.full((2, 5), -5.0)
        good_cardinality[0, 2] = 5.0
        good_cardinality[1, 1] = 5.0
        bad_cardinality = -good_cardinality
        good = _clause_activation_calibration_loss(
            active_logits=good_active,
            cardinality_logits=good_cardinality,
            clause_valid=clause_valid,
            multi_group_weight=1.0,
        )
        bad = _clause_activation_calibration_loss(
            active_logits=bad_active,
            cardinality_logits=bad_cardinality,
            clause_valid=clause_valid,
            multi_group_weight=1.0,
        )
        good_total = good[0] + good[1] + good[2]
        bad_total = bad[0] + bad[1] + bad[2]
        self.assertLess(good_total.item(), bad_total.item())
        self.assertEqual(good[3]["exact"].item(), 1.0)
        self.assertEqual(good[3]["multi_exact"].item(), 1.0)
        self.assertEqual(good[3]["cardinality_accuracy"].item(), 1.0)
        self.assertEqual(good[3]["multi_gradient_fraction"].item(), 0.5)
        bad_total.backward()
        self.assertIsNotNone(bad_active.grad)

    def test_v95_zero_init_visual_binding_adapter_matches_v93_teacher(self):
        torch.manual_seed(95)
        module = EntityRelationAffordanceField(
            text_dim=10,
            video_dim=16,
            action_dim=12,
            projection_dim=8,
            hidden_dim=8,
            num_heads=2,
            max_clauses=4,
            camera_count=2,
            visual_aspect_ratio=2.0,
            role_adapter_enabled=True,
            role_adapter_hidden_dim=8,
            role_adapter_teacher_enabled=True,
            balanced_role_adapter_enabled=True,
            balanced_role_adapter_hidden_dim=8,
        )
        module.eval()
        module.requires_grad_(False)
        adapter = module.balanced_role_binding_adapter
        self.assertIsNotNone(adapter)
        adapter.train()
        adapter.requires_grad_(True)
        _, _, outputs, metrics = module(
            base_goal_queries=torch.randn(2, 3, 12),
            base_goal_embedding=torch.randn(2, 8),
            language_hidden=torch.randn(2, 6, 10),
            language_mask=torch.ones(2, 6, dtype=torch.bool),
            current_video_hidden=torch.randn(2, 8, 16),
        )
        for role in ("subject", "reference"):
            self.assertTrue(
                torch.equal(
                    outputs[f"{role}_attention"],
                    outputs[f"teacher_{role}_attention"],
                )
            )
        for name in ("grasp", "goal", "interaction"):
            self.assertTrue(
                torch.equal(
                    outputs[f"{name}_anchor"],
                    outputs[f"teacher_{name}_anchor"],
                )
            )
        self.assertEqual(
            metrics["pgc_v9_balanced_role_adapter_subject_delta_norm"].item(),
            0.0,
        )
        repair_loss = outputs["subject_attention"][..., 0].pow(2).mean()
        repair_loss.backward()
        self.assertIsNotNone(adapter.subject_output.weight.grad)
        self.assertTrue(
            all(
                parameter.grad is None
                for name, parameter in module.named_parameters()
                if not name.startswith("balanced_role_binding_adapter.")
            )
        )

    def test_gate_aligned_attention_loss_rejects_role_swaps(self):
        def outputs(subject_logits, reference_logits):
            subject_logits = subject_logits.reshape(1, 1, 4).requires_grad_()
            reference_logits = reference_logits.reshape(1, 1, 4).requires_grad_()
            return {
                "active_logits": torch.full((1, 1), 5.0),
                "predicate_logits": torch.nn.functional.one_hot(
                    torch.ones(1, 1, dtype=torch.long), num_classes=11
                ).float()
                * 5.0,
                "subject_similarity": subject_logits,
                "reference_similarity": reference_logits,
                "subject_attention": subject_logits.softmax(dim=-1),
                "reference_attention": reference_logits.softmax(dim=-1),
                "subject_visibility_logits": torch.full((1, 1), 5.0),
                "reference_visibility_logits": torch.full((1, 1), 5.0),
                "subject_view_visibility_logits": torch.full((1, 1, 2), 5.0),
                "reference_view_visibility_logits": torch.full((1, 1, 2), 5.0),
                "subject_view_centers": torch.zeros(1, 1, 2, 2),
                "reference_view_centers": torch.zeros(1, 1, 2, 2),
                "subject_position": torch.zeros(1, 1, 3),
                "reference_position": torch.zeros(1, 1, 3),
                "subject_queries": torch.zeros(1, 1, 4),
                "reference_queries": torch.zeros(1, 1, 4),
                "subject_token": torch.zeros(1, 1, 4),
                "reference_token": torch.zeros(1, 1, 4),
                "predicate_truth_logits": torch.zeros(1, 1),
                "phase_logits": torch.zeros(1, 1, 3),
                "grasp_anchor": torch.zeros(1, 1, 3),
                "goal_anchor": torch.zeros(1, 1, 3),
                "interaction_anchor": torch.zeros(1, 1, 3),
            }

        labels = {
            "predicate_ids": torch.ones(1, 1, dtype=torch.long),
            "clause_valid": torch.ones(1, 1, dtype=torch.bool),
            "predicate_truth": torch.zeros(1, 1),
            "predicate_truth_valid": torch.ones(1, 1, dtype=torch.bool),
            "phase_ids": torch.zeros(1, 1, dtype=torch.long),
            "phase_valid": torch.ones(1, 1, dtype=torch.bool),
        }
        for role, entity_id in (("subject", 11), ("reference", 21)):
            mask = torch.zeros(1, 1, 2, 4)
            if role == "subject":
                mask[..., :2] = 1.0
            else:
                mask[..., 2:] = 1.0
            labels[f"{role}_masks"] = mask
            labels[f"{role}_mask_valid"] = torch.ones(1, 1, dtype=torch.bool)
            labels[f"{role}_view_visible"] = torch.ones(
                1, 1, 2, dtype=torch.bool
            )
            labels[f"{role}_view_centers"] = torch.zeros(1, 1, 2, 2)
            labels[f"{role}_positions"] = torch.zeros(1, 1, 3)
            labels[f"{role}_position_valid"] = torch.ones(
                1, 1, dtype=torch.bool
            )
            labels[f"{role}_entity_ids"] = torch.full(
                (1, 1), entity_id, dtype=torch.long
            )
        for name in ("grasp", "goal", "interaction"):
            labels[f"{name}_anchors"] = torch.zeros(1, 1, 3)
            labels[f"{name}_anchor_valid"] = torch.ones(
                1, 1, dtype=torch.bool
            )

        weights = ERAFLossWeights(
            objective_version=3,
            mask=0.0,
            attention_mask=2.0,
            entity=0.0,
            relation=0.0,
            anchor=0.0,
            position=0.0,
            role_swap=2.0,
            role_overlap=1.0,
            role_swap_margin=0.20,
            role_assignment=4.0,
            role_assignment_temperature=0.10,
            role_assignment_hard_weight=2.0,
            phase=0.0,
        )
        correct_outputs = outputs(
            torch.tensor([5.0, 5.0, -5.0, -5.0]),
            torch.tensor([-5.0, -5.0, 5.0, 5.0]),
        )
        swapped_outputs = outputs(
            torch.tensor([-5.0, -5.0, 5.0, 5.0]),
            torch.tensor([5.0, 5.0, -5.0, -5.0]),
        )
        correct_loss, correct_metrics = entity_relation_affordance_loss(
            correct_outputs, labels, weights=weights
        )
        swapped_loss, swapped_metrics = entity_relation_affordance_loss(
            swapped_outputs, labels, weights=weights
        )
        self.assertLess(correct_loss.item(), swapped_loss.item())
        self.assertEqual(correct_metrics["pgc_v9_role_swap_accuracy"].item(), 1.0)
        self.assertEqual(swapped_metrics["pgc_v9_role_swap_accuracy"].item(), 0.0)
        self.assertEqual(
            correct_metrics["pgc_v9_role_assignment_accuracy"].item(), 1.0
        )
        self.assertEqual(
            swapped_metrics["pgc_v9_role_assignment_accuracy"].item(), 0.0
        )
        self.assertLess(
            correct_metrics["loss_pgc_v9_role_assignment"].item(),
            swapped_metrics["loss_pgc_v9_role_assignment"].item(),
        )
        self.assertLess(
            correct_metrics["loss_pgc_v9_attention_mask"].item(),
            swapped_metrics["loss_pgc_v9_attention_mask"].item(),
        )
        swapped_loss.backward()
        subject_gradient = swapped_outputs["subject_similarity"].grad
        reference_gradient = swapped_outputs["reference_similarity"].grad
        self.assertLess(subject_gradient[..., :2].mean().item(), 0.0)
        self.assertGreater(subject_gradient[..., 2:].mean().item(), 0.0)
        self.assertGreater(reference_gradient[..., :2].mean().item(), 0.0)
        self.assertLess(reference_gradient[..., 2:].mean().item(), 0.0)

    def test_wrong_entity_candidate_uses_same_state_entity_or_fallback_patch(self):
        base_queries, base_embedding, (_, _, outputs, _) = self._forward()
        wrong_indices = outputs["subject_attention"].argmin(dim=-1)
        fallback = torch.gather(
            outputs["visual_tokens"].unsqueeze(1).expand(
                -1, wrong_indices.shape[1], -1, -1
            ),
            2,
            wrong_indices.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, -1, 1, outputs["visual_tokens"].shape[-1]),
        ).squeeze(2)
        predicted_ids = outputs["predicate_logits"].argmax(dim=-1)
        is_binary = (predicted_ids >= 1) & (predicted_ids <= 6)
        expected = torch.where(
            is_binary.unsqueeze(-1), outputs["reference_token"], fallback
        )
        captured = {}

        def capture_subject(_module, inputs):
            captured["subject"] = inputs[0].detach().clone()

        handle = self.module.entity_grounder.position_head.register_forward_pre_hook(
            capture_subject
        )
        try:
            self.module.negative_goal_queries(
                base_goal_queries=base_queries,
                base_goal_embedding=base_embedding,
                outputs=outputs,
                kind="entity",
            )
        finally:
            handle.remove()
        self.assertTrue(torch.equal(captured["subject"], expected))

    def test_structural_ablation_paths_are_distinct(self):
        entity_only = EntityRelationAffordanceField(
            text_dim=10, video_dim=16, action_dim=12, projection_dim=8,
            hidden_dim=8, num_heads=2, entity_only=True, use_anchors=False,
        )
        self.assertTrue(entity_only.entity_only)
        self.assertFalse(entity_only.use_anchors)
        no_anchor = EntityRelationAffordanceField(
            text_dim=10, video_dim=16, action_dim=12, projection_dim=8,
            hidden_dim=8, num_heads=2, entity_only=False, use_anchors=False,
        )
        self.assertFalse(no_anchor.entity_only)
        self.assertFalse(no_anchor.use_anchors)


class PGCERAFIntegrationTest(unittest.TestCase):
    def test_v5_migration_is_exact_frozen_and_round_trips(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_path = root / "base.pt"
            v5_path = root / "v5.pt"
            v9_path = root / "v9.pt"
            v5 = tiny_pgc_fastwam(version=5)
            torch.save({"format": "fastwam_full_v1", "mot": v5.mot.state_dict()}, base_path)
            v5.load_checkpoint(base_path)
            v5.save_checkpoint(v5_path, step=4000)
            v9 = tiny_pgc_fastwam(version=9, v9_stage="grounding")
            v9.load_checkpoint(v5_path)
            v9.eval()
            final_video = torch.randn(2, 8, 16)
            neutral_video = torch.randn(2, 8, 16)
            context = torch.randn(2, 5, 10)
            context_mask = torch.ones(2, 5, dtype=torch.bool)
            with torch.no_grad():
                v5_queries, v5_embedding, _ = v5._encode_policy_guard_goal(
                    final_video_hidden=final_video,
                    video_tokens_per_frame=8,
                    context=context,
                    context_mask=context_mask,
                )
                v9_queries, v9_embedding, _, _ = v9._encode_policy_guard_eraf(
                    final_video_hidden=final_video,
                    current_visual_hidden=neutral_video,
                    video_tokens_per_frame=8,
                    context=context,
                    context_mask=context_mask,
                    language_context_len=5,
                )
            self.assertTrue(torch.equal(v9_queries, v5_queries))
            self.assertTrue(torch.equal(v9_embedding, v5_embedding))
            v9.prepare_trainable_parameters()
            trainable = {
                name
                for name, parameter in v9.named_parameters()
                if parameter.requires_grad
            }
            self.assertTrue(trainable)
            self.assertTrue(
                all(
                    name.startswith(
                        "policy_guard_modules.entity_relation_affordance."
                    )
                    for name in trainable
                )
            )
            self.assertFalse(
                any(
                    parameter.requires_grad
                    for parameter in v9.mot.parameters()
                )
            )
            self.assertFalse(v9.lora_enabled)
            v9.save_checkpoint(v9_path, step=1500)
            payload = torch.load(v9_path, map_location="cpu", weights_only=False)
            self.assertEqual(payload["format"], "fastwam_policy_guard_v9")
            metadata = payload["architecture_metadata"]
            self.assertEqual(
                metadata["warm_start_contract"], "exact_pgc_v5_sidecars"
            )
            self.assertEqual(metadata["privileged_supervision"], "training_only")
            self.assertEqual(metadata["deployment_inputs"], "rgb_language_proprio")
            self.assertEqual(metadata["eraf_grounding_objective_version"], 2)
            self.assertEqual(metadata["eraf_attention_mask_weight"], 2.0)
            self.assertEqual(metadata["eraf_role_swap_weight"], 2.0)
            self.assertEqual(metadata["eraf_role_overlap_weight"], 1.0)
            self.assertEqual(metadata["eraf_role_swap_margin"], 0.20)
            restored = tiny_pgc_fastwam(version=9, v9_stage="action")
            restored.load_checkpoint(v9_path)
            for key, value in v9.policy_guard_modules.state_dict().items():
                self.assertTrue(
                    torch.equal(
                        value,
                        restored.policy_guard_modules.state_dict()[key],
                    ),
                    key,
                )

    def test_v9r1_grounding_can_upgrade_to_v9r2_assignment_objective(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_path = root / "base.pt"
            v5_path = root / "v5.pt"
            v9r1_path = root / "v9r1.pt"
            v9r2_path = root / "v9r2.pt"
            base = tiny_pgc_fastwam(version=5)
            torch.save(
                {"format": "fastwam_full_v1", "mot": base.mot.state_dict()},
                base_path,
            )
            base.load_checkpoint(base_path)
            base.save_checkpoint(v5_path, step=4000)

            v9r1 = tiny_pgc_fastwam(
                version=9,
                v9_stage="grounding",
                v9_grounding_objective_version=2,
            )
            v9r1.load_checkpoint(v5_path)
            v9r1.save_checkpoint(v9r1_path, step=1500)

            v9r2 = tiny_pgc_fastwam(
                version=9,
                v9_stage="grounding",
                v9_grounding_objective_version=3,
            )
            v9r2.load_checkpoint(v9r1_path)
            self.assertEqual(
                v9r2.policy_guard_eraf_grounding_objective_version, 3
            )
            for key, value in v9r1.policy_guard_modules.state_dict().items():
                self.assertTrue(
                    torch.equal(
                        value,
                        v9r2.policy_guard_modules.state_dict()[key],
                    ),
                    key,
                )
            v9r2.save_checkpoint(v9r2_path, step=2500)
            metadata = torch.load(
                v9r2_path, map_location="cpu", weights_only=False
            )["architecture_metadata"]
            self.assertEqual(metadata["eraf_grounding_objective_version"], 3)
            self.assertEqual(metadata["eraf_role_assignment_weight"], 4.0)
            self.assertEqual(
                metadata["eraf_role_assignment_temperature"], 0.10
            )
            self.assertEqual(
                metadata["eraf_role_assignment_hard_weight"], 2.0
            )

            restored = tiny_pgc_fastwam(
                version=9,
                v9_stage="action",
                v9_grounding_objective_version=3,
            )
            restored.load_checkpoint(v9r2_path)

    def test_v9r1_grounding_can_upgrade_to_v9r3_frozen_role_adapter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_path = root / "base.pt"
            v5_path = root / "v5.pt"
            v9r1_path = root / "v9r1.pt"
            v9r3_path = root / "v9r3.pt"
            base = tiny_pgc_fastwam(version=5)
            torch.save(
                {"format": "fastwam_full_v1", "mot": base.mot.state_dict()},
                base_path,
            )
            base.load_checkpoint(base_path)
            base.save_checkpoint(v5_path, step=4000)

            v9r1 = tiny_pgc_fastwam(
                version=9,
                v9_stage="grounding",
                v9_grounding_objective_version=2,
            )
            v9r1.load_checkpoint(v5_path)
            v9r1.save_checkpoint(v9r1_path, step=1500)

            v9r3 = tiny_pgc_fastwam(
                version=9,
                v9_stage="grounding",
                v9_grounding_objective_version=4,
            )
            v9r3.load_checkpoint(v9r1_path)
            old_state = v9r1.policy_guard_modules.state_dict()
            new_state = v9r3.policy_guard_modules.state_dict()
            for key, value in old_state.items():
                self.assertTrue(torch.equal(value, new_state[key]), key)
            adapter = v9r3.policy_guard_modules[
                "entity_relation_affordance"
            ].role_assignment_adapter
            self.assertIsNotNone(adapter)
            self.assertEqual(
                int(adapter.subject_output.weight.count_nonzero().item()), 0
            )
            self.assertEqual(
                int(adapter.reference_output.weight.count_nonzero().item()), 0
            )

            torch.manual_seed(17)
            inputs = {
                "base_goal_queries": torch.randn(2, 3, 12),
                "base_goal_embedding": torch.randn(2, 8),
                "language_hidden": torch.randn(2, 6, 10),
                "language_mask": torch.ones(2, 6, dtype=torch.bool),
                "current_video_hidden": torch.randn(2, 8, 16),
            }
            v9r1_eraf = v9r1.policy_guard_modules[
                "entity_relation_affordance"
            ].eval()
            v9r3_eraf = v9r3.policy_guard_modules[
                "entity_relation_affordance"
            ].eval()
            with torch.no_grad():
                old_queries, old_embedding, old_outputs, _ = v9r1_eraf(**inputs)
                new_queries, new_embedding, new_outputs, _ = v9r3_eraf(**inputs)
            self.assertTrue(torch.equal(old_queries, new_queries))
            self.assertTrue(torch.equal(old_embedding, new_embedding))
            for key in (
                "subject_attention",
                "reference_attention",
                "subject_position",
                "reference_position",
                "relation_hidden",
                "goal_anchor",
            ):
                self.assertTrue(
                    torch.equal(old_outputs[key], new_outputs[key]), key
                )

            v9r3.save_checkpoint(v9r3_path, step=2500)
            payload = torch.load(
                v9r3_path, map_location="cpu", weights_only=False
            )
            metadata = payload["architecture_metadata"]
            self.assertEqual(metadata["eraf_grounding_objective_version"], 4)
            self.assertEqual(metadata["eraf_role_assignment_weight"], 1.0)
            self.assertEqual(metadata["eraf_role_assignment_hard_weight"], 0.5)
            self.assertEqual(
                metadata["eraf_role_adapter_trainable_scope"],
                "role_assignment_adapter_only",
            )

            restored = tiny_pgc_fastwam(
                version=9,
                v9_stage="action",
                v9_grounding_objective_version=4,
            )
            restored.load_checkpoint(v9r3_path)

    def test_v9r3_trains_only_adapter_then_freezes_it_for_action(self):
        grounding = tiny_pgc_fastwam(
            version=9,
            v9_stage="grounding",
            v9_grounding_objective_version=4,
        )
        grounding.prepare_trainable_parameters()
        trainable = {
            name
            for name, parameter in grounding.named_parameters()
            if parameter.requires_grad
        }
        self.assertTrue(trainable)
        self.assertTrue(
            all(
                name.startswith(
                    "policy_guard_modules.entity_relation_affordance."
                    "role_assignment_adapter."
                )
                for name in trainable
            )
        )
        groups = grounding.policy_guard_optimizer_groups(5.0e-5)
        self.assertEqual(len(groups), 1)
        self.assertEqual(
            groups[0]["pgc_v9_group"], "entity_relation_affordance"
        )

        action = tiny_pgc_fastwam(
            version=9,
            v9_stage="action",
            v9_grounding_objective_version=4,
        )
        action.prepare_trainable_parameters()
        action_trainable = {
            name
            for name, parameter in action.named_parameters()
            if parameter.requires_grad
        }
        self.assertTrue(action_trainable)
        self.assertTrue(
            all(
                name.startswith(
                    "policy_guard_modules.action_chunk_proposal."
                )
                for name in action_trainable
            )
        )

    def test_v9r3_can_upgrade_to_v94_structured_role_adapter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_path = root / "base.pt"
            v5_path = root / "v5.pt"
            v9r1_path = root / "v9r1.pt"
            v9r3_path = root / "v9r3.pt"
            v9r4_path = root / "v9r4.pt"
            base = tiny_pgc_fastwam(version=5)
            torch.save(
                {"format": "fastwam_full_v1", "mot": base.mot.state_dict()},
                base_path,
            )
            base.load_checkpoint(base_path)
            base.save_checkpoint(v5_path, step=4000)
            v9r1 = tiny_pgc_fastwam(
                version=9,
                v9_stage="grounding",
                v9_grounding_objective_version=2,
            )
            v9r1.load_checkpoint(v5_path)
            v9r1.save_checkpoint(v9r1_path, step=1500)
            v9r3 = tiny_pgc_fastwam(
                version=9,
                v9_stage="grounding",
                v9_grounding_objective_version=4,
            )
            v9r3.load_checkpoint(v9r1_path)
            v9r3.save_checkpoint(v9r3_path, step=2500)

            v9r4 = tiny_pgc_fastwam(
                version=9,
                v9_stage="grounding",
                v9_grounding_objective_version=5,
            )
            v9r4.load_checkpoint(v9r3_path)
            old_state = v9r3.policy_guard_modules.state_dict()
            new_state = v9r4.policy_guard_modules.state_dict()
            for key, value in old_state.items():
                self.assertTrue(torch.equal(value, new_state[key]), key)
            adapter = v9r4.policy_guard_modules[
                "entity_relation_affordance"
            ].structured_role_assignment_adapter
            self.assertIsNotNone(adapter)
            self.assertEqual(
                int(adapter.subject_output.weight.count_nonzero().item()), 0
            )
            self.assertEqual(
                int(adapter.reference_output.weight.count_nonzero().item()), 0
            )

            torch.manual_seed(94)
            inputs = {
                "base_goal_queries": torch.randn(2, 3, 12),
                "base_goal_embedding": torch.randn(2, 8),
                "language_hidden": torch.randn(2, 6, 10),
                "language_mask": torch.ones(2, 6, dtype=torch.bool),
                "current_video_hidden": torch.randn(2, 8, 16),
            }
            with torch.no_grad():
                old_result = v9r3.policy_guard_modules[
                    "entity_relation_affordance"
                ].eval()(**inputs)
                new_result = v9r4.policy_guard_modules[
                    "entity_relation_affordance"
                ].eval()(**inputs)
            self.assertTrue(torch.equal(old_result[0], new_result[0]))
            self.assertTrue(torch.equal(old_result[1], new_result[1]))
            for key in (
                "subject_attention",
                "reference_attention",
                "subject_position",
                "reference_position",
                "relation_hidden",
                "goal_anchor",
            ):
                self.assertTrue(
                    torch.equal(old_result[2][key], new_result[2][key]), key
                )

            v9r4.save_checkpoint(v9r4_path, step=3500)
            payload = torch.load(
                v9r4_path, map_location="cpu", weights_only=False
            )
            metadata = payload["architecture_metadata"]
            self.assertEqual(metadata["eraf_grounding_objective_version"], 5)
            self.assertEqual(
                metadata["eraf_role_adapter_trainable_scope"],
                "structured_role_assignment_adapter_only",
            )
            self.assertEqual(metadata["eraf_structured_assignment_weight"], 2.0)
            restored = tiny_pgc_fastwam(
                version=9,
                v9_stage="action",
                v9_grounding_objective_version=5,
            )
            restored.load_checkpoint(v9r4_path)

    def test_v94_trains_only_structured_adapter_then_freezes_for_action(self):
        grounding = tiny_pgc_fastwam(
            version=9,
            v9_stage="grounding",
            v9_grounding_objective_version=5,
        )
        grounding.prepare_trainable_parameters()
        trainable = {
            name
            for name, parameter in grounding.named_parameters()
            if parameter.requires_grad
        }
        self.assertTrue(trainable)
        self.assertTrue(
            all(
                name.startswith(
                    "policy_guard_modules.entity_relation_affordance."
                    "structured_role_assignment_adapter."
                )
                for name in trainable
            )
        )
        action = tiny_pgc_fastwam(
            version=9,
            v9_stage="action",
            v9_grounding_objective_version=5,
        )
        action.prepare_trainable_parameters()
        action_trainable = {
            name
            for name, parameter in action.named_parameters()
            if parameter.requires_grad
        }
        self.assertTrue(action_trainable)
        self.assertTrue(
            all(
                name.startswith("policy_guard_modules.action_chunk_proposal.")
                for name in action_trainable
            )
        )

    def test_v93_can_upgrade_to_v95_balanced_visual_binding(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_path = root / "base.pt"
            v5_path = root / "v5.pt"
            v9r1_path = root / "v9r1.pt"
            v9r3_path = root / "v9r3.pt"
            v9r5_path = root / "v9r5.pt"
            base = tiny_pgc_fastwam(version=5)
            torch.save(
                {"format": "fastwam_full_v1", "mot": base.mot.state_dict()},
                base_path,
            )
            base.load_checkpoint(base_path)
            base.save_checkpoint(v5_path, step=4000)
            v9r1 = tiny_pgc_fastwam(
                version=9,
                v9_stage="grounding",
                v9_grounding_objective_version=2,
            )
            v9r1.load_checkpoint(v5_path)
            v9r1.save_checkpoint(v9r1_path, step=1500)
            v9r3 = tiny_pgc_fastwam(
                version=9,
                v9_stage="grounding",
                v9_grounding_objective_version=4,
            )
            v9r3.load_checkpoint(v9r1_path)
            v9r3.save_checkpoint(v9r3_path, step=2500)

            v9r5 = tiny_pgc_fastwam(
                version=9,
                v9_stage="grounding",
                v9_grounding_objective_version=6,
            )
            v9r5.load_checkpoint(v9r3_path)
            old_state = v9r3.policy_guard_modules.state_dict()
            new_state = v9r5.policy_guard_modules.state_dict()
            for key, value in old_state.items():
                self.assertTrue(torch.equal(value, new_state[key]), key)
            adapter = v9r5.policy_guard_modules[
                "entity_relation_affordance"
            ].balanced_role_binding_adapter
            self.assertIsNotNone(adapter)
            self.assertEqual(
                int(adapter.subject_output.weight.count_nonzero().item()), 0
            )
            self.assertEqual(
                int(adapter.reference_output.weight.count_nonzero().item()), 0
            )
            v9r5.prepare_trainable_parameters()
            trainable = {
                name
                for name, parameter in v9r5.named_parameters()
                if parameter.requires_grad
            }
            self.assertTrue(trainable)
            self.assertTrue(
                all(
                    name.startswith(
                        "policy_guard_modules.entity_relation_affordance."
                        "balanced_role_binding_adapter."
                    )
                    for name in trainable
                )
            )
            v9r5.save_checkpoint(v9r5_path, step=3500)
            payload = torch.load(v9r5_path, map_location="cpu", weights_only=False)
            metadata = payload["architecture_metadata"]
            self.assertEqual(metadata["eraf_grounding_objective_version"], 6)
            self.assertEqual(
                metadata["eraf_role_adapter_trainable_scope"],
                "balanced_visual_role_binding_adapter_only",
            )
            restored = tiny_pgc_fastwam(
                version=9,
                v9_stage="action",
                v9_grounding_objective_version=6,
            )
            restored.load_checkpoint(v9r5_path)

    def test_v93_can_upgrade_to_v96_global_hard_curriculum(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_path = root / "base.pt"
            v5_path = root / "v5.pt"
            v9r1_path = root / "v9r1.pt"
            v9r3_path = root / "v9r3.pt"
            v9r6_path = root / "v9r6.pt"
            base = tiny_pgc_fastwam(version=5)
            torch.save(
                {"format": "fastwam_full_v1", "mot": base.mot.state_dict()},
                base_path,
            )
            base.load_checkpoint(base_path)
            base.save_checkpoint(v5_path, step=4000)
            v9r1 = tiny_pgc_fastwam(
                version=9,
                v9_stage="grounding",
                v9_grounding_objective_version=2,
            )
            v9r1.load_checkpoint(v5_path)
            v9r1.save_checkpoint(v9r1_path, step=1500)
            v9r3 = tiny_pgc_fastwam(
                version=9,
                v9_stage="grounding",
                v9_grounding_objective_version=4,
            )
            v9r3.load_checkpoint(v9r1_path)
            v9r3.save_checkpoint(v9r3_path, step=2500)

            v9r6 = tiny_pgc_fastwam(
                version=9,
                v9_stage="grounding",
                v9_grounding_objective_version=7,
            )
            v9r6.load_checkpoint(v9r3_path)
            v9r6.prepare_trainable_parameters()
            trainable = {
                name
                for name, parameter in v9r6.named_parameters()
                if parameter.requires_grad
            }
            self.assertTrue(trainable)
            self.assertTrue(
                all("balanced_role_binding_adapter" in name for name in trainable)
            )
            v9r6.save_checkpoint(v9r6_path, step=3000)
            payload = torch.load(v9r6_path, map_location="cpu", weights_only=False)
            metadata = payload["architecture_metadata"]
            self.assertEqual(metadata["eraf_grounding_objective_version"], 7)
            self.assertEqual(
                metadata["eraf_role_adapter_trainable_scope"],
                "global_hard_curriculum_balanced_visual_role_binding_adapter_only",
            )
            self.assertEqual(
                metadata["eraf_hard_role_curriculum"],
                "v9_3_audited_native_hard_easy_1_1",
            )
            self.assertEqual(metadata["eraf_ddp_group_balance"], "global_count_exact")
            self.assertEqual(
                metadata["eraf_geometry_preservation_scope"],
                "all_active_clauses",
            )
            restored = tiny_pgc_fastwam(
                version=9,
                v9_stage="action",
                v9_grounding_objective_version=7,
            )
            restored.load_checkpoint(v9r6_path)

    def test_v96_to_v97_to_v98_checkpoint_upgrade_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_path = root / "base.pt"
            v5_path = root / "v5.pt"
            v9r1_path = root / "v9r1.pt"
            v9r3_path = root / "v9r3.pt"
            v9r6_path = root / "v9r6.pt"
            v9r7_path = root / "v9r7.pt"
            v9r8_path = root / "v9r8.pt"
            base = tiny_pgc_fastwam(version=5)
            torch.save(
                {"format": "fastwam_full_v1", "mot": base.mot.state_dict()},
                base_path,
            )
            base.load_checkpoint(base_path)
            base.save_checkpoint(v5_path, step=4000)
            v9r1 = tiny_pgc_fastwam(
                version=9,
                v9_stage="grounding",
                v9_grounding_objective_version=2,
            )
            v9r1.load_checkpoint(v5_path)
            v9r1.save_checkpoint(v9r1_path, step=1500)
            v9r3 = tiny_pgc_fastwam(
                version=9,
                v9_stage="grounding",
                v9_grounding_objective_version=4,
            )
            v9r3.load_checkpoint(v9r1_path)
            v9r3.save_checkpoint(v9r3_path, step=2500)
            v9r6 = tiny_pgc_fastwam(
                version=9,
                v9_stage="grounding",
                v9_grounding_objective_version=7,
            )
            v9r6.load_checkpoint(v9r3_path)
            v9r6.save_checkpoint(v9r6_path, step=3000)

            v9r7 = tiny_pgc_fastwam(
                version=9,
                v9_stage="grounding",
                v9_grounding_objective_version=8,
            )
            v9r7.load_checkpoint(v9r6_path)
            v9r7.prepare_trainable_parameters()
            trainable = {
                name
                for name, parameter in v9r7.named_parameters()
                if parameter.requires_grad
            }
            self.assertTrue(trainable)
            self.assertTrue(
                all("balanced_role_binding_adapter" in name for name in trainable)
            )
            v9r7.save_checkpoint(v9r7_path, step=3250)
            metadata = torch.load(
                v9r7_path, map_location="cpu", weights_only=False
            )["architecture_metadata"]
            self.assertEqual(metadata["eraf_grounding_objective_version"], 8)
            self.assertEqual(
                metadata["eraf_role_evidence"],
                "exclusive_subject_reference_support",
            )
            self.assertEqual(
                metadata["eraf_role_gate"],
                "exclusive_accuracy_with_full_mask_localization",
            )
            self.assertEqual(metadata["eraf_exclusive_role_coverage_min"], 0.5)
            self.assertEqual(
                metadata["eraf_role_adapter_trainable_scope"],
                "exclusive_evidence_global_hard_curriculum_"
                "balanced_visual_role_binding_adapter_only",
            )
            restored = tiny_pgc_fastwam(
                version=9,
                v9_stage="action",
                v9_grounding_objective_version=8,
            )
            restored.load_checkpoint(v9r7_path)

            v9r8 = tiny_pgc_fastwam(
                version=9,
                v9_stage="grounding",
                v9_grounding_objective_version=9,
            )
            v9r8.load_checkpoint(v9r7_path)
            # The migration contract is exact for every pre-existing V9.7
            # sidecar tensor.  Check that contract directly before comparing
            # forward paths so an actual checkpoint-loading regression cannot
            # be hidden by a numerical tolerance below.
            v97_state = v9r7.policy_guard_modules.state_dict()
            v98_state = v9r8.policy_guard_modules.state_dict()
            for key, value in v97_state.items():
                self.assertTrue(torch.equal(value, v98_state[key]), key)
            clause_adapter = v9r8.policy_guard_modules[
                "entity_relation_affordance"
            ].clause_activation_adapter
            self.assertIsNotNone(clause_adapter)
            self.assertEqual(
                int(clause_adapter.active_output.weight.count_nonzero().item()),
                0,
            )
            self.assertEqual(
                int(clause_adapter.active_output.bias.count_nonzero().item()),
                0,
            )
            v9r7.eval()
            v9r8.eval()
            final_video = torch.randn(2, 8, 16)
            current_video = torch.randn(2, 8, 16)
            context = torch.randn(2, 5, 10)
            context_mask = torch.ones(2, 5, dtype=torch.bool)
            base_goal_queries = torch.randn(2, 3, 12)
            base_goal_embedding = torch.randn(2, 8)
            eraf_inputs = {
                "base_goal_queries": base_goal_queries,
                "base_goal_embedding": base_goal_embedding,
                "language_hidden": context,
                "language_mask": context_mask,
                "current_video_hidden": current_video,
            }
            with torch.no_grad():
                base_roles = v9r7.policy_guard_modules[
                    "entity_relation_affordance"
                ].role_decoder(context, context_mask)
                calibrated_roles = clause_adapter(
                    clause_hidden=base_roles["clause_hidden"],
                    active_logits=base_roles["active_logits"],
                )
                v98_eraf = v9r8.policy_guard_modules[
                    "entity_relation_affordance"
                ](**eraf_inputs)
                v98_encoded = v9r8._encode_policy_guard_eraf(
                    final_video_hidden=final_video,
                    current_visual_hidden=current_video,
                    video_tokens_per_frame=8,
                    context=context,
                    context_mask=context_mask,
                    language_context_len=5,
                )
            # The newly inserted adapter itself is an exact identity.  The
            # enclosing attention stack is checked numerically because two
            # independently constructed MHA modules are not guaranteed to be
            # bitwise reproducible on every PyTorch backend.
            self.assertTrue(
                torch.equal(
                    calibrated_roles["active_logits"],
                    base_roles["active_logits"],
                )
            )
            # Check the upgrade at the actual insertion boundary instead of
            # comparing two independent MultiheadAttention executions. Some
            # PyTorch CPU backends produce harmless last-bit differences even
            # when their state dictionaries and inputs are identical. The
            # exact shared-state check above plus these same-forward checks
            # prove the migration contract without backend-dependent noise.
            self.assertTrue(
                torch.equal(
                    v98_eraf[2]["active_logits"],
                    v98_eraf[2]["base_active_logits"],
                )
            )
            self.assertTrue(
                torch.equal(
                    v98_eraf[2]["clause_active_residual"],
                    torch.zeros_like(v98_eraf[2]["clause_active_residual"]),
                )
            )
            self.assertTrue(torch.isfinite(v98_eraf[0]).all())
            self.assertTrue(torch.isfinite(v98_eraf[1]).all())
            self.assertTrue(torch.isfinite(v98_encoded[0]).all())
            self.assertTrue(torch.isfinite(v98_encoded[1]).all())
            v9r8.prepare_trainable_parameters()
            trainable = {
                name
                for name, parameter in v9r8.named_parameters()
                if parameter.requires_grad
            }
            self.assertTrue(trainable)
            self.assertTrue(
                all("clause_activation_adapter" in name for name in trainable)
            )
            v9r8.save_checkpoint(v9r8_path, step=3750)
            metadata = torch.load(
                v9r8_path, map_location="cpu", weights_only=False
            )["architecture_metadata"]
            self.assertEqual(metadata["eraf_grounding_objective_version"], 9)
            self.assertEqual(
                metadata["eraf_clause_activation_contract"],
                "zero_init_cross_clause_active_logit_residual",
            )
            self.assertEqual(
                metadata["eraf_clause_cardinality_supervision"],
                "balanced_active_bce_plus_count_ce_plus_multi_worst_slot",
            )
            self.assertEqual(
                metadata["eraf_clause_gate"],
                "multi_clause_exact_at_least_80pct",
            )
            self.assertEqual(
                metadata["eraf_role_adapter_trainable_scope"],
                "clause_activation_calibration_adapter_only",
            )
            restored_v98 = tiny_pgc_fastwam(
                version=9,
                v9_stage="action",
                v9_grounding_objective_version=9,
            )
            restored_v98.load_checkpoint(v9r8_path)

    def test_v98_to_v99_view_scheduler_upgrade_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_path = root / "base.pt"
            v98_path = root / "v98.pt"
            v99_path = root / "v99.pt"
            v98 = tiny_pgc_fastwam(
                version=9,
                v9_stage="grounding",
                v9_grounding_objective_version=9,
            )
            torch.save(
                {"format": "fastwam_full_v1", "mot": v98.mot.state_dict()},
                base_path,
            )
            v98.load_checkpoint(base_path)
            # Represent a trained V9.8 rather than relying on its initial
            # zero route. V9.9 must preserve a non-trivial predecessor path.
            v98_eraf_module = v98.policy_guard_modules[
                "entity_relation_affordance"
            ]
            with torch.no_grad():
                v98_eraf_module.query_delta_projection.weight.normal_(std=0.02)
                v98_eraf_module.query_delta_projection.bias.normal_(std=0.02)
                v98_eraf_module.embedding_delta_projection.weight.normal_(std=0.02)
                v98_eraf_module.embedding_delta_projection.bias.normal_(std=0.02)
                v98_eraf_module.clause_activation_adapter.active_output.weight.normal_(
                    std=0.02
                )
                v98_eraf_module.clause_activation_adapter.active_output.bias.normal_(
                    std=0.02
                )
            v98.save_checkpoint(v98_path, step=3750)

            v99 = tiny_pgc_fastwam(
                version=9,
                v9_stage="grounding",
                v9_grounding_objective_version=10,
            )
            v99.load_checkpoint(v98_path)
            v98_state = v98.policy_guard_modules.state_dict()
            v99_state = v99.policy_guard_modules.state_dict()
            for key, value in v98_state.items():
                self.assertTrue(torch.equal(value, v99_state[key]), key)
            eraf = v99.policy_guard_modules["entity_relation_affordance"]
            self.assertIsNotNone(eraf.entity_grounder.view_fusion_adapter)
            self.assertIsNotNone(eraf.clause_execution_scheduler)
            self.assertEqual(
                int(
                    eraf.entity_grounder.view_fusion_adapter.output.weight
                    .count_nonzero()
                    .item()
                ),
                0,
            )
            self.assertEqual(
                int(eraf.clause_execution_scheduler.output.weight.count_nonzero()),
                0,
            )
            eraf_inputs = {
                "base_goal_queries": torch.randn(2, 3, 12),
                "base_goal_embedding": torch.randn(2, 8),
                "language_hidden": torch.randn(2, 5, 10),
                "language_mask": torch.ones(2, 5, dtype=torch.bool),
                "current_video_hidden": torch.randn(2, 8, 16),
            }
            v98.eval()
            v99.eval()
            with torch.no_grad():
                v98_output = v98_eraf_module(**eraf_inputs)
                v99_output = eraf(**eraf_inputs)
            torch.testing.assert_close(
                v99_output[0], v98_output[0], rtol=0.0, atol=1.0e-6
            )
            torch.testing.assert_close(
                v99_output[1], v98_output[1], rtol=0.0, atol=1.0e-6
            )
            self.assertTrue(
                torch.equal(
                    v99_output[2]["subject_attention"],
                    v99_output[2]["subject_base_attention"],
                )
            )
            self.assertTrue(
                torch.equal(
                    v99_output[2]["clause_routing_multiplier"],
                    torch.ones_like(
                        v99_output[2]["clause_routing_multiplier"]
                    ),
                )
            )
            v99.prepare_trainable_parameters()
            trainable = {
                name
                for name, parameter in v99.named_parameters()
                if parameter.requires_grad
            }
            self.assertTrue(trainable)
            allowed = (
                "balanced_role_binding_adapter",
                "clause_activation_adapter",
                "view_visibility_head",
                "view_fusion_adapter",
                "clause_execution_scheduler",
            )
            self.assertTrue(
                all(any(part in name for part in allowed) for name in trainable)
            )
            self.assertTrue(any("view_fusion_adapter" in name for name in trainable))
            self.assertTrue(
                any("clause_execution_scheduler" in name for name in trainable)
            )
            v99.save_checkpoint(v99_path, step=4750)
            metadata = torch.load(
                v99_path, map_location="cpu", weights_only=False
            )["architecture_metadata"]
            self.assertEqual(metadata["eraf_grounding_objective_version"], 10)
            self.assertEqual(
                metadata["eraf_view_fusion_contract"],
                "per_view_local_attention_visibility_gated_zero_init_residual",
            )
            self.assertEqual(
                metadata["eraf_clause_scheduler_contract"],
                "first_active_unfinished_predicate_zero_init_residual_route",
            )
            self.assertEqual(
                metadata["eraf_role_adapter_trainable_scope"],
                "clause_activation_plus_balanced_role_plus_"
                "visibility_gated_view_fusion_plus_unfinished_clause_scheduler",
            )
            restored = tiny_pgc_fastwam(
                version=9,
                v9_stage="action",
                v9_grounding_objective_version=10,
            )
            restored.load_checkpoint(v99_path)

    def test_v99_to_v910_exclusive_all_entity_upgrade_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_path = root / "base.pt"
            v99_path = root / "v99.pt"
            v910_path = root / "v910.pt"
            v99 = tiny_pgc_fastwam(
                version=9,
                v9_stage="grounding",
                v9_grounding_objective_version=10,
            )
            torch.save(
                {"format": "fastwam_full_v1", "mot": v99.mot.state_dict()},
                base_path,
            )
            v99.load_checkpoint(base_path)
            eraf99 = v99.policy_guard_modules[
                "entity_relation_affordance"
            ]
            with torch.no_grad():
                for output in (
                    eraf99.balanced_role_binding_adapter.subject_output,
                    eraf99.balanced_role_binding_adapter.reference_output,
                ):
                    output.weight.normal_(std=0.02)
                    output.bias.normal_(std=0.02)
            v99.save_checkpoint(v99_path, step=4750)

            v910 = tiny_pgc_fastwam(
                version=9,
                v9_stage="grounding",
                v9_grounding_objective_version=11,
            )
            v910.load_checkpoint(v99_path)
            state99 = v99.policy_guard_modules.state_dict()
            state910 = v910.policy_guard_modules.state_dict()
            self.assertEqual(state99.keys(), state910.keys())
            for key, value in state99.items():
                self.assertTrue(torch.equal(value, state910[key]), key)

            eraf_inputs = {
                "base_goal_queries": torch.randn(2, 3, 12),
                "base_goal_embedding": torch.randn(2, 8),
                "language_hidden": torch.randn(2, 5, 10),
                "language_mask": torch.ones(2, 5, dtype=torch.bool),
                "current_video_hidden": torch.randn(2, 8, 16),
            }
            v99.eval()
            v910.eval()
            with torch.no_grad():
                output99 = eraf99(**eraf_inputs)
                output910 = v910.policy_guard_modules[
                    "entity_relation_affordance"
                ](**eraf_inputs)
            for index in (0, 1):
                torch.testing.assert_close(
                    output910[index], output99[index], rtol=0.0, atol=0.0
                )
            for name in (
                "subject_attention",
                "reference_attention",
                "active_logits",
                "clause_execution_logits",
            ):
                torch.testing.assert_close(
                    output910[2][name], output99[2][name], rtol=0.0, atol=0.0
                )

            v910.prepare_trainable_parameters()
            trainable = {
                name
                for name, parameter in v910.named_parameters()
                if parameter.requires_grad
            }
            self.assertTrue(trainable)
            self.assertTrue(
                all("balanced_role_binding_adapter" in name for name in trainable)
            )
            v910.save_checkpoint(v910_path, step=5750)
            metadata = torch.load(
                v910_path, map_location="cpu", weights_only=False
            )["architecture_metadata"]
            self.assertEqual(metadata["eraf_grounding_objective_version"], 11)
            self.assertEqual(
                metadata["eraf_all_entity_role_contract"],
                "exclusive_evidence_same_state_all_entity_bipartite_assignment",
            )
            self.assertEqual(
                metadata["eraf_multi_clause_gate_contract"],
                "semantic_exact_with_exclusive_role_evidence",
            )
            self.assertEqual(
                metadata["eraf_role_adapter_trainable_scope"],
                "exclusive_all_entity_balanced_visual_role_binding_adapter_only",
            )
            restored = tiny_pgc_fastwam(
                version=9,
                v9_stage="action",
                v9_grounding_objective_version=11,
            )
            restored.load_checkpoint(v910_path)

    def test_v910_to_v911_clause_tuple_upgrade_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_path = root / "base.pt"
            v910_path = root / "v910.pt"
            v911_path = root / "v911.pt"
            v910 = tiny_pgc_fastwam(
                version=9,
                v9_stage="grounding",
                v9_grounding_objective_version=11,
            )
            torch.save(
                {"format": "fastwam_full_v1", "mot": v910.mot.state_dict()},
                base_path,
            )
            v910.load_checkpoint(base_path)
            v910.save_checkpoint(v910_path, step=5750)

            v911 = tiny_pgc_fastwam(
                version=9,
                v9_stage="grounding",
                v9_grounding_objective_version=12,
            )
            v911.load_checkpoint(v910_path)
            state910 = v910.policy_guard_modules.state_dict()
            state911 = v911.policy_guard_modules.state_dict()
            self.assertEqual(state910.keys(), state911.keys())
            for key, value in state910.items():
                self.assertTrue(torch.equal(value, state911[key]), key)

            v911.prepare_trainable_parameters()
            trainable = {
                name
                for name, parameter in v911.named_parameters()
                if parameter.requires_grad
            }
            self.assertTrue(trainable)
            self.assertTrue(
                all("balanced_role_binding_adapter" in name for name in trainable)
            )
            v911.save_checkpoint(v911_path, step=6250)
            payload = torch.load(
                v911_path, map_location="cpu", weights_only=False
            )
            metadata = payload["architecture_metadata"]
            self.assertEqual(metadata["eraf_grounding_objective_version"], 12)
            self.assertEqual(
                metadata["eraf_clause_tuple_contract"],
                "exclusive_same_state_subject_predicate_reference_assignment",
            )
            self.assertEqual(
                metadata["eraf_clause_tuple_curriculum_contract"],
                "v9_10_audit_native_hard_easy_plus_historical_strict_1_1_1_1",
            )
            self.assertEqual(
                metadata["eraf_role_adapter_trainable_scope"],
                "audited_hard_clause_tuple_balanced_visual_"
                "role_binding_adapter_only",
            )
            self.assertEqual(metadata["eraf_clause_tuple_assignment_weight"], 4.0)
            restored = tiny_pgc_fastwam(
                version=9,
                v9_stage="action",
                v9_grounding_objective_version=12,
            )
            restored.load_checkpoint(v911_path)

    def test_v911_to_v912_closed_loop_rebinding_upgrade_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_path = root / "base.pt"
            v911_path = root / "v911.pt"
            v912_path = root / "v912.pt"
            v911 = tiny_pgc_fastwam(
                version=9,
                v9_stage="grounding",
                v9_grounding_objective_version=12,
            )
            torch.save(
                {"format": "fastwam_full_v1", "mot": v911.mot.state_dict()},
                base_path,
            )
            v911.load_checkpoint(base_path)
            v911.save_checkpoint(v911_path, step=6250)

            v912 = tiny_pgc_fastwam(
                version=9,
                v9_stage="grounding",
                v9_grounding_objective_version=13,
            )
            v912.load_checkpoint(v911_path)
            adapter = v912.policy_guard_modules[
                "entity_relation_affordance"
            ].closed_loop_phase_rebinding_adapter
            self.assertIsNotNone(adapter)
            for output in (
                adapter.subject_output,
                adapter.reference_output,
                adapter.truth_output,
                adapter.phase_output,
            ):
                self.assertEqual(int(output.weight.count_nonzero()), 0)
                self.assertEqual(int(output.bias.count_nonzero()), 0)

            v912.prepare_trainable_parameters()
            trainable = {
                name
                for name, parameter in v912.named_parameters()
                if parameter.requires_grad
            }
            self.assertTrue(trainable)
            self.assertTrue(
                all(
                    "closed_loop_phase_rebinding_adapter" in name
                    for name in trainable
                )
            )
            self.assertFalse(any(parameter.requires_grad for parameter in v912.mot.parameters()))
            v912.save_checkpoint(v912_path, step=7250)
            payload = torch.load(v912_path, map_location="cpu", weights_only=False)
            metadata = payload["architecture_metadata"]
            self.assertEqual(metadata["eraf_grounding_objective_version"], 13)
            self.assertEqual(
                metadata["eraf_closed_loop_rebinding_contract"],
                "zero_init_second_pass_role_truth_phase_and_clause_route",
            )
            self.assertEqual(
                metadata["eraf_closed_loop_state_contract"],
                "immutable_base_correct_replan_exact_simulator_state",
            )
            self.assertEqual(
                metadata["eraf_closed_loop_curriculum_contract"],
                "offline_native_closed_loop_native_historical_strict_1_1_1_1",
            )
            self.assertEqual(
                metadata["eraf_role_adapter_trainable_scope"],
                "closed_loop_phase_rebinding_adapter_only",
            )

            restored = tiny_pgc_fastwam(
                version=9,
                v9_stage="action",
                v9_grounding_objective_version=13,
            )
            restored.load_checkpoint(v912_path)
            expected_state = v912.policy_guard_modules.state_dict()
            restored_state = restored.policy_guard_modules.state_dict()
            self.assertEqual(expected_state.keys(), restored_state.keys())
            for name, value in expected_state.items():
                self.assertTrue(torch.equal(value, restored_state[name]), name)

    def test_v911_to_v913_phase_safe_memory_upgrade_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_path = root / "base.pt"
            v911_path = root / "v911.pt"
            v913_path = root / "v913.pt"
            v911 = tiny_pgc_fastwam(
                version=9,
                v9_stage="grounding",
                v9_grounding_objective_version=12,
            )
            torch.save(
                {"format": "fastwam_full_v1", "mot": v911.mot.state_dict()},
                base_path,
            )
            v911.load_checkpoint(base_path)
            v911.save_checkpoint(v911_path, step=6250)

            v913 = tiny_pgc_fastwam(
                version=9,
                v9_stage="grounding",
                v9_grounding_objective_version=14,
            )
            # The production V9.13 wrapper disables every frozen V9.11
            # auxiliary loss.  These values intentionally differ from the
            # source checkpoint and must not be treated as restored geometry.
            v913.policy_guard_eraf_loss_weights = replace(
                v913.policy_guard_eraf_loss_weights,
                attention_mask=0.0,
                role_swap=0.0,
                role_overlap=0.0,
                role_assignment=0.0,
                role_assignment_hard_weight=0.0,
            )
            v913.load_checkpoint(v911_path)
            eraf = v913.policy_guard_modules["entity_relation_affordance"]
            self.assertIsNone(eraf.closed_loop_phase_rebinding_adapter)
            self.assertIsNotNone(eraf.phase_safe_clause_memory)
            self.assertEqual(
                int(eraf.phase_safe_clause_memory.state_output.weight.count_nonzero()),
                0,
            )
            self.assertEqual(
                int(
                    eraf.phase_safe_clause_memory.scheduler_output.weight.count_nonzero()
                ),
                0,
            )

            v913.prepare_trainable_parameters()
            trainable = {
                name
                for name, parameter in v913.named_parameters()
                if parameter.requires_grad
            }
            self.assertTrue(trainable)
            self.assertTrue(
                all("phase_safe_clause_memory" in name for name in trainable)
            )
            self.assertFalse(any(p.requires_grad for p in v913.mot.parameters()))
            v913.save_checkpoint(v913_path, step=7250)
            payload = torch.load(v913_path, map_location="cpu", weights_only=False)
            metadata = payload["architecture_metadata"]
            self.assertEqual(metadata["eraf_grounding_objective_version"], 14)
            self.assertEqual(
                metadata["deployment_inputs"],
                "rgb_language_proprio_previous_policy_state",
            )
            self.assertEqual(
                metadata["eraf_role_adapter_trainable_scope"],
                "phase_safe_temporal_clause_memory_only",
            )
            self.assertEqual(
                metadata["eraf_phase_safe_memory_contract"],
                "explicit_cross_replan_pending_holding_retry_completed",
            )
            self.assertEqual(
                metadata["eraf_geometry_protection_contract"],
                "frozen_v9_11_no_query_token_anchor_or_heatmap_residual",
            )
            self.assertEqual(
                metadata["eraf_policy_state_contract"],
                "explicit_caller_owned_reset_per_episode",
            )

            restored = tiny_pgc_fastwam(
                version=9,
                v9_stage="action",
                v9_grounding_objective_version=14,
            )
            restored.policy_guard_eraf_loss_weights = replace(
                restored.policy_guard_eraf_loss_weights,
                attention_mask=0.0,
                role_swap=0.0,
                role_overlap=0.0,
                role_assignment=0.0,
                role_assignment_hard_weight=0.0,
            )
            restored.load_checkpoint(v913_path)
            expected_state = v913.policy_guard_modules.state_dict()
            restored_state = restored.policy_guard_modules.state_dict()
            self.assertEqual(expected_state.keys(), restored_state.keys())
            for name, value in expected_state.items():
                self.assertTrue(torch.equal(value, restored_state[name]), name)

    def test_stage_trainability_optimizer_contract_and_deployment_inputs(self):
        expected_modules = {
            "grounding": {"entity_relation_affordance"},
            "action": {"entity_relation_affordance", "action_chunk_proposal"},
            "verifier": {"verifier"},
        }
        for stage, expected in expected_modules.items():
            with self.subTest(stage=stage):
                model = tiny_pgc_fastwam(version=9, v9_stage=stage)
                model.prepare_trainable_parameters()
                trainable_modules = {
                    name.split(".")[1]
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad
                    and name.startswith("policy_guard_modules.")
                }
                self.assertEqual(trainable_modules, expected)
                groups = model.policy_guard_optimizer_groups(1.0e-4)
                self.assertEqual(
                    {group["pgc_v9_group"] for group in groups}, expected
                )
                rates = {
                    group["pgc_v9_group"]: group["lr"] for group in groups
                }
                if stage == "action":
                    self.assertEqual(rates["entity_relation_affordance"], 2.0e-5)
                    self.assertEqual(rates["action_chunk_proposal"], 1.0e-4)
                self.assertFalse(any(p.requires_grad for p in model.mot.parameters()))
        deployment_parameters = set(
            signature(EntityRelationAffordanceField.forward).parameters
        )
        self.assertEqual(
            deployment_parameters,
            {
                "self",
                "base_goal_queries",
                "base_goal_embedding",
                "language_hidden",
                "language_mask",
                "current_video_hidden",
                "policy_state",
                "proprio",
            },
        )

    def test_v914_completion_only_joint_action_contract(self):
        model = tiny_pgc_fastwam(
            version=9,
            v9_stage="action",
            v9_grounding_objective_version=14,
            v9_completion_only_memory=True,
            v9_action_joint_training=True,
        )
        state = {
            "phase_safe_memory_state_ids": torch.tensor([[0, 1, 2, 3]]),
            "phase_safe_memory_valid": torch.tensor(
                [[True, True, True, True]]
            ),
        }
        sanitized = model._policy_guard_completion_only_state(state)
        self.assertTrue(
            torch.equal(
                sanitized["phase_safe_memory_state_ids"],
                torch.tensor([[0, 0, 0, 3]]),
            )
        )
        self.assertTrue(
            torch.equal(
                sanitized["phase_safe_memory_valid"],
                torch.tensor([[False, False, False, True]]),
            )
        )
        monotonic = model._policy_guard_completion_only_state(
            {
                "phase_safe_memory_state_ids": torch.tensor([[0, 0, 0, 0]]),
                "phase_safe_memory_valid": torch.tensor(
                    [[False, False, False, False]]
                ),
            },
            previous_state=sanitized,
        )
        self.assertTrue(
            torch.equal(
                monotonic["phase_safe_memory_state_ids"],
                torch.tensor([[0, 0, 0, 3]]),
            )
        )
        self.assertTrue(monotonic["phase_safe_memory_valid"][0, 3])

        model.prepare_trainable_parameters()
        trainable = {
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        proposal_prefix = "policy_guard_modules.action_chunk_proposal."
        bridge_prefixes = tuple(
            "policy_guard_modules.entity_relation_affordance." + name + "."
            for name in (
                "base_query_projection",
                "relation_attention",
                "query_delta_projection",
                "embedding_delta_projection",
            )
        )
        self.assertTrue(trainable)
        self.assertTrue(
            all(
                name.startswith(proposal_prefix)
                or name.startswith(bridge_prefixes)
                for name in trainable
            )
        )
        self.assertTrue(any(name.startswith(proposal_prefix) for name in trainable))
        for prefix in bridge_prefixes:
            self.assertTrue(any(name.startswith(prefix) for name in trainable))
        self.assertFalse(any(p.requires_grad for p in model.mot.parameters()))
        groups = model.policy_guard_optimizer_groups(1.0e-4)
        rates = {group["pgc_v9_group"]: group["lr"] for group in groups}
        self.assertEqual(rates["entity_relation_affordance"], 2.0e-5)
        self.assertEqual(rates["action_chunk_proposal"], 1.0e-4)

    def test_v914_rejects_joint_action_without_completion_only_contract(self):
        with self.assertRaisesRegex(ValueError, "completion-only memory"):
            tiny_pgc_fastwam(
                version=9,
                v9_stage="action",
                v9_grounding_objective_version=14,
                v9_action_joint_training=True,
            )

    @staticmethod
    def _v915_bridge_outputs(batch_size=2, clauses=4, hidden_dim=8):
        return {
            "active_logits": torch.tensor(
                [[4.0, 4.0, -4.0, -4.0]]
            ).expand(batch_size, -1).clone(),
            "subject_token": torch.randn(batch_size, clauses, hidden_dim),
            "reference_token": torch.randn(batch_size, clauses, hidden_dim),
            "relation_hidden": torch.randn(batch_size, clauses, hidden_dim),
            "subject_position": torch.randn(batch_size, clauses, 3) * 0.1,
            "reference_position": torch.randn(batch_size, clauses, 3) * 0.1,
            "grasp_anchor": torch.randn(batch_size, clauses, 3) * 0.1,
            "goal_anchor": torch.randn(batch_size, clauses, 3) * 0.1,
            "interaction_anchor": torch.randn(batch_size, clauses, 3) * 0.1,
            "predicate_truth_logits": torch.randn(batch_size, clauses),
            "phase_logits": torch.randn(batch_size, clauses, 3),
            "clause_execution_probability": torch.softmax(
                torch.randn(batch_size, clauses), dim=-1
            ),
            "subject_visibility_logits": torch.full(
                (batch_size, clauses), 4.0
            ),
            "reference_visibility_logits": torch.full(
                (batch_size, clauses), 4.0
            ),
        }

    def test_v917_geometry_action_adapter_is_zero_init_and_anchor_causal(self):
        adapter = PhaseConditionedERAFGeometryActionAdapter(
            action_dim=7,
            proprio_dim=8,
            hidden_dim=16,
            max_clauses=4,
            max_abs=0.25,
        )
        action = torch.randn(2, 6, 7)
        proprio = torch.randn(2, 8)
        outputs = self._v915_bridge_outputs()
        initial, residual, metrics = adapter(
            candidate_action=action,
            eraf_outputs=outputs,
            proprio=proprio,
        )
        self.assertTrue(torch.equal(initial, action))
        self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))
        self.assertEqual(
            float(metrics["pgc_v917_geometry_action_residual_rms"]), 0.0
        )

        adapter.output_projection.weight.data.normal_(std=0.1)
        learned, _, _ = adapter(
            candidate_action=action,
            eraf_outputs=outputs,
            proprio=proprio,
        )
        changed_outputs = dict(outputs)
        changed_outputs["goal_anchor"] = outputs["goal_anchor"] + 0.4
        changed, _, _ = adapter(
            candidate_action=action,
            eraf_outputs=changed_outputs,
            proprio=proprio,
        )
        self.assertFalse(torch.equal(learned, changed))
        bypass_outputs = dict(changed_outputs)
        bypass_outputs["audit_bypass_bridge"] = torch.tensor(True)
        bypass, bypass_residual, _ = adapter(
            candidate_action=action,
            eraf_outputs=bypass_outputs,
            proprio=proprio,
        )
        self.assertTrue(torch.equal(bypass, action))
        self.assertTrue(
            torch.equal(bypass_residual, torch.zeros_like(bypass_residual))
        )

    def test_v917_geometry_action_adapter_accepts_bfloat16_sidecar_inputs(self):
        adapter = PhaseConditionedERAFGeometryActionAdapter(
            action_dim=7,
            proprio_dim=8,
            hidden_dim=16,
            max_clauses=4,
            max_abs=0.25,
        )
        action = torch.randn(2, 6, 7, dtype=torch.bfloat16)
        proprio = torch.randn(2, 8, dtype=torch.bfloat16)
        outputs = {
            name: value.to(dtype=torch.bfloat16)
            if value.is_floating_point()
            else value
            for name, value in self._v915_bridge_outputs().items()
        }
        deployed, residual, metrics = adapter(
            candidate_action=action,
            eraf_outputs=outputs,
            proprio=proprio,
        )
        self.assertEqual(deployed.dtype, torch.bfloat16)
        self.assertEqual(residual.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(deployed, action))
        self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))
        self.assertTrue(
            torch.isfinite(metrics["pgc_v917_geometry_route_confidence"])
        )

    def test_v915_action_grounding_is_zero_init_and_anchor_connected(self):
        bridge = PhaseConditionedERAFActionBridge(
            goal_dim=8,
            eraf_hidden_dim=8,
            hidden_dim=8,
            num_heads=2,
            max_clauses=4,
        )
        queries = torch.randn(2, 3, 8)
        outputs = self._v915_bridge_outputs()
        initial, metrics = bridge(goal_queries=queries, eraf_outputs=outputs)
        self.assertTrue(torch.equal(initial, queries))
        self.assertEqual(
            float(metrics["pgc_v915_action_grounding_query_delta_rms"]), 0.0
        )
        bridge.query_delta_projection.weight.data.normal_(std=0.1)
        learned, _ = bridge(goal_queries=queries, eraf_outputs=outputs)
        changed_outputs = dict(outputs)
        changed_outputs["goal_anchor"] = outputs["goal_anchor"] + 0.25
        changed, _ = bridge(
            goal_queries=queries, eraf_outputs=changed_outputs
        )
        self.assertFalse(torch.equal(learned, changed))
        swapped_outputs = dict(outputs)
        swap = torch.tensor([1, 0, 2, 3])
        for name, value in outputs.items():
            if value.ndim >= 2 and value.shape[1] == 4:
                swapped_outputs[name] = value.index_select(1, swap)
        swapped, _ = bridge(
            goal_queries=queries, eraf_outputs=swapped_outputs
        )
        self.assertFalse(torch.equal(learned, swapped))

    def test_v915_action_grounding_accepts_fp32_geometry_with_bfloat16_model(self):
        bridge = PhaseConditionedERAFActionBridge(
            goal_dim=8,
            eraf_hidden_dim=8,
            hidden_dim=8,
            num_heads=2,
            max_clauses=4,
        ).to(dtype=torch.bfloat16)
        queries = torch.randn(2, 3, 8, dtype=torch.bfloat16)
        outputs = self._v915_bridge_outputs()
        for name in ("subject_token", "reference_token", "relation_hidden"):
            outputs[name] = outputs[name].to(torch.bfloat16)
        routed, metrics = bridge(goal_queries=queries, eraf_outputs=outputs)
        self.assertEqual(routed.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(routed, queries))
        self.assertTrue(
            torch.isfinite(
                metrics["pgc_v915_action_grounding_goal_anchor_norm"]
            )
        )

    def test_v915_causal_ranking_prefers_correct_action(self):
        model = tiny_pgc_fastwam(
            version=9,
            v9_stage="action",
            v9_grounding_objective_version=15,
            v9_completion_only_memory=True,
            v9_action_joint_training=True,
        )
        batch = 2
        correct = torch.zeros(batch, 4, 7)
        target = torch.zeros_like(correct)
        negative = torch.full_like(correct, 0.5)
        clause_valid = torch.tensor(
            [[True, True, False, False], [True, True, False, False]]
        )
        target_labels = {
            "clause_valid": clause_valid,
            "subject_entity_ids": torch.tensor([[1, 2, -1, -1]] * batch),
            "reference_entity_ids": torch.tensor([[3, 4, -1, -1]] * batch),
            "goal_anchor_valid": clause_valid,
            "goal_anchors": torch.zeros(batch, 4, 3),
        }
        source_labels = {
            "clause_valid": clause_valid,
            "subject_entity_ids": torch.tensor([[5, 6, -1, -1]] * batch),
            "reference_entity_ids": torch.tensor([[7, 8, -1, -1]] * batch),
            "goal_anchor_valid": clause_valid,
            "goal_anchors": torch.ones(batch, 4, 3),
        }
        loss, metrics = model._compute_policy_guard_v915_causal_action_loss(
            correct_action=correct,
            negative_actions={
                kind: negative
                for kind in ("subject", "reference", "anchor", "clause")
            },
            target_action=target,
            action_is_pad=None,
            target_labels=target_labels,
            source_labels=source_labels,
            is_counterfactual=torch.ones(batch, dtype=torch.bool),
            direct_action_valid=torch.ones(batch, dtype=torch.bool),
            paired_language_valid=torch.ones(batch, dtype=torch.bool),
        )
        self.assertEqual(float(loss), 0.0)
        self.assertEqual(
            float(metrics["pgc_v915_action_causal_active_kinds"]), 4.0
        )
        for kind in ("subject", "reference", "anchor", "clause"):
            self.assertEqual(
                float(metrics[f"pgc_v915_{kind}_correct_action_win_rate"]),
                1.0,
            )

    def test_v916_reference_negative_uses_same_state_subject_fallback(self):
        target_outputs = self._v915_bridge_outputs(batch_size=1)
        source_outputs = self._v915_bridge_outputs(batch_size=1)
        target_outputs["reference_token"].fill_(1.0)
        target_outputs["reference_position"].fill_(1.0)
        target_outputs["subject_token"].fill_(2.0)
        target_outputs["subject_position"].fill_(2.0)
        source_outputs["reference_token"].fill_(3.0)
        source_outputs["reference_position"].fill_(3.0)
        clause_valid = torch.tensor([[True, True, False, False]])
        target_labels = {
            "clause_valid": clause_valid,
            "subject_entity_ids": torch.tensor([[1, 2, -1, -1]]),
            "reference_entity_ids": torch.tensor([[9, 9, -1, -1]]),
        }
        source_labels = {
            "clause_valid": clause_valid,
            "subject_entity_ids": torch.tensor([[4, 5, -1, -1]]),
            # The source shares the same container, so using its mask/pose is
            # not a genuine wrong-reference intervention.
            "reference_entity_ids": torch.tensor([[9, 9, -1, -1]]),
        }
        negative = FastWAM._policy_guard_v915_negative_eraf_outputs(
            target_outputs=target_outputs,
            source_outputs=source_outputs,
            kind="reference",
            target_labels=target_labels,
            source_labels=source_labels,
            reference_subject_fallback=True,
        )
        torch.testing.assert_close(
            negative["reference_token"][:, :2],
            target_outputs["subject_token"][:, :2],
        )
        torch.testing.assert_close(
            negative["reference_position"][:, :2],
            target_outputs["subject_position"][:, :2],
        )
        torch.testing.assert_close(
            negative["reference_token"][:, 2:],
            target_outputs["reference_token"][:, 2:],
        )

    def test_v916_reference_fallback_is_causally_eligible(self):
        model = tiny_pgc_fastwam(
            version=9,
            v9_stage="action",
            v9_grounding_objective_version=16,
            v9_completion_only_memory=True,
            v9_action_joint_training=True,
        )
        clause_valid = torch.tensor([[True, False, False, False]])
        target_labels = {
            "clause_valid": clause_valid,
            "subject_entity_ids": torch.tensor([[1, -1, -1, -1]]),
            "reference_entity_ids": torch.tensor([[9, -1, -1, -1]]),
            "goal_anchor_valid": clause_valid,
            "goal_anchors": torch.zeros(1, 4, 3),
        }
        source_labels = {
            "clause_valid": clause_valid,
            "subject_entity_ids": torch.tensor([[2, -1, -1, -1]]),
            "reference_entity_ids": torch.tensor([[9, -1, -1, -1]]),
            "goal_anchor_valid": clause_valid,
            "goal_anchors": torch.zeros(1, 4, 3),
        }
        correct = torch.zeros(1, 4, 7)
        loss, metrics = model._compute_policy_guard_v915_causal_action_loss(
            correct_action=correct,
            negative_actions={"reference": torch.ones_like(correct)},
            target_action=correct,
            action_is_pad=None,
            target_labels=target_labels,
            source_labels=source_labels,
            is_counterfactual=torch.ones(1, dtype=torch.bool),
            direct_action_valid=torch.ones(1, dtype=torch.bool),
            paired_language_valid=torch.ones(1, dtype=torch.bool),
        )
        self.assertEqual(float(loss), 0.0)
        self.assertEqual(float(metrics["pgc_v915_reference_eligible_rate"]), 1.0)
        self.assertEqual(
            float(metrics["pgc_v916_reference_subject_fallback_rate"]), 1.0
        )

    def test_v914_to_v915_action_grounding_upgrade_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_path = root / "base.pt"
            v914_path = root / "v914.pt"
            v915_path = root / "v915.pt"
            v914 = tiny_pgc_fastwam(
                version=9,
                v9_stage="action",
                v9_grounding_objective_version=14,
                v9_completion_only_memory=True,
                v9_action_joint_training=True,
            )
            torch.save(
                {"format": "fastwam_full_v1", "mot": v914.mot.state_dict()},
                base_path,
            )
            v914.load_checkpoint(base_path)
            v914.save_checkpoint(v914_path, step=11250)

            v915 = tiny_pgc_fastwam(
                version=9,
                v9_stage="action",
                v9_grounding_objective_version=15,
                v9_completion_only_memory=True,
                v9_action_joint_training=True,
            )
            v915.load_checkpoint(v914_path)
            bridge = v915.policy_guard_modules["eraf_action_grounding_bridge"]
            self.assertTrue(
                torch.equal(
                    bridge.query_delta_projection.weight,
                    torch.zeros_like(bridge.query_delta_projection.weight),
                )
            )
            for name, value in v914.policy_guard_modules.state_dict().items():
                self.assertTrue(
                    torch.equal(value, v915.policy_guard_modules.state_dict()[name]),
                    name,
                )
            v915.prepare_trainable_parameters()
            trainable_modules = {
                name.split(".")[1]
                for name, parameter in v915.named_parameters()
                if parameter.requires_grad
                and name.startswith("policy_guard_modules.")
            }
            self.assertEqual(
                trainable_modules,
                {
                    "entity_relation_affordance",
                    "action_chunk_proposal",
                    "eraf_action_grounding_bridge",
                },
            )
            rates = {
                group["pgc_v9_group"]: group["lr"]
                for group in v915.policy_guard_optimizer_groups(5.0e-5)
            }
            self.assertEqual(rates["eraf_action_grounding_bridge"], 1.0e-4)
            v915.save_checkpoint(v915_path, step=13250)
            payload = torch.load(v915_path, map_location="cpu", weights_only=False)
            self.assertEqual(
                payload["architecture_metadata"]["eraf_action_grounding_contract"],
                "separate_subject_reference_relation_grasp_goal_interaction_"
                "displacement_tokens_zero_init_v9_14_exact",
            )
            restored = tiny_pgc_fastwam(
                version=9,
                v9_stage="action",
                v9_grounding_objective_version=15,
                v9_completion_only_memory=True,
                v9_action_joint_training=True,
            )
            restored.load_checkpoint(v915_path)
            for name, value in v915.policy_guard_modules.state_dict().items():
                self.assertTrue(
                    torch.equal(
                        value, restored.policy_guard_modules.state_dict()[name]
                    ),
                    name,
                )

    def test_v915_to_v916_semantic_causal_upgrade_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_path = root / "base.pt"
            v915_path = root / "v915.pt"
            v916_path = root / "v916.pt"
            v915 = tiny_pgc_fastwam(
                version=9,
                v9_stage="action",
                v9_grounding_objective_version=15,
                v9_completion_only_memory=True,
                v9_action_joint_training=True,
            )
            torch.save(
                {"format": "fastwam_full_v1", "mot": v915.mot.state_dict()},
                base_path,
            )
            v915.load_checkpoint(base_path)
            v915.policy_guard_modules[
                "eraf_action_grounding_bridge"
            ].query_delta_projection.weight.data.normal_(std=0.01)
            v915.save_checkpoint(v915_path, step=13250)

            v916 = tiny_pgc_fastwam(
                version=9,
                v9_stage="action",
                v9_grounding_objective_version=16,
                v9_completion_only_memory=True,
                v9_action_joint_training=True,
            )
            v916.load_checkpoint(v915_path)
            for name, value in v915.policy_guard_modules.state_dict().items():
                self.assertTrue(
                    torch.equal(value, v916.policy_guard_modules.state_dict()[name]),
                    name,
                )
            v916.prepare_trainable_parameters()
            trainable = {
                name
                for name, parameter in v916.named_parameters()
                if parameter.requires_grad
            }
            self.assertTrue(trainable)
            self.assertTrue(
                all(
                    name.startswith(
                        "policy_guard_modules.eraf_action_grounding_bridge."
                    )
                    for name in trainable
                )
            )
            groups = v916.policy_guard_optimizer_groups(2.0e-5)
            self.assertEqual(
                [group["pgc_v9_group"] for group in groups],
                ["eraf_action_grounding_bridge"],
            )
            v916.save_checkpoint(v916_path, step=13750)
            payload = torch.load(v916_path, map_location="cpu", weights_only=False)
            metadata = payload["architecture_metadata"]
            self.assertEqual(
                metadata["eraf_action_trainable_scope"],
                "semantic_causal_action_grounding_bridge_only",
            )
            self.assertEqual(
                metadata["eraf_action_semantic_negative_contract"],
                "joint_valid_entity_id_swap_with_same_state_subject_as_"
                "reference_fallback_plus_clause_swap",
            )
            restored = tiny_pgc_fastwam(
                version=9,
                v9_stage="action",
                v9_grounding_objective_version=16,
                v9_completion_only_memory=True,
                v9_action_joint_training=True,
            )
            restored.load_checkpoint(v916_path)
            for name, value in v916.policy_guard_modules.state_dict().items():
                self.assertTrue(
                    torch.equal(value, restored.policy_guard_modules.state_dict()[name]),
                    name,
                )

    def test_v916_to_v917_direct_geometry_action_upgrade_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_path = root / "base.pt"
            v916_path = root / "v916.pt"
            v917_path = root / "v917.pt"
            v916 = tiny_pgc_fastwam(
                version=9,
                v9_stage="action",
                v9_grounding_objective_version=16,
                v9_completion_only_memory=True,
                v9_action_joint_training=True,
            )
            torch.save(
                {"format": "fastwam_full_v1", "mot": v916.mot.state_dict()},
                base_path,
            )
            v916.load_checkpoint(base_path)
            v916.save_checkpoint(v916_path, step=13750)

            v917 = tiny_pgc_fastwam(
                version=9,
                v9_stage="action",
                v9_grounding_objective_version=17,
                v9_completion_only_memory=True,
                v9_action_joint_training=True,
            )
            v917.load_checkpoint(v916_path)
            old_state = v916.policy_guard_modules.state_dict()
            new_state = v917.policy_guard_modules.state_dict()
            for name, value in old_state.items():
                self.assertTrue(torch.equal(value, new_state[name]), name)
            adapter = v917.policy_guard_modules[
                "eraf_geometry_action_adapter"
            ]
            self.assertTrue(
                torch.equal(
                    adapter.output_projection.weight,
                    torch.zeros_like(adapter.output_projection.weight),
                )
            )
            v917.prepare_trainable_parameters()
            trainable = {
                name
                for name, parameter in v917.named_parameters()
                if parameter.requires_grad
            }
            self.assertTrue(trainable)
            self.assertTrue(
                all(
                    name.startswith(
                        "policy_guard_modules.eraf_geometry_action_adapter."
                    )
                    for name in trainable
                )
            )
            groups = v917.policy_guard_optimizer_groups(2.0e-5)
            self.assertEqual(
                [group["pgc_v9_group"] for group in groups],
                ["eraf_geometry_action_adapter"],
            )
            v917.save_checkpoint(v917_path, step=14250)
            metadata = torch.load(
                v917_path, map_location="cpu", weights_only=False
            )["architecture_metadata"]
            self.assertEqual(
                metadata["eraf_action_trainable_scope"],
                "phase_conditioned_relative_geometry_action_adapter_only",
            )
            restored = tiny_pgc_fastwam(
                version=9,
                v9_stage="action",
                v9_grounding_objective_version=17,
                v9_completion_only_memory=True,
                v9_action_joint_training=True,
            )
            restored.load_checkpoint(v917_path)
            for name, value in new_state.items():
                self.assertTrue(
                    torch.equal(
                        value, restored.policy_guard_modules.state_dict()[name]
                    ),
                    name,
                )

    def test_v917_to_v918_phase_residual_upgrade_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_path = root / "base.pt"
            v917_path = root / "v917.pt"
            v918_path = root / "v918.pt"
            v917 = tiny_pgc_fastwam(
                version=9,
                v9_stage="action",
                v9_grounding_objective_version=17,
                v9_completion_only_memory=True,
                v9_action_joint_training=True,
            )
            torch.save(
                {"format": "fastwam_full_v1", "mot": v917.mot.state_dict()},
                base_path,
            )
            v917.load_checkpoint(base_path)
            v917.save_checkpoint(v917_path, step=14250)
            old_state = {
                name: value.clone()
                for name, value in v917.policy_guard_modules.state_dict().items()
            }

            v918 = tiny_pgc_fastwam(
                version=9,
                v9_stage="action",
                v9_grounding_objective_version=18,
                v9_completion_only_memory=True,
                v9_action_joint_training=True,
            )
            v918.load_checkpoint(v917_path)
            new_state = v918.policy_guard_modules.state_dict()
            self.assertEqual(set(old_state), set(new_state))
            for name, value in old_state.items():
                self.assertTrue(torch.equal(value, new_state[name]), name)
            v918.prepare_trainable_parameters()
            trainable = {
                name
                for name, parameter in v918.named_parameters()
                if parameter.requires_grad
            }
            self.assertTrue(trainable)
            self.assertTrue(
                all(
                    name.startswith(
                        "policy_guard_modules.eraf_geometry_action_adapter."
                    )
                    for name in trainable
                )
            )
            v918.save_checkpoint(v918_path, step=15250)
            metadata = torch.load(
                v918_path, map_location="cpu", weights_only=False
            )["architecture_metadata"]
            self.assertEqual(
                metadata["eraf_action_phase_residual_contract"],
                "phase_balanced_bounded_expert_minus_frozen_v9_17_"
                "candidate_prefix_residual_imitation",
            )
            self.assertEqual(
                metadata["eraf_action_trainable_scope"],
                "phase_conditioned_geometry_adapter_only_with_phase_"
                "balanced_residual_imitation",
            )

    def test_v918_to_v919_hard_phase_servo_upgrade_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_path = root / "base.pt"
            v918_path = root / "v918.pt"
            v919_path = root / "v919.pt"
            v918 = tiny_pgc_fastwam(
                version=9,
                v9_stage="action",
                v9_grounding_objective_version=18,
                v9_completion_only_memory=True,
                v9_action_joint_training=True,
            )
            torch.save(
                {"format": "fastwam_full_v1", "mot": v918.mot.state_dict()},
                base_path,
            )
            v918.load_checkpoint(base_path)
            v918.save_checkpoint(v918_path, step=15250)
            old_state = {
                name: value.clone()
                for name, value in v918.policy_guard_modules.state_dict().items()
            }

            v919 = tiny_pgc_fastwam(
                version=9,
                v9_stage="action",
                v9_grounding_objective_version=19,
                v9_completion_only_memory=True,
                v9_action_joint_training=True,
            )
            v919.load_checkpoint(v918_path)
            new_state = v919.policy_guard_modules.state_dict()
            for name, value in old_state.items():
                self.assertTrue(torch.equal(value, new_state[name]), name)
            added = set(new_state) - set(old_state)
            self.assertTrue(added)
            self.assertTrue(
                all(name.startswith("eraf_hard_routed_phase_servo.") for name in added)
            )
            v919.prepare_trainable_parameters()
            trainable = {
                name
                for name, parameter in v919.named_parameters()
                if parameter.requires_grad
            }
            self.assertTrue(trainable)
            self.assertTrue(
                all(
                    name.startswith(
                        "policy_guard_modules.eraf_hard_routed_phase_servo."
                    )
                    for name in trainable
                )
            )
            v919.save_checkpoint(v919_path, step=16250)
            metadata = torch.load(
                v919_path, map_location="cpu", weights_only=False
            )["architecture_metadata"]
            self.assertEqual(
                metadata["eraf_action_phase_servo_contract"],
                "hard_single_clause_explicit_affine_eef_phase_specific_"
                "positive_cartesian_gain_with_legacy_suppression",
            )
            self.assertEqual(
                metadata["eraf_action_trainable_scope"],
                "hard_routed_phase_servo_only",
            )

    def test_v919_to_v920_phase_compatible_waypoint_upgrade_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_path = root / "base.pt"
            v919_path = root / "v919.pt"
            v920_path = root / "v920.pt"
            v919 = tiny_pgc_fastwam(
                version=9,
                v9_stage="action",
                v9_grounding_objective_version=19,
                v9_completion_only_memory=True,
                v9_action_joint_training=True,
            )
            torch.save(
                {"format": "fastwam_full_v1", "mot": v919.mot.state_dict()},
                base_path,
            )
            v919.load_checkpoint(base_path)
            v919.save_checkpoint(v919_path, step=16250)
            old_state = {
                name: value.clone()
                for name, value in v919.policy_guard_modules.state_dict().items()
            }

            v920 = tiny_pgc_fastwam(
                version=9,
                v9_stage="action",
                v9_grounding_objective_version=20,
                v9_completion_only_memory=True,
                v9_action_joint_training=True,
            )
            v920.load_checkpoint(v919_path)
            new_state = v920.policy_guard_modules.state_dict()
            for name, value in old_state.items():
                self.assertTrue(torch.equal(value, new_state[name]), name)
            added = set(new_state) - set(old_state)
            self.assertTrue(added)
            self.assertTrue(
                all(
                    name.startswith("eraf_phase_compatible_waypoint_adapter.")
                    for name in added
                )
            )
            v920.prepare_trainable_parameters()
            trainable = {
                name
                for name, parameter in v920.named_parameters()
                if parameter.requires_grad
            }
            self.assertTrue(trainable)
            self.assertTrue(
                all(
                    name.startswith(
                        "policy_guard_modules."
                        "eraf_phase_compatible_waypoint_adapter."
                    )
                    for name in trainable
                )
            )
            v920.save_checkpoint(v920_path, step=17250)
            metadata = torch.load(
                v920_path, map_location="cpu", weights_only=False
            )["architecture_metadata"]
            self.assertEqual(
                metadata["eraf_action_waypoint_contract"],
                "hard_clause_phase_compatible_positive_progress_local_tangent_"
                "waypoint_with_privileged_training_only_compatibility_labels",
            )
            self.assertEqual(
                metadata["eraf_action_trainable_scope"],
                "phase_compatible_local_waypoint_adapter_only",
            )

    def test_v920_to_v921_expert_alignment_upgrade_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_path = root / "base.pt"
            v920_path = root / "v920.pt"
            v921_path = root / "v921.pt"
            v920 = tiny_pgc_fastwam(
                version=9,
                v9_stage="action",
                v9_grounding_objective_version=20,
                v9_completion_only_memory=True,
                v9_action_joint_training=True,
            )
            torch.save(
                {"format": "fastwam_full_v1", "mot": v920.mot.state_dict()},
                base_path,
            )
            v920.load_checkpoint(base_path)
            v920.save_checkpoint(v920_path, step=17250)
            old_state = {
                name: value.clone()
                for name, value in v920.policy_guard_modules.state_dict().items()
            }

            v921 = tiny_pgc_fastwam(
                version=9,
                v9_stage="action",
                v9_grounding_objective_version=21,
                v9_completion_only_memory=True,
                v9_action_joint_training=True,
            )
            v921.load_checkpoint(v920_path)
            new_state = v921.policy_guard_modules.state_dict()
            for name, value in old_state.items():
                self.assertTrue(torch.equal(value, new_state[name]), name)
            added = set(new_state) - set(old_state)
            self.assertTrue(added)
            self.assertTrue(
                all(
                    name.startswith("eraf_phase_expert_residual_adapter.")
                    for name in added
                )
            )
            self.assertTrue(
                torch.equal(
                    v921.policy_guard_modules[
                        "eraf_phase_expert_residual_adapter"
                    ].phase_output.weight,
                    torch.zeros_like(
                        v921.policy_guard_modules[
                            "eraf_phase_expert_residual_adapter"
                        ].phase_output.weight
                    ),
                )
            )
            v921.prepare_trainable_parameters()
            trainable = {
                name
                for name, parameter in v921.named_parameters()
                if parameter.requires_grad
            }
            self.assertTrue(trainable)
            self.assertTrue(
                all(
                    name.startswith(
                        "policy_guard_modules."
                        "eraf_phase_expert_residual_adapter."
                    )
                    for name in trainable
                )
            )
            v921.save_checkpoint(v921_path, step=18250)
            metadata = torch.load(
                v921_path, map_location="cpu", weights_only=False
            )["architecture_metadata"]
            self.assertEqual(
                metadata["eraf_action_trainable_scope"],
                "phase_specific_privileged_expert_residual_adapter_only",
            )
            self.assertEqual(
                metadata["eraf_action_expert_alignment_contract"],
                "training_only_privileged_phase_anchor_teacher_plus_deployed_"
                "full_action_prefix_residual_and_semantic_causal_ranking",
            )

    def test_v921_to_v922_clause_ranking_upgrade_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_path = root / "base.pt"
            v921_path = root / "v921.pt"
            v922_path = root / "v922.pt"
            v921 = tiny_pgc_fastwam(
                version=9,
                v9_stage="action",
                v9_grounding_objective_version=21,
                v9_completion_only_memory=True,
                v9_action_joint_training=True,
            )
            torch.save(
                {"format": "fastwam_full_v1", "mot": v921.mot.state_dict()},
                base_path,
            )
            v921.load_checkpoint(base_path)
            v921.save_checkpoint(v921_path, step=18250)
            old_state = {
                name: value.clone()
                for name, value in v921.policy_guard_modules.state_dict().items()
            }

            v922 = tiny_pgc_fastwam(
                version=9,
                v9_stage="action",
                v9_grounding_objective_version=22,
                v9_completion_only_memory=True,
                v9_action_joint_training=True,
            )
            v922.load_checkpoint(v921_path)
            new_state = v922.policy_guard_modules.state_dict()
            self.assertEqual(set(old_state), set(new_state))
            for name, value in old_state.items():
                self.assertTrue(torch.equal(value, new_state[name]), name)

            v922.prepare_trainable_parameters()
            trainable = {
                name
                for name, parameter in v922.named_parameters()
                if parameter.requires_grad
            }
            self.assertTrue(trainable)
            self.assertTrue(
                all(
                    name.startswith(
                        "policy_guard_modules."
                        "eraf_phase_expert_residual_adapter."
                    )
                    for name in trainable
                )
            )
            v922.save_checkpoint(v922_path, step=18750)
            metadata = torch.load(
                v922_path, map_location="cpu", weights_only=False
            )["architecture_metadata"]
            self.assertEqual(
                metadata["eraf_action_joint_contract"],
                "frozen_v920_stack_plus_phase_specific_expert_adapter_with_"
                "balanced_final_action_clause_ranking",
            )
            self.assertEqual(
                metadata["eraf_action_clause_ranking_contract"],
                "coherent_same_state_clause_swap_final_expert_prefix_mse_"
                "ranking_balanced_over_approach_transport_release",
            )


    def test_v913_to_v914_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_path = root / "base.pt"
            v913_path = root / "v913.pt"
            v914_path = root / "v914.pt"
            v913 = tiny_pgc_fastwam(
                version=9,
                v9_stage="grounding",
                v9_grounding_objective_version=14,
            )
            torch.save(
                {"format": "fastwam_full_v1", "mot": v913.mot.state_dict()},
                base_path,
            )
            v913.load_checkpoint(base_path)
            v913.save_checkpoint(v913_path, step=7250)

            v914 = tiny_pgc_fastwam(
                version=9,
                v9_stage="action",
                v9_grounding_objective_version=14,
                v9_completion_only_memory=True,
                v9_action_joint_training=True,
            )
            v914.load_checkpoint(v913_path)
            v914.save_checkpoint(v914_path, step=11250)
            payload = torch.load(v914_path, map_location="cpu", weights_only=False)
            metadata = payload["architecture_metadata"]
            self.assertEqual(
                metadata["deployment_inputs"],
                "rgb_language_proprio_completed_clause_bitset",
            )
            self.assertEqual(
                metadata["eraf_policy_state_contract"],
                "monotonic_completed_bitset_no_pending_holding_retry_recurrence",
            )
            self.assertEqual(
                metadata["eraf_action_joint_contract"],
                "frozen_eraf_perception_plus_action_bridge_and_proposal",
            )
            self.assertEqual(
                metadata["eraf_role_adapter_trainable_scope"],
                "frozen_eraf_perception_action_bridge_plus_proposal",
            )

            restored = tiny_pgc_fastwam(
                version=9,
                v9_stage="action",
                v9_grounding_objective_version=14,
                v9_completion_only_memory=True,
                v9_action_joint_training=True,
            )
            restored.load_checkpoint(v914_path)
            expected = v914.policy_guard_modules.state_dict()
            actual = restored.policy_guard_modules.state_dict()
            self.assertEqual(expected.keys(), actual.keys())
            for name, value in expected.items():
                self.assertTrue(torch.equal(value, actual[name]), name)

    def test_verifier_stage_builds_explicit_negatives_with_frozen_root(self):
        model = tiny_pgc_fastwam(version=9, v9_stage="verifier")
        model.prepare_trainable_parameters()
        self.assertFalse(model.training)
        final_video = torch.randn(2, 8, 16)
        neutral_video = torch.randn(2, 8, 16)
        context = torch.randn(2, 5, 10)
        context_mask = torch.ones(2, 5, dtype=torch.bool)
        _, _, outputs, _ = model._encode_policy_guard_eraf(
            final_video_hidden=final_video,
            current_visual_hidden=neutral_video,
            video_tokens_per_frame=8,
            context=context,
            context_mask=context_mask,
            language_context_len=5,
        )
        self.assertIn("wrong_entity_goal_queries", outputs)
        self.assertIn("wrong_relation_goal_queries", outputs)


class PGCERAFGroundingGateTest(unittest.TestCase):
    def test_v910_gate_uses_all_entity_exclusive_roles(self):
        def clause(all_entity_correct: bool):
            return {
                "full_mask_correct": True,
                "exclusive_mask_valid": True,
                "exclusive_mask_correct": True,
                "all_entity_exclusive_valid": True,
                "all_entity_exclusive_correct": all_entity_correct,
                "mask_overlap_iou": 0.0,
                "subject_overlap_fraction": 0.0,
                "reference_overlap_fraction": 0.0,
                "full_subject_margin": 0.2,
                "full_reference_margin": 0.2,
                "exclusive_subject_margin": 0.2,
                "exclusive_reference_margin": 0.2,
                "dataset_kind": "native",
                "predicate": "on",
                "task": "compound-task",
            }

        record = {
            "subject_top1_hits": [True, True],
            "reference_top1_hits": [True, True],
            "role_swap_correct": [True, True],
            "relation_targets": [1, 1],
            "relation_predictions": [1, 1],
            "goal_anchor_errors_m": [0.01, 0.02],
            "clause_exact": True,
            "clause_count": 2,
            "role_audit_clauses": [clause(True), clause(False)],
            "clause_exact_components": {
                "active": True,
                "predicate": True,
                "subject_localization": True,
                "reference_localization": True,
                "semantic_role": False,
            },
        }
        report = compute_grounding_gate_report([record] * 5)
        self.assertFalse(report["passed"])
        self.assertEqual(
            report["metrics"]["semantic_role_gate_scope"],
            "exclusive_all_entity",
        )
        self.assertEqual(
            report["metrics"]["pairwise_exclusive_role_accuracy"], 1.0
        )
        self.assertEqual(
            report["metrics"]["all_entity_exclusive_role_accuracy"], 0.5
        )
        self.assertEqual(
            report["multi_clause_failure_partition"]["semantic_role"], 5
        )

    def test_gate_requires_every_declared_threshold(self):
        def role_clause(correct: bool):
            return {
                "full_mask_correct": correct,
                "exclusive_mask_valid": True,
                "exclusive_mask_correct": correct,
                "mask_overlap_iou": 0.1,
                "subject_overlap_fraction": 0.1,
                "reference_overlap_fraction": 0.1,
                "full_subject_margin": 0.2 if correct else -0.2,
                "full_reference_margin": 0.2 if correct else -0.2,
                "exclusive_subject_margin": 0.2 if correct else -0.2,
                "exclusive_reference_margin": 0.2 if correct else -0.2,
                "dataset_kind": "native",
                "predicate": "in",
                "task": "task-a",
            }

        good = {
            "subject_top1_hits": [True] * 9 + [False],
            "reference_top1_hits": [True] * 9 + [False],
            "role_swap_correct": [True] * 9 + [False],
            "relation_targets": [1, 2, 1, 2],
            "relation_predictions": [1, 2, 1, 2],
            "goal_anchor_errors_m": [0.01, 0.03, 0.05],
            "clause_exact": True,
            "clause_count": 2,
            "role_audit_clauses": [
                *[role_clause(True) for _ in range(9)],
                role_clause(False),
            ],
        }
        report = compute_grounding_gate_report([good] * 5)
        self.assertTrue(report["passed"])
        self.assertEqual(report["diagnosis"], "grounding_gate_pass")
        clause_bad = dict(good, clause_exact=False)
        report = compute_grounding_gate_report([clause_bad] * 5)
        self.assertFalse(report["passed"])
        self.assertEqual(report["diagnosis"], "multi_clause_activation_failure")
        bad = dict(good, relation_predictions=[2, 1, 2, 1])
        report = compute_grounding_gate_report([bad] * 5)
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["relation_macro_f1_at_least_90pct"])

        v99_good = dict(
            good,
            single_visible_view_selection_correct=[True, True],
            clause_scheduler_correct=[True],
            clause_scheduler_confidence=[0.95],
        )
        report = compute_grounding_gate_report(
            [v99_good] * 5,
            require_view_scheduler=True,
        )
        self.assertTrue(report["passed"])
        self.assertEqual(
            report["metrics"]["single_visible_view_selection_accuracy"], 1.0
        )
        v99_bad = dict(
            v99_good,
            single_visible_view_selection_correct=[False, True],
        )
        report = compute_grounding_gate_report(
            [v99_bad] * 5,
            require_view_scheduler=True,
        )
        self.assertFalse(report["passed"])
        self.assertFalse(
            report["checks"]["single_visible_view_selection_at_least_80pct"]
        )

    def test_role_residual_audit_separates_overlap_from_binding_failure(self):
        base = {
            "subject_top1_hits": [True],
            "reference_top1_hits": [True],
            "role_swap_correct": [False],
            "relation_targets": [1],
            "relation_predictions": [1],
            "goal_anchor_errors_m": [0.01],
            "clause_exact": True,
            "clause_count": 2,
        }

        def clause(*, exclusive_correct, predicate="in", task="task-a"):
            return {
                "full_mask_correct": False,
                "exclusive_mask_valid": True,
                "exclusive_mask_correct": exclusive_correct,
                "mask_overlap_iou": 0.75,
                "subject_overlap_fraction": 0.80,
                "reference_overlap_fraction": 0.70,
                "full_subject_margin": -0.10,
                "full_reference_margin": -0.05,
                "exclusive_subject_margin": 0.20 if exclusive_correct else -0.20,
                "exclusive_reference_margin": 0.30 if exclusive_correct else -0.30,
                "dataset_kind": "counterfactual",
                "predicate": predicate,
                "task": task,
            }

        overlap_records = [
            dict(base, role_audit_clauses=[clause(exclusive_correct=True)])
            for _ in range(5)
        ]
        audit = compute_grounding_gate_report(overlap_records)[
            "role_residual_audit"
        ]
        overlap_report = compute_grounding_gate_report(overlap_records)
        self.assertTrue(overlap_report["passed"])
        self.assertFalse(
            overlap_report["diagnostics"]["full_mask_role_swap_at_least_90pct"]
        )
        self.assertEqual(
            audit["diagnosis"],
            "exclusive_role_gate_pass_full_mask_overlap_diagnostic",
        )
        self.assertEqual(
            audit["overall"]["full_mask_failure_partition"][
                "exclusive_recovers"
            ],
            5,
        )
        self.assertIn("counterfactual", audit["by_dataset_kind"])
        self.assertIn("in", audit["by_predicate"])
        self.assertIn("task-a", audit["by_task"])

        binding_records = [
            dict(base, role_audit_clauses=[clause(exclusive_correct=False)])
            for _ in range(5)
        ]
        audit = compute_grounding_gate_report(binding_records)[
            "role_residual_audit"
        ]
        self.assertFalse(compute_grounding_gate_report(binding_records)["passed"])
        self.assertEqual(
            audit["diagnosis"], "role_binding_generalization_failure"
        )
        self.assertEqual(
            audit["overall"]["full_mask_failure_partition"][
                "exclusive_still_wrong"
            ],
            5,
        )


if __name__ == "__main__":
    unittest.main()
