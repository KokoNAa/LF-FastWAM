import hashlib
import json
import tempfile
import unittest
from collections import Counter, OrderedDict
from inspect import signature
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from fastwam.datasets.lerobot.robot_video_dataset import (
    RobotVideoDataset,
    build_pgc_v96_sample_plan,
    build_pgc_v9_sample_indices,
)
from fastwam.datasets.lerobot.base_lerobot_dataset import BaseLerobotDataset
from fastwam.datasets.pgc_libero import (
    PGC_ACTION_CONVENTION_FASTWAM,
    PGC_ACTION_CONVENTION_LIBERO_ENV,
    PGC_ACTION_REPLAY_FASTWAM_TO_LIBERO_ENV,
    PGC_ENTITY_RELATION_ARRAY_NAMES,
    array_sha256,
    classify_strict_conflict,
    fastwam_actions_to_libero_env,
    libero_env_actions_to_fastwam,
    load_pgc_entity_relation_index,
    parse_libero_goal_clauses,
    provenance_pair,
    state_sha256,
    validate_strict_conflict_audit,
)
from fastwam.models.wan22.entity_relation_affordance import (
    ERAFLossWeights,
    EntityRelationAffordanceField,
    _balanced_bipartite_assignment_loss,
    _structured_all_entity_assignment_loss,
    entity_relation_affordance_loss,
)
from fastwam.models.wan22.policy_guard import infer_spatial_patch_grid
import scripts.build_pgc_libero_entity_relations as eraf_builder
from scripts.build_pgc_libero_entity_relations import (
    ARRAY_NAMES,
    _match_native_demo,
    _region_anchor,
)
from scripts.eval_pgc_v9_grounding_gate import compute_grounding_gate_report
from tests.test_policy_guard import tiny_pgc_fastwam
from fastwam.utils.samplers import ResumableEpochSampler


class PGCERAFParsingTest(unittest.TestCase):
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
            },
        )

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
    def test_gate_requires_every_declared_threshold(self):
        good = {
            "subject_top1_hits": [True] * 9 + [False],
            "reference_top1_hits": [True] * 9 + [False],
            "role_swap_correct": [True] * 9 + [False],
            "relation_targets": [1, 2, 1, 2],
            "relation_predictions": [1, 2, 1, 2],
            "goal_anchor_errors_m": [0.01, 0.03, 0.05],
            "clause_exact": True,
            "clause_count": 2,
        }
        report = compute_grounding_gate_report([good] * 5)
        self.assertTrue(report["passed"])
        bad = dict(good, relation_predictions=[2, 1, 2, 1])
        report = compute_grounding_gate_report([bad] * 5)
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["relation_macro_f1_at_least_90pct"])

    def test_role_residual_audit_separates_overlap_from_binding_failure(self):
        base = {
            "subject_top1_hits": [True],
            "reference_top1_hits": [True],
            "role_swap_correct": [False],
            "relation_targets": [1],
            "relation_predictions": [1],
            "goal_anchor_errors_m": [0.01],
            "clause_exact": False,
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
        self.assertEqual(audit["diagnosis"], "overlap_sensitive_full_mask_gate")
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
