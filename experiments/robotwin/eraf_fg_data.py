"""Raw state/label access behind the legacy compact replay payloads."""
from __future__ import annotations
from collections import OrderedDict
import json
from pathlib import Path

import numpy as np


def file_metadata(path):
    """Record file provenance without reading a checkpoint or camera archive."""
    path = Path(path).resolve()
    stat = path.stat()
    return {'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def verify_retention_capture(row):
    if row.get('retention_format') == 'robotwin_policy_retention_v2':
        if file_metadata(row['capture_path']) != row['capture_metadata']:
            raise ValueError('Retention capture metadata changed.')
    else:
        # Read historical records without changing their recorded protocol.
        from experiments.robotwin.eraf_fg_bridge import file_sha256
        if file_sha256(row['capture_path']) != row['capture_sha256']:
            raise ValueError('Retention capture archive changed.')


def retention_language(row):
    """Admit new retention captures only under their actually executed goal."""
    native, cf = bool(row.get('native_retention')), bool(row.get('cf_retention'))
    if native == cf:
        raise ValueError('Exactly one retention kind is required.')
    flag = 'full_native_episode_success' if native else 'full_cf_episode_success'
    if row.get(flag) is not True:
        raise ValueError('Retention capture lacks verified selected-goal success.')
    expected = 'correct' if native else 'counterfactual'
    if row.get('retention_condition', expected) != expected:
        raise ValueError('Retention condition contradicts its kind.')
    return 'source' if native else 'target'


def validate_retention_scene(rows):
    if not rows:
        raise ValueError('Empty retention scene.')
    language = retention_language(rows[0])
    fields = ('source_task', 'task_config', 'scene_seed', 'capture_path',
              'source_instruction', 'counterfactual_instruction', 'retention_format')
    if rows[0].get('retention_format') == 'robotwin_policy_retention_v2':
        fields += ('capture_metadata', 'teacher_checkpoint_metadata', 'teacher_checkpoint')
        for row in rows:
            for key, path_key in (('capture_metadata', 'capture_path'),
                                  ('teacher_checkpoint_metadata', 'teacher_checkpoint')):
                value = row.get(key, {})
                if (value.get('path') != row.get(path_key) or
                        not isinstance(value.get('size'), int) or value['size'] <= 0 or
                        not isinstance(value.get('mtime_ns'), int) or value['mtime_ns'] <= 0):
                    raise ValueError('Retention file metadata is missing or invalid.')
    else:
        fields += ('capture_sha256', 'teacher_checkpoint_sha256')
        if rows[0].get('retention_format') is not None or any(
                not row.get(key) for row in rows for key in fields[-2:]):
            raise ValueError('Unknown or incomplete historical retention provenance.')
    if any(retention_language(row) != language or any(row.get(k) != rows[0].get(k) for k in fields) for row in rows):
        raise ValueError('Retention scene mixes condition, capture, teacher, or instruction provenance.')
    if sorted(row['frame_index'] for row in rows) != list(range(len(rows))):
        raise ValueError('Retention scene has missing or duplicate captured states.')
    return language


def validate_cf_retention_coverage(rows, required_tasks, *, minimum_scenes=10):
    """Extend preservation to target tasks without dropping the existing three."""
    previous = {'place_a2b_right', 'place_burger_fries', 'stack_blocks_two'}
    allowed = previous | {'place_a2b_left', 'blocks_ranking_rgb'}
    required = set(required_tasks)
    if (len(required) != len(required_tasks) or not previous <= required <= allowed
            or minimum_scenes <= 0):
        raise ValueError('Declare unique CF-retention tasks including all three previous tasks.')
    scenes = {}
    for row in rows:
        if row.get('cf_retention'):
            retention_language(row)
            if row['replay_split'] == 'train':
                scenes.setdefault(row['source_task'], set()).add((row['task_config'], row['scene_seed']))
    if set(scenes) != required:
        raise ValueError(f'CF-retention training tasks differ from the declared coverage: {sorted(scenes)}.')
    counts = {task: len(values) for task, values in scenes.items()}
    if any(count < minimum_scenes for count in counts.values()):
        raise ValueError(f'Need {minimum_scenes} CF-retention training scenes per declared task: {counts}.')
    return counts


class RawReplay:
    def __init__(self, source_bank, *, label_cache=None, fg_geometry=False):
        self.raw = {}
        from experiments.robotwin.fg_geometry_replay import CorrectionGeometry
        self.corrections = CorrectionGeometry() if fg_geometry else None
        self.label_cache = Path(label_cache) if label_cache else None
        if self.label_cache:
            self.label_cache.mkdir(parents=True, exist_ok=True)
        self.labels = OrderedDict()
        for domain in ("demo_clean", "demo_randomized"):
            plan = json.loads((Path(source_bank) / domain / "plan.json").read_text())
            for pair in plan["pairs"]:
                for kind in ("native", "counterfactual"):
                    root = Path(pair[kind]["hdf5"]).parent.parent
                    for line in (root / "meta/pgc_episodes.jsonl").read_text().splitlines():
                        record = json.loads(line)
                        self.raw[domain, pair["pair_id"], kind, record["episode_index"]] = (
                            root / record["raw_hdf5"], record)

    def locate(self, row, language):
        kind = "native" if language == "source" else "counterfactual"
        frame = int(row.get(language + "_frame_index", row.get("frame_index", 0)))
        if "raw_paths" in row:
            return Path(row["raw_paths"][kind]), frame
        path, record = self.raw[row["task_config"], row["pair_id"], kind, row["episode_index"]]
        if int(record["scene_seed"]) != int(row["scene_seed"]):
            raise ValueError("Replay raw-file scene seed differs from the cached observation.")
        return path, frame

    def state(self, row, language):
        import h5py
        if row.get('fg_correction'):
            if self.corrections is None or language != 'target':
                raise ValueError('FG proprio requires the explicit target geometry protocol.')
            return self.corrections.state(row)
        if row.get("native_retention") or row.get("cf_retention"):
            with np.load(row["capture_path"], allow_pickle=False) as source:
                return source["state"][row["frame_index"]].copy()
        path, frame = self.locate(row, language)
        with h5py.File(path, "r") as handle:
            return handle["joint_action/vector"][frame].astype(np.float32)

    def attach_proprio(self, payload, row, policy):
        result = dict(payload, captured={k: dict(v) for k, v in payload["captured"].items()})
        for language, captured in result["captured"].items():
            state = policy._normalize_state(self.state(row, language)).to(policy.model.device)
            # The frozen state encoder must reproduce the actual cached token.
            import torch
            with torch.no_grad():
                projected = policy.model.proprio_encoder(state.to(policy.model.torch_dtype).unsqueeze(1))
            expected = captured["action_inputs"]["context"][:, -1:]
            if not torch.equal(projected, expected):
                raise ValueError("Raw proprio does not reproduce the frozen deployment state token.")
            captured["proprio"] = state
            captured["policy_guard_state"] = None
        return result

    def grounding(self, row, language):
        import h5py
        import torch
        if row.get('fg_correction'):
            if self.corrections is None:
                raise ValueError('FG labels require the explicit partial geometry protocol.')
            return self.corrections.labels(row, language)
        from scripts.build_pgc_robotwin_entity_relations import _generic_role_arrays, _entity_id, _phase_ids
        path, frame = self.locate(row, language)
        key = f"one_frame_v1:{path}:{frame}"
        if key not in self.labels:
            import hashlib
            stem = hashlib.sha256(key.encode()).hexdigest()
            target = self.label_cache / (stem + ".pt") if self.label_cache else None
            if target and target.exists():
                stored = torch.load(target, map_location="cpu", weights_only=True)
                if stored["raw_size"] != path.stat().st_size or stored["raw_mtime_ns"] != path.stat().st_mtime_ns:
                    raise ValueError("Grounding raw file changed after label preparation.")
                value = stored["labels"]
            else:
                with h5py.File(path, "r") as handle:
                    class FrameView:
                        def __contains__(self, name):
                            return name in handle

                        def __getitem__(self, name):
                            return handle[name][frame:frame + 1]

                    # Decode segmentation only at this frame. Phase labels
                    # still use the complete recorded position/truth history.
                    value = {prefix: _generic_role_arrays(handle=FrameView(), prefix=prefix,
                                entity_ids=[_entity_id(row["pair_id"] + f"/actor_{i}") for i in range(4)])
                             for prefix in ("source", "target")}
                    positions = handle["pgc_entity_state/entity_positions"][:]
                    for prefix, arrays in value.items():
                        indices = handle[f"pgc_entity_state/{prefix}_subject_indices"][0]
                        truth = handle[f"pgc_entity_state/{prefix}_predicate_truth"][:]
                        for clause in np.flatnonzero(arrays["clause_valid"][0]):
                            phase = _phase_ids(positions[:, indices[clause]], truth[:, clause])
                            arrays["phase_ids"][0, clause] = phase[frame]
                value = {prefix: {k: torch.from_numpy(v) for k, v in values.items()} for prefix, values in value.items()}
                if target:
                    import os
                    temp = target.with_suffix(f".{os.getpid()}.tmp")
                    torch.save({"labels": value, "raw_size": path.stat().st_size,
                                "raw_mtime_ns": path.stat().st_mtime_ns}, temp)
                    temp.replace(target)
            self.labels[key] = value
            if len(self.labels) > 2:
                self.labels.popitem(last=False)
        self.labels.move_to_end(key)
        result = {k: v.clone() for k, v in self.labels[key][language].items()}
        valid = result["clause_valid"].bool()
        # Offline expert clips have no audited memory transitions. Keep the
        # same explicit invalid temporal labels as the existing dataset.
        state = torch.zeros_like(valid, dtype=torch.long)
        state[valid & result["predicate_truth_valid"].bool() & (result["predicate_truth"] >= .5)] = 3
        result.update(phase_safe_memory_previous_state_ids=state,
                      phase_safe_memory_target_state_ids=state,
                      phase_safe_memory_state_valid=torch.zeros_like(valid),
                      phase_safe_memory_execution_target=torch.zeros(1, dtype=torch.long),
                      phase_safe_memory_execution_valid=torch.zeros(1, dtype=torch.bool),
                      phase_safe_memory_stage_id=torch.zeros(1, dtype=torch.long),
                      phase_safe_memory_stage_valid=torch.zeros(1, dtype=torch.bool))
        return result


def grounding_outputs(model, captured):
    """Ground semantic roles without recomputing irrelevant Video attention.

    The existing ERAF semantic heads read neutral pre-DiT patches and language;
    base goal queries are used only by the separately trained routing bridge.
    """
    import torch
    with torch.no_grad():
        pre = model.video_expert.pre_dit(**captured["video_inputs"])
    action = captured["action_inputs"]
    length = action["context"].shape[1] - int(action["state_only_context_mask"].sum())
    module = model.policy_guard_modules["entity_relation_affordance"]
    seeds = torch.zeros_like(model.policy_guard_modules["goal_query_seeds"].weight).unsqueeze(0)
    _, _, outputs, metrics = module(
        base_goal_queries=seeds,
        base_goal_embedding=seeds.new_zeros((1, module.projection_dim)),
        language_hidden=action["context"][:, :length], language_mask=action["context_mask"][:, :length],
        current_video_hidden=pre["tokens"].detach(), proprio=captured["proprio"], policy_state=None)
    return outputs, metrics
