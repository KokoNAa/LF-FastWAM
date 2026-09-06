#!/usr/bin/env python3
"""Measure frozen semantic geometry, phase, and truth on expert holdout states."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src')]


def trajectory_queries(rows, raw):
    """Sample existing held-out expert scenes at each observed semantic phase.

    Simulator labels select offline audit frames only; they are never policy
    inputs. This adds no training scenes and does not inspect policy rollouts.
    """
    import h5py
    import numpy as np
    from scripts.build_pgc_robotwin_entity_relations import _phase_ids
    unique = {}
    for row in rows:
        for language in ('source', 'target'):
            path, _ = raw.locate(row, language)
            unique.setdefault((str(path), language), row)
    queries = []
    for (path, language), row in sorted(unique.items()):
        with h5py.File(path, 'r') as h:
            positions = h['pgc_entity_state/entity_positions'][:]
            truth = h[f'pgc_entity_state/{language}_predicate_truth'][:]
            valid = h[f'pgc_entity_state/{language}_clause_valid'][:].astype(bool)
            indices = h[f'pgc_entity_state/{language}_subject_indices'][0]
            frames = {0, len(positions) - 1}
            for clause in np.flatnonzero(valid.any(0)):
                phase = _phase_ids(positions[:, indices[clause]], truth[:, clause])
                for value in (0, 1, 2):
                    matching = np.flatnonzero(valid[:, clause] & (phase == value))
                    if len(matching):
                        frames.add(int(matching[len(matching) // 2]))
            for frame in sorted(frames):
                queries.append((dict(row, **{language + '_frame_index': frame}), language))
    return queries


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'checkpoint', 'source-bank', 'output'):
        ap.add_argument('--' + name, required=True)
    ap.add_argument('--label-cache')
    ap.add_argument('--trajectory-holdout', action='store_true',
                    help='Audit all observed phases of the same held-out expert scenes, without training.')
    args = ap.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    import numpy as np
    import torch
    from experiments.robotwin.eraf_fg_bridge import load_policy, file_sha256
    from experiments.robotwin.eraf_fg_data import RawReplay, grounding_outputs
    from experiments.robotwin.compact_replay import ReplayPayloads
    from experiments.robotwin.same_state_repair import move_cache
    from experiments.robotwin.eraf_fg_bridge import capture_frozen_inputs
    from scripts.build_pgc_robotwin_entity_relations import WORKSPACE_MIN, WORKSPACE_MAX
    torch.set_num_threads(2)
    manifest = json.loads(Path(args.manifest).read_text())
    rows = [r for r in manifest['states'] if r['replay_split'] == 'replay_holdout'
            and not any(r.get(k) for k in ('fg_correction', 'native_retention', 'cf_retention'))]
    if not rows:
        raise ValueError('No original expert holdout states.')
    policy = load_policy(args.checkpoint, manifest)
    model = policy.model
    payloads = ReplayPayloads(rows, model.device)
    raw = RawReplay(args.source_bank, label_cache=args.label_cache)
    queries = trajectory_queries(rows, raw) if args.trajectory_holdout else [
        (row, language) for row in rows for language in ('source', 'target')]
    query_records = []
    policy.num_inference_steps = 1  # Only frozen semantic inputs are captured in trajectory mode.
    scale = torch.tensor((WORKSPACE_MAX - WORKSPACE_MIN) / 2, device=model.device)
    grouped = defaultdict(lambda: {'phase_confusion': np.zeros((3, 3), dtype=int),
        'truth_confusion': np.zeros((2, 2), dtype=int), 'positions': [], 'goals': []})
    with torch.no_grad():
        for i, (row, language) in enumerate(queries):
            path, frame = raw.locate(row, language)
            if args.trajectory_holdout:
                import h5py
                from experiments.robotwin.image_io import decode_legacy_robotwin_rgb
                from experiments.robotwin.eraf_fg_contract import CAMERAS
                from experiments.robotwin.pgc_data import array_sha256
                with h5py.File(path, 'r') as h:
                    observation = {'joint_action': {'vector': h['joint_action/vector'][frame].astype(np.float32)},
                        'observation': {c: {'rgb': decode_legacy_robotwin_rgb(h[f'observation/{c}/rgb'][frame])}
                                        for c in CAMERAS}}
                policy.reset()
                instruction = row['source_instruction' if language == 'source' else 'counterfactual_instruction']
                captured = capture_frozen_inputs(policy, observation, instruction)
                captured = move_cache(captured, model.device)
                query_records.append({'pair_id': row['pair_id'], 'scene_seed': row['scene_seed'],
                    'language': language, 'path': str(path), 'frame': frame, 'instruction': instruction,
                    'state_sha256': array_sha256(observation['joint_action']['vector']),
                    'rgb_sha256': {c: array_sha256(observation['observation'][c]['rgb']) for c in CAMERAS}})
            else:
                payload = raw.attach_proprio(payloads[row['id']], row, policy)
                captured = payload['captured'][language]
            labels = move_cache(raw.grounding(row, language), model.device)
            pred, _ = grounding_outputs(model, captured)
            group = grouped[row['pair_id'], language]
            valid = labels['clause_valid'].bool()
            phase = pred['phase_logits'].argmax(-1)
            keep = valid & labels['phase_valid'].bool()
            for truth, guess in zip(labels['phase_ids'][keep].tolist(), phase[keep].tolist()):
                group['phase_confusion'][truth, guess] += 1
            truth_pred = pred['predicate_truth_logits'].sigmoid() >= .5
            keep = valid & labels['predicate_truth_valid'].bool()
            for truth, guess in zip((labels['predicate_truth'][keep] >= .5).long().tolist(), truth_pred[keep].long().tolist()):
                group['truth_confusion'][truth, guess] += 1
            for role in ('subject', 'reference'):
                keep = valid & labels[role + '_position_valid'].bool()
                delta = (pred[role + '_position'].float() - labels[role + '_positions'].float()) * scale
                group['positions'] += delta.norm(dim=-1)[keep].tolist()
            keep = valid & labels['goal_anchor_valid'].bool()
            delta = (pred['goal_anchor'].float() - labels['goal_anchors'].float()) * scale
            group['goals'] += delta.norm(dim=-1)[keep].tolist()
            if (i + 1) % 20 == 0:
                print(f'[semantic-audit] queries={i+1}/{len(queries)}', flush=True)
    cells = []
    for (pair, language), group in sorted(grouped.items()):
        cell = {'pair_id': pair, 'language': language}
        for key in ('phase', 'truth'):
            confusion = group[key + '_confusion']
            cell[key + '_confusion_true_rows_predicted_columns'] = confusion.tolist()
            cell[key + '_accuracy'] = float(confusion.trace() / confusion.sum()) if confusion.sum() else None
        for key in ('positions', 'goals'):
            values = np.asarray(group[key]) * 100
            cell[key + '_euclidean_error_cm'] = {'count': len(values), 'mean': float(values.mean()),
                'median': float(np.median(values)), 'p90': float(np.percentile(values, 90))} if len(values) else None
        cells.append(cell)
    report = {'complete': True, 'checkpoint': args.checkpoint,
        'checkpoint_sha256': file_sha256(args.checkpoint), 'holdout_states': len(rows),
        'scene_count': len({(r['pair_id'], r['task_config'], r['scene_seed']) for r in rows}),
        'scope': 'Offline expert holdout; no policy history or closed-loop action evaluation.',
        'trajectory_holdout': args.trajectory_holdout, 'language_queries': len(queries),
        'trajectory_queries': query_records,
        'phase_ids': {'0': 'approach', '1': 'transport', '2': 'release_or_complete'}, 'cells': cells}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
