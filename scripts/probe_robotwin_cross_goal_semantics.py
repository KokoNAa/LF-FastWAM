#!/usr/bin/env python3
"""Audit both goal instructions on the same held-out expert terminal states."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src')]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ('manifest', 'checkpoint', 'source-bank', 'output'):
        ap.add_argument('--' + key, type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import h5py
    import numpy as np
    import torch
    from experiments.robotwin.cross_goal_semantics import terminal_queries, semantic_labels
    from experiments.robotwin.eraf_fg_bridge import load_policy, capture_frozen_inputs, file_sha256
    from experiments.robotwin.eraf_fg_data import RawReplay, grounding_outputs
    from experiments.robotwin.eraf_fg_contract import CAMERAS
    from experiments.robotwin.image_io import decode_legacy_robotwin_rgb
    from experiments.robotwin.pgc_data import array_sha256
    from experiments.robotwin.same_state_repair import move_cache
    from scripts.build_pgc_robotwin_entity_relations import WORKSPACE_MIN, WORKSPACE_MAX
    torch.set_num_threads(2)
    manifest = json.loads(args.manifest.read_text())
    raw = RawReplay(args.source_bank)
    queries = terminal_queries(manifest['states'], raw)
    policy = load_policy(args.checkpoint, manifest)
    policy.num_inference_steps = 1  # Capture semantic inputs only; no action-quality claim.
    model = policy.model
    scale = torch.tensor((WORKSPACE_MAX - WORKSPACE_MIN) / 2, device=model.device)
    records = []
    with torch.no_grad():
        for row, observation_language, language in queries:
            path, frame = raw.locate(row, observation_language)
            with h5py.File(path, 'r') as handle:
                observation = dict(joint_action={'vector': handle['joint_action/vector'][frame].astype(np.float32)},
                    observation={c: {'rgb': decode_legacy_robotwin_rgb(handle[f'observation/{c}/rgb'][frame])} for c in CAMERAS})
            instruction = row['source_instruction' if language == 'source' else 'counterfactual_instruction']
            policy.reset()
            captured = move_cache(capture_frozen_inputs(policy, observation, instruction), model.device)
            assert captured['policy_guard_state'] is None
            labels = move_cache(semantic_labels(raw, row, observation_language, language), model.device)
            pred, _ = grounding_outputs(model, captured)
            valid = labels['clause_valid'].bool() & labels['predicate_truth_valid'].bool()
            delta = (pred['goal_anchor'].float() - labels['goal_anchors'].float()) * scale
            clauses = [dict(clause=int(c), truth_label=bool(labels['predicate_truth'][b, c] >= .5),
                truth_probability=float(pred['predicate_truth_logits'][b, c].sigmoid()),
                active_probability=float(pred['active_logits'][b, c].sigmoid()),
                phase_prediction=int(pred['phase_logits'][b, c].argmax()),
                goal_error_cm=float(delta[b, c].norm() * 100)) for b, c in valid.nonzero().tolist()]
            records.append(dict(pair_id=row['pair_id'], task_config=row['task_config'], scene_seed=row['scene_seed'],
                observation_language=observation_language, instruction_language=language, raw_path=str(path), frame=frame,
                instruction=instruction, state_sha256=array_sha256(observation['joint_action']['vector']),
                rgb_sha256={c: array_sha256(observation['observation'][c]['rgb']) for c in CAMERAS},
                crossed=observation_language != language, clauses=clauses))
            if len(records) % 10 == 0:
                print(f'[cross-goal] {len(records)}/{len(queries)}', flush=True)
    grouped = defaultdict(lambda: dict(queries=0, clauses=0, negatives=0, false_positives=0, hits=0, goal_error_cm_sum=0.))
    for row in records:
        group = grouped[(row['pair_id'], row['crossed'])]
        group['queries'] += 1
        for c in row['clauses']:
            truth, prediction = c['truth_label'], c['truth_probability'] >= .5
            group['clauses'] += 1
            group['negatives'] += not truth
            group['false_positives'] += not truth and prediction
            group['hits'] += truth == prediction
            group['goal_error_cm_sum'] += c['goal_error_cm']
    report = dict(complete=True, format='robotwin_cross_goal_semantic_probe_v1', checkpoint=str(args.checkpoint),
        checkpoint_sha256=file_sha256(args.checkpoint), manifest=str(args.manifest), manifest_sha256=file_sha256(args.manifest),
        queries=len(records), scene_count=len({(r['pair_id'], r['task_config'], r['scene_seed']) for r in records}),
        records=records, cells=[dict(pair_id=k[0], crossed=k[1], **v) for k, v in sorted(grouped.items())],
        action_quality_evaluated=False, labels_used_as_model_inputs=False, policy_history='none',
        scope='Held-out expert endpoints under both goals. Cross-goal phase/history labels are invalidated; physical truth and geometry remain simulator annotations. This is a semantic audit, not a CF policy score.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
