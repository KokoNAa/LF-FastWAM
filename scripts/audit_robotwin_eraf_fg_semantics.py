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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'checkpoint', 'source-bank', 'output'):
        ap.add_argument('--' + name, required=True)
    ap.add_argument('--label-cache')
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
    scale = torch.tensor((WORKSPACE_MAX - WORKSPACE_MIN) / 2, device=model.device)
    grouped = defaultdict(lambda: {'phase_confusion': np.zeros((3, 3), dtype=int),
        'truth_confusion': np.zeros((2, 2), dtype=int), 'positions': [], 'goals': []})
    with torch.no_grad():
        for i, row in enumerate(rows):
            payload = raw.attach_proprio(payloads[row['id']], row, policy)
            for language in ('source', 'target'):
                labels = move_cache(raw.grounding(row, language), model.device)
                pred, _ = grounding_outputs(model, payload['captured'][language])
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
            if (i + 1) % 10 == 0:
                print(f'[semantic-audit] states={i+1}/{len(rows)}', flush=True)
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
        'scope': 'Offline original expert holdout; no policy history or closed-loop action evaluation.',
        'phase_ids': {'0': 'approach', '1': 'transport', '2': 'release_or_complete'}, 'cells': cells}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
