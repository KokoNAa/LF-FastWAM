import json
import sys

import numpy as np
import pytest

from experiments.robotwin.eraf_fg_data import file_metadata
from scripts import curate_robotwin_retention_pool as pool


@pytest.mark.parametrize('case', ['valid_partial', 'initial_goal', 'duplicate', 'evaluation_seed'])
def test_pool_preserves_partial_status_and_rejects_invalid_successes(tmp_path, monkeypatch, case):
    teacher = tmp_path / 'teacher.pt'
    teacher.write_bytes(b'teacher fixture')
    parent = tmp_path / 'parent.json'
    parent.write_text(json.dumps({'states': []}))
    rows = []
    for index in range(10):
        seed = (91200000 if case == 'evaluation_seed' else 80810000) + index
        capture = tmp_path / f'{seed}.npz'
        state = np.zeros(60)
        state[0], state[7], state[14] = (0., .1, .2)
        if case == 'initial_goal' and index == 0:
            state[0], state[14] = .2, 0.
        np.savez(capture, initial_physical_state=state, executed_actions=np.zeros((48, 14)))
        rows.append(dict(id=f'cf_{seed}', source_task='blocks_ranking_rgb', task_config='demo_clean',
                         scene_seed=seed, replay_split='train', cf_retention=True,
                         retention_condition='counterfactual', full_cf_episode_success=True,
                         retention_format='robotwin_policy_retention_v2',
                         source_instruction='RGB', counterfactual_instruction='BGR', frame_index=0,
                         capture_path=str(capture), capture_metadata=file_metadata(capture),
                         teacher_checkpoint=str(teacher), teacher_checkpoint_metadata=file_metadata(teacher)))
    source = tmp_path / 'partial.json'
    source.write_text(json.dumps(dict(source_task='blocks_ranking_rgb', condition='counterfactual',
                                     complete=False, requested_scenes=12, successful_scenes=10,
                                     attempted_scenes=40, states=rows)))
    output = tmp_path / 'pool.json'
    collections = [str(source)] * (2 if case == 'duplicate' else 1)
    monkeypatch.setattr(sys, 'argv', ['pool', '--collections', *collections,
                                    '--parent-manifest', str(parent), '--task', 'blocks_ranking_rgb',
                                    '--scenes', '10', '--output', str(output)])
    if case == 'valid_partial':
        pool.main()
        report = json.loads(output.read_text())
        assert report['complete'] and report['successful_scenes'] == 10
        assert report['original_collectors_all_complete'] is False
        assert report['source_collections'][0]['complete'] is False
        assert json.loads(source.read_text())['complete'] is False
        assert len(report['initial_goal_audit']) == 10
    else:
        with pytest.raises(ValueError):
            pool.main()
        assert not output.exists()
