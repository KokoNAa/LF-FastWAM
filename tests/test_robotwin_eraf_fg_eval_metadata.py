import json
import sys
import pytest
from scripts import eval_robotwin_eraf_fg as evaluation


def test_summarize_without_file_scans_still_rejects_changed_checkpoint(tmp_path, monkeypatch):
    from experiments.robotwin import eraf_fg_bridge
    monkeypatch.setattr(eraf_fg_bridge, 'file_sha256', lambda _: pytest.fail('Unexpected full-file scan'))
    checkpoint = tmp_path/'model.pt'; checkpoint.write_bytes(b'checkpoint')
    catalog = tmp_path/'catalog'; output = tmp_path/'results'
    canonical = catalog/'place_a2b_left/demo_clean/correct/episodes.jsonl'
    canonical.parent.mkdir(parents=True)
    episode = {'scene_seed': 123, 'episode_index': 0, 'source_instruction': 'left',
               'counterfactual_instruction': 'right'}
    canonical.write_text(json.dumps(episode)+'\n')
    for condition in ('correct', 'counterfactual'):
        path = output/'place_a2b_left/demo_clean'/condition; path.mkdir(parents=True)
        (path/'episodes.jsonl').write_text(json.dumps(episode)+'\n')
        (path/'summary.json').write_text(json.dumps({'total_episodes': 1}))
        (path/'initial_states.json').write_text(json.dumps([{'scene_seed': 123, 'sha256': 'same'}]))
        (path/'complete.json').write_text(json.dumps({'complete': True, 'checkpoint': str(checkpoint),
            'skip_file_hashes': True, 'checkpoint_metadata': evaluation.file_metadata(checkpoint),
            'canonical_metadata': evaluation.file_metadata(canonical), 'eraf': 'off',
            'policy_kind': 'repair', 'memory_mode': 'carry', 'deployment': {'action_horizon': 32}}))
    monkeypatch.setattr(sys, 'argv', ['eval', 'summarize', '--output', str(output),
        '--checkpoint', str(checkpoint), '--catalog-root', str(catalog), '--tasks', 'place_a2b_left',
        '--episodes', '1', '--skip-file-hashes'])
    evaluation.main()
    assert json.loads((output/'summary.json').read_text())['episodes'] == 2
    checkpoint.write_bytes(b'changed checkpoint')
    with pytest.raises(ValueError, match='checkpoint changed'):
        evaluation.main()
