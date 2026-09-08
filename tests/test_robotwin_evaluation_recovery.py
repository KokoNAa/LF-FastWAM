import json
from pathlib import Path

import pytest

from scripts.recover_robotwin_cf_evaluation import (
    DEPLOYMENT, assert_source_idle, metadata, validate_complete_cell, worker_command,
)


@pytest.fixture
def cell(tmp_path):
    checkpoint = tmp_path / 'model.pt'; checkpoint.write_bytes(b'fixture identity only')
    catalog = tmp_path / 'catalog.jsonl'
    expected = dict(scene_seed=90200000, episode_index=0, source_instruction='RGB',
                    counterfactual_instruction='BGR', initial_physical_state_sha256='state-hash')
    catalog.write_text(json.dumps(expected)+'\n')
    directory = tmp_path / 'completed'; directory.mkdir()
    row = expected | dict(checkpoint=str(checkpoint), condition='counterfactual',
                          selected_goal='counterfactual', policy_instruction='BGR')
    (directory/'episodes.jsonl').write_text(json.dumps(row)+'\n')
    (directory/'initial_states.json').write_text(json.dumps([dict(scene_seed=90200000, sha256='state-hash')]))
    (directory/'summary.json').write_text(json.dumps(dict(total_episodes=1)))
    (directory/'complete.json').write_text(json.dumps(dict(complete=True, checkpoint=str(checkpoint),
        checkpoint_metadata=metadata(checkpoint), canonical_metadata=metadata(catalog), skip_file_hashes=True,
        deployment=DEPLOYMENT, eraf='on', policy_kind='repair', memory_mode='carry')))
    return directory, dict(checkpoint=checkpoint, catalog=catalog, episodes=1, eraf='on')


def test_complete_cell_can_be_retained(cell):
    directory, args = cell
    assert validate_complete_cell(directory, **args)


def test_partial_cell_is_not_mistaken_for_complete(cell):
    directory, args = cell
    (directory/'complete.json').unlink()
    (directory/'episodes.jsonl').write_text('{"interrupted":')
    assert not validate_complete_cell(directory, **args)
    assert (directory/'episodes.jsonl').read_text() == '{"interrupted":'


@pytest.mark.parametrize('field,value', [
    ('condition', 'correct'), ('selected_goal', 'source'), ('policy_instruction', 'RGB'),
    ('scene_seed', 90200001), ('checkpoint', '/different.pt'),
])
def test_mixed_episode_contract_rejected(cell, field, value):
    directory, args = cell
    path = directory/'episodes.jsonl'; row = json.loads(path.read_text()); row[field] = value
    path.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError): validate_complete_cell(directory, **args)


def test_complete_marker_cannot_hide_truncation(cell):
    directory, args = cell
    (directory/'episodes.jsonl').write_text('')
    with pytest.raises(ValueError): validate_complete_cell(directory, **args)


def test_changed_checkpoint_rejected(cell):
    directory, args = cell
    args['checkpoint'].write_bytes(b'changed weight')
    with pytest.raises(ValueError): validate_complete_cell(directory, **args)


def test_wrong_initial_scene_rejected(cell):
    directory, args = cell
    (directory/'initial_states.json').write_text(json.dumps([dict(scene_seed=90200000, sha256='wrong')]))
    with pytest.raises(ValueError): validate_complete_cell(directory, **args)


def test_live_source_owner_rejected(tmp_path):
    proc = tmp_path/'proc/987654'; proc.mkdir(parents=True)
    (proc/'stat').write_text('987654 (python) S 0')
    (proc/'cmdline').write_bytes(b'python\0/scripts/eval_robotwin_eraf_fg.py\0--output\0/runs/source/eval\0')
    with pytest.raises(ValueError): assert_source_idle(Path('/runs/source'), proc_root=tmp_path/'proc')
    (proc/'stat').write_text('987654 (python) Z 0')
    assert_source_idle(Path('/runs/source'), proc_root=tmp_path/'proc')


def test_recovery_command_uses_original_weight_and_eval_runtime():
    plan = dict(manifest='/data/manifest.json', groups={'original_five': dict(catalog='/catalog', episodes=6)},
                arms={'eraf_fg':dict(eraf='on')})
    cmd = worker_command(Path('/old-runtime'), plan, Path('/source'), Path('/new-output'),
                         'original_five', 'blocks_ranking_rgb', 'eraf_fg', 2)
    assert cmd[2] == '/old-runtime/scripts/eval_robotwin_eraf_fg.py'
    assert cmd[cmd.index('--checkpoint')+1] == '/source/eraf_fg/joint/step_000200.pt'
    assert cmd[cmd.index('--output')+1] == '/new-output/eval_original_five/eraf_fg/dev'
    assert '--resume' not in cmd and '--steps' not in cmd
