from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.robotwin.expanded_fg import SPECS
from experiments.robotwin.expanded_fg_preparation import audit_extension,merge_mask_shards
from test_robotwin_expanded_fg import record


def bank_fixture():
    parent=dict(complete=True,states=[dict(id='old',pair_id='place_a2b_left_to_right',
        task_config='demo_clean',scene_seed=123,replay_split='train')],stats_path='fixed-stats')
    records=[];added=[]
    for t,task in enumerate(('place_empty_cup','move_pillbottle_pad')):
        spec=next(s for s in SPECS if s.source_task==task)
        for split,n,base in [('train',24,86000100),('replay_holdout',6,87000100)]:
            for i in range(n):
                seed=base+t*4000+i
                r=record(spec)|dict(scene_seed=seed,replay_split=split,frame_path=f'raw/{seed}.npz',
                                    frame_sha256='f'*64,controls_sha256='e'*64)
                records.append(r)
                for frame in range(0,83,8):
                    added.append(r|dict(id=f'{task}_{seed}_{frame}',payload='missing',fg_correction=True,
                        frame_index=frame,reference_valid_actions=min(32,83-frame),policy_memory='none',
                        frozen_input_protocol='robotwin_eraf_fg_pre_dit_v1'))
    target=deepcopy(parent);target['states'].extend(added)
    return parent,target,dict(complete=True,records=records)


def test_extension_preserves_old_rows_and_all_real_corrective_windows():
    report=audit_extension(*bank_fixture(),check_files=False)
    assert report['parent_rows']==1 and report['added_scenes']==60 and report['added_rows']==660
    assert report['all_goal_tail_windows_present']


@pytest.mark.parametrize('change', ['old_row','stats','tail','mask','source','memory','duplicate','scene','split','file'])
def test_extension_rejects_changed_parent_leakage_and_incomplete_full_goal_supervision(change):
    parent,target,collection=bank_fixture()
    if change=='old_row':target['states'][0]['id']='changed'
    if change=='stats':target['stats_path']='changed'
    if change=='tail':target['states'].pop()
    if change=='mask':target['states'][-1]['reference_valid_actions']=32
    if change=='source':target['states'][-1]['counterfactual_instruction']='wrong'
    if change=='memory':target['states'][-1]['policy_memory']='gold'
    if change=='duplicate':target['states'].append(deepcopy(target['states'][-1]))
    if change=='scene':collection['records'].pop()
    if change=='split':collection['records'][0]['replay_split']='replay_holdout'
    with pytest.raises(ValueError):audit_extension(parent,target,collection,check_files=change=='file')


def test_preparation_wait_checks_actual_source_command_and_kernel_start(tmp_path):
    from scripts.run_robotwin_expanded_fg_preparation import source_process
    proc=tmp_path/'123';proc.mkdir()
    fields=['S']+['0']*18+['6789']
    (proc/'stat').write_text('123 (python) '+' '.join(fields))
    (proc/'cmdline').write_bytes(b'python\0/repo/run_robotwin_expanded_fg_pilots.py\0--output\0/runs/source\0')
    assert source_process(123,Path('/runs/source'),proc_root=tmp_path)=='6789'
    with pytest.raises(ValueError):source_process(123,Path('/runs/other'),proc_root=tmp_path)
    (proc/'cmdline').write_bytes(b'python\0unrelated.py\0/runs/source\0')
    with pytest.raises(ValueError):source_process(123,Path('/runs/source'),proc_root=tmp_path)
    fields[0]='Z';(proc/'stat').write_text('123 (python) '+' '.join(fields))
    assert source_process(123,Path('/runs/source'),proc_root=tmp_path) is None
    assert source_process(999,Path('/runs/source'),proc_root=tmp_path) is None


def test_mask_aggregate_rechecks_complete_parent_and_added_scene_coverage(tmp_path):
    from experiments.robotwin.eraf_fg_bridge import file_sha256
    from experiments.robotwin.fg_mask_replay import SCHEMA
    root=tmp_path/'masks';root.mkdir()
    manifest=tmp_path/'manifest.json'
    rows=[]
    for i in range(2):
        r=dict(pair_id='place_empty_cup_on_to_front',task_config='demo_clean',scene_seed=86000100+i,
            replay_split='train',fg_correction=True,frame_path=f'/raw/{i}.npz',frame_sha256='a'*64,controls_sha256='b'*64)
        rows.append(r)
    manifest.write_text(json.dumps(dict(states=rows)))
    for i,r in enumerate(rows):
        folder=root/f'shard{i}';folder.mkdir()
        labels=folder/'labels.npz';labels.write_bytes(b'test identity')
        scene=r|dict(complete=True,all_rgb_frames_equal=True,phase_labels_valid=False,temporal_labels_valid=False,
                     replay_state_max_abs=0.,frames=83,labels=str(labels),labels_sha256=file_sha256(labels))
        (folder/'plan.json').write_text(json.dumps(dict(manifest_sha256=file_sha256(manifest),shard_index=i,num_shards=2)))
        (folder/'complete.json').write_text(json.dumps(dict(complete=True,schema=SCHEMA,scenes=[scene])))
    result=merge_mask_shards(root,manifest,2)
    assert result['scenes']==2 and result['frames']==166
    with pytest.raises(FileExistsError):merge_mask_shards(root,manifest,2)


def test_missing_mask_shard_cannot_publish_a_complete_aggregate(tmp_path):
    root=tmp_path/'masks';root.mkdir()
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    with pytest.raises(FileNotFoundError):merge_mask_shards(root,manifest,2)
    assert not (root/'complete.json').exists()
