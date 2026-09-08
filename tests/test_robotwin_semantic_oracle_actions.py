from copy import deepcopy
import pytest
import torch

from scripts.probe_robotwin_semantic_oracle_actions import TASKS, selected_rows, validate_oracle, oracle_route


def rows():
    return [dict(id=f'{task}-{split}-{seed}-{frame}', pair_id='synthetic', source_task=task,
                 task_config='demo_clean', replay_split=split, scene_seed=seed, frame_index=frame)
            for task in TASKS for split,seeds in [('train',[1]),('replay_holdout',[2,3])]
            for seed in seeds for frame in ([0] if task in TASKS[:2] and split=='replay_holdout' else [0,5,10])]


def test_selection_uses_declared_states_not_input_order():
    actual=selected_rows(rows())
    assert actual==selected_rows(list(reversed(rows())))
    assert len(actual)==20
    assert sum(r['replay_split']=='replay_holdout' for r in actual)==12
    assert all(r['frame_index'] in (0,5) for r in actual)


def test_selection_excludes_corrective_and_retention_rows():
    original=rows()
    noise=[dict(r,id='earlier-'+r['id'],**{kind:True}) for r in original
           for kind in ('fg_correction','cf_retention','native_retention')]
    assert selected_rows(original+noise)==selected_rows(original)


def test_scene_split_leakage_rejected():
    original=rows()
    with pytest.raises(ValueError,match='leakage'):
        selected_rows(original+[dict(original[0],id='leak',replay_split='replay_holdout')])


def test_missing_initial_rejected():
    with pytest.raises(ValueError,match='initial'):
        selected_rows([r for r in rows() if not (r['scene_seed']==1 and r['frame_index']==0)])


def labels():
    result={k:torch.ones(1,2,dtype=torch.bool) for k in ('clause_valid','subject_mask_valid',
        'reference_mask_valid','subject_position_valid','reference_position_valid','goal_anchor_valid',
        'phase_valid','predicate_truth_valid')}
    result.update({k:torch.zeros(1,2) for k in ('predicate_ids','subject_masks','reference_masks',
        'subject_positions','reference_positions','goal_anchors','predicate_truth','phase_ids')})
    return result


@pytest.mark.parametrize('key',['phase_valid','subject_position_valid','reference_position_valid',
                              'goal_anchor_valid','predicate_truth_valid'])
def test_invalid_active_label_rejected(key):
    value=labels(); value[key][0,0]=False
    with pytest.raises(ValueError,match='Missing active'):
        validate_oracle(value)
    value['clause_valid'][0,0]=False
    assert validate_oracle(value)


class Route:
    def route_oracle(self,*,oracle): return oracle['sentinel']


class Model:
    def __init__(self): self.policy_guard_modules={'entity_relation_affordance':Route()}
    def _encode_policy_guard_goal(self,*,policy_guard_eraf_oracle=None):
        return self.policy_guard_modules['entity_relation_affordance'].route_oracle(oracle=policy_guard_eraf_oracle)


def test_production_injection_count_and_restore():
    model=Model();original=model._encode_policy_guard_goal
    with torch.no_grad(), oracle_route(model,{'sentinel':17}) as counts:
        assert model._encode_policy_guard_goal()==17
    assert counts==dict(encode=1,route=1)
    assert model._encode_policy_guard_goal==original
    assert '_encode_policy_guard_goal' not in vars(model)


def test_route_must_actually_run():
    with torch.no_grad(), pytest.raises(ValueError,match='every production'):
        with oracle_route(Model(),{}): pass


def test_exception_restores_instance_override():
    model=Model(); model._encode_policy_guard_goal=lambda **kw:1
    original=model._encode_policy_guard_goal
    with pytest.raises(RuntimeError,match='inference failed'):
        with oracle_route(model,{}): raise RuntimeError('inference failed')
    assert model._encode_policy_guard_goal is original


def test_training_use_rejected_and_restored():
    model=Model()
    with pytest.raises(ValueError,match='inference-only'):
        with oracle_route(model,{}): model._encode_policy_guard_goal()
    assert '_encode_policy_guard_goal' not in vars(model)


def summaries():
    from scripts.run_robotwin_semantic_oracle_actions import ARMS
    records=[dict(id=str(i),task='cup',kind='initial',split='replay_holdout',scene_seed=i,
                  frame=0,instruction='front',state_sha256='s',rgb_sha256={'c':'i'},
                  reference_sha256='a',valid_sha256='v',oracle_label_sha256={'position':'p'},
                  action_mse_first24=dict(learned=2.,semantic_oracle=1.,eraf_off=3.,repeat=2.)) for i in range(20)]
    return {a:dict(complete=True,input_hashes_stable=True,records=deepcopy(records)) for a in ARMS}


def test_paired_diagnostic_requires_actual_input_identity():
    from scripts.run_robotwin_semantic_oracle_actions import compare
    value=summaries()
    assert compare(value)['actual_inputs_equal']
    value['eraf_fg']['records'][0]['oracle_label_sha256']['position']='changed'
    with pytest.raises(ValueError,match='differ'):compare(value)


def test_incomplete_diagnostic_cannot_be_summarized():
    from scripts.run_robotwin_semantic_oracle_actions import compare
    value=summaries();value['eraf_only']['complete']=False
    with pytest.raises(ValueError,match='complete'):compare(value)
