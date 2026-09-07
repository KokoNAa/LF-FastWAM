import json
from pathlib import Path
import pytest

from scripts.summarize_robotwin_cf_priority_campaign import summarize, OLD_TASKS, NEW_TASKS


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root):
    selection = {n: {'name': n + '_200', 'step': 200} for n in ('ordinary_cf', 'fg_only', 'eraf_fg')}
    put(root / 'protocol.json', {'arms': {k: {} for k in selection}, 'code_commit': 'training_commit',
                                'joint_steps': 400, 'interface_steps': 100})
    put(root / 'selection.json', selection); put(root / 'driver.json', {'complete': True})
    for matrix in ('eval_original_five', 'eval_additional_five', 'eval_eraf_bypass'):
        put(root / matrix / 'driver.json', dict(complete=True, matched_initial_states=True))
        tasks = NEW_TASKS if matrix == 'eval_additional_five' else OLD_TASKS
        names = ['selected_eraf_fg_bypass'] if matrix == 'eval_eraf_bypass' else ['all5_off200'] + [s['name'] for s in selection.values()]
        if matrix == 'eval_original_five':
            names += [arm + '_400' for arm in selection]
            put(root / 'eval_original_five_plan.json',
                {'models': {n: {'checkpoint': '/weights/' + n + '.pt'} for n in names}})
        for task in tasks:
            for name in names:
                n = 3 if task in NEW_TASKS else 6
                hits = 0 if n == 3 else (4 if name == 'eraf_fg_200' else 3)
                path = root / matrix / name / 'dev' / task / 'demo_clean/counterfactual'
                checkpoint = '/weights/' + ('eraf_fg_200' if name == 'selected_eraf_fg_bypass' else name) + '.pt'
                put(path / 'complete.json', dict(complete=True, checkpoint=checkpoint))
                put(path / 'initial_states.json', list(range(n)))
                records = [dict(scene_seed=i, episode_index=i, source_instruction='source',
                    counterfactual_instruction='target', policy_instruction='target', selected_goal='counterfactual',
                    initial_source_goal_success=False, initial_counterfactual_goal_success=False,
                    step_limit=1200, selected_goal_success=i < hits) for i in range(n)]
                (path / 'episodes.jsonl').write_text('\n'.join(json.dumps(r) for r in records) + '\n')


def test_ten_task_macro_gains_and_real_module_bypass_are_not_microaverages(tmp_path):
    fixture(tmp_path); report = summarize(tmp_path)
    assert report['complete'] and report['eraf_fg_strictly_best']
    assert report['aggregates']['eraf_fg_200']['macro_cf'] == pytest.approx(1/3)
    assert report['aggregates']['eraf_fg_200']['gained_scenes'] == 5
    assert report['aggregates']['eraf_fg_200']['lost_scenes'] == 0
    assert len(report['all_original_five_candidates']) == 7
    assert all(r['lost_by_bypass'] == [3] for r in report['same_checkpoint_eraf_bypass'].values())
    assert not report['correct_evaluated'] and not report['independent_test']
    p = tmp_path / 'eval_additional_five/eraf_fg_200/dev/place_empty_cup/demo_clean/counterfactual/complete.json'
    put(p, dict(complete=True, checkpoint='/different.pt'))
    with pytest.raises(ValueError, match='different checkpoints'):
        summarize(tmp_path)


def test_incomplete_or_unmatched_scene_cannot_produce_a_winning_report(tmp_path):
    fixture(tmp_path)
    p = tmp_path / 'eval_original_five/eraf_fg_200/dev/blocks_ranking_rgb/demo_clean/counterfactual/initial_states.json'
    put(p, [99])
    with pytest.raises(ValueError, match='exact CF scenes'):
        summarize(tmp_path)
    put(tmp_path / 'driver.json', {'complete': False})
    with pytest.raises(ValueError, match='incomplete'):
        summarize(tmp_path)
