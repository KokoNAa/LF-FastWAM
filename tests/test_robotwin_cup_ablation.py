import json
from pathlib import Path
import pytest
import torch
from scripts.run_robotwin_cup_ablation import training_audit


def artifacts(tmp_path):
    parent = tmp_path/'parent.pt'
    before = {'mot_trainable': {'block.action.lora': torch.zeros(2), 'block.video.lora': torch.zeros(2)},
              'policy_guard': {'guard': torch.zeros(2)}}
    torch.save(before, parent)
    after = {'mot_trainable': {'block.action.lora': torch.ones(2), 'block.video.lora': torch.zeros(2)},
             'policy_guard': {'guard': torch.zeros(2)}, 'optimizer_steps': 2, 'parent_checkpoint': str(parent)}
    joint = tmp_path/'joint'; joint.mkdir()
    torch.save(after, joint/'step_000002.pt')
    for rank in range(3):
        rows = [{'step': step, 'grad_norm': .1,
                 'examples': [{'id': f'cup_ordinary_{rank}_{step}_{i}', 'ordinary_cf_control': True}
                              for i in range(4)]} for step in (1, 2)]
        (joint/f'rank{rank}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return parent, after


def test_real_update_audit_rejects_frozen_parameter_changes(tmp_path):
    parent, after = artifacts(tmp_path)
    result = training_audit(tmp_path, parent, 2, 3)
    assert result['examples'] == result['cup_examples'] == 24
    assert result['changed_action_tensors'] == 1
    after['mot_trainable']['block.video.lora'][0] = 1
    torch.save(after, tmp_path/'joint/step_000002.pt')
    with pytest.raises(ValueError, match='frozen non-action'):
        training_audit(tmp_path, parent, 2, 3)


def test_training_audit_rejects_incomplete_rank(tmp_path):
    parent, _ = artifacts(tmp_path)
    path = tmp_path/'joint/rank2.jsonl'
    path.write_text(path.read_text().splitlines()[0]+'\n')
    with pytest.raises(ValueError, match='Incomplete optimizer-step'):
        training_audit(tmp_path, parent, 2, 3)
