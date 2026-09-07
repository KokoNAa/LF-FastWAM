import pytest
import torch

from experiments.robotwin.cross_goal_semantics import cross_capture, semantic_labels, semantic_row, paired_cross_goal_batch


def capture(text, state, image):
    text = torch.full((1, 3, 2), float(text))
    state = torch.full((1, 1, 2), float(state))
    return dict(proprio=state, policy_guard_state=None,
        video_inputs=dict(x=torch.tensor([image]), context=text.clone(), context_mask=torch.ones(1, 3)),
        action_inputs=dict(context=torch.cat([text, state], 1), context_mask=torch.ones(1, 4),
                           state_only_context_mask=torch.tensor([[False, False, False, True]])))


def test_crossed_text_preserves_physical_observation_and_its_state_token():
    captured = dict(source=capture(1, 10, 100), target=capture(2, 20, 200))
    out = cross_capture(captured, 'source', 'target')
    assert torch.equal(out['video_inputs']['x'], torch.tensor([100]))
    assert torch.equal(out['proprio'], captured['source']['proprio'])
    assert torch.equal(out['action_inputs']['context'][:, -1:], captured['source']['action_inputs']['context'][:, -1:])
    assert torch.equal(out['action_inputs']['context'][:, :3], captured['target']['action_inputs']['context'][:, :3])
    assert torch.equal(out['video_inputs']['context'], captured['target']['video_inputs']['context'])
    assert torch.equal(captured['source']['action_inputs']['context'][:, :3], torch.ones(1, 3, 2))
    captured['target']['action_inputs']['state_only_context_mask'] = torch.tensor([[True, False, False, False]])
    with pytest.raises(ValueError):
        cross_capture(captured, 'source', 'target')


class Raw:
    def locate(self, row, language):
        return row['raw_paths']['native' if language == 'source' else 'counterfactual'], row[language + '_frame_index']

    def grounding(self, row, language):
        assert row['raw_paths'] == {'native': 'native.h5', 'counterfactual': 'native.h5'}
        assert row['target_frame_index'] == row['source_frame_index'] == 17
        assert language == 'target'
        return self.labels


def test_crossed_labels_use_actual_scene_other_predicate_and_no_invented_history():
    raw = Raw()
    raw.labels = dict(predicate_truth=torch.tensor([[0.]]), goal_anchors=torch.tensor([[[1., 2., 3.]]]),
                      phase_valid=torch.tensor([[True]]), phase_safe_memory_state_valid=torch.tensor([[False]]))
    row = dict(raw_paths={'native': 'native.h5', 'counterfactual': 'target.h5'}, source_frame_index=17, target_frame_index=29)
    labels = semantic_labels(raw, row, 'source', 'target')
    assert not labels['phase_valid'].any()
    assert not labels['phase_safe_memory_state_valid'].any()
    assert torch.equal(labels['predicate_truth'], raw.labels['predicate_truth'])
    assert torch.equal(labels['goal_anchors'], raw.labels['goal_anchors'])
    assert raw.labels['phase_valid'].all()  # No mutation of the cached original labels.
    assert row['raw_paths']['counterfactual'] == 'target.h5'
    with pytest.raises(ValueError):
        semantic_row(raw, dict(row, fg_correction=True), 'source', 'target')


def test_paired_groups_preserve_collective_schedule_and_full_goal_samples():
    batch = [(dict(id=i), 'source' if i % 2 else 'target', 3 <= i < 6) for i in range(12)]
    out = paired_cross_goal_batch(batch, 3)
    for rank in range(3):
        assert out[6 + rank][0] is out[rank][0]
        assert out[6 + rank][1] == out[rank][1]
        assert out[6 + rank][3] != out[rank][3]
    assert out[3:6] == [(r, l, p, l) for r, l, p in batch[3:6]]
    assert out[9:] == [(r, l, p, l) for r, l, p in batch[9:]]
    assert [sum(x[1] != x[3] for x in out[r::3]) for r in range(3)] == [1, 1, 1]
    with pytest.raises(ValueError):
        paired_cross_goal_batch([(dict(id=i), 'source', i == 0) for i in range(12)], 3)
    with pytest.raises(ValueError):
        paired_cross_goal_batch(batch[:6], 3)
