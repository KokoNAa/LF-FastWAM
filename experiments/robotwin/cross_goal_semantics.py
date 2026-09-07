"""Counter-goal semantic views; never reuse expert actions under a new goal."""
from __future__ import annotations


def paired_cross_goal_batch(batch, world_size):
    """Pair one common rank group with its opposite goal; keep FG slots intact.

    Three ranks must enter the same loss/validity branch together. Replacing
    an entire common group preserves the existing collective call schedule.
    """
    if world_size < 1 or len(batch) % world_size:
        raise ValueError('The semantic batch must contain complete rank groups.')
    common = []
    for start in range(0, len(batch), world_size):
        kinds = {bool(partial) for _, _, partial in batch[start:start + world_size]}
        if len(kinds) != 1:
            raise ValueError('Ranks would enter different semantic loss branches.')
        if kinds == {False}:
            common.append(start)
    if len(common) < 2:
        raise ValueError('Cross-goal pairing needs at least two common rank groups.')
    result = [(row, language, partial, language) for row, language, partial in batch]
    source, target = common[:2]
    for rank in range(world_size):
        row, language, partial = batch[source + rank]
        if language not in ('source', 'target') or any(row.get(k) for k in ('fg_correction', 'native_retention', 'cf_retention')):
            raise ValueError('Only ordinary expert observations may be paired across goals.')
        result[target + rank] = (row, language, False, 'target' if language == 'source' else 'source')
    return result


def semantic_row(raw, row, observation_language, instruction_language):
    if observation_language not in ('source', 'target') or instruction_language not in ('source', 'target'):
        raise ValueError('Declare physical trajectory and instruction branches separately.')
    if any(row.get(k) for k in ('fg_correction', 'native_retention', 'cf_retention')):
        raise ValueError('Cross-goal views require ordinary paired expert scene annotations.')
    path, frame = raw.locate(row, observation_language)
    return dict(row, raw_paths={'native': str(path), 'counterfactual': str(path)},
                source_frame_index=frame, target_frame_index=frame, frame_index=frame)


def cross_capture(captured, observation_language, instruction_language):
    """Use another goal's text while preserving this observation and state token."""
    from experiments.robotwin.decision_language_replay import replace_language
    if observation_language not in ('source', 'target') or instruction_language not in ('source', 'target'):
        raise ValueError('Unknown language branch.')
    if observation_language == instruction_language:
        return captured[observation_language]
    donor = captured[instruction_language]['action_inputs']
    mask = donor['state_only_context_mask']
    nstate = int(mask.sum())
    length = donor['context'].shape[1] - nstate
    if nstate != 1 or bool(mask[..., :length].any()) or not bool(mask[..., length:].all()):
        raise ValueError('Expected one trailing proprioception context token.')
    return replace_language(captured[observation_language], donor['context'][:, :length],
                            donor['context_mask'][:, :length])


def semantic_labels(raw, row, observation_language, instruction_language):
    labels = raw.grounding(semantic_row(raw, row, observation_language, instruction_language), instruction_language)
    if observation_language != instruction_language:
        # The other goal was not being executed in this trajectory. Its physical
        # predicates and geometry are observed; its execution history is not.
        labels = {k: v.clone() for k, v in labels.items()}
        for key in ('phase_valid', 'phase_safe_memory_state_valid', 'phase_safe_memory_execution_valid',
                    'phase_safe_memory_stage_valid'):
            if key in labels:
                labels[key].zero_()
    return labels


def terminal_queries(rows, raw):
    """Four semantic queries per held-out scene: two trajectories × two goals."""
    import h5py
    from experiments.robotwin.pgc_data import ROBOTWIN_TEN_TASK_EXTRA_SPECS, ROBOTWIN_REPLACEMENT_PAIR_SPECS
    required = {s.pair_id for s in (*ROBOTWIN_TEN_TASK_EXTRA_SPECS, *ROBOTWIN_REPLACEMENT_PAIR_SPECS)}
    scenes = {}
    for row in rows:
        if row['pair_id'] in required and row['replay_split'] == 'replay_holdout' and not any(
                row.get(k) for k in ('fg_correction', 'native_retention', 'cf_retention')):
            scenes.setdefault((row['pair_id'], row['task_config'], row['scene_seed']), row)
    if {key[0] for key in scenes} != required:
        raise ValueError('Every additional task needs held-out expert scenes.')
    result = []
    for _, row in sorted(scenes.items()):
        for observation_language in ('source', 'target'):
            path, _ = raw.locate(row, observation_language)
            with h5py.File(path, 'r') as handle:
                frame = len(handle['joint_action/vector']) - 1
            if frame < 1:
                raise ValueError('Expert trajectory has no executed terminal state.')
            physical = dict(row, **{observation_language + '_frame_index': frame})
            for language in ('source', 'target'):
                result.append((physical, observation_language, language))
    return result
