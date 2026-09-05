"""Seen-template language diversity for paired ranking and stacking positives."""
from __future__ import annotations

import json
from pathlib import Path
import re
import string


def seen_instruction_pairs(repo, pair_id, count=8):
    tasks = {
        'blocks_ranking_rgb_to_bgr': ('blocks_ranking_rgb',
            {'A': 'red block', 'B': 'green block', 'C': 'blue block'},
            {'A': 'blue block', 'B': 'green block', 'C': 'red block'}),
        'stack_blocks_two_green_on_red_to_red_on_green': ('stack_blocks_two',
            {'A': 'red block', 'B': 'green block'},
            {'A': 'green block', 'B': 'red block'}),
    }
    if pair_id not in tasks:
        return []
    task, source, target = tasks[pair_id]
    data = json.loads((Path(repo) / 'third_party/RoboTwin/description/task_instruction' / (task + '.json')).read_text())
    pairs = []
    for template in data['seen']:
        try:
            fields = {field for _, field, _, _ in string.Formatter().parse(template) if field}
        except ValueError:
            continue
        # Arm choices can depend on the scene; retain templates specifying goals
        # without arms. Literal colors cannot be reversed by slot substitution.
        if not fields or not fields <= source.keys() or re.search(r'\b(red|green|blue)\b', template.lower()):
            continue
        pair = {k: template.format(**slots) for k, slots in [('source', source), ('target', target)]}
        if pair not in pairs:
            pairs.append(pair)
    if len(pairs) > count:
        pairs = [pairs[i * len(pairs) // count] for i in range(count)]
    return pairs


def replace_language(captured, context, mask):
    """Replace frozen text tokens, keeping the captured state token and masks."""
    import torch
    output = dict(captured)
    length = context.shape[1]
    for key in ('video_inputs', 'action_inputs'):
        inputs = dict(captured[key])
        old, old_mask = inputs['context'], inputs['context_mask']
        if old.shape[1] not in (length, length + 1):
            raise ValueError('Expected fixed-length text followed by an optional state token.')
        inputs['context'] = torch.cat((context.to(old), old[:, length:]), dim=1)
        inputs['context_mask'] = torch.cat((mask.to(old_mask), old_mask[:, length:]), dim=1)
        output[key] = inputs
    return output


def build_seen_contexts(model, repo, pair_ids):
    contexts = {}
    for pair_id in sorted(set(pair_ids)):
        pairs = seen_instruction_pairs(repo, pair_id)
        if pairs:
            contexts[pair_id] = [{language: model.encode_prompt(text)
                                 for language, text in pair.items()} for pair in pairs]
    return contexts
