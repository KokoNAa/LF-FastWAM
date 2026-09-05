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


def bound_spatial_instruction_pairs(repo, pair_id, slots, count=4):
    """Name the actual A/B objects using training-only object descriptions."""
    source_direction, target_direction = {
        'place_a2b_left_to_right': ('left', 'right'),
        'place_a2b_right_to_left': ('right', 'left'),
    }[pair_id]
    description = Path(repo) / 'third_party/RoboTwin/description'
    data = json.loads((description / 'task_instruction' / f'place_a2b_{source_direction}.json').read_text())
    templates = []
    for template in data['seen']:
        fields = set(re.findall(r'{([^}]+)}', template))
        if fields == {'A', 'B'} and re.search(r'\b' + source_direction + r'\b', template):
            if template not in templates:
                templates.append(template)
    names = {}
    for key in ('A', 'B'):
        asset = slots['{' + key + '}']
        path = description / 'objects_description' / (asset + '.json')
        names[key] = json.loads(path.read_text())['seen']
    pairs = []
    for i, template in enumerate(templates[:count]):
        values = {key: 'the ' + names[key][i % len(names[key])] for key in names}
        # Swap directions BEFORE rendering names: object descriptions may
        # themselves contain a directional word that must remain unchanged.
        reverse = re.sub(r'\b' + source_direction + r'\b', target_direction, template)
        pairs.append({'source': template.format(**values), 'target': reverse.format(**values)})
    return pairs


def build_seen_contexts(model, repo, rows):
    specs = {}
    for row in rows:
        if row['replay_split'] != 'train':
            continue
        key = row.get('language_replay_key', row['pair_id'])
        if key not in specs:
            specs[key] = row.get('seen_instruction_pairs') or seen_instruction_pairs(repo, row['pair_id'])
    prompts = sorted({text for pairs in specs.values() for pair in pairs for text in pair.values()})
    encoded = {}
    for start in range(0, len(prompts), 4):
        batch = prompts[start:start + 4]
        context, mask = model.encode_prompt(batch)
        encoded.update({text: (context[i:i+1], mask[i:i+1]) for i, text in enumerate(batch)})
    print(f'[language] seen_prompts={len(prompts)} paired_keys={sum(bool(v) for v in specs.values())}', flush=True)
    return {key: [{language: encoded[text] for language, text in pair.items()} for pair in pairs]
            for key, pairs in specs.items() if pairs}
