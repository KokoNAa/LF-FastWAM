"""Store later observations without duplicating frozen text and attention masks."""
from __future__ import annotations

from collections import OrderedDict


def capture_delta(captured, parent):
    import torch
    delta = {}
    for group, values in captured.items():
        changes = {}
        for key, value in values.items():
            old = parent[group][key]
            if isinstance(value, torch.Tensor):
                if torch.equal(value, old):
                    continue
                if (key == 'context' and value.shape == old.shape
                        and torch.equal(value[:, :-1], old[:, :-1])):
                    changes[key] = {'last_token': value[:, -1:].clone()}
                else:
                    changes[key] = value
            elif value != old:
                changes[key] = value
        if changes:
            delta[group] = changes
    return delta


def restore_capture(parent, delta):
    import torch
    result = {group: dict(values) for group, values in parent.items()}
    for group, changes in delta.items():
        for key, value in changes.items():
            if isinstance(value, dict) and set(value) == {'last_token'}:
                result[group][key] = torch.cat((parent[group][key][:, :-1], value['last_token']), dim=1)
            else:
                result[group][key] = value
    return result


class ReplayPayloads:
    """Bound CPU parent storage and materialize just the current GPU sample."""
    def __init__(self, rows, device, capacity=32):
        self.paths = {row['id']: row['payload'] for row in rows}
        self.device = device
        self.capacity = capacity
        self.parents = OrderedDict()

    def _parent(self, path):
        import torch
        if path not in self.parents:
            self.parents[path] = torch.load(path, map_location='cpu', weights_only=True)
            if len(self.parents) > self.capacity:
                self.parents.popitem(last=False)
        self.parents.move_to_end(path)
        return self.parents[path]

    def __getitem__(self, key):
        import torch
        from experiments.robotwin.same_state_repair import move_cache
        payload = torch.load(self.paths[key], map_location='cpu', weights_only=True)
        if payload.get('format') == 'robotwin_compact_replay_v1':
            parent = self._parent(payload['parent_payload'])
            payload = {'references': payload['references'], 'captured': {
                language: restore_capture(parent['captured'][language], changes)
                for language, changes in payload['capture_deltas'].items()}}
        return move_cache(payload, self.device)
