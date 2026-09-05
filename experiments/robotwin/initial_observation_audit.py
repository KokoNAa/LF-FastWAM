"""Optional read-only first-policy-input audit for matched CIS evaluations."""
from __future__ import annotations
import json
import os
from pathlib import Path
import re

import numpy as np

from experiments.robotwin.no_eraf_probe import CAMERAS, observation_hash, typed_hash


class InitialObservationAudit:
    def __init__(self, directory):
        self.directory = Path(directory).expanduser().resolve()
        self.reset()

    @classmethod
    def from_environment(cls):
        value = os.environ.get('FASTWAM_ROBOTWIN_INITIAL_OBSERVATION_AUDIT', '').strip()
        return cls(value) if value else None

    def reset(self):
        self.metadata = None
        self.written = False

    def begin(self, metadata):
        required = {'pair_id', 'source_task', 'task_config', 'scene_seed', 'episode_index',
                    'source_instruction', 'counterfactual_instruction', 'policy_instruction',
                    'condition', 'checkpoint'}
        if not required <= metadata.keys():
            raise ValueError('Initial observation audit requires complete CIS episode metadata.')
        self.metadata = dict(metadata)

    def record(self, observation, instruction):
        if self.written:
            return
        if self.metadata is None or instruction != self.metadata['policy_instruction']:
            raise ValueError('Initial observation audit episode/instruction mismatch.')
        arrays = {'state': np.asarray(observation['joint_action']['vector']),
                  **{c: np.asarray(observation['observation'][c]['rgb']) for c in CAMERAS}}
        if arrays['state'].shape != (14,) or not np.isfinite(arrays['state']).all():
            raise ValueError('Expected a finite14D initial robot state.')
        for camera in CAMERAS:
            x = arrays[camera]
            if x.ndim != 3 or x.shape[-1] != 3 or x.dtype != np.uint8:
                raise ValueError('Expected raw RGB camera input.')
        parts = [str(self.metadata[k]) for k in ('pair_id', 'task_config', 'condition', 'scene_seed', 'episode_index')]
        if any(not re.fullmatch('[a-zA-Z0-9_-]+', p) for p in parts):
            raise ValueError('Invalid audit filename field.')
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / ('__'.join(parts) + '.json')
        value = {'format': 'robotwin_initial_policy_observation_v1', 'metadata': self.metadata,
                 'observation_sha256': observation_hash(arrays), 'state': arrays['state'].tolist(),
                 'arrays': {k: {'sha256': typed_hash(v), 'shape': list(v.shape), 'dtype': str(v.dtype)}
                            for k, v in arrays.items()}}
        with path.open('x') as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write('\n')
        self.written = True
