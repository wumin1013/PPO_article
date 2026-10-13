"""One-change pilot: fixed geometry sensing with policy-selected control D.

Keep all 26 coordinates, policy architecture, optimizer, time objective, KCM,
decision period and braking from macro preview_time. Replace only the six
multiscale geometry coordinates by sensing at fixed physical distances.
"""
from pathlib import Path
import hashlib
import json
import math

import numpy as np

import preview_macro_study as core

OUT = core.OUT.parent/'sensor_study'
BASE_SOURCE = core.source_state
BASE_CONFIG = core.configuration
BASE_ENV = core.MacroEnvironment
SENSOR_BASE_MM = 3.0


def fixed_geometry(raw):
    """Read geometry without mutating preview, reference filter or progress."""
    _, _, s, _, _ = raw._project_onto_progress_path(raw.current_position, update=False)
    tangent = float(raw._tangent_angle_at_s(float(s)))
    values = []
    for scale in raw.lookahead_obs_scales:
        distance = SENSOR_BASE_MM*float(scale)
        point = raw._interpolate_progress_point_at_s(float(s)+distance)
        angle = raw._tangent_angle_at_s(float(s)+distance)
        values.extend([raw._wrap_angle(angle-tangent)/math.pi,
                       float(np.clip(np.linalg.norm(np.asarray(point)-raw.current_position)/distance, 0., 1.))])
    return np.asarray(values, dtype=np.float32)


class SensorEnvironment(BASE_ENV):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sense()

    def _sense(self):
        if self.observation_dim != 26 or self.inner.raw.enable_curvature_obs:
            raise ValueError('Registered sensor replacement requires 26D state without curvature coordinates')
        self.state = self.state.copy()
        geometry = fixed_geometry(self.inner.raw)
        if geometry.shape != (6,):
            raise ValueError('Exactly three registered sensing scales required')
        self.state[12:18] = geometry
        self.trace[-1].update({f'sensor_geometry_{i}': float(v) for i, v in enumerate(geometry)})

    def step(self, action):
        _, reward, done, info = super().step(action)
        self._sense()
        return self.state.copy(), reward, done, info


def configuration(candidate, method, seed):
    if candidate != 'preview_time':
        raise ValueError('Sensor candidate changes preview_time geometry sensing only')
    cfg = BASE_CONFIG(candidate, method, seed)
    cfg['revision'].update(study='fixed-geometry-sensing-control-preview-v1',
        sensor_base_mm=SENSOR_BASE_MM, sensor_distances_mm=[1.5, 3., 6.],
        sensor_state_coordinates=list(range(12, 18)),
        candidate_change='Only geometry observation coordinates 12:18 use fixed sensing ranges, shared by all fixed/adaptive methods',
        preview_timing='Observe fixed geometry and preceding held D; policy selects new feed and control D; KCM executes for 20 ms; then observe again',
        initialization='Fresh common MLP means [0,.05,(3-.8)/3.8], std [.03,.05,.15]; no pretraining',
        likelihood='One raw latent Gaussian draw scored per 20 ms macro decision; inactive dimensions masked',
        preview_initial_std=.15,
        physical_std_known_limitation='Additive std floor active; no inherited inactive-floor claim')
    return cfg


def source_state():
    state = BASE_SOURCE()
    state['source_files'][Path(__file__).relative_to(core.old.ROOT).as_posix()] = core.old.sha(__file__)
    state['source_sha256'] = hashlib.sha256(json.dumps(state['source_files'], sort_keys=True).encode()).hexdigest()
    return state


def prepare():
    core.old.check_protocol()
    value = dict(version='fixed-geometry-sensing-control-preview-v1', source=source_state(),
        manifest_sha256=core.old.sha(core.old.DATA/'manifest.json'),
        candidate='preview_time', methods=list(core.METHODS), seed=101,
        pilot_macro_decisions=4096, smoke_macro_decisions=32,
        change='Only six geometry observation coordinates: D-dependent .5D/D/2D -> fixed 1.5/3/6mm',
        unchanged='26D network, action semantics, previous-D and reference-memory state, initialization, PPO update, 20ms decisions, 1ms KCM, reward and complete rest audit',
        initial_screen=['fixed3', 'adaptive'],
        advance_rule='Run short/long fixed controls only if adaptive completes all validation paths and paired time is better than fixed3',
        test_access=False, data_role='Single-seed development screening; no repeatability or holdout claim',
        machine_execution_authorized=False)
    target = OUT/'protocol.json'
    if target.exists() and core.old.read(target) != value:
        raise ValueError('Keep frozen sensing study; use a new version for further changes')
    if not target.exists():
        core.old.write(target, value)
    return value


def install():
    core.OUT = OUT
    core.source_state = source_state
    core.configuration = configuration
    core.MacroEnvironment = SensorEnvironment
    core.prepare = prepare
    return core


if __name__ == '__main__':
    install().main()
