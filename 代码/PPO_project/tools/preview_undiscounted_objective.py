"""Finite-episode time objective with unchanged execution and terminal audit.

This independent candidate is not installed by importing it. Configure an
existing coherent-distance config and explicitly select UndiscountedEnvironment.
The matching PPO must consume that config's gamma=1; all other learning
settings, physical commands and the audited +10/-20 terminal values remain.
"""
from __future__ import annotations

import copy
import math

from preview_coherent_distance import CoherentDistanceEnvironment


VERSION = 'finite-episode-undiscounted-time-v1'


def configure(cfg):
    """Copy a coherent preview_time config and remove physical discounting."""
    revision = cfg['revision']
    if revision['candidate'] != 'preview_time' or 'coherent_distance' not in revision:
        raise ValueError('Requires the coherent-distance preview_time candidate')
    max_steps = revision['episode_max_steps']
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError('Requires a finite positive integer episode max_steps')
    if cfg['environment']['max_steps'] != max_steps:
        raise ValueError('environment.max_steps and revision.episode_max_steps must agree')
    if cfg['environment']['interpolation_period'] != 0.001:
        raise ValueError('This isolated objective requires the original 1 ms physical period')
    result = copy.deepcopy(cfg)
    result['ppo']['gamma'] = 1.0
    result['revision'].update(
        undiscounted_objective=VERSION,
        objective='New ordered coverage minus full physical elapsed seconds plus audited terminal value',
        discount_change='Only remove physical discount at both macro PPO and tick/brake/terminal reward levels')
    return result


class UndiscountedEnvironment(CoherentDistanceEnvironment):
    """Revalue each original macro after its unchanged physical execution."""

    def __init__(self, cfg, path_id, reference=None):
        if (cfg['revision'].get('undiscounted_objective') != VERSION
                or cfg['revision']['candidate'] != 'preview_time'
                or cfg['ppo']['gamma'] != 1.0):
            raise ValueError('Use configure(cfg) and gamma=1 with this environment')
        configure(cfg)  # Revalidate the effective horizon and 1 ms assumptions.
        super().__init__(cfg, path_id, reference=reference)

    def step(self, action):
        trace_start = len(self.trace)
        physics_before = self.physics_steps
        state, _, done, info = super().step(action)
        physical_count = self.physics_steps - physics_before
        rows = self.trace[trace_start:]
        # The original physical rows already contain ordered delta - 0.001.
        # Excluding the initial row prevents rewarding reset-time coverage.
        reward = math.fsum(row['reward'] for row in rows[:physical_count])
        if len(rows) > physical_count:
            # Brake/rest rows have no progress reward. The parent duration
            # includes both phases; subtract only the appended tail once.
            tail_time = self.last_macro_duration_s - physical_count * 0.001
            reward -= tail_time
        if done:
            reward += 10.0 if self.result['success'] else -20.0
        return state, float(reward), done, info
