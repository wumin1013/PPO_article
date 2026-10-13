"""Diagnostic first-order action hold; the policy still decides every20ms."""
import numpy as np
from gate_band_state import GateBandRuntime


class LinearActionHoldRuntime(GateBandRuntime):
    """Interpolate physical action endpoints, never states or executed motion."""
    def __init__(self, reference, *, interpolate_heads=(), **kwargs):
        super().__init__(reference, **kwargs)
        self.interpolate_heads = tuple(interpolate_heads)
        assert set(self.interpolate_heads) <= {0, 1, 2}
        self._executed_actions = None

    def guidance_target(self):
        if self._executed_actions is not None:
            fraction = min((self.ticks-self._action_start_tick+1)/20., 1.)
            action = self._action_goal.copy()
            if fraction < 1.:
                for h in self.interpolate_heads:
                    action[h] = self._action_origin[h] + fraction*(self._action_goal[h]-self._action_origin[h])
            self.held = action
            self._executed_actions.append(action.copy())
        return super().guidance_target()

    def step(self, latent):
        self._action_origin = self.held.copy()
        self._action_goal = self.decode_action(latent, self.fixed_d)
        self._action_start_tick = self.ticks
        self._executed_actions = []
        observation, reward, done, info = super().step(latent)
        executed = np.full((len(info['rows']), 3), np.nan)
        count = len(self._executed_actions)
        assert count == int((info['rows'][:, 12] == 0).sum())
        if count:
            executed[:count] = self._executed_actions
        info['executed_actions_1ms'] = executed
        info['requested_action'] = self._action_goal.copy()
        self._executed_actions = None
        return observation, reward, done, info
