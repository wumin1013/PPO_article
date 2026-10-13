"""Explicit monotone feed-output calibration; no added policy input or memory."""
import numpy as np
from compact_control import CompactPolicy


class CalibratedFeedPolicy(CompactPolicy):
    def __init__(self, gain=1., seed=922302):
        super().__init__(14,3,seed)
        if gain < 1.:
            raise ValueError('The registered feed-response probe only tests gain>=1')
        self.feed_gain=float(gain)

    def act(self, observation, deterministic=False):
        if not deterministic:
            raise ValueError('Diagnostic parameter probe, not a PPO sampling distribution')
        action,logp,value=super().act(observation,True)
        # z=.6 corresponds to80mm/s. The unchanged executor clips at100mm/s.
        action=np.array(action,copy=True)
        action[1]+= (self.feed_gain-1.)*max(float(action[1])-.6,0.)
        return action,logp,value
