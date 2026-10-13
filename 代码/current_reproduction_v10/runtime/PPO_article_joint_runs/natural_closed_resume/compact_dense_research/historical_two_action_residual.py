"""Genuine two-output residual actor with a single global executed preview.

The historical common backbone remains shared. Its internal preview proposal is
available as a feature, but never changes the executed constant preview. The
new actor, sampled Gaussian and PPO likelihood all have exactly two dimensions.
"""
import numpy as np
import torch
from compact_control import CompactPolicy
from historical_residual_ppo import HistoricalResidualRuntime


class TwoActionResidualPolicy(CompactPolicy):
    def __init__(self,dimension=17,action_dim=2,seed=922302):
        if (dimension,action_dim)!=(17,2):raise ValueError('Expected17inputs and exactly2outputs')
        super().__init__(dimension,action_dim,seed)
        with torch.no_grad():
            self.actor[-1].bias.zero_()
            self.log_std.copy_(torch.log(torch.tensor([.01,.04])))


class HistoricalTwoActionRuntime(HistoricalResidualRuntime):
    def __init__(self,reference,*,dimension=17,action_dim=2,preview_rule_mm=4.,
                 allow_swing=True,potential_speed=30.,remaining_encoding='log'):
        if (dimension,action_dim)!=(17,2):raise ValueError('Expected17inputs and exactly2outputs')
        if not .8<=preview_rule_mm<=12.:raise ValueError('Preview outside shared permitted range')
        super().__init__(reference,dimension=17,action_dim=3,preview_rule_mm=preview_rule_mm,
            allow_swing=allow_swing,potential_speed=potential_speed,remaining_encoding=remaining_encoding)
        self.env.fixed_d=float(preview_rule_mm)
        # Rebind before any observation/action so all inherited geometry/context
        # components see the actual fixed-preview execution from the start.
        self.base.bind(self.env)
        self.action_dim=2

    def step(self,action):
        action=np.asarray(action,dtype=float)
        if action.shape!=(2,) or not np.isfinite(action).all():
            raise ValueError('Exactly2finite raw residual actions required')
        result=super().step(np.r_[action,0.])
        result[3]['sampled_residual_action']=action.copy()
        result[3]['actor_output_dimension']=2
        assert result[3]['held_action'][2]==self.env.fixed_d
        return result
