"""Exactly14inputs and2policy outputs on the same physical execution chain.

The global nonlearned preview is an environment parameter. This is an independent
two-output network, not a three-output actor whose last sample is ignored.
"""
import numpy as np
import torch
from compact_control import CompactPolicy
from compact_farvector_runtime import FarVectorRuntime
from farvector_time_runtime import FarVectorTimeRuntime


class DirectTwoPolicy(CompactPolicy):
    def __init__(self,seed=922302):
        super().__init__(14,2,seed)

    def distribution(self,x):
        return torch.distributions.Normal(self.actor(x),self.log_std.clamp(-8.,-.3).exp())

    def initialize_common_layers(self,three_state):
        state=self.state_dict()
        for key in state:
            source=three_state[key]
            if key in ['actor.4.weight','actor.4.bias','log_std']:
                source=source[:2]
            assert state[key].shape==source.shape,(key,state[key].shape,source.shape)
            state[key]=source.clone()
        self.load_state_dict(state)


class DirectTwoRuntime(FarVectorRuntime):
    def __init__(self,reference,*,preview_rule_mm=3.,potential_speed=100.):
        if not .8<=preview_rule_mm<=12.:
            raise ValueError('Nonlearned preview must stay inside the common range')
        super().__init__(reference,potential_speed=potential_speed)
        self.preview_rule_mm=float(preview_rule_mm)
        self.action_dim=2
        self.env.fixed_d=self.preview_rule_mm
        self.env.held[2]=self.preview_rule_mm

    def step(self,action):
        action=np.asarray(action,dtype=float)
        if action.shape!=(2,) or not np.isfinite(action).all():
            raise ValueError('Exactly two finite actor outputs required')
        # The common three-channel physical interface receives the configured
        # environment preview; the policy does not sample or predict it.
        result=super().step(np.r_[action,0.])
        assert result[3]['held_action'][2]==self.preview_rule_mm
        result[3]['actor_output_dimension']=2
        return result

    def restore(self,state):
        super().restore(state)
        self.env.fixed_d=self.preview_rule_mm
        self.env.held[2]=self.preview_rule_mm


class DirectTwoTimeRuntime(DirectTwoRuntime,FarVectorTimeRuntime):
    pass
