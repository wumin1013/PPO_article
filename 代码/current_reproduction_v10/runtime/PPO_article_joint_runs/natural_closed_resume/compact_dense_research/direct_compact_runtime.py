"""Fourteen direct policy inputs on the common executor, without a teacher at run time."""
import math
import numpy as np
from compact_control import CompactRuntime, feature_names, local_progress
from turn_aware_projector import environment_factory


class DirectCompactRuntime:
    def __init__(self,reference,*,dimension=14,action_dim=3,preview_rule_mm=3.,
                 allow_swing=True,potential_speed=30.,remaining_encoding='log'):
        if (dimension,action_dim,allow_swing,remaining_encoding)!=(14,3,True,'log'):
            raise ValueError('This compression pilot requires14inputs and3direct outputs')
        self.env=environment_factory(reference)
        self.env.dimension=14
        self.potential_speed=float(potential_speed)

    def __getattr__(self,name):return getattr(self.env,name)

    def observation(self):
        local=CompactRuntime.observation(self.env)
        remaining=max(0.,self.env.corridor.arcs[-1]-local_progress(self.env))
        local[-1]=math.log1p(remaining/30.)/8.
        return local

    def feature_names(self):
        names=feature_names(14)
        names[-1]='log1p_remaining_distance_over30_div8'
        return names

    def potential(self):
        return (self.env.progress()[0]-self.env.corridor.arcs[-1])/self.potential_speed

    def step(self,action):
        action=np.asarray(action,dtype=float)
        if action.shape!=(3,) or not np.isfinite(action).all():raise ValueError('Three finite direct outputs required')
        # Encoding changes only coordinates; the same physical decoder, guide,
        # projector, coverage checker and terminal execution are retained.
        latent=np.arctanh(np.clip(action,-1.+1e-10,1.-1e-10))
        phi=self.potential()
        _,_,done,info=self.env.step(latent)
        reward=-info['elapsed_s']+(0. if done else self.potential())-phi
        if done and not self.env.success:
            remaining=max(0.,self.env.corridor.arcs[-1]-self.env.progress()[0])
            reward-=self.env.max_ticks*.001+1.+remaining/20.
        return self.observation(),float(reward),done,info

    def state(self):
        return dict(physical=self.env.state(),projector_counts=dict(self.env.experimental_projector.counts))

    def restore(self,state):
        self.env.restore(state['physical'])
        self.env.experimental_projector.counts=dict(state['projector_counts'])
