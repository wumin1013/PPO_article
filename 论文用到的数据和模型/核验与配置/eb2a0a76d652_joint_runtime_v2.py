"""Bound global geometry observations and expose a wider common preview range."""
import math
import numpy as np
from joint_runtime import JointRuntime


class JointRuntimeV2(JointRuntime):
    max_preview = 12.

    @staticmethod
    def decode_action(latent,fixed_d=None):
        bounded = np.tanh(np.asarray(latent,float))
        if bounded.shape != (3,) or not np.isfinite(bounded).all():
            raise ValueError('Three finite latent actions required')
        return np.array([math.pi/2*bounded[0],50.*(1+bounded[1]),
                         6.4+5.6*bounded[2] if fixed_d is None else float(fixed_d)])

    def observation(self):
        observation = super().observation()
        observation[10] *= 4.6/12.
        # Distant gates/endpoints must not saturate the shared local encoder.
        observation[11:15] = np.clip(observation[11:15],-1.,1.)
        return observation
