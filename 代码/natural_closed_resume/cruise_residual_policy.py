"""A continuous local-observation feed residual with fitted shared parameters."""
import numpy as np


class CruiseResidualPolicy:
    def __init__(self, base, upper, amplitude, width, recovery_width=None):
        self.base=base;self.upper=float(upper)
        self.amplitude=float(amplitude);self.width=float(width)
        self.recovery_width=float(width if recovery_width is None else recovery_width)

    def bind(self,env):
        self.base.bind(env)
        return env

    def feature(self,observation):
        x=np.asarray(observation)
        directions=x[42:62].reshape(10,2).astype(float)
        angles=np.unwrap(np.arctan2(directions[:,1],directions[:,0]))
        bend=np.max(np.abs(angles-angles[0])*np.clip(x[62:72],0.,1.))
        # Reflection-even, continuous basis of existing local geometry/dynamics.
        # No position, corner ID, progress, time or target speed enters this basis.
        return float(np.exp(-((bend/self.width)**2+(x[7]/self.recovery_width)**2+(x[5]/.1)**2)))

    def act(self,observation,deterministic=True):
        x=np.asarray(observation).copy();x[15]=min(float(x[15]),self.upper)
        latent=self.base.act(x,deterministic)[0].astype(float)
        latent[1]+=self.amplitude*self.feature(observation)
        return latent,0.,0.
