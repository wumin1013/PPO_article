"""Explicit observation support restriction; never a hidden action or speed rule."""
import numpy as np


class RemainingSupportPolicy:
    def __init__(self,base,upper):
        self.base=base;self.upper=float(upper)

    def transform(self,observation):
        x=np.asarray(observation).copy()
        x[15]=min(float(x[15]),self.upper)
        return x

    def bind(self,env):
        self.base.bind(env)
        return env

    def act(self,observation,deterministic=True):
        return self.base.act(self.transform(observation),deterministic)
