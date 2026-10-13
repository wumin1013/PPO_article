"""Learned three-action residual on observed cross-track error at visible bends."""
import numpy as np


def turn_feedback_features(observation):
    x=np.asarray(observation,dtype=float)
    directions=x[42:62].reshape(10,2)
    angles=np.unwrap(np.arctan2(directions[:,1],directions[:,0]))
    bend=float(np.max(np.abs(angles-angles[0])*np.clip(x[62:72],0.,1.)))
    support=bend**4/(bend**4+.25**4)
    norm=max(float(np.hypot(x[6],x[7])),1e-8)
    error=(x[7]*x[0]+x[6]*x[1])/norm
    return support*np.array([error,abs(error),abs(error)])


class TurnFeedbackPolicy:
    def __init__(self,base,coefficients):self.base=base;self.coefficients=np.asarray(coefficients,dtype=float)
    def bind(self,env):return self.base.bind(env)
    def act(self,observation,deterministic=True):
        latent=self.base.act(observation,deterministic)[0].astype(float)
        return latent+self.coefficients*turn_feedback_features(observation),0.,0.
