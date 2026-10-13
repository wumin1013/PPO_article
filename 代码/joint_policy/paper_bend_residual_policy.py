"""A learned three-action residual on visible tangent change, over a frozen backbone."""
import numpy as np
from paper_state_support_policy import RemainingSupportPolicy


def bend_feature(observation):
    d=np.asarray(observation)[42:62].reshape(10,2).astype(float)
    angles=np.unwrap(np.arctan2(d[:,1],d[:,0]))
    delta=(angles-angles[0])*np.clip(np.asarray(observation)[62:72],0.,1.)
    value=delta[np.argmax(np.abs(delta))]
    odd=float(np.tanh(value/.5))
    return np.array([odd,abs(odd),abs(odd)])


class BendResidualPolicy(RemainingSupportPolicy):
    def __init__(self,base,upper,coefficients,arm='joint'):
        super().__init__(base,upper)
        self.coefficients=np.asarray(coefficients,float)
        self.arm=arm

    def act(self,observation,deterministic=True):
        original=self.base.act(self.transform(observation),deterministic)[0]
        action=original.astype(float)+bend_feature(observation)*self.coefficients
        if self.arm=='no_steering':action[0]=0.
        if self.arm=='constant_feed':action[1]=20.  # tanh(20) is exactly one in the common float64 decoder.
        return action,0.,0.
