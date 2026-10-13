"""Add a fitted feed residual on distributed visible curvature, not corner IDs."""
import numpy as np
from cruise_residual_policy import CruiseResidualPolicy


def arc_feature(observation,curvature_scale):
    x=np.asarray(observation)
    indices=np.array([3,5,7,9]);distances=np.array([3.,7.,15.,30.])
    directions=x[42:62].reshape(10,2)[indices].astype(float)
    angles=np.unwrap(np.arctan2(directions[:,1],directions[:,0]))
    curvature=np.abs(np.diff(angles)/np.diff(distances))
    # Numerical floor removes straight-line roundoff, not a physical speed cap.
    minimum=max(float(curvature.min())-1e-8,0.)
    valid=float(np.min(x[62+indices]))
    return valid*minimum**2/(minimum**2+float(curvature_scale)**2)


class ArcCruiseResidualPolicy:
    def __init__(self,base,upper,amplitude,width,arc_amplitude,arc_scale,recovery_width=None):
        self.base=CruiseResidualPolicy(base,upper,amplitude,width,recovery_width)
        self.arc_amplitude=float(arc_amplitude);self.arc_scale=float(arc_scale)

    def bind(self,env):return self.base.bind(env)

    def act(self,observation,deterministic=True):
        latent=self.base.act(observation,deterministic)[0]
        if self.arc_amplitude:
            latent=latent.copy();latent[1]+=self.arc_amplitude*arc_feature(observation,self.arc_scale)
        return latent,0.,0.


def build_policy(base,spec,upper=.1412694752216339):
    if 'arc_amplitude' in spec:
        policy=ArcCruiseResidualPolicy(base,upper,spec['amplitude'],spec['width'],spec['arc_amplitude'],spec['arc_scale'],spec.get('recovery_width'))
    else:
        policy=CruiseResidualPolicy(base,upper,spec['amplitude'],spec['width'],spec.get('recovery_width'))
    if 'turn_feedback' in spec:
        from turn_feedback_policy import TurnFeedbackPolicy
        policy=TurnFeedbackPolicy(policy,spec['turn_feedback'])
    return policy
