"""Shared local-observation residual for paired conditional parameter fitting."""
import numpy as np
from turn_feedback_policy import turn_feedback_features

class CoordinatedCruisePolicy:
    def __init__(self,base,spec):self.base=base;self.spec=dict(spec)
    def bind(self,env):return self.base.bind(env)
    def act(self,observation,deterministic=True):
        x=np.asarray(observation).copy()
        x[[16,17,19]]*=self.spec.get('global_fraction',1.)
        latent=self.base.act(x,deterministic)[0].astype(float)
        directions=x[42:62].reshape(10,2);angles=np.unwrap(np.arctan2(directions[:,1],directions[:,0]))
        bend=float(np.max(abs(angles-angles[0])*np.clip(x[62:72],0.,1.)))
        heading=abs(np.arctan2(x[7],x[6]))
        low,high=self.spec['low'],self.spec['high']
        if not high>low:raise ValueError('Cruise support width must be positive')
        fraction=np.clip((high-max(bend,heading))/(high-low),0.,1.)
        support=fraction*fraction*(3.-2.*fraction)
        latent[1]+=self.spec['amplitude']*support
        latent+=np.asarray(self.spec.get('feedback_delta',[0.,0.,0.]))*turn_feedback_features(x)
        return latent,0.,0.
