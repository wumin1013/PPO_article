"""Small continuous three-action residual using visible bend and tracking error."""
import numpy as np
from coordinated_cruise_policy import CoordinatedCruisePolicy
from turn_feedback_policy import turn_feedback_features


def bend_features(observation):
    x=np.asarray(observation,dtype=float)
    d=x[42:62].reshape(10,2)
    a=np.unwrap(np.arctan2(d[:,1],d[:,0]))
    bend=float(np.max(np.abs(a-a[0])*np.clip(x[62:72],0.,1.)))
    h=abs(float(np.arctan2(x[7],x[6])))
    magnitude=max(bend,h)
    support=np.clip((magnitude-.80)/.15,0.,1.)
    support=support*support*(3.-2.*support)
    centers=np.array([.97,1.40,2.30]);widths=np.array([.16,.37,.38])
    basis=np.exp(-.5*((magnitude-centers)/widths)**2)*support
    tracking=turn_feedback_features(x)
    # Steering/preview use error feedback; feed also acts before error grows.
    return basis, np.array([tracking[0],max(tracking[1],.15*support),tracking[2]])


class BendConditionedPolicy:
    def __init__(self,base,spec):
        self.base=CoordinatedCruisePolicy(base,spec['cruise_base'])
        self.weights=np.asarray(spec['bend_weights'],dtype=float).reshape(3,3)
    def bind(self,env):return self.base.bind(env)
    def act(self,observation,deterministic=True):
        latent=self.base.act(observation,deterministic)[0].astype(float)
        basis,feedback=bend_features(observation)
        latent+=self.weights@basis*feedback
        return latent,0.,0.
