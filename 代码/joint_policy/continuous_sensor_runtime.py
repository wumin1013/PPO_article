"""Continuous local reference sensing; the physical executor is inherited intact."""
import numpy as np
from joint_runtime import SENSOR_DISTANCES
from joint_runtime_v5 import JointRuntimeV5


SENSOR_HALF_WIDTH_MM = 1.0  # 100 mm/s * 20 ms / 2, shared across every geometry.


def continuous_sensor_observation(env, observation=None):
    result=env.observation().copy() if observation is None else observation.copy()
    progress,index=env.progress()
    total=env.corridor.arcs[-1]
    center=np.clip(progress+SENSOR_DISTANCES,0.,total)
    lower=np.maximum(0.,center-SENSOR_HALF_WIDTH_MM)
    upper=np.minimum(total,center+SENSOR_HALF_WIDTH_MM)
    a,_=env.sample_reference(lower);b,_=env.sample_reference(upper)
    chords=(b-a)/(upper-lower)[:,None]
    tangent=env.corridor.tangents[index]
    frame=np.stack([tangent,[-tangent[1],tangent[0]]],axis=1)
    result[42:62]=(chords@frame).ravel()
    return result


class ContinuousSensorRuntime(JointRuntimeV5):
    def observation(self):
        return continuous_sensor_observation(self,super().observation())


def environment_factory(reference,*,fixed_d=None,max_time=None):
    length=np.linalg.norm(np.diff(reference,axis=0),axis=1).sum()
    return ContinuousSensorRuntime(reference,fixed_d=fixed_d,
        max_time=max(5.,float(length)/20.+1.),
        closed=bool(np.linalg.norm(reference[0]-reference[-1])<1e-9))
