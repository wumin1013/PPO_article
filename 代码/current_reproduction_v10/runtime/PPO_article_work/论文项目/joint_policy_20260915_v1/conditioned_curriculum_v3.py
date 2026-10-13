"""Use the same frozen paths with the time-scaled shaping potential."""
import numpy as np
from joint_runtime_v3 import JointRuntimeV3


def environment_factory(reference,*,fixed_d=None,max_time=None):
    length=np.linalg.norm(np.diff(reference,axis=0),axis=1).sum()
    return JointRuntimeV3(reference,fixed_d=fixed_d,max_time=max(5.,float(length)/20.+1.),
        closed=bool(np.linalg.norm(reference[0]-reference[-1])<1e-9))
