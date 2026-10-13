"""A shared extra speed-deficit objective; the physical executor is unchanged."""
import numpy as np
from direct_compact_runtime import DirectCompactRuntime


class SpeedDeficitRuntime(DirectCompactRuntime):
    def step(self,action):
        x,reward,done,info=super().step(action)
        rows=info['rows']
        deficit_s=float(np.maximum(100.-np.linalg.norm(rows[:,2:4],axis=1),0.).sum()*.001/100.)
        info['additional_speed_deficit_penalty_s']=deficit_s
        return x,reward-deficit_s,done,info
