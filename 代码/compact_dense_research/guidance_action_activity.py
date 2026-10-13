"""Observe preview influence on visited physical states without changing commands."""
import math
import numpy as np
from continuous_gate_guidance import continuous_gate_target


def observe_preview_activity(runtime):
    env=runtime.env;original=env.guidance_target;maximum=[0.]
    def heading(point):
        vector=point-env.position
        return math.atan2(vector[1],vector[0]) if np.linalg.norm(vector)>1e-12 else env.heading
    def target():
        result=original();base=heading(result);distance=env.held[2]
        try:
            for alternative in [.8,6.,12.]:
                env.held[2]=alternative
                # Bypass the execution logger: one real guide call per tick.
                angle=heading(continuous_gate_target(env))
                maximum[0]=max(maximum[0],abs((angle-base+math.pi)%(2*math.pi)-math.pi))
        finally:
            env.held[2]=distance
        return result
    env.guidance_target=target
    def reset():maximum[0]=0.
    def value():return maximum[0]
    return reset,value
