"""Ordered pure-pursuit circle exit; geometry only, no speed or turn templates."""
import math
import numpy as np


def circle_exit_target(env,progress,distance):
    """First forward circle exit on the ordered reference branch.

    If no future line segment intersects the lookahead circle, preserve the
    existing arc query. Near the terminal endpoint contained in the circle,
    return that endpoint. Neither routine changes physical constraints.
    """
    _,index=env.progress()
    radius=float(distance);position=env.position;corridor=env.corridor
    for i in range(index,len(corridor.lengths)):
        offset=max(0.,progress-corridor.arcs[i]) if i==index else 0.
        offset=min(float(corridor.lengths[i]),offset)
        start=corridor.starts[i]+offset*corridor.tangents[i]
        length=float(corridor.lengths[i])-offset
        delta=start-position;tangent=corridor.tangents[i]
        projection=float(delta@tangent)
        discriminant=projection**2+radius**2-float(delta@delta)
        if discriminant<0:continue
        exit_distance=-projection+math.sqrt(max(discriminant,0.))
        if -1e-10<=exit_distance<=length+1e-10:
            return start+np.clip(exit_distance,0.,length)*tangent,tangent
        endpoint=start+length*tangent
        if np.linalg.norm(endpoint-position)>radius+1e-9 and exit_distance<0:
            continue
    if np.linalg.norm(corridor.reference[-1]-position)<=radius+1e-9:
        return corridor.reference[-1].copy(),corridor.tangents[-1]
    return env.sample_reference(progress+radius)
