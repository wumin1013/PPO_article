"""Continuous signed slack exactly equivalent to the recorded heading-count guard."""
from functools import lru_cache
import numpy as np
from duration_joint_metrics import guard_names


@lru_cache(maxsize=2048)
def raw_heading_slack(trace, allowed_count):
    with np.load(trace) as z:
        ref, rows = z['reference'].copy(), z['rows'].copy()
    u,v=np.diff(ref,axis=0);u=u/np.linalg.norm(u);v=v/np.linalg.norm(v)
    angle=float(np.arctan2(u[0]*v[1]-u[1]*v[0],u@v))
    normal=np.sign(angle)*np.array([-u[1],u[0]])
    speed=np.linalg.norm(rows[:,2:4],axis=1)
    heading=np.unwrap(np.arctan2(rows[:,2:4]@normal,rows[:,2:4]@u))[speed>5.]
    limit=abs(angle)+np.deg2rad(.5)
    actual_count=int((heading>limit).sum())
    if allowed_count>=len(heading):
        margin=1.
    else:
        # At most N strict exceedances iff the (N+1)-th largest value is <= limit.
        boundary=np.partition(heading,-allowed_count-1)[-allowed_count-1]
        margin=float((limit-boundary)/np.deg2rad(.5))
    assert (actual_count<=allowed_count)==(margin>=0.)
    return margin,actual_count


def learning_guards(records,baseline,original):
    values=np.asarray(original,dtype=float).copy();n=len(records)
    names=guard_names();offset=names.index('shape.heading_overshoot_gt05_duration_s')
    assert len(values)==n*(11+len(names))
    for i,(r,b) in enumerate(zip(records,baseline)):
        if not r['success']:continue
        allowed=int(round(b['shape']['heading_overshoot_gt05_duration_s']/.001))
        margin,count=raw_heading_slack(str(r['trace']),allowed)
        assert count==int(round(r['shape']['heading_overshoot_gt05_duration_s']/.001))
        row=11*n+len(names)*i+offset
        assert (values[row]>=0.)==(margin>=0.)
        values[row]=margin
    return values
