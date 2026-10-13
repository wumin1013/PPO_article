"""Exact event-count margins for unsmoothed XY and incoming lateral heading."""
from functools import lru_cache
import numpy as np
from duration_joint_metrics import guard_names
from cruise_duration_credit import event_learning_guards


def count_margin(values,limit,allowed,scale):
    """At most allowed strict exceedances iff this signed order statistic is >=0."""
    actual=int(np.count_nonzero(values>limit))
    margin=1. if allowed>=len(values) else float((limit-np.partition(values,-allowed-1)[-allowed-1])/scale)
    assert (actual<=allowed)==(margin>=0.)
    return margin,actual


@lru_cache(maxsize=2048)
def shape_count_margins(trace,allowed02,allowed005,allowed_reverse):
    with np.load(trace) as z:ref,rows=z['reference'],z['rows']
    u,v=np.diff(ref,axis=0);u=u/np.linalg.norm(u);v=v/np.linalg.norm(v)
    sign=np.sign(u[0]*v[1]-u[1]*v[0])
    outward=-(rows[:,:2]-ref[1])@(sign*np.array([-v[1],v[0]]))
    speed=np.linalg.norm(rows[:,2:4],axis=1)
    heading=np.unwrap(np.arctan2(rows[:,2:4]@(sign*np.array([-u[1],u[0]])),rows[:,2:4]@u))
    apex=int(np.linalg.norm(rows[:,:2]-ref[1],axis=1).argmin())
    eligible=(np.arange(len(rows))<apex)&(speed>5.)&(rows[:,12]==0)
    return [count_margin(outward,.02,allowed02,.005),
            count_margin(outward,.005,allowed005,.005),
            count_margin(-heading[eligible],np.deg2rad(.05),allowed_reverse,np.deg2rad(.05))]


def shape_event_learning_guards(records,baseline,original):
    values=event_learning_guards(records,baseline,original)
    names=guard_names();n=len(records)
    keys=['shape.out_edge_outward_gt02_duration_s','shape.out_edge_outward_gt005_duration_s',
          'shape.reverse_lateral_significant_duration_s']
    for i,(r,b) in enumerate(zip(records,baseline)):
        if not r['success']:continue
        allowed=[int(round(b['shape'][k.split('.')[1]]/.001)) for k in keys]
        margins=shape_count_margins(str(r['trace']),*allowed)
        for key,(margin,count) in zip(keys,margins):
            assert count==int(round(r['shape'][key.split('.')[1]]/.001))
            row=11*n+len(names)*i+names.index(key)
            assert (values[row]>=-1e-8)==(margin>=0.)
            values[row]=margin
    return values
