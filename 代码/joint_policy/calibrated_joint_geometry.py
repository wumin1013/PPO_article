"""Diagonal physical response model that reproduces observed signed axis probes."""
import numpy as np
from audit_function_constraint_interference import distances


def calibrated_geometry(source,current,paired,queries):
    margins,linear,quadratic=[],[],[]
    for index,(record,query) in enumerate(zip(current['records'],queries)):
        assert record['path']==query['path']
        with np.load(source/record['trace']) as z:
            ref,nominal=z['reference'].copy(),z['rows'][:,:2].copy()
        q=np.asarray(query['sampled_ticks'])-1
        base=distances(ref,nominal[q])
        g=np.zeros((len(q),len(paired)))
        h=np.zeros_like(g)
        for axis,pair in enumerate(paired):
            values=[]
            valid=[]
            for candidate in pair:
                r=candidate['records'][index]
                with np.load(source/r['trace']) as z:xy=z['rows'][:,:2].copy()
                available=np.ones(len(q),dtype=bool) if r['success'] else q<len(xy)
                value=np.zeros(len(q))
                # A completed stopped path stays at its recorded endpoint.
                # Failed trajectories are never extended beyond observed ticks.
                value[available]=distances(ref,xy[np.minimum(q[available],len(xy)-1)])
                values.append(value);valid.append(available)
            low,high=values
            lo,hi=valid
            both=lo&hi
            g[both,axis]=.5*(high[both]-low[both])
            h[both,axis]=.5*(high[both]+low[both])-base[both]
            g[hi&~lo,axis]=high[hi&~lo]-base[hi&~lo]
            g[lo&~hi,axis]=base[lo&~hi]-low[lo&~hi]
        margins.append(1.25-base);linear.append(g);quadratic.append(h)
    margin,g,h=np.concatenate(margins),np.vstack(linear),np.vstack(quadratic)
    checks=[]
    for axis,pair in enumerate(paired):
        for sign,candidate in zip((-1,1),pair):
            if not all(r['success'] for r in candidate['records']):continue
            minimum=float((margin-sign*g[:,axis]-h[:,axis]).min())
            checks.append(dict(tag=candidate['tag'],minimum_predicted_point_margin_mm=minimum))
    bad=[r for r in checks if r['minimum_predicted_point_margin_mm'] < -1e-7]
    return margin,g,h,dict(known_complete_axes=len(checks),false_exclusions=len(bad),
        complete_axis_checks=checks,
        scope='Exact sampled-axis point-distance calibration; no cross-axis or complete-segment safety certificate.')
