"""Same acceptance contract; exact nearest-segment broad phase for distance statistics only.

Position rows include the initial point. Vector differences use zero initial
velocity/acceleration. Failures retain prefix metrics but have no completion time.
"""
import numpy as np


def metrics(reference, positions, dt=.001, tolerance=1.25, limits=(100.,2000.,62500.),
            declared_success=True, reason=None, geometry=True):
    from runtime_bootstrap import setup
    setup()
    from geometry_spec import Corridor, GateTracker
    ref=np.asarray(reference,float); p=np.asarray(positions,float)
    if len(p)<2 or not np.isfinite(p).all():
        return dict(success=False,reason=reason or 'empty_or_nonfinite_trace',complete_time_s=None,samples=len(p))
    v=np.diff(p,axis=0)/dt
    a=np.diff(np.vstack([np.zeros(2),v]),axis=0)/dt
    j=np.diff(np.vstack([np.zeros(2),a]),axis=0)/dt
    norms=[np.linalg.norm(z,axis=1) for z in (v,a,j)]
    peaks=[float(z.max(initial=0)) for z in norms]
    ratios=np.asarray(peaks)/limits
    physical=bool(np.all(ratios<=1.+2e-5))
    rest=(norms[0]<=.001)&(norms[1]<=.01)
    nonrest=np.flatnonzero(~rest)
    rest_ticks=len(rest)-(int(nonrest[-1])+1 if len(nonrest) else 0)
    resting=rest_ticks*dt>=.02-1e-10
    corridor=Corridor(ref,epsilon=tolerance,closed=bool(np.linalg.norm(ref[0]-ref[-1])<1e-9))
    from exact_distance_index_v5 import SegmentDistanceIndex
    distances=SegmentDistanceIndex(ref).distances(p)
    geo=None
    if geometry:
        tracker=GateTracker(corridor,p[0])
        for point in p[1:]:tracker.update(point)
        geo=tracker.snapshot(endpoint_tolerance=tolerance)
    complete=bool(declared_success and physical and resting and geo is not None and geo['geometry_complete'])
    reasons=[]
    if not declared_success:reasons.append(reason or 'executor_failure')
    if not physical:reasons.append('vector_limit_violation')
    if not resting:reasons.append('terminal_rest_incomplete')
    if geo is not None and not geo['geometry_complete']:reasons.append('geometry_or_ordered_completion_failure')
    return dict(success=complete,reason='success' if complete else ';'.join(reasons),complete_time_s=(len(p)-1)*dt if complete else None,
        elapsed_prefix_s=(len(p)-1)*dt,samples=len(p),dt_s=dt,tolerance_mm=tolerance,limits=list(limits),
        vector_peaks=peaks,vector_limit_ratios=ratios.tolist(),kinematic_pass=physical,rest_pass=resting,
        terminal_rest_s=rest_ticks*dt,geometry=geo,max_polyline_deviation_mm=max(distances),
        rms_polyline_deviation_mm=float(np.sqrt(np.mean(np.square(distances)))),
        executed_length_mm=float(norms[0].sum()*dt),source='independent finite differences from positions; continuous command-capsule union and ordered gate audit',
        numerical_scope='piecewise linear commands; vector backward differences, not continuous servo dynamics')
