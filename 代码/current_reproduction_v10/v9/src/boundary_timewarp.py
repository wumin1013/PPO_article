"""Finite boundary reparameterization of an already planned command trajectory.

The suffix and its last three predecessor positions are retained exactly.
Every candidate is checked using discrete planar vector limits. Geometry is
checked independently before adopting any result. Policy planning observations
are not relabelled as observations of the retimed execution.
"""
import numpy as np
from scipy.interpolate import CubicSpline

START_S=.020
END_S=.120
JOIN_S=.200
MAX_SAVING_S=.020
LIMITS=np.array([100.,2000.,62500.])


def candidates(positions,dt):
    positions=np.asarray(positions,float)
    join=round(JOIN_S/dt)
    if len(positions)<join+4:
        raise ValueError('Retain unchanged trajectory shorter than the boundary horizon')
    # Interpolant is evaluated only inside this original finite prefix.
    source_t=np.arange(join+1)*dt
    interpolant=CubicSpline(source_t,positions[:join+1],axis=0)
    out=[]
    for saved_ticks in range(round(MAX_SAVING_S/dt),0,-1):
        duration=JOIN_S-saved_ticks*dt
        target_t=np.arange(join-saved_ticks+1)*dt
        u=np.clip((target_t-START_S)/(END_S-START_S),0.,1.)
        blend=u**3*(10.-15.*u+6.*u*u)
        query=target_t+saved_ticks*dt*blend
        prefix=interpolant(query)
        # Exact samples outside the warp, including all finite-difference
        # boundary states and unchanged initial jerk ramp.
        before=target_t<=START_S
        after=target_t>=END_S
        prefix[before]=positions[np.flatnonzero(before)]
        prefix[after]=positions[np.flatnonzero(after)+saved_ticks]
        assert np.array_equal(prefix[-3:],positions[join-2:join+1])
        proof=np.vstack((prefix,positions[join+1:join+4]))
        v=np.diff(proof,axis=0)/dt
        a=np.diff(np.vstack((np.zeros(2),v)),axis=0)/dt
        j=np.diff(np.vstack((np.zeros(2),a)),axis=0)/dt
        peaks=np.array([np.linalg.norm(x,axis=1).max() for x in (v,a,j)])
        out.append((prefix,dict(saved_ticks=saved_ticks,saved_s=saved_ticks*dt,
            original_join_s=JOIN_S,new_join_s=duration,vector_peaks=peaks.tolist(),
            vector_pass=bool(np.all(peaks<=LIMITS*(1+1e-10))),
            exact_join_position_velocity_acceleration=True)))
    return out


def geometry_prefix(reference,prefix,original_join,*,tolerance):
    from runtime_bootstrap import setup
    setup()
    from geometry_spec import Corridor,GateTracker
    corridor=Corridor(reference,epsilon=tolerance,
                      closed=bool(np.linalg.norm(reference[0]-reference[-1])<1e-9))
    trial=GateTracker(corridor,prefix[0])
    expected=GateTracker(corridor,original_join[0])
    for p in original_join[1:]:expected.update(p)
    for p in prefix[1:]:trial.update(p)
    return bool(trial.inside and trial.frontier==expected.frontier and
                trial.pending_negative==expected.pending_negative)
