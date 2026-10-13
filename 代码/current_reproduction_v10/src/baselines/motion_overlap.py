"""Independent jerk-limited S-curve motion overlap (SCMO).

This is NOT a reproduction of Li et al. (2023) or Xu et al. (2026).
All positions and limits use mm and seconds. At most two point-to-point
motions overlap. A single common time dilation enforces vector limits over
each full polynomial interval, including startup and braking. Corner witness
times provide ordered vertex coverage. Geometric bounds concern commanded
motion only, not tracking error or swept cutter/workpiece clearance.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
import numpy as np


@dataclass
class SCurve:
    length: float
    phase_t: np.ndarray
    phase_j: np.ndarray
    phase_x: np.ndarray
    phase_v: np.ndarray
    phase_a: np.ndarray

    @property
    def duration(self):
        return float(self.phase_t[-1])

    def position(self, time):
        t = np.clip(np.asarray(time, dtype=float), 0, self.duration)
        idx = np.clip(np.searchsorted(self.phase_t, t, side="right")-1, 0, len(self.phase_j)-1)
        h = t-self.phase_t[idx]
        return self.phase_x[idx]+self.phase_v[idx]*h+self.phase_a[idx]*h*h/2+self.phase_j[idx]*h**3/6

    def time_at_position(self, distance):
        low, high = 0., self.duration
        for _ in range(45):
            mid = (low+high)/2
            if self.position(mid)<distance: low=mid
            else: high=mid
        return (low+high)/2


def _scurve(length, vmax, amax, jmax):
    """Exact rest-to-rest, zero-acceleration seven-phase scalar S curve."""
    tj = min(amax/jmax, np.sqrt(vmax/jmax))
    ta = max(0., vmax/(jmax*tj)-tj)
    d_at_v = vmax*(2*tj+ta)
    if length >= d_at_v:
        tv = (length-d_at_v)/vmax
    else:
        tv = 0.
        if length >= 2*amax**3/jmax**2:
            tj = amax/jmax
            ta = max(0., (np.sqrt(tj*tj+4*length/amax)-3*tj)/2)
        else:
            tj, ta = (length/(2*jmax))**(1/3), 0.
    durations = np.array([tj,ta,tj,tv,tj,ta,tj])
    jerks = jmax*np.array([1.,0.,-1.,0.,-1.,0.,1.])
    keep = durations > 1e-14
    durations, jerks = durations[keep], jerks[keep]
    times = np.r_[0.,np.cumsum(durations)]
    x = np.zeros(len(times)); v = x.copy(); a = x.copy()
    for k, (h,j) in enumerate(zip(durations,jerks)):
        x[k+1] = x[k]+v[k]*h+a[k]*h*h/2+j*h**3/6
        v[k+1] = v[k]+a[k]*h+j*h*h/2
        a[k+1] = a[k]+j*h
    if not np.isclose(x[-1],length,rtol=2e-11,atol=1e-10):
        raise ArithmeticError("S-curve displacement mismatch")
    return SCurve(length,times,jerks,x,v,a)


def _corner_bound(left, right, overlap):
    """Maximum min(remaining incoming distance, outgoing distance)."""
    lo, hi = 0., overlap
    for _ in range(45):
        t = (lo+hi)/2
        rem = left.length-float(left.position(left.duration-overlap+t))
        adv = float(right.position(t))
        if rem > adv:
            lo = t
        else:
            hi = t
    witness = (lo+hi)/2
    bound = float(right.position(witness))
    return bound, witness


def _clean_reference(reference):
    p = np.asarray(reference,dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 2 or not np.isfinite(p).all():
        raise ValueError("reference must be a finite N x 2 array with N >= 2")
    p = p[np.r_[True,np.linalg.norm(np.diff(p,axis=0),axis=1)>1e-12]]
    if len(p)<2:
        raise ValueError("reference has no nonzero segment")
    # Only exact straight, forward vertices are removed; no geometric fitting.
    keep = [p[0]]
    for i in range(1,len(p)-1):
        u, v = p[i]-keep[-1], p[i+1]-p[i]
        cross = u[0]*v[1]-u[1]*v[0]
        if abs(cross) <= 1e-12*np.linalg.norm(u)*np.linalg.norm(v) and np.dot(u,v)>0:
            continue
        keep.append(p[i])
    return np.asarray(keep+[p[-1]])


def _limits(limits):
    def pick(keys):
        for key in keys:
            if isinstance(limits,dict) and key in limits:
                return float(limits[key])
            if hasattr(limits,key):
                return float(getattr(limits,key))
        raise KeyError("missing one of "+str(keys))
    result = [pick(("v","v_max","vmax","max_velocity")),
              pick(("a","a_max","amax","max_acceleration")),
              pick(("j","j_max","jmax","max_jerk"))]
    if not np.isfinite(result).all() or min(result)<=0:
        raise ValueError("kinematic limits must be positive and finite")
    return result


def _interval_peak_v(v, a, j, h):
    # v(h*z) is quadratic; roots of d||v||^2/dz give every interior extremum.
    coeff = np.stack([v,a*h,j*h*h/2])
    squared = sum((np.convolve(coeff[:,k],coeff[:,k]) for k in range(2)),np.zeros(5))
    deriv = np.polynomial.polynomial.polyder(squared)
    roots = np.polynomial.polynomial.polyroots(np.trim_zeros(deriv,"b")) if np.any(deriv) else []
    candidates = [0.,1.]+[float(r.real) for r in roots if abs(r.imag)<1e-8 and 0<r.real<1]
    return max(float(np.linalg.norm(v+a*h*z+j*(h*z)**2/2)) for z in candidates)


def _build_events(curves,directions,starts):
    changes = {}
    def add(t,value):
        key = float(t)
        changes[key] = changes.get(key,np.zeros(2))+value
    for c, d, start in zip(curves,directions,starts):
        previous = 0.
        for t,j in zip(c.phase_t[:-1],c.phase_j):
            add(start+t,d*(j-previous)); previous=j
        add(start+c.duration,-d*previous)
    times = np.asarray(sorted(changes))
    n = len(times)-1
    pos = np.zeros((n+1,2)); vel=pos.copy(); acc=pos.copy(); jerk=np.zeros((n,2))
    # Reconstruct event states from the exact local S curves. Integrating one
    # long global jerk-difference sum accumulated roundoff over thousands of
    # events and caused terminal residuals on the original 44 m program.
    # Superposition is the same trajectory; each independent segment ends
    # exactly and contributes a constant displacement thereafter.
    for c,d,start in zip(curves,directions,starts):
        local=np.clip(times-start,0.,c.duration)
        idx=np.clip(np.searchsorted(c.phase_t,local,side='right')-1,0,len(c.phase_j)-1)
        h=local-c.phase_t[idx]
        x=c.phase_x[idx]+c.phase_v[idx]*h+c.phase_a[idx]*h*h/2+c.phase_j[idx]*h**3/6
        x=np.where(times>=start+c.duration,c.length,x)
        active=(times>=start)&(times<start+c.duration)
        v=np.where(active,c.phase_v[idx]+c.phase_a[idx]*h+c.phase_j[idx]*h*h/2,0.)
        a=np.where(active,c.phase_a[idx]+c.phase_j[idx]*h,0.)
        pos+=x[:,None]*d;vel+=v[:,None]*d;acc+=a[:,None]*d
        mid=.5*(times[:-1]+times[1:])-start
        phase=np.clip(np.searchsorted(c.phase_t,mid,side='right')-1,0,len(c.phase_j)-1)
        j=np.where((mid>=0)&(mid<c.duration),c.phase_j[phase],0.)
        jerk+=j[:,None]*d
    peaks = np.zeros(3)
    for i,h in enumerate(np.diff(times)):
        current=jerk[i]
        peaks[0] = max(peaks[0],_interval_peak_v(vel[i],acc[i],current,h))
        peaks[1] = max(peaks[1],np.linalg.norm(acc[i]),np.linalg.norm(acc[i]+current*h))
        peaks[2] = max(peaks[2],np.linalg.norm(current))
    return times,pos,vel,acc,jerk,peaks


def plan(reference, tolerance=1.25, limits=None, dt=.001, rest_time=.02,
         overlap_fraction=.5):
    """Return uniformly sampled commands plus continuous-interval certificates.

    Only one fixed overlap fraction (0.5) is the default, not a test-tuned
    hyperparameter. The final moving time is rounded upward to an integer dt,
    using additional common dilation; then at least 20 ms of final rest is
    included. 'jerk' is the right-sided analytic jerk (bounded jumps allowed).
    Invalid input raises ValueError; caller should retain that failed record.
    """
    began = perf_counter()
    if not np.isfinite([tolerance,dt,rest_time,overlap_fraction]).all() or tolerance<=0 or dt<=0 or rest_time<.02 or not 0<=overlap_fraction<=.5:
        raise ValueError("positive tolerance/dt, rest_time >= 0.02, overlap fraction in [0,0.5] required")
    vmax,amax,jmax = _limits(limits or dict(v=100.,a=2000.,j=62500.))
    p = _clean_reference(reference)
    vectors = np.diff(p,axis=0); lengths=np.linalg.norm(vectors,axis=1)
    directions = vectors/lengths[:,None]
    curves = [_scurve(float(L),vmax,amax,jmax) for L in lengths]
    starts = np.zeros(len(curves)); overlaps=[]; witnesses=[]; geometric_bounds=[]; vertex_bounds=[]
    for i in range(len(curves)-1):
        left,right=curves[i:i+2]
        # A corner witness has ||p-corner|| = b ||d_next-d_prev||.
        angle_factor = np.linalg.norm(directions[i+1]-directions[i])
        b_allowed = tolerance*.98/max(1.,angle_factor)
        high=overlap_fraction*min(left.duration,right.duration)
        # Scalar rest-to-rest S curves are symmetric: the permitted overlap
        # is the sum of the two first-passage times to b_allowed.
        overlap=min(high,left.time_at_position(min(b_allowed,left.length))+
                    right.time_at_position(min(b_allowed,right.length)))
        b,witness=_corner_bound(left,right,overlap)
        starts[i+1]=starts[i]+left.duration-overlap
        overlaps.append(overlap); witnesses.append(starts[i+1]+witness)
        geometric_bounds.append(b); vertex_bounds.append(b*angle_factor)
    times,pos,vel,acc,jerk,peaks=_build_events(curves,directions,starts)
    # The deviation between a C2 curve and its chord is <= A*dt^2/8.
    # Reserve one percent of tolerance for complete sampled-command chords.
    chord_scale=np.sqrt(peaks[1]*dt*dt/(8*.01*tolerance))
    minimum_scale=max(1.,peaks[0]/vmax,np.sqrt(peaks[1]/amax),np.cbrt(peaks[2]/jmax),chord_scale)*(1+1e-10)
    moving_steps=int(np.ceil(times[-1]*minimum_scale/dt))
    scale=moving_steps*dt/times[-1]
    end=moving_steps*dt
    # Position-derived acceleration/jerk require extra repeated samples after
    # arrival. Two extra ticks retain >=20 ms of rest under those checks.
    final_steps=int(np.ceil(rest_time/dt))+2
    ts=np.arange(moving_steps+final_steps+1)*dt
    unscaled=np.minimum(ts/scale,times[-1])
    idx=np.clip(np.searchsorted(times,unscaled,side="right")-1,0,len(jerk)-1)
    h=unscaled-times[idx]
    sampled_pos=pos[idx]+vel[idx]*h[:,None]+acc[idx]*h[:,None]**2/2+jerk[idx]*h[:,None]**3/6+p[0]
    sampled_vel=(vel[idx]+acc[idx]*h[:,None]+jerk[idx]*h[:,None]**2/2)/scale
    sampled_acc=(acc[idx]+jerk[idx]*h[:,None])/scale**2
    sampled_jerk=jerk[idx]/scale**3
    rested=ts>=end-1e-12
    endpoint_residual=float(np.linalg.norm(sampled_pos[moving_steps]-p[-1]))
    terminal_velocity_residual=float(np.linalg.norm(sampled_vel[moving_steps]))
    terminal_acceleration_residual=float(np.linalg.norm(sampled_acc[moving_steps]))
    if max(endpoint_residual,terminal_velocity_residual,terminal_acceleration_residual)>1e-5:
        raise ArithmeticError("event integration terminal residual exceeds 1e-5")
    sampled_pos[rested]=p[-1]; sampled_vel[rested]=0.;sampled_acc[rested]=0.;sampled_jerk[rested]=0.
    metrics=dict(method="independent_scurve_motion_overlap",is_published_method_reproduction=False,
        implementation_version="1.2-local-event-superposition",
        status="success",dt_s=dt,moving_time_s=end,completion_time_s=float(ts[-1]),
        terminal_rest_s=final_steps*dt,vector_speed_max=float(peaks[0]/scale),
        vector_acceleration_max=float(peaks[1]/scale**2),vector_jerk_max=float(peaks[2]/scale**3),
        commanded_path_deviation_bound_mm=max(geometric_bounds,default=0.),
        sampled_command_chord_deviation_bound_mm=max(geometric_bounds,default=0.)+peaks[1]/scale**2*dt*dt/8,
        sampled_command_chord_to_continuous_bound_mm=peaks[1]/scale**2*dt*dt/8,
        ordered_vertex_witness_bound_mm=max(vertex_bounds,default=0.),
        geometry_certificate="continuous bound from at most two overlapping monotone segment motions",
        kinematic_certificate="polynomial interval extrema, vector Euclidean norms",
        original_vertices=len(reference),planning_vertices=len(p),common_time_dilation=scale,
        maximum_concurrent_motions=2,overlap_fraction=overlap_fraction,
        endpoint_residual_before_roundoff_mm=endpoint_residual,
        terminal_velocity_residual_before_roundoff=terminal_velocity_residual,
        terminal_acceleration_residual_before_roundoff=terminal_acceleration_residual,
        planning_wall_time_s=perf_counter()-began)
    return dict(time=ts,positions=sampled_pos,velocities=sampled_vel,
        accelerations=sampled_acc,jerk=sampled_jerk,metrics=metrics,
        corner_witness_times=np.asarray(witnesses)*scale,
        event_times=times*scale,event_positions=pos+p[0],
        event_velocities=vel/scale,event_accelerations=acc/scale**2,event_jerk=jerk/scale**3)
