"""Sencer-style fixed geometry with upstream Ruckig straight-block scheduling.

Ruckig 0.12.2 is used unchanged for its native one-dimensional state-to-state
problem. This is a CNC adapter, not the native Li-2023 or Xu-2026 algorithm and
not a whole-path time-optimality claim. Cartesian curve caps and the full-path
acceptance contract are shared with the retained S2 comparator.
"""
from pathlib import Path
import math
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'literature/ruckig_vendor'))
from baselines.legacy_s2 import prepare_geometry, sha


def transition_duration(v0,v1,acceleration,jerk):
    difference=abs(v1-v0)
    if difference <= acceleration*acceleration/jerk:
        return 2.*math.sqrt(difference/jerk)
    return difference/acceleration+acceleration/jerk


def straight(length,v0,v1,limits):
    from ruckig import Ruckig, InputParameter, Trajectory, Result
    inp=InputParameter(1)
    inp.current_position=[0.];inp.current_velocity=[float(v0)];inp.current_acceleration=[0.]
    inp.target_position=[float(length)];inp.target_velocity=[float(v1)];inp.target_acceleration=[0.]
    inp.max_velocity=[float(limits[0])];inp.min_velocity=[0.]
    inp.max_acceleration=[float(limits[1])];inp.max_jerk=[float(limits[2])]
    trajectory=Trajectory(1)
    result=Ruckig(1).calculate(inp,trajectory)
    if result not in (Result.Working,Result.Finished):
        raise RuntimeError('Upstream Ruckig failed: '+str(result))
    return trajectory


def plan(reference,*,dt=.001,tolerance=1.25,limits=(100.,2000.,62500.),corner_scale=1.,source_geometry=None):
    started=time.perf_counter()
    curve,g,s,c,source,extrapolated=prepare_geometry(reference,dt=dt,tolerance=tolerance,limits=limits,corner_scale=corner_scale,source_geometry=source_geometry)
    effective=(limits[0],.999*limits[1],.999*limits[2])
    # The same forward/backward compatibility solver uses exact third-order
    # zero-endpoint-acceleration reachability instead of quintic reachability.
    s.transition_duration=transition_duration
    compatible=s.schedule(curve.curve_lengths_mm,curve.curve_speed_caps_mm_s,curve.straight_lengths_mm,
                          F=effective[0],A=effective[1],J=effective[2])
    speeds=[0.,*compatible.curve_speeds_mm_s,0.]
    phases=[];t0=s0=0.
    for i,length in enumerate(curve.straight_lengths_mm):
        if length>1e-12:
            tr=straight(length,speeds[i],speeds[i+1],effective)
            phases.append((t0,s0,tr.duration,'ruckig_straight',tr))
            t0+=tr.duration;s0+=length
        if i<len(curve.curve_lengths_mm):
            length=curve.curve_lengths_mm[i];speed=speeds[i+1]
            duration=length/speed
            phases.append((t0,s0,duration,'curve_cruise',speed))
            t0+=duration;s0+=length
    if abs(s0-curve.length)>1e-7:raise ValueError('Arc-length assembly mismatch')
    if not math.isfinite(t0) or t0+.025>1200.:
        raise ValueError('The finite complete-command time budget is exceeded')
    times=np.arange(math.ceil(t0/dt)+math.ceil(.025/dt)+1)*dt
    sigma=np.full(len(times),curve.length,dtype=float)
    scalar=np.zeros((len(times),3));summaries=[]
    for start,offset,duration,kind,payload in phases:
        begin=np.searchsorted(times,start,side='left');end=np.searchsorted(times,start+duration,side='left')
        local=times[begin:end]-start
        if kind=='curve_cruise':
            sigma[begin:end]=offset+payload*local;scalar[begin:end,0]=payload
        else:
            states=[payload.at_time(float(t)) for t in local]
            # A genuine phase may lie wholly between two execution ticks.
            # Empty phase sampling must retain the (0,3) state-array shape.
            values=np.asarray([[x[0][0],x[1][0],x[2][0]] for x in states]).reshape(-1,3)
            sigma[begin:end]=offset+values[:,0];scalar[begin:end,:2]=values[:,1:]
        summaries.append(dict(kind=kind,start_s=start,duration_s=duration,start_length_mm=offset))
    if np.any(np.diff(sigma)<-1e-8):raise ValueError('Ruckig adapter reversed arc-length progress')
    positions=curve.position(sigma)
    meta=dict(method='S2-R: Sencer-style geometry with Ruckig straight scheduling',version='1.1-empty-phase-safe',
        upstream_version='0.12.2',upstream_doi='10.15607/RSS.2021.XVII.015',
        native_scope='One-dimensional straight-block state-to-state third-order timing',
        adaptation='Curve-constant vector-safe feed; exact zero-acceleration boundary states; forward/backward compatibility',
        optimality_scope='Upstream Ruckig straight subproblems only; no whole-path optimum claim',
        limits=list(limits),internal_limits=list(effective),tolerance_mm=tolerance,corner_scale=corner_scale,
        corner_error_parameter_mm=g.CORNER_ERROR_MM,small_angle_extrapolated_degrees=extrapolated,
        source_sha256={name:sha(source/name) for name in ['geometry.py','scheduler.py']},
        adapter_sha256=sha(__file__),geometry=curve.metadata,
        execution=dict(continuous_motion_duration_s=t0,complete_command_time_s=float(times[-1]),
                       curve_cruise_speeds_mm_s=list(compatible.curve_speeds_mm_s),scheduled_phases=summaries),
        planning_wall_s=time.perf_counter()-started)
    return positions,meta
