"""Pure assessment of the existing bounded vector cubic stopping family."""
import copy
import time
import numpy as np
from predictive_action_candidate_v5 import snapshot


def stopping_extension(env):
    """Audit a speculative full stop; never alter or execute the real state."""
    import joint_runtime as joint
    from geometry_spec import GateTracker
    started,cpu=time.perf_counter(),time.process_time()
    state=snapshot(env)
    p0=env.position.copy();v0=env.velocity.copy();a0=env.acceleration.copy()
    rows=np.empty((0,13));positions=np.asarray([p0]);error=None
    info={};first_bad=None
    witness=GateTracker(env.corridor,env.corridor.reference[0])
    for key,value in state['tracker'].items():setattr(witness,key,copy.deepcopy(value))
    try:
        plan=joint.terminal.plan_planar_terminal_brake(p0,v0,a0,dt=joint.DT,limits=joint.terminal.LIMITS)
        cols=plan['columns'];vv=np.c_[cols['velocity_x_mm_s'],cols['velocity_y_mm_s']]
        current=p0.copy();pp=[p0.copy()];rr=[];old_v=v0.copy();old_a=a0.copy()
        for index,velocity in enumerate(vv[1:],1):
            acceleration=(velocity-old_v)/joint.DT;jerk=(acceleration-old_a)/joint.DT
            current=current+joint.DT*velocity
            witness.update(current)
            if first_bad is None and not witness.inside:first_bad=index
            pp.append(current.copy())
            rr.append(np.r_[current,velocity,acceleration,jerk,velocity,0.,0.,1.])
            old_v=velocity;old_a=acceleration
        positions=np.asarray(pp);rows=np.asarray(rr)
        info.update(braking_time_s=plan['braking_time_s'],total_tail_time_s=plan['total_tail_time_s'])
    except Exception as exc:
        error=type(exc).__name__+': '+str(exc)
    v=np.diff(positions,axis=0)/joint.DT
    a=np.diff(np.vstack([v0,v]),axis=0)/joint.DT
    j=np.diff(np.vstack([a0,a]),axis=0)/joint.DT
    peaks=[float(np.linalg.norm(x,axis=1).max(initial=0.)) for x in (v,a,j)]
    vector=bool(np.all(np.asarray(peaks)/[100.,2000.,62500.]<=1.00002))
    rest=(np.linalg.norm(v,axis=1)<=.001)&(np.linalg.norm(a,axis=1)<=.01)
    nonrest=np.flatnonzero(~rest)
    rest_count=len(rest)-(int(nonrest[-1])+1 if len(nonrest) else 0)
    if not len(nonrest):rest_count+=state['core']['rest_ticks']
    next_gate=(env.corridor.gates[witness.frontier] if witness.frontier<len(env.corridor.gates) else None)
    signed=None if next_gate is None else float((positions[-1]-next_gate.point)@next_gate.normal)
    # A stopped state beyond the next unvisited gate is not a certified
    # forward continuation. No gate location/channel or task rule is changed.
    forward=bool(signed is None or signed<=1e-9)
    geometry=bool(witness.inside and forward)
    feasible=bool(error is None and len(rows)>0 and geometry and vector and rest_count*joint.DT>=.02-1e-10)
    info.update(feasible=feasible,error=error,every_interval_inside=bool(witness.inside),
        next_unvisited_gate_not_behind_stop=forward,next_unvisited_gate_signed_mm=signed,
        initial_frontier=state['tracker']['frontier'],final_frontier=witness.frontier,
        first_outside_tick=first_bad,first_outside_time_s=None if first_bad is None else first_bad*joint.DT,
        vector_pass=vector,vector_peaks=peaks,terminal_rest_s=rest_count*joint.DT,
        initial_position_mm=p0.tolist(),initial_velocity_mm_s=v0.tolist(),initial_acceleration_mm_s2=a0.tolist(),
        stopping_length_mm=float(np.linalg.norm(np.diff(positions,axis=0),axis=1).sum()),
        stop_position_mm=positions[-1].tolist(),
        assessment_wall_s=time.perf_counter()-started,assessment_cpu_s=time.process_time()-cpu,
        family='Unmodified common first-feasible nonreversing bounded vector cubic, 98% jerk reserve, 23ms discrete rest tail',
        interpretation='Feasibility of this one stopping family only; failure is not proof that all braking controls are impossible; this stop is not executed')
    return rows,positions,info
