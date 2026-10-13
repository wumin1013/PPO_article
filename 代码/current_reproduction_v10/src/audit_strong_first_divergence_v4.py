"""Read-only reconstruction at first historical/v3 command disagreement."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
import math
import numpy as np
from common import ROOT, write_json
from runtime_bootstrap import setup
setup()
from execution_settings import configure_dt
configure_dt(.001)
import runtime_configuration as rc
import continuous_gate_guidance as guide
from compact_farvector_runtime import FarVectorRuntime


def ray_record(env, gate, vector):
    start=env.position
    h0=float((start-gate.point)@gate.normal)
    rate=float(vector@gate.normal)
    parameter=-h0/rate if abs(rate)>1e-15 else None
    crossing=start+parameter*vector if parameter is not None else None
    return dict(h0_mm=h0,normal_rate=rate,ray_parameter=parameter,
        crossing=None if crossing is None else crossing.tolist(),
        forward=bool(parameter is not None and parameter>0),
        within_target_segment=bool(parameter is not None and 0<=parameter<=1),
        channel_ok=bool(crossing is not None and env.corridor.gate_contains(gate,crossing)))


def first(array, eps):
    inds=np.flatnonzero(np.max(abs(array),axis=1)>eps)
    return int(inds[0]) if len(inds) else None


report=[]
for case in ('dense','original'):
    oldfile=ROOT/f'baseline_records/supplementary_reproduction/records/J-NNC/{case}_1ms.npz'
    newfile=ROOT/f'results/strong_checkpoint_v4/v3/{case}.npz'
    with np.load(oldfile) as f:
        old={k:f[k] for k in f.files}
    with np.load(newfile) as f:
        new={k:f[k] for k in f.files}
    n=min(len(old['rows']),len(new['rows']))
    delta=new['rows'][:n]-old['rows'][:n]
    index=first(delta[:,0:2],1e-9)
    tolerance=1.25 if case=='dense' else 1.2480086195146001
    rc.configure_runtime('v3',tolerance=tolerance)
    wrapper=FarVectorRuntime(old['reference'],potential_speed=100.)
    env=wrapper.env
    for row in old['rows'][:index]:
        env.tracker.update(row[:2])
    prev=old['rows'][index-1]
    env.position=prev[:2].copy();env.velocity=prev[2:4].copy();env.acceleration=prev[4:6].copy()
    env.heading=math.atan2(prev[9],prev[8]);env.ticks=index
    env.held=old['actions'][index//20].copy()
    gate=env.corridor.gates[env.tracker.frontier]
    guide.active_gate_transitions=rc._ORIGINAL
    historical_target=rc._ORIGINAL_TARGET(env)
    historical_active=rc._ORIGINAL(env,guide.GATE_TRANSITION_WIDTH_MM)
    guide.active_gate_transitions=rc._local_transitions
    local_target=rc._ORIGINAL_TARGET(env)
    local_active=rc._local_transitions(env,guide.GATE_TRANSITION_WIDTH_MM)
    ordered_target=rc._ordered_target(env)
    alpha=env.held[0]
    rotation=np.array([[math.cos(alpha),-math.sin(alpha)],[math.sin(alpha),math.cos(alpha)]])
    direction=historical_target-env.position
    rotated=rotation@direction
    beta=math.atan2(direction[1],direction[0])
    wrap=lambda x:(x+math.pi)%(2*math.pi)-math.pi
    filtered_heading=wrap(env.heading+(1-math.exp(-.001/.012))*wrap(beta+alpha-env.heading))
    filtered_direction=np.array([math.cos(filtered_heading),math.sin(filtered_heading)])
    record=dict(case=case,first_position_disagreement_index_0based=index,
        first_position_disagreement_time_s=(index+1)*.001,
        first_desired_disagreement_index_0based=first(delta[:,8:10],1e-9),
        first_any_disagreement_index_0based=first(delta,1e-9),
        position=env.position.tolist(),velocity=env.velocity.tolist(),
        heading_rad=env.heading,held_physical_action=env.held.tolist(),
        frontier=env.tracker.frontier,gate_vertex=gate.vertex_index,
        gate_point=gate.point.tolist(),gate_normal=gate.normal.tolist(),
        source_progress_mm=env.progress()[0],gate_arc_mm=float(env.corridor.arcs[gate.vertex_index]),
        historical_target=historical_target.tolist(),local_transitions_target=local_target.tolist(),
        ordered_target=ordered_target.tolist(),
        historical_active=[dict(vertex=g.vertex_index,distance=float(d)) for g,d in historical_active],
        local_active=[dict(vertex=g.vertex_index,distance=float(d)) for g,d in local_active],
        local_target_delta_mm=float(np.linalg.norm(local_target-historical_target)),
        ordered_target_delta_mm=float(np.linalg.norm(ordered_target-local_target)),
        bare_los=ray_record(env,gate,direction),
        steering_rotated_los=ray_record(env,gate,rotated),
        filtered_heading_ray=ray_record(env,gate,filtered_direction),
        reconstructed_historical_desired= (env.held[1]*filtered_direction).tolist(),
        saved_historical_desired=old['rows'][index,8:10].tolist(),
        saved_v3_desired=new['rows'][index,8:10].tolist(),
        historical_vs_v3_preceding_row_max_abs=float(abs(delta[:index]).max()),
        context='No physical replay or source modification; tracker reconstructed by feeding saved pre-disagreement command positions, heading recovered from desired vector.')
    report.append(record)
    print(case,record,flush=True)
write_json(ROOT/'audit/strong_first_divergence_v4.json',report)
