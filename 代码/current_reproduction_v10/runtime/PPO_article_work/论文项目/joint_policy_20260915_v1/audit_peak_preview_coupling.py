"""Exact saved-state factorization of large decision-boundary target jumps."""
import math
import time
import numpy as np
import torch
from canonical_gate_state import checkpoint_environment_factory
from probe_coordinated_suffix import reconstruct
from audit_current_command_coordinates import command_signals
from joint_runtime import wrap
from search_complete_return import read
from study_io import BASE, register, receipt, write_json, sha


def main():
    parent=BASE/'canonical_gate_band_per_corner_shape_priority_learning_20260918'
    timing=BASE/'canonical_gate_band_curvature_gate_timing_20260918'
    data=read(parent/'results.json');model=parent/data['chosen']['model']
    events=read(timing/'results.json')['records'];pool=data['full_candidate']
    traces={r['original_path']:BASE/t for r,t in zip(pool['results'],pool['traces'])}
    stage=register('canonical_gate_band_peak_preview_coupling_20260918',
        [__file__,model,parent/'results.json',parent/'receipt.json',timing/'results.json',
         timing/'receipt.json',BASE/'probe_coordinated_suffix.py',BASE/'continuous_gate_guidance.py']+
        [traces[p] for p in range(16)],
        dict(new_physics_ticks=0,new_actor_decisions=0,saved_prefix_geometry_ticks=20000,
             fixed_state_guide_queries=32,wall_s=60),
        'All16 largest ordinary LOS jumps occur at20ms action boundaries, '
        'and12 curvature peaks precede the0.1mm gate band. At each saved '
        'largest jump, reconstruct the SAME exact command-start state and '
        'query the unchanged guide at old versus new preview. Decompose the '
        'recorded signed LOS jump into one-tick geometry/position advancement '
        'and current preview-action change, then add the independently known '
        'steering-residual change to reconstruct the full target jump. Record '
        'feed changes too. This is an exact conditional action-to-target '
        'causal calculation on the saved state; it does NOT prove subsequent '
        'curvature/speed causation or identify why training learned the change. '
        'Only if it exposes a substantial uncancelled target jump may a '
        'small whole-path physical intervention test that downstream link.')
    start=time.perf_counter();costs=dict(saved_prefix_geometry_ticks=0)
    md=torch.load(model,map_location='cpu',weights_only=False)
    factory=checkpoint_environment_factory(md['architecture']);records=[]
    for event in events:
        path=event['path'];tick=event['los_jump_tick'];assert tick%20==0 and tick>0
        k=tick//20
        with np.load(traces[path]) as z:
            ref,rows,actions=[z[name].copy() for name in ('reference','rows','actions')]
        env=reconstruct(ref,rows,actions,tick,costs,env_factory=factory)
        u,v=np.diff(ref,axis=0);sign=np.sign(u[0]*v[1]-u[1]*v[0]);origin=math.atan2(u[1],u[0])
        old,new=actions[k-1],actions[k]
        def query(action):
            env.held=action.copy();delta=env.guidance_target()-env.position
            return sign*wrap(math.atan2(delta[1],delta[0])-origin)*180/math.pi
        old_at_new,new_at_new=query(old),query(new)
        signal=command_signals(ref,rows,actions)
        preview_change=new_at_new-old_at_new
        geometry_change=old_at_new-signal['los_deg'][tick-1]
        residual_change=float(sign*(new[0]-old[0])*180/math.pi)
        target_jump=float(signal['combined_deg'][tick]-signal['combined_deg'][tick-1])
        error=abs(preview_change+geometry_change+residual_change-target_jump)
        assert abs(new_at_new-signal['los_deg'][tick])<1e-8 and error<1e-8
        records.append(dict(path=path,decision=k,old_preview_mm=float(old[2]),new_preview_mm=float(new[2]),
            preview_jump_mm=float(new[2]-old[2]),preview_target_contribution_deg=float(preview_change),
            geometry_target_contribution_deg=float(geometry_change),
            steering_target_contribution_deg=residual_change,total_target_jump_deg=target_jump,
            decomposition_error_deg=float(error),old_feed_mm_s=float(old[1]),new_feed_mm_s=float(new[1]),
            old_residual_deg=float(sign*old[0]*180/math.pi),new_residual_deg=float(sign*new[0]*180/math.pi),
            curvature_peak_delay_ms=int(event['peak_tick']-tick),
            peak_speed_mm_s=event['peak_speed_mm_s'],
            physical_peak_curvature_mm_inv=event['physical_peak_curvature_mm_inv']))
        assert time.perf_counter()-start<60 and costs['saved_prefix_geometry_ticks']<=20000
    write_json(stage/'results.json',dict(parent_sha256=sha(model),records=records,costs=costs,
        target_jump_decomposition_verified=True,downstream_physical_causality_proven=False,
        historical_learning_origin_proven=False,goal_complete=False))
    receipt(stage,dict(status='completed_exact_saved_preview_target_factorization',
        costs=costs,new_physics_ticks=0,new_actor_decisions=0,wall_s=time.perf_counter()-start,goal_complete=False))
    for r in records:print('PREVIEW_JUMP_FACTOR',r,flush=True)


if __name__=='__main__':main()
