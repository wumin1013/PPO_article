"""Finite feed-only reachability probes in omitted curved-route cruise support."""
import time
import numpy as np
import torch
from canonical_gate_state import CanonicalGatePolicy,checkpoint_environment_factory
from probe_coordinated_suffix import reconstruct,encode
from complete_corner_metrics import complete_corner_metrics
from complete_motion_metrics import complete_motion_metrics
from audit_function_constraint_interference import command_segment_margin
from validate_actions import difference_audit
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_segment_cruise_support_20260918'
    support=read(source/'results.json');model=BASE/support['parent_model']
    parent=read(model.parent/'results.json')['full_candidate']
    windows=support['selected_probe_windows'];assert len(windows)==4
    stage=register('canonical_gate_band_segment_cruise_probe_20260918',
        [__file__,model,source/'results.json',source/'receipt.json',BASE/'probe_coordinated_suffix.py',
         BASE/'joint_runtime.py',BASE/'complete_motion_metrics.py',BASE/'complete_corner_metrics.py',
         BASE/'audit_function_constraint_interference.py',BASE/'validate_actions.py'],
        dict(paths=[r['path'] for r in windows],new_suffixes=4,max_feed_intervention_decisions=10,
             feed_request_mm_s=99.5,actor_calls=1500,new_physics_ticks=40000,
             saved_prefix_geometry_ticks=8000,wall_s=90,training_updates=0),
        'Test the four conservatively selected saved windows from the support audit, '
        'two isolated and two multisegment training paths. At most10 consecutive20ms '
        'decisions change ONLY feed request to99.5mm/s. Steering and preview remain '
        'the unmodified policy outputs at each actual state, and all subsequent '
        'decisions resume the original policy through coverage, braking and rest. '
        'Do not clamp trajectories or alter V/A/J. Reuse the exact prefix and nominal '
        'full-path result. Report independent full-command dynamics, continuous '
        'Euclidean corridor, complete time and cruise, and isolated-corner shape. '
        'Complete feasible cruise gain establishes local physical reachability only; '
        'shape regressions indicate required joint learning rather than a deployable '
        'speed rule. No intervention is a learned candidate or final acceptance.')
    start=time.perf_counter();costs=dict(actor_calls=0,new_physics_ticks=0,saved_prefix_geometry_ticks=0)
    md=torch.load(model,map_location='cpu',weights_only=False);policy=CanonicalGatePolicy(256)
    policy.load_state_dict(md['policy']);factory=checkpoint_environment_factory(md['architecture'])
    records=[]
    for w in windows:
        path=w['path'];anchor=w['window_start'];frame=20*anchor
        with np.load(BASE/w['trace']) as z:ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
        baseline=next(r for r in parent['results'] if r['original_path']==path)
        assert baseline['success']
        env=reconstruct(ref,rows,actions,frame,costs,env_factory=factory)
        chunks=[];held=[];requests=[]
        while not env.done:
            assert time.perf_counter()-start<85 and costs['actor_calls']<1500 and costs['new_physics_ticks']<37880
            latent=policy.act(env.observation(),True)[0];costs['actor_calls']+=1
            if not chunks:assert np.array_equal(env.decode_action(latent),actions[anchor])
            if len(chunks)<w['probe_decisions']:
                original=env.decode_action(latent);altered=original.copy();altered[1]=99.5
                latent=encode(altered);decoded=env.decode_action(latent)
                assert np.max(abs(decoded-altered))<1e-12
                requests.append(dict(decision=anchor+len(chunks),policy_action=original.tolist(),
                                     diagnostic_action=decoded.tolist()))
            _,_,_,info=env.step(latent);chunks.append(info['rows']);held.append(info['held_action'])
            costs['new_physics_ticks']+=len(info['rows'])
        full=np.vstack([rows[:frame],*chunks]);all_actions=np.vstack([actions[:anchor],held])
        audit=difference_audit(ref[0],full);corridor=command_segment_margin(ref,full[:,:2])
        success=bool(env.success and audit['kinematic_pass'] and audit['rest_pass'] and corridor['max_distance_upper_mm']<=1.25+1e-8)
        nominal_motion=complete_motion_metrics(ref,rows);motion=complete_motion_metrics(ref,full)
        name=f'path{path}.npz';np.savez_compressed(stage/name,reference=ref,rows=full,actions=all_actions,anchor=anchor)
        record=dict(path=path,success=success,reason=env.reason,trace=name,requests=requests,
            audit=audit,continuous_corridor=corridor,geometry=env.tracker.snapshot(1.25),
            baseline_motion=nominal_motion,motion=motion,
            complete_time_gain_s=baseline['completion_time_s']-len(full)*.001 if success else None,
            complete_cruise99_gain_s=motion['cruise_ge99_s']-nominal_motion['cruise_ge99_s'] if success else None)
        if len(ref)==3:
            record['baseline_shape']=complete_corner_metrics(ref,rows)
            record['shape']=complete_corner_metrics(ref,full)
            names=('in_edge_outward_mm','out_edge_outward_mm','full_physical_reverse_deg','heading_overshoot_deg')
            record['isolated_shape_changes']={n:record['shape'][n]-record['baseline_shape'][n] for n in names}
        records.append(record);write_json(stage/'results.json',dict(parent_sha256=sha(model),records=records,
            costs=costs,diagnostic_only=True,goal_complete=False))
        print('SEGMENT_CRUISE_PROBE',path,success,record['complete_time_gain_s'],record['complete_cruise99_gain_s'],
              record.get('isolated_shape_changes'),flush=True)
    receipt(stage,dict(status='completed_segment_cruise_request_probe',costs=costs,
        successes=sum(r['success'] for r in records),wall_s=time.perf_counter()-start,
        diagnostic_only=True,goal_complete=False))


if __name__=='__main__':main()
