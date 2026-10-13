"""Equal initial target-jump interventions through preview or steering only."""
import math
import time
import argparse
import numpy as np
import torch
from scipy.optimize import brentq
from canonical_gate_state import CanonicalGatePolicy, checkpoint_environment_factory
from probe_coordinated_suffix import reconstruct, encode
from audit_current_command_coordinates import command_signals
from complete_corner_metrics import complete_corner_metrics
from complete_motion_metrics import complete_motion_metrics
from audit_function_constraint_interference import command_segment_margin
from validate_actions import difference_audit
from joint_runtime import wrap
from search_complete_return import read
from study_io import BASE, register, receipt, write_json, sha


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--temporal',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1)
    parent=BASE/'canonical_gate_band_per_corner_shape_priority_learning_20260918'
    cause=BASE/'canonical_gate_band_peak_preview_coupling_20260918'
    data=read(parent/'results.json');model=parent/data['chosen']['model'];pool=data['full_candidate']
    events=read(cause/'results.json')['records'];ids=[6,7,12,13,14,15]
    traces={r['original_path']:BASE/t for r,t in zip(pool['results'],pool['traces'])}
    late=BASE/'canonical_gate_band_matched_target_jump_probe_20260918'
    extra_sources=([late/'results.json',late/'receipt.json'] if args.temporal else [])
    stage=register('canonical_gate_band_target_jump_temporal_probe_20260918' if args.temporal else
        'canonical_gate_band_matched_target_jump_probe_20260918',
        [__file__,model,parent/'results.json',parent/'receipt.json',cause/'results.json',cause/'receipt.json',
         BASE/'probe_coordinated_suffix.py',BASE/'complete_corner_metrics.py',BASE/'validate_actions.py',
         BASE/'audit_function_constraint_interference.py']+[traces[p] for p in ids]+extra_sources,
        dict(paths=ids,arms=['early_only','early_and_late'] if args.temporal else ['preview_only','steering_only'],new_complete_suffixes=12,
             changed_decisions_per_suffix=2 if args.temporal else 1,target_jump_fraction=.5,wall_s=120,
             actor_calls=1800,new_physics_ticks=40000,saved_prefix_geometry_ticks=5000),
        ('Temporal2x2 intervention: reuse the nominal and completed late-steering-only '
         'arms exactly. Add half the saved signed target jump to steering ONE '
         'decision earlier, either alone or together with the same negative '
         'steering offset at the originally identified jump decision. Apply '
         'offsets to the original policy outputs at each ACTUAL changed state, '
         'so the combined arm includes real state/action interaction. Preview '
         'and feed remain the policy outputs. Changes are diagnostic, offline '
         'path/event selections, never a runtime rule or policy teacher. This '
         'distinguishes redistribution of turning over time from suppressing '
         'a late target jump. All12 previous suffixes remain immutable; only '
         'the12 missing early/combined cells run. Report interaction as '
         'combined-minus-early-minus-late-plus-baseline on full-path metrics. '
         'No candidate adoption, training success, or physical-limit changes. '
         if args.temporal else
        'Saved fixed-state decomposition attributes large sharp-turn target jumps '
        'mostly to preview changes that steering does not cancel. Test the next '
        'causal link with two matched single-decision interventions on six existing '
        'left/right80,140,155degree training paths. At the registered largest '
        'jump, halve the SIGNED combined target jump at the exact same state, '
        'using either only preview (a scalar root within the old/new preview '
        'interval) or only steering; feed remains identical. Verify both produce '
        'the same initial target. Hold this action for the ordinary20ms decision '
        'then restore full unmodified learned feedback through original braking '
        'and rest. The two arms expose different within-decision geometry while '
        'matching their initial target, so differences cannot be attributed to '
        'initial heading size. Keep original executor and constraints. Reuse '
        'baseline and prefix; record all failures. No prefix time benefit, '
        'deployment, teacher labels or learned-success claim. Adoption is '
        'forbidden for these interventions; any promising mechanism must later '
        'be learned and pass unchanged complete shape, cruise and fairness checks.'))
    start=time.perf_counter();deadline=start+115
    costs=dict(actor_calls=0,new_physics_ticks=0,saved_prefix_geometry_ticks=0)
    md=torch.load(model,map_location='cpu',weights_only=False)
    policy=CanonicalGatePolicy(256);policy.load_state_dict(md['policy'])
    factory=checkpoint_environment_factory(md['architecture']);records=[]
    late_data=read(late/'results.json') if args.temporal else None
    if late_data:assert late_data['parent_sha256']==sha(model)
    try:
        for path in ids:
            event=next(r for r in events if r['path']==path);jump_decision=event['decision']
            k=jump_decision-int(args.temporal);tick=20*k
            with np.load(traces[path]) as z:
                ref,rows,actions=[z[name].copy() for name in ('reference','rows','actions')]
            env=reconstruct(ref,rows,actions,tick,costs,env_factory=factory);state=env.state()
            u,v=np.diff(ref,axis=0);sign=np.sign(u[0]*v[1]-u[1]*v[0]);origin=math.atan2(u[1],u[0])
            old,new=actions[k-1],actions[k];signal=command_signals(ref,rows,actions)
            nominal_target=signal['combined_deg'][tick]
            half_jump=.5*event['total_target_jump_deg']
            desired_target=nominal_target+(1. if args.temporal else -1.)*half_jump
            def target(action):
                env.held=action.copy();delta=env.guidance_target()-env.position
                return sign*wrap(math.atan2(delta[1],delta[0])+action[0]-origin)*180/math.pi
            assert abs(target(new)-nominal_target)<1e-8
            def preview_residual(d):
                action=new.copy();action[2]=d
                return target(action)-desired_target
            if args.temporal:
                interventions={name:new.copy() for name in ('early_only','early_and_late')}
                for action in interventions.values():action[0]+=sign*math.radians(half_jump)
            else:
                root=brentq(preview_residual,old[2],new[2],xtol=1e-12)
                interventions={name:new.copy() for name in ('preview_only','steering_only')}
                interventions['preview_only'][2]=root
                interventions['steering_only'][0]-=sign*math.radians(half_jump)
            baseline_shape=complete_corner_metrics(ref,rows)
            for arm,action in interventions.items():
                env.restore(state);initial_target=target(action)
                assert abs(initial_target-desired_target)<1e-8
                latent=encode(action);assert np.max(abs(env.decode_action(latent)-action))<1e-12
                chunks=[];held=[]
                while not env.done:
                    assert time.perf_counter()<deadline and costs['actor_calls']<1800 and costs['new_physics_ticks']<37800
                    if chunks:
                        latent=policy.act(env.observation(),True)[0];costs['actor_calls']+=1
                        if args.temporal and arm=='early_and_late' and len(chunks)==1:
                            follow=env.decode_action(latent)
                            follow[0]-=sign*math.radians(half_jump)
                            latent=encode(follow)
                    _,_,_,info=env.step(latent);chunks.append(info['rows']);held.append(info['held_action'])
                    costs['new_physics_ticks']+=len(info['rows'])
                full=np.vstack([rows[:tick],*chunks]);all_actions=np.vstack([actions[:k],held])
                audit=difference_audit(ref[0],full)
                corridor=command_segment_margin(ref,full[:,:2],tolerance=1e-10)
                success=bool(env.success and audit['kinematic_pass'] and audit['rest_pass'] and
                    corridor['max_distance_upper_mm']<=1.25+1e-8)
                name=f'path{path}_{arm}.npz'
                np.savez_compressed(stage/name,reference=ref,rows=full,actions=all_actions)
                shape=complete_corner_metrics(ref,full) if success else None
                motion=complete_motion_metrics(ref,full) if success else None
                r=dict(path=path,arm=arm,decision=k,nominal_action=new.tolist(),intervention_action=action.tolist(),
                    initial_combined_target_deg=initial_target,nominal_combined_target_deg=nominal_target,
                    success=success,reason=env.reason,trace=name,audit=audit,corridor=corridor,
                    geometry=env.tracker.snapshot(1.25),baseline_shape=baseline_shape,shape=shape,motion=motion,
                    complete_time_change_s=(len(full)-len(rows))*.001 if success else None,
                    shape_changes={key:shape[key]-baseline_shape[key] for key in
                        ('in_edge_outward_mm','out_edge_outward_mm','full_physical_reverse_deg',
                         'heading_overshoot_deg','full_normalized_energy','full_spatial_peak_curvature_mm_inv',
                         'full_middle80_turn_arc_mm','apex_speed_mm_s')} if success else None)
                records.append(r)
                if args.temporal:
                    control=next(v for v in late_data['records'] if v['path']==path and v['arm']=='steering_only')
                    assert control['decision']==jump_decision and control['success']
                    r['reused_late_record']=dict(stage=late.name,path=path,arm='steering_only',trace=control['trace'])
                    if arm=='early_and_late' and success:
                        early=next(v for v in records if v['path']==path and v['arm']=='early_only')
                        if early['success']:
                            r['factorial_shape_interaction']={key:r['shape_changes'][key]-early['shape_changes'][key]-control['shape_changes'][key] for key in r['shape_changes']}
                write_json(stage/'results.json',dict(parent_sha256=sha(model),records=records,costs=costs,
                    diagnostic_only=True,learned_candidate_adopted=False,goal_complete=False))
                print('MATCHED_TARGET_JUMP',path,arm,success,r['complete_time_change_s'],r['shape_changes'],flush=True)
        assert costs['saved_prefix_geometry_ticks']<=5000
        receipt(stage,dict(status='completed_matched_target_jump_physical_probe',costs=costs,
            successes=sum(r['success'] for r in records),wall_s=time.perf_counter()-start,
            diagnostic_only=True,learned_candidate_adopted=False,goal_complete=False))
    except Exception as error:
        receipt(stage,dict(status='failed_matched_target_jump_physical_probe',error=str(error),
            costs=costs,completed_suffixes=len(records),wall_s=time.perf_counter()-start,
            diagnostic_only=True,learned_candidate_adopted=False,goal_complete=False))
        raise


if __name__=='__main__':main()
