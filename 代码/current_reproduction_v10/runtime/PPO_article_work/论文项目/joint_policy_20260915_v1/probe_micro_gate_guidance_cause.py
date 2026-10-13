"""One-tick guide-phase interventions with original coverage auditing intact."""
import time
import numpy as np
import torch
from geometry_spec import audit_commands
from continuous_sensor_runtime import environment_factory
from kernel_local_feedback_policy import KernelLocalFeedbackPolicy
from probe_coordinated_suffix import reconstruct
from probe_one_decision_credit import score_complete
from audit_xy_curvature_resolution import curvature_metrics
from validate_actions import difference_audit
from search_complete_return import read
from study_io import BASE,register,receipt,write_json


def main():
    torch.set_num_threads(1)
    parent=BASE/'continuous_sensor_matched_function_transfer_20260917'
    selected=next(r for r in read(parent/'results.json')['candidates'] if r['arm']=='updated')
    composition=BASE/'continuous_sensor_complete_composition_20260917'
    plus=read(composition/'results.json')['candidates'][1]
    j=next(i for i,r in enumerate(selected['results']) if r['original_path']==15)
    cases=[('advance_updated',composition/'step+0.125.pt',BASE/plus['traces'][j],1),
           ('delay_parent',parent/'updated.pt',BASE/selected['traces'][j],0)]
    stage=register('one_ms_gate_guidance_causal_probe_20260917',
        [__file__,BASE/'continuous_sensor_runtime.py',BASE/'joint_runtime.py',
         BASE/'geometry_spec.py',BASE/'probe_coordinated_suffix.py',
         BASE/'continuous_sensor_composition_event_audit_20260917/results.json']+
        [p for _,m,t,_ in cases for p in (m,t)],
        dict(path=15,anchor_decision=36,altered_guide_tick=725,suffixes=2,
            actor_calls=200,new_physics_ticks=5000,saved_geometry_ticks=1500,wall_s=60),
        'Saved1ms gate reconstruction shows the parent crossing within command724 '
        '(fraction.9872), but the positive continuous-sensor update crosses in725 '
        '(fraction.0021). At physical state tick725, force ONLY the guide progress '
        'query to the opposite adjacent reference segment for this single1ms '
        'execution: advance the positive candidate, and delay the parent. Keep '
        'the actual GateTracker, corridor, all V/A/J constraints, decision actions, '
        'terminal logic and coverage measurements intact. The probe is between '
        'decision boundaries, so actor observations/actions are unchanged until '
        'normal feedback responds. Resume through full stop. These intentionally '
        'counterfactual guide queries diagnose micro-event causality; no schedule '
        'or rule from the probe is deployed. Compare bidirectional shape effects '
        'and independently reconstruct ordered coverage from resulting XY.')
    start=time.perf_counter();costs=dict(actor_calls=0,new_physics_ticks=0,saved_prefix_geometry_ticks=0)
    records=[]
    for label,model,trace,forced_segment in cases:
        with np.load(trace) as z:ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
        _,events=audit_commands(ref,np.vstack([ref[0],rows[:,:2]]))
        prior=reconstruct(ref,rows,actions,720,costs);env=environment_factory(ref);env.restore(prior.state())
        p=KernelLocalFeedbackPolicy(256);p.load_state_dict(torch.load(model,map_location='cpu',weights_only=False)['policy'])
        original_progress=env.progress;interventions=[]
        def progress():
            value=original_progress()
            if env.ticks!=725:return value
            assert env.tracker.frontier!=forced_segment
            arc=env.corridor.arcs[forced_segment]+np.clip((env.position-env.corridor.starts[forced_segment])@
                env.corridor.tangents[forced_segment],0.,env.corridor.lengths[forced_segment])
            interventions.append(dict(tick=env.ticks,actual_frontier=env.tracker.frontier,
                old_progress=float(value[0]),diagnostic_progress=float(arc)))
            return float(arc),forced_segment
        env.progress=progress
        chunks=[];held=[]
        while not env.done:
            assert time.perf_counter()-start<60 and costs['actor_calls']<200 and costs['new_physics_ticks']<2880
            a=p.act(env.observation(),True)[0];costs['actor_calls']+=1
            if not chunks:assert np.array_equal(env.decode_action(a),actions[36])
            _,_,_,info=env.step(a);chunks.append(info['rows']);held.append(info['held_action'])
            costs['new_physics_ticks']+=len(info['rows'])
        assert len(interventions)==1
        full=np.vstack([rows[:720],*chunks]);all_actions=np.vstack([actions[:36],held])
        check=difference_audit(ref[0],full);coverage,after_events=audit_commands(ref,np.vstack([ref[0],full[:,:2]]),endpoint_tolerance=1.25)
        success=bool(env.success and check['kinematic_pass'] and check['rest_pass'] and coverage['geometry_complete'])
        np.savez_compressed(stage/f'{label}.npz',reference=ref,rows=full,actions=all_actions)
        record=dict(label=label,success=success,reason=env.reason,trace=f'{label}.npz',
            interventions=interventions,original_events=events,result_events=after_events,
            audit=check,independent_coverage=coverage,full_time_s=len(full)*.001 if success else None,
            baseline=score_complete(ref,rows,actions),**score_complete(ref,full,all_actions,env),
            curvature=curvature_metrics(ref,full))
        records.append(record);write_json(stage/'results.json',dict(records=records,costs=costs,goal_complete=False,diagnostic_only=True))
        print('MICRO_GATE_CAUSE',label,success,interventions,
            {k:(record['baseline']['shape'][k],record['shape'][k]) for k in ('out_edge_outward_mm','full_physical_reverse_deg')},flush=True)
    receipt(stage,dict(status='completed_one_ms_gate_guidance_cause_probe',costs=costs,
        complete_suffixes=sum(r['success'] for r in records),wall_s=time.perf_counter()-start,goal_complete=False))


if __name__=='__main__':main()
