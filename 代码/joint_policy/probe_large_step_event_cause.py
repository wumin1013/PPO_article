"""Bidirectional single-sensor, single-decision causal interventions."""
import time
import numpy as np
import torch
from kernel_local_feedback_policy import KernelLocalFeedbackPolicy
from probe_coordinated_suffix import reconstruct
from probe_one_decision_credit import score_complete
from audit_xy_curvature_resolution import curvature_metrics
from validate_actions import difference_audit
from search_complete_return import read
from study_io import BASE, register, receipt, write_json


def main():
    torch.set_num_threads(1)
    audit=BASE/'large_step_feedback_event_audit_20260917'
    source=read(audit/'results.json')
    parent=BASE/'recentered_reverse_shape_credit_learning_20260917'
    large=BASE/'full_feedback_geometry_fixed_learning_20260917'
    selection=read(parent/'training_selection.json')
    stage=register('large_step_single_sensor_causal_probe_20260917',
        [__file__,audit/'results.json',audit/'receipt.json',parent/selection['model'],
         large/'candidate_0.5.pt',large/'candidate_1.pt',BASE/'probe_coordinated_suffix.py',
         BASE/'kernel_local_feedback_policy.py',BASE/'joint_runtime.py',BASE/'joint_runtime_v5.py'],
        dict(paths=[8,15],suffixes=4,actor_calls=500,new_physics_ticks=12000,
             saved_prefix_geometry_ticks=6000,wall_s=90,training_updates=0),
        'Test the specific current nonmonotone regressions. At path8 decision29 and '
        'path15 decision35, swap only the two point-tangent components of the4.6mm '
        'sensor, keeping all other actual observations, physical states, parameters '
        'and dynamics fixed. Test both candidate receiving parent sensor and parent '
        'receiving candidate sensor. Apply once, then resume each unchanged actor '
        'through complete braking/rest. Baselines reuse saved traces. Adoption: none; '
        'a causal mechanism finding requires rescue and reverse impairment, not just '
        'temporal coincidence. These are intentionally counterfactual observations, '
        'not learned or deployable sensing. Stop after four suffixes; no extra grid.')
    started=time.perf_counter();costs=dict(actor_calls=0,new_physics_ticks=0,saved_prefix_geometry_ticks=0)
    records=[]
    for path,candidate,anchor in ((8,1.,29),(15,.5,35)):
        record=next(r for r in source['records'] if r['path']==path)
        data={}
        for step in (0.,candidate):
            trace=BASE/next(t['trace'] for t in record['traces'] if t['step']==step)
            with np.load(trace) as z:ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
            env=reconstruct(ref,rows,actions,anchor*20,costs)
            assert env.tracker.frontier==0
            data[step]=(ref,rows,actions,env.state(),env.observation())
        for recipient,donor in ((candidate,0.),(0.,candidate)):
            ref,rows,actions,state,obs=data[recipient]
            model=parent/selection['model'] if recipient==0 else large/f'candidate_{recipient:g}.pt'
            p=KernelLocalFeedbackPolicy(256)
            p.load_state_dict(torch.load(model,map_location='cpu',weights_only=False)['policy'])
            env.restore(state);original=p.act(obs,True)[0];costs['actor_calls']+=1
            assert np.array_equal(env.decode_action(original),actions[anchor])
            altered=obs.copy();altered[50:52]=data[donor][4][50:52]
            assert np.array_equal(obs[:50],altered[:50]) and np.array_equal(obs[52:],altered[52:])
            action=p.act(altered,True)[0];costs['actor_calls']+=1
            changed_action=(env.decode_action(action)-env.decode_action(original)).tolist()
            chunks=[];held=[]
            while not env.done:
                assert time.perf_counter()-started<90 and costs['new_physics_ticks']<10000 and costs['actor_calls']<490
                if chunks:action=p.act(env.observation(),True)[0];costs['actor_calls']+=1
                _,_,_,info=env.step(action);chunks.append(info['rows']);held.append(info['held_action'])
                costs['new_physics_ticks']+=len(info['rows'])
            full=np.vstack([rows[:anchor*20],*chunks]);all_actions=np.vstack([actions[:anchor],held])
            assert np.array_equal(full[:anchor*20],rows[:anchor*20])
            check=difference_audit(ref[0],full)
            success=bool(env.success and check['kinematic_pass'] and check['rest_pass'])
            name=f'path{path}_recipient{recipient:g}.npz'
            np.savez_compressed(stage/name,reference=ref,rows=full,actions=all_actions)
            value=score_complete(ref,full,all_actions,env)
            record_out=dict(path=path,recipient_step=recipient,donor_step=donor,anchor=anchor,
                changed_sensor_distance_mm=4.6,changed_action=changed_action,success=success,
                reason=env.reason,trace=name,audit=check,geometry=env.tracker.snapshot(1.25),
                full_time_s=len(full)*.001 if success else None,**value,
                curvature=curvature_metrics(ref,full),baseline=score_complete(ref,rows,actions))
            records.append(record_out)
            write_json(stage/'results.json',dict(records=records,costs=costs,diagnostic_only=True,goal_complete=False))
            names=('in_edge_outward_mm','out_edge_outward_mm','full_physical_reverse_deg')
            print('SENSOR_CAUSE',path,recipient,success,changed_action,
                {k:(record_out['baseline']['shape'][k],value['shape'][k]) for k in names},flush=True)
    assert costs['saved_prefix_geometry_ticks']<=6000 and costs['new_physics_ticks']<=12000
    receipt(stage,dict(status='completed_bidirectional_single_sensor_cause_probe',costs=costs,
        suffixes=len(records),complete_suffixes=sum(r['success'] for r in records),wall_s=time.perf_counter()-started))


if __name__=='__main__':main()
