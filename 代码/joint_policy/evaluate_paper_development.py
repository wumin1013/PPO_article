"""One bounded development evaluation of the locked checkpoint; no training."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import json
from pathlib import Path
import time
import numpy as np
import torch

from deep_joint_credit_policy import load_credit_policy
from learned_action_pulse_policy import environment_factory
from analyze_nonlinear_joint_response import measure
from audit_xy_curvature_resolution import curvature_metrics
from geometry_spec import audit_commands
from study_io import BASE, register, receipt, sha, write_json


def main():
    torch.set_num_threads(1)
    review=BASE/'paper_evidence_review'
    frozen=json.loads((review/'frozen_evaluation_candidate.json').read_text(encoding='utf-8'))
    model=BASE/frozen['model'];assert sha(model)==frozen['sha256']
    datafile=BASE/'learning_preparation_v4/paths.json'
    refs=json.loads(datafile.read_text(encoding='utf-8'))['validation'];assert len(refs)==12
    # Same runtime used by the saved candidate. Its optional pulse remains zero.
    stage=register('paper_evidence_review/development',
        [__file__,review/'frozen_evaluation_candidate.json',review/'evidence_matrix.json',datafile,
         review/'learning_costs/receipt.json',review/'closed_paths/receipt.json']+
        [BASE/n for n in ('deep_joint_credit_policy.py','observable_memory_joint_policy.py','joint_teacher_state_policy.py',
            'learned_action_pulse_policy.py','gate_band_state.py','canonical_gate_state.py','continuous_gate_guidance.py',
            'continuous_sensor_runtime.py','joint_runtime_v5.py','joint_runtime.py','geometry_spec.py',
            'analyze_nonlinear_joint_response.py','validate_actions.py','complete_motion_metrics.py','complete_corner_metrics.py')],
        dict(complete_development_paths=list(range(12)),maximum_policy_versions=1,wall_s=1800,
             maximum_actor_calls=6000,maximum_simulation_ticks=100000,training_updates=0,
             fitting_updates=0,parameter_optimization_updates=0,sealed_test_paths=0),
        'Fill only the missing complete12 development evaluation for the immutable worst_guard checkpoint. '
        'Same serialized nested actor, same runtime and20ms hold/1ms execution; verify optional pulse is zero. '
        'Every route starts with reset policy memory and ends in physical success/failure. Save failed prefixes '
        'as diagnostic data with complete_time=null. Stop after12 routes or hard resource limits; no optimizer, '
        'model change, alternate candidate, or sealed test. These previously-used development paths are not independent tests. '
        'No adoption or superiority inference from this one-model evaluation.')
    started=time.perf_counter();deadline=started+1800.
    policy=load_credit_policy(torch.load(model,map_location='cpu',weights_only=False))
    records=[];costs=dict(actor_calls=0,physics_ticks=0);stop_reason=None
    for index,points in enumerate(refs):
        if time.perf_counter()>=deadline or costs['actor_calls']>=6000 or costs['physics_ticks']>=100000:
            stop_reason='batch_resource_limit';break
        ref=np.asarray(points,float);env=policy.bind(environment_factory(ref))
        chunks=[];obs=[];actions=[];latents=[];executed=[];error=None
        try:
            while not env.done:
                if time.perf_counter()>=deadline or costs['actor_calls']>=6000 or costs['physics_ticks']>=100000:
                    raise TimeoutError('Registered development evaluation resource limit')
                x=env.observation();latent=policy.act(x,True)[0];costs['actor_calls']+=1
                assert not np.any(env.pending_pulse), '20ms action-hold contract requires zero pulse'
                _,_,_,info=env.step(latent)
                rows=info['rows'];chunks.append(rows);obs.append(x);actions.append(info['held_action']);latents.append(latent)
                executed.append(info['executed_actions_1ms']);costs['physics_ticks']+=len(rows)
            rows=np.vstack(chunks)
            result=measure(ref,rows,env)
            geometry,_=audit_commands(ref,np.vstack([ref[0],rows[:,:2]]),epsilon=1.25,
                endpoint_tolerance=1.25,closed=bool(np.linalg.norm(ref[0]-ref[-1])<1e-9))
            result['independent_geometry']=geometry
            result['success']=bool(result['success'] and geometry['geometry_complete'])
            if not result['success']:result['complete_time_s']=None
            if len(ref)==3:result['curvature']=curvature_metrics(ref,rows)
        except Exception as exc:
            error=f'{type(exc).__name__}: {exc}'
            rows=np.vstack(chunks) if chunks else np.empty((0,13))
            result=dict(success=False,reason='evaluation_exception',error=error,complete_time_s=None)
        file=stage/f'path{index}.npz'
        action_rows=np.vstack(executed) if executed else np.empty((0,3))
        np.savez_compressed(file,reference=ref,rows=rows,observations=np.asarray(obs),actions=np.asarray(actions),
                            latents=np.asarray(latents),executed_actions_1ms=action_rows)
        ordinary=np.flatnonzero(rows[:,12]==0)
        hold=True
        if len(ordinary)>1:
            changed=np.flatnonzero(np.any(action_rows[1:]!=action_rows[:-1],axis=1))+1
            changed=changed[(rows[changed,12]==0)&(rows[changed-1,12]==0)]
            hold=bool(np.all(changed%20==0))
        result.update(path=index,trace=str(file.relative_to(BASE)),trace_sha256=sha(file),
                      elapsed_prefix_or_complete_s=len(rows)*.001,ordinary_20ms_action_hold=hold,
                      actor_decisions=len(actions),closed=bool(np.linalg.norm(ref[0]-ref[-1])<1e-9))
        assert hold
        records.append(result)
        write_json(stage/'results.json',dict(model_sha256=sha(model),records=records,costs=costs,
            development_only=True,training_updates=0,sealed_test_opened=False))
        print('DEVELOPMENT',index,result['success'],result.get('reason'),result['complete_time_s'],
              'prefix_s',result['elapsed_prefix_or_complete_s'],flush=True)
        if error:
            stop_reason=error;break
    assert sha(model)==frozen['sha256']
    summary=dict(planned_paths=12,evaluated_paths=len(records),successes=sum(r['success'] for r in records),
        failures=[dict(path=r['path'],reason=r.get('reason'),elapsed_prefix_s=r['elapsed_prefix_or_complete_s']) for r in records if not r['success']],
        not_evaluated_paths=list(range(len(records),12)),
        complete_time_sum_s=sum(r['complete_time_s'] for r in records) if len(records)==12 and all(r['success'] for r in records) else None,
        successful_paths_time_sum_s=sum(r['complete_time_s'] for r in records if r['success']),
        cruise99_sum_successful_s=sum(r['motion']['cruise_ge99_s'] for r in records if r['success']),
        longest_cruise99_successful_s=max((r['motion']['longest_cruise_ge99_s'] for r in records if r['success']),default=0.),
        all_ordinary_actions_held_20ms=all(r['ordinary_20ms_action_hold'] for r in records))
    write_json(stage/'results.json',dict(model_sha256=sha(model),records=records,costs=costs,summary=summary,
        development_only=True,training_updates=0,sealed_test_opened=False,comparison_not_budget_matched=True))
    receipt(stage,dict(status='completed_locked_candidate_development_evaluation' if len(records)==12 else 'stopped_bounded_development_evaluation',
        model_sha256=sha(model),summary=summary,costs=costs,wall_s=time.perf_counter()-started,
        stop_reason=stop_reason,results_sha256=sha(stage/'results.json'),
        ppo_updates=0,offline_fitting_updates=0,parameter_optimization_updates=0,
        algorithm_optimization_rounds=0,sealed_test_opened=False,formal_adoption=False,goal_complete=False))
    print('DEVELOPMENT_COMPLETE',json.dumps(summary),flush=True)


if __name__=='__main__': main()
