"""One bounded public long-path evaluation, preserving any failure prefix."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import json
import time
import numpy as np
import torch
from deep_joint_credit_policy import load_credit_policy
from learned_action_pulse_policy import environment_factory
from complete_motion_metrics import complete_motion_metrics
from validate_actions import difference_audit
from study_io import BASE, ROOT, register, receipt, sha, write_json


def main():
    torch.set_num_threads(1)
    review=BASE/'paper_evidence_review';frozen=json.loads((review/'frozen_evaluation_candidate.json').read_text(encoding='utf-8'))
    model=BASE/frozen['model'];assert sha(model)==frozen['sha256']
    ref_file=ROOT/'论文项目/revision_20260913_preview_efficiency_v2/geometry_design_v1/references/mixed_lengths_large.csv'
    ref_hash='33c0740b65bb44c8ace801a71d55fd7c39b07442daea3bc2880730ea56e3f082'
    assert sha(ref_file)==ref_hash
    stage=register('paper_evidence_review/public_long',
        [__file__,review/'frozen_evaluation_candidate.json',review/'development/receipt.json',ref_file,
         BASE/'s2_frozen_command_new_geometry_v1/results.json',BASE/'v7_fixed_long_mixed_v1/evaluation.json',
         BASE/'learned_action_pulse_policy.py',BASE/'observable_memory_joint_policy.py',
         BASE/'complete_motion_metrics.py',BASE/'validate_actions.py'],
        dict(public_development_paths=['mixed_lengths_large'],models=1,wall_s=1200,
            maximum_actor_calls=80000,maximum_simulation_ticks=1600000,training_updates=0,
            fitting_updates=0,parameter_optimization_updates=0,sealed_test_paths=0),
        'Fill the single public complete long-path case already fixed in the8-case evidence matrix. '
        'Same immutable checkpoint and same20ms/1ms runtime as development12. No training, '
        'parameter selection, alternate candidate, or sealed geometry. Keep failed or budget-stopped '
        'prefixes with complete_time=null. Stop once the path succeeds/fails or20min expires. '
        'Reuse existing S2/fixed results only as historical system comparisons, not matched-budget proof.')
    started=time.perf_counter();deadline=started+1200
    ref=np.loadtxt(ref_file,delimiter=',',skiprows=1)
    policy=load_credit_policy(torch.load(model,map_location='cpu',weights_only=False))
    env=policy.bind(environment_factory(ref))
    assert env.max_ticks<=1600000
    chunks=[];actions=[];observations=[];executed=[];latents=[]
    costs=dict(actor_calls=0,physics_ticks=0);error=None;hold=True;last_progress=started
    try:
        while not env.done:
            if time.perf_counter()>=deadline or costs['actor_calls']>=80000 or costs['physics_ticks']>=1600000:
                raise TimeoutError('Registered public long-path evaluation resource limit')
            x=env.observation();latent=policy.act(x,True)[0];costs['actor_calls']+=1
            assert not np.any(env.pending_pulse), 'Optional1ms pulse must remain zero'
            _,_,_,info=env.step(latent)
            chunks.append(info['rows']);actions.append(info['held_action']);observations.append(x);latents.append(latent)
            executed.append(info['executed_actions_1ms']);costs['physics_ticks']+=len(info['rows'])
            ordinary=info['executed_actions_1ms'][info['rows'][:,12]==0]
            hold &= not len(ordinary) or np.all(ordinary==ordinary[0])
            assert hold
            if time.perf_counter()-last_progress>=30:
                last_progress=time.perf_counter()
                progress=dict(wall_s=last_progress-started,actor_calls=costs['actor_calls'],
                    physics_ticks=costs['physics_ticks'],passed_gates=env.tracker.frontier,
                    total_gates=len(env.corridor.gates),complete=False)
                write_json(stage/'progress.json',progress);print('LONG_PROGRESS',json.dumps(progress),flush=True)
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}'
    rows=np.vstack(chunks) if chunks else np.empty((0,13))
    audit=difference_audit(ref[0],rows) if len(rows) else None
    geometry=env.tracker.snapshot(1.25)
    success=bool(error is None and env.success and geometry['geometry_complete'] and audit['kinematic_pass'] and audit['rest_pass'])
    motion=complete_motion_metrics(ref,rows) if len(rows) else None
    file=stage/'trace.npz'
    np.savez_compressed(file,reference=ref,rows=rows,actions=np.asarray(actions),observations=np.asarray(observations),
        latents=np.asarray(latents),executed_actions_1ms=np.vstack(executed) if executed else np.empty((0,3)))
    result=dict(case='mixed_lengths_large',model_sha256=sha(model),reference_sha256=ref_hash,
        success=success,reason=env.reason if error is None else 'evaluation_exception_or_budget',error=error,
        complete_time_s=len(rows)*.001 if success else None,elapsed_prefix_or_complete_s=len(rows)*.001,
        motion=motion,motion_scope='complete' if success else 'failure_prefix_only',
        geometry=geometry,audit=audit,ordinary_20ms_hold=bool(hold),costs=costs,
        trace_sha256=sha(file),training_updates=0,public_development_only=True,
        sealed_test_opened=False,budget_matched_superiority_established=False)
    assert sha(model)==frozen['sha256']
    write_json(stage/'evaluation.json',result)
    receipt(stage,dict(status='completed_public_long_evaluation' if error is None else 'stopped_bounded_public_long_evaluation',
        success=success,model_sha256=sha(model),costs=costs,wall_s=time.perf_counter()-started,
        complete_time_s=result['complete_time_s'],elapsed_prefix_or_complete_s=result['elapsed_prefix_or_complete_s'],
        evaluation_sha256=sha(stage/'evaluation.json'),trace_sha256=sha(file),training_updates=0,
        parameter_optimization_updates=0,algorithm_optimization_rounds=0,sealed_test_opened=False,
        automatic_retry=False,goal_complete=False))
    print('PUBLIC_LONG_COMPLETE',json.dumps(result),flush=True)


if __name__=='__main__': main()
