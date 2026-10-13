"""Release pre-turn, peak, and recovery coefficients in the same learned function."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import copy
import time
import multiprocessing as mp
import numpy as np
import torch
from scipy.optimize import minimize
from learn_early_memory_joint_policy import worker_initialize,worker_evaluate,metric_functions
from learn_direct_observable_joint_policy import finish
from learn_complete_joint_teachers import parameters_in_bounds
from study_io import BASE,register,receipt,write_json,sha
from search_complete_return import read


def main():
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_peak_memory_shape_restoration_20260920'
    completed=read(source/'receipt.json');assert completed['status'].startswith('completed')
    assert not completed['research_acceptable'],'Audit the new protected model before continuing'
    reference=BASE/'canonical_gate_band_main_turn_memory_joint_learning_20260920'
    parent_file=reference/'research_candidate.pt'
    md=torch.load(source/'state_joint/candidate.pt',map_location='cpu',weights_only=False)
    old_steps=np.array(read(reference/'local_credit_model.json')['steps']).reshape(3,2,3)
    peak_steps=np.array(read(source/'local_credit_model.json')['steps']).reshape(3,1,3)
    steps=np.concatenate([old_steps,peak_steps],axis=1).reshape(-1)
    stage=register('canonical_gate_band_coupled_peak_memory_learning_20260920',
        [__file__,BASE/'observable_memory_joint_policy.py',BASE/'joint_teacher_state_policy.py',
         BASE/'learn_early_memory_joint_policy.py',BASE/'learn_direct_observable_joint_policy.py',
         BASE/'learn_shared_temporal_joint_policy.py',BASE/'learn_complete_joint_teachers.py',parent_file,
         source/'receipt.json',source/'state_joint/candidate.pt',source/'state_joint/results.json',
         source/'local_credit_model.json',source/'zero_extension_identity.json',source/'exploration_anchor.json',
         source/'support_audit.json',source/'replayed_states.npz',reference/'local_credit_model.json'],
        dict(wall_s=480,workers=3,maximum_optimizer_calls=80,total_parameters=27,
             parameter_scales=steps.tolist(),bounds=[-1.,1.],strict_reference_sha256=sha(parent_file)),
        'The fixed-old-weight peak-restoration control reduces violation but cannot satisfy '
        'the full improved-reference guards within its registered run. Retain the EXACT '
        'three-kernel causal memory function, full304 trajectory,87 exact zero-support routes, '
        'reference model and all constraints. Reuse all actual parameter vectors. Change '
        'only optimization scope: jointly release all27 coefficients at pre-turn280ms, '
        'peak-region320ms and recovery380ms training observations. Runtime inputs remain '
        'states/history only; these times are not switches or labels. Use measured axis '
        'scales and start from the saved point with smallest squared guard violation among '
        'points retaining at least half the exploration loss gain. Learn by complete '
        'physical COBYLA for80 calls/480s, accepting only measured improved candidates '
        'passing every original guard. This tests cross-stage parameter coordination, '
        'not PPO, an estimator bias, or generalization. No prior physical vector is rerun.')
    started=time.perf_counter();deadline=started+460;folder=stage/'state_joint';folder.mkdir()
    data=read(source/'baseline.json');baseline=data['baseline']['records'];selected=[r for r in baseline if r['path']==304]
    support=read(source/'support_audit.json');choice=support['selected'];assert choice['paths']==[304]
    write_json(stage/'baseline.json',data);write_json(stage/'support_audit.json',support)
    with np.load(source/'replayed_states.npz') as z:np.savez_compressed(stage/'replayed_states.npz',**{k:z[k] for k in z.files})
    score,constraints=metric_functions(data,selected,stage,md['parent'],deadline)
    history=copy.deepcopy(read(source/'state_joint/results.json')['candidates'])
    for c in history:
        c['tag']='reused_'+c['tag'];assert abs(c['score']-score(c['records']))<1e-10
        assert np.array_equal(c['measured_inequalities'],constraints(c['records']))
    zero=history[0];md['reused_candidates']=len(history)
    def feasible(c):return parameters_in_bounds(c['coefficients'],-1.,1.) and min(c['measured_inequalities'])>=-1e-8
    assert feasible(zero)
    lower=min(c['score'] for c in history)
    eligible=[c for c in history if c['score']<=zero['score']-.5*(zero['score']-lower) and parameters_in_bounds(c['coefficients'],-1.,1.)]
    anchor=min(eligible,key=lambda c:np.square(np.minimum(c['measured_inequalities'],0.)).sum())
    seed=np.array(anchor['coefficients'])
    write_json(stage/'coupled_initial_point.json',dict(tag=anchor['tag'],score=anchor['score'],
        guard_minimum=min(anchor['measured_inequalities']),coefficients=seed.tolist(),reused_vectors=len(history)))
    print('COUPLED_PEAK_START',anchor['tag'],anchor['score'],min(anchor['measured_inequalities']),len(history),flush=True)
    cache={tuple(np.round(c['coefficients'],12)):c for c in history};pool=None;blocks=[]
    try:
        pool=mp.get_context('spawn').Pool(3,worker_initialize,(md,selected,folder,deadline))
        def evaluate(u):
            key=tuple(np.round(u,12))
            if key not in cache:
                tag=f'coupled_{len(history)}';records=pool.map(worker_evaluate,[(304,u,tag)])
                c=dict(tag=tag,coefficients=np.asarray(u).tolist(),records=records,score=score(records),
                    measured_inequalities=constraints(records).tolist(),actual_rerun_paths=[304],exact_zero_support_paths=[])
                history.append(c);cache[key]=c
                best=min(filter(feasible,history),key=lambda c:c['score'])
                write_json(folder/'results.json',dict(candidates=history,best=best['tag']))
                if len(history)%5==0:print('COUPLED_PEAK',len(history),'best',best['score'],'latest',c['score'],min(c['measured_inequalities']),flush=True)
            return cache[key]
        solved=minimize(lambda z:evaluate(seed+steps*z)['score'],np.zeros(27),method='COBYLA',
            bounds=[((-1.-v)/h,(1.-v)/h) for v,h in zip(seed,steps)],
            constraints=[dict(type='ineq',fun=lambda z:np.array(evaluate(seed+steps*z)['measured_inequalities']))],
            options=dict(maxiter=80,rhobeg=2.,tol=1e-6,catol=1e-8))
        blocks.append(dict(method='all27_joint_memory_coefficients',solver=str(solved.message)))
        pool.close();pool.join();pool=None
        best=min(filter(feasible,history),key=lambda c:c['score'])
        finish(stage,md,baseline,data,best,zero,history,blocks,choice,parent_file,started)
    except TimeoutError:
        if pool is not None:pool.terminate();pool.join();pool=None
        best=min(filter(feasible,history),key=lambda c:c['score'])
        finish(stage,md,baseline,data,best,zero,history,blocks,choice,parent_file,started,budget_exhausted=True)
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_coupled_peak_memory_learning',error=str(exc),wall_s=time.perf_counter()-started,
            adopted=False,goal_complete=False));raise


if __name__=='__main__':mp.freeze_support();main()
