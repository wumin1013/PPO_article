"""Matched adaptive constrained parameter learning using actual complete returns."""
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
from duration_joint_metrics import guards,penalty
from shape_event_credit import shape_event_learning_guards
from learn_intrinsic_joint_recovery import intrinsic_guards
from learn_complete_joint_teachers import teacher_score,parameters_in_bounds
from learn_early_memory_joint_policy import worker_initialize,worker_evaluate,metric_functions
from learn_direct_observable_joint_policy import finish
from study_io import BASE,register,write_json,receipt
from search_complete_return import read


TASK={}


def initialize_task(md,selected,retained,data,stage,deadline,old):
    torch.set_num_threads(1)
    TASK.update(md=md,selected=selected,retained=retained,data=data,stage=stage,deadline=deadline,old=old)


def optimize(arm):
    stage=TASK['stage'];folder=stage/arm;folder.mkdir()
    md=copy.deepcopy(TASK['md']);selected=TASK['selected'];retained=TASK['retained']
    old=copy.deepcopy(TASK['old']);history=old.copy();zero=old[0];origin=np.asarray(zero['coefficients']).reshape(3,7,8,3)
    worker_initialize(md,selected,folder,TASK['deadline'])
    _,old_guards=metric_functions(TASK['data'],selected,stage,md['parent'],TASK['deadline'])
    kernels=[3,7];scale=np.tile([.02,.05,.05],6)
    initial=np.concatenate([origin[:,0,k].reshape(-1) for k in kernels])
    def expand(x):
        u=origin.copy()
        for j,k in enumerate(kernels):u[:,0,k]=(initial+x*scale)[9*j:9*j+9].reshape(3,3)
        return u.reshape(-1)
    def coordinates(c):
        u=np.asarray(c['coefficients']).reshape(origin.shape)
        return (np.concatenate([u[:,0,k].reshape(-1) for k in kernels])-initial)/scale
    def feasible(c):return parameters_in_bounds(c['coefficients'],-1.,1.) and min(c['augmented_measured_inequalities'])>=-1e-8
    cache={tuple(np.round(c['coefficients'],12)):c for c in old};reused=len(old);invalid=0
    with np.load(selected[0]['trace']) as z:prefix=z['rows'][:300].copy()
    def evaluate(x):
        nonlocal invalid
        u=expand(x);key=tuple(np.round(u,12))
        if not parameters_in_bounds(u,-1.,1.):
            invalid+=1
            return dict(score=1e8+float(np.square(x).sum()),augmented_learning_inequalities=[-100.]*len(zero['augmented_learning_inequalities']))
        if key not in cache:
            tag=f'{arm}_{len(history)-reused}'
            records=[worker_evaluate((104,u,tag))]
            with np.load(records[0]['trace']) as z:assert np.array_equal(z['rows'][:300],prefix)
            original=np.r_[old_guards(records),guards(records[0],selected[0])];extra=intrinsic_guards(records,selected)
            c=dict(tag=tag,coefficients=u.tolist(),records=records,
                score=float(teacher_score(records[0],selected[0],retained[0],barrier=False)+penalty(records[0])),
                measured_inequalities=original.tolist(),augmented_measured_inequalities=np.r_[original,extra].tolist(),
                augmented_learning_inequalities=np.r_[shape_event_learning_guards(records,selected,original),extra].tolist(),
                actual_rerun_paths=[104],exact_zero_support_paths=[])
            history.append(c);cache[key]=c
            best=min(filter(feasible,history),key=lambda r:r['score'])
            write_json(folder/'results.json',dict(candidates=history,best=best['tag']))
            if (len(history)-reused)%10==0:
                print('DIRECT_STABLE',arm,len(history)-reused,c['score'],min(c['augmented_measured_inequalities']),best['score'],flush=True)
        return cache[key]
    start=zero if arm=='current' else next(c for c in old if c['tag']=='metric_0_joint_0.25')
    result=None;exhausted=False
    try:
        result=minimize(lambda x:evaluate(x)['score'],coordinates(start),method='COBYLA',
            bounds=list(zip((-1-initial)/scale,(1-initial)/scale)),
            constraints=[dict(type='ineq',fun=lambda x:np.asarray(evaluate(x)['augmented_learning_inequalities']))],
            options=dict(maxiter=120,rhobeg=.5,tol=1e-5,catol=1e-8))
    except TimeoutError:exhausted=True
    best=min(filter(feasible,history),key=lambda c:c['score'])
    row=dict(arm=arm,start_tag=start['tag'],best=best['tag'],new_candidates=len(history)-reused,
        solver_message=str(result.message) if result is not None else 'registered_wall_budget',
        budget_exhausted=exhausted,rejected_out_of_bounds_proposals=invalid)
    write_json(folder/'completion.json',row)
    return dict(row,history=history[reused:])


def main():
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_stable_support_joint_credit_learning_20260920'
    parent_file=BASE/'canonical_gate_band_decoded_joint_step_learning_20260920/research_candidate.pt'
    md=torch.load(source/'state_joint/candidate.pt',map_location='cpu',weights_only=False)
    data=read(source/'baseline.json');baseline=data['baseline']['records']
    history=read(source/'state_joint/results.json')['candidates'];zero=history[0];selected=zero['records']
    retained=[next(o['baseline_shape'] for o in data['retained'] if o['path']==104)]
    stage=register('canonical_gate_band_direct_stable_joint_learning_20260920',
        [__file__,parent_file,source/'receipt.json',source/'state_joint/candidate.pt',source/'state_joint/results.json',
         source/'optimization_blocks.json',BASE/'shape_event_credit.py',BASE/'joint_teacher_state_policy.py',
         BASE/'observable_memory_joint_policy.py',BASE/'learn_early_memory_joint_policy.py',BASE/'learn_direct_observable_joint_policy.py'],
        dict(wall_s=240,workers=2,arms=['current','quarter'],maximum_evaluations_per_arm=120,
             parameter_scales=[.02,.05,.05],initial_scaled_radius=.5,coefficient_bounds=[-1.,1.]),
        'A fresh local response at the geometry-supported quarter point cannot '
        'predict even0.001mm guarded outgoing improvement. This does not prove '
        'physical or policy infeasibility. Compare two actual-return constrained '
        'COBYLA starts in the IDENTICAL18-parameter state/memory function: the '
        'protected current policy and the measured quarter point. Reuse all39 '
        'completed same-function vectors. Each new parameter vector runs the '
        'entire physical104 route; protect all current original and intrinsic '
        'guards and exact other87 routes. This adaptive trust-region method '
        'updates its interpolation from actual complete results, rather than '
        'using the fixed symmetric linear credit model. At most120 calls per '
        'arm/240s total, with no frozen-test access or PPO. Stop these arms at '
        'budget and use results to decide whether broader temporal influence '
        'is necessary; no failed-prefix speed or relaxed guard may be selected.')
    started=time.perf_counter();deadline=started+220;folder=stage/'state_joint';folder.mkdir()
    write_json(stage/'baseline.json',data);choice=dict(paths=[104],variable_kernels=[3,7])
    write_json(stage/'support_audit.json',read(source/'support_audit.json'))
    md['reused_candidates']=len(history);pool=None
    try:
        pool=mp.get_context('spawn').Pool(2,initialize_task,(md,selected,retained,data,stage,deadline,history))
        results=pool.map(optimize,['current','quarter']);pool.close();pool.join();pool=None
        blocks=[]
        for r in results:history.extend(r.pop('history'));blocks.append(r)
        def feasible(c):return parameters_in_bounds(c['coefficients'],-1.,1.) and min(c['augmented_measured_inequalities'])>=-1e-8
        best=min(filter(feasible,history),key=lambda c:c['score'])
        finish(stage,md,baseline,data,best,zero,history,blocks,choice,parent_file,started,
            budget_exhausted=any(r['budget_exhausted'] for r in results))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_direct_stable_learning',error=str(exc),wall_s=time.perf_counter()-started,
            adopted=False,goal_complete=False));raise


if __name__=='__main__':mp.freeze_support();main()
