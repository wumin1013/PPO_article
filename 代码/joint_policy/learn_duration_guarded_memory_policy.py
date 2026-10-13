"""Joint full-route learning after identifying omitted duration and speed tradeoffs."""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import copy
import time
import multiprocessing as mp
import numpy as np
import torch
from scipy.optimize import minimize, linprog
from duration_joint_metrics import guards, guard_names, penalty
from learn_early_memory_joint_policy import worker_initialize, worker_evaluate, metric_functions
from learn_direct_observable_joint_policy import finish
from learn_complete_joint_teachers import parameters_in_bounds
from study_io import BASE, register, receipt, write_json, sha
from search_complete_return import read


def main():
    torch.set_num_threads(1)
    source = BASE/'canonical_gate_band_coupled_peak_memory_learning_20260920'
    reference = BASE/'canonical_gate_band_main_turn_memory_joint_learning_20260920'
    parent_file = reference/'research_candidate.pt'
    assert sha(parent_file) == 'ab5f53db4bd77754aa6d6186a85b65a2a724e4f78cb1715948c60bdbc3ac2503'
    audit = read(source/'duration_tradeoff_audit.json')
    assert sum(c['expanded_guard_pass'] for c in audit['records']) == 1
    md = torch.load(source/'state_joint/candidate.pt',map_location='cpu',weights_only=False)
    steps = np.asarray(read(source/'registration.json')['budget']['parameter_scales'])
    stage = register('canonical_gate_band_duration_guarded_memory_learning_20260920',
        [__file__,BASE/'duration_joint_metrics.py',BASE/'observable_memory_joint_policy.py',
         BASE/'learn_early_memory_joint_policy.py',BASE/'learn_direct_observable_joint_policy.py',
         BASE/'learn_shared_temporal_joint_policy.py',parent_file,source/'receipt.json',
         source/'duration_tradeoff_audit.json',source/'state_joint/results.json',
         source/'state_joint/candidate.pt',source/'support_audit.json',source/'replayed_states.npz'],
        dict(wall_s=360,maximum_new_evaluations=90,parameters=27,bounds=[-1.,1.],
             extra_guard_names=guard_names(),maximum_cached_model_proposals=2,optimizer_calls=88),
        'Audit of302 completed vectors finds the apparent coupled gain prolongs outgoing '
        'excursion and heading overshoot, increases excursion area, and lowers exit speed '
        'and actual cruise. Keep the exact same27-parameter learned state-memory function '
        'and the ab5 reference. Retain ALL existing guards and add non-regression for '
        'recorded exit/heading durations, significant reverse lateral displacement/time, '
        'entry/exit speeds, actual and longest99cruise, total time and0.2mm shape. Add '
        'continuous polygon-interpolated excursion durations/area as gradient information, '
        'without replacing raw1ms duration checks. Objective adds squared real excursion '
        'area to existing full shape loss. Reuse all saved vectors; fit a local secant '
        'model ONLY to propose at most two joint physical validations, then up to88 scaled '
        'COBYLA evaluations. Accept only truly complete outcomes satisfying every guard. '
        'If none improves, inspect active-constraint and temporal-support conflicts instead '
        'of continuing this optimizer. Full physical parameter optimization, not PPO; '
        'no timing switches, route IDs, angle tables or sealed tests enter the policy.')
    started = time.perf_counter(); deadline = started+340
    folder = stage/'state_joint'; folder.mkdir()
    data = read(source/'baseline.json'); baseline = data['baseline']['records']
    selected = [r for r in baseline if r['path']==304]
    support = read(source/'support_audit.json'); choice = support['selected']
    assert choice['paths']==[304]
    write_json(stage/'baseline.json',data); write_json(stage/'support_audit.json',support)
    with np.load(source/'replayed_states.npz') as z:
        np.savez_compressed(stage/'replayed_states.npz',**{k:z[k] for k in z.files})
    old_score, old_guards = metric_functions(data,selected,stage,md['parent'],deadline)
    def score(records):
        return old_score(records)+sum(penalty(r) for r in records)
    def constraints(records):
        return np.concatenate([old_guards(records)]+[guards(r,b) for r,b in zip(records,selected)])
    history = copy.deepcopy(read(source/'state_joint/results.json')['candidates'])
    for c in history:
        c['tag']='reused_'+c['tag']; c['score']=score(c['records'])
        c['measured_inequalities']=constraints(c['records']).tolist()
    zero=history[0]; md['reused_candidates']=len(history)
    def feasible(c):
        return parameters_in_bounds(c['coefficients'],-1.,1.) and min(c['measured_inequalities'])>=-1e-8
    assert feasible(zero)
    cache={tuple(np.round(c['coefficients'],12)):c for c in history}
    seed=np.asarray(zero['coefficients']); c0=np.asarray(zero['measured_inequalities'])
    # Fit continuous secants in measured axis units; report residuals, never trust them as outcomes.
    x=(np.asarray([c['coefficients'] for c in history])-seed)/steps
    y=np.array([[c['score']-zero['score'],*(np.asarray(c['measured_inequalities'])-c0)] for c in history])
    distance=np.linalg.norm(x,axis=1); weights=1./(1.+distance**2)
    beta=np.linalg.lstsq(x*np.sqrt(weights[:,None]),y*np.sqrt(weights[:,None]),rcond=1e-8)[0]
    g=beta[:,0]; jac=beta[:,1:].T
    local=linprog(g,A_ub=-jac,b_ub=c0,bounds=[(-1.,1.)]*27,method='highs')
    write_json(stage/'cached_credit_model.json',dict(rank=int(np.linalg.matrix_rank(x)),
        input_vectors=len(history),steps=steps.tolist(),loss_gradient=g.tolist(),constraints_jacobian=jac.tolist(),
        weighted_rms_residual=np.sqrt(np.average((x@beta-y)**2,weights=weights,axis=0)).tolist(),
        predicted_loss_change=float(local.fun) if local.success else None,
        scaled_proposal=local.x.tolist() if local.success else None,
        status=str(local.message),actual_validation_required=True))
    print('DURATION_START',len(history),'score',zero['score'],'model_prediction',local.fun if local.success else None,flush=True)
    pool=None; blocks=[]
    try:
        pool=mp.get_context('spawn').Pool(1,worker_initialize,(md,selected,folder,deadline))
        def evaluate(u):
            key=tuple(np.round(u,12))
            if key not in cache:
                if len(history)-md['reused_candidates']>=90:raise TimeoutError('Registered evaluation limit')
                tag=f'duration_{len(history)}'
                records=pool.map(worker_evaluate,[(304,u,tag)])
                c=dict(tag=tag,coefficients=np.asarray(u).tolist(),records=records,score=score(records),
                    measured_inequalities=constraints(records).tolist(),actual_rerun_paths=[304],exact_zero_support_paths=[])
                history.append(c);cache[key]=c
                best=min(filter(feasible,history),key=lambda c:c['score'])
                write_json(folder/'results.json',dict(candidates=history,best=best['tag']))
                if (len(history)-md['reused_candidates'])%5==0:
                    print('DURATION_JOINT',len(history)-md['reused_candidates'],'best',best['score'],
                          'latest',c['score'],'guard',min(c['measured_inequalities']),flush=True)
            return cache[key]
        if local.success and local.fun<-.001:
            for scale in (1.,.25):
                c=evaluate(seed+steps*local.x*scale)
                blocks.append(dict(method='cached_secant_joint_validation',scale=scale,tag=c['tag'],
                    predicted_loss_change=float(local.fun*scale),actual_loss_change=c['score']-zero['score'],
                    measured_guard_minimum=min(c['measured_inequalities'])))
        best=min(filter(feasible,history),key=lambda c:c['score'])
        solved=minimize(lambda z:evaluate(seed+steps*z)['score'],(np.asarray(best['coefficients'])-seed)/steps,
            method='COBYLA',bounds=[((-1.-v)/h,(1.-v)/h) for v,h in zip(seed,steps)],
            constraints=[dict(type='ineq',fun=lambda z:np.array(evaluate(seed+steps*z)['measured_inequalities']))],
            options=dict(maxiter=88,rhobeg=1.,tol=1e-6,catol=1e-8))
        blocks.append(dict(method='duration_area_joint_learning',solver=str(solved.message)))
        pool.close();pool.join();pool=None
        best=min(filter(feasible,history),key=lambda c:c['score'])
        finish(stage,md,baseline,data,best,zero,history,blocks,choice,parent_file,started)
    except TimeoutError:
        if pool is not None:pool.terminate();pool.join();pool=None
        best=min(filter(feasible,history),key=lambda c:c['score'])
        finish(stage,md,baseline,data,best,zero,history,blocks,choice,parent_file,started,budget_exhausted=True)
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_duration_guarded_memory_learning',error=str(exc),
            wall_s=time.perf_counter()-started,adopted=False,goal_complete=False));raise


if __name__=='__main__':mp.freeze_support();main()
