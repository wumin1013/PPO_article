"""Validate current full-policy response calibration and close aggregate guards."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import copy
import multiprocessing as mp
import time
import numpy as np
import torch
from learn_pathwise_full_credit import pathwise_response, validation, actual_selection
from learn_full_pool_compensation import solve_response, progress
from learn_full_pool_joint_credit import initialize, evaluate, candidate_metadata
from learn_trace_resolved_joint_credit import signed_channels, read_rows
from learn_retained_joint_growth import growth_deficits
from study_io import BASE, register, receipt, write_json, sha
from search_complete_return import read


def calibrate(response_data, baseline, retained, trust, controls):
    output=copy.deepcopy(response_data)
    base,matrix,bm,mj,bounds,modes,traces=output
    u=np.array([c['parameters'] for _,c in controls])
    g=trust.T@trust/len(trust)
    gram=u@g@u.T
    ridge=.01*np.linalg.eigvalsh(gram).max()
    correction=np.linalg.solve(gram+ridge*np.eye(len(u)),u@g)
    residual=np.array([growth_deficits(c,retained)-base-matrix@z for (_,c),z in zip(controls,u)])
    matrix += residual.T@correction
    residual=np.array([np.array([r['corridor']['margin_lower_mm'] for r in c['records']])-bm-mj@z for (_,c),z in zip(controls,u)])
    mj += residual.T@correction
    for path,m in traces.items():
        residuals={k:[] for k in m['jac']}
        for (folder,c),z in zip(controls,u):
            r=next(r for r in c['records'] if r['path']==path)
            assert r['success']
            ref,rows=read_rows(folder,r)
            actual=signed_channels(ref,rows,m['base']['grid'],np.arange(len(m['base']['incoming'])))
            for k in residuals:residuals[k].append(actual[k]-m['base'][k]-m['jac'][k]@z)
        for k in residuals:m['jac'][k] += np.array(residuals[k]).T@correction
    return output,dict(controls=[c['tag'] for _,c in controls],ridge=float(ridge),gram_eigenvalues=np.linalg.eigvalsh(gram).tolist())


def main():
    torch.set_num_threads(1)
    origin=BASE/'canonical_gate_band_full_pool_joint_credit_learning_20260919'
    source=BASE/'canonical_gate_band_active_constraint_joint_learning_20260919'
    earlier=BASE/'canonical_gate_band_full_pool_compensation_learning_20260919'
    parent=BASE/'canonical_gate_band_iterative_motion_constraint_learning_20260919/research_candidate.pt'
    assert sha(parent)=='5ab47111bfcec3a511903e20f1249c91bbfb65e5ab4cd1a3d2b4a493a979c26a'
    data=read(source/'baseline.json');baseline,retained=data['baseline'],data['retained']
    with np.load(source/'parameter_space.npz') as z:directions,trust=z['directions'].copy(),z['trust'].copy()
    n=len(directions);indexed={};controls=[];held=[]
    for folder in (origin,source):
        for c in read(folder/'results.json')['candidates']:
            if c['tag'].startswith('axis'):
                for r in c['records']:r['trace']=str(folder/r['trace'])
                indexed[c['tag']]=c
    for folder,prefix in ((earlier,'compensation_joint_'),(source,'active_joint_')):
        cs={c['tag']:c for c in read(folder/'results.json')['candidates']}
        for scale,destination in ((.5,controls),(1.,held)):
            c=copy.deepcopy(cs[prefix+str(scale)])
            c['parameters'] += [0.]*(n-len(c['parameters']))
            destination.append((folder,c))
    metadata=torch.load(parent,map_location='cpu',weights_only=False)
    stage=register('canonical_gate_band_full_pool_secant_guard_learning_20260919',
        [__file__,BASE/'learn_pathwise_full_credit.py',BASE/'learn_full_pool_compensation.py',
         BASE/'learn_full_pool_joint_credit.py',source/'results.json',source/'receipt.json',
         source/'parameter_space.npz',source/'baseline.json',origin/'results.json',earlier/'results.json',parent],
        dict(wall_s=360,workers=4,new_full64_candidates=4,optimizer_s_per_arm=100,
             reused_axis_candidates=26,fit_actors=2,held_actors=2),
        'The expanded half-step passed all per-corner envelope guards but total reversal rose0.00872deg, '
        'so it was not selected. Aggregate actual acceptance had not been included in the proposal '
        'constraints. Add summed exit,reversal,energy nonregression constraints to both matched '
        'response-model arms. Independently validate a damped rank2 secant correction using two '
        'measured HALF-step actors, checking their FULL-step actors without fitting them. If total '
        'normalized shape error improves20percent and155deg peak error is no worse on both, refit '
        'using all4 completed actors. Compare unchanged-response and validated-response half/full '
        'proposals,4 maximum actual64-path candidates. Margin,trace,metric responses are corrected '
        'consistently, still predictions only. Preserve114 centers,13 directions,all physical and '
        'actual research selection criteria; no PPO,label fitting,runtime rule or final adoption.')
    started=time.perf_counter();pool=None;outcomes=[];jobs=[]
    progress(stage,True,'补齐总反转等采用条件对应的优化约束，并用未参与拟合的完整候选验证响应校准。')
    try:
        paired=[(indexed[f'axis{i}_-1'],indexed[f'axis{i}_+1']) for i in range(n)]
        original,eligibility=pathwise_response(source,baseline,retained,paired)
        fitted,fit=calibrate(original,baseline,retained,trust,controls)
        oldcheck=validation(original,source,baseline,retained,[c for _,c in held])
        newcheck=validation(fitted,source,baseline,retained,[c for _,c in held])
        olderror=sum(c['shape_prediction_error'] for c in oldcheck)
        newerror=sum(c['shape_prediction_error'] for c in newcheck)
        peak_pass=all(abs(a['path14_predicted_deficits'][5]-a['path14_actual_deficits'][5])<=
                      abs(b['path14_predicted_deficits'][5]-b['path14_actual_deficits'][5]) for a,b in zip(newcheck,oldcheck))
        accepted=bool(newerror<.8*olderror and peak_pass)
        write_json(stage/'validation.json',dict(fit=fit,previous=oldcheck,calibrated=newcheck,
            previous_error=olderror,calibrated_error=newerror,peak_pass=peak_pass,accepted=accepted))
        print('FULL_SECANT_VALIDATION',olderror,newerror,peak_pass,accepted,flush=True)
        arms=[('raw',original)]
        if accepted:
            fitted,fit=calibrate(original,baseline,retained,trust,controls+held)
            write_json(stage/'fit.json',fit);arms.append(('calibrated',fitted))
        for arm,response in arms:
            step,report=solve_response(baseline,retained,response,trust,True,min(started+240,time.perf_counter()+100),aggregate_guards=True)
            write_json(stage/(arm+'_optimizer.json'),report)
            print('FULL_SECANT_SOLVER',arm,report['message'],report['predicted_objective'],report['maximum_function_change'],flush=True)
            if np.max(abs(step))>1e-6:
                for scale in (.5,1.):
                    u=scale*step
                    if not any(np.max(abs(trust@(u-q)))<1e-8 for _,q in jobs):jobs.append((f'{arm}_joint_{scale}',u))
        refs,curved={},{}
        for r in baseline['records']:
            with np.load(r['trace']) as z:refs[r['path']]=z['reference'].copy()
            curved[r['path']]=r['curved_path']
        for tag,u in jobs:torch.save(candidate_metadata(metadata,directions,u),stage/(tag+'.pt'))
        if jobs:
            pool=mp.get_context('spawn').Pool(min(4,len(jobs)),initialize,(stage,metadata,directions,refs,curved,retained,started+350))
            for c in pool.imap_unordered(evaluate,jobs):
                outcomes.append(c)
                write_json(stage/'results.json',dict(candidates=outcomes,adopted=False,goal_complete=False))
                print('FULL_SECANT_ACTUAL',c['tag'],c['summary'],flush=True)
            pool.close();pool.join();pool=None
        selected=actual_selection(outcomes,baseline,retained)
        result=dict(candidates=outcomes,adopted=False,goal_complete=False)
        if selected:
            result['best']=selected['tag']
            torch.save(candidate_metadata(metadata,directions,selected['parameters']),stage/'research_candidate.pt')
        write_json(stage/'results.json',result)
        receipt(stage,dict(status='completed_full_pool_secant_guard_learning',best=selected['tag'] if selected else None,
            full64_candidates=len(outcomes),secant_validated=accepted,wall_s=time.perf_counter()-started,
            adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_full_pool_secant_guard_learning',error=str(exc),
            full64_candidates=len(outcomes),wall_s=time.perf_counter()-started,adopted=False,goal_complete=False));raise
    finally:
        progress(stage,False,'继续按实际逐角与总形状、全池成功及巡航筛选学习更新，原目标未完成。')


if __name__=='__main__':mp.freeze_support();main()
