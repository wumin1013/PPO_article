"""Use recorded XY/V/A consequences to constrain joint innovation learning."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import copy
import itertools
import multiprocessing as mp
import time
import numpy as np
import torch
from learn_common_horizon_constraint_credit import distances
from learn_innovation_local_compensation import initialize,evaluate
from learn_full_pool_joint_credit import candidate_metadata
from decision_innovation_policy import metadata as innovation_metadata
from learn_pathwise_full_credit import actual_selection,validation
from learn_full_pool_compensation import solve_response,progress
from study_io import BASE,register,receipt,write_json,sha
from search_complete_return import read


def load_pairs(path):
    local=BASE/'canonical_gate_band_current_bend_capacity_learning_20260919'
    fine=BASE/'canonical_gate_band_fine_full_policy_credit_learning_20260919'
    global_source=BASE/'canonical_gate_band_decision_innovation_joint_learning_20260919'
    refresh=BASE/'canonical_gate_band_innovation_local_compensation_learning_20260919'
    def indexed(folder,filename='results.json'):
        cs=read(folder/filename)['candidates']
        result={}
        for c in cs:
            if any(r['path']==path for r in c['records']):
                r=next(r for r in c['records'] if r['path']==path)
                with np.load(folder/r['trace']) as z:result[c['tag']]=z['rows'][:,:6].copy()
        return result
    a,b,c,d=indexed(local),indexed(fine),indexed(global_source),indexed(refresh,'route_refresh.json')
    pairs=[]
    for i in range(18):
        if f'fine_axis{i}_-1' in b:
            pairs.append((b[f'fine_axis{i}_-1'],b[f'fine_axis{i}_+1'],.25))
        else:pairs.append((a[f'axis{i}_-1'],a[f'axis{i}_+1'],1.))
    amps=read(refresh/'response_eligibility.json')['route_amplitudes']
    for i in range(9):
        matching=[r for r in amps if r['axis']==i and r['path']==path]
        if matching:
            amp=matching[0]['amplitude']
            pairs.append((d[f'path31_amp{amp}_axis{i}_-1'],d[f'path31_amp{amp}_axis{i}_+1'],amp))
        else:pairs.append((c[f'axis{i}_-1'],c[f'axis{i}_+1'],1.))
    return pairs


def motion_model(reference,base,pairs,count):
    base=base[:count,:6]
    jac=[];modes=[];unsupported=[]
    for i,(lo,hi,amp) in enumerate(pairs):
        lok,hik=len(lo)>=count,len(hi)>=count
        if lok and hik:delta=(hi[:count]-lo[:count])/(2*amp);mode='central_observed'
        elif hik:delta=(hi[:count]-base)/amp;mode='positive_observed'
        elif lok:delta=(base-lo[:count])/amp;mode='negative_observed'
        else:delta=np.zeros_like(base);mode='unobserved_frozen';unsupported.append(i)
        jac.append(delta);modes.append(mode)
    jac=np.stack(jac,-1)
    horizons=(0.,.005,.010,.020)
    forecast=np.concatenate([base[:,:2]+t*base[:,2:4]+.5*t*t*base[:,4:6] for t in horizons])
    derivative=np.concatenate([jac[:,:2]+t*jac[:,2:4]+.5*t*t*jac[:,4:6] for t in horizons])
    nominal=distances(reference,forecast)[0]
    active=nominal>.8
    forecast,derivative,nominal=forecast[active],derivative[active],nominal[active]
    upper=np.maximum(nominal,1.20)
    def fun(u):
        distance,_=distances(reference,forecast+np.einsum('tcn,n->tc',derivative,u))
        return (upper-distance)/.01
    def gradient(u):
        _,normal=distances(reference,forecast+np.einsum('tcn,n->tc',derivative,u))
        return -np.einsum('tc,tcn->tn',normal,derivative)/.01
    constraints=[dict(type='ineq',fun=fun,jac=gradient)]
    if unsupported:
        constraints.append(dict(type='eq',fun=lambda u:u[unsupported],jac=lambda u:np.eye(len(u))[unsupported]))
    return constraints,base,jac,dict(observed_ms=count,modes=modes,unsupported=unsupported,
        forecast_horizons_ms=[0,5,10,20],training_target_mm=1.20,
        nominal_violations_are_preserved_not_assumed_feasible=True,active_motion_constraints=int(active.sum()))


def main():
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_innovation_local_compensation_learning_20260919'
    local_source=BASE/'canonical_gate_band_current_bend_capacity_learning_20260919'
    model=torch.load(source/'response_model.pt',map_location='cpu',weights_only=False)
    response,baseline,retained=model['response'],model['baseline'],model['retained']
    parent=torch.load(local_source/'zero_extension.pt',map_location='cpu',weights_only=False)
    with np.load(local_source/'parameter_space.npz') as z:directions,local_trust=z['directions'].copy(),z['trust'].copy()
    held=read(source/'results.json')['candidates']
    r=next(r for r in baseline['records'] if r['path']==31)
    with np.load(r['trace']) as z:reference,base=z['reference'].copy(),z['rows'].copy()
    observed=[]
    for c in held:
        rr=next(r for r in c['records'] if r['path']==31)
        with np.load(source/rr['trace']) as z:observed.append((c,z['rows'].copy()))
    count=min([len(base)]+[len(rows) for _,rows in observed])
    stage=register('canonical_gate_band_innovation_motion_guard_learning_20260919',
        [__file__,BASE/'learn_common_horizon_constraint_credit.py',BASE/'learn_innovation_local_compensation.py',
         BASE/'decision_innovation_policy.py',BASE/'learn_full_pool_compensation.py',
         source/'response_model.pt',source/'results.json',source/'receipt.json',source/'route_refresh.json',
         local_source/'zero_extension.pt',local_source/'parameter_space.npz',
         BASE/'canonical_gate_band_fine_full_policy_credit_learning_20260919/results.json'],
        dict(wall_s=360,workers=2,reused_joint_checks=2,new_full64_candidates=2,
             new_response_rollouts=0,constrained_route=31,observed_ms=count),
        'Both new27-parameter proposals failed only on path31, even though scalar '
        'minimum-margin responses predicted feasibility. Test the previously useful '
        'common-horizon physical-state constraint mechanism on THIS new representation. '
        'Reuse all current local/global/route-refreshed axes; no repeated simulation. '
        'Compose physicalXY,V,A on the observed horizon shared by the two failed '
        'joint checks; allow only actual observed sides at this horizon and freeze '
        'unsupported parameters. First compare actual and predicted prefix corridor '
        'exposure for the two joint checks without fitting. Use0/5/10/20ms kinematic '
        'forecasts as TRAINING constraints only, with max(nominal_distance,1.20mm) '
        'bounds; these predictions are not real future trajectories. Require the '
        'motion constraints to reject both failed joint proposals before learning '
        'two half/full updates under every existing shape/time/cruise guard. '
        'Full64 actual coverage/braking/rest and1.25mm decide eligibility. Failed '
        'prefixes provide constraint evidence only, never completion or efficiency.')
    started=time.perf_counter();pool=None;outcomes=[]
    progress(stage,True,'复用真实XY、V、A响应，验证并约束路径31的运动趋势，再执行完整64路径候选。')
    try:
        write_json(stage/'baseline.json',dict(baseline=baseline,retained=retained))
        constraints,common,jac,audit=motion_model(reference,base,load_pairs(31),count)
        checks=[]
        for c,rows in observed:
            u=np.asarray(c['parameters']);pred=common[:,:2]+np.einsum('tcn,n->tc',jac[:,:2],u)
            checks.append(dict(tag=c['tag'],actual_common_max_distance_mm=float(distances(reference,rows[:count,:2])[0].max()),
                predicted_common_max_distance_mm=float(distances(reference,pred)[0].max()),
                maximum_xy_prediction_error_mm=float(np.linalg.norm(pred-rows[:count,:2],axis=1).max()),
                motion_guard_minimum=float(constraints[0]['fun'](u).min())))
        justified=all(c['motion_guard_minimum']<-.1 for c in checks)
        write_json(stage/'motion_credit_validation.json',dict(checks=checks,audit=audit,
            rejects_both_failed_proposals=justified,fit_to_joint_checks=False))
        print('INNOVATION_MOTION_VALIDATION',checks,justified,flush=True)
        jobs=[]
        if justified:
            global_trust=np.zeros((24,9));row=0
            for head in range(3):
                for sign in itertools.product((-1.,1.),repeat=3):
                    global_trust[row,np.arange(3)*3+head]=sign;row+=1
            trust=np.zeros((len(local_trust)+24,27));trust[:len(local_trust),:18]=local_trust
            trust[len(local_trust):,18:]=global_trust
            step,report=solve_response(baseline,retained,response,trust,True,
                started+120,aggregate_guards=True,additional_constraints=constraints)
            write_json(stage/'optimizer.json',report)
            print('INNOVATION_MOTION_SOLVER',report['message'],report['predicted_objective'],flush=True)
            if abs(step).max()>1e-6:jobs=[(f'motion_guard_joint_{s}',s*step) for s in (.5,1.)]
        refs,curved={},{}
        for r in baseline['records']:
            with np.load(r['trace']) as z:refs[r['path']]=z['reference'].copy()
            curved[r['path']]=r['curved_path']
        if jobs:
            pool=mp.get_context('spawn').Pool(2,initialize,(stage,parent,directions,refs,curved,retained,started+345))
            for c in pool.imap_unordered(evaluate,jobs):
                outcomes.append(c)
                print('INNOVATION_MOTION_ACTUAL',c['tag'],c['summary'],flush=True)
                write_json(stage/'results.json',dict(candidates=outcomes,adopted=False,goal_complete=False))
            pool.close();pool.join();pool=None
        complete=[c for c in outcomes if c['summary']['successes']==64]
        write_json(stage/'prediction_check.json',validation(response,stage,baseline,retained,complete))
        selected=actual_selection(outcomes,baseline,retained)
        result=dict(candidates=outcomes,adopted=False,goal_complete=False)
        for c in outcomes:
            u=c['parameters'];md=innovation_metadata(candidate_metadata(parent,directions,u[:18]),u[18:])
            torch.save(md,stage/(c['tag']+'.pt'))
            if c is selected:
                result['best']=c['tag'];torch.save(md,stage/'research_candidate.pt')
        write_json(stage/'results.json',result)
        receipt(stage,dict(status='completed_innovation_motion_guard_learning',motion_guard_validated=justified,
            full64_candidates=len(outcomes),best=selected['tag'] if selected else None,
            wall_s=time.perf_counter()-started,adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_innovation_motion_guard_learning',error=str(exc),
            full64_candidates=len(outcomes),wall_s=time.perf_counter()-started,adopted=False,goal_complete=False));raise
    finally:
        progress(stage,False,'根据真实运动约束学习结果继续修正三动作协调；原研究目标和完整验收要求不变。')


if __name__=='__main__':mp.freeze_support();main()
