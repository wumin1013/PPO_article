"""Combine new observed-innovation credit with cached local joint corrections."""
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
import learn_decision_innovation_joint as innovation
from decision_innovation_policy import metadata as innovation_metadata
from learn_full_pool_joint_credit import candidate_metadata
from learn_pathwise_full_credit import pathwise_response,actual_selection,validation
from learn_full_pool_compensation import solve_response,progress
from study_io import BASE,register,receipt,write_json,sha
from search_complete_return import read


LOCAL={}


def initialize(stage,parent,directions,references,curved,retained,deadline):
    innovation.initialize(stage,parent,references,curved,retained,deadline)
    LOCAL.update(parent=parent,directions=directions)


def evaluate(job):
    tag,u=job
    innovation.WORK['parent']=candidate_metadata(LOCAL['parent'],LOCAL['directions'],np.asarray(u[:18]))
    result=innovation.evaluate((tag,u[18:]))
    result['parameters']=np.asarray(u).tolist()
    write_json(innovation.WORK['stage']/(tag+'.json'),result)
    return result


def combine(local,global_response):
    assert np.max(abs(local[0]-global_response[0]))<1e-10
    assert np.array_equal(local[2],global_response[2])
    traces=copy.deepcopy(local[6])
    for path,t in traces.items():
        other=global_response[6][path]
        for k in t['jac']:
            assert np.array_equal(t['base'][k],other['base'][k])
            t['jac'][k]=np.column_stack([t['jac'][k],other['jac'][k]])
    return (local[0],np.column_stack([local[1],global_response[1]]),local[2],
            np.column_stack([local[3],global_response[3]]),local[4]+global_response[4],
            local[5]+global_response[5],traces)


def scale_path_response(model,baseline,paired,amplitudes):
    offsets={};offset=0
    for index,r in enumerate(baseline['records']):
        size=9 if 'shape' in r else 2
        offsets[r['path']]=(index,offset,size);offset+=size
    for (path,axis),amplitude in amplitudes.items():
        index,offset,size=offsets[path]
        b=baseline['records'][index];lo,hi=[c['records'][index] for c in paired[axis]]
        assert 'shape' not in b, 'Refresh currently targets only the non-isolated blocking route.'
        model[1][offset:offset+size,axis]/=amplitude
        model[3][index,axis]/=amplitude
        if b['curved_path']:
            if lo['success'] and hi['success']:
                delta=(hi['smooth_cruise99_s']-lo['smooth_cruise99_s'])/2
            elif hi['success']:delta=hi['smooth_cruise99_s']-b['smooth_cruise99_s']
            elif lo['success']:delta=b['smooth_cruise99_s']-lo['smooth_cruise99_s']
            else:delta=0.
            model[1][-1,axis]+=-5*delta*(1/amplitude-1)


def main():
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_decision_innovation_joint_learning_20260919'
    local_source=BASE/'canonical_gate_band_current_bend_capacity_learning_20260919'
    fine=BASE/'canonical_gate_band_fine_full_policy_credit_learning_20260919'
    data=read(source/'baseline.json');baseline,retained=data['baseline'],data['retained']
    parent=torch.load(local_source/'zero_extension.pt',map_location='cpu',weights_only=False)
    local=torch.load(fine/'response_model.pt',map_location='cpu',weights_only=False)['response']
    with np.load(local_source/'parameter_space.npz') as z:directions,local_trust=z['directions'].copy(),z['trust'].copy()
    cached=read(source/'results.json')['candidates']
    indexed={c['tag']:c for c in cached}
    for c in cached:
        for r in c['records']:r['trace']=str(source/r['trace'])
    paired=[(indexed[f'axis{i}_-1'],indexed[f'axis{i}_+1']) for i in range(9)]
    eligibility=read(source/'response_eligibility.json')
    blockers=[(r['path'],i) for r in eligibility for i,m in enumerate(r['modes']) if m=='frozen']
    assert blockers==[(31,1),(31,3),(31,6),(31,7)]
    stage=register('canonical_gate_band_innovation_local_compensation_learning_20260919',
        [__file__,BASE/'decision_innovation_policy.py',BASE/'learn_decision_innovation_joint.py',
         BASE/'learn_full_pool_joint_credit.py',BASE/'learn_pathwise_full_credit.py',
         BASE/'learn_full_pool_compensation.py',source/'results.json',source/'receipt.json',
         source/'response_eligibility.json',source/'baseline.json',local_source/'zero_extension.pt',
         local_source/'parameter_space.npz',fine/'response_model.pt',fine/'validation.json',fine/'receipt.json'],
        dict(wall_s=600,workers=4,reused_global_full64_candidates=18,reused_local_dimensions=18,
             blocking_path=31,blocking_axes=[1,3,6,7],refresh_amplitudes=[.125,.03125],
             maximum_new_single_route_runs=16,maximum_new_full64_candidates=2,total_joint_dimensions=27),
        'The new9-feature global innovation actor has actual large bend responses but '
        'no guarded update. Four axes lack either complete sign only on route31. Refresh '
        'ONLY that route at1/8 amplitude, and1/32 only if both signs still fail; reuse '
        'every completed other-route response. Apply route-specific finite-amplitude '
        'normalization, including exact smooth-cruise derivative contribution. Missing '
        'complete-route credit remains frozen. Combine the resulting9-column trajectory '
        'response with the already validated18-column local state-function response at '
        'the SAME physically identical5ab parent. This permits local corrections to '
        'compensate global innovation effects; neither component is adopted by itself. '
        'Jointly optimize all27 coefficients with unchanged shape,geometry,time,cruise '
        'guards; evaluate half/full proposals on all64 routes. The guide and20/1ms '
        'hold/execution are unchanged. No failed-prefix efficiency, repeated completed '
        'experiments, PPO claim, path-specific runtime behavior or acceptance relaxation.')
    started=time.perf_counter();pool=None;refresh=[];outcomes=[];amplitudes={}
    progress(stage,True,'仅补路径31缺失的完整信用，将历史特征和已有局部三动作函数联合求解，再验完整64路径。')
    try:
        write_json(stage/'baseline.json',data)
        refs,curved={},{}
        for r in baseline['records']:
            with np.load(r['trace']) as z:refs[r['path']]=z['reference'].copy()
            curved[r['path']]=r['curved_path']
        pool=mp.get_context('spawn').Pool(4,initialize,(stage,parent,directions,{31:refs[31]},
            {31:curved[31]},retained,started+580))
        pending=[1,3,6,7]
        record_index=next(i for i,r in enumerate(baseline['records']) if r['path']==31)
        for amplitude in (.125,.03125):
            jobs=[(f'path31_amp{amplitude}_axis{i}_{s:+d}',np.r_[np.zeros(18),s*amplitude*np.eye(9)[i]])
                  for i in pending for s in (-1,1)]
            batch={}
            for c in pool.imap_unordered(evaluate,jobs):
                refresh.append(c);batch[c['tag']]=c
                write_json(stage/'route_refresh.json',dict(candidates=refresh))
                print('INNOVATION_ROUTE_CREDIT',c['tag'],c['records'][0]['success'],
                      c['records'][0]['corridor']['margin_lower_mm'],flush=True)
            remaining=[]
            for i in pending:
                rlo,rhi=[batch[f'path31_amp{amplitude}_axis{i}_{s:+d}']['records'][0] for s in (-1,1)]
                for candidate,r in zip(paired[i],(rlo,rhi)):
                    r['trace']=str(stage/r['trace'])
                    candidate['records'][record_index]=r
                amplitudes[(31,i)]=amplitude
                if not rlo['success'] and not rhi['success']:remaining.append(i)
            pending=remaining
            if not pending:break
        pool.close();pool.join();pool=None
        global_response,audit=pathwise_response(stage,baseline,retained,paired)
        scale_path_response(global_response,baseline,paired,amplitudes)
        response=combine(local,global_response)
        write_json(stage/'response_eligibility.json',dict(route_modes=audit,still_missing_axes=pending,
            route_amplitudes=[dict(path=p,axis=i,amplitude=a) for (p,i),a in amplitudes.items()]))
        torch.save(dict(response=response,baseline=baseline,retained=retained),stage/'response_model.pt')
        global_trust=np.zeros((24,9));row=0
        for head in range(3):
            for sign in itertools.product((-1.,1.),repeat=3):
                global_trust[row,np.arange(3)*3+head]=sign;row+=1
        trust=np.zeros((len(local_trust)+24,27))
        trust[:len(local_trust),:18]=local_trust;trust[len(local_trust):,18:]=global_trust
        step,report=solve_response(baseline,retained,response,trust,True,
            min(started+350,time.perf_counter()+90),aggregate_guards=True)
        write_json(stage/'optimizer.json',report)
        print('INNOVATION_LOCAL_SOLVER',report['message'],report['predicted_objective'],
              report['maximum_function_change'],flush=True)
        jobs=[(f'innovation_local_joint_{s}',s*step) for s in (.5,1.)] if abs(step).max()>1e-6 else []
        if jobs:
            pool=mp.get_context('spawn').Pool(2,initialize,(stage,parent,directions,refs,curved,retained,started+580))
            for c in pool.imap_unordered(evaluate,jobs):
                outcomes.append(c)
                write_json(stage/'results.json',dict(candidates=outcomes,adopted=False,goal_complete=False))
                print('INNOVATION_LOCAL_ACTUAL',c['tag'],c['summary'],flush=True)
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
        receipt(stage,dict(status='completed_innovation_local_compensation_learning',single_route_runs=len(refresh),
            full64_candidates=len(outcomes),best=selected['tag'] if selected else None,
            wall_s=time.perf_counter()-started,adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_innovation_local_compensation_learning',error=str(exc),
            single_route_runs=len(refresh),full64_candidates=len(outcomes),wall_s=time.perf_counter()-started,
            adopted=False,goal_complete=False));raise
    finally:
        progress(stage,False,'依据历史特征与局部参数的联合补偿结果继续实现平滑形状及持续巡航。')


if __name__=='__main__':mp.freeze_support();main()
