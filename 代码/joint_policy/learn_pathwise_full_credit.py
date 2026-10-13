"""Pathwise complete-consequence derivatives from cached full-policy trials."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse
import copy
import multiprocessing as mp
import time
import numpy as np
import torch
from learn_full_pool_compensation import consistent_response, solve_response, progress
from learn_full_pool_joint_credit import initialize, evaluate, candidate_metadata
from learn_trace_resolved_joint_credit import signed_channels, read_rows, trace_deficits
from learn_retained_joint_growth import growth_deficits
from learn_absolute_shape_joint_credit import absolute_shape_response
from study_io import BASE, register, receipt, write_json, sha
from search_complete_return import read


def pathwise_response(source, baseline, retained, paired):
    base = growth_deficits(baseline, retained)
    n = len(paired)
    matrix = np.zeros((len(base), n))
    bm = np.array([r['corridor']['margin_lower_mm'] for r in baseline['records']])
    margin_jac = np.zeros((len(bm), n))
    traces, audit = {}, []
    offset = 0
    for index, r in enumerate(baseline['records']):
        size = 9 if 'shape' in r else 2
        jac = {k: [] for k in ('incoming', 'outgoing', 'omega', 'turn')}
        if 'shape' in r:
            ref, rows = read_rows(source, r)
            b = signed_channels(ref, rows)
        modes = []
        def route_value(record):
            assert record['success']
            # The other records remain the completed baseline. This supplies
            # only this route's complete consequence, never failed-prefix time.
            merged = dict(baseline, records=[record if k==index else t for k,t in enumerate(baseline['records'])], summary=dict(baseline['summary']))
            if r['curved_path']:
                merged['summary']['smooth_curved_cruise99_s'] += record['smooth_cruise99_s']-r['smooth_cruise99_s']
            return growth_deficits(merged, retained)
        for i, (lo, hi) in enumerate(paired):
            low, high = lo['records'][index], hi['records'][index]
            lok, hik = low['success'], high['success']
            mode = 'central' if lok and hik else 'positive' if hik else 'negative' if lok else 'frozen'
            modes.append(mode)
            def diff(a, l, h):
                return (h-l)/2 if mode=='central' else h-a if mode=='positive' else a-l if mode=='negative' else np.zeros_like(a)
            lv, hv = route_value(low) if lok else base, route_value(high) if hik else base
            delta = diff(base, lv, hv)
            matrix[offset:offset+size, i] = delta[offset:offset+size]
            matrix[-1, i] += delta[-1]
            margin_jac[index, i] = diff(bm[index], low['corridor']['margin_lower_mm'], high['corridor']['margin_lower_mm'])
            if 'shape' in r:
                lc = signed_channels(ref, read_rows(source, low)[1], b['grid'], np.arange(len(rows))) if lok else b
                hc = signed_channels(ref, read_rows(source, high)[1], b['grid'], np.arange(len(rows))) if hik else b
                for k in jac:
                    jac[k].append(diff(b[k], lc[k], hc[k]))
        if 'shape' in r:
            traces[r['path']] = dict(base=b, jac={k:np.column_stack(v) for k,v in jac.items()})
        audit.append(dict(path=r['path'], modes=modes))
        offset += size
    # If neither sign completes any one route, that coordinate cannot support
    # a whole-pool update. Do not silently turn missing evidence into zero risk.
    modes=['frozen' if any(a['modes'][i]=='frozen' for a in audit) else 'pathwise_complete' for i in range(n)]
    bounds=[(0.,0.) if mode=='frozen' else (-1.,1.) for mode in modes]
    return (base,matrix,bm,margin_jac,bounds,modes,traces),audit


def validation(response_data, source, baseline, retained, candidates):
    base,matrix,_,_,_,_,traces=response_data
    result=[]
    for c in candidates:
        u=np.asarray(c['parameters'])
        pred,j=trace_deficits(u,base,matrix,traces,baseline,retained)
        actual=growth_deficits(c,retained)
        indices=np.array([9*p+k for p in range(16) for k in range(1,7)])
        pa,_=absolute_shape_response(pred,j,baseline['records'],retained)
        result.append(dict(tag=c['tag'],shape_prediction_error=float(np.linalg.norm(pred[indices]-actual[indices])),
            predicted_absolute_objective=float(np.maximum(pa,0.)@np.maximum(pa,0.)),
            actual_absolute_objective=c['absolute_objective'],
            path14_predicted_deficits=pred[126:135].tolist(),path14_actual_deficits=actual[126:135].tolist()))
    return result


def actual_selection(outcomes, baseline, retained):
    base=growth_deficits(baseline,retained)
    good=[]
    guard=np.array([9*p+k for p in range(16) for k in range(1,8)])
    for c in outcomes:
        if not c['primary_preserved']:
            continue
        v=growth_deficits(c,retained)
        c['research_corner_guards_pass']=bool(np.all(v[guard]<=np.maximum(base[guard],0.)+1e-5))
        b,a=baseline['summary'],c['summary']
        c['research_acceptable']=bool(c['research_corner_guards_pass'] and c['absolute_objective']<baseline['absolute_objective']
            and a['complete_time_s']<=b['complete_time_s']+.02 and a['curved_cruise99_s']>=b['curved_cruise99_s']-.05
            and a['reverse_sum_deg']<b['reverse_sum_deg'] and a['exit_sum_mm']<=b['exit_sum_mm'] and a['energy_sum']<=b['energy_sum'])
        if c['research_acceptable']:good.append(c)
    return min(good,key=lambda c:c['absolute_objective']) if good else None


def main():
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_full_pool_joint_credit_learning_20260919'
    check=BASE/'canonical_gate_band_full_pool_compensation_learning_20260919'
    parent=BASE/'canonical_gate_band_iterative_motion_constraint_learning_20260919/research_candidate.pt'
    assert sha(parent)=='5ab47111bfcec3a511903e20f1249c91bbfb65e5ab4cd1a3d2b4a493a979c26a'
    data=read(source/'baseline.json');baseline,retained=data['baseline'],data['retained']
    indexed={c['tag']:c for c in read(source/'results.json')['candidates']}
    with np.load(source/'parameter_space.npz') as z:directions,trust=z['directions'].copy(),z['trust'].copy()
    paired=[(indexed[f'axis{i}_-1'],indexed[f'axis{i}_+1']) for i in range(len(directions))]
    held=read(check/'results.json')['candidates']
    metadata=torch.load(parent,map_location='cpu',weights_only=False)
    stage=register('canonical_gate_band_pathwise_full_credit_learning_20260919',
        [__file__,BASE/'learn_full_pool_compensation.py',BASE/'learn_full_pool_joint_credit.py',
         source/'results.json',source/'receipt.json',source/'baseline.json',source/'parameter_space.npz',
         check/'results.json',check/'receipt.json',parent],
        dict(wall_s=240,new_full64_candidates=2,reused_full64_response_candidates=16,
             held_response_checks=2,optimizer_s=90,workers=2),
        'The sign-compensation proposal completed64/64 but155deg exit/energy/peak worsened despite '
        'their predicted guards. Test route-specific COMPLETE-policy credit. A failure on path6 '
        'does not erase a completed155deg route from the same actor evaluation. Use central responses '
        'where both complete routes exist, one-sided where only one route completes, and zero where '
        'neither completes. Build route time and smooth cruise responses only from actual complete '
        'routes; never count a failed prefix. Match scalar/margin/trace eligibility per route. '
        'First check predictions on the two measured joint candidates, without fitting to them. '
        'Proceed to at most two full64 proposals only if the matched shape prediction error decreases. '
        'The learned actor,8 directions,114 centers, action trust and physical executor are unchanged. '
        'This tests credit-data eligibility, not estimator bias or PPO. No final adoption or sealed test.')
    started=time.perf_counter();pool=None;outcomes=[]
    progress(stage,True,'按路径使用完整闭环响应，先验证已测联合轨迹预测，再执行至多两个新候选。')
    try:
        previous=consistent_response(source,baseline,retained,paired)
        response_data,audit=pathwise_response(source,baseline,retained,paired)
        oldcheck=validation(previous,source,baseline,retained,held)
        newcheck=validation(response_data,source,baseline,retained,held)
        olderror=sum(x['shape_prediction_error'] for x in oldcheck)
        newerror=sum(x['shape_prediction_error'] for x in newcheck)
        write_json(stage/'validation.json',dict(previous=oldcheck,pathwise=newcheck,previous_error=olderror,
                   pathwise_error=newerror,prediction_improved=newerror<olderror,fit_to_checks=False))
        write_json(stage/'response_eligibility.json',audit)
        print('PATHWISE_VALIDATION',olderror,newerror,flush=True)
        step=np.zeros(len(directions))
        if newerror<olderror:
            step,report=solve_response(baseline,retained,response_data,trust,True,started+120)
            write_json(stage/'optimizer.json',report)
            print('PATHWISE_SOLVER',report['message'],report['predicted_objective'],report['maximum_function_change'],flush=True)
        jobs=[(f'pathwise_joint_{s}',s*step) for s in (.5,1.)] if np.max(abs(step))>1e-6 else []
        refs,curved={},{}
        for r in baseline['records']:
            with np.load(r['trace']) as z:refs[r['path']]=z['reference'].copy()
            curved[r['path']]=r['curved_path']
        for tag,u in jobs:torch.save(candidate_metadata(metadata,directions,u),stage/(tag+'.pt'))
        if jobs:
            pool=mp.get_context('spawn').Pool(2,initialize,(stage,metadata,directions,refs,curved,retained,started+230))
            for c in pool.imap_unordered(evaluate,jobs):
                outcomes.append(c)
                write_json(stage/'results.json',dict(candidates=outcomes,adopted=False,goal_complete=False))
                print('PATHWISE_ACTUAL',c['tag'],c['summary'],flush=True)
            pool.close();pool.join();pool=None
        selected=actual_selection(outcomes,baseline,retained)
        result=dict(candidates=outcomes,adopted=False,goal_complete=False)
        if selected:
            result['best']=selected['tag']
            torch.save(candidate_metadata(metadata,directions,selected['parameters']),stage/'research_candidate.pt')
        write_json(stage/'results.json',result)
        receipt(stage,dict(status='completed_pathwise_full_credit_learning',best=selected['tag'] if selected else None,
            full64_candidates=len(outcomes),wall_s=time.perf_counter()-started,adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_pathwise_full_credit_learning',error=str(exc),
            wall_s=time.perf_counter()-started,adopted=False,goal_complete=False));raise
    finally:
        progress(stage,False,'依据逐路径信用预测和真实64路径结果，修正活跃形状及几何约束的联合学习空间。')


if __name__=='__main__':mp.freeze_support();main()
