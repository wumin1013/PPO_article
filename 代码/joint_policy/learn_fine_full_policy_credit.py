"""Refresh six nonlinear current policy directions at quarter amplitude."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import copy
import multiprocessing as mp
import time
import numpy as np
import torch
from learn_pathwise_full_credit import pathwise_response,actual_selection
from learn_full_pool_compensation import solve_response,progress
from learn_full_pool_joint_credit import initialize,evaluate,candidate_metadata
from learn_trace_resolved_joint_credit import trace_deficits
from learn_retained_joint_growth import growth_deficits
from study_io import BASE,register,receipt,write_json
from search_complete_return import read


def choose_axes(baseline,paired,held):
    keys=['in_edge_outward_mm','out_edge_outward_mm','full_physical_reverse_deg',
          'full_normalized_energy','full_spatial_peak_curvature_mm_inv','apex_speed_mm_s']
    scales=np.array([.002,.005,.5,.02,.2,2.])
    weights=np.mean(np.abs([c['parameters'] for c in held]),axis=0)
    scores=[]
    for i,(lo,hi) in enumerate(paired):
        nonlinear=[];missing=0
        for b,l,h in zip(baseline['records'],lo['records'],hi['records']):
            if not l['success'] or not h['success']:
                missing+=1
                continue
            if 'shape' in b:
                nonlinear.extend([(h['shape'][k]+l['shape'][k]-2*b['shape'][k])/s for k,s in zip(keys,scales)])
        curvature=float(np.linalg.norm(nonlinear))
        score=float(weights[i]*(curvature+missing))
        scores.append(dict(axis=i,score=score,shape_second_difference=curvature,missing_complete_pairs=missing,
                           proposal_weight=float(weights[i])))
    scores.sort(key=lambda r:r['score'],reverse=True)
    return [r['axis'] for r in scores[:6]],scores


def validate(model,baseline,retained,held):
    ids=np.array([9*p+k for p in range(16) for k in range(1,7)])
    output=[]
    for c in held:
        actual=growth_deficits(c,retained)
        predicted=trace_deficits(np.asarray(c['parameters']),model[0],model[1],model[6],baseline,retained)[0]
        output.append(dict(tag=c['tag'],shape_error=float(np.linalg.norm(predicted[ids]-actual[ids])),
            reverse_error=float(np.linalg.norm(predicted[3:144:9]-actual[3:144:9])),
            peak14_error=float(predicted[131]-actual[131])))
    return output


def main():
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_current_bend_capacity_learning_20260919'
    latest=BASE/'canonical_gate_band_joint_interaction_subspace_learning_20260919'
    data=read(source/'baseline.json');baseline,retained=data['baseline'],data['retained']
    records={c['tag']:c for c in read(source/'results.json')['candidates']}
    held=[c for c in records.values() if c['tag'].startswith('bend_joint')]+read(latest/'results.json')['candidates']
    for c in records.values():
        if c['tag'].startswith('axis'):
            for r in c['records']:r['trace']=str(source/r['trace'])
    paired=[(records[f'axis{i}_-1'],records[f'axis{i}_+1']) for i in range(18)]
    axes,scores=choose_axes(baseline,paired,held)
    metadata=torch.load(source/'zero_extension.pt',map_location='cpu',weights_only=False)
    with np.load(source/'parameter_space.npz') as z:directions,trust=z['directions'].copy(),z['trust'].copy()
    stage=register('canonical_gate_band_fine_full_policy_credit_learning_20260919',
        [__file__,BASE/'learn_pathwise_full_credit.py',BASE/'learn_full_pool_compensation.py',
         BASE/'learn_full_pool_joint_credit.py',source/'results.json',source/'receipt.json',
         source/'baseline.json',source/'parameter_space.npz',source/'zero_extension.pt',latest/'results.json',latest/'receipt.json'],
        dict(wall_s=420,workers=4,refreshed_axes=axes,amplitude=.25,new_response_candidates=12,new_joint_candidates=2,
             reused_response_candidates=24,held_joint_candidates=4),
        'Complete scalar credit was worse than signed-trace credit on all current held actors. '
        'Do not reuse that failed estimator. Select six current coefficient axes with strongest '
        'measured second differences/incomplete side pairs, weighted only by proposed coefficient '
        'magnitudes (not held outcome labels). Measure +/-0.25 instead of +/-1 on full64 paths; '
        'reuse the other24 response actors. Scale derivatives by actual parameter amplitude and '
        'keep per-route complete-only eligibility consistent. Compare the resulting response '
        'model on the same four unfitted complete actors. Require20percent lower shape error '
        'and lower reversal error before at most two new joint learned updates with all existing '
        'guards. This is full-policy finite-amplitude credit validation and joint parameter '
        'optimization, not more PPO iterations, feed regression or a relaxed acceptance rule.')
    started=time.perf_counter();pool=None;outcomes=[]
    progress(stage,True,'六个高非线性参数以四分之一幅度刷新完整闭环信用，验证后联合学习。')
    try:
        write_json(stage/'axis_selection.json',dict(axes=axes,scores=scores))
        original,_=pathwise_response(source,baseline,retained,paired)
        refs,curved={},{}
        for r in baseline['records']:
            with np.load(r['trace']) as z:refs[r['path']]=z['reference'].copy()
            curved[r['path']]=r['curved_path']
        pool=mp.get_context('spawn').Pool(4,initialize,(stage,metadata,directions,refs,curved,retained,started+410))
        jobs=[(f'fine_axis{i}_{s:+d}',s*.25*np.eye(18)[i]) for i in axes for s in (-1,1)]
        for c in pool.imap_unordered(evaluate,jobs):
            outcomes.append(c)
            write_json(stage/'results.json',dict(candidates=outcomes,adopted=False,goal_complete=False))
            print('FINE_POLICY_RESPONSE',len(outcomes),c['tag'],c['summary']['successes'],flush=True)
        indexed={c['tag']:c for c in outcomes}
        revised=list(paired)
        for i in axes:revised[i]=(indexed[f'fine_axis{i}_-1'],indexed[f'fine_axis{i}_+1'])
        model,eligibility=pathwise_response(stage,baseline,retained,revised)
        for i in axes:
            model[1][:,i]/=.25;model[3][:,i]/=.25
            for m in model[6].values():
                for j in m['jac'].values():j[:,i]/=.25
        oldcheck=validate(original,baseline,retained,held);newcheck=validate(model,baseline,retained,held)
        olderror=sum(r['shape_error'] for r in oldcheck);newerror=sum(r['shape_error'] for r in newcheck)
        accepted=bool(newerror<.8*olderror and sum(r['reverse_error'] for r in newcheck)<sum(r['reverse_error'] for r in oldcheck))
        write_json(stage/'validation.json',dict(coarse=oldcheck,fine=newcheck,coarse_error=olderror,fine_error=newerror,accepted=accepted,fit_to_checks=False))
        write_json(stage/'response_eligibility.json',eligibility)
        torch.save(dict(response=model,baseline=baseline,retained=retained,axes=axes),stage/'response_model.pt')
        print('FINE_POLICY_VALIDATION',olderror,newerror,accepted,flush=True)
        jobs=[]
        if accepted:
            step,report=solve_response(baseline,retained,model,trust,True,min(started+330,time.perf_counter()+90),aggregate_guards=True)
            write_json(stage/'optimizer.json',report)
            print('FINE_POLICY_SOLVER',report['message'],report['predicted_objective'],report['maximum_function_change'],flush=True)
            if abs(step).max()>1e-6:jobs=[(f'fine_policy_joint_{s}',s*step) for s in (.5,1.)]
        for tag,u in jobs:torch.save(candidate_metadata(metadata,directions,u),stage/(tag+'.pt'))
        for c in pool.imap_unordered(evaluate,jobs):
            outcomes.append(c)
            write_json(stage/'results.json',dict(candidates=outcomes,adopted=False,goal_complete=False))
            print('FINE_POLICY_ACTUAL',c['tag'],c['summary'],flush=True)
        pool.close();pool.join();pool=None
        selected=actual_selection(outcomes,baseline,retained)
        result=dict(candidates=outcomes,adopted=False,goal_complete=False)
        if selected:
            result['best']=selected['tag'];torch.save(candidate_metadata(metadata,directions,selected['parameters']),stage/'research_candidate.pt')
        write_json(stage/'results.json',result)
        receipt(stage,dict(status='completed_fine_full_policy_credit_learning',best=selected['tag'] if selected else None,
            response_validated=accepted,full64_candidates=len(outcomes),wall_s=time.perf_counter()-started,adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_fine_full_policy_credit_learning',error=str(exc),full64_candidates=len(outcomes),
            wall_s=time.perf_counter()-started,adopted=False,goal_complete=False));raise
    finally:
        progress(stage,False,'按当前完整策略小幅响应的验证结果继续学习，保持原模型及最终验收边界。')


if __name__=='__main__':mp.freeze_support();main()
