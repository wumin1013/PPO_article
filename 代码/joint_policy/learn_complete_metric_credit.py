"""Matched prediction check of complete metric versus trace-composition credit."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
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
    metadata=torch.load(source/'zero_extension.pt',map_location='cpu',weights_only=False)
    with np.load(source/'parameter_space.npz') as z:directions,trust=z['directions'].copy(),z['trust'].copy()
    stage=register('canonical_gate_band_complete_metric_credit_learning_20260919',
        [__file__,BASE/'learn_pathwise_full_credit.py',BASE/'learn_full_pool_compensation.py',
         BASE/'learn_full_pool_joint_credit.py',source/'results.json',source/'receipt.json',
         source/'baseline.json',source/'parameter_space.npz',source/'zero_extension.pt',latest/'results.json',latest/'receipt.json'],
        dict(wall_s=180,workers=2,reused_axis_candidates=36,held_joint_candidates=4,new_candidates=2,optimizer_s=90),
        'Current signed-trace composition underpredicts reversal of full learned joint actors, '
        'including the interaction-corrected update. Compare original complete scalar metric '
        'finite responses against signed-trace-then-aggregation responses on four already '
        'measured current joint actors, with identical axis evidence and no fit to those actors. '
        'Use scalar complete-metric credit for half/full guarded updates only if shape error '
        'falls20percent and combined reversal prediction error improves. Keep every current '
        'complete-route eligibility,per-corner/aggregate/margin,time,cruise and function guard. '
        'This tests consequence aggregation at the present operating point; improvement would '
        'not prove a universal temporal-alignment cause or PPO estimator bias. No new response '
        'collection, runtime shaping, action labels, failed-prefix complete metrics or adoption.')
    started=time.perf_counter();pool=None;outcomes=[]
    progress(stage,True,'用四个已测联合候选比较完整指标信用与轨迹合成信用，验证后进行联合参数学习。')
    try:
        model,_=pathwise_response(source,baseline,retained,[(records[f'axis{i}_-1'],records[f'axis{i}_+1']) for i in range(18)])
        indices=np.array([9*p+k for p in range(16) for k in range(1,7)])
        reverse=np.array([9*p+3 for p in range(16)])
        checks=[]
        for c in held:
            u=np.array(c['parameters']);actual=growth_deficits(c,retained)
            scalar=model[0]+model[1]@u
            trace=trace_deficits(u,model[0],model[1],model[6],baseline,retained)[0]
            checks.append(dict(tag=c['tag'],scalar_error=float(np.linalg.norm(scalar[indices]-actual[indices])),
                trace_error=float(np.linalg.norm(trace[indices]-actual[indices])),
                scalar_reverse_error=float(np.linalg.norm(scalar[reverse]-actual[reverse])),
                trace_reverse_error=float(np.linalg.norm(trace[reverse]-actual[reverse])),
                scalar_total_reverse_error_deg=float(.5*(scalar[reverse]-actual[reverse]).sum()),
                trace_total_reverse_error_deg=float(.5*(trace[reverse]-actual[reverse]).sum())))
        se=sum(r['scalar_error'] for r in checks);te=sum(r['trace_error'] for r in checks)
        accepted=bool(se<.8*te and sum(r['scalar_reverse_error'] for r in checks)<sum(r['trace_reverse_error'] for r in checks))
        write_json(stage/'validation.json',dict(records=checks,scalar_error=se,trace_error=te,accepted=accepted,fit_to_checks=False))
        print('COMPLETE_METRIC_VALIDATION',se,te,accepted,flush=True)
        jobs=[]
        if accepted:
            step,report=solve_response(baseline,retained,model,trust,True,started+100,aggregate_guards=True,scalar_response=True)
            write_json(stage/'optimizer.json',report)
            print('COMPLETE_METRIC_SOLVER',report['message'],report['predicted_objective'],report['maximum_function_change'],flush=True)
            if abs(step).max()>1e-6:jobs=[(f'complete_metric_{s}',s*step) for s in (.5,1.)]
        refs,curved={},{}
        for r in baseline['records']:
            with np.load(r['trace']) as z:refs[r['path']]=z['reference'].copy()
            curved[r['path']]=r['curved_path']
        for tag,u in jobs:torch.save(candidate_metadata(metadata,directions,u),stage/(tag+'.pt'))
        if jobs:
            pool=mp.get_context('spawn').Pool(2,initialize,(stage,metadata,directions,refs,curved,retained,started+170))
            for c in pool.imap_unordered(evaluate,jobs):
                outcomes.append(c);write_json(stage/'results.json',dict(candidates=outcomes,adopted=False,goal_complete=False))
                print('COMPLETE_METRIC_ACTUAL',c['tag'],c['summary'],flush=True)
            pool.close();pool.join();pool=None
        selected=actual_selection(outcomes,baseline,retained)
        result=dict(candidates=outcomes,adopted=False,goal_complete=False)
        if selected:
            result['best']=selected['tag'];torch.save(candidate_metadata(metadata,directions,selected['parameters']),stage/'research_candidate.pt')
        write_json(stage/'results.json',result)
        receipt(stage,dict(status='completed_complete_metric_credit_learning',best=selected['tag'] if selected else None,
            scalar_validated=accepted,full64_candidates=len(outcomes),wall_s=time.perf_counter()-started,adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_complete_metric_credit_learning',error=str(exc),wall_s=time.perf_counter()-started,
            adopted=False,goal_complete=False));raise
    finally:
        progress(stage,False,'依据实测信用精度继续修正联合学习，所有形状、完整路径及泛化要求未完成。')


if __name__=='__main__':mp.freeze_support();main()
