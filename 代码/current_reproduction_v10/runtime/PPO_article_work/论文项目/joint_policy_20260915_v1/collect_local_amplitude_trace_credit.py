"""Matched amplitude control at two measured peak-error paths; reuse other data."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import copy
import multiprocessing as mp
import subprocess
import sys
import time
import numpy as np
import torch
import learn_stratified_suffix_joint_credit as suffix
from dense_trace_credit_model import build_model, values, coordinates
from learn_retained_joint_growth import growth_deficits
from study_io import BASE,register,receipt,write_json,sha
from search_complete_return import read


AMPLITUDE=np.array([.00025,.00375,.00375])


def initialize(*args):
    suffix.AMPLITUDE=AMPLITUDE.copy()
    suffix.initialize(*args)


def collect(query):
    result=suffix.collect_query(query)
    result['amplitude']=AMPLITUDE.tolist()
    write_json(suffix.WORK['stage']/f"query{query['query']}"/'result.json',result)
    return result


def main():
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_dense_margin_suffix_learning_20260919'
    sampled=BASE/'canonical_gate_band_complete_current_suffix_credit_20260919'
    cached=BASE/'canonical_gate_band_error_buffered_trace_completion_20260919'
    old=read(sampled/'query_results.json')['queries']
    plan=read(sampled/'query_plan.json');baseline=plan['baseline']
    parent=torch.load(source/'research_candidate.pt',map_location='cpu',weights_only=False)
    assert sha(source/'research_candidate.pt')==plan['parent_sha256']
    controls=[]
    for name,tags in [('canonical_gate_band_per_corner_dense_credit_20260919',
                       ['per_corner_constrained_0.0625','per_corner_constrained_0.125']),
                      ('canonical_gate_band_dense_signed_trace_completion_20260919',['signed_trace_0.0625'])]:
        folder=BASE/name
        controls.extend((folder,r) for r in read(folder/'results.json')['candidates'] if r['tag'] in tags)
    previous=controls[0][0]
    metadata=torch.load(previous/'per_corner_constrained_0.125.pt',map_location='cpu',weights_only=False)
    retained=read(BASE/'canonical_gate_band_nonlinear_feed_closed_loop_20260919/results.json')['records']
    queries=[q for q in plan['queries'] if q['path'] in (8,14)]
    stage=register('canonical_gate_band_local_amplitude_trace_credit_20260919',
        [__file__,BASE/'dense_trace_credit_model.py',BASE/'learn_stratified_suffix_joint_credit.py',
         source/'research_candidate.pt',sampled/'query_plan.json',sampled/'query_results.json',
         cached/'response_model.pt',cached/'peak_locations.json',cached/'peak_finite_amplitude_audit.json']+
        [folder/(r['tag']+'.pt') for folder,r in controls],
        dict(wall_s=240,queries=len(queries),branches_per_query=6,workers=4,paths=[8,14],
             old_amplitude=plan['amplitude'],new_amplitude=AMPLITUDE.tolist(),new_policy_candidates=0),
        'The signed half-step predicts the correct peak location but underestimates its '
        'amplitude. On155deg the central and directional single-action sums differ by '
        '0.023014mm^-1; the diagonal quadratic term explains only0.003101 of the '
        '0.014850 actual error. These are sensitivity clues, not attribution proof. '
        'At the SAME parent/states/three heads collect +/- quarter-amplitude complete '
        'suffixes for ALL decisions on the two observed-error paths8/14 only. Reuse '
        'all other paths, basis, prefix/nullspace, metrics and actual measured controls. '
        'Compare predicted shape on three ALREADY measured full policies before any '
        'new parameter learning. Use smaller-amplitude response only if combined error '
        'decreases; otherwise reject that explanation and change response calibration. '
        'No repeated completed query at the old amplitude, PPO claim, deployed probe '
        'or physical-limit change. Follow immediately with evidence-based joint learning.')
    started=time.perf_counter();pool=None;output=[]
    write_json(stage/'query_plan.json',dict(queries=queries,amplitude=AMPLITUDE.tolist(),
        parent_sha256=sha(source/'research_candidate.pt'),baseline=baseline))
    subprocess.run([sys.executable,'-B',str(BASE/'record_joint_live_status.py'),'--active',stage.name,
        '--kind','physical_response_collection_for_parameter_learning','--next',
        '用两条路径的小幅三动作后缀验证有限幅响应误差，再开展联合参数学习。'],check=True,stdout=subprocess.DEVNULL)
    try:
        pool=mp.get_context('spawn').Pool(4,initialize,(stage,parent,baseline,retained,source,started+225))
        for q in pool.imap_unordered(collect,queries):
            output.append(q)
            if len(output)%16==0 or len(output)==len(queries):
                write_json(stage/'query_results.json',dict(queries=sorted(output,key=lambda q:q['query'])))
                print('LOCAL_AMPLITUDE_CREDIT',len(output),'of',len(queries),flush=True)
        pool.close();pool.join();pool=None
        new={q['query']:q for q in output}
        mixed=[]
        for q in old:
            row=copy.deepcopy(new.get(q['query'],q))
            folder=stage if q['query'] in new else sampled
            for r in row['records']:r['trace']=str(folder/r['trace'])
            mixed.append(row)
        write_json(stage/'mixed_queries.json',dict(queries=mixed))
        model=build_model(baseline,metadata,source,sampled,sorted(mixed,key=lambda q:q['query']),retained,np.array(plan['amplitude']))
        coarse=torch.load(cached/'response_model.pt',map_location='cpu',weights_only=False)
        indices=np.array([9*p+k for p in (8,14) for k in range(2,7)])
        audit=[]
        for folder,c in controls:
            md=torch.load(folder/(c['tag']+'.pt'),map_location='cpu',weights_only=False)
            z=coordinates(model,md);actual=growth_deficits(c,retained)
            a=values(coarse,z)[0];b=values(model,z)[0]
            audit.append(dict(tag=c['tag'],coarse_error=float(np.square(a[indices]-actual[indices]).sum()),
                fine_error=float(np.square(b[indices]-actual[indices]).sum()),
                peak_predictions=[dict(path=p,coarse=float(a[9*p+5]),fine=float(b[9*p+5]),
                                      actual=float(actual[9*p+5])) for p in (8,14)]))
        coarse_error=sum(r['coarse_error'] for r in audit);fine_error=sum(r['fine_error'] for r in audit)
        torch.save(model,stage/'response_model.pt')
        write_json(stage/'model_validation.json',dict(controls=audit,coarse_error=coarse_error,
            fine_error=fine_error,improved=fine_error<coarse_error))
        print('LOCAL_AMPLITUDE_VALIDATION',audit,'errors',coarse_error,fine_error,flush=True)
        receipt(stage,dict(status='completed_local_amplitude_trace_credit',queries=len(output),
            branches=6*len(output),successes=sum(r['success'] for q in output for r in q['records']),
            model_improved=fine_error<coarse_error,wall_s=time.perf_counter()-started,adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        write_json(stage/'query_results.json',dict(queries=sorted(output,key=lambda q:q['query'])))
        receipt(stage,dict(status='failed_local_amplitude_trace_credit',error=str(exc),queries=len(output),
            wall_s=time.perf_counter()-started,adopted=False,goal_complete=False))
        raise
    finally:
        subprocess.run([sys.executable,'-B',str(BASE/'record_joint_live_status.py'),'--kind',
            'physical_response_collection_finished','--next',
            '按幅度对照的实测预测误差选择响应模型，继续联合参数学习和完整闭环验证。'],check=True,stdout=subprocess.DEVNULL)


if __name__=='__main__':mp.freeze_support();main()
