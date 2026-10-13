"""Complete decision coverage for the two near-boundary training paths."""
import copy
import multiprocessing as mp
import os
import subprocess
import sys
import time
import numpy as np
import torch
import learn_stratified_suffix_joint_credit as suffix
from learn_current_protected_joint_credit import initialize
from learn_prefix_preserving_suffix_credit import prefix_indices
from study_io import BASE,register,receipt,write_json
from search_complete_return import read


def main():
    os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_numerical_prefix_suffix_credit_20260919'
    sampled=BASE/'canonical_gate_band_current_protected_joint_credit_20260919/iteration0'
    previous=read(sampled/'query_results.json')['queries']
    baseline=next(c for c in read(source/'results.json')['candidates'] if c['tag']=='protected_0.125')
    parent=torch.load(source/'protected_0.125.pt',map_location='cpu',weights_only=False)
    retained=read(BASE/'canonical_gate_band_nonlinear_feed_closed_loop_20260919/results.json')['records']
    stage=register('canonical_gate_band_dense_boundary_suffix_credit_20260919',
        [__file__,BASE/'learn_current_protected_joint_credit.py',BASE/'learn_stratified_suffix_joint_credit.py',
         source/'protected_0.125.pt',source/'results.json',sampled/'query_results.json',sampled/'query_plan.json',
         BASE/'canonical_gate_band_linear_margin_suffix_credit_20260919/results.json',
         BASE/'canonical_gate_band_linear_margin_suffix_credit_20260919/receipt.json'],
        dict(wall_s=300,workers=4,paths=[6,9],all_unprotected_ordinary_decisions=True,
             maximum_new_queries=59,branches_per_query=6,amplitude=[.001,.015,.015]),
        'The sampled margin constraint prevented path9 failure but predicted the wrong '
        'direction of path6 actual full-policy margin. Path6 six samples represented41 '
        'post-prefix decisions; path9 six represented30. Collect every MISSING ordinary '
        'decision after the same protected prefix on these two paths, exactly once, '
        'reusing all12 existing matching state queries. Same parent,20ms intervention, '
        'amplitudes and complete feedback through rest. Each dense query contributes '
        'unit weight to constraint derivatives; no temporal upweighting. Keep sparse '
        'shape learning data and actor basis fixed for the following matched optimizer '
        'comparison. This is targeted physical response collection, not PPO, a new '
        'policy rule or final validation. Preserve all failed branches and keep failed '
        'prefixes out of efficiency. Do not open any sealed paths.')
    started=time.perf_counter();outcomes=[];missing=[];pool=None
    for r in baseline['records']:
        if r['path'] not in (6,9):continue
        with np.load(source/r['trace']) as z:ref,rows,obs=[z[k].copy() for k in ('reference','rows','observations')]
        start=len(prefix_indices(ref,rows,obs))
        indices=np.flatnonzero(rows[np.minimum(np.arange(len(obs))*20,len(rows)-1),12]==0)
        indices=indices[indices>=start]
        for d in indices:
            matches=[q for q in previous if q['path']==r['path'] and q['frame']==int(d)*20]
            if matches:
                assert len(matches)==1
                q=copy.deepcopy(matches[0]);q['reused']=True;q['constraint_weight']=1
                for record in q['records']:record['trace']=str(sampled/record['trace'])
                outcomes.append(q)
            else:
                missing.append(dict(query=108+len(missing),path=r['path'],frame=int(d)*20,
                    stratum=int(d),stratum_size=1,ordinary_decisions=len(indices),protected_prefix_decisions=start))
    assert len(missing)<=59
    write_json(stage/'query_plan.json',dict(queries=missing,reused_queries=[dict(path=q['path'],frame=q['frame']) for q in outcomes],
        total_queries=len(missing)+len(outcomes),amplitude=[.001,.015,.015]))
    write_json(stage/'query_results.json',dict(queries=outcomes))
    subprocess.run([sys.executable,'-B',str(BASE/'record_joint_live_status.py'),'--active',stage.name,
        '--kind','physical_response_collection_for_parameter_learning','--next','补齐两条临界路径逐决策响应后，以单位权重校正约束信用并继续联合优化。'],
        check=True,stdout=subprocess.DEVNULL)
    try:
        pool=mp.get_context('spawn').Pool(4,initialize,(stage,parent,baseline,retained,source,started+290))
        for q in pool.imap_unordered(suffix.collect_query,missing):
            q['reused']=False;q['constraint_weight']=1
            outcomes.append(q);write_json(stage/'query_results.json',dict(queries=outcomes))
            if sum(not x['reused'] for x in outcomes)%10==0:
                print('DENSE_BOUNDARY_CREDIT',sum(not x['reused'] for x in outcomes),'of',len(missing),
                    'all_complete',sum(r['success'] for q in outcomes for r in q['records']),flush=True)
        pool.close();pool.join();pool=None
        receipt(stage,dict(status='completed_dense_boundary_suffix_credit',new_queries=len(missing),
            reused_queries=sum(q['reused'] for q in outcomes),total_queries=len(outcomes),
            wall_s=time.perf_counter()-started,adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_dense_boundary_suffix_credit',error=str(exc),
            finished_queries=len(outcomes),wall_s=time.perf_counter()-started,adopted=False,goal_complete=False));raise
    finally:
        subprocess.run([sys.executable,'-B',str(BASE/'record_joint_live_status.py'),'--kind',
            'physical_response_collection_finished','--next','用完整逐决策边界响应重建约束信用，保持形状数据与表示不变进行联合参数对照。'],
            check=True,stdout=subprocess.DEVNULL)


if __name__=='__main__':mp.freeze_support();main()
