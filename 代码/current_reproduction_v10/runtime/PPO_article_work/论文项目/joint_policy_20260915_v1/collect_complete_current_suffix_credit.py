"""Collect every unprotected decision's complete three-action consequences."""
import argparse
import multiprocessing as mp
import os
import subprocess
import sys
import time
import numpy as np
import torch
import learn_stratified_suffix_joint_credit as suffix
from learn_current_protected_joint_credit import initialize,query_plan
from learn_prefix_preserving_suffix_credit import prefix_indices
from study_io import BASE,register,receipt,write_json,sha
from search_complete_return import read


def collect_and_save(query):
    result=suffix.collect_query(query)
    write_json(suffix.WORK['stage']/f"query{query['query']}"/'result.json',result)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_dense_margin_suffix_learning_20260919'
    data=read(source/'results.json');baseline=next(c for c in data['candidates'] if c['tag']==data['best'])
    model=source/'research_candidate.pt';parent=torch.load(model,map_location='cpu',weights_only=False)
    retained=read(BASE/'canonical_gate_band_nonlinear_feed_closed_loop_20260919/results.json')['records']
    name='canonical_gate_band_complete_current_suffix_credit_20260919'
    if args.resume:
        stage=BASE/name;assert read(stage/'query_plan.json')['parent_sha256']==sha(model)
    else:
        stage=register(name,[__file__,BASE/'learn_current_protected_joint_credit.py',
            BASE/'learn_stratified_suffix_joint_credit.py',BASE/'learn_prefix_preserving_suffix_credit.py',
            BASE/'local_suffix_credit_policy.py',BASE/'deep_joint_credit_policy.py',model,
            source/'results.json',source/'receipt.json',source/'prior_policy_margin_prediction.json'],
            dict(wall_s=1200,workers=4,maximum_queries=638,branches_per_query=6,
                 maximum_new_branches=3828,amplitude=[.001,.015,.015],basis_centers=108),
            'Dense temporal constraint credit corrected the sign error and predicted '
            'four previously measured margins within0.000017mm. Its learned candidate '
            'completed20/20 with19.97685deg reversal,0.107719mm exit,3.691s curved99 '
            'cruise, still not accepted. At this NEW current model collect EVERY '
            'ordinary decision after the same numerical protected entry prefix, on '
            'all18 curved training paths.638 states,3828 complete-feedback action '
            'interventions maximum; each state has UNIT weight for BOTH shape and '
            'margin credit. This removes temporal subsampling from the finite '
            'action-response surrogate, not finite-amplitude/nonlinear approximation. '
            'Verify exact next-decision identity at every state. Keep only108 '
            'state-basis centers, selected by the predeclared six-stratum plan, so '
            'denser credit is not conflated with more learned parameters. Save '
            'per-query receipts for reuse; no completed query is rerun on resume. '
            'The next operation is joint parameter optimization and actual full '
            'closed-loop verification, not more diagnosis-only work. No PPO claim, '
            'action label, path/angle rule, limit relaxation or sealed test.')
    started=time.perf_counter();pool=None;output=[];queries=[]
    for r in baseline['records']:
        with np.load(source/r['trace']) as z:ref,rows,obs=[z[k].copy() for k in ('reference','rows','observations')]
        start=len(prefix_indices(ref,rows,obs))
        indices=np.flatnonzero(rows[np.minimum(np.arange(len(obs))*20,len(rows)-1),12]==0)
        indices=indices[indices>=start]
        for d in indices:
            queries.append(dict(query=len(queries),path=r['path'],frame=int(d)*20,stratum=int(d),
                stratum_size=1,ordinary_decisions=len(indices),protected_prefix_decisions=start))
    assert len(queries)<=638
    basis_plan=query_plan(baseline,source,20260919)
    basis_ids=[next(q['query'] for q in queries if q['path']==b['path'] and q['frame']==b['frame']) for b in basis_plan]
    if not args.resume:
        write_json(stage/'query_plan.json',dict(queries=queries,basis_query_ids=basis_ids,source=str(source),
            baseline=baseline,parent_sha256=sha(model),amplitude=[.001,.015,.015]))
    else:
        old=read(stage/'query_plan.json');assert old['queries']==queries and old['basis_query_ids']==basis_ids
    missing=[]
    for q in queries:
        saved=stage/f"query{q['query']}"/'result.json'
        if saved.exists():output.append(read(saved))
        elif saved.parent.exists():raise RuntimeError(f'Partial query retained, recover its branch traces before restarting: {saved.parent}')
        else:missing.append(q)
    subprocess.run([sys.executable,'-B',str(BASE/'record_joint_live_status.py'),'--active',stage.name,
        '--kind','physical_response_collection_for_parameter_learning','--next','采集当前策略全部638个未保护决策的完整三动作后果，再用单位权重开展联合形状与约束学习。'],
        check=True,stdout=subprocess.DEVNULL)
    try:
        pool=mp.get_context('spawn').Pool(4,initialize,(stage,parent,baseline,retained,source,started+1185))
        for q in pool.imap_unordered(collect_and_save,missing):
            output.append(q)
            if len(output)%24==0 or len(output)==len(queries):
                write_json(stage/'query_results.json',dict(queries=sorted(output,key=lambda q:q['query'])))
                print('COMPLETE_DECISION_CREDIT',len(output),'of',len(queries),'complete',
                    sum(r['success'] for q in output for r in q['records']),'primary',
                    sum(r['credit_feasible'] for q in output for r in q['records']),flush=True)
        pool.close();pool.join();pool=None
        write_json(stage/'query_results.json',dict(queries=sorted(output,key=lambda q:q['query'])))
        receipt(stage,dict(status='completed_all_current_decision_suffix_credit',queries=len(output),
            new_queries=len(missing),branches=6*len(output),basis_centers=len(basis_ids),
            wall_s=time.perf_counter()-started,adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        write_json(stage/'query_results.json',dict(queries=sorted(output,key=lambda q:q['query'])))
        receipt(stage,dict(status='failed_all_current_decision_suffix_credit',error=str(exc),queries=len(output),
            wall_s=time.perf_counter()-started,adopted=False,goal_complete=False));raise
    finally:
        subprocess.run([sys.executable,'-B',str(BASE/'record_joint_live_status.py'),'--kind',
            'physical_response_collection_finished','--next','用已采齐的逐决策形状和余量信用进行联合参数优化及完整闭环对照。'],
            check=True,stdout=subprocess.DEVNULL)


if __name__=='__main__':mp.freeze_support();main()
