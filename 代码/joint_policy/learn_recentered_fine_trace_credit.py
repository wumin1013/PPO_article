"""Refresh complete small-amplitude credit and update the existing actor in place."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse
import copy
import multiprocessing as mp
import subprocess
import sys
import time
import numpy as np
import torch
import learn_stratified_suffix_joint_credit as suffix
from dense_trace_credit_model import build_model,values,coordinates,coefficients,objective
from guarded_trace_optimizer import solve_guarded
from deep_joint_credit_policy import load_credit_policy
from learn_prefix_preserving_suffix_credit import prefix_indices
from learn_error_buffered_trace_credit import actual_guard_audit
from learn_absolute_shape_joint_credit import absolute_actual
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


def status(stage,kind,next_action):
    subprocess.run([sys.executable,'-B',str(BASE/'record_joint_live_status.py'),'--active',stage,
        '--kind',kind,'--next',next_action],check=True,stdout=subprocess.DEVNULL)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_secant_calibrated_joint_learning_20260919'
    metadata=torch.load(source/'research_candidate.pt',map_location='cpu',weights_only=False)
    data=read(source/'results.json');baseline=next(c for c in data['candidates'] if c['tag']==data['best'])
    retained=read(BASE/'canonical_gate_band_nonlinear_feed_closed_loop_20260919/results.json')['records']
    name='canonical_gate_band_recentered_fine_trace_learning_20260919'
    if args.resume:
        stage=BASE/name
        assert read(stage/'query_plan.json')['parent_sha256']==sha(source/'research_candidate.pt')
    else:
        stage=register(name,[__file__,BASE/'dense_trace_credit_model.py',BASE/'guarded_trace_optimizer.py',
            BASE/'learn_stratified_suffix_joint_credit.py',BASE/'local_suffix_credit_policy.py',
            BASE/'learn_prefix_preserving_suffix_credit.py',source/'research_candidate.pt',source/'results.json',
            source/'receipt.json',BASE/'canonical_gate_band_local_amplitude_trace_credit_20260919/model_validation.json'],
            dict(wall_s=1560,response_collection_s=1200,solver_per_arm_s=120,
                 maximum_queries=700,branches_per_query=6,workers=4,paths=20,
                 amplitude=AMPLITUDE.tolist(),trust_scales=[.25,.5],maximum_new_candidates=2,
                 basis_centers=108,add_network_layers=False),
            'The full-policy calibrated0.5 branch is a measured20/20 complete research '
            'candidate with17.232238deg reversal,0.047908mm exit,3.745s actual curved99 '
            'cruise and all local parent guards passed. It is NOT adopted; retained86cd '
            'per-corner,full64,fairness/generalization conditions still fail or remain '
            'unperformed. Refresh all ordinary unprotected decisions at THIS new '
            'checkpoint with the smaller three-action amplitude already supported '
            'by a matched79-state control. No old-center query is mislabeled current '
            'and no completed query is repeated on resume. Preserve108 centers, '
            'features and network depth: optimize additive COEFFICIENT updates in '
            'the existing actor, with exact zero-coordinate identity to this new '
            'center and numerical protected-prefix preservation against its actor. '
            'Use dense signed-trace nonlinear aggregation, original physical margin '
            'and per-corner guards, two bounded joint radii only, then actual full20 '
            'paths. Persist per-query data and feasible optimizer iterates. This is '
            'full-suffix physical response collection and deterministic parameter '
            'learning, not PPO, feed-label fitting, runtime rules or final validation.')
    started=time.perf_counter();pool=None;outcomes=[];jobs=[];audits=[]
    queries=[]
    for r in baseline['records']:
        with np.load(source/r['trace']) as z:ref,rows,obs=[z[k].copy() for k in ('reference','rows','observations')]
        start=len(prefix_indices(ref,rows,obs))
        ordinary=np.flatnonzero(rows[np.minimum(np.arange(len(obs))*20,len(rows)-1),12]==0)
        ordinary=ordinary[ordinary>=start]
        for d in ordinary:
            queries.append(dict(query=len(queries),path=r['path'],frame=int(d)*20,stratum=int(d),
                stratum_size=1,ordinary_decisions=len(ordinary),protected_prefix_decisions=start))
    assert len(queries)<=700
    if args.resume:assert read(stage/'query_plan.json')['queries']==queries
    else:write_json(stage/'query_plan.json',dict(queries=queries,baseline=baseline,
        parent_sha256=sha(source/'research_candidate.pt'),amplitude=AMPLITUDE.tolist(),
        center_coefficients=metadata['coefficients'].tolist()))
    output=[];missing=[]
    for q in queries:
        saved=stage/f"query{q['query']}"/'result.json'
        if saved.exists():output.append(read(saved))
        elif saved.parent.exists():raise RuntimeError(f'Partial query preserved; inspect before resuming: {saved.parent}')
        else:missing.append(q)
    status(stage.name,'physical_response_collection_for_parameter_learning',
           '在新的17.232度反转研究分支上刷新全部小幅三动作完整后果，随后保持网络容量继续联合学习。')
    try:
        if missing:
            pool=mp.get_context('spawn').Pool(4,initialize,(stage,metadata,baseline,retained,source,started+1180))
            for q in pool.imap_unordered(collect,missing):
                output.append(q)
                if len(output)%24==0 or len(output)==len(queries):
                    write_json(stage/'query_results.json',dict(queries=sorted(output,key=lambda q:q['query'])))
                    print('RECENTERED_FINE_CREDIT',len(output),'of',len(queries),'complete',
                          sum(r['success'] for q in output for r in q['records']),flush=True)
            pool.close();pool.join();pool=None
        write_json(stage/'query_results.json',dict(queries=sorted(output,key=lambda q:q['query'])))
        write_json(stage/'collection_receipt.json',dict(queries=len(output),new_queries=len(missing),
            branches=6*len(output),successes=sum(r['success'] for q in output for r in q['records']),
            wall_s=time.perf_counter()-started,parent_sha256=sha(source/'research_candidate.pt')))
        status(stage.name,'complete_closed_loop_parameter_optimization',
               '已刷新当前闭环信用；在同一108中心策略上学习两个联合系数更新并验证完整路径。')
        cache=stage/'response_model.pt'
        if cache.exists():model=torch.load(cache,map_location='cpu',weights_only=False)
        else:
            model=build_model(baseline,metadata,source,stage,sorted(output,key=lambda q:q['query']),retained,
                              AMPLITUDE,center_coefficients=metadata['coefficients'].copy())
            torch.save(model,cache)
        zero=np.zeros(model['metric_jac'].shape[1])
        assert np.max(abs(coefficients(model,zero)-metadata['coefficients']))==0.
        assert np.max(abs(coordinates(model,metadata)))==0.
        identity_error=abs(objective(model,zero)[0]-absolute_actual(baseline,retained))
        assert identity_error<1e-7,identity_error
        center=load_credit_policy(metadata)
        with torch.no_grad():model['policy'].coefficients.copy_(torch.tensor(metadata['coefficients']))
        zero_actor_error=max(float(abs(model['policy'].act(x,True)[0]-center.act(x,True)[0]).max()) for x in model['protected'])
        assert zero_actor_error<=1e-12,zero_actor_error
        write_json(stage/'center_identity.json',dict(metric_error=identity_error,
            protected_actor_error=zero_actor_error,coefficient_bitwise_identity=True,
            center_count=len(metadata['centers']),network_depth_unchanged=True))
        optimization_started=time.perf_counter()
        for scale in (.25,.5):
            tag=f'recentered_joint_{scale}'
            if (stage/(tag+'.pt')).exists():
                candidate=torch.load(stage/(tag+'.pt'),map_location='cpu',weights_only=False)
                jobs.append((tag,candidate));continue
            z,audit=solve_guarded(model,zero,scale,stage/(tag+'_solver'),
                min(started+1445,optimization_started+260,time.perf_counter()+120))
            audit['tag']=tag;audits.append(audit);write_json(stage/'optimizer.json',dict(candidates=audits))
            if z is None or abs(model['anchors']@z).max()<1e-7:continue
            candidate=copy.deepcopy(metadata)
            candidate.update(coefficients=coefficients(model,z),step_scale=1.,
                learning_method='recentered_small_amplitude_joint_parameter_optimization')
            with torch.no_grad():model['policy'].coefficients.copy_(torch.tensor(candidate['coefficients']))
            prefix_error=max(float(abs(model['policy'].act(x,True)[0]-center.act(x,True)[0]).max()) for x in model['protected'])
            assert prefix_error<=1e-12,prefix_error
            audit['maximum_protected_latent_difference']=prefix_error
            torch.save(candidate,stage/(tag+'.pt'));jobs.append((tag,candidate))
        write_json(stage/'optimizer.json',dict(candidates=audits))
        if (stage/'results.json').exists():outcomes=read(stage/'results.json')['candidates']
        done={c['tag'] for c in outcomes}
        pending=[job for job in jobs if job[0] not in done]
        if pending:
            for tag,_ in pending:
                if (stage/tag).exists():raise RuntimeError(f'Partial candidate trace folder preserved: {tag}')
            pool=mp.get_context('spawn').Pool(2,suffix.initialize,(stage,metadata,baseline,retained,source,started+1550))
            for result in pool.imap_unordered(suffix.evaluate_network,pending):
                result['local_guard_audit']=actual_guard_audit(model,result) if result['summary']['successes']==20 else None
                outcomes.append(result);write_json(stage/'results.json',dict(candidates=outcomes,adopted=False,goal_complete=False))
                print('RECENTERED_FINE_ACTUAL',result['tag'],result['summary'],result['local_guard_audit'],flush=True)
            pool.close();pool.join();pool=None
        good=[c for c in outcomes if c['primary_preserved'] and c['absolute_objective']<baseline['absolute_objective']
              and c['summary']['reverse_sum_deg']<baseline['summary']['reverse_sum_deg']
              and c['summary']['exit_sum_mm']<=baseline['summary']['exit_sum_mm']
              and c['summary']['energy_sum']<=baseline['summary']['energy_sum']]
        best=min(good,key=lambda c:c['absolute_objective']) if good else None
        result=dict(candidates=outcomes,adopted=False,goal_complete=False)
        if best:
            result['best']=best['tag'];torch.save(next(md for tag,md in jobs if tag==best['tag']),stage/'research_candidate.pt')
        write_json(stage/'results.json',result)
        receipt(stage,dict(status='completed_recentered_fine_trace_joint_learning',queries=len(output),new_candidates=len(outcomes),
            best=None if best is None else best['tag'],wall_s=time.perf_counter()-started,adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        write_json(stage/'query_results.json',dict(queries=sorted(output,key=lambda q:q['query'])))
        receipt(stage,dict(status='failed_recentered_fine_trace_joint_learning',error=str(exc),queries=len(output),
            new_candidates=len(outcomes),wall_s=time.perf_counter()-started,adopted=False,goal_complete=False))
        raise
    finally:
        status('','complete_closed_loop_parameter_optimization_finished',
               '按最新当前策略响应和完整实测继续联合学习；形状、全池、公平比较和泛化未验收。')


if __name__=='__main__':mp.freeze_support();main()
