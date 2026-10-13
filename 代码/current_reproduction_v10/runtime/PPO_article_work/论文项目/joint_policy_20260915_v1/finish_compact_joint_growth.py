"""Finish only missing compact-policy responses, reusing completed raw rollouts."""
import multiprocessing as mp
import os
import subprocess
import sys
import time
import numpy as np
import torch
from scipy.optimize import minimize
from compact_joint_growth_policy import CompactJointGrowthPolicy
from complete_local_credit import complete_metric_response,state_function_features
from learn_retained_joint_growth import smooth_cruise,growth_deficits,evaluate_growth
from learn_trace_resolved_joint_credit import build_trace_response,trace_deficits
from learn_entry_preserving_trace_credit import entry_constraints
from learn_coupled_feasibility_credit import build_geometric_response
from learn_nonlinear_joint_feedback import rollout,aggregate,promotion
from optimize_complete_joint_function import initialize_worker,WORKER,residual
from analyze_nonlinear_joint_response import measure
from validate_actions import difference_audit
from study_io import BASE,register,receipt,write_json,sha
from search_complete_return import read


def initialize_resume(stage,spec,refs,architecture,deadline,source):
    initialize_worker(stage,spec,refs,architecture,deadline)
    WORKER['cache_source']=source


def audit_saved_rollout(ref,rows,actions,observations,policy,factory):
    """Rebuild only exact gate geometry and verify saved deterministic actions."""
    env=factory(ref)
    for xy in rows[:,:2]:
        env.tracker.update(xy)
    audit=difference_audit(ref[0],rows)
    geometry=env.tracker.snapshot(1.25)
    normal=np.flatnonzero(rows[np.minimum(np.arange(len(observations))*20,len(rows)-1),12]==0)
    for index in normal:
        assert np.array_equal(env.decode_action(policy.act(observations[index],True)[0]),actions[index])
    brake_witness=bool(np.any(rows[:,12]==1) and np.all(rows[-20:,12]!=0))
    env.success=bool(geometry['geometry_complete'] and audit['kinematic_pass'] and audit['rest_pass'] and brake_witness)
    env.reason='cached_full_physical_completion_verified' if env.success else 'cached_failed_rollout_verified'
    result=measure(ref,rows,env)
    result['cached_raw_audit']=dict(physics_replayed=False,geometry_ticks=len(rows),
        deterministic_ordinary_actions_checked=len(normal),braking_and_rest_witness=brake_witness,
        original_runtime_reason_unavailable=True)
    return result


def evaluate_cached(job):
    tag,coeff=job
    p,stage=WORKER['policy'],WORKER['stage']
    with torch.no_grad():p.coefficients.copy_(torch.as_tensor(coeff,dtype=torch.float32))
    folder=stage/tag
    folder.mkdir(exist_ok=True)
    records=[]
    costs=dict(actor_calls=0,physics_ticks=0,reused_raw_paths=0,reused_geometry_ticks=0)
    for path,ref in WORKER['refs'].items():
        cached=WORKER['cache_source']/tag/f'path{path}.npz'
        if cached.exists():
            with np.load(cached) as z:
                rows,actions,observations=[z[k].copy() for k in ('rows','actions','observations')]
                assert np.array_equal(z['reference'],ref)
            result=audit_saved_rollout(ref,rows,actions,observations,p,WORKER['factory'])
            result['source_trace_sha256']=sha(cached)
            trace='../'+cached.relative_to(BASE).as_posix()
            costs['reused_raw_paths']+=1
            costs['reused_geometry_ticks']+=len(rows)
        else:
            result,rows,actions,observations=rollout(p,WORKER['factory'],ref,costs,WORKER['deadline'])
            np.savez_compressed(folder/f'path{path}.npz',reference=ref,rows=rows,actions=actions,observations=observations)
            trace=f'{tag}/path{path}.npz'
        result.update(path=path,trace=trace)
        if result['success']:result['smooth_cruise99_s']=smooth_cruise(rows)
        records.append(result)
        write_json(folder/'partial_records.json',dict(coefficients=np.asarray(coeff).tolist(),records=records,costs=costs))
    summary=aggregate(records)
    summary['smooth_curved_cruise99_s']=(sum(r['smooth_cruise99_s'] for r in records if r['path'] not in (48,57))
        if all(r['success'] for r in records) else None)
    return dict(tag=tag,coefficients=np.asarray(coeff).tolist(),records=records,summary=summary,costs=costs)


def relative_record(record,source):
    path=(source/record['trace']).resolve()
    return dict(record,trace='../'+path.relative_to(BASE).as_posix())


def main():
    os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_compact_joint_growth_20260919'
    source_receipt=read(source/'receipt.json')
    assert source_receipt['status']=='failed_retained_joint_growth_learning',source_receipt
    assert 'wall budget' in source_receipt['error'].lower(),source_receipt
    previous=read(source/'results.json')
    spec=torch.load(source/'policy_spec.pt',map_location='cpu',weights_only=False)
    retained=read(BASE/'canonical_gate_band_nonlinear_feed_closed_loop_20260919/results.json')['records']
    dimension=spec['transform'].shape[1]
    done={r['tag']:r for r in previous['candidates']}
    jobs=[(f'iter0_axis{h}_{s:+d}',s*.25*np.eye(dimension)[h]) for h in range(dimension) for s in (-1,1)
          if f'iter0_axis{h}_{s:+d}' not in done]
    stage=register('canonical_gate_band_compact_joint_growth_completion_20260919',
        [__file__,BASE/'compact_joint_growth_policy.py',BASE/'learn_retained_joint_growth.py',
         BASE/'learn_trace_resolved_joint_credit.py',BASE/'learn_entry_preserving_trace_credit.py',
         BASE/'learn_coupled_feasibility_credit.py',BASE/'complete_local_credit.py',
         BASE/'optimize_complete_joint_function.py',BASE/'geometry_spec.py',BASE/'validate_actions.py',
         source/'results.json',source/'receipt.json',source/'policy_spec.pt'],
        dict(reused_finished_candidates=len(done)-1,missing_response_candidates=len(jobs),
             proposal_candidates=3,workers=4,paths=20,wall_s=300),
        'The registered compact-policy round hit its wall budget before all response '
        'candidates returned. Continue the SAME planned response collection without '
        'rerunning any finished candidate or saved path. Reuse completed result rows. '
        'For a partially returned candidate, reuse its already written raw rollout '
        'NPZ files after exact deterministic-action and ordered-gate/XY/V/A/J/brake/rest '
        'audit; this replays geometry only, never physical integration. Original '
        'runtime reason is unavailable for such cached paths and is explicitly '
        'marked; no failed prefix gets time or cruise credit. Run only missing '
        'paths, then the two originally planned joint proposal scales plus one '
        'matched L2<=1 normalized joint-step proposal, motivated by prior L2~5 '
        'composition failures. No extra response axes, seeds, data or new policy '
        'architecture. The added small-step comparison changes optimizer trust '
        'only. Preserve all acceptance limits; never adopt from surrogate values. '
        'This remains complete closed-loop parameter optimization, not PPO.')
    started=time.perf_counter()
    subprocess.run([sys.executable,'-B',str(BASE/'record_joint_live_status.py'),
        '--active',stage.name,'--kind','complete_closed_loop_parameter_optimization',
        '--next','补齐局部状态联合学习的缺失响应并执行联合提案；复用已有完整轨迹，不重跑完成实验。'],
        cwd=BASE,check=True,stdout=subprocess.DEVNULL)
    torch.save(spec,stage/'policy_spec.pt')
    candidates=[dict(r,records=[relative_record(t,source) for t in r['records']],reused_from=source.name)
                for r in previous['candidates']]
    current=next(r for r in candidates if r['tag']=='reused_retained86')
    refs={}
    for r in current['records']:
        with np.load(stage/r['trace']) as z:refs[r['path']]=z['reference'].copy()
    pool=mp.get_context('spawn').Pool(4,initialize_resume,
        (stage,spec,refs,'gate_band_canonical_kernel',started+290,source))
    proposals=[]
    def save():write_json(stage/'results.json',dict(candidates=candidates,best=current['tag'],proposals=proposals,adopted=False,goal_complete=False))
    def record(result):
        result['hard_objective']=float(np.linalg.norm(residual(result,retained))**2)
        if result['summary']['successes']==20:
            v=np.maximum(growth_deficits(result,retained),0.);result['objective']=float(v@v)
        else:result['objective']=result['hard_objective']
        result['promotion']=promotion(result['records'],retained)
        result['primary_preserved']=all(result['promotion'][k] for k in ('all_complete','inward_first_turns','entry_nonworse'))
        candidates.append(result);save()
        print('COMPACT_COMPLETION',result['tag'],result['objective'],result['summary'],flush=True)
    try:
        save()
        for result in pool.imap_unordered(evaluate_cached,jobs):record(result)
        indexed={r['tag']:r for r in candidates}
        paired=[(indexed[f'iter0_axis{h}_-1'],indexed[f'iter0_axis{h}_+1']) for h in range(dimension)]
        base,matrix,bounds,modes,_=complete_metric_response(current,paired,retained)
        base[-1]=growth_deficits(current,retained)[-1]
        for h,(lo,hi) in enumerate(paired):
            low,high=lo['summary']['smooth_curved_cruise99_s'],hi['summary']['smooth_curved_cruise99_s']
            nominal=current['summary']['smooth_curved_cruise99_s']
            matrix[-1,h]=-5*((high-low)/2 if low is not None and high is not None else
                high-nominal if high is not None else nominal-low if low is not None else 0.)
        models=build_trace_response(stage,current,paired)
        with np.load(BASE/'canonical_gate_band_retained86_parameter_geometry_20260918/geometry.npz') as z:support=z['observation'].copy()
        policy=CompactJointGrowthPolicy(spec['retained'],spec['feed'],spec['basis'],spec['transform'])
        features=state_function_features(policy,support)
        trust=(features*.25/np.array([.005,.6,.03])[None,:,None]).reshape(-1,dimension)
        margin,geometry,_=build_geometric_response(stage,current,paired)
        entry,entry_jac=entry_constraints(models)
        def objective(u):
            signed,jac=trace_deficits(u,base,matrix,models,current,retained);v=np.maximum(signed,0.)
            return float(v@v+.01*(u@u)),2*jac.T@v+.02*u
        constraints=[dict(type='ineq',fun=lambda u:np.r_[1-trust@u,1+trust@u],jac=lambda u:np.vstack([-trust,trust])),
            dict(type='ineq',fun=lambda u:(margin-geometry@u)/.01,jac=lambda u:-geometry/.01)]
        if len(entry):constraints.append(dict(type='ineq',fun=lambda u:(entry+entry_jac@u)/.001,jac=lambda u:entry_jac/.001))
        fresh=[]
        for arm in ('original_support','joint_radius'):
            c=constraints+([dict(type='ineq',fun=lambda u:1-u@u,jac=lambda u:-2*u)] if arm=='joint_radius' else [])
            sol=minimize(objective,np.zeros(dimension),jac=True,method='SLSQP',bounds=bounds,constraints=c,
                         options=dict(maxiter=250,ftol=1e-8))
            proposals.append(dict(arm=arm,solver_success=bool(sol.success),message=sol.message,
                predicted_objective=objective(sol.x)[0]-.01*(sol.x@sol.x),normalized_step=sol.x.tolist(),
                normalized_l2=float(np.linalg.norm(sol.x))))
            if sol.success:
                for scale in ((.5,1.) if arm=='original_support' else (1.,)):
                    fresh.append((f'{arm}_{scale}',.25*scale*sol.x))
        write_json(stage/'proposal.json',dict(proposals=proposals,response_modes=modes))
        print('COMPACT_PROPOSALS',[(r['arm'],r['predicted_objective'],r['normalized_l2']) for r in proposals],flush=True)
        for result in pool.imap_unordered(evaluate_growth,fresh):record(result)
        eligible=[r for r in candidates if r.get('primary_preserved')]
        current=min(eligible,key=lambda r:r['objective'])
        pool.close();pool.join();pool=None
        torch.save(dict(architecture=spec['kind'],policy_spec=spec,coefficients=np.asarray(current['coefficients']),
            environment_architecture='gate_band_canonical_kernel',learning_method='compact_complete_policy_parameter_learning'),
            stage/'research_candidate.pt')
        save()
        receipt(stage,dict(status='completed_compact_joint_growth_with_cached_completion',
            missing_response_candidates_completed=len(jobs),joint_candidates=len(fresh),best=current['tag'],
            wall_s=time.perf_counter()-started,adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        save();receipt(stage,dict(status='failed_compact_joint_growth_completion',error=str(exc),
            wall_s=time.perf_counter()-started,adopted=False,goal_complete=False))
        raise
    finally:
        subprocess.run([sys.executable,'-B',str(BASE/'record_joint_live_status.py'),
            '--kind','complete_closed_loop_parameter_optimization_finished',
            '--next','核对局部状态联合学习与联合步长对照的真实结果；继续推进形状和巡航，目标仍未完成。'],
            cwd=BASE,check=True,stdout=subprocess.DEVNULL)


if __name__=='__main__':mp.freeze_support();main()
