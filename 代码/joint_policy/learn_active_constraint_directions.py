"""Expand only blocked joint response directions; retain actor architecture."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import copy
import multiprocessing as mp
import time
import numpy as np
import torch
from scipy.linalg import null_space
from scipy.optimize import linprog
from deep_joint_credit_policy import load_credit_policy
from dense_trace_credit_model import coordinates, values, objective, coefficients
from learn_prefix_preserving_suffix_credit import prefix_indices
from learn_pathwise_full_credit import pathwise_response, actual_selection
from learn_full_pool_compensation import solve_response, progress
from learn_full_pool_joint_credit import initialize, evaluate, candidate_metadata
from learn_trace_resolved_joint_credit import trace_deficits
from learn_absolute_shape_joint_credit import absolute_shape_response
from study_io import BASE, register, receipt, write_json, sha
from search_complete_return import read


def active_audit(response_data, baseline, retained):
    base,matrix,bm,mj,_,_,traces=response_data
    v,j=trace_deficits(np.zeros(matrix.shape[1]),base,matrix,traces,baseline,retained)
    av,aj=absolute_shape_response(v,j,baseline['records'],retained)
    gradient=2*aj.T@np.maximum(av,0.)
    keys=['success','entry','exit','reverse','energy','peak','apex','first','time']
    active=[(p,k) for p in range(16) for k in range(1,8) if v[9*p+k]>=-1e-8 and np.linalg.norm(j[9*p+k])>1e-10]
    matrix=np.vstack([j[9*p+k] for p,k in active]+[-mj[i]/.01 for i,r in enumerate(baseline['records']) if bm[i]<=.0200001])
    labels=[f'path{p}_{keys[k]}' for p,k in active]+['margin_path'+str(r['path']) for i,r in enumerate(baseline['records']) if bm[i]<=.0200001]
    sol=linprog(gradient,A_ub=matrix,b_ub=np.zeros(len(matrix)),bounds=[(-1.,1.)]*len(gradient),method='highs')
    assert sol.success
    return dict(linear_descent_objective=float(sol.fun),blocking=sorted([dict(label=k,dual=float(v)) for k,v in zip(labels,sol.ineqlin.marginals)],key=lambda x:x['dual']))


def expand(metadata, baseline, original, older):
    policy=load_credit_policy(metadata)
    support,protected=[],[]
    for r in baseline['records']:
        with np.load(r['trace']) as z:ref,rows,obs=[z[k].copy() for k in ('reference','rows','observations')]
        support.append(obs)
        if not r['curved_path']:protected.append(obs)
        elif 'shape' in r:protected.append(obs[prefix_indices(ref,rows,obs)])
    support,protected=np.vstack(support),np.vstack(protected)
    f=policy.features(torch.tensor(support)).detach().numpy()*policy.output_scale.numpy()
    fp=policy.features(torch.tensor(protected)).detach().numpy()*policy.output_scale.numpy()
    nulls=[null_space(fp[:,:,h],rcond=1e-12) for h in range(3)]
    oldmeta=copy.deepcopy(metadata);oldmeta['coefficients']=metadata['coefficients'][:108]
    z=coordinates(older,oldmeta);v,j=values(older,z)
    anchors=older['anchors'];hessian=j.T@j/max(objective(older,z)[0],1.)+.04*anchors.T@anchors/len(anchors)+1e-5*np.eye(len(z))
    gradients=[('path11_peak',j[9*11+5]),('path0_reverse',j[3])]
    gradients += [('margin_path'+str(p),-older['margin_jac'][older['paths'].index(p)]) for p in (6,9,31)]
    all_directions=list(original.copy());audit=[]
    normalizer=np.array([.003,.025,.025])[None,:,None]
    def response(ds):return (np.einsum('sch,ich->shi',f,np.asarray(ds))/normalizer).reshape(-1,len(ds))
    origin=coefficients(older,np.zeros_like(z))
    for name,gradient in gradients:
        delta=-np.linalg.solve(hessian,gradient)
        direction=np.vstack([coefficients(older,delta)-origin,np.zeros((6,3))])
        for h,n in enumerate(nulls):direction[:,h]=n@(n.T@direction[:,h])
        previous=response(all_directions)
        raw=response([direction])[:,0]
        weights=np.linalg.lstsq(previous,raw,rcond=1e-10)[0]
        direction-=np.einsum('i,ich->ch',weights,np.asarray(all_directions))
        new=response([direction])[:,0]
        remainder=float(np.linalg.norm(new)/max(np.linalg.norm(raw),1e-30))
        if remainder<1e-7:
            audit.append(dict(label=name,added=False,residual_fraction=remainder));continue
        direction*=.25/np.max(abs(new))
        all_directions.append(direction)
        audit.append(dict(label=name,added=True,index=len(all_directions)-1,residual_fraction=remainder))
    ds=np.asarray(all_directions);trust=response(ds)
    prefix=np.einsum('sch,ich->shi',fp,ds)
    assert np.max(abs(prefix))<1e-11
    return ds,trust,dict(proposals=audit,protected_states=len(protected),support_states=len(support),
        prefix_max=float(np.max(abs(prefix))),centers=len(metadata['centers']),layers_added=0,
        old_gradients_only_propose_directions=True)


def main():
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_full_pool_joint_credit_learning_20260919'
    parent=BASE/'canonical_gate_band_iterative_motion_constraint_learning_20260919/research_candidate.pt'
    old=BASE/'canonical_gate_band_recentered_fine_trace_learning_20260919/response_model.pt'
    assert sha(parent)=='5ab47111bfcec3a511903e20f1249c91bbfb65e5ab4cd1a3d2b4a493a979c26a'
    data=read(source/'baseline.json');baseline,retained=data['baseline'],data['retained']
    reused=copy.deepcopy(read(source/'results.json')['candidates'])
    for c in reused:
        for r in c['records']:r['trace']=str(source/r['trace'])
    indexed={c['tag']:c for c in reused}
    with np.load(source/'parameter_space.npz') as z:original=z['directions'].copy()
    metadata=torch.load(parent,map_location='cpu',weights_only=False)
    older=torch.load(old,map_location='cpu',weights_only=False)
    stage=register('canonical_gate_band_active_constraint_joint_learning_20260919',
        [__file__,BASE/'learn_pathwise_full_credit.py',BASE/'learn_full_pool_compensation.py',
         BASE/'learn_full_pool_joint_credit.py',source/'results.json',source/'receipt.json',
         source/'baseline.json',source/'parameter_space.npz',old,parent],
        dict(wall_s=480,workers=4,max_added_directions=5,max_new_axis_candidates=10,
             new_joint_candidates=2,reused_full64_candidates=16,centers=114,layers_added=0),
        'Pathwise complete credit reduced held joint shape error from0.04761 to0.02071 but '
        'the8-dimensional local model has no guarded descent direction. Expand only the parameter '
        'search subspace, retaining114 actor centers and all existing completed trials. Propose5 '
        'directions using earlier local response gradients for path11 peak,path0 reversal and '
        'path6/9/31 margins. Old gradients propose direction only; current actual full64 +/- '
        'trials supply learning credit. Project out protected current entry/straight responses '
        'and prior8-direction action span; normalize new directions to0.25 current action trust. '
        'Recombine old and new pathwise complete responses, freezing any coordinate with a route '
        'missing both complete signs. Solve existing full-pool/per-corner guards, then test half/full '
        'joint proposals on all64. No retries of old axes, architecture expansion, PPO claim, '
        'failed-prefix time benefit, runtime rule or final model promotion.')
    started=time.perf_counter();pool=None;outcomes=[]
    progress(stage,True,'补采五个活跃约束补偿方向的完整64路径响应，联合三动作参数后实测。')
    try:
        before=pathwise_response(source,baseline,retained,[(indexed[f'axis{i}_-1'],indexed[f'axis{i}_+1']) for i in range(8)])[0]
        write_json(stage/'original_active_constraints.json',active_audit(before,baseline,retained))
        directions,trust,audit=expand(metadata,baseline,original,older)
        write_json(stage/'direction_audit.json',audit)
        np.savez_compressed(stage/'parameter_space.npz',directions=directions,trust=trust)
        write_json(stage/'baseline.json',data)
        refs,curved={},{}
        for r in baseline['records']:
            with np.load(r['trace']) as z:refs[r['path']]=z['reference'].copy()
            curved[r['path']]=r['curved_path']
        n=len(directions)
        for c in reused:c['parameters']+=([0.]*(n-8))
        pool=mp.get_context('spawn').Pool(4,initialize,(stage,metadata,directions,refs,curved,retained,started+470))
        jobs=[(f'axis{i}_{s:+d}',s*np.eye(n)[i]) for i in range(8,n) for s in (-1,1)]
        for c in pool.imap_unordered(evaluate,jobs):
            outcomes.append(c);indexed[c['tag']]=c
            write_json(stage/'results.json',dict(candidates=outcomes,reused_source=source.name,adopted=False,goal_complete=False))
            print('ACTIVE_AXIS',c['tag'],c['summary']['successes'],c['summary']['reverse_sum_deg'],flush=True)
        paired=[(indexed[f'axis{i}_-1'],indexed[f'axis{i}_+1']) for i in range(n)]
        response_data,eligibility=pathwise_response(stage,baseline,retained,paired)
        write_json(stage/'response_eligibility.json',eligibility)
        write_json(stage/'expanded_active_constraints.json',active_audit(response_data,baseline,retained))
        step,report=solve_response(baseline,retained,response_data,trust,True,min(started+380,time.perf_counter()+90))
        write_json(stage/'optimizer.json',report)
        print('ACTIVE_SOLVER',report['message'],report['predicted_objective'],report['maximum_function_change'],flush=True)
        joint=[(f'active_joint_{s}',s*step) for s in (.5,1.)] if np.max(abs(step))>1e-6 else []
        for tag,u in joint:torch.save(candidate_metadata(metadata,directions,u),stage/(tag+'.pt'))
        for c in pool.imap_unordered(evaluate,joint):
            outcomes.append(c)
            write_json(stage/'results.json',dict(candidates=outcomes,reused_source=source.name,adopted=False,goal_complete=False))
            print('ACTIVE_JOINT',c['tag'],c['summary'],flush=True)
        pool.close();pool.join();pool=None
        selected=actual_selection(outcomes,baseline,retained)
        result=dict(candidates=outcomes,reused_source=source.name,adopted=False,goal_complete=False)
        if selected:
            result['best']=selected['tag']
            torch.save(candidate_metadata(metadata,directions,selected['parameters']),stage/'research_candidate.pt')
        write_json(stage/'results.json',result)
        receipt(stage,dict(status='completed_active_constraint_joint_learning',best=selected['tag'] if selected else None,
            full64_candidates=len(outcomes),reused_full64_candidates=16,wall_s=time.perf_counter()-started,
            adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_active_constraint_joint_learning',error=str(exc),
            full64_candidates=len(outcomes),wall_s=time.perf_counter()-started,adopted=False,goal_complete=False));raise
    finally:
        progress(stage,False,'依据扩展联合响应的完整64路径结果继续大角度形状学习；冻结和泛化验收尚未开始。')


if __name__=='__main__':mp.freeze_support();main()
