"""Use validated head interactions while learning all local joint coefficients."""
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
from study_io import BASE,register,receipt,write_json
from search_complete_return import read


def augmented_response(original,prism,headmap,trust):
    n=headmap.shape[0]
    g=trust.T@trust/len(trust)
    projection=np.linalg.solve(headmap.T@g@headmap,headmap.T@g)
    assert np.max(abs(projection@headmap-np.eye(3)))<1e-10
    def augment(a,b):
        correction=b.copy();correction[:,:3]-=a@headmap
        return np.column_stack([a,correction])
    base,matrix,bm,mj,bounds,modes,traces=copy.deepcopy(original)
    matrix=augment(matrix,prism[1]);mj=augment(mj,prism[3])
    for path,m in traces.items():
        for k in m['jac']:m['jac'][k]=augment(m['jac'][k],prism[6][path]['jac'][k])
    model=(base,matrix,bm,mj,bounds+[(0.,1.)]*5,modes+['prism_feature']*5,traces)
    extended=np.column_stack([trust,np.zeros((len(trust),5))])
    orthogonal=trust@(np.eye(n)-headmap@projection)
    def equality(y):
        z=y[n:];residual=np.r_[z[:3]-projection@y[:n],z[3]-z[0]*z[1],z[4]-z[0]*z[2]]
        jac=np.zeros((5,n+5));jac[:3,:n]=-projection;jac[:3,n:n+3]=np.eye(3)
        jac[3,n:]=[-z[1],-z[0],0.,1.,0.];jac[4,n:]=[-z[2],0.,-z[0],0.,1.]
        return residual,jac
    extras=[dict(type='eq',fun=lambda y:equality(y)[0],jac=lambda y:equality(y)[1]),
        dict(type='ineq',fun=lambda y:np.array([y[n+1]-y[n+2]]),jac=lambda y:np.r_[np.zeros(n),0.,1.,-1.,0.,0.][None,:]),
        dict(type='ineq',fun=lambda y:np.r_[.5-orthogonal@y[:n],.5+orthogonal@y[:n]],
             jac=lambda y:np.column_stack([np.vstack([-orthogonal,orthogonal]),np.zeros((2*len(trust),5))]))]
    return model,extended,extras,projection


def main():
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_current_bend_capacity_learning_20260919'
    prism_source=BASE/'canonical_gate_band_complete_head_prism_learning_20260919'
    assert read(prism_source/'validation.json')['accepted']
    data=read(source/'baseline.json');baseline,retained=data['baseline'],data['retained']
    records={c['tag']:c for c in read(source/'results.json')['candidates']}
    u=np.array(records['bend_joint_1.0']['parameters'])
    for c in records.values():
        if c['tag'].startswith('axis'):
            for r in c['records']:r['trace']=str(source/r['trace'])
    metadata=torch.load(source/'zero_extension.pt',map_location='cpu',weights_only=False)
    with np.load(source/'parameter_space.npz') as z:directions,trust=z['directions'].copy(),z['trust'].copy()
    prism=torch.load(prism_source/'response_model.pt',map_location='cpu',weights_only=False)['response']
    headmap=np.zeros((18,3))
    for i in range(18):headmap[i,i%3]=u[i]
    stage=register('canonical_gate_band_joint_interaction_subspace_learning_20260919',
        [__file__,BASE/'learn_full_pool_compensation.py',BASE/'learn_pathwise_full_credit.py',
         BASE/'learn_full_pool_joint_credit.py',source/'results.json',source/'baseline.json',
         source/'parameter_space.npz',source/'zero_extension.pt',prism_source/'response_model.pt',
         prism_source/'validation.json',prism_source/'receipt.json'],
        dict(wall_s=240,workers=2,new_full64_candidates=2,optimizer_s=120,orthogonal_function_radius=.5,
             learned_parameters=18,network_layers_added=0),
        'The complete head-prism response reduced held actor error34.5percent but its3 global '
        'head weights had no guarded descent. Transfer that validated nonlinear interaction '
        'information into the18-dimensional current coefficient response. Project coefficients '
        'onto the3 proposed head functions using the saved-state action metric; replace their '
        'linear response with the validated prism response while retaining local derivatives '
        'orthogonal to this subspace. Enforce exact polynomial feature equalities, observed '
        'prism parameter domain,0.5 normalized orthogonal function radius,all route margins '
        'and original per-corner/aggregate/time/cruise guards. This is a proposal model, not a '
        'deployed function or new actor layer. Optimize the SAME18 learned actor coefficients '
        'and validate half/full on64 paths. No additional response collection, failed-prefix '
        'loss, action-label fitting, PPO claim or scientific adoption.')
    started=time.perf_counter();pool=None;outcomes=[]
    progress(stage,True,'把验证过的三动作交互信用用于18个局部参数，优化后实测完整64路径。')
    try:
        original,_=pathwise_response(source,baseline,retained,[(records[f'axis{i}_-1'],records[f'axis{i}_+1']) for i in range(18)])
        model,extended,extras,projection=augmented_response(original,prism,headmap,trust)
        np.savez_compressed(stage/'projection.npz',projection=projection,headmap=headmap)
        step,report=solve_response(baseline,retained,model,extended,False,started+120,
                                  aggregate_guards=True,additional_constraints=extras)
        write_json(stage/'optimizer.json',report)
        print('INTERACTION_SUBSPACE_SOLVER',report['message'],report['predicted_objective'],report['maximum_function_change'],flush=True)
        x=step[:18]
        jobs=[(f'interaction_subspace_{s}',s*x) for s in (.5,1.)] if abs(x).max()>1e-6 else []
        refs,curved={},{}
        for r in baseline['records']:
            with np.load(r['trace']) as z:refs[r['path']]=z['reference'].copy()
            curved[r['path']]=r['curved_path']
        for tag,p in jobs:torch.save(candidate_metadata(metadata,directions,p),stage/(tag+'.pt'))
        if jobs:
            pool=mp.get_context('spawn').Pool(2,initialize,(stage,metadata,directions,refs,curved,retained,started+230))
            for c in pool.imap_unordered(evaluate,jobs):
                outcomes.append(c)
                write_json(stage/'results.json',dict(candidates=outcomes,adopted=False,goal_complete=False))
                print('INTERACTION_SUBSPACE_ACTUAL',c['tag'],c['summary'],flush=True)
            pool.close();pool.join();pool=None
        selected=actual_selection(outcomes,baseline,retained)
        result=dict(candidates=outcomes,adopted=False,goal_complete=False)
        if selected:
            result['best']=selected['tag'];torch.save(candidate_metadata(metadata,directions,selected['parameters']),stage/'research_candidate.pt')
        write_json(stage/'results.json',result)
        receipt(stage,dict(status='completed_joint_interaction_subspace_learning',best=selected['tag'] if selected else None,
            full64_candidates=len(outcomes),wall_s=time.perf_counter()-started,adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_joint_interaction_subspace_learning',error=str(exc),wall_s=time.perf_counter()-started,
            adopted=False,goal_complete=False));raise
    finally:
        progress(stage,False,'按真实完整路径结果继续修正三动作联合学习，保持正式模型、S2与最终形状泛化门槛。')


if __name__=='__main__':mp.freeze_support();main()
