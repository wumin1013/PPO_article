"""Learn inside the prism of actually complete head-combination responses."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import copy
import multiprocessing as mp
import time
import numpy as np
import torch
from learn_current_head_interactions import factorial_model
from learn_full_pool_joint_credit import initialize,evaluate,candidate_metadata
from learn_full_pool_compensation import solve_response,progress
from learn_pathwise_full_credit import actual_selection,pathwise_response
from learn_trace_resolved_joint_credit import trace_deficits
from learn_retained_joint_growth import growth_deficits
from study_io import BASE,register,receipt,write_json
from search_complete_return import read


def features(x):
    a,b,c=x
    return np.array([a,b,c,a*b,a*c])


def main():
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_current_bend_capacity_learning_20260919'
    previous=BASE/'canonical_gate_band_current_head_interaction_learning_20260919'
    data=read(source/'baseline.json');baseline,retained=data['baseline'],data['retained']
    records={c['tag']:c for c in read(source/'results.json')['candidates']}
    u=np.asarray(records['bend_joint_1.0']['parameters'])
    held=records['bend_joint_0.5']
    full=copy.deepcopy(records['bend_joint_1.0']);full['parameters']=[1.,1.,1.]
    for r in full['records']:r['trace']=str(source/r['trace'])
    prior=copy.deepcopy(read(previous/'results.json')['candidates'])
    vertices=[c for c in prior if c['tag'] in ('head_100','head_010','head_110','head_011')]
    for c in vertices:
        for r in c['records']:r['trace']=str(previous/r['trace'])
    vertices.append(full)
    metadata=torch.load(source/'zero_extension.pt',map_location='cpu',weights_only=False)
    with np.load(source/'parameter_space.npz') as z:original,oldtrust=z['directions'].copy(),z['trust'].copy()
    headmap=np.zeros((18,3))
    for i in range(18):headmap[i,i%3]=u[i]
    directions=np.einsum('ih,icd->hcd',headmap,original)
    stage=register('canonical_gate_band_complete_head_prism_learning_20260919',
        [__file__,BASE/'learn_current_head_interactions.py',BASE/'learn_full_pool_compensation.py',
         BASE/'learn_full_pool_joint_credit.py',BASE/'learn_pathwise_full_credit.py',
         source/'results.json',source/'receipt.json',source/'baseline.json',source/'parameter_space.npz',
         source/'zero_extension.pt',previous/'results.json',previous/'receipt.json',previous/'validation.json'],
        dict(wall_s=180,workers=2,reused_complete_vertices=6,reused_half_validation=1,new_joint_candidates=2),
        'The full head cube has two path31 failures when preview changes without feed; no '
        'failed-vertex complete loss was imputed. Reuse six COMPLETE vertices including baseline '
        'in the parameter prism0<=preview_weight<=feed_weight<=1,0<=steering_weight<=1. '
        'This is a training parameter domain, not a runtime rule connecting physical feed and '
        'preview actions. Model signed consequences with basis[s,f,p,s*f,s*p]. Validate the '
        'cached all-half actor without fitting it, comparing against the original18-direction '
        'pathwise complete model. Require20percent lower shape error and nonworse155deg peak '
        'error before at most two complete64-path learned weight proposals. Enforce product '
        'equalities as absolute residuals and all existing physical,per-corner,aggregate,time '
        'and cruise guards. Complete vertices do not prove interior feasibility; every learned '
        'actor needs actual full checks. No new response rollouts, speed rule, PPO or adoption.')
    started=time.perf_counter();pool=None;outcomes=[];jobs=[]
    progress(stage,True,'复用完整组合建立联合响应，验证后在训练参数域内优化三动作权重。')
    try:
        model=factorial_model(stage,baseline,retained,vertices,features)
        for c in records.values():
            if c['tag'].startswith('axis'):
                for r in c['records']:r['trace']=str(source/r['trace'])
        old,_=pathwise_response(source,baseline,retained,[(records[f'axis{i}_-1'],records[f'axis{i}_+1']) for i in range(18)])
        before=trace_deficits(.5*u,old[0],old[1],old[6],baseline,retained)[0]
        after=trace_deficits(features([.5]*3),model[0],model[1],model[6],baseline,retained)[0]
        actual=growth_deficits(held,retained)
        indices=np.array([9*p+k for p in range(16) for k in range(1,7)])
        olderror=float(np.linalg.norm(before[indices]-actual[indices]));newerror=float(np.linalg.norm(after[indices]-actual[indices]))
        peak=9*14+5
        accepted=bool(newerror<.8*olderror and abs(after[peak]-actual[peak])<=abs(before[peak]-actual[peak]))
        write_json(stage/'validation.json',dict(old_error=olderror,prism_error=newerror,
            old_peak_error=float(before[peak]-actual[peak]),prism_peak_error=float(after[peak]-actual[peak]),
            accepted=accepted,held_actor='bend_joint_0.5',fit_to_held=False))
        print('COMPLETE_PRISM_VALIDATION',olderror,newerror,accepted,flush=True)
        torch.save(dict(response=model,baseline=baseline,retained=retained),stage/'response_model.pt')
        if accepted:
            trust=np.column_stack([oldtrust@headmap,np.zeros((len(oldtrust),2))])
            extras=[dict(type='eq',fun=lambda x:x[3:]-np.array([x[0]*x[1],x[0]*x[2]]),
                         jac=lambda x:np.array([[-x[1],-x[0],0.,1.,0.],[-x[2],0.,-x[0],0.,1.]])),
                    dict(type='ineq',fun=lambda x:np.array([x[1]-x[2]]),jac=lambda x:np.array([[0.,1.,-1.,0.,0.]]))]
            step,report=solve_response(baseline,retained,model,trust,False,min(started+100,time.perf_counter()+90),
                aggregate_guards=True,additional_constraints=extras)
            write_json(stage/'optimizer.json',report)
            print('COMPLETE_PRISM_SOLVER',report['message'],report['predicted_objective'],step[:3],flush=True)
            if abs(step[:3]).max()>1e-6:jobs=[(f'prism_joint_{s}',s*step[:3]) for s in (.5,1.)]
        refs,curved={},{}
        for r in baseline['records']:
            with np.load(r['trace']) as z:refs[r['path']]=z['reference'].copy()
            curved[r['path']]=r['curved_path']
        for tag,x in jobs:torch.save(candidate_metadata(metadata,directions,x),stage/(tag+'.pt'))
        if jobs:
            pool=mp.get_context('spawn').Pool(2,initialize,(stage,metadata,directions,refs,curved,retained,started+170))
            for c in pool.imap_unordered(evaluate,jobs):
                outcomes.append(c)
                write_json(stage/'results.json',dict(candidates=outcomes,adopted=False,goal_complete=False))
                print('COMPLETE_PRISM_ACTUAL',c['tag'],c['summary'],flush=True)
            pool.close();pool.join();pool=None
        selected=actual_selection(outcomes,baseline,retained)
        result=dict(candidates=outcomes,adopted=False,goal_complete=False)
        if selected:
            result['best']=selected['tag'];torch.save(candidate_metadata(metadata,directions,selected['parameters']),stage/'research_candidate.pt')
        write_json(stage/'results.json',result)
        receipt(stage,dict(status='completed_complete_head_prism_learning',best=selected['tag'] if selected else None,
            full64_candidates=len(outcomes),response_validated=accepted,wall_s=time.perf_counter()-started,adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_complete_head_prism_learning',error=str(exc),wall_s=time.perf_counter()-started,
            adopted=False,goal_complete=False));raise
    finally:
        progress(stage,False,'据当前三动作耦合的实测及学习结果继续机制修正，不把完整训练池成功替代最终验收。')


if __name__=='__main__':mp.freeze_support();main()
