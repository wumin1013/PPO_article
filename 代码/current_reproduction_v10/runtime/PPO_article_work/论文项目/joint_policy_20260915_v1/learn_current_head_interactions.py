"""Full-policy factorial interaction credit followed by learned joint weights."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import copy
import itertools
import multiprocessing as mp
import time
import numpy as np
import torch
from learn_full_pool_joint_credit import initialize,evaluate,candidate_metadata
from learn_full_pool_compensation import solve_response,progress
from learn_pathwise_full_credit import actual_selection
from learn_trace_resolved_joint_credit import signed_channels,read_rows,trace_deficits
from learn_retained_joint_growth import growth_deficits
from study_io import BASE,register,receipt,write_json,sha
from search_complete_return import read


def polynomial(x):
    a,b,c=x
    return np.array([a,b,c,a*b,a*c,b*c,a*b*c])


def product_constraints(x):
    a,b,c=x[:3]
    f=np.array([x[3]-a*b,x[4]-a*c,x[5]-b*c,x[6]-a*b*c])
    j=np.zeros((4,7));j[:,3:]=np.eye(4)
    j[:,:3]=[[-b,-a,0.],[-c,0.,-a],[0.,-c,-b],[-b*c,-a*c,-a*b]]
    return f,j


def factorial_model(stage,baseline,retained,vertices,feature_fn=polynomial):
    assert all(c['summary']['successes']==64 for c in vertices)
    a=np.array([feature_fn(c['parameters']) for c in vertices])
    inverse=np.linalg.inv(a)
    base=growth_deficits(baseline,retained)
    matrix=(inverse@np.array([growth_deficits(c,retained)-base for c in vertices])).T
    bm=np.array([r['corridor']['margin_lower_mm'] for r in baseline['records']])
    mj=(inverse@np.array([np.array([r['corridor']['margin_lower_mm'] for r in c['records']])-bm for c in vertices])).T
    traces={}
    for r in baseline['records']:
        if 'shape' not in r:continue
        ref,rows=read_rows(stage,r);b=signed_channels(ref,rows);differences={k:[] for k in ('incoming','outgoing','omega','turn')}
        for c in vertices:
            row=next(t for t in c['records'] if t['path']==r['path'])
            _,ys=read_rows(stage,row)
            channels=signed_channels(ref,ys,b['grid'],np.arange(len(rows)))
            for key in differences:differences[key].append(channels[key]-b[key])
        traces[r['path']]=dict(base=b,jac={k:(inverse@np.array(v)).T for k,v in differences.items()})
    return base,matrix,bm,mj,[(0.,1.)]*a.shape[1],['factorial_complete']*a.shape[1],traces


def predict(model,baseline,retained,x):
    base,matrix,_,_,_,_,traces=model
    return trace_deficits(polynomial(x),base,matrix,traces,baseline,retained)[0]


def main():
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_current_bend_capacity_learning_20260919'
    data=read(source/'baseline.json');baseline,retained=data['baseline'],data['retained']
    results={c['tag']:c for c in read(source/'results.json')['candidates']}
    full=copy.deepcopy(results['bend_joint_1.0']);held=copy.deepcopy(results['bend_joint_0.5'])
    metadata=torch.load(source/'zero_extension.pt',map_location='cpu',weights_only=False)
    with np.load(source/'parameter_space.npz') as z:original,oldtrust=z['directions'].copy(),z['trust'].copy()
    u=np.array(full['parameters']);headmap=np.zeros((18,3))
    for i in range(18):headmap[i,i%3]=u[i]
    directions=np.einsum('ih,icd->hcd',headmap,original)
    assert np.array_equal(candidate_metadata(metadata,directions,np.ones(3))['coefficients'],
                          torch.load(source/'bend_joint_1.0.pt',map_location='cpu',weights_only=False)['coefficients'])
    for c in (full,held):
        for r in c['records']:r['trace']=str(source/r['trace'])
    full['parameters']=[1.,1.,1.];held['parameters']=[.5,.5,.5]
    stage=register('canonical_gate_band_current_head_interaction_learning_20260919',
        [__file__,BASE/'learn_full_pool_joint_credit.py',BASE/'learn_full_pool_compensation.py',
         BASE/'learn_pathwise_full_credit.py',source/'results.json',source/'receipt.json',
         source/'baseline.json',source/'parameter_space.npz',source/'zero_extension.pt',source/'bend_joint_1.0.pt'],
        dict(wall_s=300,workers=4,new_factorial_candidates=6,new_joint_candidates=2,
             cached_zero_full_and_half=3,paths_per_candidate=64,optimizer_s=100),
        'The18-parameter joint step preserved64/64 but reversed more despite modeled guards. '
        'Measure current cross-head interactions before increasing capacity or training. At this '
        'exact parent and proposed learned correction, reuse zero,all-three-full,and all-three-half '
        'actors. Complete the six missing steering/feed/preview on/off vertices with full64 rollouts. '
        'If all seven nonzero vertices complete, fit a multilinear model to signed actualXY, '
        'angular/spatial turning and complete scalar/margin consequences, including all pairwise '
        'and three-way interactions. Validate on cached half actor without fitting it, comparing '
        'against the same single-head additive model. Use interaction credit for two learned '
        'bounded head-weight proposals only if shape prediction error falls20percent and155deg '
        'peak error is not worse. Optimize only inside the observed head-weight cube with all '
        'physical,per-corner,aggregate,time and cruise guards. These are learned parameter weights '
        'within continuous state functions, not temporal switches,action labels or speed rules. '
        'A failed vertex is retained as a failure, never assigned complete loss or imputed traces. '
        'This is parameter optimization plus a mechanism control, not PPO or final adoption.')
    started=time.perf_counter();pool=None;outcomes=[]
    progress(stage,True,'补齐当前三动作参数的完整闭环交互对照，验证交互信用后联合学习。')
    try:
        refs,curved={},{}
        for r in baseline['records']:
            with np.load(r['trace']) as z:refs[r['path']]=z['reference'].copy()
            curved[r['path']]=r['curved_path']
        pool=mp.get_context('spawn').Pool(4,initialize,(stage,metadata,directions,refs,curved,retained,started+290))
        jobs=[('head_'+''.join(map(str,bits)),np.array(bits,float)) for bits in itertools.product((0,1),repeat=3) if sum(bits) in (1,2)]
        for c in pool.imap_unordered(evaluate,jobs):
            outcomes.append(c)
            write_json(stage/'results.json',dict(candidates=outcomes,adopted=False,goal_complete=False))
            print('HEAD_FACTORIAL',c['tag'],c['summary'],flush=True)
        vertices=outcomes+[full]
        accepted=False;jobs=[]
        if all(c['summary']['successes']==64 for c in vertices):
            model=factorial_model(stage,baseline,retained,vertices)
            additive=copy.deepcopy(model)
            additive[1][:,3:]=0.;additive[3][:,3:]=0.
            for m in additive[6].values():
                for j in m['jac'].values():j[:,3:]=0.
            indices=np.array([9*p+k for p in range(16) for k in range(1,7)])
            actual=growth_deficits(held,retained)
            before=predict(additive,baseline,retained,[.5]*3);after=predict(model,baseline,retained,[.5]*3)
            olderror=float(np.linalg.norm(before[indices]-actual[indices]));newerror=float(np.linalg.norm(after[indices]-actual[indices]))
            peak=9*14+5
            accepted=bool(newerror<.8*olderror and abs(after[peak]-actual[peak])<=abs(before[peak]-actual[peak]))
            write_json(stage/'validation.json',dict(additive_error=olderror,interaction_error=newerror,
                additive_peak_error=float(before[peak]-actual[peak]),interaction_peak_error=float(after[peak]-actual[peak]),
                accepted=accepted,held_actor='bend_joint_0.5',fit_to_held=False))
            print('HEAD_INTERACTION_VALIDATION',olderror,newerror,accepted,flush=True)
            torch.save(dict(response=model,baseline=baseline,retained=retained),stage/'response_model.pt')
            if accepted:
                trust=np.column_stack([oldtrust@headmap,np.zeros((len(oldtrust),4))])
                equality=dict(type='eq',fun=lambda x:product_constraints(x)[0],jac=lambda x:product_constraints(x)[1])
                step,report=solve_response(baseline,retained,model,trust,False,min(started+220,time.perf_counter()+100),
                    aggregate_guards=True,additional_constraints=[equality])
                write_json(stage/'optimizer.json',report)
                print('HEAD_INTERACTION_SOLVER',report['message'],report['predicted_objective'],step[:3],flush=True)
                if np.max(abs(step[:3]))>1e-6:
                    jobs=[(f'interaction_joint_{scale}',scale*step[:3]) for scale in (.5,1.)]
        else:
            write_json(stage/'validation.json',dict(accepted=False,reason='Incomplete vertices; no complete-factorial loss model',
                failed=[dict(tag=c['tag'],paths=[r['path'] for r in c['records'] if not r['success']]) for c in vertices if c['summary']['successes']!=64]))
        for tag,x in jobs:torch.save(candidate_metadata(metadata,directions,x),stage/(tag+'.pt'))
        for c in pool.imap_unordered(evaluate,jobs):
            outcomes.append(c)
            write_json(stage/'results.json',dict(candidates=outcomes,adopted=False,goal_complete=False))
            print('HEAD_INTERACTION_ACTUAL',c['tag'],c['summary'],flush=True)
        pool.close();pool.join();pool=None
        selected=actual_selection(outcomes,baseline,retained)
        result=dict(candidates=outcomes,adopted=False,goal_complete=False)
        if selected:
            result['best']=selected['tag']
            torch.save(candidate_metadata(metadata,directions,selected['parameters']),stage/'research_candidate.pt')
        write_json(stage/'results.json',result)
        receipt(stage,dict(status='completed_current_head_interaction_learning',best=selected['tag'] if selected else None,
            full64_candidates=len(outcomes),interaction_model_accepted=accepted,wall_s=time.perf_counter()-started,
            adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_current_head_interaction_learning',error=str(exc),full64_candidates=len(outcomes),
            wall_s=time.perf_counter()-started,adopted=False,goal_complete=False));raise
    finally:
        progress(stage,False,'依据实测三动作交互与完整64路径更新继续学习；正式模型、物理边界与最终验收不变。')


if __name__=='__main__':mp.freeze_support();main()
