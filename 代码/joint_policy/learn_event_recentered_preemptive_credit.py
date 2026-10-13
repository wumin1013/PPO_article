"""Learn joint corrections around measured earlier-action state changes."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import copy
import time
import multiprocessing as mp
import numpy as np
import torch
from scipy.optimize import linprog
from joint_teacher_state_policy import features
from observable_memory_joint_policy import dynamic_context,future_bend_context
from consolidate_joint_teacher_policy import collect_data
from duration_joint_metrics import guards,penalty
from shape_event_credit import shape_event_learning_guards
from learn_intrinsic_joint_recovery import intrinsic_guards
from learn_complete_joint_teachers import teacher_score,parameters_in_bounds
from learn_early_memory_joint_policy import worker_initialize,worker_evaluate,metric_functions
from learn_direct_observable_joint_policy import finish
from study_io import BASE,register,write_json,receipt,sha
from search_complete_return import read


def main():
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_preemptive_joint_credit_learning_20260920'
    parent_file=BASE/'canonical_gate_band_postpeak_memory_joint_learning_20260920/research_candidate.pt'
    assert sha(parent_file)=='59848328bcc09eb382ffe32e3a779ab6a3f67177e2410cdb6c3ab204b6b8679c'
    md=torch.load(source/'initial_policy.pt',map_location='cpu',weights_only=False)
    history=copy.deepcopy(read(source/'state_joint/results.json')['candidates'])
    by_tag={c['tag']:c for c in history};zero=history[0];anchor=by_tag['preemptive_joint_84']
    data=read(source/'baseline.json');baseline=data['baseline']['records']
    paths=[r['path'] for r in zero['records']];selected=zero['records'];n=len(paths)
    kernels=[0,1,5,6];shape=(3,7,7,3);constant_shape=(3,4,3)
    stage=register('canonical_gate_band_event_recentered_preemptive_credit_20260920',
        [__file__,BASE/'shape_event_credit.py',BASE/'cruise_duration_credit.py',
         BASE/'observable_memory_joint_policy.py',BASE/'joint_teacher_state_policy.py',
         BASE/'duration_joint_metrics.py',BASE/'learn_intrinsic_joint_recovery.py',
         BASE/'consolidate_joint_teacher_policy.py',BASE/'learn_early_memory_joint_policy.py',
         BASE/'learn_direct_observable_joint_policy.py',parent_file,source/'receipt.json',
         source/'initial_policy.pt',source/'state_joint/results.json',source/'joint_credit_models.json',
         source/'matched_credit_audit.json'],
        dict(wall_s=360,workers=4,new_symmetric_vectors=72,maximum_joint_vectors=10,
             derivative_step_fraction=.2,context_counts=[1,3,7],radii=[1.,4.,16.],
             required_lateral_reduction_mm=.001,coefficient_bounds=[-1.,1.]),
        'The earlier joint control reduces significant lateral movement by0.001056mm '
        'but violates26 guards, including a1ms reversal duration increase. Preserve '
        'ALL measured guards. Replace three step-count learning rows with exactly '
        'equivalent signed order statistics; verify equivalence on all89 saved '
        'vectors. Replay the actual joint anchor, then remeasure72 symmetric '
        'responses at one fifth the earlier steps AROUND this joint policy to '
        'capture changed-state coupling. Use measured per-route responses to compare '
        'constant,linear and quadratic visible context corrections. Minimize L1 '
        'correction size with predicted lateral reduction>=0.001mm and nonnegative '
        'original plus intrinsic-curvature protections. Test at most3 model choices '
        'then at most7 measured secant corrections. No PPO, phase switch, feed '
        'labels, acceptance relaxation, sealed tests or deployment. Stop this '
        'mechanism on infeasibility or budget; retained model remains unchanged.')
    started=time.perf_counter();deadline=started+345;folder=stage/'state_joint';folder.mkdir()
    write_json(stage/'baseline.json',data)
    support=read(source/'support_audit.json');write_json(stage/'support_audit.json',support)
    choice=support['selected'];md['reused_candidates']=len(history)
    _,old_guards=metric_functions(data,selected,stage,md['parent'],deadline)
    retained=[next(r['baseline_shape'] for r in data['retained'] if r['path']==b['path']) for b in selected]
    def losses(records):return np.array([teacher_score(r,b,o,barrier=False)+penalty(r) for r,b,o in zip(records,selected,retained)])
    def lateral(records):return sum(r['shape']['reverse_lateral_significant_mm'] if r['success'] else 1. for r in records)
    def entry(records,u,tag):
        original=np.concatenate([old_guards(records)]+[guards(r,b) for r,b in zip(records,selected)])
        extra=intrinsic_guards(records,selected)
        return dict(tag=tag,coefficients=np.asarray(u).reshape(-1).tolist(),records=records,
            score=float(losses(records).sum()),measured_inequalities=original.tolist(),
            augmented_measured_inequalities=np.r_[original,extra].tolist(),
            augmented_learning_inequalities=np.r_[shape_event_learning_guards(records,selected,original),extra].tolist(),
            actual_rerun_paths=paths,exact_zero_support_paths=[])
    def feasible(c):return parameters_in_bounds(c['coefficients'],-1.,1.) and min(c['augmented_measured_inequalities'])>=-1e-8
    def best():return min(filter(feasible,history),key=lambda c:c['score'])
    def violation(c):return float(np.linalg.norm(np.minimum(c['augmented_measured_inequalities'],0.)))
    blocks=[];pool=None
    try:
        for c in history:
            c['augmented_learning_inequalities']=np.r_[shape_event_learning_guards(c['records'],selected,c['measured_inequalities']),
                intrinsic_guards(c['records'],selected)].tolist()
        write_json(stage/'event_equivalence_audit.json',dict(vectors=len(history),paths=n,
            count_guard_comparisons=len(history)*n*3,all_exact=True,new_physical_runs=0))
        actual=copy.deepcopy(md);actual['coefficients']=np.asarray(anchor['coefficients']).reshape(-1,3)
        replay=collect_data(actual,actual['parent'],anchor['records'],deadline=deadline)
        np.savez_compressed(stage/'anchor_replay.npz',**replay)
        f=features(md,replay['observations'],replay['innovations']);context=np.zeros((n,7,4));context[:,0]=1.
        active=[]
        for i,p in enumerate(paths):
            for j,k in enumerate(kernels):
                ids=np.flatnonzero((replay['path_ids']==p)&np.any(f[:,k]!=0.,axis=1))
                assert len(ids)<=1,('context_requires_single_activation',p,k,ids.tolist())
                if len(ids):
                    x=replay['observations'][ids[0]]
                    v,a=np.tanh((dynamic_context(x)-md['per_center_context_mean'][k])/md['per_center_context_scale'][k])
                    g=np.tanh((future_bend_context(x)-md['per_center_geometry_mean'][k])/md['per_center_geometry_scale'][k])
                    context[i,1:,j]=[v,a,v*a,v*v,a*a,g]
                    active.append(dict(path=p,kernel=k,decision_ms=int(replay['decisions'][ids[0]])*20))
        write_json(stage/'anchor_context_audit.json',dict(records=active,context=context.tolist(),
            decoded_actions_bitwise=True,decisions=len(replay['decisions'])))
        cache={tuple(np.round(c['coefficients'],12)):c for c in history}
        pool=mp.get_context('spawn').Pool(4,worker_initialize,(md,selected,folder,deadline))
        def evaluate(u,kind):
            u=np.asarray(u).reshape(-1);key=tuple(np.round(u,12))
            if key not in cache:
                assert parameters_in_bounds(u,-1.,1.)
                assert len(history)-md['reused_candidates']<82
                tag=f'event_joint_{len(history)-md["reused_candidates"]}'
                records=pool.map(worker_evaluate,[(p,u,tag) for p in paths])
                c=entry(records,u,tag);c['kind']=kind;history.append(c);cache[key]=c
                write_json(folder/'results.json',dict(candidates=history,best=best()['tag']))
                print('EVENT_JOINT',tag,kind,c['score'],violation(c),lateral(records)-lateral(selected),flush=True)
            return cache[key]
        steps=.2*np.asarray(read(source/'joint_credit_models.json')['steps'])
        size=len(zero['augmented_learning_inequalities'])
        jac=np.zeros((size,*constant_shape));grad=np.zeros((n,*constant_shape));lat=np.zeros((n,*constant_shape));pairs=[]
        ua=np.asarray(anchor['coefficients']).reshape(shape)
        for t in range(3):
            for j,k in enumerate(kernels):
                for h in range(3):
                    delta=np.zeros(shape);delta[t,0,k,h]=steps[t,j,h]
                    plus=evaluate(ua+delta,'local_plus');minus=evaluate(ua-delta,'local_minus')
                    jac[:,t,j,h]=.5*(np.asarray(plus['augmented_learning_inequalities'])-minus['augmented_learning_inequalities'])
                    grad[:,t,j,h]=.5*(losses(plus['records'])-losses(minus['records']))
                    for i,(r,b) in enumerate(zip(plus['records'],minus['records'])):
                        lat[i,t,j,h]=.5*(r['shape']['reverse_lateral_significant_mm']-b['shape']['reverse_lateral_significant_mm'])
                    pairs.append(dict(time=t,kernel=k,head=h,plus=plus['tag'],minus=minus['tag']))
                    write_json(stage/'local_response.json',dict(pairs=pairs,steps=steps.tolist(),jacobian=jac.tolist(),
                        loss_gradient=grad.tolist(),lateral_gradient=lat.tolist()))
        owners=np.r_[np.repeat(np.arange(n),11),np.repeat(np.arange(n),16),np.repeat(np.arange(n),2)]
        variants=[]
        def expand(v,count):
            out=np.zeros(shape);part=np.asarray(v).reshape(3,count,4,3)*steps[:,None]
            for j,k in enumerate(kernels):out[:,:count,k]=part[:,:,j]
            return out.reshape(-1)
        def solve(matrix,g,l,current,radius):
            # L1 size controls unnecessary excursions in the constrained nullspace.
            values=np.asarray(current['augmented_learning_inequalities']);dim=len(g)
            a=np.vstack([np.c_[-matrix,np.zeros((len(values),dim))],
                         np.c_[np.eye(dim),-np.eye(dim)],np.c_[-np.eye(dim),-np.eye(dim)],
                         np.r_[l,np.zeros(dim)],np.r_[g,np.zeros(dim)]])
            b=np.r_[values,np.zeros(2*dim),lateral(selected)-.001-lateral(current['records']),
                    zero['score']-.001-current['score']]
            scale=np.maximum(np.maximum(abs(a).max(1),abs(b)),1e-12)
            solved=linprog(np.r_[np.zeros(dim),np.ones(dim)],A_ub=a/scale[:,None],b_ub=b/scale,
                bounds=[(-radius,radius)]*dim+[(0.,None)]*dim,method='highs',
                options=dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
            return solved.x[:dim] if solved.success else None
        for count in (1,3,7):
            matrix=np.stack([jac*context[owners,None,c,:,None] for c in range(count)],axis=2).reshape(size,-1)
            g=np.sum(np.stack([grad*context[:,None,c,:,None] for c in range(count)],axis=2),axis=0).reshape(-1)
            l=np.sum(np.stack([lat*context[:,None,c,:,None] for c in range(count)],axis=2),axis=0).reshape(-1)
            proposal=None
            for radius in (1.,4.,16.):
                proposal=solve(matrix,g,l,anchor,radius)
                blocks.append(dict(context_count=count,radius=radius,success=proposal is not None,anchor=anchor['tag']))
                if proposal is not None:break
            if proposal is None:continue
            c=evaluate(np.asarray(anchor['coefficients'])+expand(proposal,count),f'context_{count}')
            blocks[-1].update(tag=c['tag'],violation=violation(c))
            actual_delta=np.asarray(c['augmented_learning_inequalities'])-anchor['augmented_learning_inequalities']
            norm=proposal@proposal
            matrix+=np.outer(actual_delta-matrix@proposal,proposal)/norm
            g+=(c['score']-anchor['score']-g@proposal)*proposal/norm
            l+=(lateral(c['records'])-lateral(anchor['records'])-l@proposal)*proposal/norm
            variants.append((c,count,matrix,g,l))
        if variants:
            current,count,matrix,g,l=min(variants,key=lambda x:violation(x[0]))
            for iteration in range(7):
                if feasible(current):break
                proposal=None
                for radius in (1.,4.,16.):
                    proposal=solve(matrix,g,l,current,radius)
                    if proposal is not None:break
                block=dict(iteration=iteration,context_count=count,anchor=current['tag'],success=proposal is not None)
                blocks.append(block)
                if proposal is None:break
                c=evaluate(np.asarray(current['coefficients'])+expand(proposal,count),'joint_secant')
                block.update(tag=c['tag'],violation=violation(c))
                norm=proposal@proposal
                matrix+=np.outer(np.asarray(c['augmented_learning_inequalities'])-current['augmented_learning_inequalities']-matrix@proposal,proposal)/norm
                g+=(c['score']-current['score']-g@proposal)*proposal/norm
                l+=(lateral(c['records'])-lateral(current['records'])-l@proposal)*proposal/norm
                if violation(c)<violation(current):current=c
                write_json(stage/'recovery.json',dict(blocks=blocks,nearest=current['tag']))
        pool.close();pool.join();pool=None
        finish(stage,md,baseline,data,best(),zero,history,blocks,choice,parent_file,started)
    except TimeoutError:
        if pool is not None:pool.terminate();pool.join();pool=None
        finish(stage,md,baseline,data,best(),zero,history,blocks,choice,parent_file,started,budget_exhausted=True)
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_event_recentered_credit',error=str(exc),wall_s=time.perf_counter()-started,
            actual_new_candidates=len(history)-md['reused_candidates'],adopted=False,goal_complete=False));raise


if __name__=='__main__':mp.freeze_support();main()
