"""Learn missing pre-peak three-action compensation with the earlier policy fixed."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import copy
import time
import multiprocessing as mp
import numpy as np
import torch
from joint_teacher_state_policy import features
from observable_memory_joint_policy import dynamic_context,future_bend_context
from consolidate_joint_teacher_policy import collect_data
from audit_context_restoration_scaling import solve_correction
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
    source=BASE/'canonical_gate_band_event_recentered_preemptive_credit_20260920'
    origin=BASE/'canonical_gate_band_preemptive_joint_credit_learning_20260920'
    parent_file=BASE/'canonical_gate_band_postpeak_memory_joint_learning_20260920/research_candidate.pt'
    assert sha(parent_file)=='59848328bcc09eb382ffe32e3a779ab6a3f67177e2410cdb6c3ab204b6b8679c'
    md=torch.load(origin/'initial_policy.pt',map_location='cpu',weights_only=False)
    old=read(source/'state_joint/results.json')['candidates']
    anchor=copy.deepcopy(next(c for c in old if c['tag']==read(source/'recovery.json')['nearest']))
    zero=copy.deepcopy(old[0]);history=[zero,anchor];md['reused_candidates']=2
    selected=zero['records'];paths=[r['path'] for r in selected];n=len(paths)
    data=read(source/'baseline.json');baseline=data['baseline']['records']
    kernels=[3,4];shape=(3,7,7,3);u0=np.asarray(anchor['coefficients']).reshape(shape)
    assert not np.any(u0[:,:,kernels])
    stage=register('canonical_gate_band_preapex_joint_compensation_20260920',
        [__file__,BASE/'shape_event_credit.py',BASE/'observable_memory_joint_policy.py',
         BASE/'joint_teacher_state_policy.py',BASE/'duration_joint_metrics.py',
         BASE/'learn_intrinsic_joint_recovery.py',BASE/'audit_context_restoration_scaling.py',
         BASE/'consolidate_joint_teacher_policy.py',BASE/'learn_early_memory_joint_policy.py',
         BASE/'learn_direct_observable_joint_policy.py',parent_file,source/'receipt.json',
         source/'state_joint/results.json',source/'remaining_support_audit.json',origin/'initial_policy.pt'],
        dict(wall_s=240,workers=4,symmetric_vectors=36,context_counts=[1,3,7],
             maximum_joint_vectors=9,kernels=kernels,coefficient_bounds=[-1.,1.],
             finite_difference_steps=[.001,.003,.003]),
        'Joint early-state response learning leaves six actual violations at '
        'event_joint_78, including105 curvature/apex speed and204 outgoing excursion. '
        'Saved support audit shows the previous stage froze observable kernels3/4 '
        'activating near300ms before actual curvature peaks332-342ms. Fix ALL '
        'existing coefficients at that failed exploratory anchor, retaining the '
        'protected postpeak policy as the embedded parent. Learn the previously '
        'zero weights on the two existing pre-peak kernels and causal memories '
        'jointly across three actions. Collect36 +/- complete physical vectors, '
        'compare constant,linear and quadratic visible-context corrections from '
        'the same measurements, then at most6 joint secant repairs. Each physical '
        'run must preserve the raw0:300ms prefix bitwise, retaining actual early '
        'lateral improvement. Keep every original measured and intrinsic guard. '
        'No acceptance relaxation, PPO, path/time policy input or sealed test. '
        'Stop if the missing action support cannot restore feasibility in240s.')
    started=time.perf_counter();deadline=started+225;folder=stage/'state_joint';folder.mkdir()
    write_json(stage/'baseline.json',data)
    support=read(source/'support_audit.json');support['selected']['variable_kernels']=kernels
    write_json(stage/'support_audit.json',support);choice=support['selected']
    _,old_guards=metric_functions(data,selected,stage,md['parent'],deadline)
    retained=[next(r['baseline_shape'] for r in data['retained'] if r['path']==b['path']) for b in selected]
    def losses(records):return np.array([teacher_score(r,b,o,barrier=False)+penalty(r) for r,b,o in zip(records,selected,retained)])
    def entry(records,u,tag):
        original=np.concatenate([old_guards(records)]+[guards(r,b) for r,b in zip(records,selected)])
        extra=intrinsic_guards(records,selected)
        return dict(tag=tag,coefficients=np.asarray(u).reshape(-1).tolist(),records=records,
            score=float(losses(records).sum()),measured_inequalities=original.tolist(),
            augmented_measured_inequalities=np.r_[original,extra].tolist(),
            augmented_learning_inequalities=np.r_[shape_event_learning_guards(records,selected,original),extra].tolist(),
            actual_rerun_paths=paths,exact_zero_support_paths=[],early_prefix_bitwise=True)
    def feasible(c):return parameters_in_bounds(c['coefficients'],-1.,1.) and min(c['augmented_measured_inequalities'])>=-1e-8
    def best():return min(filter(feasible,history),key=lambda c:c['score'])
    def violation(c):return float(np.linalg.norm(np.minimum(c['augmented_measured_inequalities'],0.)))
    blocks=[];pool=None
    try:
        actual=copy.deepcopy(md);actual['coefficients']=u0.reshape(-1,3)
        replay=collect_data(actual,actual['parent'],anchor['records'],deadline=deadline)
        np.savez_compressed(stage/'anchor_replay.npz',**replay)
        f=features(md,replay['observations'],replay['innovations']);context=np.zeros((n,7,2));context[:,0]=1.
        active=[];prefix={}
        for i,r in enumerate(anchor['records']):
            p=r['path']
            with np.load(r['trace']) as z:prefix[p]=z['rows'][:300].copy()
            for j,k in enumerate(kernels):
                ids=np.flatnonzero((replay['path_ids']==p)&np.any(f[:,k]!=0.,axis=1))
                assert len(ids)<=1
                if len(ids):
                    assert int(replay['decisions'][ids[0]])==15
                    x=replay['observations'][ids[0]]
                    v,a=np.tanh((dynamic_context(x)-md['per_center_context_mean'][k])/md['per_center_context_scale'][k])
                    g=np.tanh((future_bend_context(x)-md['per_center_geometry_mean'][k])/md['per_center_geometry_scale'][k])
                    context[i,1:,j]=[v,a,v*a,v*v,a*a,g]
                    active.append(dict(path=p,kernel=k,decision_ms=300,activation=float(abs(f[ids[0],k]).max())))
        write_json(stage/'anchor_context_audit.json',dict(records=active,context=context.tolist(),
            decoded_actions_bitwise=True,decisions=len(replay['decisions'])))
        cache={tuple(np.round(c['coefficients'],12)):c for c in history}
        pool=mp.get_context('spawn').Pool(4,worker_initialize,(md,selected,folder,deadline))
        def evaluate(u,kind):
            u=np.asarray(u).reshape(-1);key=tuple(np.round(u,12))
            if key not in cache:
                assert parameters_in_bounds(u,-1.,1.) and len(history)-2<45
                tag=f'preapex_joint_{len(history)-2}'
                records=pool.map(worker_evaluate,[(p,u,tag) for p in paths])
                for r in records:
                    with np.load(r['trace']) as z:assert np.array_equal(z['rows'][:300],prefix[r['path']]),('prefix',r['path'])
                c=entry(records,u,tag);c['kind']=kind;history.append(c);cache[key]=c
                write_json(folder/'results.json',dict(candidates=history,best=best()['tag']))
                print('PREAPEX_JOINT',tag,kind,c['score'],violation(c),flush=True)
            return cache[key]
        steps=np.broadcast_to(np.array([.001,.003,.003]),(3,2,3)).copy()
        size=len(zero['augmented_learning_inequalities']);jac=np.zeros((size,3,2,3));grad=np.zeros((n,3,2,3));pairs=[]
        for t in range(3):
            for j,k in enumerate(kernels):
                for h in range(3):
                    delta=np.zeros(shape);delta[t,0,k,h]=steps[t,j,h]
                    plus=evaluate(u0+delta,'local_plus');minus=evaluate(u0-delta,'local_minus')
                    jac[:,t,j,h]=.5*(np.asarray(plus['augmented_learning_inequalities'])-minus['augmented_learning_inequalities'])
                    grad[:,t,j,h]=.5*(losses(plus['records'])-losses(minus['records']))
                    pairs.append(dict(time=t,kernel=k,head=h,plus=plus['tag'],minus=minus['tag']))
                    write_json(stage/'local_response.json',dict(pairs=pairs,steps=steps.tolist(),jacobian=jac.tolist(),loss_gradient=grad.tolist()))
        owners=np.r_[np.repeat(np.arange(n),11),np.repeat(np.arange(n),16),np.repeat(np.arange(n),2)]
        def expand(v,count):
            out=np.zeros(shape);part=np.asarray(v).reshape(3,count,2,3)*steps[:,None]
            for j,k in enumerate(kernels):out[:,:count,k]=part[:,:,j]
            return out.reshape(-1)
        def solve(matrix,g,current):
            for radius in (1.,4.,16.):
                result=solve_correction(matrix,g,np.asarray(current['augmented_learning_inequalities']),
                    min(0.,zero['score']-.001-current['score']),radius=radius)
                if result['success']:return result,radius
            return result,radius
        variants=[]
        for count in (1,3,7):
            matrix=np.stack([jac*context[owners,None,c,:,None] for c in range(count)],axis=2).reshape(size,-1)
            g=np.sum(np.stack([grad*context[:,None,c,:,None] for c in range(count)],axis=2),axis=0).reshape(-1)
            result,radius=solve(matrix,g,anchor)
            row=dict(context_count=count,anchor=anchor['tag'],radius=radius,**result);blocks.append(row)
            if not result['success']:continue
            step=np.asarray(result['scaled_parameter_step'])
            c=evaluate(np.asarray(anchor['coefficients'])+expand(step,count),f'context_{count}')
            row.update(tag=c['tag'],violation=violation(c))
            matrix+=np.outer(np.asarray(c['augmented_learning_inequalities'])-anchor['augmented_learning_inequalities']-matrix@step,step)/(step@step)
            g+=(c['score']-anchor['score']-g@step)*step/(step@step)
            variants.append((c,count,matrix,g))
        if variants:
            current,count,matrix,g=min(variants,key=lambda x:violation(x[0]))
            for iteration in range(6):
                if feasible(current):break
                result,radius=solve(matrix,g,current)
                row=dict(iteration=iteration,context_count=count,anchor=current['tag'],radius=radius,**result);blocks.append(row)
                if not result['success']:break
                step=np.asarray(result['scaled_parameter_step'])
                c=evaluate(np.asarray(current['coefficients'])+expand(step,count),'joint_secant')
                row.update(tag=c['tag'],violation=violation(c))
                matrix+=np.outer(np.asarray(c['augmented_learning_inequalities'])-current['augmented_learning_inequalities']-matrix@step,step)/(step@step)
                g+=(c['score']-current['score']-g@step)*step/(step@step)
                if violation(c)<violation(current):current=c
                write_json(stage/'recovery.json',dict(blocks=blocks,nearest=current['tag']))
        pool.close();pool.join();pool=None
        finish(stage,md,baseline,data,best(),zero,history,blocks,choice,parent_file,started)
    except TimeoutError:
        if pool is not None:pool.terminate();pool.join();pool=None
        finish(stage,md,baseline,data,best(),zero,history,blocks,choice,parent_file,started,budget_exhausted=True)
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_preapex_compensation',error=str(exc),wall_s=time.perf_counter()-started,
            actual_new_candidates=len(history)-2,adopted=False,goal_complete=False));raise


if __name__=='__main__':mp.freeze_support();main()
