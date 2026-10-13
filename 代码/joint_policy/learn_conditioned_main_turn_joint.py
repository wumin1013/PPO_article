"""Add measured main-turn controllability to the earlier-gain recovery policy."""
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
from consolidate_joint_teacher_policy import collect_data
from joint_teacher_state_policy import features
from observable_memory_joint_policy import dynamic_context
from duration_joint_metrics import guards, penalty
from learn_complete_joint_teachers import teacher_score, parameters_in_bounds
from learn_early_memory_joint_policy import worker_initialize, worker_evaluate, metric_functions
from learn_direct_observable_joint_policy import finish
from study_io import BASE, register, receipt, write_json
from search_complete_return import read


def main():
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_approach_conditioned_recovery_learning_20260920'
    assert not read(source/'receipt.json')['research_acceptable']
    added=read(source/'clustered_main_turn_support_audit.json')
    old_md=torch.load(source/'state_joint/candidate.pt',map_location='cpu',weights_only=False)
    md=copy.deepcopy(old_md)
    md.update(centers=np.vstack([old_md['centers'],added['centers']]),radii=np.r_[old_md['radii'],added['radii']],
        per_center_mean=np.vstack([old_md['per_center_mean'],[added['mean']]*2]),
        per_center_std=np.vstack([old_md['per_center_std'],[added['std']]*2]),coefficients=np.zeros((90,3)),
        learning_method='main_turn_approach_conditioned_joint_response_learning')
    old_history=read(source/'state_joint/results.json')['candidates']
    old_anchor=next(c for c in old_history if c['tag']=='approach_joint_18')
    early=np.zeros((3,6,5,3));early[:,:,:3]=np.asarray(old_anchor['coefficients']).reshape(3,6,3,3)
    early=early.reshape(-1)
    parent_file=BASE/'canonical_gate_band_postpeak_memory_joint_learning_20260920/research_candidate.pt'
    stage=register('canonical_gate_band_conditioned_main_turn_joint_learning_20260920',
        [__file__,BASE/'observable_memory_joint_policy.py',BASE/'joint_teacher_state_policy.py',
         BASE/'duration_joint_metrics.py',BASE/'learn_early_memory_joint_policy.py',
         BASE/'learn_shared_temporal_joint_policy.py',BASE/'learn_direct_observable_joint_policy.py',
         BASE/'consolidate_joint_teacher_policy.py',parent_file,source/'receipt.json',
         source/'conditioned_credit_models.json',source/'conditioned_route_capacity.json',
         source/'clustered_main_turn_support_audit.json',source/'support_audit_correction.json',
         source/'state_joint/candidate.pt',source/'state_joint/results.json'],
        dict(wall_s=600,workers=4,new_main_turn_kernels=2,new_matched_axis_vectors=36,
             maximum_scale_replacement_vectors=36,maximum_joint_validations=6,
             context_controls=[1,3,4,6],coefficient_bounds=[-1.,1.]),
        'Conditional response models cannot repair104 even with independent route '
        'weights and a fourfold local box; blocking rows are reversal,curvature,'
        'outgoing excursion and heading. Add two compact main-turn observation '
        'kernels at least20ms before its raw physical curvature peak. Farthest-point '
        'training-state clustering avoids the25-route support of one broad kernel; '
        'the corrected positive radii touch8 existing corner routes. The earlier '
        'measured seven-route lateral gain stays fixed. Reuse36 later-axis responses '
        'after checking old kernel identity and nonzero action embedding, then '
        'measure only36 new main-turn axes (at most36 smaller replacements). Jointly '
        'fit old and new three-action responses with the ACTUAL earlier-gain '
        'constraint intercept; compare1/3/4/6 motion-context bases. Validate at '
        'most6 full shared-policy proposals. Preserve the earlier physical prefix '
        'and all original protected outcomes; no runtime time/path/angle table. '
        'This is complete-loop parameter learning, not PPO or final acceptance.')
    started=time.perf_counter();deadline=started+580;folder=stage/'state_joint';folder.mkdir()
    data=read(source/'baseline.json');baseline=data['baseline']['records']
    support=read(source/'support_audit.json');paths=support['selected']['paths']
    assert set(added['paths']).issubset(paths)
    selected=[r for r in baseline if r['path'] in paths]
    with np.load(source/'conditioned_full_states.npz') as z:training={k:z[k].copy() for k in z.files}
    f=features(md,training['observations'],training['innovations'])
    assert np.array_equal(f[:,:3],features(old_md,training['observations'],training['innovations']))
    means=[];scales=[]
    for k in (3,4):
        active=np.any(f[:,k]!=0.,axis=1);raw=dynamic_context(training['observations'][active])
        means.append(raw.mean(0));scales.append(np.maximum(raw.std(0),[5.,100.]))
    md.update(per_center_context_mean=np.vstack([old_md['per_center_context_mean'],means]),
              per_center_context_scale=np.vstack([old_md['per_center_context_scale'],scales]))
    trainable=[0,1,3,4];context=np.zeros((len(paths),6,4));context[:,0]=1.;phase_paths=[]
    for k in trainable:
        phase_paths.append(np.unique(training['path_ids'][np.any(f[:,k]!=0.,axis=1)]).tolist())
    for i,p in enumerate(paths):
        for j,k in enumerate(trainable):
            ids=np.flatnonzero((training['path_ids']==p)&np.any(f[:,k]!=0.,axis=1))
            assert len(ids)<=1
            if len(ids):
                assert int(training['decisions'][ids[0]])*20>=280
                v,a=np.tanh((dynamic_context(training['observations'][ids[0]])-md['per_center_context_mean'][k])/md['per_center_context_scale'][k])
                context[i,1:,j]=[v,a,v*a,v*v,a*a]
    choice=dict(paths=paths,decisions=int(np.any(f!=0.,axis=(1,2)).sum()),radii=md['radii'].tolist())
    write_json(stage/'baseline.json',data);write_json(stage/'support_audit.json',dict(selected=choice,
        phase_support_paths=phase_paths,physical_contexts=context.tolist(),trainable_kernel_indices=trainable,
        unchanged_other_paths=[r['path'] for r in baseline if r['path'] not in paths]))
    np.savez_compressed(stage/'replayed_states.npz',**training)
    _,old_guards=metric_functions(data,selected,stage,md['parent'],deadline)
    retained=[next(r['baseline_shape'] for r in data['retained'] if r['path']==b['path']) for b in selected]
    def losses(records):return np.asarray([teacher_score(r,b,o,barrier=False)+penalty(r) for r,b,o in zip(records,selected,retained)])
    def score(records):return float(losses(records).sum())
    def constraints(records):return np.concatenate([old_guards(records)]+[guards(r,b) for r,b in zip(records,selected)])
    zero=dict(tag='reused_identity',coefficients=[0.]*270,records=selected,score=score(selected),measured_inequalities=constraints(selected).tolist())
    anchor=copy.deepcopy(old_anchor);anchor['coefficients']=early.tolist()
    history=[zero,anchor];md['reused_candidates']=2;cache={tuple(np.round(c['coefficients'],12)):c for c in history}
    def feasible(c):return parameters_in_bounds(c['coefficients'],-1.,1.) and min(c['measured_inequalities'])>=-1e-8
    def best():return min(filter(feasible,history),key=lambda c:c['score'])
    pool=None;blocks=[]
    try:
        check=copy.deepcopy(md);check['coefficients']=early.reshape(-1,3)
        replay=collect_data(check,md['parent'],anchor['records'])
        # This vector has the early and old main-memory steering weights in one block.
        old_axis=next(c for c in old_history if c['tag']=='conditioned_53')
        embedded=np.zeros((3,6,5,3));embedded[:,:,:3]=np.asarray(old_axis['coefficients']).reshape(3,6,3,3)
        check['coefficients']=embedded.reshape(-1,3)
        axis_replay=collect_data(check,md['parent'],[r for r in old_axis['records'] if r['path']==304])
        write_json(stage/'nonzero_embedding_audit.json',dict(early_identity_decisions=len(replay['decisions']),
            two_nonzero_memory_terms_decisions=len(axis_replay['decisions']),old_features_bitwise_saved_states=len(f),
            all_old_axis_vectors_at_most_two_terms_per_action_block=True,new_physical_runs=0))
        print('MAIN_TURN_CONDITIONED_START',choice,'new_support',phase_paths[2:],flush=True)
        pool=mp.get_context('spawn').Pool(4,worker_initialize,(md,selected,folder,deadline))
        prefix={r['path']:r['first_later_kernel_ms'] for r in read(source/'conditioned_support_audit.json')['records']}
        def evaluate(u):
            u=np.asarray(u).reshape(-1);key=tuple(np.round(u,12))
            if key not in cache:
                assert parameters_in_bounds(u,-1.,1.)
                assert np.array_equal(u.reshape(3,6,5,3)[:,:,2],early.reshape(3,6,5,3)[:,:,2])
                offset=(u-early).reshape(3,6,5,3);touched=[np.any(offset[:,:,k]) for k in trainable]
                affected=sorted(set(p for k in range(4) if touched[k] for p in phase_paths[k]))
                tag=f'main_joint_{len(history)}';measured=pool.map(worker_evaluate,[(p,u,tag) for p in affected])
                by_path={r['path']:r for r in measured};records=[copy.deepcopy(by_path.get(r['path'],r)) for r in anchor['records']]
                for r,b in zip(records,anchor['records']):
                    if r['path'] in affected:
                        with np.load(r['trace']) as z:actual=z['rows'][:prefix[r['path']]].copy()
                        with np.load(b['trace']) as z:reference=z['rows'][:prefix[r['path']]].copy()
                        assert np.array_equal(actual,reference),('early_prefix_changed',tag,r['path'])
                c=dict(tag=tag,coefficients=u.tolist(),records=records,score=score(records),
                    measured_inequalities=constraints(records).tolist(),early_prefix_bitwise=True,
                    actual_rerun_paths=sorted(set(affected+[14,104,105,204,205,304,305])),
                    evaluated_paths_this_vector=affected,exact_zero_support_paths=[p for p in paths if p not in affected and p not in (14,104,105,204,205,304,305)])
                history.append(c);cache[key]=c;write_json(folder/'results.json',dict(candidates=history,best=best()['tag']))
                print('MAIN_TURN_CONDITIONED',tag,c['score'],min(c['measured_inequalities']),flush=True)
            return cache[key]
        old_model=read(source/'conditioned_credit_models.json');c0=np.asarray(anchor['measured_inequalities'])
        jac=np.zeros((len(c0),3,4,3));g=np.zeros((len(paths),3,4,3));steps=np.zeros((3,4,3))
        jac[:,:,:2]=old_model['base_constraints'];g[:,:,:2]=old_model['base_loss_gradient'];steps[:,:2]=old_model['steps'];steps[:,2:]=[.003,.01,.01]
        pairs=[]
        for t in range(3):
            for j,k in enumerate(trainable[2:],2):
                for h in range(3):
                    step=steps[t,j,h];delta=np.zeros((3,6,5,3));delta[t,0,k,h]=step
                    plus=evaluate(early+delta.reshape(-1));minus=evaluate(early-delta.reshape(-1))
                    size=max(np.max(abs(np.asarray(c['measured_inequalities'])-c0)) for c in (plus,minus))
                    if size>.1:
                        factor=max(.001,.1/size);step*=factor;steps[t,j,h]=step;delta*=factor
                        plus=evaluate(early+delta.reshape(-1));minus=evaluate(early-delta.reshape(-1))
                    jac[:,t,j,h]=.5*(np.asarray(plus['measured_inequalities'])-minus['measured_inequalities'])
                    g[:,t,j,h]=.5*(losses(plus['records'])-losses(minus['records']))
                    pairs.append(dict(time=t,local_kernel=j,policy_kernel=k,head=h,step=step,plus=plus['tag'],minus=minus['tag'],initial_response=float(size)))
                    write_json(stage/'main_turn_responses.json',dict(pairs=pairs,steps=steps.tolist()))
        owners=np.r_[np.repeat(np.arange(len(paths)),11),np.repeat(np.arange(len(paths)),16)];models=[]
        for count in (1,3,4,6):
            matrix=np.stack([jac*context[owners,None,c,:,None] for c in range(count)],axis=2).reshape(len(c0),-1)
            gradient=np.sum(np.stack([g*context[:,None,c,:,None] for c in range(count)],axis=2),axis=0).reshape(-1)
            scaling=np.maximum(np.maximum(abs(matrix).max(1),abs(c0)),1e-12)
            solved=linprog(gradient,A_ub=-matrix/scaling[:,None],b_ub=c0/scaling,bounds=[(-1.,1.)]*(36*count),method='highs',
                options=dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
            row=dict(context_count=count,parameters=36*count,success=bool(solved.success),status=str(solved.message),
                predicted_loss=float(anchor['score']+solved.fun) if solved.success else None,
                scaled_proposal=solved.x.tolist() if solved.success else None,
                predicted_guard_minimum=float((c0+matrix@solved.x).min()) if solved.success else None)
            models.append(row);print('MAIN_TURN_CREDIT',{k:v for k,v in row.items() if k!='scaled_proposal'},flush=True)
        write_json(stage/'joint_main_turn_credit_models.json',dict(models=models,steps=steps.tolist(),pairs=pairs,
            base_constraints=jac.tolist(),base_loss_gradient=g.tolist(),constraint_intercept=c0.tolist()))
        for model in [m for m in models if m['success'] and m['predicted_loss']<zero['score']-.001][:2]:
            count=model['context_count'];value=np.asarray(model['scaled_proposal']).reshape(3,count,4,3)*steps[:,None]
            direction=np.zeros((3,6,5,3))
            for j,k in enumerate(trainable):direction[:,:count,k]=value[:,:,j]
            for scale in (1.,.5,.25):
                c=evaluate(early+scale*direction.reshape(-1));blocks.append(dict(context_count=count,scale=scale,tag=c['tag'],
                    score=c['score'],guard_minimum=min(c['measured_inequalities']),early_prefix_bitwise=True))
                if best()['score']<zero['score']-.001:break
            if best()['score']<zero['score']-.001:break
        pool.close();pool.join();pool=None
        finish(stage,md,baseline,data,best(),zero,history,blocks,choice,parent_file,started)
    except TimeoutError:
        if pool is not None:pool.terminate();pool.join();pool=None
        finish(stage,md,baseline,data,best(),zero,history,blocks,choice,parent_file,started,budget_exhausted=True)
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_conditioned_main_turn_joint_learning',error=str(exc),wall_s=time.perf_counter()-started,
            adopted=False,goal_complete=False));raise


if __name__=='__main__':mp.freeze_support();main()
