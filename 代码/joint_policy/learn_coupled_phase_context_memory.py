"""Jointly learn before-jump and after-peak feedback, reusing exact one-phase evidence."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import copy
import time
import multiprocessing as mp
import numpy as np
import torch
from scipy.optimize import linprog,minimize
from consolidate_joint_teacher_policy import collect_data
from joint_teacher_state_policy import features
from observable_memory_joint_policy import dynamic_context
from duration_joint_metrics import guards,penalty
from learn_complete_joint_teachers import teacher_score,parameters_in_bounds
from learn_early_memory_joint_policy import worker_initialize,worker_evaluate,metric_functions
from learn_direct_observable_joint_policy import finish
from study_io import BASE,register,receipt,write_json,sha
from search_complete_return import read


def embed(value,kernel):
    old=np.asarray(value).reshape(3,-1,3);out=np.zeros((3,3,2,3));out[:,:old.shape[1],kernel,:]=old
    return out.reshape(-1)


def main():
    torch.set_num_threads(1)
    early=BASE/'canonical_gate_band_anticipatory_context_memory_learning_20260920'
    late=BASE/'canonical_gate_band_shared_postpeak_memory_learning_20260920'
    context_source=BASE/'canonical_gate_band_context_postpeak_memory_learning_20260920'
    assert not read(early/'receipt.json')['research_acceptable']
    pre=torch.load(early/'state_joint/candidate.pt',map_location='cpu',weights_only=False)
    post=torch.load(context_source/'state_joint/candidate.pt',map_location='cpu',weights_only=False)
    parent_file=BASE/'canonical_gate_band_postpeak_memory_joint_learning_20260920/research_candidate.pt'
    assert sha(parent_file)=='59848328bcc09eb382ffe32e3a779ab6a3f67177e2410cdb6c3ab204b6b8679c'
    md=copy.deepcopy(pre);md.update(centers=np.concatenate([pre['centers'],post['centers']]),
        radii=np.r_[pre['radii'],post['radii']],per_center_mean=np.array([pre['mean'],post['mean']]),
        per_center_std=np.array([pre['std'],post['std']]),
        per_center_context_mean=np.array([pre['dynamic_context_mean'],post['dynamic_context_mean']]),
        per_center_context_scale=np.array([pre['dynamic_context_scale'],post['dynamic_context_scale']]),
        coefficients=np.zeros((18,3)),learning_method='joint_prejump_postpeak_observable_memory_credit')
    stage=register('canonical_gate_band_coupled_phase_context_memory_learning_20260920',
        [__file__,BASE/'joint_teacher_state_policy.py',BASE/'observable_memory_joint_policy.py',
         BASE/'consolidate_joint_teacher_policy.py',BASE/'duration_joint_metrics.py',
         BASE/'learn_early_memory_joint_policy.py',BASE/'learn_shared_temporal_joint_policy.py',
         BASE/'learn_direct_observable_joint_policy.py',parent_file,early/'receipt.json',early/'local_credit_models.json',
         early/'state_joint/results.json',early/'state_joint/candidate.pt',late/'receipt.json',
         late/'state_joint/results.json',late/'local_credit_model.json',context_source/'state_joint/candidate.pt'],
        dict(wall_s=600,workers=4,phase_kernels=2,context_parameter_controls=[18,36,54],
             maximum_new_post_axis_paths=18,maximum_joint_validations=3,maximum_optimizer_calls=56,
             optimizer_calls_by_dimension={18:48,36:48,54:56},
             coefficient_bounds=[-1.,1.]),
        'Separate pre-jump and postpeak shared functions do not pass all protected '
        'outcomes. Test a SINGLE jointly optimized two-kernel causal state/history '
        'function, preserving each kernel normalization and physical-context basis. '
        'Both state responses act through all three actions; no runtime timing or '
        'route switches. Reuse all23 exact pre-only vectors after zero-extension '
        'action replay. For18 post-axis vectors, the research parent differs from '
        'old source only on304: verify every old14/104/204 decoded action under the '
        'new combined policy before reuse, physically run304 anew, and reuse all '
        'other zero-support routes. Compare18/36/54-parameter joint credit on the '
        'same measured phase responses, validate up to3 full13-route compositions '
        'to expose non-additive state interactions, and optimize at most48 calls '
        '(56 for54 parameters, the required dimension-plus-two initialization) '
        'only after a measured fully guarded gain. Keep all original physical, '
        'shape,duration,entry/exit-speed,cruise and0.2mm protections. No global '
        'root-cause, PPO-bias or generalization claim follows.')
    started=time.perf_counter();deadline=started+580;folder=stage/'state_joint';folder.mkdir()
    data=read(early/'baseline.json');baseline=data['baseline']['records'];support=read(early/'support_audit.json');choice=support['selected']
    paths=choice['paths'];selected=[r for r in baseline if r['path'] in paths]
    with np.load(early/'replayed_states.npz') as z:training={k:z[k].copy() for k in z.files}
    f=features(md,training['observations'],training['innovations']);active=np.any(f!=0.,axis=(1,2))
    assert np.unique(training['path_ids'][active]).tolist()==paths
    assert np.unique(training['path_ids'][np.any(f[:,1]!=0.,axis=1)]).tolist()==[14,104,204,304]
    contexts=np.zeros((len(paths),3,2));contexts[:,0,:]=1.
    for i,p in enumerate(paths):
        for kernel in (0,1):
            ids=np.flatnonzero((training['path_ids']==p)&np.any(f[:,kernel]!=0.,axis=1))
            assert len(ids)<=1
            if len(ids):
                raw=dynamic_context(training['observations'][ids[0]])
                contexts[i,1:,kernel]=np.tanh((raw-md['per_center_context_mean'][kernel])/md['per_center_context_scale'][kernel])
    write_json(stage/'baseline.json',data);write_json(stage/'support_audit.json',dict(selected=choice,
        phase_support_paths=[np.unique(training['path_ids'][np.any(f[:,k]!=0.,axis=1)]).tolist() for k in (0,1)],
        physical_contexts=contexts.tolist(),runtime_uses_only_observation_history=True))
    np.savez_compressed(stage/'replayed_states.npz',**training)
    _,old_guards=metric_functions(data,selected,stage,md['parent'],deadline)
    retained=[next(r['baseline_shape'] for r in data['retained'] if r['path']==b['path']) for b in selected]
    def losses(records):return np.array([teacher_score(r,b,o,barrier=False)+penalty(r) for r,b,o in zip(records,selected,retained)])
    def score(records):return float(losses(records).sum())
    def constraints(records):return np.concatenate([old_guards(records)]+[guards(r,b) for r,b in zip(records,selected)])
    old=read(early/'state_joint/results.json')['candidates'];history=[]
    for c in old:
        c=copy.deepcopy(c);c['coefficients']=embed(c['coefficients'],0).tolist();c['tag']='reused_pre_'+c['tag']
        assert abs(c['score']-score(c['records']))<1e-10
        assert np.array_equal(c['measured_inequalities'],constraints(c['records']))
        history.append(c)
    zero=history[0];cache={tuple(np.round(c['coefficients'],12)):c for c in history};md['reused_candidates']=len(history)
    check=copy.deepcopy(md);check['coefficients']=np.asarray(history[1]['coefficients']).reshape(-1,3)
    replay=collect_data(check,md['parent'],history[1]['records'])
    write_json(stage/'pre_zero_extension_identity.json',dict(paths=paths,decisions=len(replay['decisions']),decoded_actions_bitwise=True,new_physical_rollouts=0))
    def feasible(c):return parameters_in_bounds(c['coefficients'],-1.,1.) and min(c['measured_inequalities'])>=-1e-8
    def best():return min(filter(feasible,history),key=lambda c:c['score'])
    pool=None;blocks=[];reuse_audit=[]
    print('COUPLED_PHASE_START',len(history),'reused pre vectors',paths,flush=True)
    try:
        pool=mp.get_context('spawn').Pool(4,worker_initialize,(md,selected,folder,deadline))
        def evaluate(u,post_cached=None):
            key=tuple(np.round(u,12))
            if key not in cache:
                tag=f'phase_{len(history)}'
                if post_cached is not None:
                    reusable=[r for r in post_cached['records'] if r['path']!=304]
                    check=copy.deepcopy(md);check['coefficients']=np.asarray(u).reshape(-1,3)
                    replay=collect_data(check,md['parent'],reusable)
                    reuse_audit.append(dict(tag=tag,source_tag=post_cached['tag'],paths=[r['path'] for r in reusable],
                        decoded_actions_bitwise=True,decisions=len(replay['decisions'])))
                    changed=pool.map(worker_evaluate,[(304,u,tag)]);by_path={r['path']:r for r in reusable+changed}
                    records=[copy.deepcopy(by_path.get(r['path'],r)) for r in selected];actual=[304]
                    write_json(stage/'post_reuse_identity.json',reuse_audit)
                else:
                    records=pool.map(worker_evaluate,[(p,u,tag) for p in paths]);actual=paths
                c=dict(tag=tag,coefficients=np.asarray(u).tolist(),records=records,score=score(records),
                    measured_inequalities=constraints(records).tolist(),actual_rerun_paths=actual,
                    exact_zero_support_paths=[p for p in paths if p not in (14,104,204,304)] if post_cached is not None else [])
                history.append(c);cache[key]=c;write_json(folder/'results.json',dict(candidates=history,best=best()['tag']))
                print('COUPLED_PHASE',tag,'new_paths',len(actual),'score',c['score'],'guard',min(c['measured_inequalities']),flush=True)
            return cache[key]
        late_history={c['tag']:c for c in read(late/'state_joint/results.json')['candidates']}
        pre_model=read(early/'local_credit_models.json');post_model=read(late/'local_credit_model.json')
        pairs=[];steps=np.zeros((3,2,3));indexed={c['tag']:c for c in history}
        for kernel,model in enumerate((pre_model,post_model)):
            for pair in model['pairs']:
                i=pair['index'];time_block,head=divmod(i,3);steps[time_block,kernel,head]=pair['step']
                if kernel==0:
                    plus=indexed['reused_pre_'+pair['plus']];minus=indexed['reused_pre_'+pair['minus']]
                else:
                    old_plus,old_minus=late_history[pair['plus']],late_history[pair['minus']]
                    plus=evaluate(embed(old_plus['coefficients'],1),old_plus)
                    minus=evaluate(embed(old_minus['coefficients'],1),old_minus)
                pairs.append(dict(time=time_block,kernel=kernel,head=head,plus=plus['tag'],minus=minus['tag'],step=pair['step']))
        indexed={c['tag']:c for c in history};c0=np.array(zero['measured_inequalities']);jac=np.zeros((len(c0),3,2,3));g=np.zeros((len(paths),3,2,3))
        for pair in pairs:
            plus,minus=indexed[pair['plus']],indexed[pair['minus']];t,k,h=pair['time'],pair['kernel'],pair['head']
            jac[:,t,k,h]=.5*(np.asarray(plus['measured_inequalities'])-minus['measured_inequalities'])
            g[:,t,k,h]=.5*(losses(plus['records'])-losses(minus['records']))
        owners=np.r_[np.repeat(np.arange(len(paths)),11),np.repeat(np.arange(len(paths)),16)];models=[]
        for count in (1,2,3):
            matrix=np.stack([jac*contexts[owners,None,c,:,None] for c in range(count)],axis=2).reshape(len(c0),-1)
            gradient=np.sum(np.stack([g*contexts[:,None,c,:,None] for c in range(count)],axis=2),axis=0).reshape(-1)
            scaling=np.maximum(np.maximum(abs(matrix).max(1),abs(c0)),1e-12)
            solved=linprog(gradient,A_ub=-matrix/scaling[:,None],b_ub=c0/scaling,bounds=[(-1.,1.)]*(18*count),method='highs',
                options=dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
            models.append(dict(context_count=count,parameters=18*count,success=bool(solved.success),
                predicted_loss_change=float(solved.fun) if solved.success else None,
                scaled_proposal=solved.x.tolist() if solved.success else None,
                predicted_guard_minimum=float((c0+matrix@solved.x).min()) if solved.success else None))
        write_json(stage/'joint_phase_credit_models.json',dict(pairs=pairs,steps=steps.tolist(),models=models,
            base_constraints=jac.tolist(),base_per_path_loss_gradient=g.tolist(),joint_interactions_require_physical_validation=True))
        useful=[m for m in models if m['success'] and m['predicted_loss_change']<-.001 and m['predicted_guard_minimum']>=-1e-8]
        if useful:
            selected_model=min(useful,key=lambda m:m['parameters']);count=selected_model['context_count']
            local_steps=np.repeat(steps[:,None,:,:],count,axis=1).reshape(-1)
            def expand(u):
                value=np.zeros((3,3,2,3));value[:,:count]=np.asarray(u).reshape(3,count,2,3);return value.reshape(-1)
            direction=np.array(selected_model['scaled_proposal'])
            for scale in (1.,.5,.25):
                c=evaluate(expand(local_steps*direction*scale));blocks.append(dict(method='joint_phase_composition',scale=scale,tag=c['tag'],
                    predicted_change=selected_model['predicted_loss_change']*scale,actual_change=c['score']-zero['score'],
                    guard_minimum=min(c['measured_inequalities'])))
            if best()['score']<zero['score']-.001:
                start=np.asarray(best()['coefficients']).reshape(3,3,2,3)[:,:count].reshape(-1)/local_steps
                solution=minimize(lambda z:evaluate(expand(local_steps*z))['score'],start,method='COBYLA',
                    bounds=[(-1./h,1./h) for h in local_steps],
                    constraints=[dict(type='ineq',fun=lambda z:np.asarray(evaluate(expand(local_steps*z))['measured_inequalities']))],
                    options=dict(maxiter=max(48,18*count+2),rhobeg=1.,tol=1e-6,catol=1e-8))
                blocks.append(dict(method='coupled_phase_learning_after_actual_gain',solver=str(solution.message)))
        pool.close();pool.join();pool=None;finish(stage,md,baseline,data,best(),zero,history,blocks,choice,parent_file,started)
    except TimeoutError:
        if pool is not None:pool.terminate();pool.join();pool=None
        finish(stage,md,baseline,data,best(),zero,history,blocks,choice,parent_file,started,budget_exhausted=True)
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_coupled_phase_context_learning',error=str(exc),wall_s=time.perf_counter()-started,
            adopted=False,goal_complete=False));raise


if __name__=='__main__':mp.freeze_support();main()
