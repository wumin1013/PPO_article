"""Shared three-action credit before the main guide jump on existing training paths."""
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
from joint_teacher_state_policy import features,state_pair
from observable_memory_joint_policy import dynamic_context
from localize_joint_shape_residual import characterize
from duration_joint_metrics import guards,penalty,guard_names
from learn_complete_joint_teachers import teacher_score,parameters_in_bounds
from learn_early_memory_joint_policy import worker_initialize,worker_evaluate,metric_functions
from learn_direct_observable_joint_policy import finish
from study_io import BASE,register,receipt,write_json,sha
from search_complete_return import read


def main():
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_postpeak_memory_joint_learning_20260920'
    parent_file=source/'research_candidate.pt';parent=torch.load(parent_file,map_location='cpu',weights_only=False)
    assert sha(parent_file)=='59848328bcc09eb382ffe32e3a779ab6a3f67177e2410cdb6c3ab204b6b8679c'
    previous=BASE/'canonical_gate_band_zero_margin_context_credit_20260920'
    assert not read(previous/'receipt.json')['research_acceptable']
    result=read(source/'results.json');chosen=next(c for c in result['candidates'] if c['tag']==result['best'])
    data=read(source/'baseline.json');data['baseline']=copy.deepcopy(chosen);baseline=data['baseline']['records']
    target=next(r for r in baseline if r['path']==304)
    decision=int(characterize(target['trace'])['target_jumps'][0]['time_ms'])//20-1
    stage=register('canonical_gate_band_anticipatory_context_memory_learning_20260920',
        [__file__,BASE/'observable_memory_joint_policy.py',BASE/'duration_joint_metrics.py',
         BASE/'joint_teacher_state_policy.py',BASE/'consolidate_joint_teacher_policy.py',
         BASE/'learn_early_memory_joint_policy.py',BASE/'learn_shared_temporal_joint_policy.py',
         BASE/'learn_direct_observable_joint_policy.py',parent_file,source/'receipt.json',
         source/'raw_result_audit.json',source/'saved_lateral_motion_audit.json',
         source/'energy_resolution_audit.json',previous/'receipt.json',previous/'action_rounding_audit.json'],
        dict(wall_s=600,workers=4,maximum_base_axis_vectors=36,maximum_joint_validations=3,
             maximum_optimizer_calls=48,maximum_parameters=27,context_controls=[9,18,27],
             coefficient_bounds=[-1.,1.],extra_guard_names=guard_names()),
        'Postpeak corrections leave the large pre-existing curvature peak unchanged and '
        'only improve reversal marginally. Contextual postpeak repair also fails actual '
        'guards; action-rounding audit does not explain every failed correction. Keep '
        'the verified598 research parent and shift the trainable state response to the '
        'observed decision before its largest guide-target jump. Choose a shared kernel '
        'radius1.2 times the largest nearest-state distance among the four existing155 '
        'training geometries, then audit ALL88 trajectories to identify every affected '
        'route. No time/path/angle value is an input or switch. Learn instantaneous and '
        'two causal-memory three-action weights; compare constant, speed-conditioned and '
        'speed-plus-tangential-acceleration local credit from the SAME matched full-route '
        'responses. Use the smallest representation predicting a strict feasible descent, '
        'validate up to three joint steps and run at most48 parameter-optimization calls '
        'only after an actual fully guarded improvement. All affected routes are run '
        'from start through complete braking/rest; zero-support routes are reused by '
        'deterministic induction. Original and duration/speed/cruise/multiresolution '
        'guards remain unchanged. This is full-loop parameter learning on training '
        'paths, not PPO, final shape acceptance, or independent generalization.')
    started=time.perf_counter();deadline=started+580;folder=stage/'state_joint';folder.mkdir()
    with np.load(source/'replayed_states.npz') as z:training={k:z[k].copy() for k in z.files}
    update=collect_data(parent,parent,[target]);keep=training['path_ids']!=304
    training={k:np.concatenate([v[keep],update[k]]) for k,v in training.items()}
    assert not np.any(training['labels'])
    a,b=state_pair(training['observations'],training['innovations']);both=np.vstack([a,b])
    mean=both.mean(0);std=np.maximum(both.std(0),.05);a,b=(a-mean)/std,(b-mean)/std
    index=np.flatnonzero((training['path_ids']==304)&(training['decisions']==decision)).item();center=a[index]
    distances=np.minimum(np.sqrt(np.mean((a-center)**2,axis=1)),np.sqrt(np.mean((b-center)**2,axis=1)))
    radius=1.2*max(float(distances[training['path_ids']==p].min()) for p in (14,104,204,304))
    md=dict(architecture='observable_memory_joint_radial',parent=parent,mean=mean,std=std,
        centers=center[None],radii=np.array([radius]),coefficients=np.zeros((9,3)),memory_decays=[.5,.8],
        dynamic_context_count=2,accepted=False,reused_candidates=1,
        learning_method='anticipatory_shared_complete_joint_context_credit',action_interval_ms=20,physical_tick_ms=1)
    f=features(md,training['observations'],training['innovations']);active=np.any(f!=0.,axis=(1,2))
    paths=np.unique(training['path_ids'][active]).tolist();selected=[r for r in baseline if r['path'] in paths]
    assert len(paths)==13 and all('shape' in r for r in selected)
    assert all(np.sum(active&(training['path_ids']==p))==1 for p in paths)
    indices=[np.flatnonzero(active&(training['path_ids']==p)).item() for p in paths]
    context=dynamic_context(training['observations'][indices]);ctx_mean=context.mean(0);ctx_scale=np.maximum(context.std(0),[5.,100.])
    md.update(dynamic_context_mean=ctx_mean,dynamic_context_scale=ctx_scale)
    context=np.column_stack([np.ones(len(paths)),np.tanh((context-ctx_mean)/ctx_scale)])
    choice=dict(paths=paths,widths=[radius],decisions=int(active.sum()))
    write_json(stage/'support_audit.json',dict(selected=choice,training_only_center_ms=20*decision,
        active_states=[dict(path=int(p),decision_ms=int(t*20)) for p,t in zip(training['path_ids'][active],training['decisions'][active])],
        physical_context=context.tolist(),context_mean=ctx_mean.tolist(),context_scale=ctx_scale.tolist(),
        unaffected_paths=[r['path'] for r in baseline if r['path'] not in paths]))
    np.savez_compressed(stage/'replayed_states.npz',**training);write_json(stage/'baseline.json',data)
    identity=collect_data(md,parent,[target]);assert not np.any(identity['labels'])
    write_json(stage/'zero_identity.json',dict(path=304,decisions=len(identity['decisions']),decoded_actions_bitwise=True,
        other_parent_traces_unchanged_in_prior_raw_audit=True,new_physical_rollouts=0))
    old_score,old_guards=metric_functions(data,selected,stage,parent,deadline)
    retained=[next(r['baseline_shape'] for r in data['retained'] if r['path']==b['path']) for b in selected]
    def losses(records):return np.array([teacher_score(r,b,o,barrier=False)+penalty(r) for r,b,o in zip(records,selected,retained)])
    def score(records):return float(losses(records).sum())
    def constraints(records):return np.concatenate([old_guards(records)]+[guards(r,b) for r,b in zip(records,selected)])
    zero=dict(tag='reused_identity',coefficients=[0.]*27,records=selected,score=score(selected),
              measured_inequalities=constraints(selected).tolist())
    history=[zero];cache={tuple(np.zeros(27)):zero};blocks=[];pool=None
    def feasible(c):return parameters_in_bounds(c['coefficients'],-1.,1.) and min(c['measured_inequalities'])>=-1e-8
    def best():return min(filter(feasible,history),key=lambda c:c['score'])
    print('ANTICIPATORY_CONTEXT_SUPPORT',choice,'score',zero['score'],flush=True)
    try:
        pool=mp.get_context('spawn').Pool(4,worker_initialize,(md,selected,folder,deadline))
        def evaluate(u):
            key=tuple(np.round(u,12))
            if key not in cache:
                tag=f'anticipatory_{len(history)}';records=pool.map(worker_evaluate,[(p,u,tag) for p in paths])
                c=dict(tag=tag,coefficients=np.asarray(u).tolist(),records=records,score=score(records),
                    measured_inequalities=constraints(records).tolist(),actual_rerun_paths=paths,exact_zero_support_paths=[])
                history.append(c);cache[key]=c
                write_json(folder/'results.json',dict(candidates=history,best=best()['tag']))
                if len(history)%4==0:print('ANTICIPATORY_CONTEXT',len(history)-1,'best',best()['score'],
                    'latest',c['score'],min(c['measured_inequalities']),flush=True)
            return cache[key]
        base_indices=np.array([0,1,2,9,10,11,18,19,20]);steps=np.tile([.003,.01,.01],3);pairs=[]
        for i,(column,h) in enumerate(zip(base_indices,steps)):
            u=np.zeros(27);u[column]=h;plus=evaluate(u);minus=evaluate(-u)
            size=max(np.max(abs(np.asarray(c['measured_inequalities'])-zero['measured_inequalities'])) for c in (plus,minus))
            if size>.1:
                h*=max(.001,.1/size);steps[i]=h;u[column]=h;plus=evaluate(u);minus=evaluate(-u)
            pairs.append(dict(index=i,column=int(column),step=h,plus=plus['tag'],minus=minus['tag'],initial_response=float(size)))
            write_json(stage/'matched_response_progress.json',dict(pairs=pairs,actual_vectors=len(history)-1))
        indexed={c['tag']:c for c in history};jac=[];g=[];c0=np.asarray(zero['measured_inequalities'])
        for pair in pairs:
            plus,minus=indexed[pair['plus']],indexed[pair['minus']]
            jac.append(.5*(np.asarray(plus['measured_inequalities'])-minus['measured_inequalities']))
            g.append(.5*(losses(plus['records'])-losses(minus['records'])))
        jac=np.asarray(jac).T;g=np.asarray(g).T;owners=np.r_[np.repeat(np.arange(len(paths)),11),np.repeat(np.arange(len(paths)),16)]
        models=[]
        for count in (1,2,3):
            matrix=np.concatenate([jac.reshape(-1,3,1,3)*context[owners,None,k:k+1,None] for k in range(count)],axis=2).reshape(len(c0),-1)
            gradient=np.sum(g.reshape(len(paths),3,1,3)*context[:,None,:count,None],axis=0).reshape(-1)
            scaling=np.maximum(np.maximum(np.max(abs(matrix),axis=1),abs(c0)),1e-12)
            solved=linprog(gradient,A_ub=-matrix/scaling[:,None],b_ub=c0/scaling,bounds=[(-1.,1.)]*(9*count),method='highs',
                options=dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
            models.append(dict(context_count=count,parameters=9*count,success=bool(solved.success),status=str(solved.message),
                predicted_loss_change=float(solved.fun) if solved.success else None,
                scaled_proposal=solved.x.tolist() if solved.success else None,
                predicted_guard_minimum=float((c0+matrix@solved.x).min()) if solved.success else None))
        write_json(stage/'local_credit_models.json',dict(pairs=pairs,steps=steps.tolist(),models=models,
            base_jacobian=jac.tolist(),base_per_path_loss_gradient=g.tolist(),derived_context_models_require_actual_validation=True))
        useful=[m for m in models if m['success'] and m['predicted_loss_change']<-.001 and m['predicted_guard_minimum']>=-1e-8]
        if useful:
            chosen_model=min(useful,key=lambda m:m['parameters']);count=chosen_model['context_count']
            local_steps=np.repeat(steps.reshape(3,1,3),count,axis=1).reshape(-1)
            def expand(u):
                value=np.zeros((3,3,3));value[:,:count,:]=np.asarray(u).reshape(3,count,3);return value.reshape(-1)
            direction=np.asarray(chosen_model['scaled_proposal'])
            for scale in (1.,.5,.25):
                c=evaluate(expand(local_steps*direction*scale));blocks.append(dict(method='shared_anticipatory_credit',
                    parameters=9*count,scale=scale,tag=c['tag'],predicted_change=chosen_model['predicted_loss_change']*scale,
                    actual_change=c['score']-zero['score'],actual_minimum=min(c['measured_inequalities'])))
            if best()['score']<zero['score']-.001:
                start=np.asarray(best()['coefficients']).reshape(3,3,3)[:,:count,:].reshape(-1)/local_steps
                solved=minimize(lambda z:evaluate(expand(local_steps*z))['score'],start,method='COBYLA',
                    bounds=[(-1./h,1./h) for h in local_steps],
                    constraints=[dict(type='ineq',fun=lambda z:np.asarray(evaluate(expand(local_steps*z))['measured_inequalities']))],
                    options=dict(maxiter=48,rhobeg=1.,tol=1e-6,catol=1e-8))
                blocks.append(dict(method='joint_learning_after_actual_guarded_gain',parameters=9*count,solver=str(solved.message)))
        pool.close();pool.join();pool=None
        finish(stage,md,baseline,data,best(),zero,history,blocks,choice,parent_file,started)
    except TimeoutError:
        if pool is not None:pool.terminate();pool.join();pool=None
        finish(stage,md,baseline,data,best(),zero,history,blocks,choice,parent_file,started,budget_exhausted=True)
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_anticipatory_context_memory_learning',error=str(exc),wall_s=time.perf_counter()-started,
            adopted=False,goal_complete=False));raise


if __name__=='__main__':mp.freeze_support();main()
