"""Learn joint causal corrections at main-turn onset and post-apex recovery."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import copy
import time
import multiprocessing as mp
import numpy as np
import torch
from scipy.optimize import minimize,linprog
from consolidate_joint_teacher_policy import collect_data
from joint_teacher_state_policy import state_pair,features
from localize_joint_shape_residual import characterize
from learn_early_memory_joint_policy import worker_initialize,worker_evaluate,metric_functions
from learn_direct_observable_joint_policy import finish
from learn_complete_joint_teachers import parameters_in_bounds
from study_io import BASE,register,receipt,write_json,sha
from search_complete_return import read


def main():
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_compact_memory_joint_credit_learning_20260920'
    parent_file=source/'research_candidate.pt';audit=read(source/'raw_result_audit.json')
    assert audit['model_sha256']==sha(parent_file) and audit['changed_paths']==[304]
    result=read(source/'results.json');chosen=next(c for c in result['candidates'] if c['tag']==result['best'])
    data=read(source/'baseline.json');data['baseline']=copy.deepcopy(chosen)
    baseline=data['baseline']['records'];target=next(r for r in baseline if r['path']==304)
    location=characterize(target['trace'])
    jump=int(location['target_jumps'][0]['time_ms'])//20-1
    reverse=max((r for r in location['negative_turn_intervals'] if r['start_ms']>location['apex_ms']),key=lambda r:r['reverse_deg'])
    recovery=int(reverse['start_ms'])//20-1
    assert 0<=jump<recovery
    stage=register('canonical_gate_band_main_turn_memory_joint_learning_20260920',
        [__file__,BASE/'observable_memory_joint_policy.py',BASE/'joint_teacher_state_policy.py',
         BASE/'consolidate_joint_teacher_policy.py',BASE/'learn_early_memory_joint_policy.py',
         BASE/'learn_direct_observable_joint_policy.py',BASE/'learn_shared_temporal_joint_policy.py',
         BASE/'learn_complete_joint_teachers.py',BASE/'multi_corner_raw_metrics.py',parent_file,
         source/'receipt.json',source/'raw_result_audit.json',source/'saved_lateral_motion_audit.json',
         source/'results.json',source/'replayed_states.npz'],
        dict(wall_s=480,workers=3,coefficient_bounds=[-1.,1.],parameters=18,
             maximum_axis_vectors=72,maximum_composed_vectors=6,maximum_optimizer_calls=80,
             nominal_axis_steps=[.003,.01,.01],maximum_normalized_axis_response=.1,
             training_path=304,center_decisions_for_sample_selection=[jump,recovery]),
        'The accepted local training update changed only304 and improved early lateral/exit '
        'metrics slightly, but left the primary155deg turn curvature and roughly0.166mm '
        'outgoing excursion unresolved. Its kernel is tied to an early observed state, '
        'well before the dominant requested-heading jump and post-apex reverse interval. '
        'Freeze that verified research parent and learn a new THREE-ACTION causal memory '
        'function on two actual observable states: one decision before the largest target '
        'jump, and one before the dominant post-apex reversal. These indices select training '
        'samples only and are not runtime inputs/switches. Each kernel radius is half the '
        'nearest other-route saved-state distance. Audit all88 paths; replay only changed304 '
        'states with the full parent policy history. Reuse all87 zero-support routes by '
        'deterministic induction, and execute the entire affected route from start through '
        'braking/rest for each vector. Learn18 constant/fast-memory/slow-memory coefficients '
        'jointly. Obtain symmetric physical responses, shrink overly large axis responses '
        'once, form linear and diagonal-conservative complete-return credit models, and '
        'independently validate up to6 composed updates. Only after a real protected gain '
        'run up to80 scaled COBYLA calls within the same480s budget. Preserve every original '
        'guard and the formally retained86cdfe model. This localized state-function capacity '
        'study does not demonstrate generalization, ideal tangency, or a PPO estimator property.')
    started=time.perf_counter();deadline=started+460;folder=stage/'state_joint';folder.mkdir()
    parent=torch.load(parent_file,map_location='cpu',weights_only=False)
    with np.load(source/'replayed_states.npz') as z:training={k:z[k].copy() for k in z.files}
    update=collect_data(parent,parent,[target]);keep=training['path_ids']!=304
    training={k:np.concatenate([v[keep],update[k]]) for k,v in training.items()}
    assert not np.any(training['labels'])
    np.savez_compressed(stage/'replayed_states.npz',**training)
    a,b=state_pair(training['observations'],training['innovations']);both=np.vstack([a,b])
    mean=both.mean(0);std=np.maximum(both.std(0),.05);a,b=(a-mean)/std,(b-mean)/std
    selected_indices=[np.flatnonzero((training['path_ids']==304)&(training['decisions']==d)).item() for d in (jump,recovery)]
    centers=a[selected_indices];other=training['path_ids']!=304
    radii=np.array([.5*np.minimum(np.sqrt(np.mean((a-c)**2,axis=1)),np.sqrt(np.mean((b-c)**2,axis=1)))[other].min() for c in centers])
    assert np.all(radii>1e-10),'Indistinguishable observed states require a different representation'
    md=dict(architecture='observable_memory_joint_radial',parent=parent,mean=mean,std=std,centers=centers,radii=radii,
        coefficients=np.zeros((6,3)),memory_decays=[.5,.8],accepted=False,reused_candidates=1,
        learning_method='main_turn_causal_memory_complete_return_joint_learning',action_interval_ms=20,physical_tick_ms=1)
    f=features(md,training['observations'],training['innovations']);active=np.any(f!=0.,axis=(1,2))
    paths=np.unique(training['path_ids'][active]).tolist();assert paths==[304]
    choice=dict(paths=paths,widths=radii.tolist(),decisions=int(active.sum()))
    write_json(stage/'support_audit.json',dict(selected=choice,training_only_center_decisions=[jump,recovery],
        parent_state_action_replay=True,parent_trace_localization=location,
        active_decisions_ms=(20*training['decisions'][active]).tolist(),
        unaffected_paths=[r['path'] for r in baseline if r['path'] not in paths]))
    write_json(stage/'baseline.json',data)
    selected=[target];score,constraints=metric_functions(data,selected,stage,parent,deadline)
    zero=dict(tag='reused_identity',coefficients=[0.]*18,records=selected,score=score(selected),measured_inequalities=constraints(selected).tolist())
    history=[zero];cache={tuple(np.zeros(18)):zero};pool=None;blocks=[]
    print('MAIN_TURN_SUPPORT',choice,'centers_ms',[jump*20,recovery*20],'score',zero['score'],flush=True)
    def feasible(c):return parameters_in_bounds(c['coefficients'],-1.,1.) and min(c['measured_inequalities'])>=-1e-8
    try:
        pool=mp.get_context('spawn').Pool(3,worker_initialize,(md,selected,folder,deadline))
        def evaluate(u):
            key=tuple(np.round(u,12))
            if key not in cache:
                tag=f'main_{len(history)}'
                records=pool.map(worker_evaluate,[(304,u,tag)])
                c=dict(tag=tag,coefficients=np.asarray(u).tolist(),records=records,score=score(records),
                    measured_inequalities=constraints(records).tolist(),actual_rerun_paths=[304],exact_zero_support_paths=[])
                history.append(c);cache[key]=c
                best=min(filter(feasible,history),key=lambda c:c['score'])
                write_json(folder/'results.json',dict(candidates=history,best=best['tag']))
                if len(history)%5==0:print('MAIN_TURN',len(history)-1,'best',best['score'],'latest',c['score'],min(c['measured_inequalities']),flush=True)
            return cache[key]
        steps=np.tile([.003,.01,.01],6);pairs=[]
        for i,h in enumerate(steps):
            u=np.zeros(18);u[i]=h;plus=evaluate(u);minus=evaluate(-u)
            size=max(np.max(abs(np.asarray(c['measured_inequalities'])-zero['measured_inequalities'])) for c in (plus,minus))
            if size>.1:
                h*=max(.001,.1/size);steps[i]=h;u[i]=h;plus=evaluate(u);minus=evaluate(-u)
            pairs.append(dict(index=i,step=h,plus=plus['tag'],minus=minus['tag'],initial_normalized_response=float(size)))
            write_json(stage/'matched_response_progress.json',dict(pairs=pairs,completed_axes=len(pairs),actual_new_vectors=len(history)-1))
        central=[];even=[];g=[];q=[];indexed={c['tag']:c for c in history}
        for p in pairs:
            plus,minus=indexed[p['plus']],indexed[p['minus']]
            a,b=np.asarray(plus['measured_inequalities']),np.asarray(minus['measured_inequalities'])
            central.append(.5*(a-b));even.append(abs(.5*(a+b)-zero['measured_inequalities']))
            g.append(.5*(plus['score']-minus['score']));q.append(max(0.,.5*(plus['score']+minus['score'])-zero['score']))
        jac=np.array(central).T;curvature=np.array(even).T;g,q=np.array(g),np.array(q);c0=np.array(zero['measured_inequalities'])
        linear=linprog(g,A_ub=-jac,b_ub=c0,bounds=[(-1.,1.)]*18,method='highs');proposals=[]
        if linear.success and linear.fun<-.001:proposals.append(('linear',linear.x))
        predicted=lambda z:c0+jac@z-curvature@(z*z)
        objective=lambda z:float(g@z+q@(z*z))
        fitted=minimize(objective,.25*linear.x if proposals else np.zeros(18),method='SLSQP',bounds=[(-1.,1.)]*18,
            constraints=[dict(type='ineq',fun=predicted)],options=dict(maxiter=500,ftol=1e-10))
        if objective(fitted.x)<-.001 and predicted(fitted.x).min()>=-1e-7:proposals.insert(0,('diagonal_conservative',fitted.x))
        write_json(stage/'local_credit_model.json',dict(pairs=pairs,steps=steps.tolist(),central_constraints=jac.tolist(),
            absolute_even_constraints=curvature.tolist(),central_loss=g.tolist(),nonnegative_even_loss=q.tolist(),
            linear_predicted_loss_change=float(linear.fun) if linear.success else None,
            conservative_predicted_loss_change=objective(fitted.x),conservative_minimum=float(predicted(fitted.x).min()),
            proposals=[dict(method=m,scaled_parameters=z.tolist()) for m,z in proposals]))
        for method,z in proposals:
            for scale in (1.,.5,.25):
                c=evaluate(steps*z*scale);blocks.append(dict(method=method,scale=scale,tag=c['tag'],
                    actual_loss_change=c['score']-zero['score'],predicted_loss_change=objective(scale*z),
                    actual_minimum=min(c['measured_inequalities']),passes_original_guards=feasible(c)))
                write_json(stage/'composed_credit_validation.json',blocks)
        best=min(filter(feasible,history),key=lambda c:c['score'])
        if best['score']<zero['score']-.001:
            learned=minimize(lambda z:evaluate(steps*z)['score'],np.asarray(best['coefficients'])/steps,method='COBYLA',
                bounds=[(-1./h,1./h) for h in steps],
                constraints=[dict(type='ineq',fun=lambda z:np.asarray(evaluate(steps*z)['measured_inequalities']))],
                options=dict(maxiter=80,rhobeg=2.,tol=1e-6,catol=1e-8))
            blocks.append(dict(method='joint_learning_after_verified_gain',solver=str(learned.message)))
        pool.close();pool.join();pool=None
        best=min(filter(feasible,history),key=lambda c:c['score'])
        finish(stage,md,baseline,data,best,zero,history,blocks,choice,parent_file,started)
    except TimeoutError:
        if pool is not None:pool.terminate();pool.join();pool=None
        best=min(filter(feasible,history),key=lambda c:c['score'])
        finish(stage,md,baseline,data,best,zero,history,blocks,choice,parent_file,started,budget_exhausted=True)
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_main_turn_memory_learning',error=str(exc),wall_s=time.perf_counter()-started,
            adopted=False,goal_complete=False));raise


if __name__=='__main__':mp.freeze_support();main()
