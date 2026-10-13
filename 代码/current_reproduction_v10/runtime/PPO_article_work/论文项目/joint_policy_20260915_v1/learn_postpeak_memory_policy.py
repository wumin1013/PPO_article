"""Learn state-conditioned joint credit after the curvature peak and before maximum excursion."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import copy
import time
import multiprocessing as mp
import numpy as np
import torch
from scipy.optimize import minimize, linprog
from consolidate_joint_teacher_policy import collect_data
from joint_teacher_state_policy import features, state_pair
from localize_joint_shape_residual import characterize
from geometric_turn_metrics import spatial_turn_profile
from duration_joint_metrics import guards, guard_names, penalty
from learn_early_memory_joint_policy import worker_initialize, worker_evaluate, metric_functions
from learn_direct_observable_joint_policy import finish
from learn_complete_joint_teachers import parameters_in_bounds
from study_io import BASE, register, receipt, write_json, sha
from search_complete_return import read


def main():
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_duration_guarded_memory_learning_20260920'
    previous=BASE/'canonical_gate_band_coupled_peak_memory_learning_20260920'
    reference=BASE/'canonical_gate_band_main_turn_memory_joint_learning_20260920'
    assert not read(source/'receipt.json')['research_acceptable']
    assert read(source/'memory_capacity_audit.json')['new_physical_rollouts']==0
    parent_file=reference/'research_candidate.pt';parent=torch.load(parent_file,map_location='cpu',weights_only=False)
    assert sha(parent_file)=='ab5f53db4bd77754aa6d6186a85b65a2a724e4f78cb1715948c60bdbc3ac2503'
    data=read(source/'baseline.json');baseline=data['baseline']['records'];selected=[r for r in baseline if r['path']==304]
    location=characterize(selected[0]['trace'])
    decision=int(location['outgoing_min_ms'])//20
    with np.load(selected[0]['trace']) as z:ref,rows=z['reference'].copy(),z['rows'].copy()
    xy=np.vstack([ref[0],rows[:,:2]]);profile=spatial_turn_profile(xy)
    peak_arc=profile['grid'][1:-1][abs(profile['turn']).argmax()]
    raw_arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
    peak_ms=int(np.searchsorted(raw_arc,peak_arc))-1
    assert peak_ms<20*decision<location['outgoing_min_ms']
    stage=register('canonical_gate_band_postpeak_memory_joint_learning_20260920',
        [__file__,BASE/'duration_joint_metrics.py',BASE/'observable_memory_joint_policy.py',
         BASE/'joint_teacher_state_policy.py',BASE/'consolidate_joint_teacher_policy.py',
         BASE/'learn_early_memory_joint_policy.py',BASE/'learn_direct_observable_joint_policy.py',
         BASE/'learn_shared_temporal_joint_policy.py',parent_file,source/'receipt.json',
         source/'memory_capacity_audit.json',source/'state_joint/results.json',
         previous/'replayed_states.npz',previous/'raw_result_audit.json'],
        dict(wall_s=400,maximum_axis_evaluations=36,maximum_composed_evaluations=6,
             maximum_optimizer_calls=64,parameters=9,coefficient_bounds=[-1.,1.],
             center_selection_decision=decision,extra_guard_names=guard_names()),
        'The27-weight duration-guarded run failed; saved exact action replay shows '
        'independent state kernels only at280/320/380ms, while curvature peaks341ms and '
        'exit excursion reaches its maximum379ms. Freeze the protected ab5 research '
        'parent and add ONE observed-state kernel sampled at the last decision before '
        'maximum outgoing excursion, strictly after curvature peak. Its9 learned '
        'instantaneous/fast/slow-memory three-action weights can adjust recovery without '
        'changing the recorded approach, apex or peak. Radius is half the nearest '
        'other-route OR earlier-target-state distance. Sample index is offline selection '
        'only; runtime sees state/history, never time/path labels. Reuse87 exactly '
        'zero-support routes. Verify zero action extension and then measure symmetric '
        'full-route physical responses of all9 parameters, fit constrained local credit, '
        'validate at most6 joint proposals, and optimize at most64 calls only if measured '
        'responses predict a feasible descent. Preserve every old and added duration, '
        'speed,cruise,0.2mm and physical guard. This is a local representation/credit '
        'capacity test with learned parameters, not generalization or PPO training.')
    started=time.perf_counter();deadline=started+380;folder=stage/'state_joint';folder.mkdir()
    with np.load(previous/'replayed_states.npz') as z:training={k:z[k].copy() for k in z.files}
    # All other routes have bitwise identical observations/actions under the frozen parent.
    raw=read(previous/'raw_result_audit.json');assert raw['unchanged_other_paths_bitwise']
    update=collect_data(parent,parent,selected);keep=training['path_ids']!=304
    training={k:np.concatenate([v[keep],update[k]]) for k,v in training.items()}
    assert not np.any(training['labels'])
    a,b=state_pair(training['observations'],training['innovations']);both=np.vstack([a,b])
    mean=both.mean(0);std=np.maximum(both.std(0),.05);a,b=(a-mean)/std,(b-mean)/std
    index=np.flatnonzero((training['path_ids']==304)&(training['decisions']==decision)).item()
    center=a[index];distance=np.minimum(np.sqrt(np.mean((a-center)**2,axis=1)),np.sqrt(np.mean((b-center)**2,axis=1)))
    protected=(training['path_ids']!=304)|(training['decisions']<decision)
    radius=.5*float(distance[protected].min());assert radius>1e-10
    md=dict(architecture='observable_memory_joint_radial',parent=parent,mean=mean,std=std,
        centers=center[None],radii=np.array([radius]),coefficients=np.zeros((3,3)),
        memory_decays=[.5,.8],accepted=False,reused_candidates=1,
        learning_method='postpeak_observable_state_complete_joint_credit',action_interval_ms=20,physical_tick_ms=1)
    f=features(md,training['observations'],training['innovations']);active=np.any(f!=0.,axis=(1,2))
    assert np.unique(training['path_ids'][active]).tolist()==[304]
    assert np.all(training['decisions'][active]>=decision)
    identity=collect_data(md,parent,selected);assert not np.any(identity['labels'])
    choice=dict(paths=[304],widths=[radius],decisions=int(active.sum()))
    write_json(stage/'support_audit.json',dict(selected=choice,actual_zero_action_replay_decisions=len(identity['decisions']),
        original_peak_ms=peak_ms,original_excursion_max_ms=location['outgoing_min_ms'],
        training_only_center_ms=20*decision,active_decisions_ms=(20*training['decisions'][active]).tolist(),
        protected_prefix_before_ms=20*decision,unaffected_paths=[r['path'] for r in baseline if r['path']!=304]))
    np.savez_compressed(stage/'replayed_states.npz',**training);write_json(stage/'baseline.json',data)
    old_score,old_guards=metric_functions(data,selected,stage,parent,deadline)
    def score(records):return old_score(records)+sum(penalty(r) for r in records)
    def constraints(records):return np.concatenate([old_guards(records)]+[guards(r,b) for r,b in zip(records,selected)])
    zero=dict(tag='reused_identity',coefficients=[0.]*9,records=selected,score=score(selected),
              measured_inequalities=constraints(selected).tolist())
    history=[zero];cache={tuple(np.zeros(9)):zero};blocks=[];pool=None
    def feasible(c):return parameters_in_bounds(c['coefficients'],-1.,1.) and min(c['measured_inequalities'])>=-1e-8
    print('POSTPEAK_SUPPORT',choice,20*decision,'peak',peak_ms,'score',zero['score'],flush=True)
    try:
        pool=mp.get_context('spawn').Pool(1,worker_initialize,(md,selected,folder,deadline))
        def evaluate(u):
            key=tuple(np.round(u,12))
            if key not in cache:
                tag=f'postpeak_{len(history)}';records=pool.map(worker_evaluate,[(304,u,tag)])
                # Full rollout supplies an independent causal prefix identity check.
                with np.load(records[0]['trace']) as z:
                    assert np.array_equal(z['rows'][:decision*20],rows[:decision*20])
                c=dict(tag=tag,coefficients=np.asarray(u).tolist(),records=records,score=score(records),
                    measured_inequalities=constraints(records).tolist(),actual_rerun_paths=[304],
                    exact_zero_support_paths=[],prefix_motion_bitwise=True)
                history.append(c);cache[key]=c;best=min(filter(feasible,history),key=lambda c:c['score'])
                write_json(folder/'results.json',dict(candidates=history,best=best['tag']))
                if len(history)%5==0:print('POSTPEAK',len(history)-1,'best',best['score'],'latest',c['score'],min(c['measured_inequalities']),flush=True)
            return cache[key]
        steps=np.tile([.003,.01,.01],3);pairs=[]
        for i,h in enumerate(steps):
            u=np.zeros(9);u[i]=h;plus=evaluate(u);minus=evaluate(-u)
            size=max(np.max(abs(np.asarray(c['measured_inequalities'])-zero['measured_inequalities'])) for c in (plus,minus))
            if size>.1:
                h*=max(.001,.1/size);steps[i]=h;u[i]=h;plus=evaluate(u);minus=evaluate(-u)
            pairs.append(dict(index=i,step=h,plus=plus['tag'],minus=minus['tag'],initial_response=float(size)))
        indexed={c['tag']:c for c in history};g=[];q=[];jac=[];even=[];c0=np.asarray(zero['measured_inequalities'])
        for p in pairs:
            plus,minus=indexed[p['plus']],indexed[p['minus']]
            a,b=np.asarray(plus['measured_inequalities']),np.asarray(minus['measured_inequalities'])
            jac.append(.5*(a-b));even.append(abs(.5*(a+b)-c0))
            g.append(.5*(plus['score']-minus['score']));q.append(max(0.,.5*(plus['score']+minus['score'])-zero['score']))
        jac,even,g,q=np.array(jac).T,np.array(even).T,np.array(g),np.array(q)
        linear=linprog(g,A_ub=-jac,b_ub=c0,bounds=[(-1.,1.)]*9,method='highs');proposals=[]
        if linear.success and linear.fun<-.001:proposals.append(('linear',linear.x))
        predict=lambda z:c0+jac@z-even@(z*z)
        objective=lambda z:float(g@z+q@(z*z))
        fit=minimize(objective,.25*linear.x if proposals else np.zeros(9),method='SLSQP',bounds=[(-1.,1.)]*9,
            constraints=[dict(type='ineq',fun=predict)],options=dict(maxiter=300,ftol=1e-10))
        if objective(fit.x)<-.001 and predict(fit.x).min()>=-1e-7:proposals.insert(0,('diagonal_conservative',fit.x))
        write_json(stage/'local_credit_model.json',dict(pairs=pairs,steps=steps.tolist(),central_constraints=jac.tolist(),
            absolute_even_constraints=even.tolist(),central_loss=g.tolist(),nonnegative_even_loss=q.tolist(),
            linear_predicted_change=float(linear.fun) if linear.success else None,
            conservative_predicted_change=objective(fit.x),conservative_guard_minimum=float(predict(fit.x).min())))
        for method,z in proposals:
            for scale in (1.,.5,.25):
                c=evaluate(steps*z*scale);blocks.append(dict(method=method,scale=scale,tag=c['tag'],
                    predicted_change=objective(scale*z),actual_change=c['score']-zero['score'],
                    measured_guard_minimum=min(c['measured_inequalities'])))
        if proposals:
            best=min(filter(feasible,history),key=lambda c:c['score'])
            solved=minimize(lambda z:evaluate(steps*z)['score'],np.asarray(best['coefficients'])/steps,method='COBYLA',
                bounds=[(-1./h,1./h) for h in steps],
                constraints=[dict(type='ineq',fun=lambda z:np.asarray(evaluate(steps*z)['measured_inequalities']))],
                options=dict(maxiter=64,rhobeg=1.,tol=1e-6,catol=1e-8))
            blocks.append(dict(method='joint_learning_after_response_validation',solver=str(solved.message)))
        pool.close();pool.join();pool=None;best=min(filter(feasible,history),key=lambda c:c['score'])
        finish(stage,md,baseline,data,best,zero,history,blocks,choice,parent_file,started)
    except TimeoutError:
        if pool is not None:pool.terminate();pool.join();pool=None
        best=min(filter(feasible,history),key=lambda c:c['score'])
        finish(stage,md,baseline,data,best,zero,history,blocks,choice,parent_file,started,budget_exhausted=True)
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_postpeak_memory_joint_learning',error=str(exc),wall_s=time.perf_counter()-started,
            adopted=False,goal_complete=False));raise


if __name__=='__main__':mp.freeze_support();main()
