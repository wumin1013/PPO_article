"""Test and learn a shared postpeak three-action memory on four training geometries."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import copy
import argparse
import time
import multiprocessing as mp
import numpy as np
import torch
from scipy.optimize import minimize,linprog
from joint_teacher_state_policy import features
from consolidate_joint_teacher_policy import collect_data
from duration_joint_metrics import guards,guard_names,penalty
from learn_early_memory_joint_policy import worker_initialize,worker_evaluate,metric_functions
from learn_direct_observable_joint_policy import finish
from learn_complete_joint_teachers import parameters_in_bounds
from study_io import BASE,register,receipt,write_json,sha
from search_complete_return import read


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dynamic-context',action='store_true')
    args=parser.parse_args();context_count=3 if args.dynamic_context else 1;dimensions=9*context_count
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_postpeak_memory_joint_learning_20260920'
    parent_file=BASE/'canonical_gate_band_main_turn_memory_joint_learning_20260920/research_candidate.pt'
    assert read(source/'raw_result_audit.json')['unchanged_other_paths_bitwise']
    assert read(source/'receipt.json')['research_acceptable']
    md=torch.load(source/'research_candidate.pt',map_location='cpu',weights_only=False)
    seed=np.asarray(md['coefficients']).reshape(-1).copy();md['radii']=6.*md['radii']
    response_source=BASE/'canonical_gate_band_shared_postpeak_memory_learning_20260920'
    context_audit=None
    def extend(value):
        result=np.zeros((3,context_count,3));result[:,0,:]=np.asarray(value).reshape(3,3)
        return result.reshape(-1)
    if args.dynamic_context:
        context_audit=read(response_source/'context_credit_audit.json')
        assert context_audit['selected_context_count']==3
        md.update(dynamic_context_count=2,dynamic_context_mean=np.array(context_audit['context_mean']),
                  dynamic_context_scale=np.array(context_audit['context_scale']),coefficients=extend(seed).reshape(-1,3))
        seed=extend(seed)
    data=read(source/'baseline.json');baseline=data['baseline']['records']
    with np.load(source/'replayed_states.npz') as z:training={k:z[k].copy() for k in z.files}
    f=features(md,training['observations'],training['innovations']);active=np.any(f!=0.,axis=(1,2))
    paths=np.unique(training['path_ids'][active]).tolist();assert paths==[14,104,204,304]
    selected=[r for r in baseline if r['path'] in paths]
    stage=register('canonical_gate_band_context_postpeak_memory_learning_20260920' if args.dynamic_context else
                   'canonical_gate_band_shared_postpeak_memory_learning_20260920',
        [__file__,BASE/'duration_joint_metrics.py',BASE/'observable_memory_joint_policy.py',
         BASE/'joint_teacher_state_policy.py',BASE/'learn_early_memory_joint_policy.py',
         BASE/'learn_shared_temporal_joint_policy.py',BASE/'learn_direct_observable_joint_policy.py',
         source/'research_candidate.pt',source/'receipt.json',source/'raw_result_audit.json',
         source/'saved_lateral_motion_audit.json',source/'support_audit.json',parent_file]+
         ([response_source/'receipt.json',response_source/'context_credit_audit.json',
           response_source/'state_joint/results.json',BASE/'audit_postpeak_context_credit.py'] if args.dynamic_context else []),
        dict(wall_s=480,workers=4,parameters=dimensions,kernel_width_multiplier=6.,paths=paths,
             maximum_axis_evaluations=2*dimensions*2,maximum_composed_evaluations=6,maximum_optimizer_calls=48,
             coefficient_bounds=[-1.,1.],extra_guard_names=guard_names()),
        ('Measured wide-kernel constant weights have no shared feasible credit direction. '
         'The SAME18 matched complete responses predict no direction with speed alone, but '
         'a feasible joint direction with speed AND tangential acceleration. Implement '
         'these two reflection-even continuous physical contexts on the radial feature '
         'before the causal memory update, preserving zero-context-weight identity. '
         'Reuse every19 prior parameter vector by exact zero extension; verify decoded '
         'actions across all four saved transfer routes. Measure the added context axes '
         'in the real full loop, compare their derivatives to the chain-rule prediction, '
         'then validate composed updates and perform bounded joint optimization. '
         'No path-specific parameters or speed prescriptions. This changes learned '
         'state/history representation, not physical constraints or final acceptance. '
         if args.dynamic_context else '')+
        'The narrow postpeak memory gives a verified but very small improvement only on304. '
        'Saved-state support with the same radial metric and sixfold width includes exactly '
        'four existing155deg training geometries14/104/204/304. Keep the same nine three-action '
        'instant/fast/slow weights and protected ab5 parent. First evaluate direct transfer '
        'of the narrow learned weights in the complete loop, then measure symmetric local '
        'responses about the zero function on all four routes, infer shared constrained '
        'credit and verify up to six proposed joint updates. If measured responses predict '
        'descent, run at most48 scaled COBYLA calls. Every candidate receives all four '
        'complete trajectories; other84 routes have exact zero support and are reused. '
        'No route-specific parameter, timed switch or angle-speed table is introduced. '
        'All original physics and added duration,speed,cruise,multiresolution guards remain. '
        'Protect the narrow304 improvement as well as all reference routes. This is a '
        'shared-training capacity comparison, not independent generalization or PPO.')
    started=time.perf_counter();deadline=started+460;folder=stage/'state_joint';folder.mkdir()
    # Protect the verified narrow improvement on304 without changing the policy parent.
    narrow=read(source/'results.json');narrow=next(c for c in narrow['candidates'] if c['tag']==narrow['best'])
    protected=copy.deepcopy(data)
    protected['baseline']=copy.deepcopy(narrow)
    selected_protected=[r for r in narrow['records'] if r['path'] in paths]
    write_json(stage/'baseline.json',protected)
    write_json(stage/'zero_function_baseline.json',data)
    choice=dict(paths=paths,widths=md['radii'].tolist(),decisions=int(active.sum()))
    write_json(stage/'support_audit.json',dict(selected=choice,
        active_states=[dict(path=int(p),decision_ms=int(t*20)) for p,t in zip(training['path_ids'][active],training['decisions'][active])],
        unaffected_paths=[r['path'] for r in baseline if r['path'] not in paths]))
    np.savez_compressed(stage/'replayed_states.npz',**training)
    old_score,old_guards=metric_functions(protected,selected_protected,stage,md['parent'],deadline)
    def score(records):return old_score(records)+sum(penalty(r) for r in records)
    def constraints(records):return np.concatenate([old_guards(records)]+[guards(r,b) for r,b in zip(records,selected_protected)])
    zero=dict(tag='reused_zero_function',coefficients=[0.]*dimensions,records=selected,score=score(selected),
              measured_inequalities=constraints(selected).tolist())
    history=[zero];cache={tuple(np.zeros(dimensions)):zero};md['reused_candidates']=1
    if args.dynamic_context:
        previous=read(response_source/'state_joint/results.json')['candidates']
        for c in previous[1:]:
            c=copy.deepcopy(c);c['coefficients']=extend(c['coefficients']).tolist();c['tag']='reused_'+c['tag']
            assert abs(c['score']-score(c['records']))<1e-10
            assert np.array_equal(c['measured_inequalities'],constraints(c['records']))
            history.append(c);cache[tuple(np.round(c['coefficients'],12))]=c
        md['reused_candidates']=len(history)
        check=copy.deepcopy(md);check['coefficients']=seed.reshape(-1,3)
        replay=collect_data(check,md['parent'],previous[1]['records'])
        write_json(stage/'context_zero_extension_identity.json',dict(paths=paths,decisions=len(replay['decisions']),
            decoded_actions_bitwise=True,reused_vectors=len(history),new_physical_rollouts=0))
    def feasible(c):return parameters_in_bounds(c['coefficients'],-1.,1.) and min(c['measured_inequalities'])>=-1e-8
    pool=None;blocks=[]
    # Unlike the earlier modules, zero is not necessarily feasible relative to narrow304.
    def best_candidate():
        good=[c for c in history if feasible(c)]
        return min(good,key=lambda c:c['score']) if good else None
    print('SHARED_POSTPEAK_SUPPORT',choice,'zero_guard',min(zero['measured_inequalities']),flush=True)
    def finalize(exhausted=False):
        best=best_candidate()
        if best is None:
            torch.save(md,folder/'candidate.pt')
            write_json(folder/'results.json',dict(candidates=history,best=None))
            write_json(stage/'optimization_blocks.json',blocks)
            write_json(stage/'results.json',dict(candidates=[],research_acceptable=False,adopted=False,goal_complete=False))
            receipt(stage,dict(status='completed_shared_postpeak_no_feasible_update',research_acceptable=False,
                new_candidates=len(history)-md['reused_candidates'],source_model_sha256=sha(parent_file),wall_s=time.perf_counter()-started,
                budget_exhausted=exhausted,adopted=False,goal_complete=False))
            print('SHARED_POSTPEAK_COMPLETE no feasible update',len(history)-1,flush=True)
        else:
            reference=dict(zero,score=score(selected_protected))
            finish(stage,md,narrow['records'],protected,best,reference,history,blocks,choice,parent_file,started,budget_exhausted=exhausted)
    try:
        pool=mp.get_context('spawn').Pool(4,worker_initialize,(md,selected,folder,deadline))
        def evaluate(u):
            key=tuple(np.round(u,12))
            if key not in cache:
                tag=f'shared_postpeak_{len(history)}'
                records=pool.map(worker_evaluate,[(p,u,tag) for p in paths])
                c=dict(tag=tag,coefficients=np.asarray(u).tolist(),records=records,score=score(records),
                    measured_inequalities=constraints(records).tolist(),actual_rerun_paths=paths,exact_zero_support_paths=[])
                history.append(c);cache[key]=c;best=best_candidate()
                write_json(folder/'results.json',dict(candidates=history,best=best['tag'] if best else None))
                if len(history)%5==0:print('SHARED_POSTPEAK',len(history)-1,'best',best['score'] if best else None,
                    'latest',c['score'],min(c['measured_inequalities']),flush=True)
            return cache[key]
        transfer=evaluate(seed)
        blocks.append(dict(method='direct_weight_transfer',tag=transfer['tag'],guard_minimum=min(transfer['measured_inequalities'])))
        base_steps=(np.array(read(response_source/'local_credit_model.json')['steps']) if args.dynamic_context else
                    np.tile([.003,.01,.01],3))
        steps=np.repeat(base_steps.reshape(3,1,3),context_count,axis=1).reshape(-1);pairs=[]
        for i,h in enumerate(steps):
            u=np.zeros(dimensions);u[i]=h;plus=evaluate(u);minus=evaluate(-u)
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
        linear=linprog(g,A_ub=-jac,b_ub=c0,bounds=[(-1.,1.)]*dimensions,method='highs');proposals=[]
        if linear.success and linear.fun<-.001:proposals.append(('linear',linear.x))
        predict=lambda z:c0+jac@z-even@(z*z)
        objective=lambda z:float(g@z+q@(z*z))
        fit=minimize(objective,linear.x if proposals else seed/steps,method='SLSQP',bounds=[(-1.,1.)]*dimensions,
            constraints=[dict(type='ineq',fun=predict)],options=dict(maxiter=400,ftol=1e-10))
        if objective(fit.x)<-.001 and predict(fit.x).min()>=-1e-7:proposals.insert(0,('diagonal_conservative',fit.x))
        write_json(stage/'local_credit_model.json',dict(pairs=pairs,steps=steps.tolist(),central_constraints=jac.tolist(),
            absolute_even_constraints=even.tolist(),central_loss=g.tolist(),nonnegative_even_loss=q.tolist(),
            linear_predicted_change=float(linear.fun) if linear.success else None,
            linear_status=str(linear.message),conservative_predicted_change=objective(fit.x),
            conservative_guard_minimum=float(predict(fit.x).min())))
        if args.dynamic_context:
            predicted=next(r for r in context_audit['models'] if r['context_count']==context_count)
            old_steps=np.repeat(base_steps.reshape(3,1,3),context_count,axis=1).reshape(-1)
            ratio=steps/old_steps
            expected_jac=np.asarray(predicted['central_constraints'])*ratio
            expected_g=np.asarray(predicted['central_loss'])*ratio
            write_json(stage/'context_credit_validation.json',dict(
                constraint_derivative_max_error=float(abs(jac-expected_jac).max()),
                loss_derivative_max_error=float(abs(g-expected_g).max()),
                predicted_model_feasible=predicted['feasible'],measured_model_feasible=bool(linear.success),
                predicted_model_loss_change=predicted['predicted_loss_change'],
                measured_model_loss_change=float(linear.fun) if linear.success else None,
                original_constraints_and_reference_unchanged=True))
        for method,z in proposals:
            for scale in (1.,.5,.25):
                c=evaluate(steps*z*scale);blocks.append(dict(method=method,scale=scale,tag=c['tag'],
                    predicted_change=objective(scale*z),actual_change=c['score']-zero['score'],
                    measured_guard_minimum=min(c['measured_inequalities'])))
        if proposals:
            best=best_candidate();initial=np.asarray((best or transfer)['coefficients'])/steps
            solved=minimize(lambda z:evaluate(steps*z)['score'],initial,method='COBYLA',
                bounds=[(-1./h,1./h) for h in steps],constraints=[dict(type='ineq',fun=lambda z:np.asarray(evaluate(steps*z)['measured_inequalities']))],
                options=dict(maxiter=48,rhobeg=1.,tol=1e-6,catol=1e-8))
            blocks.append(dict(method='shared_joint_learning',solver=str(solved.message)))
        pool.close();pool.join();pool=None;finalize()
    except TimeoutError:
        if pool is not None:pool.terminate();pool.join();pool=None
        finalize(True)
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_shared_postpeak_memory_learning',error=str(exc),wall_s=time.perf_counter()-started,
            adopted=False,goal_complete=False));raise


if __name__=='__main__':mp.freeze_support();main()
