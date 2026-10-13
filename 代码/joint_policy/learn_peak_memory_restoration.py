"""Restore actual shape guards with a learned peak-region three-action memory."""
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
from joint_teacher_state_policy import features,state_pair
from geometric_turn_metrics import spatial_turn_profile
from learn_early_memory_joint_policy import worker_initialize,worker_evaluate,metric_functions
from learn_direct_observable_joint_policy import finish
from learn_complete_joint_teachers import parameters_in_bounds
from study_io import BASE,register,receipt,write_json,sha
from search_complete_return import read


def expand_coefficients(value):
    return np.concatenate([np.asarray(value).reshape(3,2,3),np.zeros((3,1,3))],axis=1).reshape(-1)


def main():
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_main_turn_memory_joint_learning_20260920'
    parent_file=source/'research_candidate.pt';audit=read(source/'raw_result_audit.json')
    assert audit['model_sha256']==sha(parent_file) and audit['changed_paths']==[304]
    outer=read(source/'results.json');selected_full=next(c for c in outer['candidates'] if c['tag']==outer['best'])
    data=read(source/'baseline.json');data['baseline']=copy.deepcopy(selected_full);baseline=data['baseline']['records']
    selected=[next(r for r in baseline if r['path']==304)]
    old=read(source/'state_joint/results.json');reference=next(c for c in old['candidates'] if c['tag']==old['best'])
    md=torch.load(parent_file,map_location='cpu',weights_only=False)
    stage=register('canonical_gate_band_peak_memory_shape_restoration_20260920',
        [__file__,BASE/'observable_memory_joint_policy.py',BASE/'joint_teacher_state_policy.py',
         BASE/'consolidate_joint_teacher_policy.py',BASE/'learn_early_memory_joint_policy.py',
         BASE/'learn_shared_temporal_joint_policy.py',BASE/'learn_direct_observable_joint_policy.py',
         BASE/'learn_complete_joint_teachers.py',BASE/'geometric_turn_metrics.py',parent_file,
         source/'receipt.json',source/'raw_result_audit.json',source/'saved_lateral_motion_audit.json',
         source/'energy_resolution_audit.json',source/'state_joint/results.json',source/'replayed_states.npz'],
        dict(wall_s=480,workers=3,new_peak_weights=9,total_weights=27,maximum_axis_vectors=36,
             maximum_joint_validations=6,maximum_restoration_optimizer_calls=80,
             nominal_steps=[.003,.02,.02],original_guard_reference_sha256=sha(parent_file)),
        'Main-turn learning has a protected candidate, but the largest measured reversal '
        'improvement fails a small energy guard. Four sampling phases at0.1/0.2mm do not '
        'justify dismissing that energy increase as numerical error. Keep the STRICT '
        'latest accepted research-reference guards, including its improved peak curvature '
        'and apex speed. Choose the lowest-loss saved physical point satisfying all other '
        'guards; retain it explicitly as unaccepted exploration. Add a third continuous '
        'observable-state kernel one decision before its actual spatial-curvature peak. '
        'The peak index selects a training state only, never a runtime time/path rule. '
        'Keep the two learned old kernels and their memory traces exactly; append nine '
        'zero coefficients for instantaneous/fast/slow peak residuals. Radius is half '
        'the nearest other-route saved-state distance; audit all88 routes. Verify zero '
        'extension by decoded-action replay for both accepted and exploratory304 traces, '
        'and reuse all prior full physical vectors. Freeze the old18 exploratory weights '
        'while learning nine new THREE-ACTION peak weights from symmetric complete-route '
        'responses, local constrained credit, independent joint validation, and up to80 '
        'scaled restoration calls in480s. Only original-guard-passing full outcomes may '
        'form a research checkpoint. No metric/physics limit is relaxed, no teacher '
        'schedule is deployed, and no generalization or PPO claim follows.')
    started=time.perf_counter();deadline=started+460;folder=stage/'state_joint';folder.mkdir()
    score,constraints=metric_functions(data,selected,stage,md['parent'],deadline)
    history=[]
    for c in old['candidates']:
        c=copy.deepcopy(c);c['tag']='reused_'+c['tag'];c['coefficients']=expand_coefficients(c['coefficients']).tolist()
        c['score']=score(c['records']);c['measured_inequalities']=constraints(c['records']).tolist();history.append(c)
    zero=next(c for c in history if c['tag']=='reused_'+reference['tag'])
    history.remove(zero);history.insert(0,zero)
    def feasible(c):return parameters_in_bounds(c['coefficients'],-1.,1.) and min(c['measured_inequalities'])>=-1e-8
    assert feasible(zero)
    safe_indices=[i for i in range(len(zero['measured_inequalities'])) if i not in (3,4,5)]
    anchor=min((c for c in history if np.min(np.asarray(c['measured_inequalities'])[safe_indices])>=-1e-8
        and parameters_in_bounds(c['coefficients'],-1.,1.)),key=lambda c:c['score'])
    anchor_old=next(c for c in old['candidates'] if 'reused_'+c['tag']==anchor['tag'])
    anchor_md=copy.deepcopy(md);anchor_md['coefficients']=np.asarray(anchor_old['coefficients']).reshape(-1,3)
    anchor_md.update(accepted=False,training_exploration_only=True,final_shape_guards_pass=False)
    torch.save(anchor_md,stage/'unaccepted_exploration_anchor.pt')
    write_json(stage/'exploration_anchor.json',dict(tag=anchor['tag'],score=anchor['score'],
        original_guard_minimum=min(anchor['measured_inequalities']),adopted=False))
    with np.load(source/'replayed_states.npz') as z:training={k:z[k].copy() for k in z.files}
    replay=collect_data(anchor_md,md['parent'],anchor['records']);keep=training['path_ids']!=304
    training={k:np.concatenate([v[keep],replay[k]]) for k,v in training.items()}
    np.savez_compressed(stage/'replayed_states.npz',**training)
    r=anchor['records'][0]
    with np.load(r['trace']) as z:ref,rows=z['reference'].copy(),z['rows'].copy()
    xy=np.vstack([ref[0],rows[:,:2]]);spatial=spatial_turn_profile(xy)
    arc_at_peak=spatial['grid'][1:-1][abs(spatial['turn']).argmax()]
    raw_arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
    peak_ms=int(np.searchsorted(raw_arc,arc_at_peak))-1;decision=max(0,peak_ms//20-1)
    a,b=state_pair(training['observations'],training['innovations']);a,b=(a-md['mean'])/md['std'],(b-md['mean'])/md['std']
    index=np.flatnonzero((training['path_ids']==304)&(training['decisions']==decision)).item();center=a[index]
    distances=np.minimum(np.sqrt(np.mean((a-center)**2,axis=1)),np.sqrt(np.mean((b-center)**2,axis=1)))
    radius=.5*float(distances[training['path_ids']!=304].min());assert radius>1e-10
    previous_f=features(md,training['observations'],training['innovations'])
    md.update(centers=np.vstack([md['centers'],center]),radii=np.r_[md['radii'],radius],
        coefficients=np.asarray(zero['coefficients']).reshape(-1,3),reused_candidates=len(history),
        learning_method='peak_region_causal_memory_shape_guard_restoration')
    f=features(md,training['observations'],training['innovations']);assert np.array_equal(f[:,:2,:],previous_f)
    active=np.any(f!=0.,axis=(1,2));paths=np.unique(training['path_ids'][active]).tolist();assert paths==[304]
    identity=[]
    for c in (zero,anchor):
        test_md=copy.deepcopy(md);test_md['coefficients']=np.asarray(c['coefficients']).reshape(-1,3)
        checked=collect_data(test_md,md['parent'],c['records'])
        identity.append(dict(tag=c['tag'],decisions=len(checked['path_ids']),decoded_actions_bitwise=True,
                             recorded_target_velocities_bitwise=True))
    write_json(stage/'zero_extension_identity.json',dict(original_features_bitwise=True,physical_reruns=0,checks=identity))
    choice=dict(paths=paths,widths=md['radii'].tolist(),decisions=int(active.sum()))
    write_json(stage/'support_audit.json',dict(selected=choice,peak_ms=peak_ms,center_training_decision_ms=20*decision,
        unaffected_paths=[r['path'] for r in baseline if r['path']!=304],scope='Localized training capacity; not generalization.'))
    write_json(stage/'baseline.json',data)
    print('PEAK_MEMORY_SUPPORT',choice,'peak_ms',peak_ms,'center_ms',20*decision,'anchor',anchor['tag'],anchor['score'],min(anchor['measured_inequalities']),flush=True)
    indices=np.array([6,7,8,15,16,17,24,25,26]);seed=np.array(anchor['coefficients'])
    cache={tuple(np.round(c['coefficients'],12)):c for c in history};pool=None;blocks=[]
    try:
        pool=mp.get_context('spawn').Pool(3,worker_initialize,(md,selected,folder,deadline))
        def evaluate(u):
            key=tuple(np.round(u,12))
            if key not in cache:
                tag=f'peak_{len(history)}';records=pool.map(worker_evaluate,[(304,u,tag)])
                c=dict(tag=tag,coefficients=np.asarray(u).tolist(),records=records,score=score(records),
                    measured_inequalities=constraints(records).tolist(),actual_rerun_paths=[304],exact_zero_support_paths=[])
                history.append(c);cache[key]=c
                best=min(filter(feasible,history),key=lambda c:c['score'])
                write_json(folder/'results.json',dict(candidates=history,best=best['tag']))
                if len(history)%5==0:print('PEAK_MEMORY',len(history),'best',best['score'],'latest',c['score'],min(c['measured_inequalities']),flush=True)
            return cache[key]
        steps=np.tile([.003,.02,.02],3);pairs=[]
        for j,(i,h) in enumerate(zip(indices,steps)):
            u=seed.copy();u[i]+=h;plus=evaluate(u);u[i]-=2*h;minus=evaluate(u)
            size=max(np.max(abs(np.asarray(c['measured_inequalities'])-anchor['measured_inequalities'])) for c in (plus,minus))
            if size>.1:
                h*=max(.001,.1/size);steps[j]=h;u=seed.copy();u[i]+=h;plus=evaluate(u);u[i]-=2*h;minus=evaluate(u)
            pairs.append(dict(index=int(i),step=h,plus=plus['tag'],minus=minus['tag']))
            write_json(stage/'response_progress.json',dict(pairs=pairs,axes=len(pairs)))
        lookup={c['tag']:c for c in history};central=[];even=[];g=[];q=[]
        for p in pairs:
            plus,minus=lookup[p['plus']],lookup[p['minus']];a,b=np.array(plus['measured_inequalities']),np.array(minus['measured_inequalities'])
            central.append(.5*(a-b));even.append(abs(.5*(a+b)-anchor['measured_inequalities']))
            g.append(.5*(plus['score']-minus['score']));q.append(max(0.,.5*(plus['score']+minus['score'])-anchor['score']))
        jac=np.array(central).T;curvature=np.array(even).T;g,q=np.array(g),np.array(q);c0=np.array(anchor['measured_inequalities'])
        predicted=lambda z:c0+jac@z-curvature@(z*z)
        objective=lambda z:float(g@z+q@(z*z))
        linear=linprog(g,A_ub=-jac,b_ub=c0,bounds=[(-1.,1.)]*9,method='highs');proposals=[]
        if linear.success:proposals.append(('linear',linear.x))
        fitted=minimize(objective,.25*linear.x if proposals else np.zeros(9),method='SLSQP',bounds=[(-1.,1.)]*9,
            constraints=[dict(type='ineq',fun=predicted)],options=dict(maxiter=500,ftol=1e-10))
        if predicted(fitted.x).min()>=-1e-7:proposals.insert(0,('diagonal_conservative',fitted.x))
        write_json(stage/'local_credit_model.json',dict(pairs=pairs,steps=steps.tolist(),central_constraints=jac.tolist(),
            absolute_even_constraints=curvature.tolist(),anchor_constraints=c0.tolist(),central_loss=g.tolist(),nonnegative_even_loss=q.tolist(),
            linear_message=str(linear.message),conservative_message=str(fitted.message),conservative_minimum=float(predicted(fitted.x).min()),
            proposals=[dict(method=m,scaled_parameters=z.tolist()) for m,z in proposals]))
        def expand(z):
            u=seed.copy();u[indices]+=steps*z;return u
        for method,z in proposals:
            for scale in (1.,.5,.25):
                c=evaluate(expand(scale*z));blocks.append(dict(method=method,scale=scale,tag=c['tag'],score=c['score'],
                    actual_minimum=min(c['measured_inequalities']),passes_original_guards=feasible(c)))
                write_json(stage/'composed_credit_validation.json',blocks)
        compatible=[c for c in history if np.array_equal(np.delete(c['coefficients'],indices),np.delete(seed,indices))]
        initial=min(compatible,key=lambda c:np.square(np.minimum(c['measured_inequalities'],0.)).sum())
        learned=minimize(lambda z:evaluate(expand(z))['score'],(np.array(initial['coefficients'])[indices]-seed[indices])/steps,
            method='COBYLA',bounds=[((-1.-seed[i])/h,(1.-seed[i])/h) for i,h in zip(indices,steps)],
            constraints=[dict(type='ineq',fun=lambda z:np.array(evaluate(expand(z))['measured_inequalities']))],
            options=dict(maxiter=80,rhobeg=2.,tol=1e-6,catol=1e-8))
        blocks.append(dict(method='peak_weight_shape_restoration',solver=str(learned.message)))
        pool.close();pool.join();pool=None
        best=min(filter(feasible,history),key=lambda c:c['score'])
        finish(stage,md,baseline,data,best,zero,history,blocks,choice,parent_file,started)
    except TimeoutError:
        if pool is not None:pool.terminate();pool.join();pool=None
        best=min(filter(feasible,history),key=lambda c:c['score'])
        finish(stage,md,baseline,data,best,zero,history,blocks,choice,parent_file,started,budget_exhausted=True)
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_peak_memory_restoration',error=str(exc),wall_s=time.perf_counter()-started,
            adopted=False,goal_complete=False));raise


if __name__=='__main__':mp.freeze_support();main()
