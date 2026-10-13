"""Calibrate action sensitivities, then validate complete-return joint updates."""
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
from learn_early_memory_joint_policy import worker_initialize,worker_evaluate,metric_functions
from learn_direct_observable_joint_policy import finish
from learn_complete_joint_teachers import parameters_in_bounds
from joint_teacher_state_policy import features,motion_geometry_context,state_pair
from multi_corner_raw_metrics import augment
from study_io import BASE,register,receipt,write_json,sha
from search_complete_return import read


def main():
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group()
    group.add_argument('--context-memory',action='store_true');group.add_argument('--compact-memory',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1)
    response_source=BASE/'canonical_gate_band_early_observable_memory_joint_learning_20260920'
    source=(BASE/'canonical_gate_band_memory_context_joint_credit_learning_20260920' if args.compact_memory else
            BASE/'canonical_gate_band_memory_calibrated_joint_credit_learning_20260920'
            if args.context_memory else response_source)
    previous=read(source/'receipt.json');assert previous['status'].startswith('completed')
    assert not previous['research_acceptable'],'Audit a useful previous model first'
    parent_file=BASE/'canonical_gate_band_direct_observable_joint_learning_20260920/research_candidate.pt'
    metadata_source=(response_source if args.compact_memory else source)/'state_joint/candidate.pt'
    md=torch.load(metadata_source,map_location='cpu',weights_only=False)
    assert not np.any(md['coefficients'])
    indices=np.array([3,4,5,9,10,11,15,16,17])
    responses=read(response_source/'matched_memory_head_response_audit.json')['single_coefficient_responses']
    by_index={r['index']:r for r in responses}
    steps=np.array([np.clip(abs(by_index[i]['coefficient_change'])*.025/
        max(by_index[i]['maximum_normalized_constraint_change'],1e-8),1e-4,.025) for i in indices])
    with np.load(source/'replayed_states.npz') as z:training={k:z[k].copy() for k in z.files}
    affine=BASE/'canonical_gate_band_early_local_affine_joint_learning_20260920'
    context_ratios=[]
    compact_audit=None
    if args.compact_memory:
        pair=state_pair(training['observations'],training['innovations'])
        distances=np.minimum(*[np.sqrt(np.mean(((x-md['mean'])/md['std']-md['centers'][1])**2,axis=1)) for x in pair])
        protected=training['path_ids']<64
        radius=.5*float(distances[protected].min());assert 0.<radius<md['radii'][1]
        md['radii']=md['radii'].copy();md['radii'][1]=radius
        active=np.any(features(md,training['observations'],training['innovations'])[:,1,:]!=0.,axis=1)
        assert not np.any(active&protected)
        compact_audit=dict(radius=radius,selection='Half the nearest original64 saved-state distance; never a runtime path test.',
            affected_paths=np.unique(training['path_ids'][active]).tolist(),decisions=int(active.sum()),
            original64_exact_zero_support=True,remaining_center_frozen_zero=True,
            final_generalization_claim=False)
    if args.context_memory:
        old_md=torch.load(affine/'state_joint/candidate.pt',map_location='cpu',weights_only=False)
        for key in ('mean','std','centers','radii'):assert np.array_equal(md[key],old_md[key])
        md.update(context_memory=True,memory_decays=[.5,.8,.8,.8],center_context=old_md['center_context'],
                  context_scale=old_md['context_scale'],coefficients=np.zeros((10,3)),
                  learning_method='contextual_causal_memory_complete_return_joint_credit')
        ff=features(md,training['observations'],training['innovations'])[:,1,:]
        context=np.tanh((motion_geometry_context(training['observations'])[:,None,:]-md['center_context'])/md['context_scale'])
        context_ratios=[float(np.clip(np.linalg.norm(ff)/max(np.linalg.norm(ff*context[:,1,k,None]),1e-12),1.,10.)) for k in range(2)]
        indices=np.r_[indices,[21,22,23,27,28,29]]
        steps=np.concatenate([steps]+[np.minimum(.025,steps[6:9]*ratio) for ratio in context_ratios])
    dimension=md['coefficients'].size
    stage=register('canonical_gate_band_compact_memory_joint_credit_learning_20260920' if args.compact_memory else
                   'canonical_gate_band_memory_context_joint_credit_learning_20260920' if args.context_memory
                   else 'canonical_gate_band_memory_calibrated_joint_credit_learning_20260920',
        [__file__,BASE/'learn_early_memory_joint_policy.py',BASE/'observable_memory_joint_policy.py',
         BASE/'joint_teacher_state_policy.py',BASE/'learn_shared_temporal_joint_policy.py',
         BASE/'learn_complete_joint_teachers.py',BASE/'multi_corner_raw_metrics.py',parent_file,
         source/'receipt.json',source/'state_joint/results.json',metadata_source,
         response_source/'memory_zero_extension_identity.json',response_source/'memory_execution_audit.json',
         response_source/'matched_memory_head_response_audit.json',source/'support_audit.json',source/'replayed_states.npz']+
        ([source/'local_credit_conflict_audit.json',source/'local_credit_model.json',affine/'state_joint/candidate.pt',
          BASE/'consolidate_joint_teacher_policy.py'] if args.context_memory else []),
        dict(wall_s=360,workers=3,center_path=304,coefficient_indices=indices.tolist(),
             calibrated_steps=steps.tolist(),target_normalized_response=.025,
             matched_axis_vectors=2*len(indices),maximum_composed_validations=6,coefficient_bounds=[-1.,1.],
             context_memory=args.context_memory,parameters=dimension,context_feature_scale_ratios=context_ratios,
             compact_support=compact_audit,maximum_postvalidation_optimizer_calls=60 if args.compact_memory else 0),
        ('Both shared memory credit models have no protected joint descent despite individual '
         'route descent. Test local functional capacity by narrowing ONLY the304 training-state '
         'kernel to half the nearest original64 state distance, retaining the same causal '
         'memory form, parent and all physical/final guards. The runtime sees continuous '
         'observations only; path IDs select training samples/support audits and never actions. '
         'The other center stays frozen zero. Audit support over ALL88 saved routes and '
         'actually execute every affected full route. Reuse only the zero residual because '
         'nonzero old vectors define a different function. This is a localized training '
         'capacity control, not a generalization solution or a trajectory template. '
         'Collect18 matched axis responses, validate up to6 joint proposals, and only after '
         'an actual qualifying gain optimize that learned9-parameter function for at most60 '
         'additional calls in the same360s budget using the measured parameter scales. '
         'Keep the original retained model and all87 zero-support trajectories. '
         if args.compact_memory else '')+
        ('The measured nine-parameter local model permits isolated-route descent on14,104,204,304 '
         'but no descent when every original route guard is combined. Test a richer shared '
         'OBSERVATION-HISTORY function while retaining every old coefficient and causal trace. '
         'Append two0.8-decay traces of past kernel activation multiplied by tanh-normalized '
         'observed speed and forward gate range, using the verified local-affine feature '
         'centers/scales. Thus history retains motion/geometry context after instantaneous '
         'support vanishes. Learn all three heads; no path/time switch or angle-speed table. '
         'Zero-extend all113 previous vectors exactly, including all18 matched responses. '
         'Check actual decoded-action identity on saved nonzero-memory304 and multi-corner18 '
         'routes. Only the six new304 context weights require12 new +/- responses at the '
         'same zero parent. Feature-energy ratios calibrate their initial perturbations, '
         'not actuator commands. Rebuild the15-dimensional joint credit models and validate '
         'composed updates through unchanged full physical routes and original guards. '
         'Local-model failure is not global impossibility or evidence of PPO bias. '
         if args.context_memory else '')+
        'Same0.05 simplex perturbations produced radically unequal complete physical responses '
        'across steering/feed/preview and memory traces. Keep the declared memory function, '
        'parent,all original final guards and complete-route evaluator. Freeze the206 center '
        'and calibrate the declared instantaneous/memory304 coefficients from measured '
        'response magnitudes; this changes parameter coordinates, not action or speed rules. '
        'At the SAME accepted zero-residual parent, collect +/- each small calibrated axis '
        'through full closed-loop coverage/braking/rest. Reuse every exact prior vector. '
        'Estimate central constraint/loss differences and even response terms. Solve both '
        'linear and diagonal-conservative local credit models inside the measured box. '
        'Independently validate useful composed joint updates at1,1/2,1/4 scale, at most6 '
        'new vectors. Neither a predicted feasible point nor a local-cone failure establishes '
        'global feasibility or PPO bias. If the actual original guards all pass and loss '
        'improves, audit the full training pool; otherwise change the mechanism rather than '
        'extend the same solver run. Physical constraints and deployment chain remain intact.')
    started=time.perf_counter();deadline=started+340;folder=stage/'state_joint';folder.mkdir()
    data=read(source/'baseline.json');baseline=data['baseline']['records']
    support=read(source/'support_audit.json');choice=support['selected']
    if args.compact_memory:
        choice=dict(paths=compact_audit['affected_paths'],width=compact_audit['radius'],decisions=compact_audit['decisions'])
        support=dict(selected=choice,compact_support=compact_audit,
            unaffected_paths=[r['path'] for r in baseline if r['path'] not in choice['paths']])
    selected=[r for r in baseline if r['path'] in choice['paths']]
    for r in selected:augment(r)
    write_json(stage/'baseline.json',data);write_json(stage/'support_audit.json',support)
    np.savez_compressed(stage/'replayed_states.npz',**training)
    f=features(md,training['observations'],training['innovations'])
    score,constraints=metric_functions(data,selected,stage,md['parent'],deadline)
    history=([dict(tag='identity',coefficients=[0.]*dimension,records=selected,score=score(selected),
                   measured_inequalities=constraints(selected).tolist())] if args.compact_memory else
             copy.deepcopy(read(source/'state_joint/results.json')['candidates']))
    pool=None;blocks=[]
    for c in history:
        if args.context_memory:c['coefficients']+= [0.]*12
        c['tag']='reused_'+c['tag'];assert abs(c['score']-score(c['records']))<1e-10
        assert np.array_equal(c['measured_inequalities'],constraints(c['records']))
    zero=history[0];cache={tuple(np.round(c['coefficients'],12)):c for c in history}
    md['reused_candidates']=len(history)
    def feasible(c):return parameters_in_bounds(c['coefficients'],-1.,1.) and min(c['measured_inequalities'])>=-1e-8
    try:
        if args.context_memory:
            from consolidate_joint_teacher_policy import collect_data
            check=next(c for c in history if c['tag'].endswith('calibrated_108'))
            check_md=copy.deepcopy(md);check_md['coefficients']=np.asarray(check['coefficients']).reshape(-1,3)
            check_records=[r for r in check['records'] if r['path'] in (18,304)]
            replay=collect_data(check_md,md['parent'],check_records)
            write_json(stage/'context_zero_extension_identity.json',dict(paths=[18,304],decisions=len(replay['path_ids']),
                actual_decoded_actions_bitwise=True,recorded_target_velocities_bitwise=True,
                previous_nonzero_memory_trace=check['tag'],new_physical_rollouts=0))
        pool=mp.get_context('spawn').Pool(3,worker_initialize,(md,selected,folder,deadline))
        def evaluate(u):
            key=tuple(np.round(u,12))
            if key not in cache:
                tag=f'calibrated_{len(history)}';active=np.any(np.asarray(u).reshape(-1,2,3)!=0.,axis=(0,2))
                zero_paths=[r['path'] for r in selected if not np.any(f[training['path_ids']==r['path']][:,active,:])]
                measured=pool.map(worker_evaluate,[(r['path'],u,tag) for r in selected if r['path'] not in zero_paths])
                by_path={r['path']:r for r in measured}
                records=[by_path.get(r['path'],dict(copy.deepcopy(r),reused_exact_zero_support=True)) for r in selected]
                for r in records:augment(r)
                c=dict(tag=tag,coefficients=np.asarray(u).tolist(),records=records,score=score(records),
                    measured_inequalities=constraints(records).tolist(),exact_zero_support_paths=zero_paths,
                    actual_rerun_paths=[r['path'] for r in selected if r['path'] not in zero_paths])
                history.append(c);cache[key]=c
                best=min(filter(feasible,history),key=lambda c:c['score'])
                write_json(folder/'results.json',dict(candidates=history,best=best['tag']))
                print('CALIBRATED_RESPONSE',tag,c['score'],min(c['measured_inequalities']),flush=True)
            return cache[key]
        central=[];even=[];gradient=[];loss_even=[];pairs=[]
        for i,h in zip(indices,steps):
            u=np.zeros(dimension);u[i]=h;plus=evaluate(u);minus=evaluate(-u)
            a,b=np.asarray(plus['measured_inequalities']),np.asarray(minus['measured_inequalities'])
            central.append(.5*(a-b));even.append(abs(.5*(a+b)-zero['measured_inequalities']))
            gradient.append(.5*(plus['score']-minus['score']))
            loss_even.append(max(0.,.5*(plus['score']+minus['score'])-zero['score']))
            pairs.append(dict(index=int(i),step=float(h),plus=plus['tag'],minus=minus['tag']))
            write_json(stage/'matched_response_progress.json',dict(pairs=pairs,completed_axes=len(pairs),
                common_parent_coefficients=[0.]*dimension,new_physical_vectors=len(history)-md['reused_candidates']))
        jac=np.asarray(central).T;curvature=np.asarray(even).T;g=np.asarray(gradient);q=np.asarray(loss_even)
        c0=np.asarray(zero['measured_inequalities'])
        linear=linprog(g,A_ub=-jac,b_ub=c0,bounds=[(-1.,1.)]*len(indices),method='highs')
        proposals=[]
        if linear.success and float(g@linear.x)<-.001:proposals.append(('linear',linear.x))
        def predicted(z):return c0+jac@z-curvature@(z*z)
        model=lambda z:float(g@z+q@(z*z))
        initial=np.zeros(len(indices)) if not proposals else .25*linear.x
        quadratic=minimize(model,initial,method='SLSQP',bounds=[(-1.,1.)]*len(indices),
            constraints=[dict(type='ineq',fun=predicted)],options=dict(maxiter=500,ftol=1e-10))
        if model(quadratic.x)<-.001 and predicted(quadratic.x).min()>=-1e-7:
            proposals.insert(0,('diagonal_conservative',quadratic.x))
        write_json(stage/'local_credit_model.json',dict(indices=indices.tolist(),steps=steps.tolist(),pairs=pairs,
            central_constraints=jac.tolist(),absolute_even_constraints=curvature.tolist(),
            central_loss=g.tolist(),nonnegative_even_loss=q.tolist(),
            linear_message=str(linear.message),linear_predicted_loss_change=float(linear.fun) if linear.success else None,
            conservative_message=str(quadratic.message),conservative_predicted_loss_change=model(quadratic.x),
            conservative_minimum=float(predicted(quadratic.x).min()),
            proposals=[dict(method=m,scaled_parameters=z.tolist()) for m,z in proposals],
            local_prediction_only=True))
        for method,z in proposals:
            for scale in (1.,.5,.25):
                u=np.zeros(dimension);u[indices]=steps*z*scale;c=evaluate(u)
                blocks.append(dict(method=method,scale=scale,tag=c['tag'],actual_loss_change=c['score']-zero['score'],
                    predicted_loss_change=model(scale*z),predicted_minimum=float(predicted(scale*z).min()),
                    actual_minimum=min(c['measured_inequalities']),passes_original_guards=feasible(c)))
                write_json(stage/'composed_credit_validation.json',blocks)
        best=min(filter(feasible,history),key=lambda c:c['score'])
        if args.compact_memory and best['score']<zero['score']-.001:
            seed=np.asarray(best['coefficients'])[indices]/steps
            def expand(z):
                u=np.zeros(dimension);u[indices]=steps*z;return u
            learned=minimize(lambda z:evaluate(expand(z))['score'],seed,method='COBYLA',
                bounds=[(-1./h,1./h) for h in steps],
                constraints=[dict(type='ineq',fun=lambda z:np.asarray(evaluate(expand(z))['measured_inequalities']))],
                options=dict(maxiter=60,rhobeg=2.,tol=1e-6,catol=1e-8))
            blocks.append(dict(method='scaled_complete_joint_learning_after_actual_feasible_gain',solver=str(learned.message)))
        pool.close();pool.join();pool=None
        best=min(filter(feasible,history),key=lambda c:c['score'])
        finish(stage,md,baseline,data,best,zero,history,blocks,choice,parent_file,started)
    except TimeoutError:
        if pool is not None:pool.terminate();pool.join();pool=None
        best=min(filter(feasible,history),key=lambda c:c['score'])
        finish(stage,md,baseline,data,best,zero,history,blocks,choice,parent_file,started,budget_exhausted=True)
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_memory_calibrated_credit',error=str(exc),wall_s=time.perf_counter()-started,
            adopted=False,goal_complete=False));raise


if __name__=='__main__':mp.freeze_support();main()
