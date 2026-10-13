"""Learn a shared observable-reference residual from full closed-loop responses."""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import copy
import time
import multiprocessing as mp
import numpy as np
import torch
from scipy.optimize import linprog
from audit_saved_lateral_motion import lateral_motion
from intrinsic_curve_metrics import intrinsic_metrics
from joint_teacher_state_policy import features, state_pair, reference_oriented_observation, INDICES
from consolidate_joint_teacher_policy import collect_data
from saved_memory_action_identity import SavedMemoryActionIdentity
from gate_band_state import GateBandRuntime
from duration_joint_metrics import guards, penalty
from shape_event_credit import shape_event_learning_guards
from learn_intrinsic_joint_recovery import intrinsic_guards
from learn_complete_joint_teachers import teacher_score, parameters_in_bounds
from learn_early_memory_joint_policy import worker_initialize, worker_evaluate, metric_functions
from learn_direct_observable_joint_policy import finish
from audit_context_restoration_scaling import solve_correction
from study_io import BASE, register, write_json, receipt, sha
from search_complete_return import read


def descent(jac, gradient, values, u, steps, indices, radius):
    """Constrained response-model descent; the proposal still needs real execution."""
    a = -jac[:, indices]
    scale = np.maximum(np.maximum(np.max(abs(a), axis=1), abs(values)), 1e-12)
    bounds = [(max(-radius, (-1-u[i])/steps[i]), min(radius, (1-u[i])/steps[i])) for i in indices]
    solved = linprog(gradient[indices], A_ub=a/scale[:, None], b_ub=values/scale,
                     bounds=bounds, method='highs', options=dict(
                         primal_feasibility_tolerance=1e-9, dual_feasibility_tolerance=1e-9))
    delta = np.zeros(len(u))
    if solved.success:
        delta[indices] = solved.x
    return dict(success=bool(solved.success), message=str(solved.message),
                scaled_parameter_step=delta.tolist(), predicted_loss_change=float(gradient@delta),
                predicted_guard_minimum=float(np.min(values+jac@delta)))


def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    source = BASE/'canonical_gate_band_reference_supported_joint_learning_20260920'
    parent_file = source/'research_candidate.pt'
    assert sha(parent_file) == 'ded2a93f85f3503a8c09bf59fd79e48c8f2cd187874f07789c53b8863c936229'
    assert read(source/'selection_review.json')['protected_research_candidate']
    parent = torch.load(parent_file, map_location='cpu', weights_only=False)
    result = read(source/'results.json')
    current = next(c for c in result['candidates'] if c['tag'] == result['best'])
    data = copy.deepcopy(read(source/'baseline.json'))
    data['baseline'] = copy.deepcopy(current)
    baseline = data['baseline']['records']
    corners = [r for r in baseline if 'shape' in r]
    targets = sorted({max(corners, key=lambda r:r['shape'][key])['path'] for key in
                      ('out_edge_outward_mm', 'reverse_lateral_significant_mm')})
    stage = register('canonical_gate_band_shared_reference_transfer_joint_learning_20260920',
        [__file__, parent_file, source/'receipt.json', source/'selection_review.json', source/'results.json',
         source/'baseline.json'] + [BASE/name for name in (
             'joint_teacher_state_policy.py', 'observable_memory_joint_policy.py',
             'saved_memory_action_identity.py', 'consolidate_joint_teacher_policy.py',
             'learn_early_memory_joint_policy.py', 'learn_direct_observable_joint_policy.py',
             'duration_joint_metrics.py', 'shape_event_credit.py', 'learn_intrinsic_joint_recovery.py',
             'audit_context_restoration_scaling.py')],
        dict(wall_s=600, workers=4, target_selection='largest saved exit and significant lateral errors',
             target_paths=targets, centers_per_target=3, radius_choices=[.05,.1,.2,.3,.6],
             maximum_supported_paths=12, response_steps=[.004,.01,.01],
             comparison_arms=['steering_only','feed_only','preview_only','joint'],
             proposal_radii=[4.,16.], maximum_joint_corrections=4, coefficient_bounds=[-1.,1.]),
        'The observed-reference representation and remeasured joint response recovered a real '
        'feasible improvement on104. Test its use on the remaining training errors instead of '
        'more104-only tuning. Select observed states at lateral onset, before physical curvature '
        'peak, and just after that peak on the two worst saved corners. Time and path select '
        'training observations only; neither enters the deployed function. All centers use the '
        'same observed-reference geometry metric, compact support and two causal memories. '
        'Choose the broadest registered support affecting at most12 isolated training corners. '
        'Measure symmetric steering/feed/preview parameter responses through COMPLETE routes, '
        'compare single-head versus joint feasible descent using the same samples, then permit '
        'four true-return corrections. Retain every existing shape, physical, time, cruise, '
        'duration and intrinsic curvature guard. Failure triggers measured support/response '
        'review, not identical training extension.600s. Parameter learning, not PPO or a '
        'generalization claim. The guide, projection and braking chain remain unchanged.')
    folder = stage/'state_joint'
    folder.mkdir()
    deadline = started+580.
    write_json(stage/'baseline.json', data)
    history, blocks, pool = [], [], None
    try:
        observations, path_ids, decisions, selectors = [], [], [], []
        for r in baseline:
            with np.load(r['trace']) as z:
                obs, rows = z['observations'].copy(), z['rows'].copy()
            observations.extend(obs)
            path_ids.extend([r['path']]*len(obs))
            decisions.extend(range(len(obs)))
            if r['path'] in targets:
                lateral = lateral_motion(r['trace'])
                peak = intrinsic_metrics(rows)['peak_ms']
                ds = [max(0,lateral['intervals'][0]['start_ms']//20-1), max(0,peak//20-1), peak//20+1]
                selectors.extend(dict(path=r['path'],decision=d) for d in dict.fromkeys(ds))
        obs, ids, decision = np.asarray(observations), np.asarray(path_ids), np.asarray(decisions)
        innovations = np.zeros((len(obs),3))
        raw, _ = state_pair(reference_oriented_observation(obs), innovations)
        count = len(selectors)
        mean, std, weights = [np.repeat(parent[k][-1:], count, axis=0) for k in
                             ('per_center_mean','per_center_std','per_center_distance_weights')]
        assert not np.any(weights[:,len(INDICES):]), 'Zero innovation cache requires zero metric weights'
        centers = np.array([(raw[np.flatnonzero((ids==s['path'])&(decision==s['decision'])).item()]-mean[k])/std[k]
                            for k,s in enumerate(selectors)])
        md = dict(architecture='observable_memory_joint_radial', parent=parent, centers=centers,
            mean=mean[0], std=std[0], per_center_mean=mean, per_center_std=std,
            per_center_distance_weights=weights, per_center_reference_frame=np.ones(count,dtype=bool),
            memory_decays=[.5,.8], coefficients=np.zeros((3*count,3)), accepted=False,
            action_interval_ms=20, physical_tick_ms=1, reused_candidates=1,
            learning_method='shared_observed_reference_complete_joint_parameter_learning')
        choices = []
        corner_ids = {r['path'] for r in corners}
        for width in (.05,.1,.2,.3,.6):
            md['radii'] = np.full(count,width)
            f = features(md,obs,innovations)
            active = np.any(f!=0.,axis=(1,2))
            paths = sorted(np.unique(ids[active]).tolist())
            choices.append(dict(width=width,paths=paths,decisions=int(active.sum())))
        eligible = [c for c in choices if set(targets)<=set(c['paths'])<=corner_ids and len(c['paths'])<=12]
        assert eligible, 'Registered support choices exceed evaluator scope; revise representation'
        choice = max(eligible,key=lambda c:c['width'])
        md['radii'] = np.full(count,choice['width'])
        choice['variable_kernels'] = list(range(count))
        paths = choice['paths']
        selected = [r for r in baseline if r['path'] in paths]
        write_json(stage/'support_audit.json',dict(selected=choice,choices=choices,
            selectors_for_training_only=selectors,all_training_paths=len(baseline),
            outside_support_exact_zero=True,no_runtime_time_or_path_inputs=True))
        torch.save(md,stage/'initial_policy.pt')
        print('SHARED_REFERENCE_SUPPORT',targets,choice,flush=True)
        replay = collect_data(md,parent,selected,include_latents=True,deadline=deadline)
        replay['held_actions'] = np.vstack([np.load(r['trace'])['actions'] for r in selected])
        np.savez_compressed(stage/'affected_replay.npz',**replay)
        identity = SavedMemoryActionIdentity(md,replay,GateBandRuntime.decode_action)
        dimensions = count*9
        assert all(identity.unchanged(p,np.zeros(dimensions)) for p in paths)
        write_json(stage/'initial_action_identity.json',dict(decisions=len(replay['path_ids']),
            decoded_actions_exact=True,zero_support_paths=[r['path'] for r in baseline if r['path'] not in paths]))
        _, old_guards = metric_functions(data,selected,stage,parent,deadline)
        retained = [next(o['baseline_shape'] for o in data['retained'] if o['path']==r['path']) for r in selected]
        def entry(records,u,tag,actual,unchanged):
            original = np.concatenate([old_guards(records)]+[guards(r,b) for r,b in zip(records,selected)])
            extra = intrinsic_guards(records,selected)
            score = sum(teacher_score(r,b,o,barrier=False)+penalty(r)+
                (4*(r['shape']['reverse_lateral_total_mm']/.005)**2 if r['success'] else 1e6)
                for r,b,o in zip(records,selected,retained))
            return dict(tag=tag,coefficients=np.asarray(u).tolist(),records=records,score=float(score),
                measured_inequalities=original.tolist(),augmented_measured_inequalities=np.r_[original,extra].tolist(),
                augmented_learning_inequalities=np.r_[shape_event_learning_guards(records,selected,original),extra].tolist(),
                actual_rerun_paths=actual,exact_zero_support_paths=unchanged)
        zero = entry(copy.deepcopy(selected),np.zeros(dimensions),'reused_parent',[],paths)
        history.append(zero)
        def feasible(c):
            return parameters_in_bounds(c['coefficients'],-1.,1.) and min(c['augmented_measured_inequalities'])>=-1e-8
        def best():
            return min(filter(feasible,history),key=lambda c:c['score'])
        cache = {tuple(zero['coefficients']):zero}
        pool = mp.get_context('spawn').Pool(4,worker_initialize,(md,selected,folder,deadline))
        def evaluate(u,kind):
            u = np.asarray(u)
            assert parameters_in_bounds(u,-1.,1.)
            key = tuple(u.tolist())
            if key not in cache:
                if time.perf_counter()>=deadline:
                    raise TimeoutError('Shared reference joint learning wall budget')
                unchanged = [p for p in paths if identity.unchanged(p,u)]
                actual = [p for p in paths if p not in unchanged]
                tag = f'{kind}_{len(history)}'
                measured = pool.map(worker_evaluate,[(p,u,tag) for p in actual])
                by = {r['path']:r for r in measured}
                c = entry([by.get(r['path'],copy.deepcopy(r)) for r in selected],u,tag,actual,unchanged)
                history.append(c)
                cache[key] = c
                write_json(folder/'results.json',dict(candidates=history,best=best()['tag']))
                if len(history)%8==0 or kind!='response':
                    print('SHARED_REFERENCE',tag,c['score'],min(c['augmented_measured_inequalities']),best()['tag'],flush=True)
            return cache[key]
        steps = np.tile([.004,.01,.01],dimensions//3)
        pairs = []
        for j in range(dimensions):
            delta = np.eye(dimensions)[j]*steps
            pairs.append((evaluate(delta,'response'),evaluate(-delta,'response')))
        jac = np.array([(np.array(a['augmented_learning_inequalities'])-b['augmented_learning_inequalities'])/2 for a,b in pairs]).T
        gradient = np.array([(a['score']-b['score'])/2 for a,b in pairs])
        write_json(stage/'measured_joint_credit.json',dict(steps=steps.tolist(),jacobian=jac.tolist(),
            loss_gradient=gradient.tolist(),response_pairs=[[a['tag'],b['tag']] for a,b in pairs]))
        proposals = []
        for head,name in ((0,'steering_only'),(1,'feed_only'),(2,'preview_only'),(None,'joint')):
            indices = np.arange(dimensions) if head is None else np.arange(head,dimensions,3)
            for radius in (4.,16.):
                solved = descent(jac,gradient,np.array(zero['augmented_learning_inequalities']),
                                 np.zeros(dimensions),steps,indices,radius)
                block = dict(arm=name,radius=radius,solution=solved)
                if solved['success'] and solved['predicted_loss_change'] < -.001:
                    measured = evaluate(np.array(solved['scaled_parameter_step'])*steps,name)
                    block.update(measured=measured['tag'],actual_guard_min=min(measured['augmented_measured_inequalities']))
                    if head is None:
                        proposals.append(measured)
                blocks.append(block)
                write_json(stage/'optimization_blocks.json',blocks)
        if proposals:
            current = min(proposals,key=lambda c:max(0.,-min(c['augmented_measured_inequalities'])))
            for iteration in range(4):
                if feasible(current):
                    break
                u = np.array(current['coefficients'])
                aug_jac = np.vstack([jac,np.diag(steps),-np.diag(steps)])
                values = np.r_[current['augmented_learning_inequalities'],1+u,1-u]
                solved = solve_correction(aug_jac,gradient,values,max(0.,zero['score']-current['score']),radius=4.)
                block = dict(arm='actual_return_recovery',anchor=current['tag'],solution=solved)
                if not solved['success']:
                    blocks.append(block)
                    break
                delta = np.array(solved['scaled_parameter_step'])
                measured = evaluate(u+delta*steps,'recovery')
                block.update(measured=measured['tag'],actual_guard_min=min(measured['augmented_measured_inequalities']))
                blocks.append(block)
                norm = float(delta@delta)
                if norm <= 1e-18:
                    break
                observed = np.array(measured['augmented_learning_inequalities'])-current['augmented_learning_inequalities']
                jac += np.outer(observed-jac@delta,delta)/norm
                gradient += (measured['score']-current['score']-gradient@delta)*delta/norm
                current = measured
        pool.close()
        pool.join()
        pool = None
        finish(stage,md,baseline,data,best(),zero,history,blocks,choice,parent_file,started)
    except TimeoutError:
        if pool is not None:
            pool.terminate()
            pool.join()
        if history:
            finish(stage,md,baseline,data,best(),zero,history,blocks,choice,parent_file,started,budget_exhausted=True)
        else:
            receipt(stage,dict(status='failed_shared_reference_preparation',actual_new_candidates=0,
                wall_s=time.perf_counter()-started,adopted=False,goal_complete=False))
        raise
    except Exception as exc:
        if pool is not None:
            pool.terminate()
            pool.join()
        receipt(stage,dict(status='failed_shared_reference_joint_learning',error=str(exc),
            actual_new_candidates=max(0,len(history)-1),wall_s=time.perf_counter()-started,
            adopted=False,goal_complete=False))
        raise


if __name__ == '__main__':
    mp.freeze_support()
    main()
