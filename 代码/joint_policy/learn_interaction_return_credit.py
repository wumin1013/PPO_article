"""Matched additive and interaction-aware credit on shared complete-route returns."""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import copy
import itertools
import time
import multiprocessing as mp
import numpy as np
import torch
from scipy.optimize import minimize
from gate_band_state import GateBandRuntime
from saved_memory_action_identity import SavedMemoryActionIdentity
from duration_joint_metrics import guards, penalty
from shape_event_credit import shape_event_learning_guards
from learn_intrinsic_joint_recovery import intrinsic_guards
from learn_complete_joint_teachers import teacher_score, parameters_in_bounds
from learn_early_memory_joint_policy import worker_initialize, worker_evaluate, metric_functions
from learn_direct_observable_joint_policy import finish
from audit_early_geometry_outcomes import constraint_names
from study_io import BASE, register, receipt, write_json, sha
from search_complete_return import read


def response_features(x, interactions):
    x = np.asarray(x)
    terms = [x, x*x]
    if interactions:
        terms.append(np.stack([x[...,i]*x[...,j] for i in range(4) for j in range(i+1,4)],axis=-1))
    return np.concatenate(terms,axis=-1)


def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    source = BASE/'canonical_gate_band_shared_reference_transfer_joint_learning_20260920'
    parent_file = BASE/'canonical_gate_band_reference_supported_joint_learning_20260920/research_candidate.pt'
    assert sha(parent_file)=='ded2a93f85f3503a8c09bf59fd79e48c8f2cd187874f07789c53b8863c936229'
    assert read(source/'receipt.json')['status'].startswith('completed')
    old = read(source/'state_joint/results.json')['candidates']
    md = torch.load(source/'initial_policy.pt',map_location='cpu',weights_only=False)
    data = read(source/'baseline.json')
    baseline = data['baseline']['records']
    support = read(source/'support_audit.json')
    choice = support['selected']
    paths = choice['paths']
    selected = [r for r in baseline if r['path'] in paths]
    joint = np.array(next(c for c in old if c['tag']=='joint_111')['coefficients'])
    feed = np.array(next(c for c in old if c['tag']=='feed_only_110')['coefficients'])
    indices = np.arange(len(joint)).reshape(3,6,3)
    early = indices[:,[0,3],:].ravel()
    late = indices[:,[1,2,4,5],:]
    basis = np.zeros((len(joint),4))
    basis[early,0] = joint[early]
    basis[late[:,:,0].ravel(),1] = joint[late[:,:,0].ravel()]
    basis[late[:,:,1:].ravel(),2] = joint[late[:,:,1:].ravel()]
    basis[:,3] = feed
    assert np.array_equal(basis[:,:3].sum(axis=1),joint)
    assert np.linalg.matrix_rank(basis)==4
    grid = [np.asarray(v,dtype=float) for v in itertools.product((0.,1.),repeat=4)]
    grid += [np.eye(4)[i]*.5 for i in range(4)] + [np.full(4,.5)]
    validation = np.random.default_rng(20260920).uniform(.2,.8,(8,4))
    stage = register('canonical_gate_band_interaction_return_credit_learning_20260920',
        [__file__,parent_file,source/'receipt.json',source/'state_joint/results.json',
         source/'initial_policy.pt',source/'affected_replay.npz',source/'support_audit.json',
         source/'shared_credit_review.json'] + [BASE/n for n in (
             'joint_teacher_state_policy.py','observable_memory_joint_policy.py',
             'saved_memory_action_identity.py','learn_early_memory_joint_policy.py',
             'duration_joint_metrics.py','shape_event_credit.py','learn_intrinsic_joint_recovery.py')],
        dict(wall_s=420,workers=4,training_design_points=len(grid),fresh_validation_points=8,
             fit_models=['additive_quadratic','pair_interaction_quadratic'],
             factors=['early_three_actions_and_memories','later_steering_and_memories',
                      'later_feed_preview_and_memories','measured_feed_recovery'],
             proposal_bound=[0.,1.],maximum_new_proposals_per_model=3,
             coefficient_bounds=[-1.,1.]),
        'Change the UPDATE CREDIT MODEL, keeping shared observable-state representation, '
        'parent, executor, training paths, objective and all acceptance guards identical. '
        'The previous additive local response predicts feasible joint updates but real '
        'complete trajectories violate several shape and cruise guards. Group its shared '
        'weight direction into early actions, later steering, later feed/preview, plus the '
        'measured feasible feed compensation. Collect a full binary factorial plus four '
        'axis midpoints and center. Fit matched additive quadratic versus quadratic with '
        'all pair interactions to the SAME complete route returns. Eight preregistered '
        'fresh parameter combinations are used only to evaluate prediction, never to fit '
        'these two compared models. Both models propose at most3 constrained weight '
        'updates with identical bounds and optimization starts. Execute every new proposal '
        'through complete routes and select only original-guard feasible improvement. '
        'This tests cross-time/cross-action credit for a shared policy on seven training '
        'corners, not route schedules or feed-label fitting, not PPO and not final '
        'generalization. No new state kernels or acceptance relaxation.420s.')
    folder = stage/'state_joint'
    folder.mkdir()
    deadline = started+400.
    write_json(stage/'baseline.json',data)
    write_json(stage/'support_audit.json',support)
    torch.save(md,stage/'initial_policy.pt')
    write_json(stage/'factor_design.json',dict(basis=basis.tolist(),training=[x.tolist() for x in grid],
        validation=validation.tolist(),no_runtime_factor_or_phase_inputs=True))
    with np.load(source/'affected_replay.npz') as z:
        replay = {k:z[k].copy() for k in z.files}
    identity = SavedMemoryActionIdentity(md,replay,GateBandRuntime.decode_action)
    _, old_guards = metric_functions(data,selected,stage,md['parent'],deadline)
    retained = [next(o['baseline_shape'] for o in data['retained'] if o['path']==r['path']) for r in selected]
    def entry(records,u,tag,actual,unchanged):
        original = np.concatenate([old_guards(records)]+[guards(r,b) for r,b in zip(records,selected)])
        extra = intrinsic_guards(records,selected)
        score = sum(teacher_score(r,b,o,barrier=False)+penalty(r)+
            (4*(r['shape']['reverse_lateral_total_mm']/.005)**2 if r['success'] else 1e6)
            for r,b,o in zip(records,selected,retained))
        return dict(tag=tag,coefficients=u.tolist(),records=records,score=float(score),
            measured_inequalities=original.tolist(),augmented_measured_inequalities=np.r_[original,extra].tolist(),
            augmented_learning_inequalities=np.r_[shape_event_learning_guards(records,selected,original),extra].tolist(),
            actual_rerun_paths=actual,exact_zero_support_paths=unchanged)
    history = copy.deepcopy(old)
    zero = history[0]
    assert entry(selected,np.zeros(len(joint)),'check',[],paths)['score']==zero['score']
    md['reused_candidates'] = len(history)
    def feasible(c):
        return parameters_in_bounds(c['coefficients'],-1.,1.) and min(c['augmented_measured_inequalities'])>=-1e-8
    def best():
        return min(filter(feasible,history),key=lambda c:c['score'])
    cache = {tuple(c['coefficients']):c for c in history}
    pool, blocks = None, []
    try:
        pool = mp.get_context('spawn').Pool(4,worker_initialize,(md,selected,folder,deadline))
        def evaluate(x,kind):
            u = basis@np.asarray(x)
            assert parameters_in_bounds(u,-1.,1.)
            key = tuple(u.tolist())
            if key not in cache:
                if time.perf_counter()>=deadline:
                    raise TimeoutError('Interaction-return credit wall budget')
                unchanged = [p for p in paths if identity.unchanged(p,u)]
                actual = [p for p in paths if p not in unchanged]
                tag = f'{kind}_{len(history)}'
                measured = pool.map(worker_evaluate,[(p,u,tag) for p in actual])
                by = {r['path']:r for r in measured}
                c = entry([by.get(r['path'],copy.deepcopy(r)) for r in selected],u,tag,actual,unchanged)
                c['credit_factors'] = np.asarray(x).tolist()
                history.append(c)
                cache[key] = c
                write_json(folder/'results.json',dict(candidates=history,best=best()['tag']))
                print('INTERACTION_RETURN',tag,c['score'],min(c['augmented_measured_inequalities']),flush=True)
            return cache[key]
        train = [evaluate(x,'factorial') for x in grid]
        y0 = np.r_[zero['score'],zero['augmented_learning_inequalities']]
        values = np.array([np.r_[c['score'],c['augmented_learning_inequalities']] for c in train])-y0
        models = []
        for interactions in (False,True):
            design = response_features(np.array(grid),interactions)
            coefficient,_,rank,singular = np.linalg.lstsq(design,values,rcond=None)
            assert rank==design.shape[1]
            models.append(dict(name='interaction' if interactions else 'additive',interactions=interactions,
                coefficients=coefficient.tolist(),rank=int(rank),condition=float(singular[0]/singular[-1]),
                training_max_error=np.max(abs(design@coefficient-values),axis=0).tolist()))
        write_json(stage/'matched_credit_models.json',dict(models=models,training_tags=[c['tag'] for c in train],
            same_training_data=True,zero_anchor_exact=True,validation_used_for_fit=False))
        valid = [evaluate(x,'unfitted_validation') for x in validation]
        actual = np.array([np.r_[c['score'],c['augmented_learning_inequalities']] for c in valid])
        names = ['score']+constraint_names(paths)
        comparisons = []
        for model in models:
            predicted = y0+response_features(validation,model['interactions'])@np.array(model['coefficients'])
            error = predicted-actual
            per_metric = np.sqrt(np.mean(error*error,axis=0))
            comparisons.append(dict(name=model['name'],score_rmse=float(per_metric[0]),
                guard_rmse=float(np.sqrt(np.mean(error[:,1:]**2))),
                prediction_errors={name:float(e) for name,e in zip(names,per_metric)},
                predicted_feasible=[bool(v[1:].min()>=-1e-8) for v in predicted],
                actual_feasible=[feasible(c) for c in valid],validation_tags=[c['tag'] for c in valid]))
        write_json(stage/'unfitted_combination_validation.json',dict(comparisons=comparisons,
            validation_used_for_fit=False,validation_scope='New weight combinations on training paths; not independent path generalization.'))
        print('MATCHED_CREDIT_VALIDATION',[(c['name'],c['score_rmse'],c['guard_rmse']) for c in comparisons],flush=True)
        for model in models:
            coefficient = np.array(model['coefficients'])
            def predict(x):
                return y0+response_features(x,model['interactions'])@coefficient
            proposals = []
            for initial in (np.zeros(4),np.array([0.,0.,0.,1.]),np.full(4,.25),np.full(4,.5)):
                solved = minimize(lambda x:predict(x)[0],initial,method='SLSQP',bounds=[(0.,1.)]*4,
                    constraints=[dict(type='ineq',fun=lambda x:predict(x)[1:])],
                    options=dict(maxiter=120,ftol=1e-10))
                if np.min(predict(solved.x)[1:])>=-1e-8 and not any(np.max(abs(solved.x-x))<1e-8 for x in proposals):
                    proposals.append(solved.x)
            proposals = sorted(proposals,key=lambda x:predict(x)[0])[:3]
            for x in proposals:
                if predict(x)[0]>=zero['score']-.001:
                    continue
                c = evaluate(x,model['name']+'_update')
                blocks.append(dict(model=model['name'],factors=x.tolist(),predicted_score=float(predict(x)[0]),
                    predicted_guard_minimum=float(predict(x)[1:].min()),measured=c['tag'],actual_score=c['score'],
                    actual_guard_minimum=min(c['augmented_measured_inequalities'])))
            write_json(stage/'optimization_blocks.json',blocks)
        pool.close()
        pool.join()
        pool = None
        finish(stage,md,baseline,data,best(),zero,history,blocks,choice,parent_file,started)
        write_json(stage/'algorithm_decision.json',dict(parent_sha256=sha(parent_file),
            best=best()['tag'],new_candidates=len(history)-md['reused_candidates'],
            new_feasible_joint_candidates=[c['tag'] for c in history[md['reused_candidates']:] if feasible(c)
                and np.any(np.array(c['coefficients'])[::3]) and np.any(np.array(c['coefficients'])[2::3])],
            model_comparisons=[{k:c[k] for k in ('name','score_rmse','guard_rmse')} for c in comparisons],
            formal_adoption=False,goal_complete=False,ppo_estimator_bias_claim=False))
    except TimeoutError:
        if pool is not None:
            pool.terminate()
            pool.join()
        finish(stage,md,baseline,data,best(),zero,history,blocks,choice,parent_file,started,budget_exhausted=True)
    except Exception as exc:
        if pool is not None:
            pool.terminate()
            pool.join()
        receipt(stage,dict(status='failed_interaction_return_credit',error=str(exc),
            actual_new_candidates=len(history)-md['reused_candidates'],wall_s=time.perf_counter()-started,
            adopted=False,goal_complete=False))
        raise


if __name__ == '__main__':
    mp.freeze_support()
    main()
