"""Equal-capacity geometry and dynamics conditioning with identical PPO credit data."""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import copy
import argparse
import time
import multiprocessing as mp
import numpy as np
import torch
from observable_memory_joint_policy import actor_features,coefficient_context
from joint_teacher_state_policy import OUTPUT_SCALE,features
from reflection_policy import reflect_observation
from head_counterfactual_rollouts import exploration_sigma
from train_constraint_counterfactual_ppo import credit_problem,solve_update
from saved_memory_action_identity import SavedMemoryActionIdentity
from gate_band_state import GateBandRuntime
from learn_complete_joint_teachers import teacher_score,parameters_in_bounds
from duration_joint_metrics import penalty
from learn_early_memory_joint_policy import worker_initialize,worker_evaluate
from learn_direct_observable_joint_policy import finish
from study_io import BASE,register,write_json,receipt,sha
from search_complete_return import read


def contextualize(original,episodes,mode):
    all_obs = []
    observations = {}
    for episode in episodes:
        with np.load(episode['record']['trace']) as z:
            obs = z['observations'].copy()
        observations[episode['path'],episode['seed']] = obs
        all_obs.extend(obs)
    all_obs = np.asarray(all_obs)
    context = coefficient_context(all_obs,mode)
    mean = context.mean(0)
    std = np.maximum(context.std(0),[5.,100.] if mode=='dynamic' else [1.,1.])
    md = copy.deepcopy(original)
    count = len(md['centers'])
    md.update(dynamic_context_count=2,dynamic_context_basis='linear',coefficient_context=mode,
        dynamic_context_mean=mean,dynamic_context_scale=std,exploration_geometry_only=True,
        coefficients=np.zeros((9*count,3)),learning_method='matched_context_conditional_constraint_ppo')
    transformed = copy.deepcopy(episodes)
    old_indices = np.concatenate([np.arange(count)+i*3*count for i in range(3)])
    for episode in transformed:
        obs = observations[episode['path'],episode['seed']]
        samples = {s['decision']:s for s in episode['samples']}
        memory = np.zeros((2,3*count,3))
        previous = np.zeros((3*count,3))
        alpha = np.asarray(md['memory_decays'])[:,None,None]
        for j,x in enumerate(obs):
            f = actor_features(md,x,np.zeros(3))
            # The support metric has exactly zero weight on innovations.
            assert np.array_equal(f[:count],features(original,x,np.zeros(3)))
            memory = alpha*memory+(1.-alpha)*previous
            previous = f.copy()
            if j not in samples:
                continue
            design = np.concatenate([f,*memory],axis=0)*OUTPUT_SCALE
            s = samples[j]
            assert np.array_equal(design[old_indices],np.array(s['design'])), ('original_feature_identity',mode,episode['path'],j)
            assert np.array_equal(exploration_sigma(md,design),np.array(s['sigma']))
            mirrored = reflect_observation(torch.as_tensor(x)).numpy()
            assert np.array_equal(actor_features(md,mirrored,np.zeros(3)),f*np.array([-1.,1.,1.]))
            s['design'] = design.tolist()
    audit = dict(mode=mode,parameters=md['coefficients'].size,context_mean=mean.tolist(),
        context_scale=std.tolist(),query_states=sum(len(e['samples']) for e in transformed),
        original_geometry_and_memory_features_bitwise=True,initial_exploration_sigma_bitwise=True,
        reflection_bitwise=True,no_future_context=True,no_action_labels=True)
    return md,transformed,audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scaled-solver',action='store_true')
    parser.add_argument('--roundoff-solver',action='store_true')
    parser.add_argument('--warm-solver',action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    started = time.perf_counter()
    source = BASE/'canonical_gate_band_head_counterfactual_ppo_collection_20260920'
    previous = BASE/'canonical_gate_band_conditional_sample_credit_ppo_20260920'
    replay_file = BASE/'canonical_gate_band_shared_reference_transfer_joint_learning_20260920/affected_replay.npz'
    parent_file = BASE/'canonical_gate_band_reference_supported_joint_learning_20260920/research_candidate.pt'
    assert not read(previous/'receipt.json')['research_acceptable']
    assert sha(parent_file)=='ded2a93f85f3503a8c09bf59fd79e48c8f2cd187874f07789c53b8863c936229'
    original = torch.load(source/'initial_policy.pt',map_location='cpu',weights_only=False)
    data = read(source/'baseline.json')
    baseline = data['baseline']['records']
    support = read(source/'support_audit.json')
    choice = support['selected']
    paths = choice['paths']
    selected = [r for r in baseline if r['path'] in paths]
    episodes = read(source/'on_policy_episodes.json')['episodes']
    episodes.sort(key=lambda r:(r['path'],r['seed']))
    with np.load(replay_file) as z:
        replay = {k:z[k].copy() for k in z.files}
    stage = register('canonical_gate_band_warm_context_counterfactual_ppo_20260920' if args.warm_solver else
                     'canonical_gate_band_roundoff_context_counterfactual_ppo_20260920' if args.roundoff_solver else
                     'canonical_gate_band_scaled_context_counterfactual_ppo_20260920' if args.scaled_solver
                     else 'canonical_gate_band_matched_context_counterfactual_ppo_20260920',
        [__file__,parent_file,source/'initial_policy.pt',source/'on_policy_episodes.json',
         source/'ppo_update_audit.json',source/'receipt.json',previous/'receipt.json',
         previous/'constraint_update_audit.json',replay_file]+[BASE/n for n in (
             'observable_memory_joint_policy.py','joint_teacher_state_policy.py',
             'saved_memory_action_identity.py','head_counterfactual_rollouts.py',
             'train_constraint_counterfactual_ppo.py','duration_joint_metrics.py',
             'shape_event_credit.py','learn_intrinsic_joint_recovery.py','learn_direct_observable_joint_policy.py')],
        dict(wall_s=300,workers=4,new_stochastic_episodes=0,reused_counterfactual_suffixes=144,
             arms=['geometry','dynamic'],parameters_per_arm=162,kl_bounds=[1e-5,1e-4],
             maximum_new_policies=4,support_equalities=True,conditional_draws_per_head=3,
             numerical_objective_scale=1e6 if args.scaled_solver else 1.,
             roundoff_prune=args.roundoff_solver),
        ('Initialize each numerical optimizer from the exact same embedded previously '
         'measured 54-parameter surrogate-feasible direction at its corresponding KL '
         'budget. Both actors still use the same old behavior distribution and PPO '
         'objective. The warm witness failed real trajectory guards; it is never '
         'declared acceptable. This tests solver initialization after zero-start '
         'scaling and rigorously bounded pruning failed to produce updates. ' if args.warm_solver else '')+
        ('Remove from the internal SLSQP active set ONLY channels whose normalized '
         'variation is bounded below1e-12 over the entire original coefficient-coordinate '
         'box. Recheck ALL original surrogate channels and every physical guard after '
         'optimization. Same objective scale1, data, coordinates, bounds and KL. This '
         'addresses numerically inert active constraints, without a physical tolerance '
         'change. ' if args.roundoff_solver else '')+
        ('Nested-coordinate audit preserved old nonzero feasible surrogate witnesses in both '
         'actors, but the original solver stopped at zero. Change ONLY positive numerical '
         'objective scale to1e6, preserving the mathematical surrogate, constraints, actor '
         'coordinates, data, KL limits and iteration budget. This is a numerical optimizer '
         'control; improved surrogate optimization is not physical acceptance. ' if args.scaled_solver else '')+
        'Algorithm-level representation comparison after conditional constraint-credit '
        'updates still miss real guards. Keep stable observed-reference GEOMETRY '
        'support and separate it from state-dependent three-action control amplitudes. '
        'Compare equal162-parameter actors: two geometric context scalars versus speed '
        'and tangential acceleration, each multiplying the SAME six state kernels and '
        'two causal memories. Context is stored when observed, not read from a future '
        'state. Reuse the SAME8 on-policy episodes and144 independent head replacement '
        'returns; reconstruct expanded features from their recorded observations. '
        'Zero contextual coefficients must reproduce the old mean and Gaussian '
        'exploration variance exactly, so this is still the same behavior policy. '
        'Keep three conditional draws per head, leave-one-out advantages,all203 '
        'constraint-credit channels,unsampled-support output equations,two KL '
        'budgets and solver settings identical. The two arms have equal capacity; '
        'comparison to earlier54 weights also changes capacity and is not attributed '
        'to dynamics alone. Every new mean policy must complete every changed route '
        'and pass original guards. No new kernels,path/time/angle rules,action labels, '
        'new data collection,final acceptance,independent test or claim that PPO '
        'credit problems have all been solved.300s then revise from actual evidence.')
    deadline = started+280.
    folder = stage/'state_joint'
    folder.mkdir()
    write_json(stage/'baseline.json',data)
    write_json(stage/'support_audit.json',support)
    scale = read(source/'ppo_update_audit.json')['global_advantage_scale']
    retained = [next(o['baseline_shape'] for o in data['retained'] if o['path']==r['path']) for r in selected]
    history,blocks,models = [],[],{}
    pool = None
    try:
        for mode in ('geometry','dynamic'):
            md,transformed,audit = contextualize(original,episodes,mode)
            models[mode] = md
            torch.save(md,stage/(mode+'_initial_policy.pt'))
            write_json(stage/(mode+'_feature_identity.json'),audit)
            identity = SavedMemoryActionIdentity(md,replay,GateBandRuntime.decode_action)
            assert all(identity.unchanged(p,np.zeros(md['coefficients'].size)) for p in paths)
            credit_folder = stage/(mode+'_credit')
            credit_folder.mkdir()
            metrics,identity,matrix,noise,advantage,credit,weights,parent_matrix,basis = credit_problem(
                credit_folder,md,data,selected,transformed,replay,deadline,True)
            def entry(records,u,tag):
                first,measured,learning = metrics(records)
                score = sum(teacher_score(r,b,o,barrier=False)+penalty(r)+
                    (4*(r['shape']['reverse_lateral_total_mm']/.005)**2 if r['success'] else 1e6)
                    for r,b,o in zip(records,selected,retained))
                return dict(tag=tag,context_arm=mode,coefficients=np.asarray(u).tolist(),records=records,score=float(score),
                    measured_inequalities=first.tolist(),augmented_measured_inequalities=measured.tolist(),
                    augmented_learning_inequalities=learning.tolist(),actual_rerun_paths=[],exact_zero_support_paths=paths)
            if not history:
                zero = entry(copy.deepcopy(selected),np.zeros(md['coefficients'].size),'reused_parent')
                history.append(zero)
            def feasible(c):
                return parameters_in_bounds(c['coefficients'],-1.,1.) and min(c['augmented_measured_inequalities'])>=-1e-8
            def best():
                return min(filter(feasible,history),key=lambda c:c['score'])
            pool = mp.get_context('spawn').Pool(4,worker_initialize,(md,selected,folder,deadline))
            for kl in (1e-5,1e-4):
                if time.perf_counter()>=deadline:
                    raise TimeoutError('Matched context PPO wall budget')
                initial = None
                if args.warm_solver:
                    old_history = read(previous/'state_joint/results.json')['candidates']
                    witness = next(c for c in old_history if c['tag']==f'constraint_and_support_kl{kl}')
                    ids = np.array([h+3*(k+18*m) for m in range(3) for k in range(6) for h in range(3)])
                    initial = np.zeros(md['coefficients'].size)
                    initial[ids] = witness['coefficients']
                u,report = solve_update(matrix,noise,advantage/scale,credit,weights,
                    np.array(zero['augmented_learning_inequalities']),parent_matrix,basis,kl,
                    numerical_scale=1e6 if args.scaled_solver else 1.,roundoff_prune=args.roundoff_solver,
                    initial_coefficients=initial)
                block = dict(context_arm=mode,kl_bound=kl,optimizer=report)
                if len(u)==md['coefficients'].size and report.get('surrogate_gain',0.)>1e-9:
                    assert parameters_in_bounds(u,-1.,1.)
                    tag = f'{mode}_kl{kl}'
                    actor = copy.deepcopy(md)
                    actor['coefficients'] = u.reshape(-1,3)
                    torch.save(actor,stage/(tag+'.pt'))
                    unchanged = [p for p in paths if identity.unchanged(p,u)]
                    actual = [p for p in paths if p not in unchanged]
                    records = pool.map(worker_evaluate,[(p,u,tag) for p in actual])
                    by = {r['path']:r for r in records}
                    c = entry([by.get(r['path'],copy.deepcopy(r)) for r in selected],u,tag)
                    c.update(actual_rerun_paths=actual,exact_zero_support_paths=unchanged)
                    history.append(c)
                    block.update(measured=tag,actual_score=c['score'],guard_min=min(c['augmented_measured_inequalities']),
                                 unchanged_paths=unchanged)
                    write_json(folder/'results.json',dict(candidates=history,best=best()['tag']))
                blocks.append(block)
                write_json(stage/'matched_context_updates.json',blocks)
                print('MATCHED_CONTEXT_PPO',block,flush=True)
            pool.close()
            pool.join()
            pool = None
        chosen = best()
        md = models[chosen['context_arm']]
        md['reused_candidates'] = 1
        torch.save(md,stage/'initial_policy.pt')
        finish(stage,md,baseline,data,chosen,zero,history,blocks,choice,parent_file,started)
        receipt(stage,dict(status='completed_matched_context_counterfactual_ppo',new_stochastic_episodes=0,
            reused_counterfactual_suffixes=144,deterministic_policies=len(history)-1,selected=chosen['tag'],
            research_acceptable=(stage/'research_candidate.pt').exists(),source_model_sha256=sha(parent_file),
            wall_s=time.perf_counter()-started,adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:
            pool.terminate()
            pool.join()
        receipt(stage,dict(status='failed_matched_context_counterfactual_ppo',error=str(exc),
            deterministic_policies=max(0,len(history)-1),wall_s=time.perf_counter()-started,
            adopted=False,goal_complete=False))
        raise


if __name__ == '__main__':
    mp.freeze_support()
    main()
