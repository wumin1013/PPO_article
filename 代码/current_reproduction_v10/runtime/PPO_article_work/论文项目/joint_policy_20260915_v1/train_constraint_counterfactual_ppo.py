"""Constrained PPO surrogates from measured per-head, per-metric complete returns."""
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
from scipy.linalg import null_space
from scipy.optimize import minimize
from head_counterfactual_rollouts import NOISE_SCALE,exploration_sigma
from saved_memory_action_identity import SavedMemoryActionIdentity
from gate_band_state import GateBandRuntime
from joint_teacher_state_policy import OUTPUT_SCALE
from duration_joint_metrics import guards, penalty
from shape_event_credit import shape_event_learning_guards
from learn_intrinsic_joint_recovery import intrinsic_guards
from learn_complete_joint_teachers import teacher_score, parameters_in_bounds
from learn_early_memory_joint_policy import worker_initialize, worker_evaluate, metric_functions
from learn_direct_observable_joint_policy import finish
from audit_early_geometry_outcomes import constraint_names
from study_io import BASE, register, write_json, receipt, sha
from search_complete_return import read


def design_matrix(design, sigma):
    normalized = np.divide(design,sigma[:,None,:],out=np.zeros_like(design),where=sigma[:,None,:]>0.)
    result = np.zeros((len(design),3,design.shape[1]*3))
    for h in range(3):
        result[:,h,h::3] = normalized[:,:,h]
    return result


def credit_problem(stage, md, data, selected, episodes, replay, deadline, reuse_conditional=False):
    _, old_guards = metric_functions(data,selected,stage,md['parent'],deadline)
    def metrics(records):
        original = np.concatenate([old_guards(records)]+[guards(r,b) for r,b in zip(records,selected)])
        extra = intrinsic_guards(records,selected)
        return original, np.r_[original,extra], np.r_[shape_event_learning_guards(records,selected,original),extra]
    def replace(r):
        return [r if b['path']==r['path'] else b for b in selected]
    identity = SavedMemoryActionIdentity(md,replay,GateBandRuntime.decode_action)
    samples, advantages, credit, weights, conditional_noise = [], [], [], [], []
    paths = {e['path'] for e in episodes}
    for episode in episodes:
        actual = metrics(replace(episode['record']))[2]
        by_query = {(r['decision'],r['head'],r['repeat']):r for r in episode['counterfactuals']}
        f,memory,_,_ = identity.paths[episode['path']]
        supported = int(np.count_nonzero(np.any(f!=0.,axis=(1,2)) | np.any(memory!=0.,axis=(1,2,3))))
        replicate_count = sum(e['path']==episode['path'] for e in episodes)
        for s in episode['samples']:
            samples.append(s)
            rows, reward_rows, noise_rows = [], [], []
            for h in range(3):
                q = [metrics(replace(by_query[s['decision'],h,j]['record']))[2] for j in range(2)]
                if reuse_conditional:
                    values = np.array([actual,*q])
                    rewards = np.array([episode['observed_return']]+[by_query[s['decision'],h,j]['total_return'] for j in range(2)])
                    # Each sample's baseline contains only the OTHER independent head draws.
                    rows.append(values-(values.sum(0)-values)/2.)
                    reward_rows.append(rewards-(rewards.sum()-rewards)/2.)
                    noise_rows.append([s['noise'][h]]+[by_query[s['decision'],h,j]['replacement_noise'] for j in range(2)])
                else:
                    rows.append(actual-np.mean(q,axis=0))
            credit.append(rows)
            advantages.append(reward_rows if reuse_conditional else s['counterfactual_advantage'])
            conditional_noise.append(noise_rows if reuse_conditional else s['noise'])
            # Stratified queries supply their predeclared inverse inclusion probability.
            query_weight = s.get('query_weight')
            weights.append((supported/len(episode['samples']) if query_weight is None else query_weight)/replicate_count)
    design = np.array([s['design'] for s in samples])
    sigma = np.array([s['sigma'] for s in samples])
    matrix = design_matrix(design,sigma)
    parent_matrix, equality = [], []
    for p,(f,memory,_,_) in identity.paths.items():
        d = np.concatenate([f,memory[:,0],memory[:,1]],axis=1)*OUTPUT_SCALE
        sd = exploration_sigma(md,d)
        a = design_matrix(d,sd).reshape(-1,matrix.shape[-1])
        a = a[np.any(a!=0.,axis=1)]
        parent_matrix.extend(a)
        if p not in paths:
            equality.extend(a)
    equality = np.array(equality)
    protected_basis = null_space(equality,rcond=1e-12)
    report = dict(sampled_paths=sorted(paths),unsampled_supported_paths=sorted(set(identity.paths)-paths),
        equality_rows=len(equality),equality_rank=matrix.shape[-1]-protected_basis.shape[1],
        protected_parameter_dimensions=protected_basis.shape[1],constraint_channels=np.asarray(credit).shape[-1],
        conditional_draws_per_head=3 if reuse_conditional else 1,
        no_new_physical_data=True,not_a_claim_of_unbiased_full_trajectory_constraint_gradient=True)
    write_json(stage/'credit_coverage_audit.json',report)
    np.savez_compressed(stage/'measured_constraint_credit.npz',matrix=matrix,noise=np.array(conditional_noise),
        advantages=np.array(advantages),credit=np.array(credit),weights=np.array(weights),
        parent_matrix=np.array(parent_matrix),equality=equality,protected_basis=protected_basis)
    print('CONSTRAINT_CREDIT_COVERAGE',report,flush=True)
    return metrics, identity, matrix, np.array(conditional_noise), np.array(advantages), np.array(credit), np.array(weights), np.array(parent_matrix), protected_basis


def solve_update(matrix, noise, advantage, credit, weights, baseline, parent_matrix, basis, kl_limit,
                 numerical_scale=1., roundoff_prune=False, initial_coefficients=None):
    if noise.ndim==2:
        noise, advantage, credit = noise[...,None],advantage[...,None],credit[:,:,None,:]
    a = NOISE_SCALE*np.einsum('nhd,dk->nhk',matrix,basis)
    parent = NOISE_SCALE*parent_matrix@basis
    weighted_credit = credit*weights[:,None,None,None]/noise.shape[-1]
    active = np.max(abs(weighted_credit),axis=(0,1,2))>0.
    assert np.all(baseline[~active]>=0.)
    weighted_credit, baseline = weighted_credit[:,:,:,active],baseline[active]
    scale = np.maximum(np.max(abs(weighted_credit),axis=(0,1,2)),1e-12)
    # Bound each channel's variation across the ENTIRE existing coordinate box.
    # Some protected outputs leave only roundoff-sized matrix entries. Keeping their
    # zero-margin inequalities in SLSQP can impose spurious active half-spaces.
    shift_bound = .5*np.sum(abs(a),axis=-1)
    ratio_bound = np.expm1(np.minimum(abs(noise)*shift_bound[...,None]+.5*shift_bound[...,None]**2,700.))
    variation_bound = np.einsum('nhk,nhkm->m',ratio_bound,abs(weighted_credit))/scale
    live = variation_bound>1e-12 if roundoff_prune else np.ones(len(scale),dtype=bool)
    def quantities(z):
        shift = np.einsum('nhd,d->nh',a,z)
        ratio = np.exp(-.5*((noise-shift[...,None])**2-noise**2))
        derivative = ratio[...,None]*(noise-shift[...,None])[...,None]*a[:,:,None,:]
        return shift,ratio,derivative
    def objective(z):
        _,ratio,derivative = quantities(z)
        clipped = np.clip(ratio,.8,1.2)
        active = ratio*advantage <= clipped*advantage
        value = -np.minimum(ratio*advantage,clipped*advantage).sum()/(len(noise)*noise.shape[-1])
        gradient = -np.einsum('nhk,nhkd->d',advantage*active,derivative)/(len(noise)*noise.shape[-1])
        return value,gradient
    def constraints(z, full=False):
        shift,ratio,derivative = quantities(z)
        delta = np.einsum('nhk,nhkm->m',ratio-1.,weighted_credit)
        jac = np.einsum('nhkd,nhkm->md',derivative,weighted_credit)
        parent_shift = parent@z
        keep = np.ones(len(scale),dtype=bool) if full else live
        values = np.r_[((baseline+delta)/scale)[keep],kl_limit-.5*np.mean(np.sum(shift*shift,axis=1)),
                       kl_limit-.5*np.mean(parent_shift*parent_shift)*3.]
        gradients = np.vstack([(jac/scale[:,None])[keep],-np.einsum('nh,nhd->d',shift,a)/len(noise),
                               -3.*parent_shift@parent/len(parent)])
        return values,gradients
    zero = np.zeros(basis.shape[1])
    if not len(zero):
        return zero,dict(success=False,message='No parameter direction preserves unsampled support')
    # Positive objective scaling changes stopping precision, not the PPO objective or feasible set.
    initial = zero if initial_coefficients is None else basis.T@np.asarray(initial_coefficients)/NOISE_SCALE
    assert np.max(abs(initial),initial=0.)<=.5+1e-12
    assert np.min(constraints(initial,full=True)[0])>=-1e-10
    solved = minimize(lambda z:numerical_scale*objective(z)[0],initial,
        jac=lambda z:numerical_scale*objective(z)[1],method='SLSQP',
        bounds=[(-.5,.5)]*len(zero),constraints=[dict(type='ineq',fun=lambda z:constraints(z)[0],
        jac=lambda z:constraints(z)[1])],options=dict(maxiter=160,ftol=1e-12))
    return NOISE_SCALE*basis@solved.x,dict(success=bool(solved.success),message=str(solved.message),
        surrogate_gain=float(objective(zero)[0]-objective(solved.x)[0]),
        minimum_predicted_normalized_margin=float(constraints(solved.x,full=True)[0].min()),
        coordinate_norm=float(np.linalg.norm(solved.x)), numerical_objective_scale=float(numerical_scale),
        roundoff_pruned_channels=int(np.sum(~live)),
        pruned_full_box_variation_bound=float(np.max(variation_bound[~live],initial=0.)),
        initial_surrogate_gain=float(objective(zero)[0]-objective(initial)[0]),
        all_original_surrogate_channels_rechecked=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reuse-conditional',action='store_true')
    parser.add_argument('--stratified-data',action='store_true')
    parser.add_argument('--correct-reward-inclusion',action='store_true')
    args = parser.parse_args()
    if args.correct_reward_inclusion:args.stratified_data=True
    if args.stratified_data:
        args.reuse_conditional = True
    torch.set_num_threads(1)
    started = time.perf_counter()
    source = BASE/('canonical_gate_band_stratified_head_credit_ppo_update_20260920' if args.stratified_data
                   else 'canonical_gate_band_head_counterfactual_ppo_collection_20260920')
    replay_file = BASE/'canonical_gate_band_shared_reference_transfer_joint_learning_20260920/affected_replay.npz'
    parent_file = BASE/'canonical_gate_band_reference_supported_joint_learning_20260920/research_candidate.pt'
    assert read(source/'receipt.json')['counterfactual_suffixes']==144
    assert sha(parent_file)=='ded2a93f85f3503a8c09bf59fd79e48c8f2cd187874f07789c53b8863c936229'
    md = torch.load(source/'initial_policy.pt',map_location='cpu',weights_only=False)
    assert not np.any(md['coefficients'])
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
    stage = register('canonical_gate_band_stratified_weighted_constraint_credit_ppo_20260920' if args.correct_reward_inclusion else
                     'canonical_gate_band_stratified_constraint_credit_ppo_20260920' if args.stratified_data else
                     'canonical_gate_band_conditional_sample_credit_ppo_20260920' if args.reuse_conditional
                     else 'canonical_gate_band_constraint_counterfactual_ppo_20260920',
        [__file__,source/'receipt.json',source/'on_policy_episodes.json',source/'ppo_update_audit.json',
         source/'shared_credit_review.json',source/'initial_policy.pt',replay_file,parent_file]+[BASE/n for n in (
             'head_counterfactual_rollouts.py','observable_memory_joint_policy.py','joint_teacher_state_policy.py',
             'saved_memory_action_identity.py','learn_early_memory_joint_policy.py','duration_joint_metrics.py',
             'shape_event_credit.py','learn_intrinsic_joint_recovery.py','learn_direct_observable_joint_policy.py')],
        dict(wall_s=240,new_stochastic_episodes=0,reused_counterfactual_suffixes=144,
             arms=['constraint_credit','constraint_and_unsampled_support_credit'],
             kl_bounds=[1e-5,1e-4],maximum_new_deterministic_policies=4,workers=4,
             same_actor_and_training_batch=True,coefficient_bounds=[-1.,1.],
             conditional_draws_per_head=3 if args.reuse_conditional else 1),
        ('Correct the scalar reward surrogate to use the SAME normalized inverse '
         'query-inclusion weights already used by constraint-credit channels. The '
         'previous stratified constraint stage weighted constraints only and is '
         'therefore an incomplete sampling-control comparison. Keep its saved batch, '
         'actor, conditional head draws, numerical solver and two KL limits; collect '
         'nothing new. Record the correction rather than rewriting prior results. '
         if args.correct_reward_inclusion else '')+
        ('Use the newly completed STRATIFIED query batch with exact recorded inverse '
         'inclusion probabilities; common-return and head-counterfactual Adam updates '
         'both failed actual guards on this batch. Keep the54-weight actor and203 '
         'constraint channels. Compare the same two support-protection arms and two '
         'KL limits against the completed uniform-query conditional-credit experiment. '
         'No new stochastic or conditional collection, and no more epochs. ' if args.stratified_data else '')+
        ('Use ALL THREE already measured conditional head-action draws, not only the '
         'original draw in the score update. Each draw uses a leave-one-out baseline '
         'from the other two independent draws. Head samples remain separate; never '
         'invent a trajectory with their three counterfactual actions combined. '
         'Same24 states,144 suffixes,mean policy,Gaussian law,global advantage scale, '
         'constraints,support equalities and two KL budgets as the previous update. '
         'This tests conditional-sampling credit, not extra data or extra epochs. '
         if args.reuse_conditional else '')+
        'Head-specific counterfactual PPO improves scalar loss but worsens105 lateral '
        'motion and uncollected shared-support paths14/305. Reuse every stochastic '
        'episode and144 complete suffixes; compute per-head counterfactual credit for '
        'ALL original constraint channels, not only a scalar weighted return. Optimize '
        'the same clipped PPO actor surrogate subject to per-path constraint-credit '
        'surrogate margins and conditional Gaussian KL measured on both sampled and '
        'saved parent states. Compare this constrained update with/without null-space '
        'protection of actor outputs on the unsampled supported paths14/205/305. '
        'Such equations operate only during learning; no runtime path/time switch. '
        'Sample-count weights account for the registered random subset; no claim '
        'of exact unbiased full-trajectory gradients or deterministic constraint '
        'guarantees from stochastic surrogates. Both arms start from identical zero '
        'weights and same on-policy batch; this is a matched constrained PPO update '
        'using saved interaction data, not new interaction collection or value-label '
        'fitting. Validate at two identical KL budgets on ALL seven affected complete '
        'routes and retain all original physical/shape/duration/curve/time/cruise '
        'guards. Preserve81 zero-support routes by induction.240s; failure triggers '
        'coverage/gradient analysis rather than more of the same epochs.')
    deadline = started+220.
    folder = stage/'state_joint'
    folder.mkdir()
    write_json(stage/'baseline.json',data)
    write_json(stage/'support_audit.json',support)
    torch.save(md,stage/'initial_policy.pt')
    pool, history, blocks = None, [], []
    try:
        metrics, identity, matrix, noise, advantage, credit, weights, parent_matrix, protected_basis = credit_problem(
            stage,md,data,selected,episodes,replay,deadline,args.reuse_conditional)
        scale = read(source/'ppo_update_audit.json')['global_advantage_scale']
        advantage = advantage/scale
        if args.correct_reward_inclusion:
            inclusion=weights/weights.mean()
            advantage=advantage*inclusion.reshape((len(weights),)+(1,)*(advantage.ndim-1))
            write_json(stage/'reward_inclusion_correction.json',dict(weights=inclusion.tolist(),
                same_inclusion_law_as_constraints=True,old_stage='canonical_gate_band_stratified_constraint_credit_ppo_20260920',
                old_constraint_weights_correct=True,old_reward_weights_missing=True,new_collection=0))
        retained = [next(o['baseline_shape'] for o in data['retained'] if o['path']==r['path']) for r in selected]
        def entry(records,u,tag):
            original,measured,learning = metrics(records)
            score = sum(teacher_score(r,b,o,barrier=False)+penalty(r)+
                (4*(r['shape']['reverse_lateral_total_mm']/.005)**2 if r['success'] else 1e6)
                for r,b,o in zip(records,selected,retained))
            return dict(tag=tag,coefficients=np.asarray(u).tolist(),records=records,score=float(score),
                measured_inequalities=original.tolist(),augmented_measured_inequalities=measured.tolist(),
                augmented_learning_inequalities=learning.tolist(),
                actual_rerun_paths=[] if tag=='reused_parent' else paths,
                exact_zero_support_paths=paths if tag=='reused_parent' else [])
        zero = entry(copy.deepcopy(selected),np.zeros(md['coefficients'].size),'reused_parent')
        history.append(zero)
        def feasible(c):
            return parameters_in_bounds(c['coefficients'],-1.,1.) and min(c['augmented_measured_inequalities'])>=-1e-8
        def best():
            return min(filter(feasible,history),key=lambda c:c['score'])
        md['reused_candidates'] = 1
        pool = mp.get_context('spawn').Pool(4,worker_initialize,(md,selected,folder,deadline))
        for name,basis in (('constraint_credit',np.eye(md['coefficients'].size)),
                           ('constraint_and_support',protected_basis)):
            for kl in (1e-5,1e-4):
                if time.perf_counter()>=deadline:
                    raise TimeoutError('Constrained PPO wall budget')
                u,report = solve_update(matrix,noise,advantage,credit,weights,
                    np.array(zero['augmented_learning_inequalities']),parent_matrix,basis,kl)
                block = dict(arm=name,kl_bound=kl,optimizer=report)
                if len(u)==md['coefficients'].size and report.get('surrogate_gain',0.)>1e-9:
                    assert parameters_in_bounds(u,-1.,1.)
                    tag = f'{name}_kl{kl}'
                    duplicate=next((c for c in history if np.array_equal(np.asarray(c['coefficients']),u)),None)
                    if duplicate is not None:
                        block.update(reused_identical_policy=duplicate['tag'],actual_score=duplicate['score'],
                            guard_min=min(duplicate['augmented_measured_inequalities']))
                        blocks.append(block);write_json(stage/'constraint_update_audit.json',blocks)
                        print('REUSED_IDENTICAL_PPO_POLICY',block,flush=True)
                        continue
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
                                 unchanged_action_paths=unchanged)
                    write_json(folder/'results.json',dict(candidates=history,best=best()['tag']))
                blocks.append(block)
                write_json(stage/'constraint_update_audit.json',blocks)
                print('CONSTRAINED_HEAD_PPO',block,flush=True)
        pool.close()
        pool.join()
        pool = None
        finish(stage,md,baseline,data,best(),zero,history,blocks,choice,parent_file,started)
        receipt(stage,dict(status='completed_constraint_counterfactual_ppo_update',
            new_stochastic_episodes=0,reused_counterfactual_suffixes=144,
            deterministic_policies=len(history)-1,selected=best()['tag'],
            research_acceptable=(stage/'research_candidate.pt').exists(),
            source_model_sha256=sha(parent_file),wall_s=time.perf_counter()-started,
            adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:
            pool.terminate()
            pool.join()
        receipt(stage,dict(status='failed_constraint_counterfactual_ppo_update',error=str(exc),
            deterministic_policies=max(0,len(history)-1),wall_s=time.perf_counter()-started,
            adopted=False,goal_complete=False))
        raise


if __name__ == '__main__':
    mp.freeze_support()
    main()
