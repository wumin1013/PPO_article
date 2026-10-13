"""Matched PPO actor updates: common return versus head-specific causal credit."""
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
from head_counterfactual_rollouts import initialize as initialize_collection, episode, NOISE_SCALE
from saved_memory_action_identity import SavedMemoryActionIdentity
from gate_band_state import GateBandRuntime
from joint_teacher_state_policy import OUTPUT_SCALE
from duration_joint_metrics import guards, penalty
from shape_event_credit import shape_event_learning_guards
from learn_intrinsic_joint_recovery import intrinsic_guards
from learn_complete_joint_teachers import teacher_score, parameters_in_bounds
from learn_early_memory_joint_policy import worker_initialize, worker_evaluate, metric_functions
from learn_direct_observable_joint_policy import finish
from study_io import BASE, register, receipt, write_json, sha
from search_complete_return import read


def ppo_updates(samples, scale, sample_weights=False):
    design = torch.tensor(np.array([s['design'] for s in samples]),dtype=torch.float64)
    noise = torch.tensor(np.array([s['noise'] for s in samples]),dtype=torch.float64)
    sigma = torch.tensor(np.array([s['sigma'] for s in samples]),dtype=torch.float64)
    common = np.repeat(np.array([s['baseline_advantage'] for s in samples])[:,None],3,axis=1)
    causal = np.array([s['counterfactual_advantage'] for s in samples])
    inclusion = np.array([s.get('query_weight',1.) for s in samples],dtype=float) if sample_weights else np.ones(len(samples))
    inclusion /= inclusion.mean()
    checkpoints, logs = [], []
    for name, values in (('common_return',common),('head_counterfactual',causal)):
        weights = torch.nn.Parameter(torch.zeros(design.shape[1:],dtype=torch.float64))
        optimizer = torch.optim.Adam([weights],lr=2e-4)
        advantage = torch.tensor(values*inclusion[:,None]/scale,dtype=torch.float64)
        assert not torch.any((design*weights).sum(1)/sigma)
        for epoch in range(1,7):
            shift = (design*weights).sum(1)/sigma
            logratio = -.5*((noise-shift).square()-noise.square())
            ratio = logratio.exp()
            loss = -torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage).sum(1).mean()
            exact_kl = .5*shift.square().sum(1).mean()
            if float(exact_kl.detach()) > .01:
                logs.append(dict(arm=name,epoch=epoch,stopped_for_kl=True,kl=float(exact_kl.detach())))
                break
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([weights],.5)
            optimizer.step()
            logs.append(dict(arm=name,epoch=epoch,loss=float(loss.detach()),kl_before=float(exact_kl.detach())))
            if epoch in (1,6):
                checkpoints.append(dict(tag=f'{name}_epoch{epoch}',arm=name,epoch=epoch,
                                        coefficients=weights.detach().numpy().ravel().tolist()))
    return checkpoints, logs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--complete-initialization',action='store_true')
    parser.add_argument('--stratified-credit',action='store_true')
    parser.add_argument('--resume-stratified-update',action='store_true')
    args = parser.parse_args()
    if args.resume_stratified_update:
        args.stratified_credit = True
    torch.set_num_threads(1)
    started = time.perf_counter()
    source = BASE/'canonical_gate_band_shared_reference_transfer_joint_learning_20260920'
    parent_file = BASE/'canonical_gate_band_reference_supported_joint_learning_20260920/research_candidate.pt'
    assert sha(parent_file)=='ded2a93f85f3503a8c09bf59fd79e48c8f2cd187874f07789c53b8863c936229'
    md = torch.load(source/'initial_policy.pt',map_location='cpu',weights_only=False)
    md['residual_output_precision'] = 'float64'
    md['learning_method'] = 'matched_on_policy_head_counterfactual_ppo'
    data = read(source/'baseline.json')
    baseline = data['baseline']['records']
    support = read(source/'support_audit.json')
    choice = support['selected']
    paths = choice['paths']
    selected = [r for r in baseline if r['path'] in paths]
    with np.load(source/'affected_replay.npz') as z:
        replay = {k:z[k].copy() for k in z.files}
    identity = SavedMemoryActionIdentity(md,replay,GateBandRuntime.decode_action)
    assert all(identity.unchanged(p,np.zeros(md['coefficients'].size)) for p in paths)
    training_paths = [104,105,204,304]
    old_source = BASE/'canonical_gate_band_head_counterfactual_ppo_collection_20260920'
    old_episodes = (read(old_source/'on_policy_episodes.json')['episodes'] if args.stratified_credit else [])
    jobs = []
    sampling_audit = []
    for p in training_paths:
        f, memory, _, _ = identity.paths[p]
        nonzero = np.flatnonzero(np.any(f!=0.,axis=(1,2)) | np.any(memory!=0.,axis=(1,2,3)))
        for replicate in range(2):
            seed = 20260920+p*10+replicate
            rng = np.random.default_rng(seed+300000)
            if args.stratified_credit:
                design = np.concatenate([f,memory[:,0],memory[:,1]],axis=1)*OUTPUT_SCALE
                energy = np.sum(design*design,axis=(1,2))
                ordered = nonzero[np.argsort(energy[nonzero])[::-1]]
                cumulative = np.cumsum(energy[ordered])/np.sum(energy[ordered])
                first = min(max(int(np.searchsorted(cumulative,1./3.))+1,1),len(ordered)-2)
                second = min(max(int(np.searchsorted(cumulative,2./3.))+1,first+1),len(ordered)-1)
                strata = np.split(ordered,[first,second])
                queries = sorted(int(rng.choice(s)) for s in strata)
                weights = {q:next(len(s) for s in strata if q in s) for q in queries}
                cached = next(e for e in old_episodes if e['path']==p and e['seed']==seed)
                jobs.append((p,seed,queries,cached,weights))
                sampling_audit.append(dict(path=p,seed=seed,strata=[s.tolist() for s in strata],
                    selected=queries,inverse_inclusion_probability=weights,
                    parent_design_energy_fraction=float(energy[queries].sum()/energy.sum())))
            else:
                queries = sorted(rng.choice(nonzero,size=min(3,len(nonzero)),replace=False).tolist())
                jobs.append((p,seed,queries))
    if args.complete_initialization:
        failed = read(BASE/'canonical_gate_band_head_counterfactual_ppo_20260920/receipt.json')
        assert failed['actual_new_episodes']==0 and failed['ppo_optimizer_steps']==0
    collected_stage = BASE/'canonical_gate_band_stratified_head_credit_ppo_20260920'
    if args.resume_stratified_update:
        previous_receipt = read(collected_stage/'receipt.json')
        assert previous_receipt['status'].startswith('failed') and previous_receipt['completed_episodes']==8
        assert previous_receipt['deterministic_checkpoints']==0
        assert not (collected_stage/'ppo_update_audit.json').exists()
    additional_sources = ([collected_stage/'receipt.json',collected_stage/'on_policy_episodes.json',
        collected_stage/'sampling_design.json'] if args.resume_stratified_update else [])
    stage = register('canonical_gate_band_stratified_head_credit_ppo_update_20260920' if args.resume_stratified_update else
                     'canonical_gate_band_stratified_head_credit_ppo_20260920' if args.stratified_credit else
                     'canonical_gate_band_head_counterfactual_ppo_collection_20260920' if args.complete_initialization
                     else 'canonical_gate_band_head_counterfactual_ppo_20260920',
        [__file__,parent_file,source/'initial_policy.pt',source/'baseline.json',source/'support_audit.json',
         source/'affected_replay.npz',BASE/'canonical_gate_band_interaction_return_credit_learning_20260920/known_feasible_credit_audit.json',
         BASE/'common_future_action_credit_learning_20260917/receipt.json']+additional_sources+[BASE/n for n in (
             'head_counterfactual_rollouts.py','observable_memory_joint_policy.py','joint_teacher_state_policy.py',
             'saved_memory_action_identity.py','learn_early_memory_joint_policy.py','duration_joint_metrics.py',
             'shape_event_credit.py','learn_intrinsic_joint_recovery.py','learn_direct_observable_joint_policy.py')],
        dict(wall_s=600,workers=4,training_paths=training_paths,episodes=8,queries_per_episode=3,
             independent_head_replacements=2,maximum_counterfactual_suffixes=144,noise_coefficient_scale=NOISE_SCALE,
             jobs=[dict(path=j[0],seed=j[1],queries=j[2]) for j in jobs],
             new_stochastic_episodes=0 if args.stratified_credit else 8,
             actor_epochs=6,learning_rate=2e-4,ppo_clip=.2,kl_stop=.01,gradient_clip=.5,
             checkpoints=[1,6],deterministic_validation_paths=paths),
        ('RESUME: all conditional collection finished previously; a weight dtype error '
         'occurred before any optimizer update. Correct dtype and reuse the entire '
         'saved batch. No stochastic episode or conditional suffix is repeated. ' if args.resume_stratified_update else '')+
        ('CHANGE FOR THIS STAGE: reuse the same8 saved stochastic episodes and the same '
         'Gaussian behavior actor. Reconstruct only needed prefixes with bitwise checks. '
         'Collect at most144 NEW conditional suffixes at THREE queries per episode, '
         'one uniform draw from each of three parent-design-energy strata. Strata are '
         'defined BEFORE looking at stochastic returns, from existing deterministic '
         'parent states. Weight each query by its stratum size (inverse inclusion '
         'probability). The old uniform design has constant weights. Preserve the '
         'original advantage scale,54 weights, Adam, learning rate, epochs, KL and '
         'both common/head-counterfactual arms. Reuse overlapping conditional samples '
         'without rerunning. Design energy is not asserted to equal actual credit '
         'importance. This tests credit sampling coverage, not added episode count '
         'or training epochs. No deployed query rule. The original baseline control '
         'results are reused, not re-evaluated. ' if args.stratified_credit else '')+
        'Original protocol context: Direct PPO actor mechanism comparison after approximate return models wrongly '
        'excluded measured feasible points. Collect8 NEW on-policy complete stochastic '
        'episodes from the same shared54-weight state/history actor. Sample independent '
        'Gaussian heads with observable-feature-scaled variance. Query decisions are a '
        'fixed random subset of parent-supported training decisions, common to both arms; '
        'no claim of an unbiased whole-trajectory gradient from this selected subset. '
        'For each query replace only ONE current head by two independent conditional '
        'draws, leave other heads fixed, and rerun the policy on real future states with '
        'common independent future innovations until complete stop or true failure. '
        'Unlike the historical common-future experiment, do not replace all three '
        'heads together and do not broadcast one joint counterfactual advantage. '
        'Compare common full-episode return minus fixed parent-path return against '
        'per-head observed return minus its two counterfactual complete returns. '
        'Same actor, trajectories, query subset, global scaling, Adam, per-head clipped '
        'likelihood objective, KL limit and epoch checkpoints. No fitted value critic '
        'or response surrogate; no feed labels. This IS new PPO interaction collection '
        'and clipped actor training, with a conditional counterfactual baseline arm. '
        'Validate BOTH checkpoints of BOTH arms on all seven affected paths; the other '
        '81 routes are protected by exact zero-support induction. Require all original '
        'physical/shape/cruise/time/duration/curvature guards before research promotion. '
        'No weaker physical limits, deployed query switches, final adoption, estimator '
        'bias claim, sealed tests or generalization claim.600s; failures lead to gradient '
        'and representation review, never unchanged extension.')
    deadline = started+580.
    folder = stage/'state_joint'
    folder.mkdir()
    raw_folder = stage/'episodes'
    raw_folder.mkdir()
    torch.save(md,stage/'initial_policy.pt')
    write_json(stage/'baseline.json',data)
    write_json(stage/'support_audit.json',support)
    if args.stratified_credit:
        write_json(stage/'sampling_design.json',dict(rows=sampling_audit,
            source_receipt_sha256=sha(old_source/'receipt.json'),
            source_episode_file_sha256=sha(old_source/'on_policy_episodes.json'),
            stochastic_episode_reuse=True,queries_per_episode_unchanged=3))
    write_json(stage/'zero_policy_identity.json',dict(all_supported_decoded_actions_exact=True,
        parent_sha256=sha(parent_file),snapshot_arithmetic='float64 residual with exact zero extension'))
    _,old_guards = metric_functions(data,selected,stage,md['parent'],deadline)
    retained = [next(o['baseline_shape'] for o in data['retained'] if o['path']==r['path']) for r in selected]
    def entry(records,u,tag):
        original = np.concatenate([old_guards(records)]+[guards(r,b) for r,b in zip(records,selected)])
        extra = intrinsic_guards(records,selected)
        score = sum(teacher_score(r,b,o,barrier=False)+penalty(r)+
            (4*(r['shape']['reverse_lateral_total_mm']/.005)**2 if r['success'] else 1e6)
            for r,b,o in zip(records,selected,retained))
        return dict(tag=tag,coefficients=np.asarray(u).tolist(),records=records,score=float(score),
            measured_inequalities=original.tolist(),augmented_measured_inequalities=np.r_[original,extra].tolist(),
            augmented_learning_inequalities=np.r_[shape_event_learning_guards(records,selected,original),extra].tolist(),
            actual_rerun_paths=[] if tag=='reused_parent' else paths,
            exact_zero_support_paths=paths if tag=='reused_parent' else [])
    zero = entry(copy.deepcopy(selected),np.zeros(md['coefficients'].size),'reused_parent')
    history = [zero]
    md['reused_candidates'] = 1
    def feasible(c):
        return parameters_in_bounds(c['coefficients'],-1.,1.) and min(c['augmented_measured_inequalities'])>=-1e-8
    def best():
        return min(filter(feasible,history),key=lambda c:c['score'])
    pool, episodes, updates = None, [], []
    try:
        # Validate initializer inputs here: a worker initializer failure otherwise respawns indefinitely.
        initialize_collection(md,selected,data['retained'],raw_folder,deadline)
        if args.resume_stratified_update:
            episodes = read(collected_stage/'on_policy_episodes.json')['episodes']
            write_json(stage/'on_policy_episodes.json',dict(episodes=episodes))
        else:
            pool = mp.get_context('spawn').Pool(4,initialize_collection,(md,selected,data['retained'],raw_folder,deadline))
            for row in pool.imap_unordered(episode,jobs):
                episodes.append(row)
                write_json(stage/'on_policy_episodes.json',dict(episodes=episodes))
                print('HEAD_CF_EPISODE',row['path'],row['seed'],row['record']['success'],
                      len(row['samples']),len(row['counterfactuals']),flush=True)
            pool.close()
            pool.join()
            pool = None
        episodes.sort(key=lambda r:(r['path'],r['seed']))
        samples = [dict(s,path=e['path'],seed=e['seed']) for e in episodes for s in e['samples']]
        assert len(samples)>=12 and len({s['path'] for s in samples})>=2, 'Insufficient actual query support'
        scale = (read(old_source/'ppo_update_audit.json')['global_advantage_scale'] if args.stratified_credit
                 else max(float(np.std([s['baseline_advantage'] for s in samples])),1e-6))
        checkpoints, updates = ppo_updates(samples,scale,sample_weights=args.stratified_credit)
        write_json(stage/'ppo_update_audit.json',dict(updates=updates,checkpoints=checkpoints,
            global_advantage_scale=scale,same_scale_both_arms=True,sample_count=len(samples),
            critic_updates=0,head_probability_initial_ratio_exactly_one=True,
            complete_future_common_noise=True,all_snapshot_checks_bitwise=True))
        pool = mp.get_context('spawn').Pool(4,worker_initialize,(md,selected,folder,deadline))
        for checkpoint in checkpoints:
            tag, u = checkpoint['tag'],np.array(checkpoint['coefficients'])
            actor = copy.deepcopy(md)
            actor['coefficients'] = u.reshape(-1,3)
            torch.save(actor,stage/(tag+'.pt'))
            records = pool.map(worker_evaluate,[(p,u,tag) for p in paths])
            by = {r['path']:r for r in records}
            c = entry([by[r['path']] for r in selected],u,tag)
            history.append(c)
            write_json(folder/'results.json',dict(candidates=history,best=best()['tag']))
            print('HEAD_CF_PPO_CHECKPOINT',tag,c['score'],min(c['augmented_measured_inequalities']),flush=True)
        pool.close()
        pool.join()
        pool = None
        finish(stage,md,baseline,data,best(),zero,history,updates,choice,parent_file,started)
        receipt(stage,dict(status='completed_matched_head_counterfactual_ppo',episodes=len(episodes),
            new_stochastic_episodes=0 if args.stratified_credit else len(episodes),
            new_counterfactual_suffixes=0 if args.resume_stratified_update else sum(e['new_counterfactual_suffixes'] for e in episodes),
            reused_counterfactual_suffixes=sum(len(e['counterfactuals']) if args.resume_stratified_update else e['reused_counterfactual_suffixes'] for e in episodes),
            counterfactual_suffixes=sum(len(e['counterfactuals']) for e in episodes),
            actor_optimizer_steps=sum('loss' in u for u in updates),critic_optimizer_steps=0,
            deterministic_checkpoints=len(history)-1,selected=best()['tag'],
            research_acceptable=(stage/'research_candidate.pt').exists(),
            source_model_sha256=sha(parent_file),wall_s=time.perf_counter()-started,
            adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:
            pool.terminate()
            pool.join()
        receipt(stage,dict(status='failed_matched_head_counterfactual_ppo',error=str(exc),
            completed_episodes=len(episodes),deterministic_checkpoints=len(history)-1,
            wall_s=time.perf_counter()-started,adopted=False,goal_complete=False))
        raise


if __name__ == '__main__':
    mp.freeze_support()
    main()
