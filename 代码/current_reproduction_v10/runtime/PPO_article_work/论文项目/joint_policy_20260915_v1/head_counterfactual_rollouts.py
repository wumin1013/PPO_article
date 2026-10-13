"""On-policy noisy episodes and single-head, common-future counterfactual returns."""
import copy
import time
import numpy as np
import torch
from observable_memory_joint_policy import ObservableMemoryJointPolicy
from joint_teacher_state_policy import OUTPUT_SCALE
from gate_band_state import environment_factory
from analyze_nonlinear_joint_response import measure
from learn_complete_joint_teachers import teacher_score
from duration_joint_metrics import penalty


WORK = {}
NOISE_SCALE = .02


def initialize(metadata, records, retained, folder, deadline):
    torch.set_num_threads(1)
    WORK.update(metadata=metadata,records={r['path']:r for r in records},
        retained={r['path']:r['baseline_shape'] for r in retained if 'baseline_shape' in r},
        folder=folder,deadline=deadline)


def exploration_sigma(metadata,design):
    exploration_design = design
    if metadata.get('exploration_geometry_only'):
        k = len(metadata['centers'])
        blocks = 1+len(metadata['memory_decays'])
        count = design.shape[-2]//blocks
        ids = np.concatenate([np.arange(k)+i*count for i in range(blocks)])
        exploration_design = design[...,ids,:]
    return NOISE_SCALE*np.sqrt(np.sum(exploration_design*exploration_design,axis=-2))


def action_distribution(policy, observation):
    mean = policy.act(observation,True)[0]
    f = policy.last_instant_features
    design = np.concatenate([f,*policy.last_memory_features],axis=0)*OUTPUT_SCALE
    sigma = exploration_sigma(policy.metadata,design)
    return mean, design, sigma


def cost(record, baseline, retained):
    if not record['success']:
        return 1e6
    return float(teacher_score(record,baseline,retained,barrier=False)+penalty(record)+
                 4*(record['shape']['reverse_lateral_total_mm']/.005)**2)


def episode(job):
    path, seed, query_decisions, *extra = job
    cached = extra[0] if extra else None
    query_weights = extra[1] if len(extra)>1 else {}
    md = WORK['metadata']
    baseline, retained = WORK['records'][path], WORK['retained'][path]
    with np.load(baseline['trace']) as z:
        reference = z['reference'].copy()
    policy = ObservableMemoryJointPolicy(copy.deepcopy(md))
    env = policy.bind(environment_factory(reference))
    noise = np.random.default_rng(seed).normal(size=(env.max_ticks//20+2,3))
    cached_arrays = None
    if cached is not None:
        with np.load(cached['record']['trace']) as z:
            cached_arrays = {k:z[k].copy() for k in ('rows','actions','observations','noise')}
        assert np.array_equal(noise,cached_arrays['noise'])
    alternative_rng = np.random.default_rng(seed+100000)
    chunks, actions, observations, snapshots, samples = [], [], [], {}, []
    actor_calls = ticks = 0
    tag = f'p{path}_seed{seed}'
    while not env.done and (cached is None or env.decisions<=max(query_decisions)):
        if time.perf_counter()>=WORK['deadline']:
            raise TimeoutError('Head-counterfactual episode budget')
        index = env.decisions
        x = env.observation()
        mean, design, sigma = action_distribution(policy,x)
        # Noise is independent across action heads and decisions; no AR likelihood shortcut.
        latent = mean.astype(float)+sigma*noise[index]
        if index in query_decisions and np.all(sigma>0.):
            snapshots[index] = copy.deepcopy((env,policy))
            samples.append(dict(decision=index,design=design,noise=noise[index].copy(),sigma=sigma,
                                observation=x.copy(),mean=mean.copy(),query_weight=query_weights.get(index)))
        _,_,_,info = env.step(latent)
        chunks.append(info['rows'])
        actions.append(info['held_action'])
        observations.append(x)
        actor_calls += 1
        ticks += len(info['rows'])
    full, held, observed = np.vstack(chunks), np.asarray(actions), np.asarray(observations)
    if cached is None:
        episode_file = WORK['folder']/(tag+'.npz')
        np.savez_compressed(episode_file,reference=reference,rows=full,actions=held,observations=observed,noise=noise)
        record = measure(reference,full,env)
        record.update(path=path,trace=str(episode_file),curved_path=True)
    else:
        for actual,key in ((full,'rows'),(held,'actions'),(observed,'observations')):
            assert np.array_equal(actual,cached_arrays[key][:len(actual)]), ('cached_prefix_identity',path,key)
        full,held,observed = [cached_arrays[k] for k in ('rows','actions','observations')]
        record = copy.deepcopy(cached['record'])
    observed_return = -cost(record,baseline,retained)
    controls = []
    cached_controls = {} if cached is None else {(r['decision'],r['head'],r['repeat']):r for r in cached['counterfactuals']}
    new_suffixes = reused_suffixes = 0
    for sample in samples:
        index = sample['decision']
        audit_env, audit_policy = copy.deepcopy(snapshots[index])
        audit_mean, audit_design, audit_sigma = action_distribution(audit_policy,audit_env.observation())
        assert np.array_equal(audit_mean,sample['mean'])
        assert np.array_equal(audit_design,sample['design'])
        _,_,_,audit_info = audit_env.step(audit_mean.astype(float)+audit_sigma*noise[index])
        assert np.array_equal(audit_info['rows'],full[index*20:index*20+len(audit_info['rows'])])
        actor_calls += 1
        ticks += len(audit_info['rows'])
        sample['head_returns'] = []
        for head in range(3):
            head_values = []
            for repeat in range(2):
                if (index,head,repeat) in cached_controls:
                    r = copy.deepcopy(cached_controls[index,head,repeat])
                    controls.append(r)
                    head_values.append(r['total_return'])
                    reused_suffixes += 1
                    continue
                branch_env, branch_policy = copy.deepcopy(snapshots[index])
                branch_chunks, branch_actions, branch_obs = [], [], []
                # A new draw in this head, independent of the original current draw.
                alternative = float(alternative_rng.normal())
                while not branch_env.done:
                    if time.perf_counter()>=WORK['deadline']:
                        raise TimeoutError('Head-counterfactual complete suffix budget')
                    j = branch_env.decisions
                    x = branch_env.observation()
                    mean, _, sigma = action_distribution(branch_policy,x)
                    epsilon = noise[j].copy()
                    if j==index:
                        epsilon[head] = alternative
                    latent = mean.astype(float)+sigma*epsilon
                    _,_,_,info = branch_env.step(latent)
                    branch_chunks.append(info['rows'])
                    branch_actions.append(info['held_action'])
                    branch_obs.append(x)
                    actor_calls += 1
                    ticks += len(info['rows'])
                rows = np.vstack([full[:index*20],*branch_chunks])
                branch_held = np.vstack([held[:index],branch_actions])
                branch_observed = np.vstack([observed[:index],branch_obs])
                file = WORK['folder']/f'{tag}_d{index}_h{head}_r{repeat}.npz'
                np.savez_compressed(file,reference=reference,rows=rows,actions=branch_held,
                                    observations=branch_observed,changed_decision=index,
                                    changed_head=head,replacement_standard_normal=alternative)
                r = measure(reference,rows,branch_env)
                r.update(path=path,trace=str(file),curved_path=True)
                value = -cost(r,baseline,retained)
                head_values.append(value)
                controls.append(dict(decision=index,head=head,repeat=repeat,replacement_noise=alternative,
                                     total_return=value,record=r))
                new_suffixes += 1
            sample['head_returns'].append(head_values)
        sample['counterfactual_advantage'] = observed_return-np.mean(sample['head_returns'],axis=1)
        sample['observed_return'] = observed_return
        sample['baseline_advantage'] = observed_return+cost(baseline,baseline,retained)
        for key in ('design','noise','sigma','observation','mean','counterfactual_advantage'):
            sample[key] = np.asarray(sample[key]).tolist()
    return dict(path=path,seed=seed,query_decisions=list(query_decisions),samples=samples,
                observed_return=observed_return,record=record,counterfactuals=controls,
                costs=dict(actor_calls=actor_calls,physical_ticks=ticks),
                snapshot_next20ms_bitwise=True,new_counterfactual_suffixes=new_suffixes,
                reused_counterfactual_suffixes=reused_suffixes,cached_prefix_bitwise=cached is not None,
                method='new_on_policy_interaction' if cached is None else 'new_conditional_suffixes_saved_on_policy_episodes')
