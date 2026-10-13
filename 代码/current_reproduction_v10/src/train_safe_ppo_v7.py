"""Registered seed101 smoke and bounded paired PPO pilot on training paths only."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from common import ROOT,sha,utc,write_json
from strong_policy_v4 import tensor_hash,STATIC_KEYS,load_strong_policy
from train_strong_fit_v4 import equivalent_loss
import safe_ppo_interface_v7 as interface
from metrics_exact_index_v5 import metrics

OUT=ROOT/'results/safe_ppo_v7'
PRIOR=ROOT/'results/recovery_training_v6'
SEED=101


def read(path):return json.loads(Path(path).read_text('utf-8'))
def arrays(path):
    with np.load(path) as data:return {key:data[key].copy() for key in data.files}


def register(phase,steps,wall):
    folder=OUT/phase
    if folder.exists():raise FileExistsError('Never overwrite registered PPO evidence')
    if phase=='pilot':assert read(OUT/'smoke/completion_validation.json')['status']=='passed'
    previous=read(PRIOR/'registration.json');source=read(ROOT/'results/geometric_resolution_check_v5_4/registration.json')['source_sha256'].copy()
    source.update({name:sha(ROOT/'src'/name) for name in ('safe_ppo_interface_v7.py','train_safe_ppo_v7.py','strong_ppo_v4.py','strong_policy_v4.py','train_strong_fit_v4.py')})
    paths=[row for row in previous['cases'] if row['id'] in (['train_box'] if phase=='smoke' else ['train_box','train_sine','train_bends','train_bezier','dense'])]
    assert len(paths)==(1 if phase=='smoke' else 5)
    steps=96 if phase=='smoke' else steps
    assert steps in (96,512,1024,2048)
    fits={arm:dict(path=str((PRIOR/('seed101_'+arm)/'final.pt').relative_to(ROOT)),
        sha256=sha(PRIOR/('seed101_'+arm)/'final.pt'),receipt_sha256=sha(PRIOR/('seed101_'+arm)/'receipt.json')) for arm in ('adaptive','fixed')}
    reg=dict(created_utc=utc(),version=interface.VERSION,phase=phase,seed=SEED,arms=['adaptive','fixed'],
        initialization='Each arm starts from its own seed101 fresh v6 supervised fit, fixed24000 steps; same data and rebuilt Wendland assets; shared historical teacher only upstream. This is not complete historical PPO reconstruction.',
        fits=fits,paths=paths,dt_s=.001,source_sha256=source,
        data_sha256=sha(PRIOR/'training_data/supervision.npz'),feature_assets_sha256=sha(PRIOR/'training_data/rebuilt_static_assets.pt'),
        data_registration_sha256=sha(PRIOR/'registration.json'),runtime_manifest_sha256=sha(ROOT/'runtime/source_manifest.json'),
        controller='Frozen v5_4 rules plus v5_4 physical-time resolution entry; request injection only. No concurrent efficiency candidate is used.',
        fixed_rule='Exactly2 policy coordinates. Fixed3mm is a core environment parameter, applies to raw/corrected/geometric actions and is logged at every actual command; geometric steering is recomputed against the3mm guide.',
        likelihood='Diagonal Gaussian density of the original sampled policy request. Corrected/geometric/stop outputs are environment transitions and never substitute for sampled actions in PPO. All converted-request transitions are retained.',
        distribution=dict(base_std=interface.STD.tolist(),std_feed='exp(clamp(log_std_feed,-8,-2)) + relu(abs(mean_feed)-1)',
            other_std='exp(clamp(log_std,-8,-2))',deterministic_mean='Unmodified fitted raw actor mean',straight_through=False,
            physical_mapping='Unchanged common clipped action decoder',noise_pairing='Always draw3 standard Normal coordinates, fixed arm consumes first2'),
        objective='Undiscounted episodic complete motion time with same geometric potential and explicit task-failure penalty',
        reward='-all_actual_elapsed_s + Phi(next_or_zero_if_terminated) - Phi(previous) - task_failure_penalty',
        failure_penalty='Physical task budget +1 +remaining_reference_mm/20; used for true failure/exception, never external wall truncation',
        gamma=1.,gae_lambda_per20ms=.95,gae='lambda**(actual_elapsed_s/0.02), grouped by path/episode; bootstrap nonterminal rollout and external truncations',
        decisions=steps,rollout=96 if phase=='smoke' else 128,epochs=2,minibatch=32,clip=.2,
        actor_lr=1e-5,critic_lr=1e-4,critic_loss_coefficient=.5,separate_gradient_norm_caps=.5,
        anchor=dict(coefficient=.1,rows_per_group=16,groups='original/dense x prior/recovery',equivalent_loss_scales=[.015,.15,.3]),
        paths_sampling='Fixed round-robin across persistent training environments; reset only terminated episodes',
        random_streams=dict(action=SEED+200000,critic=SEED+100000,minibatch=SEED+300000,anchor=SEED+400000),
        wall_limit_s_per_arm=wall,checkpoints='Initial, each completed rollout update, fixed final only',
        selection='Fixed final registered decision budget; no best-checkpoint or validation selection',
        evaluation_runs=previous['evaluation_runs'] if phase=='pilot' else [],evaluation_wall_limit_s_per_case=300.,
        no_validation_optimization=True,no_five_seed=True,no_holdout=True,no_long_original=True,
        model_version_new=True,all_failed_and_truncated_episodes_retained=True)
    write_json(folder/'registration.json',reg);verify(phase);print('SAFE_PPO_REGISTERED',phase,steps,sha(folder/'registration.json'),flush=True)


def verify(phase):
    reg=read(OUT/phase/'registration.json')
    assert all(sha(ROOT/'src'/name)==digest for name,digest in reg['source_sha256'].items())
    assert sha(PRIOR/'training_data/supervision.npz')==reg['data_sha256']
    assert sha(PRIOR/'training_data/rebuilt_static_assets.pt')==reg['feature_assets_sha256']
    assert sha(ROOT/'runtime/source_manifest.json')==reg['runtime_manifest_sha256']
    assert all(sha(ROOT/row['reference'])==row['reference_sha256'] for row in reg['paths'])
    assert all(sha(ROOT/value['path'])==value['sha256'] for value in reg['fits'].values())
    return reg


def save_policy(path,policy,reg,step,optimizers,extra):
    torch.save(dict(format='safe_request_ppo_v7',version=interface.VERSION,policy=policy.state_dict(),
        seed=SEED,arm=policy.arm,fixed_preview_mm=3.,steps_completed=step,
        registration_sha256=sha(OUT/reg['phase']/'registration.json'),
        action_rng_state=policy.action_generator.get_state(),trainable_tensor_sha256=tensor_hash(dict(policy.named_parameters())),
        actor_optimizer=optimizers[0].state_dict(),critic_optimizer=optimizers[1].state_dict(),**extra),path)


def optimize(policy,records,data,optimizers,rng,anchor_rng,reg,target,index,anchor_hash):
    ob=torch.tensor(np.asarray([r['observation'] for r in records]),dtype=torch.float32)
    ac=torch.tensor(np.asarray([r['sample'] for r in records]),dtype=torch.float32)
    old=torch.tensor([r['logp'] for r in records],dtype=torch.float32)
    means=torch.tensor(np.asarray([r['old_mean'] for r in records]),dtype=torch.float32)
    scales=torch.tensor(np.asarray([r['old_std'] for r in records]),dtype=torch.float32)
    old_replay=torch.distributions.Normal(means,scales).log_prob(ac).sum(-1)
    assert float(abs(old_replay-old).max())<2e-5
    advantages,returns=interface.physical_time_gae(records,reg['gamma'],reg['gae_lambda_per20ms'])
    adv=torch.tensor(advantages);adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8);ret=torch.tensor(returns)
    with torch.no_grad():pre_error=float(abs(policy.dist(ob).log_prob(ac).sum(-1)-old).max())
    assert pre_error<.002,'Batched vs sampled-action likelihood discrepancy'
    np.savez_compressed(target/f'batch_{index:04d}.npz',observations=ob.numpy(),sampled_actions=ac.numpy(),old_logp=old.numpy(),
        old_mean=means.numpy(),old_std=scales.numpy(),advantages=advantages,returns=returns,
        rewards=np.array([r['reward'] for r in records]),duration_s=np.array([r['elapsed_s'] for r in records]),
        sources=np.array([r['source'] for r in records]),episode=np.array([r['episode'] for r in records]))
    x,y,preview,joint,groups=data;stats=[];anchor_ids=[];started=time.perf_counter()
    for epoch in range(reg['epochs']):
        order=rng.permutation(len(records))
        for begin in range(0,len(records),reg['minibatch']):
            ids=order[begin:begin+reg['minibatch']]
            anchors=np.concatenate([g[anchor_rng.integers(len(g),size=reg['anchor']['rows_per_group'])] for g in groups])
            anchor_hash.update(anchors.tobytes());anchor_ids.append(anchors)
            dist=policy.dist(ob[ids]);ratio=(dist.log_prob(ac[ids]).sum(-1)-old[ids]).exp()
            surrogate=-torch.minimum(ratio*adv[ids],ratio.clamp(1-reg['clip'],1+reg['clip'])*adv[ids]).mean()
            predicted=policy.actor.from_features(x[anchors],preview[anchors],joint[anchors])[...,:policy.action_dim]
            anchor=equivalent_loss(predicted,y[anchors],torch.tensor(reg['anchor']['equivalent_loss_scales'][:policy.action_dim]))
            critic=reg['critic_loss_coefficient']*(policy.value(ob[ids])-ret[ids]).square().mean()
            loss=surrogate+reg['anchor']['coefficient']*anchor+critic;assert torch.isfinite(loss)
            for optimizer in optimizers:optimizer.zero_grad(set_to_none=True)
            loss.backward()
            ag=torch.nn.utils.clip_grad_norm_(list(policy.actor.parameters())+[policy.log_std],reg['separate_gradient_norm_caps'])
            cg=torch.nn.utils.clip_grad_norm_(policy.critic.parameters(),reg['separate_gradient_norm_caps'])
            assert torch.isfinite(ag) and torch.isfinite(cg)
            for optimizer in optimizers:optimizer.step()
            stats.append([float(surrogate.detach()),float(anchor.detach()),float(critic.detach()),float(ag),float(cg)])
    np.savez_compressed(target/f'anchors_{index:04d}.npz',indices=np.asarray(anchor_ids))
    with torch.no_grad():kl=float(torch.distributions.kl_divergence(torch.distributions.Normal(means,scales),policy.dist(ob)).sum(-1).mean())
    return dict(update=index,transitions=len(records),optimizer_steps=len(stats),ppo_sample_presentations=len(records)*reg['epochs'],
        anchor_sample_presentations=len(anchor_ids)*64,loss_means=np.mean(stats,axis=0).tolist(),kl=kl,
        sampled_logp_replay_max_error=float(abs(old_replay-old).max()),batched_logp_max_error=pre_error,wall_s=time.perf_counter()-started)


def train(phase,arm):
    reg=verify(phase);target=OUT/phase/arm;target.mkdir(parents=True,exist_ok=False)
    (target/'episodes').mkdir();(target/'batches').mkdir();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    started,cpu=time.perf_counter(),time.process_time();deadline=started+reg['wall_limit_s_per_arm']
    fit=ROOT/reg['fits'][arm]['path'];policy,_=interface.from_fit(fit);fitted,_=load_strong_policy(fit)
    raw=arrays(PRIOR/'training_data/supervision.npz');x=torch.tensor(raw['observations']);y=torch.tensor(raw['targets_three' if arm=='adaptive' else 'targets_two'])
    groups=[np.flatnonzero(raw['group']==i) for i in range(4)]
    with torch.no_grad():
        assert torch.equal(policy.mean(x[:128]),fitted.mean(x[:128]))
        features=[policy.actor.features(part) for part in x.split(256)]
        preview=torch.cat([f[0] for f in features]);joint=torch.cat([f[1] for f in features])
    data=(x,y,preview,joint,groups);optimizers=(torch.optim.Adam(list(policy.actor.parameters())+[policy.log_std],lr=reg['actor_lr']),torch.optim.Adam(policy.critic.parameters(),lr=reg['critic_lr']))
    rng=np.random.default_rng(SEED+300000);anchor_rng=np.random.default_rng(SEED+400000);anchor_hash=hashlib.sha256()
    snapshots=[];transitions=[];episodes=[];current={};buffer=[];updates=[];next_episode=0;completed=0;error=None
    def checkpoint(step,name):
        path=target/name;save_policy(path,policy,reg,step,optimizers,dict(anchor_rng_state=anchor_rng.bit_generator.state,minibatch_rng_state=rng.bit_generator.state))
        snapshots.append(dict(step=step,file=name,sha256=sha(path)));write_json(target/'checkpoints.json',snapshots)
        return path
    initial=checkpoint(0,'initial.pt')
    def start_episode(row):
        nonlocal next_episode
        number=next_episode;next_episode+=1;reference=np.loadtxt(ROOT/row['reference'],delimiter=',',skiprows=1)
        env=interface.SafeEpisode(reference,arm,fit,target/'episodes'/f'{number:04d}_{row["id"]}'/'controller',
            tolerance=row['evaluation_tolerance_mm'],deadline=deadline)
        return dict(row=row,id=number,env=env,records=[],actor_hash_at_creation=tensor_hash(dict(policy.actor.named_parameters())))
    def retain(item,reason):
        episode=item['env'];row=item['row'];folder=target/'episodes'/f'{item["id"]:04d}_{row["id"]}'
        controller=episode.close();records=item['records'];positions=np.asarray(episode.positions)
        rr=np.vstack(episode.rows) if episode.rows else np.empty((0,13))
        assert len(rr)+1==len(positions)
        np.savez_compressed(folder/'actual.npz',reference=episode.reference,positions=positions,rows=rr,
            held_preview_mm=np.array(episode.held_previews),observations=np.asarray([r['observation'] for r in records]),
            sampled_actions=np.asarray([r['sample'] for r in records]),rewards=np.array([r['reward'] for r in records]))
        write_json(folder/'transitions.json',records)
        last=records[-1] if records else None;terminated=bool(last and last['terminated']);success=bool(last and last['success'])
        audit=metrics(episode.reference,positions,interface.DT,row['evaluation_tolerance_mm'],declared_success=success,reason=reason)
        total_reward=sum(r['reward'] for r in records);elapsed=sum(r['elapsed_s'] for r in records)
        expected=(-elapsed+records[-1]['phi_after']-records[0]['phi_before']-sum(r['failure_penalty'] for r in records)) if records else 0.
        assert abs(total_reward-expected)<1e-9
        result=dict(path=row['id'],episode=item['id'],terminated=terminated,budget_truncated=not terminated,
            executor_success=success,reason=reason,decisions=len(records),elapsed_s=elapsed,total_reward=total_reward,
            telescoping_reward_error=abs(total_reward-expected),metrics=audit,controller=controller,
            actor_hash_at_creation=item['actor_hash_at_creation'],trace_sha256=sha(folder/'actual.npz'))
        write_json(folder/'result.json',result);episodes.append(result);write_json(target/'episodes.json',episodes)
    preparation=time.perf_counter()-started
    try:
        for row in reg['paths']:current[row['id']]=start_episode(row)
        for step in range(1,reg['decisions']+1):
            if time.perf_counter()>=deadline:raise TimeoutError('Registered total arm wall cap')
            row=reg['paths'][(step-1)%len(reg['paths'])];item=current[row['id']];env=item['env'];obs=env.env.observation()
            sampling_start=time.perf_counter();sample,logp,value=policy.act(obs)
            with torch.no_grad():dist=policy.dist(torch.tensor(obs));mean=dist.mean.numpy().copy();std=dist.scale.numpy().copy()
            next_obs,outcome=env.advance(obs,sample)
            with torch.no_grad():next_value=0. if outcome['terminated'] else float(policy.value(torch.tensor(next_obs)))
            record=dict(**outcome,step=step,path=row['id'],episode=item['id'],observation=obs.tolist(),sample=sample.tolist(),
                logp=logp,value=value,next_value=next_value,old_mean=mean.tolist(),old_std=std.tolist(),update_at_sampling=len(updates),
                sampling_and_execution_wall_s=time.perf_counter()-sampling_start,
                sampled_feed_mm_s=float(50*(1+np.clip(sample[1],-1,1))),mean_feed_mm_s=float(50*(1+np.clip(mean[1],-1,1))))
            item['records'].append(record);transitions.append(record);buffer.append(record);completed=step
            if env.ended:
                retain(item,record['reason']);current.pop(row['id'])
                if outcome['truncated']:raise TimeoutError(record['error'])
                current[row['id']]=start_episode(row)
            if len(buffer)==reg['rollout']:
                updates.append(optimize(policy,buffer,data,optimizers,rng,anchor_rng,reg,target/'batches',len(updates)+1,anchor_hash));buffer=[]
                checkpoint(step,f'checkpoint_{step:05d}.pt');write_json(target/'updates.json',updates);write_json(target/'transitions.json',transitions)
                print('SAFE_PPO_UPDATE',phase,arm,step,'episodes',len(episodes),'success',sum(e['metrics']['success'] for e in episodes),flush=True)
    except Exception as exc:error=type(exc).__name__+': '+str(exc)
    finally:
        for item in current.values():retain(item,'budget_truncated' if error is None else error)
        write_json(target/'transitions.json',transitions);write_json(target/'updates.json',updates)
        final=checkpoint(completed,'final.pt' if error is None and completed==reg['decisions'] else 'interrupted.pt')
        loaded,_=interface.load_policy(final)
        with torch.no_grad():assert torch.equal(loaded.mean(x[:128]),policy.mean(x[:128]))
        for key in STATIC_KEYS:assert torch.equal(fitted.state_dict()['actor.'+key],policy.state_dict()['actor.'+key])
        verify(phase)
        deltas=np.array([r['sampled_feed_mm_s']-r['mean_feed_mm_s'] for r in transitions]);samples=np.array([r['sample'][1] for r in transitions])
        receipt=dict(status='completed' if error is None and completed==reg['decisions'] else 'incomplete',error=error,
            phase=phase,arm=arm,steps=completed,updates=len(updates),model_sha256=sha(final),initial_sha256=sha(initial),
            registration_sha256=sha(OUT/phase/'registration.json'),anchor_stream_sha256=anchor_hash.hexdigest(),
            cost=dict(preparation_wall_s=preparation,total_wall_s=time.perf_counter()-started,total_cpu_s=time.process_time()-cpu,
                actual_physics_ticks=sum(r['tick_after']-r['tick_before'] for r in transitions),
                online_executor_wall_s=sum(r['executor_wall_s'] for r in transitions),optimization_wall_s=sum(r['wall_s'] for r in updates),
                ppo_sample_presentations=sum(r['ppo_sample_presentations'] for r in updates),anchor_sample_presentations=sum(r['anchor_sample_presentations'] for r in updates)),
            complete_successful_episodes=sum(e['metrics']['success'] for e in episodes),terminated_episodes=sum(e['terminated'] for e in episodes),
            budget_truncated_episodes=sum(e['budget_truncated'] for e in episodes),all_episodes_retained=True,
            sampled_physical_feed=dict(mean_absolute_delta_mm_s=float(abs(deltas).mean()) if len(deltas) else None,
                fraction_delta_at_least_1_mm_s=float((abs(deltas)>=1).mean()) if len(deltas) else None,
                saturation_fraction=float((abs(samples)>=1).mean()) if len(samples) else None,
                delta_quantiles_mm_s=np.quantile(deltas,[0,.25,.5,.75,1]).tolist() if len(deltas) else []),
            no_historical_online_teacher=True,no_final_test=True)
        write_json(target/'receipt.json',receipt);print(json.dumps(receipt),flush=True)


def audit(phase):
    reg=verify(phase);receipts={arm:read(OUT/phase/arm/'receipt.json') for arm in reg['arms']}
    assert all(r['status']=='completed' and r['steps']==reg['decisions'] for r in receipts.values())
    assert receipts['adaptive']['anchor_stream_sha256']==receipts['fixed']['anchor_stream_sha256']
    details=[]
    for arm in reg['arms']:
        folder=OUT/phase/arm;records=read(folder/'transitions.json');episodes=read(folder/'episodes.json')
        assert len(records)==reg['decisions'] and sum(e['decisions'] for e in episodes)==len(records)
        assert all(r['cached_request_calls']==1 and r['likelihood_uses_original_sample'] for r in records)
        assert all(len(r['sample'])==(3 if arm=='adaptive' else 2) for r in records)
        means=torch.tensor([r['old_mean'] for r in records]);std=torch.tensor([r['old_std'] for r in records]);samples=torch.tensor([r['sample'] for r in records])
        replay=torch.distributions.Normal(means,std).log_prob(samples).sum(-1)
        log_error=float(abs(replay-torch.tensor([r['logp'] for r in records])).max());assert log_error<2e-5
        for e in episodes:
            data=arrays(folder/'episodes'/f'{e["episode"]:04d}_{e["path"]}'/'actual.npz')
            assert len(data['rows'])+1==len(data['positions']) and np.array_equal(data['rows'][:,:2],data['positions'][1:])
            assert e['telescoping_reward_error']<1e-9
            if arm=='fixed':assert np.all(data['held_preview_mm']==3.)
        assert any(e['metrics']['success'] for e in episodes),'Smoke needs at least one completed full training episode per arm'
        details.append(dict(arm=arm,steps=len(records),logp_max_error=log_error,complete_successful_episodes=receipts[arm]['complete_successful_episodes'],
            physical_feed=receipts[arm]['sampled_physical_feed'],cost=receipts[arm]['cost']))
    result=dict(status='passed',created_utc=utc(),phase=phase,details=details,
        same_budget=True,same_anchor_row_stream=True,fixed_preview_always3mm=True,original_sample_likelihood_verified=True,
        complete_episode_time_and_failures_retained=True,all_truncations_bootstrapped=True,
        registration_sha256=sha(OUT/phase/'registration.json'))
    write_json(OUT/phase/'completion_validation.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['register','train','audit']);p.add_argument('--phase',choices=['smoke','pilot'],required=True);p.add_argument('--arm',choices=['adaptive','fixed']);p.add_argument('--steps',type=int,default=1024);p.add_argument('--wall',type=float,default=900.)
    a=p.parse_args();torch.set_num_threads(1)
    if a.mode=='register':register(a.phase,a.steps,a.wall)
    elif a.mode=='train':assert a.arm;train(a.phase,a.arm)
    else:audit(a.phase)
