"""Exact saved PPO replay: locate updates that turn fixed entry targets outward."""
import copy
import argparse
import math
import time
import numpy as np
import torch
from conditional_feedback_policy import ConditionalLocalGeometryPolicy
from train_bounded_shape_ppo import update
from train_pilot import save_torch
from probe_coordinated_suffix import reconstruct
from search_complete_return import read
from study_io import BASE, register, receipt, write_json, sha


def flat_gradient(value, parameters):
    return torch.cat([x.reshape(-1) for x in torch.autograd.grad(value,parameters,retain_graph=True)])


def query_targets(policy, queries):
    obs=torch.tensor(np.asarray([q['observation'] for q in queries]))
    latent=policy.distribution(obs).mean
    bounded=torch.tanh(latent);angles=[]
    for i,q in enumerate(queries):
        env=q['env'];d=6.4+5.6*bounded[i,2];s,_=env.progress()
        arc=torch.clamp(d+s,0.,float(env.corridor.arcs[-1]))
        segment=min(np.searchsorted(env.corridor.arcs,float(arc.detach()),side='right')-1,len(env.corridor.lengths)-1)
        target=arc.new_tensor(env.corridor.starts[segment])+(arc-float(env.corridor.arcs[segment]))*arc.new_tensor(env.corridor.tangents[segment])
        relative=target-arc.new_tensor(env.position)
        combined=torch.atan2(relative[1],relative[0])+math.pi/2*bounded[i,0]-q['incoming_heading']
        angles.append(q['sign']*torch.atan2(torch.sin(combined),torch.cos(combined)))
    return torch.stack(angles),latent


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume-partial',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1)
    source=BASE/'local_geometry_continuous_reward_ppo_20260916'
    initial=BASE/'matched_endpoint_representation_learning_20260916/mask_endpoint_bearing_50.pt'
    cases=BASE/'preswing_mechanism_chain_20260916'
    stage=BASE/'preswing_learning_origin_replay_20260916'
    if not args.resume_partial:
        stage=register('preswing_learning_origin_replay_20260916',
        [__file__,initial,source/'policy_6713.pt',source/'results.json',source/'paired_groups.json',
         source/'episode_results.json',cases/'receipt.json',cases/'results.json',
         BASE/'conditional_feedback_policy.py',BASE/'train_bounded_shape_ppo.py',
         BASE/'train_local_geometry_ppo.py',BASE/'bounded_shape_return.py'],
        dict(replayed_actor_calls=6713,replayed_updates=6,new_physics_ticks=0,
             new_training_decisions=0,wall_s=150,query_paths=list(range(6,14))),
        'Replay the original first6713 saved stochastic actor samples and six PPO updates '
        'bitwise from the recorded seed and initial weights. No new environment training. '
        'Track signed combined LOS+steering target at the same eight current preswing '
        'states across updates; this measures learned-function change at fixed states, '
        'not each historical policy own rollout shape. Decompose pre-update PPO score '
        'gradient by original training path and dot it with the gradient of the mean '
        'signed combined target. This is an SGD directional diagnostic, not exact Adam '
        'attribution. Save exact optimizer/RNG/batch episode references before each update '
        'for controlled optimizer-loss interventions afterward. Verify all original '
        'update statistics and final6713 tensors bitwise. Since the paired actor uses '
        'whole-episode leave-one-out advantages, critic error is not a direct actor '
        'advantage source in this run. Do not infer the entire historical cause from '
        'one gradient sign; identify reproducible transition updates for causal testing.')
    started=time.perf_counter();costs=dict(saved_prefix_geometry_ticks=0,replayed_actor_calls=0,replayed_updates=0)
    queries=[]
    for b in read(cases/'results.json')['baselines']:
        index=b['path'];anchor=b['decomposition']['anchor']
        with np.load(BASE/'exit_local_credit_parameter_learning_20260916/training_1'/f'{index}.npz') as z:
            ref,rows,actions=[z[k] for k in ('reference','rows','actions')]
        env=reconstruct(ref,rows,actions,anchor*20,costs);u=ref[1]-ref[0]
        queries.append(dict(path=index,anchor=anchor,env=env,observation=env.observation(),
            sign=b['decomposition']['sign'],incoming_heading=math.atan2(u[1],u[0])))
    torch.manual_seed(9161043)
    p=ConditionalLocalGeometryPolicy(72,mask_endpoint=True)
    p.load_state_dict(torch.load(initial,map_location='cpu',weights_only=False)['policy'])
    torch.manual_seed(9161043);p.reset_context()
    optimizer={'actor':torch.optim.Adam(p.actor_parameters(),lr=3e-5),
               'critic':torch.optim.Adam(p.critic.parameters(),lr=.001)}
    if args.resume_partial:
        assert not (stage/'receipt.json').exists() and not (stage/'results.json').exists()
        recovered=torch.load(stage/'before_update_1.pt',map_location='cpu',weights_only=False)
        p.load_state_dict(recovered['policy'])
        for k,v in optimizer.items():v.load_state_dict(recovered['optimizer'][k])
        torch.set_rng_state(recovered['torch_rng'])
        costs.update(prior_failed_attempt_actor_calls=1194,prior_failed_attempt_updates=1,
                     already_sampled_prefix_reused=1194)
        write_json(stage/'failure_recovery.json',dict(
            reason='First update exactly replayed, then result serialization rejected a numpy int64 path-count field.',
            recovery='Reuse pre-update1 policy/optimizer/RNG; do not repeat its1194 stochastic samples. Recompute its one optimizer update because post-update state was not saved.',
            repaired_source_sha256=sha(__file__),prior_actor_calls=1194,prior_optimizer_updates=1,new_physics_ticks=0))
    expected=read(source/'results.json')['updates']
    episodes=read(source/'episode_results.json')['episodes']
    batch=[];group=[];batch_meta=[];reports=[];count=0
    with torch.no_grad():initial_targets=query_targets(p,queries)[0].numpy()
    for meta in episodes:
        with np.load(source/'episodes'/meta['trace']) as z:
            data={k:z[k].copy() for k in ('observation','latent','previous_observation','previous_latent','context_live','old_log','reward','value','terminal')}
        p.reset_context()
        if not args.resume_partial or count>=1194:
            for j,obs in enumerate(data['observation']):
                a,logp,value=p.act(obs)
                assert np.array_equal(a,data['latent'][j]) and logp==data['old_log'][j] and value==data['value'][j]
                costs['replayed_actor_calls']+=1
        count+=len(data['reward'])
        batch.append(data);batch_meta.append(meta);group.append(data)
        if len(group)==4:
            total=np.array([d['reward'].sum() for d in group])
            baseline=(total.sum()-total)/3;scale=max(float(total.std()),.02)
            for d,a in zip(group,total-baseline):d['paired_advantage']=np.full(len(d['reward']),a/scale)
            group=[]
        if count!=expected[len(reports)]['decisions']:continue
        assert not group
        assert time.perf_counter()-started<150
        checkpoint=f'before_update_{len(reports)+1}.pt'
        save_torch(stage/checkpoint,dict(policy=copy.deepcopy(p.state_dict()),
            optimizer=copy.deepcopy({k:v.state_dict() for k,v in optimizer.items()}),
            torch_rng=torch.get_rng_state(),episode_numbers=[m['episode'] for m in batch_meta],
            query_observation=np.asarray([q['observation'] for q in queries]),
            query_paths=[q['path'] for q in queries],query_anchors=[q['anchor'] for q in queries]))
        tensor={k:torch.as_tensor(np.concatenate([d[k] for d in batch]),dtype=torch.bool if k=='context_live' else torch.float32)
                for k in ('observation','latent','previous_observation','previous_latent','context_live')}
        adv=torch.tensor(np.concatenate([d['paired_advantage'] for d in batch]).astype(np.float32))
        adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
        target,_=query_targets(p,queries);parameters=p.actor_parameters()
        qgrad=flat_gradient(target.mean(),parameters)
        logp=p.evaluate_conditioned(tensor['observation'],tensor['latent'],tensor['previous_observation'],tensor['previous_latent'],tensor['context_live'])[0]
        ids=np.concatenate([np.full(len(d['reward']),m['path']) for m,d in zip(batch_meta,batch)])
        contributions=[]
        for path in sorted(set(ids)):
            mask=torch.tensor(ids==path)
            g=flat_gradient((logp*adv*mask).mean(),parameters)
            contributions.append(dict(path=int(path),decisions=int(mask.sum()),
                score_gradient_norm=float(g.norm()),target_directional_derivative=float(g@qgrad),
                successes=sum(m['success'] for m in batch_meta if m['path']==path),
                episodes=int(sum(m['path']==path for m in batch_meta)),
                returns=[float(d['reward'].sum()) for m,d in zip(batch_meta,batch) if m['path']==path]))
        before=target.detach().numpy();stats=update(p,optimizer,batch,paired_advantage=True)
        assert all(stats[k]==expected[len(reports)][k] for k in stats)
        with torch.no_grad():after=query_targets(p,queries)[0].numpy()
        reports.append(dict(update=len(reports)+1,decisions=count,before_signed_target_deg=np.rad2deg(before).tolist(),
            after_signed_target_deg=np.rad2deg(after).tolist(),exact_update_statistics=True,
            mean_target_change_deg=float(np.rad2deg(after-before).mean()),path_score_gradients=contributions,
            replay_state=checkpoint))
        costs['replayed_updates']=len(reports)
        print('ORIGIN_REPLAY',len(reports),count,'mean_delta_deg',reports[-1]['mean_target_change_deg'],
              'targets',np.round(np.rad2deg(after),4).tolist(),flush=True)
        write_json(stage/'results.json',dict(initial_targets_deg=np.rad2deg(initial_targets).tolist(),
            query_paths=[q['path'] for q in queries],updates=reports,costs=costs,historical_cause_fully_established=False))
        batch=[];batch_meta=[]
        if count==6713:break
    saved=torch.load(source/'policy_6713.pt',map_location='cpu',weights_only=False)['policy']
    assert count==6713 and len(reports)==6
    assert all(torch.equal(value,saved[key]) for key,value in p.state_dict().items())
    receipt(stage,dict(status='completed_exact_preswing_origin_replay',costs=costs,
        original_checkpoint_bitwise_equal=True,new_physics_ticks=0,new_training_decisions=0,
        wall_s=time.perf_counter()-started))


if __name__=='__main__':main()
