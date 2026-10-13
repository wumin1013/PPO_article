"""Matched on-policy refinement with or without the discrete onset-label cost."""
import argparse
import math
import time
import traceback
import numpy as np
import torch
from conditional_feedback_policy import ConditionalLocalGeometryPolicy,RHO
from bounded_shape_return import terminal_shape_cost
from train_bounded_shape_ppo import update
from train_pilot import evaluate,save_torch
from joint_runtime_v5 import environment_factory
from validate_actions import difference_audit
from probe_constrained_parameter_step import summarize
from review_endpoint_representation import onset
from complete_corner_metrics import complete_corner_metrics
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=['discrete','continuous'],required=True)
    parser.add_argument('--side-coverage',choices=['incoming','two_edges']);args=parser.parse_args()
    if args.side_coverage and args.variant!='continuous':parser.error('Side-coverage comparison uses continuous reward in both arms')
    discrete=args.variant=='discrete';seed=9161087 if args.side_coverage else 9161043
    training_budget=18000 if args.side_coverage else 24000
    wall_budget=500 if args.side_coverage else 600
    two_edges=args.side_coverage=='two_edges'
    torch.set_num_threads(1);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    initial=BASE/'matched_endpoint_representation_learning_20260916/mask_endpoint_bearing_50.pt'
    if args.side_coverage:initial=BASE/'local_geometry_continuous_reward_ppo_20260916/policy_18645.pt'
    paths=read(BASE/'learning_preparation_v4/paths.json');indices=list(range(len(paths['training'])))
    stage_name=f'{args.side_coverage}_side_coverage_ppo_20260916' if args.side_coverage else f'local_geometry_{args.variant}_reward_ppo_20260916'
    stage=register(stage_name,
        [__file__,initial,initial.parent/'receipt.json',BASE/'discrete_first_label_reward_audit_20260916/receipt.json',
         BASE/'conditional_feedback_policy.py',BASE/'local_geometry_feedback_policy.py',BASE/'frame_feedback_policy.py',
         BASE/'bounded_shape_return.py',BASE/'train_bounded_shape_ppo.py',BASE/'joint_policy.py',BASE/'joint_runtime.py',
         BASE/'joint_runtime_v2.py',BASE/'joint_runtime_v3.py',BASE/'joint_runtime_v5.py',BASE/'terminal_candidate_runtime_v1.py',
         BASE/'geometry_spec.py',BASE/'learning_preparation_v4/paths.json',BASE/'complete_corner_metrics.py']+
         ([BASE/'complete_two_edge_shape_audit_20260916/receipt.json'] if args.side_coverage else []),
        dict(training_decisions=training_budget,evaluation_decisions=14000,physics_ticks=1000000,wall_s=wall_budget,seed=seed,
             training_pool=indices,group_size=4,batch_minimum_decisions=1024,actor_lr=.00003,critic_lr=.001,
             epochs=6,minibatch=256,clip=.2,kl_stop=.03,ar_rho=RHO,latent_std=list(ConditionalLocalGeometryPolicy.exploration_std),
             checkpoint_thresholds=list(range(6000,training_budget+1,6000)),discrete_first_turn_label=discrete,
             side_coverage=args.side_coverage or 'incoming',
             advantage='leave-one-out complete returns / max(same-path return std,0.02s); unchanged batch normalization',gamma=1.,gae_lambda=1.),
        ('Matched single-factor side-coverage comparison: same retained18645 actor and critic, fresh identical '
        'optimizers, seed, original64 training paths, action representation/exploration, original local '
        'energy/reversal windows and allowances, and18000-decision/500s budgets. The only difference is '
        'the existing positional term: original inbound-half-plane excursion versus maximum outside either '
        'edge half-plane over the full isolated-corner XY. Keep weights,0.05mm allowance, bounded terminal '
        'cost and full failure charge. This is geometric reward coverage, not a deployed template. '
        'Retain only checkpoints with12/12 complete development paths, no new meaningful entry excursion '
        'and substantial true exit-shape improvement (>=50% median exit-excursion reduction versus the '
        'unchanged initial model); inspect full training pool before broader use. Any retained model still '
        'requires full tangency/shape/cruise and fair validation. Stop at budget and analyze; do not extend unchanged. '
        if args.side_coverage else
        'Matched reward-mechanism experiment: both arms share the endpoint-masked50-update learned actor, '
        'all64 existing training paths, seed, three direct latent actions, AR exploration, optimizer and '
        '24000-decision budget. The only arm difference is inclusion/removal of the discrete0.5deg/20ms '
        'first-turn quality term; retain spatial bending, outward excursion and excess reversal terms, '
        'bounded terminal accounting and original full-time/failure penalties. This differs from earlier '
        'coefficient PPO in architecture/action support/exploration, not a single-factor comparison to it. '
        'Physical limits, ordered coverage and terminal stop are unchanged. Initial model has9/12 '
        'development completion; every failed prefix stays a failure. Stop this bounded experiment and '
        'analyze actual shape, cruise and completion; do not extend unchanged merely because return rises.'))
    saved=torch.load(initial,map_location='cpu',weights_only=False)
    policy=ConditionalLocalGeometryPolicy(72,mask_endpoint=True);policy.load_state_dict(saved['policy'])
    torch.manual_seed(seed);policy.reset_context()
    optimizer={'actor':torch.optim.Adam(policy.actor_parameters(),lr=3e-5),
               'critic':torch.optim.Adam(policy.critic.parameters(),lr=.001)}
    started=time.perf_counter();deadline=started+wall_budget
    costs=dict(training_actor_calls=0,training_physics_ticks=0,evaluation_actor_calls=0,evaluation_physics_ticks=0,
               ppo_updates=0,actor_minibatches=0,critic_minibatches=0)
    (stage/'episodes').mkdir();episodes=[];groups=[];group=[];batch=[];batch_size=0;updates=[];queue=[];checkpoints=[];next_check=6000;last_check=-1
    validation=[np.asarray(p) for p in paths['validation']]
    def checkpoint():
        nonlocal last_check
        n=costs['training_actor_calls']
        if n==last_check:return
        model=stage/f'policy_{n}.pt';save_torch(model,dict(policy=policy.state_dict(),observation_dim=72,mode='joint',
            architecture='local_geometry_feedback',mask_endpoint=True,frame='heading',history_independent=True,
            learning_method=f'{args.side_coverage}_side_coverage_PPO' if args.side_coverage else f'local_geometry_{args.variant}_reward_PPO',training_decisions=n,parent_sha256=sha(initial),
            exploration=dict(rho=RHO,std=list(policy.exploration_std))))
        last_check=n
        reserve=sum(math.ceil(environment_factory(p).max_ticks/20) for p in validation)
        if costs['evaluation_actor_calls']+reserve>14000 or time.perf_counter()>deadline-25:
            checkpoints.append(dict(decisions=n,model_sha256=sha(model),evaluation='deferred_budget'))
            write_json(stage/'checkpoints.json',dict(checkpoints=checkpoints,costs=costs));return
        tag=f'validation_{n}';results=evaluate(policy,validation,stage,tag,None,costs,deadline,environment_factory)
        row=summarize(stage,tag,results,list(range(12)));row.update(decisions=n,model_sha256=sha(model),final_acceptance=False)
        for i in range(4):
            with np.load(stage/tag/f'{i}.npz') as z:
                row['paths'][i]['onset_sensitivity']=onset(z['reference'],z['rows'])
                row['paths'][i]['complete_shape']=complete_corner_metrics(z['reference'],z['rows'])
        checkpoints.append(row);write_json(stage/'checkpoints.json',dict(checkpoints=checkpoints,costs=costs))
        print('CHECKPOINT',args.variant,n,row['successes'],row['full_time_s'],
            [(round(r['shape']['outward_excursion_mm'],4),round(r['shape']['reverse_turning_total_deg'],2),round(r['shape']['spatial']['middle80_turn_arc_length_mm'],2)) for r in row['paths'][:4]],
            'long95',row['paths'][-1]['cruise_ge95_s'],flush=True)
        if args.side_coverage:print('EXIT',args.side_coverage,[(round(r['complete_shape']['out_edge_outward_mm'],4),round(r['complete_shape']['heading_overshoot_deg'],2),round(r['complete_shape']['full_physical_reverse_deg'],2)) for r in row['paths'][:4]],flush=True)
    try:
        reason=None
        while True:
            if not queue:queue=[int(i) for i in rng.permutation(indices) for _ in range(4)]
            index=queue.pop();ref=np.asarray(paths['training'][index]);env=environment_factory(ref)
            reserve=(4-len(group))*math.ceil(env.max_ticks/20)
            if costs['training_actor_calls']+reserve>training_budget:reason='complete_group_decision_reserve';break
            if time.perf_counter()>deadline-75:reason='wall_reserve';break
            if costs['training_physics_ticks']+costs['evaluation_physics_ticks']+(4-len(group))*env.max_ticks>900000:reason='physical_tick_reserve';break
            policy.reset_context();data={k:[] for k in ('observation','latent','previous_observation','previous_latent','context_live','old_log','base_reward','value','terminal')};chunks=[];held=[]
            while not env.done:
                obs=env.observation();latent,logp,value=policy.act(obs)
                _,reward,done,info=env.step(latent)
                for k,v in dict(observation=obs,latent=latent,old_log=logp,base_reward=reward,value=value,terminal=done,**policy.last_context).items():data[k].append(v)
                chunks.append(info['rows']);held.append(info['held_action']);costs['training_actor_calls']+=1;costs['training_physics_ticks']+=len(info['rows'])
            rows=np.vstack(chunks);actions=np.asarray(held)
            shape_cost,details=terminal_shape_cost(ref,rows,actions,discrete_first_label=discrete,two_edges=two_edges)
            data={k:np.asarray(v) for k,v in data.items()};data['reward']=data['base_reward']-shape_cost
            audit=difference_audit(ref[0],rows);success=bool(env.success and audit['kinematic_pass'] and audit['rest_pass'])
            length=np.linalg.norm(np.diff(ref,axis=0),axis=1).sum();expected=-env.ticks*.001+length/100
            if not env.success:expected-=env.max_ticks*.001+1+max(0,length-env.progress()[0])/20
            assert abs(data['base_reward'].sum()-expected)<1e-8
            number=len(episodes);trace=f'episode_{number}.npz';np.savez_compressed(stage/'episodes'/trace,reference=ref,rows=rows,actions=actions,shape_cost=shape_cost,**data)
            episodes.append(dict(episode=number,path=index,success=success,reason=env.reason,full_time_s=env.ticks*.001 if success else None,
                base_return=float(data['base_reward'].sum()),shaped_return=float(data['reward'].sum()),shape_cost=details,trace=trace,audit=audit,geometry=env.tracker.snapshot(1.25)))
            batch.append(data);batch_size+=len(actions);group.append((number,index,data))
            if len(group)==4:
                assert len({g[1] for g in group})==1
                total=np.array([g[2]['reward'].sum() for g in group]);baseline=(total.sum()-total)/3;scale=max(float(total.std()),.02)
                for g,adv in zip(group,total-baseline):g[2]['paired_advantage']=np.full(len(g[2]['reward']),adv/scale)
                groups.append(dict(episodes=[g[0] for g in group],path=index,returns=total.tolist(),leave_one_out_baselines=baseline.tolist(),advantage_scale=scale));group=[]
                write_json(stage/'paired_groups.json',dict(groups=groups,own_return_excluded=True))
            write_json(stage/'episode_results.json',dict(episodes=episodes,costs=costs))
            if batch_size<1024 or group:continue
            stats=update(policy,optimizer,batch,paired_advantage=True);costs['ppo_updates']+=1
            for key in ['actor_minibatches','critic_minibatches']:costs[key]+=stats[key]
            stats.update(decisions=costs['training_actor_calls'],episodes=len(batch),successes=sum(e['success'] for e in episodes[-len(batch):]))
            updates.append(stats);batch=[];batch_size=0;write_json(stage/'updates.json',dict(updates=updates,costs=costs))
            print('UPDATE',args.variant,stats['decisions'],stats['successes'],'/',stats['episodes'],'KL',round(stats['mean_kl'],5),flush=True)
            if costs['training_actor_calls']>=next_check:checkpoint();next_check+=6000
        assert not group
        if batch:
            stats=update(policy,optimizer,batch,paired_advantage=True);costs['ppo_updates']+=1
            for key in ['actor_minibatches','critic_minibatches']:costs[key]+=stats[key]
            stats.update(decisions=costs['training_actor_calls'],final_complete_episode_batch=True);updates.append(stats)
        checkpoint()
        save_torch(stage/'final_optimizer_state.pt',dict(optimizer={k:v.state_dict() for k,v in optimizer.items()},torch_rng=torch.get_rng_state(),generator_rng=rng.bit_generator.state,decisions=costs['training_actor_calls']))
        write_json(stage/'results.json',dict(checkpoints=checkpoints,updates=updates,costs=costs,stop_reason=reason,
            visited_training_paths=sorted({e['path'] for e in episodes}),goal_complete=False))
        receipt(stage,dict(status='completed_matched_local_geometry_PPO',variant=args.variant,side_coverage=args.side_coverage,costs=costs,wall_s=time.perf_counter()-started))
    except Exception:
        write_json(stage/'failure.json',dict(traceback=traceback.format_exc(),costs=costs,wall_s=time.perf_counter()-started));raise


if __name__=='__main__':main()
