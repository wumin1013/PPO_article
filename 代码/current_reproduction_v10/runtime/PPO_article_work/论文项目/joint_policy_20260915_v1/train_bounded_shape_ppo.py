"""Full-feedback PPO refinement with bounded, spatially assigned shape cost.

Complete episodes, gamma=lambda=1, original failure charges, history-correct
AR8 likelihood, separate actor/critic updates and measured KL stopping.
No teacher timing schedule is present during collection or evaluation.
"""
import time
import math
import traceback
import argparse
import numpy as np
import torch
from conditional_feedback_policy import ConditionalFeedbackPolicy,ConditionalFramePolicy,RHO,STD
from bounded_shape_return import bounded_shape_cost,terminal_shape_cost
from joint_runtime_v5 import environment_factory
from joint_policy import advantages
from search_complete_return import read
from train_pilot import evaluate,save_torch
from isolated_corner_shape import corner_shape
from geometric_turn_metrics import geometric_turn_metrics
from validate_actions import difference_audit
from study_io import BASE,register,receipt,write_json,sha


def update(policy,optimizer,episodes,*,paired_advantage=False,actor_sample_mask=None):
    keys=['observation','latent','previous_observation','previous_latent','context_live','old_log','reward','value','terminal']
    batch={k:np.concatenate([e[k] for e in episodes]) for k in keys}
    next_values=np.concatenate([np.r_[e['value'][1:],0.] for e in episodes])
    adv,returns=advantages(batch['reward'],batch['value'],next_values,batch['terminal'],lam=1.)
    if paired_advantage:
        adv=np.concatenate([e['paired_advantage'] for e in episodes]).astype(np.float32)
    exact=np.concatenate([np.cumsum(e['reward'][::-1])[::-1] for e in episodes])
    assert np.max(abs(returns-exact))<2e-5
    tensors={k:torch.as_tensor(v,dtype=torch.bool if k=='context_live' else torch.float32) for k,v in batch.items()}
    advantage=torch.tensor(adv);advantage=(advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-8)
    targets=torch.tensor(returns);actor_steps=0;stopped=False;logs=[]
    # Optional diagnostic loss intervention; default retains the original update.
    actor_mask=None if actor_sample_mask is None else torch.as_tensor(actor_sample_mask,dtype=torch.float32)
    if actor_mask is not None:
        assert actor_mask.shape==advantage.shape and bool(((actor_mask==0)|(actor_mask==1)).all())
    with torch.no_grad():
        fresh,_,_=policy.evaluate_conditioned(tensors['observation'],tensors['latent'],tensors['previous_observation'],tensors['previous_latent'],tensors['context_live'])
        density_error=float((fresh-tensors['old_log']).abs().max())
        assert density_error<1e-4,density_error
    for epoch in range(6):
        for idx in torch.randperm(len(adv)).split(256):
            logp,entropy,value=policy.evaluate_conditioned(tensors['observation'][idx],tensors['latent'][idx],tensors['previous_observation'][idx],tensors['previous_latent'][idx],tensors['context_live'][idx])
            logratio=logp-tensors['old_log'][idx];ratio=logratio.exp();kl=float(((ratio-1)-logratio).mean().detach())
            surrogate=torch.minimum(ratio*advantage[idx],ratio.clamp(.8,1.2)*advantage[idx])
            loss=-(surrogate if actor_mask is None else surrogate*actor_mask[idx]).mean()
            if not stopped and kl<=.03:
                optimizer['actor'].zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(policy.actor_parameters(),.5);optimizer['actor'].step();actor_steps+=1
            else:stopped=True
            value_loss=.5*(value-targets[idx]).square().mean()
            optimizer['critic'].zero_grad(set_to_none=True);value_loss.backward();torch.nn.utils.clip_grad_norm_(policy.critic.parameters(),1.);optimizer['critic'].step()
            logs.append([float(loss.detach()),float(value_loss.detach()),kl])
    return dict(actor_minibatches=actor_steps,critic_minibatches=len(logs),actor_kl_stop=stopped,
        mean_actor_loss=float(np.mean(logs,axis=0)[0]),mean_critic_loss=float(np.mean(logs,axis=0)[1]),mean_kl=float(np.mean(logs,axis=0)[2]),
        density_error_before_update=density_error,complete_episode_mc_error=float(np.max(abs(returns-exact))))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--terminal-shape-credit',action='store_true')
    parser.add_argument('--matched-baseline',choices=['critic','paired']);parser.add_argument('--frame-refinement',action='store_true');args=parser.parse_args()
    if args.frame_refinement:args.matched_baseline='paired'
    if args.matched_baseline:args.terminal_shape_credit=True
    training_budget=15000 if args.matched_baseline else 20000
    wall_budget=900 if args.matched_baseline else 1200
    cost_function=terminal_shape_cost if args.terminal_shape_credit else bounded_shape_cost
    torch.set_num_threads(1);torch.manual_seed(916719);rng=np.random.default_rng(916719)
    source=BASE/'interior_teacher_feedback_learning_20260916';initial=source/'epoch_50.pt'
    if args.matched_baseline:
        source=BASE/'terminal_shape_full_feedback_ppo_20260916';initial=source/'policy_10511.pt'
    if args.frame_refinement:
        source=BASE/'history_independent_heading_frame_feedback_learning_20260916';initial=source/'epoch_500.pt'
    indices=[4,5,6,8,9,11,12,13,14,15,18,22,26,34,50,53]
    stage_name=f'matched_{args.matched_baseline}_return_ppo_20260916' if args.matched_baseline else 'terminal_shape_full_feedback_ppo_20260916' if args.terminal_shape_credit else 'bounded_shape_full_feedback_ppo_20260916'
    if args.frame_refinement:stage_name='history_independent_heading_return_ppo_20260916'
    stage=register(stage_name,[__file__,initial,source/'receipt.json',
        BASE/'bounded_shape_return_audit_20260916/receipt.json',BASE/'bounded_shape_return.py',BASE/'conditional_feedback_policy.py',
        BASE/'dynamic_residual_policy.py',BASE/'reflection_policy.py',BASE/'geometric_turn_metrics.py',BASE/'joint_policy.py',
        BASE/'joint_runtime.py',BASE/'joint_runtime_v2.py',BASE/'joint_runtime_v3.py',BASE/'joint_runtime_v5.py',
        BASE/'terminal_candidate_runtime_v1.py',BASE/'geometry_spec.py',BASE/'learning_preparation_v4/paths.json'],
        dict(training_decisions=training_budget,evaluation_decisions=20000,physics_ticks=1500000,wall_s=wall_budget,seed=916719,
             train_indices=indices,batch_minimum_complete_episode_decisions=1024,ppo_epochs=6,minibatch=256,
             gamma=1.,gae_lambda=1.,actor_learning_rate=.00003,critic_learning_rate=.001,clip=.2,kl_stop=.03,
             ar_rho=RHO,fixed_latent_std=list(STD),validation_decision_thresholds=[5000,10000,15000,20000],
             shape_credit='terminal' if args.terminal_shape_credit else 'retrospective_spatial',
             matched_path_group_size=4 if args.matched_baseline else 1,actor_baseline=args.matched_baseline or 'critic',
             shape_cost='isolated corners: C=1-exp(-(1.25E/angle^2+5*incoming_violation+2*normalized_excess_reversal));0<=C<=1; other paths retain base return'),
        ('Closed-loop complete-return refinement of the new history-independent heading actor at500 supervised updates. It has11/12 complete development paths, broader150deg transition and lower successful-corner reversals, but125deg violates the corridor before the gate. Use original16 training geometries and4 independent episodes per path; group leave-one-out return avoids relying on its untrained critic for actor updates. Keep same AR8 noise, terminal bounded shape objective and failure charges. This is a new initialization/architecture refinement, not evidence that paired returns outperformed the earlier critic arm. Stop15000 decisions; reject absent stable actual shape, success and cruise gains. No teacher schedule at runtime. ' if args.frame_refinement else
        'Matched baseline mechanism comparison: both arms initialize at the first12/12 causal-credit checkpoint, use groups of4 independent complete episodes on identical training paths, same seed, action noise, updates and15000-decision budgets. Critic arm uses state-value MC reward-to-go; paired arm replaces the actor advantage by whole-episode return minus the other3 complete returns, constant within each episode. Critic is still fitted in both arms, but is not used by paired actor. No per-group variance scaling, no changed failure charge or discarded failed prefix. This is a credit/baseline mechanism comparison, not a claim that baseline source alone differs from all earlier ungrouped runs. Retain only actual shape/cruise/full-completion gains; stop both budgets without unchanged extension. ' if args.matched_baseline else
        'One-factor causal-credit correction: same initial weights, seed, training paths, actor/critic hyperparameters and budgets as bounded_shape_full_feedback_ppo. Unchanged total trajectory shape cost is charged at termination because retrospective normalization makes past spatial charges depend on future actions. Adopt only with actual completion/shape/cruise improvement; no unchanged extension. ' if args.terminal_shape_credit else '')+
        'Full-network PPO from a feedback model that learned wider turns but remains unsafe on mixed paths. Same16 training paths and unchanged physics. Optimize actual complete/failure returns instead of regression targets, with bounded shape cost and complete-episode MC credit. Algorithm/reward/exploration refinement is not an isolated superiority claim over CEM or imitation. Stop at registered budgets; preserve every failure and checkpoint. No teacher schedule or per-path parameters at run time; no sealed tests.')
    saved=torch.load(initial,map_location='cpu',weights_only=False)
    policy=ConditionalFramePolicy(72,frame=saved['frame'],history_independent=saved.get('history_independent',False)) if args.frame_refinement else ConditionalFeedbackPolicy(72)
    policy.load_state_dict(saved['policy'])
    optimizer={'actor':torch.optim.Adam(policy.actor_parameters(),lr=3e-5),'critic':torch.optim.Adam(policy.critic.parameters(),lr=1e-3)}
    paths=read(BASE/'learning_preparation_v4/paths.json');validation=[np.asarray(p) for p in paths['validation']]
    started=time.perf_counter();deadline=started+wall_budget
    costs=dict(training_actor_calls=0,training_physics_ticks=0,evaluation_actor_calls=0,evaluation_physics_ticks=0,ppo_updates=0,actor_minibatches=0,critic_minibatches=0)
    (stage/'episodes').mkdir();episode_records=[];updates=[];checkpoints=[];batch=[];batch_size=0;queue=[];next_validation=5000;last_evaluation=-1
    group=[];paired_groups=[]
    evaluation_reserve=sum(math.ceil(environment_factory(p).max_ticks/20) for p in validation)
    def checkpoint():
        nonlocal last_evaluation
        n=costs['training_actor_calls']
        if n==last_evaluation:return
        model_path=stage/f'policy_{n}.pt'
        save_torch(model_path,dict(policy=policy.state_dict(),observation_dim=72,mode='joint',architecture=saved.get('architecture','reflection_preview_residual'),
            frame=saved.get('frame'),history_independent=saved.get('history_independent',False),
            learning_method=f'matched_{args.matched_baseline}_return_PPO' if args.matched_baseline else 'terminal_shape_full_feedback_PPO' if args.terminal_shape_credit else 'bounded_spatial_shape_full_feedback_PPO',training_decisions=n,exploration=dict(rho=RHO,std=STD),parent_sha256=sha(initial)))
        last_evaluation=n
        if costs['evaluation_actor_calls']+evaluation_reserve>20000:
            checkpoints.append(dict(decisions=n,model_sha256=sha(model_path),evaluation='deferred_budget',final_shape_acceptance=False))
            write_json(stage/'checkpoints.json',dict(checkpoints=checkpoints,costs=costs))
            return
        label=f'validation_{n}';results=evaluate(policy,validation,stage,label,None,costs,deadline,environment_factory)
        shapes=[];cruise=[]
        for i in range(12):
            with np.load(stage/label/f'{i}.npz') as z:
                speed=np.linalg.norm(z['rows'][:,2:4],axis=1);cruise.append(dict(path=i,peak=float(speed.max()),ge95_seconds=float((speed>=95).sum()*.001)))
                if i<4:
                    s=corner_shape(z['reference'],z['rows'],z['actions']);s['spatial']=geometric_turn_metrics(z['reference'],z['rows']);shapes.append(s)
        row=dict(decisions=n,successes=sum(r['success'] for r in results),
            full_time_s=sum(r['completion_time_s'] for r in results) if all(r['success'] for r in results) else None,
            shapes=shapes,cruise=cruise,model_sha256=sha(model_path),final_shape_acceptance=False)
        checkpoints.append(row);last_evaluation=n
        write_json(stage/'checkpoints.json',dict(checkpoints=checkpoints,costs=costs))
        print('CHECKPOINT',n,row['successes'],row['full_time_s'],[(round(s['turn_deg']),round(s['outward_excursion_mm'],3),round(s['reverse_turning_total_deg'],2),round(s['spatial']['bending_energy_mm_inv'],2)) for s in shapes],flush=True)
    try:
        while True:
            if not queue:queue=[int(i) for i in rng.permutation(indices) for _ in range(4 if args.matched_baseline else 1)]
            path_index=int(queue.pop());ref=np.asarray(paths['training'][path_index]);env=environment_factory(ref)
            reserve=(4-len(group)) if args.matched_baseline else 1
            if costs['training_actor_calls']+reserve*math.ceil(env.max_ticks/20)>training_budget or time.perf_counter()>deadline-120:break
            if costs['training_physics_ticks']+costs['evaluation_physics_ticks']+env.max_ticks>1400000:break
            policy.reset_context();data={k:[] for k in ('observation','latent','previous_observation','previous_latent','context_live','old_log','base_reward','value','terminal')};chunks=[];held=[]
            while not env.done:
                obs=env.observation();latent,logp,value=policy.act(obs)
                _,reward,done,info=env.step(latent);context=policy.last_context
                for k,v in dict(observation=obs,latent=latent,old_log=logp,base_reward=reward,value=value,terminal=done,**context).items():data[k].append(v)
                chunks.append(info['rows']);held.append(info['held_action']);costs['training_actor_calls']+=1;costs['training_physics_ticks']+=len(info['rows'])
            rows=np.vstack(chunks);actions=np.asarray(held);shape_cost,shape_details=cost_function(ref,rows,actions)
            data={k:np.asarray(v) for k,v in data.items()};data['reward']=data['base_reward']-shape_cost
            audit=difference_audit(ref[0],rows);success=bool(env.success and audit['kinematic_pass'] and audit['rest_pass'])
            length=np.linalg.norm(np.diff(ref,axis=0),axis=1).sum()
            expected=-env.ticks*.001+length/100
            if not env.success:expected-=env.max_ticks*.001+1+max(0,length-env.progress()[0])/20
            assert abs(data['base_reward'].sum()-expected)<1e-8
            number=len(episode_records);trace=f'episode_{number}.npz'
            np.savez_compressed(stage/'episodes'/trace,reference=ref,rows=rows,actions=actions,shape_cost=shape_cost,**data)
            episode_records.append(dict(episode=number,path=path_index,success=success,reason=env.reason,full_time_s=env.ticks*.001 if success else None,
                base_return=float(data['base_reward'].sum()),shaped_return=float(data['reward'].sum()),shape_cost=shape_details,trace=trace,
                audit=audit,geometry=env.tracker.snapshot(1.25)))
            batch.append(data);batch_size+=len(actions)
            if args.matched_baseline:
                group.append((number,path_index,data))
                if len(group)==4:
                    assert len({e[1] for e in group})==1
                    totals=np.array([e[2]['reward'].sum() for e in group]);baselines=(totals.sum()-totals)/3
                    for entry,advantage_value in zip(group,totals-baselines):entry[2]['paired_advantage']=np.full(len(entry[2]['reward']),advantage_value)
                    paired_groups.append(dict(episodes=[e[0] for e in group],path=path_index,complete_returns=totals.tolist(),leave_one_out_baselines=baselines.tolist()))
                    write_json(stage/'paired_groups.json',dict(groups=paired_groups,own_return_excluded=True))
                    group=[]
            write_json(stage/'episode_results.json',dict(episodes=episode_records,costs=costs))
            if batch_size<1024 or group:continue
            statistics=update(policy,optimizer,batch,paired_advantage=args.matched_baseline=='paired');costs['ppo_updates']+=1
            costs['actor_minibatches']+=statistics['actor_minibatches'];costs['critic_minibatches']+=statistics['critic_minibatches']
            statistics.update(decisions=costs['training_actor_calls'],episodes=len(batch),successes=sum(e['success'] for e in episode_records[-len(batch):]))
            updates.append(statistics);batch=[];batch_size=0
            write_json(stage/'updates.json',dict(updates=updates,costs=costs))
            print('UPDATE',costs['training_actor_calls'],statistics['successes'],'/',statistics['episodes'],'KL',round(statistics['mean_kl'],5),flush=True)
            if costs['training_actor_calls']>=next_validation:
                checkpoint();next_validation+=5000
        if batch:
            assert not group,'Never update an incomplete paired group'
            statistics=update(policy,optimizer,batch,paired_advantage=args.matched_baseline=='paired');costs['ppo_updates']+=1
            costs['actor_minibatches']+=statistics['actor_minibatches'];costs['critic_minibatches']+=statistics['critic_minibatches']
            statistics.update(decisions=costs['training_actor_calls'],episodes=len(batch),final_partial_batch_of_complete_episodes=True);updates.append(statistics)
        checkpoint()
        write_json(stage/'results.json',dict(checkpoints=checkpoints,updates=updates,costs=costs,goal_complete=False,
            boundary='Full-feedback PPO refinement pilot. Complete success, true XY shape and cruising are evaluated independently of training return. No final fair-baseline, independent-seed or sealed-test claim.'))
        receipt(stage,dict(status='completed_bounded_full_feedback_PPO',costs=costs,wall_s=time.perf_counter()-started))
    except Exception:
        write_json(stage/'failure.json',dict(traceback=traceback.format_exc(),costs=costs,wall_s=time.perf_counter()-started,preserved=True))
        raise


if __name__=='__main__':main()
