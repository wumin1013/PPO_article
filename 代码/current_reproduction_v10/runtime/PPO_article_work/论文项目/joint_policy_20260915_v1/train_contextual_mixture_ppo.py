"""PPO in learned function space, using the full existing training pool."""
import time
import math
import traceback
import argparse
import numpy as np
import torch
from contextual_mixture_policy import ContextualMixturePolicy,DeployedContextualView
from conditional_feedback_policy import RHO
from bounded_shape_return import terminal_shape_cost
from train_bounded_shape_ppo import update
from train_pilot import evaluate,save_torch
from joint_runtime_v5 import environment_factory
from validate_actions import difference_audit
from isolated_corner_shape import corner_shape
from geometric_turn_metrics import geometric_turn_metrics
from reflection_policy import reflect_observation
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--path-balanced',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1);torch.manual_seed(9161009);rng=np.random.default_rng(9161009)
    a_path=BASE/'constrained_parameter_step_probe_20260916/candidate_1.pt';b_path=BASE/'heading_frame_feedback_learning_20260916/epoch_2000.pt'
    paths=read(BASE/'learning_preparation_v4/paths.json');indices=list(range(len(paths['training'])))
    stage=register('path_balanced_contextual_ppo_20260916' if args.path_balanced else 'contextual_learned_function_ppo_20260916',[__file__,BASE/'contextual_mixture_policy.py',BASE/'conditional_feedback_policy.py',
        BASE/'frame_feedback_policy.py',BASE/'reflection_policy.py',BASE/'joint_policy.py',BASE/'train_bounded_shape_ppo.py',BASE/'bounded_shape_return.py',
        a_path,b_path,BASE/'learned_head_factorial_probe_20260916/receipt.json',BASE/'learned_mixture_pareto_review_20260916/receipt.json',
        BASE/'joint_runtime.py',BASE/'joint_runtime_v2.py',BASE/'joint_runtime_v3.py',BASE/'joint_runtime_v5.py',BASE/'terminal_candidate_runtime_v1.py',
        BASE/'geometry_spec.py',BASE/'isolated_corner_shape.py',BASE/'geometric_turn_metrics.py',BASE/'learning_preparation_v4/paths.json'],
        dict(training_decisions=24000,evaluation_decisions=15000 if args.path_balanced else 6000,physics_ticks=1000000,wall_s=900,seed=9161009,training_pool=indices,
             group_size=4,batch_minimum_decisions=1024,actor_lr=.0003,critic_lr=.001,epochs=6,minibatch=256,clip=.2,kl_stop=.03,
             ar_rho=RHO,coefficient_latent_std=[.25]*3,coefficient_map='clip raw sampled coefficients to[0,1]; PPO density uses unclipped samples',
             reward='unchanged full-time/failure reward plus existing bounded terminal shape cost',evaluation_thresholds=[5000,10000,15000,20000,24000],
             advantage_scaling='within-path return std, floor0.02s, followed by unchanged batch normalization' if args.path_balanced else 'batch normalization only'),
        ('Matched single-factor credit-scaling arm against contextual_learned_function_ppo: identical initialization, seed,64-path pool, grouped collection, physical action support, original reward/failure charges and PPO settings. Divide leave-one-out advantages by max(group return std,0.02s); the floor is one decision duration. This reweights gradient contributions across paths; it does not preserve the original unweighted expected-return objective. Every same-path return ordering and success/failure penalty is retained. The saved-data gradient audit motivates this test, not a causal claim about historical preswing. Fix checkpoint saving independently of evaluation budget; supplement all planned development checks. Adopt only actual shape/cruise/full-success gains; otherwise stop, do not continue unchanged. ' if args.path_balanced else '')+
        'Learn three state-dependent mixture coefficients jointly with PPO while freezing two learned feedback functions. Exact baseline initialization is the broad-turn0.5 parameter-step policy. Explore within their state-dependent latent-action intervals rather than perturbing all network outputs uniformly; all physical constraints and original LOS remain unchanged. Coefficients are reflection-even and use physical/geometry observations with prior command inputs masked. Broaden sampling from prior16 paths to all existing training paths only; no sealed paths. Use4 independent complete episodes per path and leave-one-out total-return advantages; critic is fitted but not used by actor. This changes representation, exploration scale and curriculum together as a new mechanism recipe, not a single-factor superiority claim. Stop24000 decisions/900s; evaluate actual XY, success and cruise, not coefficients or training reward. No trajectory lookup, angle tables, probe schedules or runtime search.')
    a=torch.load(a_path,map_location='cpu',weights_only=False);b=torch.load(b_path,map_location='cpu',weights_only=False)
    policy=ContextualMixturePolicy(72,a['policy'],b['policy']);view=DeployedContextualView(policy)
    # Offline contract checks: no new physical ticks.
    for ref in paths['training']:
        env=environment_factory(np.asarray(ref))
        if len(ref)==3:
            u,v=np.diff(np.asarray(ref),axis=0);assert abs(np.arctan2(u[0]*v[1]-u[1]*v[0],u@v))>1e-6
    with np.load(BASE/'interior_teacher_feedback_learning_20260916/training_data.npz') as z:x=torch.tensor(z['observation'][:100])
    maximum_error=0.
    for obs in x.numpy():
        sampled,logp,_=policy.act(obs);ctx=policy.last_context
        with torch.no_grad():
            new_log=policy.evaluate_conditioned(torch.tensor(obs),torch.tensor(sampled),torch.tensor(ctx['previous_observation']),torch.tensor(ctx['previous_latent']),torch.tensor(ctx['context_live']))[0]
            expected=policy.shape_function.distribution(torch.tensor(obs)).mean.numpy()
        maximum_error=max(maximum_error,abs(logp-float(new_log)))
        assert np.array_equal(view.act(obs,True)[0],expected)
    with torch.no_grad():
        raw=torch.randn(len(x),3);left=policy.environment_latent(x,raw);right=policy.environment_latent(reflect_observation(x),raw)
        assert np.max(abs(right-left*np.array([-1.,1.,1.])))<1e-6
    assert maximum_error<1e-5
    write_json(stage/'contract.json',dict(training_pool_size=len(indices),conditional_density_error=maximum_error,initial_deterministic_actions_exact=True,
        reflection_action_equivariance=True,frozen_components=True,physics_ticks=0))
    # Restore the declared RNG stream after contract-only random draws.
    torch.manual_seed(9161009);policy.reset_context()
    optimizer={'actor':torch.optim.Adam(policy.actor_parameters(),lr=.0003),'critic':torch.optim.Adam(policy.critic.parameters(),lr=.001)}
    validation=[np.asarray(r) for r in paths['validation']];started=time.perf_counter();deadline=started+900
    costs=dict(training_actor_calls=0,training_physics_ticks=0,evaluation_actor_calls=0,evaluation_physics_ticks=0,ppo_updates=0,actor_minibatches=0,critic_minibatches=0)
    (stage/'episodes').mkdir();episodes=[];group=[];groups=[];batch=[];batch_size=0;queue=[];updates=[];checkpoints=[];next_check=5000;last_check=-1
    def checkpoint():
        nonlocal last_check
        n=costs['training_actor_calls']
        if n==last_check:return
        model=stage/f'policy_{n}.pt';save_torch(model,dict(policy=policy.state_dict(),observation_dim=72,mode='joint',architecture='contextual_learned_function_mixture',
            learning_method='PPO_on_sampled_mixture_coefficients',training_decisions=n,parents=[sha(a_path),sha(b_path)],coefficient_clip=[0.,1.],exploration=dict(rho=RHO,std=policy.exploration_std)))
        last_check=n
        reserve=sum(math.ceil(environment_factory(r).max_ticks/20) for r in validation)
        if costs['evaluation_actor_calls']+reserve>(15000 if args.path_balanced else 6000):
            checkpoints.append(dict(decisions=n,model_sha256=sha(model),evaluation='deferred_budget',final_acceptance=False))
            write_json(stage/'checkpoints.json',dict(checkpoints=checkpoints,costs=costs))
            return
        tag=f'validation_{n}';result=evaluate(view,validation,stage,tag,None,costs,deadline,environment_factory)
        shapes=[];cruise=[]
        for i in range(12):
            with np.load(stage/tag/f'{i}.npz') as z:
                speed=np.linalg.norm(z['rows'][:,2:4],axis=1);cruise.append(dict(path=i,peak=float(speed.max()),ge95_seconds=float((speed>=95).sum()*.001)))
                if i<4:
                    s=corner_shape(z['reference'],z['rows'],z['actions']);s['spatial']=geometric_turn_metrics(z['reference'],z['rows']);shapes.append(s)
        row=dict(decisions=n,successes=sum(r['success'] for r in result),full_time_s=sum(r['completion_time_s'] for r in result) if all(r['success'] for r in result) else None,
            shapes=shapes,cruise=cruise,model_sha256=sha(model),final_acceptance=False)
        checkpoints.append(row);last_check=n;write_json(stage/'checkpoints.json',dict(checkpoints=checkpoints,costs=costs))
        print('CHECKPOINT',n,row['successes'],row['full_time_s'],[(round(s['turn_deg']),round(s['outward_excursion_mm'],3),round(s['reverse_turning_total_deg'],2),round(s['spatial']['bending_energy_mm_inv'],2)) for s in shapes],flush=True)
    try:
        while True:
            if not queue:queue=[int(i) for i in rng.permutation(indices) for _ in range(4)]
            index=queue.pop();reference=np.asarray(paths['training'][index]);env=environment_factory(reference)
            reserve=(4-len(group))*math.ceil(env.max_ticks/20)
            if costs['training_actor_calls']+reserve>24000 or time.perf_counter()>deadline-120:break
            if costs['training_physics_ticks']+costs['evaluation_physics_ticks']+(4-len(group))*env.max_ticks>900000:break
            policy.reset_context();data={k:[] for k in ('observation','latent','previous_observation','previous_latent','context_live','old_log','base_reward','value','terminal')};chunks=[];held=[]
            while not env.done:
                obs=env.observation();latent,logp,value=policy.act(obs)
                _,reward,done,info=env.step(policy.environment_latent(obs,latent));context=policy.last_context
                for key,value_ in dict(observation=obs,latent=latent,old_log=logp,base_reward=reward,value=value,terminal=done,**context).items():data[key].append(value_)
                chunks.append(info['rows']);held.append(info['held_action']);costs['training_actor_calls']+=1;costs['training_physics_ticks']+=len(info['rows'])
            rows=np.vstack(chunks);actions=np.asarray(held);shape_cost,shape_details=terminal_shape_cost(reference,rows,actions)
            data={k:np.asarray(v) for k,v in data.items()};data['reward']=data['base_reward']-shape_cost
            audit=difference_audit(reference[0],rows);success=bool(env.success and audit['kinematic_pass'] and audit['rest_pass'])
            length=np.linalg.norm(np.diff(reference,axis=0),axis=1).sum();expected=-env.ticks*.001+length/100
            if not env.success:expected-=env.max_ticks*.001+1+max(0,length-env.progress()[0])/20
            assert abs(data['base_reward'].sum()-expected)<1e-8
            number=len(episodes);filename=f'episode_{number}.npz';np.savez_compressed(stage/'episodes'/filename,reference=reference,rows=rows,actions=actions,shape_cost=shape_cost,**data)
            episodes.append(dict(episode=number,path=index,success=success,reason=env.reason,full_time_s=env.ticks*.001 if success else None,
                base_return=float(data['base_reward'].sum()),shaped_return=float(data['reward'].sum()),shape_cost=shape_details,trace=filename,audit=audit,geometry=env.tracker.snapshot(1.25)))
            batch.append(data);batch_size+=len(actions);group.append((number,index,data))
            if len(group)==4:
                assert len({g[1] for g in group})==1
                total=np.array([g[2]['reward'].sum() for g in group]);baseline=(total.sum()-total)/3
                scale=max(float(total.std()),.02) if args.path_balanced else 1.
                for g,adv in zip(group,total-baseline):g[2]['paired_advantage']=np.full(len(g[2]['reward']),adv/scale)
                groups.append(dict(episodes=[g[0] for g in group],path=index,returns=total.tolist(),leave_one_out_baselines=baseline.tolist(),advantage_scale=scale));group=[]
                write_json(stage/'paired_groups.json',dict(groups=groups,own_return_excluded=True))
            write_json(stage/'episode_results.json',dict(episodes=episodes,costs=costs))
            if batch_size<1024 or group:continue
            stats=update(policy,optimizer,batch,paired_advantage=True);costs['ppo_updates']+=1
            for key in ('actor_minibatches','critic_minibatches'):costs[key]+=stats[key]
            stats.update(decisions=costs['training_actor_calls'],episodes=len(batch),successes=sum(r['success'] for r in episodes[-len(batch):]))
            updates.append(stats);batch=[];batch_size=0;write_json(stage/'updates.json',dict(updates=updates,costs=costs))
            print('UPDATE',costs['training_actor_calls'],stats['successes'],'/',stats['episodes'],'KL',round(stats['mean_kl'],5),flush=True)
            if costs['training_actor_calls']>=next_check:checkpoint();next_check+=5000
        if batch:
            assert not group
            stats=update(policy,optimizer,batch,paired_advantage=True);costs['ppo_updates']+=1
            for key in ('actor_minibatches','critic_minibatches'):costs[key]+=stats[key]
            stats.update(decisions=costs['training_actor_calls'],final_complete_episode_batch=True);updates.append(stats)
        checkpoint()
        save_torch(stage/'final_optimizer_state.pt',dict(optimizer={k:v.state_dict() for k,v in optimizer.items()},torch_rng=torch.get_rng_state(),generator_rng=rng.bit_generator.state,decisions=costs['training_actor_calls']))
        # All learned base-function tensors must still be frozen and unchanged.
        for name,parent in [('shape_function',a),('cruise_function',b)]:
            assert all(torch.equal(v,getattr(policy,name).state_dict()[k]) for k,v in parent['policy'].items())
        write_json(stage/'results.json',dict(checkpoints=checkpoints,updates=updates,costs=costs,visited_training_paths=sorted({e['path'] for e in episodes}),frozen_components_exact=True,goal_complete=False))
        receipt(stage,dict(status='completed_bounded_contextual_function_PPO',costs=costs,wall_s=time.perf_counter()-started))
    except Exception:
        write_json(stage/'failure.json',dict(traceback=traceback.format_exc(),costs=costs));raise


if __name__=='__main__':main()
