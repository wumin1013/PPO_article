"""Matched marginal-noise pilot: change only feed/preview correlation.

Both arms use the same wide Gaussian marginals and AR24 time scale. This is
an exploration-support diagnostic, not a comparison to old AR8 training.
"""
import time
import math
import numpy as np
import torch
from dynamic_residual_policy import PreviewResidualPolicy
from joint_runtime_v5 import environment_factory
from isolated_corner_shape import corner_shape
from geometric_turn_metrics import geometric_turn_metrics
from validate_actions import difference_audit
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    source=BASE/'preview_state_residual_learning_20260916';model=source/'best_generation_5.pt'
    stage=register('correlated_exploration_support_probe_20260916',[__file__,model,source/'receipt.json',
        BASE/'dynamic_residual_policy.py',BASE/'wide_joint_action_coverage_audit_20260916/receipt.json',
        BASE/'learning_preparation_v4/paths.json'],
        dict(training_updates=0,maximum_episodes=24,training_paths=[8,9,14,15],seeds=[916301,916302,916303],
             marginal_std=[.03,.4,1.],ar_time_decisions=24,feed_preview_correlations=[0.,-.6],
             decisions=3000,physics_ticks=100000,wall_s=180),
        'Matched Gaussian marginal/time-scale contrast of independent versus negatively correlated feed/preview innovations. Frozen learned feedback policy, paired seeds/paths, complete stops and failures retained. Marginal widths target an underexplored reachable action region; no optimizer update or hand-coded corner policy.')
    torch.set_num_threads(1);saved=torch.load(model,map_location='cpu',weights_only=False)
    policy=PreviewResidualPolicy(saved['observation_dim']);policy.load_state_dict(saved['policy'])
    paths=read(BASE/'learning_preparation_v4/paths.json')['training'];started=time.perf_counter()
    costs=dict(decisions=0,physics_ticks=0,training_updates=0);results=[];rho=math.exp(-1/24)
    for correlation in (0.,-.6):
        label='independent' if correlation==0 else 'correlated'
        covariance=np.diag(np.array([.03,.4,1.])**2);covariance[1,2]=covariance[2,1]=correlation*.4
        factor=np.linalg.cholesky(covariance)
        for seed in (916301,916302,916303):
            for path_index in (8,9,14,15):
                ref=np.asarray(paths[path_index]);env=environment_factory(ref)
                rng=np.random.default_rng(seed+path_index);previous_residual=None;chunks=[];actions=[];observations=[];latents=[];logps=[];rewards=[]
                while not env.done:
                    assert costs['decisions']<3000 and costs['physics_ticks']<97800 and time.perf_counter()-started<180
                    obs=env.observation();mean=policy.act(obs,True)[0]
                    scale=1. if previous_residual is None else math.sqrt(1-rho*rho)
                    center=mean if previous_residual is None else mean+rho*previous_residual
                    latent=center+scale*factor@rng.normal(size=3)
                    distribution=torch.distributions.MultivariateNormal(torch.as_tensor(center),scale_tril=torch.as_tensor(scale*factor))
                    logp=float(distribution.log_prob(torch.as_tensor(latent)))
                    previous_residual=latent-mean
                    _,reward,_,info=env.step(latent);chunks.append(info['rows']);actions.append(info['held_action'])
                    observations.append(obs);latents.append(latent);logps.append(logp);rewards.append(reward)
                    costs['decisions']+=1;costs['physics_ticks']+=len(info['rows'])
                rows=np.vstack(chunks);held=np.asarray(actions);audit=difference_audit(ref[0],rows)
                success=bool(env.success and audit['kinematic_pass'] and audit['rest_pass'])
                ticks=np.arange(len(held))*20;xy=np.vstack([ref[0],rows[:,:2]])[ticks]
                tangent=(ref[1]-ref[0])/np.linalg.norm(ref[1]-ref[0]);remaining=-(xy-ref[1])@tangent
                apex=np.argmin(np.linalg.norm(rows[:,:2]-ref[1],axis=1))
                region=(remaining>=4)&(remaining<=12)&(ticks<=apex)
                joint=region&(held[:,2]>=6)&(held[:,1]>=20)&(held[:,1]<=55)&(np.abs(held[:,0])<=math.radians(10))
                run=0;longest=0
                for flag in joint:run=run+1 if flag else 0;longest=max(longest,run)
                shape=corner_shape(ref,rows,held) if len(rows)>21 and np.linalg.norm(rows[:,2:4],axis=1).max()>5 else None
                spatial=geometric_turn_metrics(ref,rows) if np.linalg.norm(np.diff(np.vstack([ref[0],rows[:,:2]]),axis=0),axis=1).sum()>.4 else None
                tag=f'{label}_{seed}_{path_index}'
                np.savez_compressed(stage/f'{tag}.npz',reference=ref,rows=rows,actions=held,observation=np.asarray(observations),
                    latent=np.asarray(latents),conditional_log_probability=np.asarray(logps),base_reward=np.asarray(rewards),covariance=covariance,rho=rho)
                result=dict(arm=label,seed=seed,path=path_index,trace=f'{tag}.npz',success=success,reason=env.reason,
                    full_time_s=env.ticks*.001 if success else None,elapsed_s=env.ticks*.001,audit=audit,geometry=env.tracker.snapshot(1.25),
                    coordinated_decisions=int(joint.sum()),longest_coordinated_run_decisions=longest,
                    region_decisions=int(region.sum()),shape=shape,spatial=spatial,base_return=sum(rewards))
                results.append(result);write_json(stage/'partial_results.json',dict(results=results,costs=costs))
                print(label,seed,path_index,success,env.reason,'joint',int(joint.sum()),'run',longest,flush=True)
    summary=[]
    for arm in ('independent','correlated'):
        selected=[r for r in results if r['arm']==arm]
        summary.append(dict(arm=arm,episodes=len(selected),successes=sum(r['success'] for r in selected),
            records_with_joint_run_ge60ms=sum(r['longest_coordinated_run_decisions']>=3 for r in selected),
            successes_with_joint_run_ge60ms=sum(r['longest_coordinated_run_decisions']>=3 and r['success'] for r in selected)))
    write_json(stage/'results.json',dict(results=results,summary=summary,costs=costs,goal_complete=False,
        boundary='24 frozen-policy stochastic episodes; finite sample support diagnostic, not proof of learning effectiveness or final generalization.'))
    receipt(stage,dict(status='completed',costs=costs,wall_s=time.perf_counter()-started,model_sha256=sha(model)))
    print(summary,flush=True)


if __name__=='__main__':main()
