"""Isolated input/reward contract, termination ranking, and fresh-model checks."""
import json
from pathlib import Path
import numpy as np
import torch
from spatial_learning_candidate import environment_factory, fresh_policy, spatial_feature_names
from study_io import BASE, register, receipt, write_json


def main():
    stage=register('spatial_learning_contract_v1',[Path(__file__),BASE/'spatial_learning_candidate.py',
        BASE/'compact_observation_candidate.py',BASE/'event_observation_candidate.py',
        BASE/'joint_policy.py',BASE/'learning_preparation_v4/paths.json',
        BASE/'event_horizon_information_probe_v2/results.json'],
        dict(environment_constructions=85,physical_macros=9,physical_ticks_upper=19080,
             model_constructions=4,batched_actor_forwards=4,batched_critic_forwards=4,training_updates=0),
        'Verify spatial22 finite observations and paired fresh initialization; normal/success/failure execution unchanged under each separate learning-side switch.')
    paths=json.loads((BASE/'learning_preparation_v4/paths.json').read_text(encoding='utf-8'))
    all_refs=[np.asarray(ref) for ref in paths['training']+paths['validation']]
    observations={72:[],22:[]}
    for ref in all_refs:
        env=environment_factory(ref,representation='spatial22')
        observations[22].append(env.observation())
        env.spatial=False
        observations[72].append(env.observation())
    for dim,values in observations.items():
        assert np.asarray(values).shape==(76,dim) and np.isfinite(values).all()
    initializations=[]
    for dim in (72,22):
        for mode in ('joint','fixed'):
            policy=fresh_policy(dim,mode=mode,seed=917)
            obs=torch.tensor(np.asarray(observations[dim]))
            with torch.no_grad():
                distribution=policy.distribution(obs)
                values=policy.value(obs)
            assert torch.equal(values,torch.zeros_like(values))
            assert torch.equal(distribution.mean,distribution.mean[0].expand_as(distribution.mean))
            initializations.append(dict(dim=dim,mode=mode,
                parameters=sum(p.numel() for p in policy.parameters()),
                actor_parameters=sum(p.numel() for p in policy.actor.parameters())+policy.log_std.numel(),
                latent_mean=distribution.mean[0].tolist(),value_zero=True,
                active_mask=policy.active_mask.tolist()))
    assert all(row['latent_mean']==initializations[0]['latent_mean'] for row in initializations)
    # Read the new saved counterexample; no repeat physical rollout of that pair.
    old=json.loads((BASE/'event_horizon_information_probe_v2/results.json').read_text(encoding='utf-8'))
    # Encoding itself is audited below on normal, true-terminal and timeout transitions.
    variants=(('original72','original'),('spatial22','original'),('original72','unshaped'))
    scenarios=[]
    action=np.array(initializations[0]['latent_mean'])
    ticks=0
    for scenario in ('normal','success','timeout'):
        ref=np.asarray([[0.,0.],[20.,0.]]) if scenario=='success' else all_refs[25]
        envs=[environment_factory(ref,representation=rep,reward_mode=reward) for rep,reward in variants]
        for env in envs:
            if scenario=='success':
                env.position[:]=[19.8,0.]
                env.tracker.position=env.position.copy()
            elif scenario=='timeout':
                env.max_ticks=1
        before=envs[0].potential()
        steps=[env.step(action) for env in envs]
        ticks+=sum(len(step[3]['rows']) for step in steps)
        after=0. if envs[0].done else envs[0].potential()
        assert all(np.array_equal(steps[0][3]['rows'],step[3]['rows']) for step in steps[1:])
        assert steps[0][1]==steps[1][1]
        assert np.array_equal(steps[0][0],steps[2][0])
        assert abs(steps[2][1]-(steps[0][1]-after+before))<1e-12
        assert all(env.done==envs[0].done and env.reason==envs[0].reason for env in envs)
        if scenario=='success':
            assert all(env.success for env in envs)
        elif scenario=='timeout':
            assert all(env.reason=='task_timeout' for env in envs)
            assert steps[2][1]<-envs[2].max_ticks*.001
        scenarios.append(dict(scenario=scenario,reason=envs[0].reason,
            rewards=[step[1] for step in steps],physical_identity=True,
            reward_only_observation_identity=True,state_only_reward_identity=True))
    write_json(stage/'results.json',dict(initializations=initializations,
        spatial_feature_names=spatial_feature_names(),initial_state_paths=76,scenarios=scenarios,
        geometry_caveat=old['immediate_alias']['interpretation'],
        dimension_is_candidate_not_minimality_claim=True,training_updates=0,goal_complete=False))
    receipt(stage,dict(status='passed_new_learning_input_and_reward_contract',
        environment_constructions=85,physical_macros=9,physics_ticks=ticks,
        model_constructions=4,batched_actor_forwards=4,batched_critic_forwards=4,training_updates=0))
    print(json.dumps(dict(initializations=initializations,scenarios=scenarios,physics_ticks=ticks),ensure_ascii=False))


if __name__=='__main__':
    main()
