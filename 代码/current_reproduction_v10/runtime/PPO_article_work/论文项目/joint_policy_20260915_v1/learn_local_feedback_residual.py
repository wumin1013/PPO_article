"""Fitted local policy improvement from complete counterfactual suffix returns."""
import time
import numpy as np
import torch
from local_residual_policy import LocalResidualPolicy
from reflection_policy import reflect_observation
from probe_coordinated_suffix import reconstruct
from probe_constrained_parameter_step import summarize
from train_constrained_frame_search import rank
from train_pilot import evaluate,save_torch
from joint_runtime_v5 import environment_factory
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    torch.set_num_threads(1)
    source=BASE/'constrained_parameter_step_probe_20260916';initial=source/'candidate_1.pt'
    probe=BASE/'local_feedback_credit_probe_20260916'
    stage=register('local_counterfactual_feedback_learning_20260916',
        [__file__,BASE/'local_residual_policy.py',initial,source/'training_results.json',probe/'receipt.json',probe/'results.json',
         BASE/'probe_coordinated_suffix.py',BASE/'frame_feedback_policy.py',BASE/'reflection_policy.py',BASE/'joint_runtime_v5.py'],
        dict(training_evaluation_decisions=7000,development_decisions=1500,physics_ticks=185000,wall_s=300,
             ridge=.001,parameter_steps=[.25,.5,1.],feature_count=97,trainable_parameters=291,
             block_weights=dict(original_feedback_states=1.,profitable_interventions=1.,branch_continuation_states=1.)),
        'Fit a frozen-feature reflected linear residual using complete-return improving local branch labels. '
        'Only six previously probed training paths supply positive labels. All16 original training trajectories '
        'and unperturbed positive-branch continuations regularize toward zero residual. Same successful/failure '
        'return definitions as the probe; every failure remains recorded, no failed prefix labels counted as success. '
        'Choose one highest-return improving successful branch per anchor(delta return>0.002s), then weighted ridge '
        'regression in normalized latent units. Evaluate parameter steps.25,.5,1 on all16 training paths; select '
        'one by predeclared full-success/incoming/reversal/time+energy rank, write selection before development. '
        'The deployed model sees only observations, no anchor/timing/angle labels or runtime simulation. '
        'This is model-assisted fitted policy improvement, not PPO. No repeated fitting if no true shape/cruise gain.')
    started=time.perf_counter();costs=dict(training_actor_calls=0,training_physics_ticks=0,evaluation_actor_calls=0,evaluation_physics_ticks=0,saved_prefix_geometry_ticks=0)
    saved=torch.load(initial,map_location='cpu',weights_only=False);policy=LocalResidualPolicy(saved['policy'])
    indices=[4,5,6,8,9,11,12,13,14,15,18,22,26,34,50,53]
    baseline_observations=[]
    for i in range(16):
        with np.load(source/'training_3'/f'{i}.npz') as z:ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
        observations=[];env=reconstruct(ref,rows,actions,(len(actions)-1)*20,costs,observations)
        observations.append(env.observation());observations=np.asarray(observations)
        for obs,action in zip(observations,actions):
            assert np.array_equal(env.decode_action(policy.act(obs,True)[0]),action)
        baseline_observations.append(observations)
    original=np.concatenate(baseline_observations)
    data=read(probe/'results.json');chosen=[]
    for base in data['baselines']:
        for anchor in base['anchors']:
            candidates=[r for r in data['records'] if r['path']==base['path'] and r['anchor']==anchor and r['success'] and r['delta_return']>.002]
            if candidates:chosen.append(max(candidates,key=lambda r:r['delta_return']))
    positive_x=[];positive_y=[];continuation=[]
    for item in chosen:
        with np.load(probe/item['trace']) as z:
            n=int(z['intervention_decisions']);positive_x.append(z['observation'][:n]);positive_y.append((z['latent'][:n]-z['base_latent'][:n])/np.array([.02,.3,.3]))
            continuation.append(z['observation'][n:])
    positive=np.concatenate(positive_x);target=np.concatenate(positive_y);tail=np.concatenate(continuation)
    x=np.concatenate([original,positive,tail]);y=np.concatenate([np.zeros((len(original),3)),target,np.zeros((len(tail),3))])
    w=np.concatenate([np.full(len(block),1/len(block)) for block in [original,positive,tail]])/3
    with torch.no_grad():features=policy.features(torch.tensor(x)).numpy().astype(float)
    solution=[];fit=[]
    for head in range(3):
        f=features[:,head];gram=f.T@(w[:,None]*f)+.001*np.eye(f.shape[1]);rhs=f.T@(w*y[:,head])
        weights=np.linalg.solve(gram,rhs);solution.append(weights)
        fit.append(dict(head=head,weighted_mse=float(np.sum(w*(f@weights-y[:,head])**2)),weight_norm=float(np.linalg.norm(weights))))
    solution=torch.tensor(np.asarray(solution),dtype=torch.float32)
    with torch.no_grad():
        policy.weights.copy_(solution)
        xx=torch.tensor(x[:100]);error=float((policy.mean(reflect_observation(xx))-policy.mean(xx)*xx.new_tensor([-1.,1.,1.])).abs().max())
        assert error<1e-6
    np.savez_compressed(stage/'training_data.npz',observation=x,target_residual=y,weight=w)
    write_json(stage/'fit.json',dict(chosen_branches=[dict(path=r['path'],anchor=r['anchor'],trace=r['trace'],delta_return=r['delta_return']) for r in chosen],
        baseline_states=len(original),positive_states=len(positive),continuation_states=len(tail),fit=fit,reflection_error=error,
        baseline_decode_exact=True,physics_ticks_for_reconstruction=0))
    paths=read(BASE/'learning_preparation_v4/paths.json');training=[np.asarray(paths['training'][i]) for i in indices]
    baseline=next(r for r in read(source/'training_results.json')['candidates'] if r['alpha']==.5);baseline['rank']=rank(baseline)
    results=[]
    for step in [.25,.5,1.]:
        policy.weights.copy_(solution*step)
        name=f'candidate_{step:g}.pt';save_torch(stage/name,dict(policy=policy.state_dict(),observation_dim=72,
            architecture='frozen_feature_local_residual',learning_method='local_counterfactual_fitted_policy_improvement',
            parent_sha256=sha(initial),parameter_step=step))
        before_calls=costs['evaluation_actor_calls'];before_ticks=costs['evaluation_physics_ticks']
        tag=f'training_{step:g}';evaluated=evaluate(policy,training,stage,tag,None,costs,started+300,environment_factory)
        costs['training_actor_calls']+=costs['evaluation_actor_calls']-before_calls;costs['evaluation_actor_calls']=before_calls
        costs['training_physics_ticks']+=costs['evaluation_physics_ticks']-before_ticks;costs['evaluation_physics_ticks']=before_ticks
        result=summarize(stage,tag,evaluated,indices);result.update(step=step,model=name);result['rank']=rank(result);results.append(result)
        write_json(stage/'training_results.json',dict(baseline=baseline,candidates=results,costs=costs))
        print('LOCAL_FITTED',step,result['rank'],result['full_time_s'],flush=True)
        assert costs['training_actor_calls']<=7000
    best=max(results,key=lambda r:r['rank'])
    selected=best if best['rank']>baseline['rank'] and best['successes']==16 else None
    write_json(stage/'selection.json',dict(selected=selected,basis='training only; no development evaluation before this selection',baseline_rank=baseline['rank']))
    development=None
    if selected:
        policy.load_state_dict(torch.load(stage/selected['model'],map_location='cpu',weights_only=False)['policy'])
        evaluated=evaluate(policy,[np.asarray(p) for p in paths['validation']],stage,'validation',None,costs,started+300,environment_factory)
        development=summarize(stage,'validation',evaluated,list(range(12)))
        print('LOCAL_DEVELOPMENT',development['successes'],development['full_time_s'],flush=True)
    assert costs['evaluation_actor_calls']<=1500 and costs['training_physics_ticks']+costs['evaluation_physics_ticks']<=185000
    write_json(stage/'results.json',dict(development=development,selected=selected,costs=costs,goal_complete=False))
    receipt(stage,dict(status='completed_local_fitted_policy_improvement',costs=costs,wall_s=time.perf_counter()-started))


if __name__=='__main__':main()
