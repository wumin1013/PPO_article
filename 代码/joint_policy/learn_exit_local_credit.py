"""Finite-suffix return direction with a training-state action trust region."""
import time
import copy
import numpy as np
import torch
from local_geometry_feedback_policy import LocalGeometryFeedbackPolicy
from frame_feedback_policy import heading_frame
from reflection_policy import reflect_observation
from probe_coordinated_suffix import reconstruct
from bounded_shape_return import terminal_shape_cost
from complete_corner_metrics import complete_corner_metrics
from train_pilot import evaluate,save_torch
from joint_runtime_v5 import environment_factory
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


@torch.no_grad()
def last_layer_features(policy,observations):
    x=torch.as_tensor(observations,dtype=torch.float32).clone()
    x[...,8:11]=x.new_tensor([0.,.3,.25]);x[...,13:15]=0.
    a=policy.actor[:4](heading_frame(x));b=policy.actor[:4](heading_frame(reflect_observation(x)))
    odd=(a-b)/2;even=(a+b)/2
    one=torch.ones_like(a[...,:1]);zero=torch.zeros_like(one)
    return torch.stack([torch.cat([odd,zero],-1),torch.cat([even,one],-1),torch.cat([even,one],-1)],1).numpy().astype(float)


def summary(folder,results,trace_paths=None):
    shapes=[];total_cost=0.;times=[]
    for i,row in enumerate(results):
        with np.load(trace_paths[i] if trace_paths is not None else folder/f'{i}.npz') as z:
            if len(z['reference'])==3:
                shapes.append(dict(path=row.get('original_path',i),success=bool(row['success']),failed_prefix=not bool(row['success']),
                                   **complete_corner_metrics(z['reference'],z['rows'])))
                c,_=terminal_shape_cost(z['reference'],z['rows'],z['actions'],discrete_first_label=False,two_edges=True)
                total_cost+=float(c.sum())
        times.append(row.get('completion_time_s'))
    success=sum(r['success'] for r in results)
    return dict(successes=success,full_time_s=sum(times) if success==len(results) else None,
        time_plus_shape_s=sum(times)+total_cost if success==len(results) else None,
        median_exit_mm=float(np.median([s['out_edge_outward_mm'] for s in shapes])) if success==len(results) and shapes else None,
        max_exit_mm=max(s['out_edge_outward_mm'] for s in shapes) if success==len(results) and shapes else None,
        max_entry_mm=max(s['in_edge_outward_mm'] for s in shapes) if success==len(results) and shapes else None,
        full_reverse_sum_deg=sum(s['full_physical_reverse_deg'] for s in shapes) if success==len(results) else None,
        comparable_complete_pool=success==len(results),shapes=shapes)


def main():
    torch.set_num_threads(1)
    initial=BASE/'local_geometry_continuous_reward_ppo_20260916/policy_18645.pt'
    pool=BASE/'local_geometry_full_training_audit_20260916/training_pool'
    probe=BASE/'exit_action_credit_probe_20260916'
    stage=register('exit_local_credit_parameter_learning_20260916',
        [__file__,initial,BASE/'local_geometry_feedback_policy.py',BASE/'frame_feedback_policy.py',
         BASE/'reflection_policy.py',BASE/'bounded_shape_return.py',BASE/'complete_corner_metrics.py',
         BASE/'probe_coordinated_suffix.py',probe/'receipt.json',probe/'policy_direction_audit.json',
         pool/'summary.json',BASE/'learning_preparation_v4/paths.json'],
        dict(new_probe_decisions=0,training_evaluation_decisions=16000,development_decisions=1500,
             physics_ticks=370000,wall_s=300,parameter_count=291,ridge=.001,parameter_steps=[.25,.5,1.],
             maximum_latent_change_on_saved_support=[.025,.3,.3]),
        'A bounded alternative credit-learning mechanism, not PPO or a controlled comparison to PPO. '
        'Use existing complete-suffix central return sensitivities at16 training states to derive one '
        'linear actor-output-layer update. Precondition with path-balanced hidden-feature covariance '
        'over all64 original training trajectories plus0.001 ridge. Scale to at most one registered '
        'latent probe amplitude on all saved states and query states, then evaluate three steps. '
        'No angle rules, probe timings or online search enter deployment. Reuse parent physics and '
        'verify observation reconstruction/actor feature identities. Retain only64/64 complete training '
        'success, entry excursion<=max(parent,0.05mm), median exit reduction>=20%, no increased maximum '
        'exit excursion, no increased full reversal sum, and lower time+unchanged two-edge shape cost. '
        'Write training selection before at most one12-path development evaluation. Otherwise stop this '
        'direction; do not repeatedly refit or extend. No sealed tests or claim of exact differentiable Q.')
    started=time.perf_counter();costs=dict(saved_prefix_geometry_ticks=0,evaluation_actor_calls=0,evaluation_physics_ticks=0)
    saved=torch.load(initial,map_location='cpu',weights_only=False)
    policy=LocalGeometryFeedbackPolicy(mask_endpoint=True);policy.load_state_dict(saved['policy'])
    observations=[];weights=[]
    for i in range(64):
        with np.load(pool/f'{i}.npz') as z:ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
        obs=[];env=reconstruct(ref,rows,actions,(len(actions)-1)*20,costs,observations=obs)
        obs.append(env.observation());obs=np.asarray(obs)
        for x,held in zip(obs,actions):assert np.array_equal(env.decode_action(policy.act(x,True)[0]),held)
        observations.append(obs);weights.extend([1./(64*len(obs))]*len(obs))
    observations=np.concatenate(observations);weights=np.asarray(weights)
    features=last_layer_features(policy,observations)
    base_weights=np.column_stack([saved['policy']['actor.4.weight'].numpy(),saved['policy']['actor.4.bias'].numpy()])
    with torch.no_grad():
        predicted=policy.distribution(torch.tensor(observations)).mean.numpy()
        feature_error=float(np.max(abs(np.einsum('nhf,hf->nh',features,base_weights)-predicted)))
    assert feature_error<1e-6
    query=read(probe/'policy_direction_audit.json')['records'];raw=read(probe/'results.json')['records']
    qobs=[];sensitivities=[]
    for row in query:
        match=next(r for r in raw if r['path']==row['path'] and r['anchor']==row['anchor'])
        with np.load(probe/match['trace']) as z:qobs.append(z['observation'][0])
        sensitivities.append(row['finite_offset_return_sensitivity'])
    qobs=np.asarray(qobs);sensitivities=np.asarray(sensitivities)
    qf=last_layer_features(policy,qobs);scale=np.array([.025,.3,.3]);direction=[]
    for head in range(3):
        f=features[:,head];fisher=f.T@(weights[:,None]*f)+.001*np.eye(f.shape[1])
        gradient=np.mean(qf[:,head]*(sensitivities[:,head]*scale[head])[:,None],axis=0)
        direction.append(np.linalg.solve(fisher,gradient))
    direction=np.asarray(direction)
    support=np.concatenate([features,qf]);maximum=np.max(abs(np.einsum('nhf,hf->nh',support,direction)))
    direction/=max(maximum,1e-12)
    latent_direction=direction*scale[:,None]
    normalized_changes=np.einsum('nhf,hf->nh',support,direction)
    assert abs(normalized_changes).max()<=1+1e-10
    predicted_gain=float(np.mean(np.sum(np.einsum('nhf,hf->nh',qf,latent_direction)*sensitivities,axis=1)))
    assert predicted_gain>0
    np.savez_compressed(stage/'training_support.npz',observation=observations,weight=weights,
                        query_observation=qobs,finite_return_sensitivity=sensitivities,parameter_direction=latent_direction)
    write_json(stage/'direction.json',dict(feature_identity_max_error=feature_error,baseline_decode_exact=True,
        registered_support_states=len(observations),local_linearized_mean_gain=predicted_gain,
        max_normalized_latent_change=float(abs(normalized_changes).max()),
        limit='Trust region only certifies saved support, not unseen states; local return sensitivity comes from finite60ms offsets.'))
    baseline=summary(pool,read(pool/'summary.json')['results'])
    paths=read(BASE/'learning_preparation_v4/paths.json');training=[np.asarray(p) for p in paths['training']]
    records=[]
    for step in (.25,.5,1.):
        policy.load_state_dict(saved['policy'])
        with torch.no_grad():
            policy.actor[-1].weight.add_(torch.tensor(step*latent_direction[:,:-1],dtype=torch.float32))
            policy.actor[-1].bias.add_(torch.tensor(step*latent_direction[:,-1],dtype=torch.float32))
        name=f'candidate_{step:g}.pt'
        save_torch(stage/name,dict(policy=policy.state_dict(),observation_dim=72,architecture='local_geometry_feedback',
             mask_endpoint=True,frame='heading',history_independent=True,parent_sha256=sha(initial),
             learning_method='finite_suffix_return_natural_parameter_direction',parameter_step=step))
        tag=f'training_{step:g}'
        result=evaluate(policy,training,stage,tag,None,costs,started+300,environment_factory)
        row=summary(stage/tag,result);row.update(step=step,model=name)
        row['eligible']=bool(row['successes']==64 and row['max_entry_mm']<=max(.05,baseline['max_entry_mm']) and
            row['median_exit_mm']<=.8*baseline['median_exit_mm'] and row['max_exit_mm']<=baseline['max_exit_mm'] and
            row['full_reverse_sum_deg']<=baseline['full_reverse_sum_deg'] and row['time_plus_shape_s']<baseline['time_plus_shape_s'])
        records.append(row)
        write_json(stage/'training_results.json',dict(baseline=baseline,candidates=records,costs=costs))
        print('LOCAL_CREDIT',step,{k:row[k] for k in ['successes','time_plus_shape_s','median_exit_mm','max_exit_mm','max_entry_mm','full_reverse_sum_deg','eligible']},flush=True)
        assert costs['evaluation_actor_calls']<=16000
    eligible=[r for r in records if r['eligible']]
    selected=min(eligible,key=lambda r:r['time_plus_shape_s']) if eligible else None
    write_json(stage/'selection.json',dict(selected=selected,training_only=True,goal_complete=False))
    development=None
    if selected:
        policy.load_state_dict(torch.load(stage/selected['model'],map_location='cpu',weights_only=False)['policy'])
        before=costs['evaluation_actor_calls']
        result=evaluate(policy,[np.asarray(p) for p in paths['validation']],stage,'development',None,costs,started+300,environment_factory)
        development=summary(stage/'development',result)
        assert costs['evaluation_actor_calls']-before<=1500
    assert costs['evaluation_physics_ticks']<=370000
    write_json(stage/'results.json',dict(selected=selected,development=development,costs=costs,goal_complete=False))
    receipt(stage,dict(status='completed_bounded_local_return_parameter_learning',costs=costs,
                       selected_model=selected['model'] if selected else None,wall_s=time.perf_counter()-started))


if __name__=='__main__':main()
