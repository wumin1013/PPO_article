"""Matched reward-extent targets from the same coordinated training suffixes."""
import time
import copy
import hashlib
import numpy as np
import torch
from local_geometry_feedback_policy import LocalGeometryFeedbackPolicy
from learn_exit_local_credit import last_layer_features,summary
from probe_coordinated_suffix import reconstruct
from train_pilot import evaluate,save_torch
from joint_runtime_v5 import environment_factory
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def policy_digest(policy):
    h=hashlib.sha256()
    for name,tensor in sorted(policy.state_dict().items()):
        h.update(name.encode('utf-8'));h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def evaluate_unique(policy,paths,stage,tag,costs,deadline,cache):
    geometry=hashlib.sha256(b''.join(np.asarray(p).tobytes() for p in paths)).hexdigest()
    key=(policy_digest(policy),geometry)
    if key in cache:
        return dict(copy.deepcopy(cache[key]),evaluation_reused=True)
    result=evaluate(policy,paths,stage,tag,None,costs,deadline,environment_factory)
    row=dict(summary(stage/tag,result),evaluation_tag=tag,evaluation_reused=False)
    cache[key]=copy.deepcopy(row)
    return row


def main():
    torch.set_num_threads(1)
    source=BASE/'exit_local_credit_parameter_learning_20260916'
    pool=source/'training_1';initial=source/'candidate_1.pt'
    probe=BASE/'entry_action_interaction_probe_20260916'
    stage=register('matched_entry_interaction_feedback_20260916',
        [__file__,initial,source/'receipt.json',probe/'receipt.json',probe/'results.json',
         BASE/'learn_exit_local_credit.py',BASE/'local_geometry_feedback_policy.py',BASE/'bounded_shape_return.py',
         BASE/'complete_corner_metrics.py',BASE/'learning_preparation_v4/paths.json'],
        dict(training_evaluation_decisions=19000,development_decisions=3000,physics_ticks=500000,
             wall_s=500,arms=['window','full'],parameter_steps=[.5,1.],ridge=.001,
             reference_states='all64 new-parent saved trajectories',block_weights=[1/3,1/3,1/3],
             parameter_count=291,latent_scales=[.025,.3,.3]),
        'Matched finite-counterfactual fitted feedback experiment. Both arms share new parent actor, '
        'same112 completed factorial suffixes, same all64-path reference-state support, linear '
        'reflected actor-output features, ridge and step opportunities. Only target selection differs: '
        'existing local energy/reversal extent versus complete-path extent; both retain the same '
        'two-edge positional term, weights, allowances, bounded cost and time/failure return. '
        'Choose the best complete branch at each anchor only if its return gain exceeds0.005s, '
        'then learn its three latent offsets on intervention observations while matching parent on '
        'original states and branch continuations. No labels from failures, no live trajectories or '
        'anchor rules at deployment. This learning method differs from prior PPO/natural-direction '
        'experiments; no single-factor causal claim across those methods. Require64/64, maximum '
        'entry<=0.05mm, median exit<=80% parent, maximum exit no worse, full reversal sum no worse, '
        'and lower complete time+current two-edge cost before selecting one development candidate '
        'per arm. Stop this fit if none qualifies; do not extend unchanged.')
    started=time.perf_counter();costs=dict(saved_prefix_geometry_ticks=0,evaluation_actor_calls=0,evaluation_physics_ticks=0)
    saved=torch.load(initial,map_location='cpu',weights_only=False)
    policy=LocalGeometryFeedbackPolicy(mask_endpoint=True);policy.load_state_dict(saved['policy'])
    original=[];original_weights=[]
    for i in range(64):
        with np.load(pool/f'{i}.npz') as z:ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
        obs=[];env=reconstruct(ref,rows,actions,(len(actions)-1)*20,costs,observations=obs)
        obs.append(env.observation());obs=np.asarray(obs)
        for x,action in zip(obs,actions):assert np.array_equal(env.decode_action(policy.act(x,True)[0]),action)
        original.append(obs);original_weights.extend([1/(64*len(obs))]*len(obs))
    original=np.concatenate(original);original_weights=np.asarray(original_weights)
    data=read(probe/'results.json');baseline=summary(pool,read(pool/'summary.json')['results'])
    paths=read(BASE/'learning_preparation_v4/paths.json');training=[np.asarray(p) for p in paths['training']]
    records=[];targets={};scale=np.array([.025,.3,.3]);evaluation_cache={}
    for extent in ('window','full'):
        chosen=[]
        for base in data['baselines']:
            for anchor in base['anchors']:
                candidates=[r for r in data['records'] if r['path']==base['path'] and r['anchor']==anchor and r['success'] and r[f'return_{extent}']-base[f'return_{extent}']>.005]
                if candidates:chosen.append(max(candidates,key=lambda r:r[f'return_{extent}']))
        assert chosen
        positive=[];target=[];tail=[]
        for row in chosen:
            with np.load(probe/row['trace']) as z:
                n=int(z['intervention_decisions']);x=z['observation'][:n]
                with torch.no_grad():mean=policy.distribution(torch.tensor(x)).mean.numpy()
                positive.append(x);target.append((z['latent'][:n]-mean)/scale);tail.append(z['observation'][n:])
        positive=np.concatenate(positive);target=np.concatenate(target);tail=np.concatenate(tail)
        x=np.concatenate([original,positive,tail])
        y=np.concatenate([np.zeros((len(original),3)),target,np.zeros((len(tail),3))])
        w=np.r_[original_weights,np.full(len(positive),1/len(positive)),np.full(len(tail),1/len(tail))]/3
        f=last_layer_features(policy,x);weights=[]
        for head in range(3):
            a=f[:,head];weights.append(np.linalg.solve(a.T@(w[:,None]*a)+.001*np.eye(a.shape[1]),a.T@(w*y[:,head])))
        weights=np.asarray(weights);unscaled_max=float(abs(np.einsum('nhf,hf->nh',f,weights)).max())
        weights/=max(1.,unscaled_max);delta=weights*scale[:,None]
        np.savez_compressed(stage/f'{extent}_training_data.npz',observation=x,target_residual=y,weight=w,parameter_delta=delta)
        targets[extent]=dict(chosen=[dict(path=r['path'],anchor=r['anchor'],bits=r['bits'],trace=r['trace']) for r in chosen],
                             positive_states=len(positive),tail_states=len(tail),unscaled_max=unscaled_max)
        write_json(stage/'targets.json',targets)
        for step in (.5,1.):
            policy.load_state_dict(saved['policy'])
            with torch.no_grad():
                policy.actor[-1].weight.add_(torch.tensor(step*delta[:,:-1],dtype=torch.float32))
                policy.actor[-1].bias.add_(torch.tensor(step*delta[:,-1],dtype=torch.float32))
            model=f'{extent}_{step:g}.pt'
            save_torch(stage/model,dict(policy=policy.state_dict(),observation_dim=72,architecture='local_geometry_feedback',
                mask_endpoint=True,frame='heading',history_independent=True,learning_method='matched_factorial_suffix_feedback_fit',
                extent=extent,step=step,parent_sha256=sha(initial)))
            tag=f'{extent}_training_{step:g}'
            row=evaluate_unique(policy,training,stage,tag,costs,started+500,evaluation_cache)
            row.update(extent=extent,step=step,model=model)
            row['eligible']=bool(row['successes']==64 and row['max_entry_mm']<=.05 and
                row['median_exit_mm']<=.8*baseline['median_exit_mm'] and row['max_exit_mm']<=baseline['max_exit_mm'] and
                row['full_reverse_sum_deg']<=baseline['full_reverse_sum_deg'] and row['time_plus_shape_s']<baseline['time_plus_shape_s'])
            records.append(row);write_json(stage/'training_results.json',dict(baseline=baseline,candidates=records,costs=costs))
            print('MATCHED_FACTORIAL',extent,step,{k:row[k] for k in ('successes','median_exit_mm','max_exit_mm','max_entry_mm','full_reverse_sum_deg','time_plus_shape_s','eligible')},flush=True)
            assert costs['evaluation_actor_calls']<=19000
        policy.load_state_dict(saved['policy'])
    selected=[]
    for extent in ('window','full'):
        eligible=[r for r in records if r['extent']==extent and r['eligible']]
        if eligible:selected.append(min(eligible,key=lambda r:r['time_plus_shape_s']))
    write_json(stage/'selection.json',dict(selected=selected,training_only=True,goal_complete=False))
    development=[];before=costs['evaluation_actor_calls']
    for row in selected:
        policy.load_state_dict(torch.load(stage/row['model'],map_location='cpu',weights_only=False)['policy'])
        tag=row['extent']+'_development'
        reviewed=evaluate_unique(policy,[np.asarray(p) for p in paths['validation']],stage,tag,costs,started+500,evaluation_cache)
        development.append(dict(extent=row['extent'],model=row['model'],**reviewed))
    assert costs['evaluation_actor_calls']-before<=3000 and costs['evaluation_physics_ticks']<=500000
    write_json(stage/'results.json',dict(selected=selected,development=development,costs=costs,goal_complete=False))
    receipt(stage,dict(status='completed_matched_factorial_feedback_fit',costs=costs,
                       selected_models=[r['model'] for r in selected],wall_s=time.perf_counter()-started))


if __name__=='__main__':main()
