"""Locate shared action-credit gradients before paying for more branches."""
import math
import numpy as np
import torch
from conditional_feedback_policy import ConditionalLocalGeometryPolicy
from replay_preswing_learning_origin import flat_gradient,query_targets
from probe_coordinated_suffix import reconstruct
from search_complete_return import read
from study_io import BASE,register,receipt,write_json


def main():
    torch.set_num_threads(1)
    source=BASE/'local_geometry_continuous_reward_ppo_20260916'
    branch=BASE/'common_future_action_credit_learning_20260917'
    revised=BASE/'common_future_credit_objective_interaction_20260917'
    replay=BASE/'preswing_learning_origin_replay_20260916'
    stage=register('common_future_head_credit_gradient_audit_20260917',
        [__file__,branch/'receipt.json',branch/'advantages.npz',revised/'receipt.json',
         revised/'counterfactual_labels.json',replay/'before_update_1.pt',
         BASE/'conditional_feedback_policy.py',BASE/'replay_preswing_learning_origin.py'],
        dict(new_physics_ticks=0,new_decisions=0,optimizer_updates=0,wall_s=60),
        'Saved-data attribution for the unsuccessful common-future actor update. Split the '
        'initial score gradient by original path and Gaussian action head; contract each '
        'with the signed combined entry-target gradient at the same8 historical queries. '
        'These are infinitesimal score-gradient directions, not Adam or physical-shape '
        'causal effects. Locate whether shared joint-action advantages still put outward '
        'pressure through steering, feed or preview. No additional training/probes here.')
    before=torch.load(replay/'before_update_1.pt',map_location='cpu',weights_only=False)
    policy=ConditionalLocalGeometryPolicy(72,mask_endpoint=True);policy.load_state_dict(before['policy'])
    metadata=read(source/'episode_results.json')['episodes'];selected=[metadata[i] for i in before['episode_numbers']]
    loaded=[]
    for m in selected:
        with np.load(source/'episodes'/m['trace']) as z:loaded.append({k:z[k].copy() for k in z.files})
    queries=[];costs=dict(saved_prefix_geometry_ticks=0)
    for path,anchor in zip(before['query_paths'],before['query_anchors']):
        with np.load(BASE/'exit_local_credit_parameter_learning_20260916/training_1'/f'{path}.npz') as z:
            ref,rows,actions=[z[k] for k in ('reference','rows','actions')]
        env=reconstruct(ref,rows,actions,anchor*20,costs);u,v=np.diff(ref,axis=0)
        queries.append(dict(path=path,env=env,observation=env.observation(),
            sign=float(np.sign(u[0]*v[1]-u[1]*v[0])),incoming_heading=math.atan2(u[1],u[0])))
    parameters=policy.actor_parameters();target,_=query_targets(policy,queries);qgrad=flat_gradient(target.mean(),parameters)
    tensors={k:torch.as_tensor(np.concatenate([d[k] for d in loaded]),dtype=torch.bool if k=='context_live' else torch.float32)
             for k in ('observation','latent','previous_observation','previous_latent','context_live')}
    dist=policy.conditional(tensors['observation'],tensors['previous_observation'],tensors['previous_latent'],tensors['context_live'])
    logp=dist.log_prob(tensors['latent'])
    ids=np.concatenate([np.full(len(d['reward']),m['path']) for m,d in zip(selected,loaded)])
    newlabels=read(revised/'counterfactual_labels.json')['labels'];records=[]
    for arm in ('original_objective_action_credit','revised_objective_action_credit'):
        advantages=[]
        with np.load(branch/'advantages.npz') as z:
            for k in range(0,len(loaded),4):
                totals=[];advs=[]
                for m,d in zip(selected[k:k+4],loaded[k:k+4]):
                    if arm.startswith('original'):
                        totals.append(float(d['reward'].sum()));advs.append(z[f'episode_{m["episode"]}'].copy())
                    else:
                        rows=[r for r in newlabels if r['episode']==m['episode']]
                        totals.append(rows[0]['observed_return']);advs.append(np.array([r['advantage'] for r in rows]))
                scale=max(float(np.std(totals)),.02);advantages.extend([a/scale for a in advs])
        a=torch.tensor(np.concatenate(advantages).astype(np.float32));a=(a-a.mean())/(a.std(unbiased=False)+1e-8)
        contributions=[]
        for path in sorted(set(ids)):
            mask=torch.tensor(ids==path)
            for head in range(3):
                g=flat_gradient((logp[:,head]*a*mask).mean(),parameters)
                contributions.append(dict(path=int(path),head=head,target_direction=float(g@qgrad),score_gradient_norm=float(g.norm())))
        total=flat_gradient((logp.sum(-1)*a).mean(),parameters)
        direct=float(total@qgrad);parts=sum(r['target_direction'] for r in contributions)
        assert abs(direct-parts)<1e-4*max(1.,abs(direct))
        records.append(dict(arm=arm,directional_derivative=direct,contributions=contributions))
        print('HEAD_CREDIT',arm,'total',direct,'parts',contributions,flush=True)
    write_json(stage/'results.json',dict(records=records,costs=costs,goal_complete=False))
    receipt(stage,dict(status='completed_common_future_head_gradient_attribution',new_physics_ticks=0,new_decisions=0))


if __name__=='__main__':main()
