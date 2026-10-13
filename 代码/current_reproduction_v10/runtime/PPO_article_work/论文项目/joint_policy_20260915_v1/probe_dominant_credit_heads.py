"""Factor dominant joint-action rescue credits into their three actual heads."""
import itertools
import time
import numpy as np
import torch
from conditional_feedback_policy import ConditionalLocalGeometryPolicy
from probe_coordinated_suffix import reconstruct
from bounded_shape_return import terminal_shape_cost
from validate_actions import difference_audit
from search_complete_return import read
from study_io import BASE,register,receipt,write_json


def main():
    torch.set_num_threads(1)
    source=BASE/'local_geometry_continuous_reward_ppo_20260916'
    branch=BASE/'common_future_action_credit_learning_20260917'
    replay=BASE/'preswing_learning_origin_replay_20260916'
    raw=read(branch/'counterfactual_results.json')['records']
    chosen=[r for r in raw if r['path']==31 and r['counterfactual_success']]
    assert len(chosen)==7 and {r['episode'] for r in chosen}=={1}
    stage=register('dominant_common_future_credit_head_factorial_20260917',
        [__file__,branch/'receipt.json',branch/'common_future_rng.pt',branch/'counterfactual_results.json',
         BASE/'common_future_head_credit_gradient_audit_20260917/results.json',
         replay/'before_update_1.pt',source/'episodes/episode_1.npz',
         BASE/'conditional_feedback_policy.py',BASE/'probe_coordinated_suffix.py',BASE/'bounded_shape_return.py'],
        dict(new_suffixes=42,reused_branches=14,actor_calls=6000,new_physics_ticks=140000,wall_s=120),
        'Resolve joint-action credit attribution at7 path31 states whose conditional-mean '
        'rescues contribute99.65% of that path squared raw action credit and dominate the '
        'outward query-target score gradient. Complete2x2x2 current-innovation head factorial: '
        'for each head use observed innovation or conditional mean, preserve the same future '
        'Gaussian stream and normal AR feedback. Reuse000 observed failure and111 completed '
        'mean branch; run remaining6 combinations at each exact state. This is a targeted '
        'mechanism diagnosis, not an unbiased population estimate or deployed intervention. '
        'Determine whether steering is necessary/sufficient for each rescue, or is being '
        'assigned a shared credit caused by other heads. Every branch continues to actual '
        'completion/failure with unchanged physical constraints and original failure costs. '
        'No policy update, development or sealed testing in this probe.')
    started=time.perf_counter();costs=dict(actor_calls=0,new_physics_ticks=0,saved_prefix_geometry_ticks=0)
    saved=torch.load(replay/'before_update_1.pt',map_location='cpu',weights_only=False)
    p=ConditionalLocalGeometryPolicy(72,mask_endpoint=True);p.load_state_dict(saved['policy'])
    streams=torch.load(branch/'common_future_rng.pt',map_location='cpu',weights_only=False)
    episode_index=streams['episodes'].index(1)
    with np.load(source/'episodes/episode_1.npz') as z:d={k:z[k].copy() for k in z.files}
    ref=d['reference'];rows=d['rows'];actions=d['actions'];original_return=float(d['reward'].sum())
    records=[]
    for base in chosen:
        anchor=base['anchor'];env=reconstruct(ref,rows,actions,anchor*20,costs);state=env.state()
        obs=env.observation()
        with torch.no_grad():mean=p.conditional(torch.tensor(obs),torch.tensor(d['previous_observation'][anchor]),
            torch.tensor(d['previous_latent'][anchor]),torch.tensor(d['context_live'][anchor])).mean.numpy()
        costs['actor_calls']+=1
        records.extend([dict(anchor=anchor,bits=[0,0,0],success=False,reused=True,total_return=original_return),
            dict(anchor=anchor,bits=[1,1,1],success=True,reused=True,total_return=base['counterfactual_return'])])
        for bits in itertools.product((0,1),repeat=3):
            if all(bits) or not any(bits):continue
            env.restore(state);torch.set_rng_state(streams['after'][episode_index][anchor])
            a=np.where(bits,mean,d['latent'][anchor]);p.previous_observation=obs.copy();p.previous_latent=a.copy()
            _,_,_,info=env.step(a);chunks=[info['rows']];held=[info['held_action']]
            costs['new_physics_ticks']+=len(info['rows'])
            while not env.done:
                assert time.perf_counter()-started<120
                assert costs['actor_calls']<6000 and costs['new_physics_ticks']<137880
                a,_,_=p.act(env.observation());costs['actor_calls']+=1
                _,_,_,info=env.step(a);chunks.append(info['rows']);held.append(info['held_action'])
                costs['new_physics_ticks']+=len(info['rows'])
            suffix=np.vstack(chunks);suffix_actions=np.asarray(held)
            full=np.vstack([rows[:anchor*20],suffix]);all_actions=np.vstack([actions[:anchor],suffix_actions])
            audit=difference_audit(ref[0],full);complete=bool(env.success and audit['kinematic_pass'] and audit['rest_pass'])
            shape,_=terminal_shape_cost(ref,full,all_actions,discrete_first_label=False)
            length=float(np.linalg.norm(np.diff(ref,axis=0),axis=1).sum())
            total=-len(full)*.001+length/100-float(shape.sum())
            if not env.success:total-=env.max_ticks*.001+1+max(0.,length-env.progress()[0])/20
            name=f'a{anchor}_'+''.join(map(str,bits))+'.npz'
            np.savez_compressed(stage/name,reference=ref,suffix_rows=suffix,suffix_actions=suffix_actions,
                anchor=anchor,original_episode=1,bits=bits)
            row=dict(anchor=anchor,bits=list(bits),success=complete,reused=False,total_return=total,
                full_time_s=len(full)*.001 if complete else None,reason=env.reason,trace=name,audit=audit)
            records.append(row)
        print('HEAD_FACTORIAL',anchor,[(r['bits'],r['success']) for r in records if r['anchor']==anchor],flush=True)
        write_json(stage/'results.json',dict(records=records,costs=costs,goal_complete=False))
    receipt(stage,dict(status='completed_dominant_head_credit_factorial',costs=costs,
        new_suffixes=42,new_complete=sum(r['success'] for r in records if not r['reused']),wall_s=time.perf_counter()-started))


if __name__=='__main__':main()
