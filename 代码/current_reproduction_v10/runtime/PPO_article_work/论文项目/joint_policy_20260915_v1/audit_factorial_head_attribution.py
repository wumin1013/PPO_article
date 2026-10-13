"""Contract saved factorial head credits with the original entry-target gradient."""
import math
from datetime import datetime, timezone
import numpy as np
import torch
from conditional_feedback_policy import ConditionalLocalGeometryPolicy
from replay_preswing_learning_origin import flat_gradient, query_targets
from probe_coordinated_suffix import reconstruct
from search_complete_return import read
from study_io import BASE, register, receipt, write_json, sha


def main():
    torch.set_num_threads(1)
    source=BASE/'local_geometry_continuous_reward_ppo_20260916'
    branch=BASE/'common_future_action_credit_learning_20260917'
    factorial=BASE/'dominant_common_future_credit_head_factorial_20260917'
    replay=BASE/'preswing_learning_origin_replay_20260916'
    stage=register('factorial_head_credit_attribution_20260917',
        [__file__,factorial/'receipt.json',factorial/'results.json',branch/'advantages.npz',
         replay/'before_update_1.pt',BASE/'audit_common_future_head_credit.py'],
        dict(new_physics_ticks=0,new_decisions=0,optimizer_updates=0,wall_s=60),
        'Reuse the completed7-state factorial. Compare shared versus per-head conditional '
        'mean credit with exactly the original group scale, global centering and denominator. '
        'Contract individual score gradients with the same signed8 entry targets. Selected '
        'states are diagnostic only; no selective-credit algorithm or physical adoption.')
    before=torch.load(replay/'before_update_1.pt',map_location='cpu',weights_only=False)
    p=ConditionalLocalGeometryPolicy(72,mask_endpoint=True);p.load_state_dict(before['policy'])
    selected=[read(source/'episode_results.json')['episodes'][i] for i in before['episode_numbers']]
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
    params=p.actor_parameters();target,_=query_targets(p,queries);qgrad=flat_gradient(target.mean(),params)
    scaled=[];scales=[]
    with np.load(branch/'advantages.npz') as z:
        for k in range(0,len(loaded),4):
            scale=max(float(np.std([d['reward'].sum() for d in loaded[k:k+4]])),.02)
            for m in selected[k:k+4]:scaled.extend(z[f'episode_{m["episode"]}']/scale);scales.append(scale)
    a=torch.tensor(np.asarray(scaled),dtype=torch.float32);center=float(a.mean());spread=float(a.std(unbiased=False))
    d=loaded[1];t={k:torch.tensor(d[k],dtype=torch.bool if k=='context_live' else torch.float32)
                  for k in ('observation','latent','previous_observation','previous_latent','context_live')}
    logp=p.conditional(t['observation'],t['previous_observation'],t['previous_latent'],t['context_live']).log_prob(t['latent'])
    saved=read(factorial/'results.json')['records'];records=[]
    for anchor in sorted({r['anchor'] for r in saved}):
        parts={tuple(r['bits']):r for r in saved if r['anchor']==anchor}
        obs=parts[(0,0,0)]['total_return'];shared=obs-parts[(1,1,1)]['total_return']
        row=dict(anchor=anchor,shared_advantage=shared,heads=[])
        for h in range(3):
            bits=tuple(int(j==h) for j in range(3));headadv=obs-parts[bits]['total_return']
            score=flat_gradient(logp[anchor,h],params);projection=float(score@qgrad)/len(a)
            row['heads'].append(dict(head=h,head_advantage=headadv,only_head_rescues=parts[bits]['success'],
                shared_direction=projection*(shared/scales[1]-center)/(spread+1e-8),
                head_direction=projection*(headadv/scales[1]-center)/(spread+1e-8),
                head_direction_without_center=projection*(headadv/scales[1])/(spread+1e-8)))
        records.append(row)
    summary={str(h):{name:sum(r['heads'][h][name] for r in records)
                    for name in ('shared_direction','head_direction','head_direction_without_center')} for h in range(3)}
    write_json(stage/'results.json',dict(records=records,summary=summary,scale=scales[1],global_center=center,
        global_spread=spread,costs=costs,goal_complete=False))
    receipt(stage,dict(status='completed_factorial_head_attribution',new_physics_ticks=0,new_decisions=0))
    status_path=BASE/'当前执行状态.json';status=read(status_path)
    for name in ('causal_shape_small_segment_check_20260917','common_future_action_credit_learning_20260917',
                 'common_future_credit_objective_interaction_20260917','common_future_head_credit_gradient_audit_20260917',
                 'dominant_common_future_credit_head_factorial_20260917',stage.name):
        r=read(BASE/name/'receipt.json');assert r['registration_sha256']==sha(BASE/name/'registration.json')
        if not any(x['stage']==name for x in status['completed']):
            status['completed'].append(dict(stage=name,receipt_sha256=sha(BASE/name/'receipt.json')))
    status['updated_utc']=datetime.now(timezone.utc).isoformat();status['goal_complete']=False
    status['running']=None;status['active_experiment']=None;status['observed_research_processes']=[]
    status['next_action']='Apply the diagnosed action-credit correction using matched controls; preserve retained strong model and verify physical shape.'
    status['latest_findings'].append('Completed common-future1194 branches and reward-by-credit2x2:all arms8/8 but reversal worsens under common-future shared credit;none adopted. Seven dominant rescues have different head causes and interactions;42 factorial suffixes901calls20511ticks completed. Per-head score attribution is saved in '+stage.name+'.')
    write_json(status_path,status)
    print('SUMMARY',summary,flush=True)
    for r in records:print('STATE',r,flush=True)


if __name__=='__main__':main()
