"""Physical causal test on the exact failed episodes that drove outward learning."""
import itertools
import time
import numpy as np
import torch
from local_geometry_feedback_policy import LocalGeometryFeedbackPolicy
from joint_runtime_v5 import environment_factory
from audit_preswing_mechanism_chain import objective
from validate_actions import difference_audit
from search_complete_return import read
from study_io import BASE,register,receipt,write_json


def main():
    torch.set_num_threads(1)
    source=BASE/'local_geometry_continuous_reward_ppo_20260916'
    model=BASE/'matched_endpoint_representation_learning_20260916/mask_endpoint_bearing_50.pt'
    phase=read(BASE/'preswing_origin_temporal_loss_interventions_20260916/phase_definition.json')
    onsets={r['episode']:r['first_outgoing_preview_decision'] for r in phase['onsets']}
    stage=register('historical_failure_credit_physical_probe_20260916',
        [__file__,model,source/'episodes/episode_16.npz',source/'episodes/episode_17.npz',
         source/'episode_results.json',BASE/'preswing_origin_temporal_loss_interventions_20260916/receipt.json',
         BASE/'preswing_origin_temporal_loss_interventions_20260916/phase_definition.json',
         BASE/'local_geometry_feedback_policy.py',BASE/'audit_preswing_mechanism_chain.py'],
        dict(new_counterfactuals=14,reproduction_checks=2,actor_calls=3000,new_physics_ticks=70000,
             wall_s=150,new_training_decisions=0,late_window_starts_decisions_before_preview=6),
        'Mechanism diagnosis on original failed episodes16/17, whose paired negative '
        'advantages drove outward entry targets in the verified first PPO update. '
        'Freeze the original actor and preserve its saved latent disturbance tape with '
        'closed-loop action z_saved+(mean_new-mean_saved). First reproduce both original '
        'failed trajectories bitwise; these explicit verification costs are counted. '
        'Then2x2x2 factors remove only: early steering disturbance before first outgoing '
        'preview; feed disturbance from120ms before that preview; preview-distance '
        'disturbance over the same late interval. Keep other components untouched. '
        'After the saved failed tape ends, use the identical deterministic original '
        'actor continuation for every surviving branch. Thus results are conditional '
        'causal interventions, not expected policy returns or deployable action rules. '
        'Complete every rescued path or retain true failure; no failure-prefix speed '
        'claim. Test whether late feed/preview corrections can prevent failure while '
        'all original early steering disturbances remain.')
    started=time.perf_counter();costs=dict(actor_calls=0,new_physics_ticks=0,reproduction_physics_ticks=0)
    p=LocalGeometryFeedbackPolicy(mask_endpoint=True)
    p.load_state_dict(torch.load(model,map_location='cpu',weights_only=False)['policy'])
    records=[];reproductions=[]
    for episode in (16,17):
        with np.load(source/'episodes'/f'episode_{episode}.npz') as z:
            ref,rows,actions,observations,latent=[z[k].copy() for k in ('reference','rows','actions','observation','latent')]
        means=np.asarray([p.act(x,True)[0] for x in observations]);costs['actor_calls']+=len(means)
        onset=onsets[episode];late_start=max(0,onset-6)
        for bits in itertools.product((0,1),repeat=3):
            env=environment_factory(ref);chunks=[];held=[];newobs=[];count=0
            while not env.done:
                assert time.perf_counter()-started<150 and costs['actor_calls']<3000
                assert costs['new_physics_ticks']<67880
                x=env.observation();mean=p.act(x,True)[0];costs['actor_calls']+=1
                action=latent[count]+(mean-means[count]) if count<len(latent) else mean.copy()
                if count<len(latent):
                    if bits[0] and count<onset:action[0]=mean[0]
                    if bits[1] and count>=late_start:action[1]=mean[1]
                    if bits[2] and count>=late_start:action[2]=mean[2]
                _,_,_,info=env.step(action);chunks.append(info['rows']);held.append(info['held_action']);newobs.append(x)
                costs['new_physics_ticks']+=len(info['rows']);count+=1
            full=np.vstack(chunks);all_actions=np.asarray(held)
            if not any(bits):
                assert np.array_equal(full,rows) and np.array_equal(all_actions,actions)
                costs['reproduction_physics_ticks']+=len(full)
                reproductions.append(dict(episode=episode,bitwise_rows=True,bitwise_actions=True,
                    reason=env.reason,physics_ticks=len(full)))
                continue
            audit=difference_audit(ref[0],full);success=bool(env.success and audit['kinematic_pass'] and audit['rest_pass'])
            name=f'ep{episode}_'+''.join(map(str,bits))+'.npz'
            np.savez_compressed(stage/name,reference=ref,rows=full,actions=all_actions,
                observations=np.asarray(newobs),original_preview_onset=onset,late_start=late_start)
            row=dict(episode=episode,bits=list(bits),success=success,reason=env.reason,
                full_time_s=len(full)*.001 if success else None,trace=name,audit=audit,
                original_early_steering_preserved=not bool(bits[0]),
                deterministic_tail_decisions=max(0,count-len(latent)),**objective(ref,full,all_actions,env))
            records.append(row)
            print('HISTORICAL_CAUSE',episode,bits,success,env.reason,
                'entry',round(row['shape']['in_edge_outward_mm'],4),
                'time',row['full_time_s'],flush=True)
            write_json(stage/'results.json',dict(records=records,reproductions=reproductions,costs=costs,
                intervention_deployed=False,historical_cause_fully_established=False))
    receipt(stage,dict(status='completed_historical_failed_episode_factorial',costs=costs,
        counterfactuals=len(records),complete_counterfactuals=sum(r['success'] for r in records),
        original_reproduction_checks=len(reproductions),wall_s=time.perf_counter()-started))


if __name__=='__main__':main()
