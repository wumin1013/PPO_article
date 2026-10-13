"""Bounded post-gate feed-only interventions; no learned-policy claim."""
import time
import numpy as np
import torch
from reflection_policy import ReflectionPolicy
from probe_coordinated_suffix import reconstruct,encode,read
from joint_runtime_v5 import environment_factory
from isolated_corner_shape import corner_shape
from geometric_turn_metrics import geometric_turn_metrics
from validate_actions import difference_audit
from study_io import BASE,register,receipt,write_json,sha


def main():
    source=BASE/'balanced_shape_time_learning_20260916';traces=source/'validation_5';model=source/'best_generation_5.pt'
    stage=register('gate_feed_jump_probe_20260916',[__file__,model,source/'receipt.json',
        BASE/'gate_observation_effect_audit_20260916/receipt.json',BASE/'probe_coordinated_suffix.py',
        BASE/'geometric_turn_metrics.py']+[traces/f'{i}.npz' for i in (1,2,3)],
        dict(new_suffixes=6,reused_baselines=3,actor_calls=1200,physics_ticks=24000,wall_s=120,training_updates=0,
             intervention='Keep pre-gate feed for 3 or 6 decisions starting at first decision after gate; learned steering and D unchanged.'),
        'Test causal effect of the observed post-gate feed jump on spatial concentration and exit reversal. No training, no deployment; original physical prefix reused and verified exactly.')
    torch.set_num_threads(1);saved=torch.load(model,map_location='cpu',weights_only=False)
    policy=ReflectionPolicy(saved['observation_dim']);policy.load_state_dict(saved['policy'])
    costs=dict(actor_calls=0,new_physics_ticks=0,saved_prefix_geometry_ticks=0,training_updates=0)
    results=[];started=time.perf_counter();original=read(traces/'summary.json')['results']
    for index in (1,2,3):
        with np.load(traces/f'{index}.npz') as d:ref,rows,actions=[d[k].copy() for k in ('reference','rows','actions')]
        tracker=environment_factory(ref).tracker
        for j,row in enumerate(rows):
            tracker.update(row[:2])
            if tracker.frontier:
                gate_tick=j+1;break
        frame=((gate_tick+19)//20)*20
        held_feed=float(actions[(gate_tick-1)//20,1])
        env=reconstruct(ref,rows,actions,frame,costs);state=env.state()
        latent=policy.act(env.observation(),deterministic=True)[0];costs['actor_calls']+=1
        assert np.array_equal(env.decode_action(latent),actions[frame//20])
        baseline=dict(path=index,hold_decisions=0,reused=True,success=original[index]['success'],
            time_s=original[index]['completion_time_s'],shape=corner_shape(ref,rows,actions),spatial=geometric_turn_metrics(ref,rows))
        results.append(baseline)
        for count_limit in (3,6):
            env.restore(state);chunks=[];held=[];count=0
            while not env.done:
                assert time.perf_counter()-started<120
                assert costs['actor_calls']<1200 and costs['new_physics_ticks']<21800
                latent=policy.act(env.observation(),deterministic=True)[0];costs['actor_calls']+=1
                if count<count_limit:
                    requested=env.decode_action(latent);requested[1]=held_feed;latent=encode(requested)
                _,_,_,info=env.step(latent);chunks.append(info['rows']);held.append(info['held_action']);count+=1
                costs['new_physics_ticks']+=len(info['rows'])
            full=np.vstack([rows[:frame],*chunks]);fullactions=np.vstack([actions[:frame//20],held])
            audit=difference_audit(ref[0],full);success=bool(env.success and audit['kinematic_pass'] and audit['rest_pass'])
            tag=f'{index}_hold{count_limit}'
            np.savez_compressed(stage/f'{tag}.npz',reference=ref,rows=full,actions=fullactions,
                intervention_start_tick=frame,intervention_decisions=count_limit)
            result=dict(path=index,hold_decisions=count_limit,held_feed_mm_s=held_feed,gate_tick=gate_tick,frame=frame,
                success=success,reason=env.reason,time_s=env.ticks*.001 if success else None,elapsed_s=env.ticks*.001,
                audit=audit,geometry=env.tracker.snapshot(1.25),shape=corner_shape(ref,full,fullactions),
                spatial=geometric_turn_metrics(ref,full),spatial_coarse=geometric_turn_metrics(ref,full,.2))
            results.append(result);write_json(stage/'partial_results.json',dict(results=results,costs=costs))
            print(tag,success,result['time_s'],'E',round(result['spatial']['bending_energy_mm_inv'],2),
                'reverse',round(result['shape']['reverse_turning_total_deg'],2),flush=True)
    write_json(stage/'results.json',dict(results=results,costs=costs,goal_complete=False,
        boundary='Feed-only offline counterfactual suffixes. No learned candidate, no optimality or generalization claim.'))
    receipt(stage,dict(status='completed',costs=costs,wall_s=time.perf_counter()-started,model_sha256=sha(model)))


if __name__=='__main__':main()
