"""Factorial feed/preview suffix intervention on the newly learned inside approach."""
import time
import json
import numpy as np
import torch
from reflection_policy import ReflectionPolicy
from probe_coordinated_suffix import reconstruct, encode, read
from isolated_corner_shape import corner_shape
from validate_actions import difference_audit
from study_io import BASE, register, receipt, sha, write_json


def main():
    source=BASE/'broader_shape_learning_20260916'; traces=source/'validation_4'
    model=source/'best_generation_4.pt'
    assert (source/'receipt.json').exists()
    stage=register('learned_shape_coordination_probe_20260916',
        [__file__,model,source/'receipt.json',source/'generations.json',traces/'summary.json',
         BASE/'reflection_policy.py',BASE/'probe_coordinated_suffix.py',BASE/'isolated_corner_shape.py']+
        [traces/f'{i}.npz' for i in (1,2,3)],
        dict(new_suffixes=9,reused_baselines=3,intervention_decisions=8,
             feed_offset_mm_s=15.,preview_offset_mm=2.,actor_calls=1200,physics_ticks=24000,wall_s=120,
             training_updates=0),
        'At the learned inside approach near x=-10 mm, isolate feed +15, preview +2, and their interaction for eight decisions. No steering override. Saved prefixes/baselines reused; finite reachability/coordination evidence only.')
    torch.set_num_threads(1);incoming=torch.load(model,map_location='cpu',weights_only=False)
    policy=ReflectionPolicy(incoming['observation_dim']);policy.load_state_dict(incoming['policy'])
    costs=dict(actor_calls=0,new_physics_ticks=0,saved_prefix_geometry_ticks=0,training_updates=0)
    started=time.perf_counter();results=[]
    original=read(traces/'summary.json')['results']
    for index in (1,2,3):
        with np.load(traces/f'{index}.npz') as d:
            ref,rows,actions=[d[k].copy() for k in ('reference','rows','actions')]
        crossing=int(np.flatnonzero(rows[:,0]-ref[1,0]>=-10)[0]);frame=((crossing+1)//20)*20
        env=reconstruct(ref,rows,actions,frame,costs);state=env.state()
        latent,_,_=policy.act(env.observation(),deterministic=True);costs['actor_calls']+=1
        assert np.array_equal(env.decode_action(latent),actions[frame//20])
        baseline=dict(path=index,feed=False,preview=False,reused=True,success=original[index]['success'],
            time_s=original[index]['completion_time_s'],shape=corner_shape(ref,rows,actions))
        results.append(baseline)
        for feed,preview in ((True,False),(False,True),(True,True)):
            env.restore(state);chunks=[];held=[];count=0
            while not env.done:
                assert time.perf_counter()-started<120
                assert costs['actor_calls']<1200 and costs['new_physics_ticks']<21800
                latent,_,_=policy.act(env.observation(),deterministic=True);costs['actor_calls']+=1
                if count<8:
                    requested=env.decode_action(latent)+np.array([0.,15.*feed,2.*preview])
                    latent=encode(requested)
                _,_,_,info=env.step(latent);chunks.append(info['rows']);held.append(info['held_action']);count+=1
                costs['new_physics_ticks']+=len(info['rows'])
            full=np.vstack([rows[:frame],*chunks]);fullactions=np.vstack([actions[:frame//20],held])
            audit=difference_audit(ref[0],full)
            success=bool(env.success and audit['kinematic_pass'] and audit['rest_pass'])
            tag=f'{index}_{int(feed)}{int(preview)}'
            np.savez_compressed(stage/f'{tag}.npz',reference=ref,rows=full,actions=fullactions,
                intervention_start_tick=frame,intervention_decisions=8)
            result=dict(path=index,feed=feed,preview=preview,reused=False,success=success,reason=env.reason,
                elapsed_s=env.ticks*.001,time_s=env.ticks*.001 if success else None,
                delta_time_s=env.ticks*.001-baseline['time_s'] if success else None,
                audit=audit,geometry=env.tracker.snapshot(1.25),shape=corner_shape(ref,full,fullactions))
            results.append(result)
            write_json(stage/'partial_results.json',dict(results=results,costs=costs))
            print(tag,success,result['delta_time_s'],
                {k:round(result['shape'][k],3) for k in ('outward_excursion_mm','reverse_turning_total_deg','apex_speed_mm_s','apex_distance_mm')},flush=True)
    write_json(stage/'results.json',dict(results=results,costs=costs,
        boundary='Finite interventions are not a learned deployed policy or proof of optimality; used only to diagnose coordinated action reachability.',goal_complete=False))
    receipt(stage,dict(status='completed_learned_shape_coordination_probe',costs=costs,model_sha256=sha(model),
        new_suffixes=9,wall_s=time.perf_counter()-started,goal_complete=False))


if __name__=='__main__':main()
