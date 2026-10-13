"""Small feed/preview reachability grid with zero steering; diagnostic only."""
import time
import numpy as np
import torch
from reflection_policy import ReflectionPolicy
from probe_coordinated_suffix import reconstruct,encode,read
from geometric_turn_metrics import geometric_turn_metrics
from isolated_corner_shape import corner_shape
from validate_actions import difference_audit
from study_io import BASE,register,receipt,write_json,sha


def main():
    source=BASE/'balanced_shape_time_learning_20260916';model=source/'best_generation_5.pt';file=source/'validation_5/3.npz'
    stage=register('wide_turn_reachability_probe_20260916',[__file__,model,file,source/'receipt.json',
        BASE/'probe_coordinated_suffix.py',BASE/'geometric_turn_metrics.py',BASE/'shape_pareto_candidate_review_20260916/receipt.json'],
        dict(new_suffixes=6,reused_baselines=1,training_updates=0,actor_calls=1200,physics_ticks=24000,wall_s=120,
             steering_deg=0,feeds_mm_s=[30,40],previews_mm=[4,6,8],intervention_decisions=24,trigger='last decision boundary at x approximately -12 mm'),
        'Offline 2x3 feed/preview reachability grid, with steering zero in all six cells. Three controls differ from the policy baseline, so no single-factor baseline attribution. Cell comparisons isolate feed or D with the other fixed. Same original physical constraints. Not learned or deployable evidence.')
    torch.set_num_threads(1);saved=torch.load(model,map_location='cpu',weights_only=False)
    policy=ReflectionPolicy(saved['observation_dim']);policy.load_state_dict(saved['policy'])
    with np.load(file) as d:ref,rows,actions=[d[k].copy() for k in ('reference','rows','actions')]
    crossing=int(np.flatnonzero(rows[:,0]-ref[1,0]>=-12)[0]);frame=((crossing+1)//20)*20
    costs=dict(actor_calls=0,new_physics_ticks=0,saved_prefix_geometry_ticks=0,training_updates=0)
    env=reconstruct(ref,rows,actions,frame,costs);state=env.state()
    assert np.array_equal(env.decode_action(policy.act(env.observation(),True)[0]),actions[frame//20]);costs['actor_calls']+=1
    baseline=dict(reused=True,success=True,time_s=read(source/'validation_5/summary.json')['results'][3]['completion_time_s'],
        shape=corner_shape(ref,rows,actions),spatial=geometric_turn_metrics(ref,rows))
    results=[];started=time.perf_counter()
    for feed in (30.,40.):
        for preview in (4.,6.,8.):
            env.restore(state);chunks=[];held=[];count=0
            while not env.done:
                assert time.perf_counter()-started<120
                assert costs['actor_calls']<1200 and costs['new_physics_ticks']<21800
                latent=encode([0.,feed,preview]) if count<24 else policy.act(env.observation(),True)[0]
                costs['actor_calls']+=1
                _,_,_,info=env.step(latent);chunks.append(info['rows']);held.append(info['held_action']);count+=1
                costs['new_physics_ticks']+=len(info['rows'])
            full=np.vstack([rows[:frame],*chunks]);fullactions=np.vstack([actions[:frame//20],held])
            audit=difference_audit(ref[0],full);success=bool(env.success and audit['kinematic_pass'] and audit['rest_pass'])
            tag=f'feed{int(feed)}_D{int(preview)}'
            np.savez_compressed(stage/f'{tag}.npz',reference=ref,rows=full,actions=fullactions,
                intervention_start_tick=frame,intervention_decisions=24)
            result=dict(feed=feed,preview=preview,success=success,reason=env.reason,elapsed_s=env.ticks*.001,
                time_s=env.ticks*.001 if success else None,audit=audit,geometry=env.tracker.snapshot(1.25),
                shape=corner_shape(ref,full,fullactions),spatial=geometric_turn_metrics(ref,full),
                spatial_coarse=geometric_turn_metrics(ref,full,.2))
            results.append(result);write_json(stage/'partial_results.json',dict(baseline=baseline,results=results,costs=costs))
            print(tag,success,env.reason,'time',result['time_s'],'E',round(result['spatial']['bending_energy_mm_inv'],2),
                'width',round(result['spatial']['middle80_turn_arc_length_mm'],2),'apex',round(result['shape']['apex_distance_mm'],2),flush=True)
    write_json(stage/'results.json',dict(baseline=baseline,results=results,costs=costs,goal_complete=False,
        boundary='Six finite action interventions; no learned policy, no optimality or generalization. Failure prefixes are not counted as fast completions.'))
    receipt(stage,dict(status='completed',costs=costs,wall_s=time.perf_counter()-started))


if __name__=='__main__':main()
