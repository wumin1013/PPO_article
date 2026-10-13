"""One directed steering-amplitude contrast after the completed factorial probe."""
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from joint_policy import JointPolicy
from validate_actions import difference_audit
from probe_coordinated_suffix import read, reconstruct, encode, shape
from study_io import BASE,register,receipt,write_json,sha


def main():
    prior=BASE/'coordinated_suffix_probe_20260916'
    source=BASE/'ar_conditional_joint_20260916';traces=source/'final_validation'
    sources=[Path(__file__),BASE/'probe_coordinated_suffix.py',prior/'receipt.json',prior/'results.json',source/'final.pt',source/'receipt.json']
    sources += [traces/f'{i}.npz' for i in (1,2,3)]
    stage=register('inner_approach_amplitude_probe_20260916',sources,
        dict(new_suffixes=3,steering_offset_degrees=8,feed_offset_mm_s=5,preview_offset_mm=2,
             intervention_decisions=8,physics_ticks=15000,actor_calls=753,prefix_geometry_ticks=3000,wall_s=180,training_updates=0),
        'The +2-degree factorial improved time but all approach medians remained outside. Test only +8-degree steering in the same +5 feed/+2 D combination at the same three anchors; no grid search, reused baselines and no training.')
    costs=dict(new_suffixes=0,new_physics_ticks=0,actor_calls=0,saved_prefix_geometry_ticks=0,training_updates=0)
    started=time.perf_counter();results=[];torch.set_num_threads(1)
    incoming=torch.load(source/'final.pt',map_location='cpu',weights_only=False)
    policy=JointPolicy(incoming['observation_dim'],mode='joint');policy.load_state_dict(incoming['policy'])
    for g in read(prior/'results.json')['groups']:
        i=g['path']
        with np.load(traces/f'{i}.npz') as data:
            ref,rows,actions=(data[k].copy() for k in ('reference','rows','actions'))
        frame=g['anchor_tick'];env=reconstruct(ref,rows,actions,frame,costs);chunks=[];helds=[];count=0
        latent,_,_=policy.act(env.observation(),deterministic=True);costs['actor_calls']+=1
        assert np.array_equal(env.decode_action(latent),actions[frame//20])
        while not env.done:
            assert time.perf_counter()-started<180 and costs['actor_calls']<753 and costs['new_physics_ticks']<12880
            latent,_,_=policy.act(env.observation(),deterministic=True);costs['actor_calls']+=1
            if count<8:latent=encode(env.decode_action(latent)+np.array([math.radians(8),5.,2.]))
            _,_,_,info=env.step(latent);chunks.append(info['rows']);helds.append(info['held_action']);count+=1
            costs['new_physics_ticks']+=len(info['rows'])
        full=np.vstack([rows[:frame],*chunks]);helds=np.vstack([actions[:frame//20],helds]);audit=difference_audit(ref[0],full)
        success=bool(env.success and audit['kinematic_pass'] and audit['rest_pass']);costs['new_suffixes']+=1
        old=next(r for r in g['records'] if r['bits']==[1,1,1]);baseline=g['records'][0]
        result=dict(path=i,angle_deg=g['angle_deg'],success=success,reason=env.reason,
            completion_time_s=env.ticks*.001 if success else None,
            delta_from_original_s=env.ticks*.001-baseline['completion_time_s'] if success else None,
            delta_from_two_degrees_s=env.ticks*.001-old['completion_time_s'] if success else None,
            geometry=env.tracker.snapshot(1.25),audit=audit,diagnostics=shape(ref,full,helds))
        results.append(result);np.savez_compressed(stage/f'{i}.npz',reference=ref,rows=full,actions=helds)
        write_json(stage/'partial_results.json',dict(results=results,costs=costs))
    write_json(stage/'results.json',dict(results=results,costs=costs,
        boundary='Three finite counterfactual suffixes, one amplitude chosen before execution; no learned inner approach, optimality or generalization claim.'))
    receipt(stage,dict(status='completed_directed_inner_approach_probe',**costs,wall_s=time.perf_counter()-started))
    print(json.dumps(results,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
