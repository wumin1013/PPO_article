"""Same-physical-state network counterfactuals at saved ordered-gate events.

Hybrid observations are diagnostics, not physical trajectories or valid-policy
performance claims. They isolate network sensitivity, not training causation.
"""
import math
import numpy as np
import torch
from joint_runtime_v5 import environment_factory
from joint_runtime import wrap
from reflection_policy import ReflectionPolicy
from study_io import BASE,register,receipt,write_json,sha


def main():
    torch.set_num_threads(1)
    model=BASE/'balanced_shape_time_learning_20260916/best_generation_5.pt'
    files=[BASE/'balanced_shape_time_learning_20260916/validation_5'/f'{i}.npz' for i in (1,2,3)]
    stage=register('gate_observation_effect_audit_20260916',[__file__,model]+files,
        dict(physics_ticks=0,maximum_model_queries=40,training_updates=0,wall_s=120),
        'At exact saved physical states compare old/new gate observations, then restore one observation group at a time. Test if frame/geometry change causes an immediate network feed response, without attributing why learning produced it.')
    saved=torch.load(model,map_location='cpu',weights_only=False)
    policy=ReflectionPolicy(saved['observation_dim']);policy.load_state_dict(saved['policy'])
    groups={'local_dynamic':list(range(8)),'held_action':list(range(8,11)),
            'gate_and_endpoint':list(range(11,15)),'progress_and_status':list(range(15,22)),
            'relative_preview_points':list(range(22,42)),'preview_directions':list(range(42,62)),
            'preview_validity':list(range(62,72))}
    records=[];queries=0
    for file in files:
        with np.load(file) as d:ref,rows,actions=d['reference'],d['rows'],d['actions']
        env=environment_factory(ref)
        for i,row in enumerate(rows):
            if row[12]>.5:break
            if i%20==0:env.held=actions[i//20].copy()
            env.position=row[:2].copy();env.velocity=row[2:4].copy();env.acceleration=row[4:6].copy()
            env.heading=math.atan2(row[9],row[8]);env.ticks=i+1
            old_frontier=env.tracker.frontier
            old_obs=env.observation()
            env.tracker.update(env.position)
            if env.tracker.frontier==old_frontier:continue
            new_obs=env.observation()
            def query(obs):
                nonlocal queries
                queries+=1
                latent=policy.act(obs,deterministic=True)[0]
                return env.decode_action(latent)*np.array([180/math.pi,1,1])
            old_action=query(old_obs);new_action=query(new_obs)
            hybrids={}
            for group,indices in groups.items():
                mixed=new_obs.copy();mixed[indices]=old_obs[indices]
                hybrids[group]=dict(action=query(mixed).tolist(),observation_change_l2=float(np.linalg.norm(new_obs[indices]-old_obs[indices])))
            tangent_old=env.corridor.tangents[old_frontier];tangent_new=env.corridor.tangents[min(env.tracker.frontier,len(env.corridor.lengths)-1)]
            records.append(dict(path=file.stem,tick=i+1,position=env.position.tolist(),speed=float(np.linalg.norm(env.velocity)),
                velocity_heading_deg=math.degrees(math.atan2(row[3],row[2])),
                frame_rotation_deg=math.degrees(wrap(math.atan2(tangent_new[1],tangent_new[0])-math.atan2(tangent_old[1],tangent_old[0]))),
                old_action_deg_mm_s_mm=old_action.tolist(),new_action_deg_mm_s_mm=new_action.tolist(),
                feed_delta_at_identical_physical_state=float(new_action[1]-old_action[1]),hybrid_restore=hybrids,
                trace_sha256=sha(file)))
            np.savez_compressed(stage/f'observations_{file.stem}.npz',old=old_obs,new=new_obs)
            print(file.stem,'feed',round(old_action[1],2),'->',round(new_action[1],2),
                  'dynamic_restore',round(hybrids['local_dynamic']['action'][1],2),
                  'geometry_restore',round(hybrids['relative_preview_points']['action'][1],2),flush=True)
    assert queries<=40
    write_json(stage/'results.json',dict(records=records,model_queries=queries,physics_ticks=0,
        boundary='Same-state actor sensitivity to observed gate update is directly measured. Hybrid inputs can be off-manifold; neither performance nor training-origin causation follows. No new physical rollout.',goal_complete=False))
    receipt(stage,dict(status='completed',model_queries=queries,physics_ticks=0,training_updates=0))


if __name__=='__main__':main()
