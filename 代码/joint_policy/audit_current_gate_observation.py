"""Same-physical-state gate observation audit of the current continuous sensor actor."""
import math
import time
import numpy as np
import torch
from geometry_spec import audit_commands
from continuous_sensor_runtime import environment_factory
from kernel_local_feedback_policy import KernelLocalFeedbackPolicy
from frame_feedback_policy import heading_frame
from search_complete_return import read
from study_io import BASE,register,receipt,write_json


def main():
    torch.set_num_threads(1)
    parent=BASE/'continuous_sensor_matched_function_transfer_20260917'
    selected=next(r for r in read(parent/'results.json')['candidates'] if r['arm']=='updated')
    guide=BASE/'continuous_gate_guidance_mechanism_probe_20260917'
    pool=read(guide/'results.json')['candidates'][0]
    stage=register('current_gate_observation_continuity_audit_20260917',
        [__file__,parent/'updated.pt',guide/'results.json',BASE/'continuous_sensor_runtime.py',
         BASE/'frame_feedback_policy.py',BASE/'joint_runtime.py',BASE/'kernel_local_feedback_policy.py'],
        dict(saved_paths=29,new_physics_ticks=0,actor_calls=600,wall_s=60),
        'The guide target is now continuous across the sampled micro-gate transition, '
        'but the actor observation still uses the original gate-selected progress and '
        'frame. At every actual gate state of the current guide/sensor branch, hold '
        'XY,v,a,heading,held actions and time fixed and compare old/new gate frontier '
        'observations and actor means. Inspect heading-frame encoded differences so '
        'a mere coordinate rotation is not confused with information discontinuity. '
        'This measures direct representation-to-action effects, not a complete '
        'trajectory cause or a learned repair. Use existing raw traces only.')
    start=time.perf_counter();p=KernelLocalFeedbackPolicy(256)
    p.load_state_dict(torch.load(parent/'updated.pt',map_location='cpu',weights_only=False)['policy'])
    records=[]
    for result,trace in zip(pool['results'],pool['traces']):
        with np.load(BASE/trace) as z:ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
        _,events=audit_commands(ref,np.vstack([ref[0],rows[:,:2]]));env=environment_factory(ref)
        for e in events:
            tick=e['command_segment'];r=rows[tick]
            env.position=r[:2].copy();env.velocity=r[2:4].copy();env.acceleration=r[4:6].copy()
            env.heading=math.atan2(r[9],r[8]);env.held=actions[min(tick//20,len(actions)-1)].copy()
            env.ticks=tick+1;env.decisions=tick//20+1
            observations=[];progress=[]
            for frontier in (e['gate'],e['gate']+1):
                env.tracker.frontier=frontier
                observations.append(env.observation());progress.append(env.progress()[0])
            x=torch.tensor(np.asarray(observations));encoded=heading_frame(x)
            with torch.no_grad():means=p.distribution(x).mean.numpy()
            held=np.array([env.decode_action(a) for a in means])
            records.append(dict(path=result['original_path'],gate=e['gate'],tick=tick+1,
                observations=np.asarray(observations).tolist(),progress=progress,
                action_before=held[0].tolist(),action_after=held[1].tolist(),action_jump=(held[1]-held[0]).tolist(),
                heading_encoded_feature_jump=(encoded[1]-encoded[0]).numpy().tolist()))
    assert len(records)*2<=600 and time.perf_counter()-start<60
    write_json(stage/'results.json',dict(records=records,actor_calls=len(records)*2,new_physics_ticks=0,goal_complete=False))
    receipt(stage,dict(status='completed_current_gate_observation_audit',gate_events=len(records),
        actor_calls=len(records)*2,new_physics_ticks=0,wall_s=time.perf_counter()-start))
    for r in records:
        if r['path'] in (3,8,13,15):print('CURRENT_GATE_OBSERVATION_JUMP',r['path'],r['action_jump'],flush=True)


if __name__=='__main__':main()
