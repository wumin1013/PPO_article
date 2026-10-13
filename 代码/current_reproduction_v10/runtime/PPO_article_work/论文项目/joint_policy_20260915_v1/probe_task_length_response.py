"""Actor responses to distant path length with identical local control inputs."""
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from joint_runtime_v5 import environment_factory
from joint_policy import JointPolicy
from smooth_policy_candidate import SmoothExplorationPolicy
from study_io import BASE, register, receipt, write_json, sha


def main():
    model_names=['joint_independent_refinement_v7','fixed3_gsde_refinement_v7']
    stage=register('task_length_actor_probe_20260916',
        [Path(__file__),BASE/'joint_policy.py',BASE/'smooth_policy_candidate.py',
         BASE/'joint_runtime.py',BASE/'joint_runtime_v2.py',BASE/'joint_runtime_v3.py',BASE/'joint_runtime_v5.py',
         BASE/'feed_update_context_20260916/receipt.json']+
        [BASE/n/f for n in model_names for f in ('final.pt','receipt.json','selection.json')],
        dict(environment_constructions=120,model_loads=2,actor_batch_forwards=2,physics_ticks=0,training_decisions=0,wall_s=120),
        'Separate distant length/global task inputs from identical local geometry and dynamics. Physically specified synthetic states, not executed successful trajectories, not a zero-input ablation.')
    torch.set_num_threads(1)
    observations=[];metadata=[];started=time.perf_counter()
    global_slots=[13,14,15,16,17,19]
    keep=np.array([i for i in range(72) if i not in global_slots])
    for angle in (0,50,95,125,150):
        theta=math.radians(angle)
        for x in (-30.,-15.,-5.):
            for speed in (40.,80.):
                baseline=None
                for tail in (60.,600.,6000.,30000.):
                    ref=np.array([[-60.,0.],[0.,0.],[tail*math.cos(theta),tail*math.sin(theta)]])
                    env=environment_factory(ref)
                    env.position=np.array([x,0.]);env.velocity=np.array([speed,0.])
                    env.acceleration=np.zeros(2);env.heading=0.;env.held=np.array([0.,speed,3.])
                    env.ticks=1000;env.decisions=50
                    obs=env.observation()
                    if baseline is None:baseline=obs
                    assert np.allclose(obs[keep],baseline[keep],rtol=0,atol=1e-7)
                    observations.append(obs)
                    metadata.append(dict(angle=angle,x=x,speed=speed,tail=tail,global_observation=obs[global_slots].tolist()))
    batch=torch.as_tensor(np.array(observations),dtype=torch.float32)
    records=[]
    for name in model_names:
        checkpoint=BASE/name/'final.pt';digest=sha(checkpoint)
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        cls=SmoothExplorationPolicy if state['sampler']=='gsde8' else JointPolicy
        policy=cls(state['observation_dim'],mode=state['mode'])
        policy.load_state_dict(state['policy'])
        with torch.no_grad():mean=policy.distribution(batch).mean.numpy()
        outputs=np.column_stack((90*np.tanh(mean[:,0]),50*(1+np.tanh(mean[:,1])),
                    np.full(len(mean),3.) if state['mode']=='fixed' else 6.4+5.6*np.tanh(mean[:,2])))
        deltas=outputs[3::4]-outputs[0::4]
        records.append(dict(stage=name,model_sha256=digest,outputs=outputs.tolist(),
            long_minus_short=dict(steering_degrees_range=[float(deltas[:,0].min()),float(deltas[:,0].max())],
                feed_mm_s_range=[float(deltas[:,1].min()),float(deltas[:,1].max())],
                feed_mean=float(deltas[:,1].mean()),preview_mm_range=[float(deltas[:,2].min()),float(deltas[:,2].max())])))
        assert sha(checkpoint)==digest
    result=dict(states=metadata,groups=records,global_slots=global_slots,
        other_66_inputs_identical_within_1e7=True,scope='Tail lengths change true task duration and global features together, while local 30-mm geometry, dynamics and control history match. Synthetic queries show sensitivity only; no optimal-action, feasibility, or generalization claim.',
        wall_s=time.perf_counter()-started,physics_ticks=0,training_decisions=0,goal_complete=False)
    write_json(stage/'results.json',result)
    receipt(stage,dict(status='completed_task_length_actor_probe',environment_constructions=len(observations),
        model_loads=2,actor_batch_forwards=2,physics_ticks=0,training_decisions=0,wall_s=result['wall_s']))
    print(json.dumps([{k:v for k,v in r.items() if k!='outputs'} for r in records],ensure_ascii=False,indent=2))


if __name__=='__main__':main()
