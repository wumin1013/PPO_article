"""Verify half-step replay continuation against uninterrupted raw commands."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,time
import numpy as np
import torch
BASE=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
from resume_resolution_candidate import replay
from run_natural_resolution import set_resolution,physical_audit
from deep_joint_credit_policy import load_credit_policy
from learned_action_pulse_policy import environment_factory
from arc_cruise_policy import build_policy
from study_io import sha,write_json

torch.set_num_threads(1);dt=.0005;set_resolution(dt);start=time.perf_counter()
stage=OUT/'resolution_replay_cache_verification';stage.mkdir(exist_ok=False)
write_json(stage/'registration.json',dict(kind='software_continuation_verification',wall_budget_s=120,ppo_updates=0,
    execution_dt_s=dt,decision_dt_s=.020,source_sha256=sha(Path(__file__)),replay_source_sha256=sha(OUT/'resume_resolution_candidate.py')))
selected=json.loads((OUT/'joint_actual_context_feedback_training/selected_policy.json').read_text(encoding='utf-8'))
md=torch.load(selected['model'],map_location='cpu',weights_only=False)
reference=np.array([[-50.,0.],[-25.,0.],[0.,0.],[0.,25.],[0.,50.]])
def rollout(policy,env):
    chunks=[];observations=[];actions=[];times=[]
    while not env.done:
        if time.perf_counter()>start+110:raise TimeoutError('Software check budget')
        x=env.observation();latent=policy.act(x,True)[0];observations.append(x);times.append(env.ticks*dt)
        _,_,_,info=env.step(latent);chunks.append(info['rows']);actions.append(info['held_action'])
    return dict(reference=reference,rows=np.vstack(chunks),observations=np.array(observations),actions=np.array(actions),decision_times_s=np.array(times),dt_s=dt)
p=build_policy(load_credit_policy(md),selected['spec']);e=environment_factory(reference);p.bind(e);full=rollout(p,e)
assert e.success and physical_audit(reference,full['rows'],dt)['rest_pass']
n=30;prefix=dict(reference=reference,rows=full['rows'][:40*n],observations=full['observations'][:n],actions=full['actions'][:n],decision_times_s=full['decision_times_s'][:n],dt_s=dt)
p=build_policy(load_credit_policy(md),selected['spec']);e=environment_factory(reference)
short=dict(prefix,rows=prefix['rows'][:800],observations=prefix['observations'][:20],actions=prefix['actions'][:20])
replay(p,e,short,dt,start+110,stage/'cache','same_verified_full_source')
p=build_policy(load_credit_policy(md),selected['spec']);e=environment_factory(reference)
identity=replay(p,e,prefix,dt,start+110,stage/'cache','same_verified_full_source')
assert identity['cached_verified_decisions']==20
tail=rollout(p,e);joined=np.vstack([prefix['rows'],tail['rows']])
assert e.success and joined.shape==full['rows'].shape
maximum=np.max(abs(joined-full['rows']),axis=0)
assert np.array_equal(joined,full['rows']),maximum
np.savez_compressed(stage/'verification.npz',uninterrupted=full['rows'],resumed=joined)
write_json(stage/'receipt.json',dict(status='passed',wall_s=time.perf_counter()-start,identity=identity,raw_column_max_errors=maximum.tolist(),
    complete_time_s=len(joined)*dt,complete_time_equal=True,ppo_updates=0,goal_complete=False))
print('HALF_STEP_CONTINUATION_VERIFIED',len(joined)*dt,maximum.tolist(),time.perf_counter()-start)
