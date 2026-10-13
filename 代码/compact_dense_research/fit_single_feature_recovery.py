"""Three fixed one-feature candidates from actual current failed-prefix inputs."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import json
import time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
from evaluate_historical_dense import ROOT,write,sha,verify_bound_inputs
from variable_radius_recovery_policy import policy_from
from fit_matched_local_recovery import call


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    torch.set_num_threads(1)
    verify_bound_inputs()
    seed=923203
    sup=ROOT/'experimental_supplement_20260923'
    parent_stage=sup/'strict_prefix_recovery'/f'seed_{seed}'
    parent=parent_stage/'checkpoint_0000.pt'
    previous=sup/'matched_local_recovery'/f'seed_{seed}'
    probe=sup/'strict_prefix_next_probe'/f'seed_{seed}'/'half_ms'
    trials=read(probe/'trials.json')
    valid=[t for t in trials if t['kind']=='offset' and t['robust_local_candidate']]
    choice=min(valid,key=lambda t:(float(np.linalg.norm(np.array(t['delta'])/[.002,.02,.05])),-t['sampled_margin_mm'],t['trial']))
    assert choice['trial']==22
    delta=np.array(choice['delta'],dtype=np.float32)
    started=time.perf_counter()
    stage=sup/'single_feature_recovery'/f'seed_{seed}'
    stage.mkdir(parents=True,exist_ok=False)
    bindings={str(p):sha(p) for p in [parent,probe/'registration.json',probe/'trials.json',Path(__file__),ROOT/'variable_radius_recovery_policy.py']}
    write(stage/'registration.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),parent_checkpoint=str(parent),parent_checkpoint_sha256=sha(parent),source_sha256=bindings,
        training_kind='Offline single static-feature coefficient construction; zero PPO',
        evidence_gap='The strict-prefix repair progresses to13.4265s but exhausts exactly inactive features. Existing feature116 can be removed without any change to protected actor outputs in float32.',
        procedure='For fixed pulse indices0/4/8 after verified arrival, reuse feature116 at the actual parent observation. Radius<=0.8nearest protected input distance. Match the smallest-offset robust local probe request at that one observed state.',
        candidates=[0,4,8],basis_count=192,inputs=14,outputs=3,policy_memory_scalars=0,delta=delta.tolist(),local_teacher_trial=22,
        adoption='Each candidate must pass actual-arrival local check. Select by local margin then earlier pulse for complete evaluation. Full bothcases/bothsteps/all34/currentbaselines required.',
        stop='Three fixed centers, any protected-output change, infeasible action target, source change or120s; no full-success inference from local probes.',wall_budget_s=120,new_ppo_updates=0))
    saved=torch.load(parent,map_location='cpu',weights_only=False)
    incumbent=policy_from(saved,seed).eval()
    original=[]
    # Prior complete original traces are anchor data; the parent's own new full
    # evaluation remains separate. Source identity is checked and not relabelled.
    for key in ['1ms','half_ms']:
        path=previous/f'evaluation/eval_0000_original_{key}.npz'
        result=read(path.with_suffix('.json'))
        assert result['success'] and result['trace_sha256']==sha(path)
        assert result['checkpoint_sha256']==sha(previous/'checkpoint_0000.pt')
        with np.load(path) as data:original.append(data['observations'].copy().astype(np.float32))
        for p in [path,path.with_suffix('.json')]:bindings[str(p)]=sha(p)
    paths=[parent_stage/f'evaluation/eval_0000_dense_{key}.npz' for key in ['1ms','half_ms']]
    dense=[]
    for path in paths:
        result=read(path.with_suffix('.json'))
        assert result['trace_sha256']==sha(path) and result['checkpoint_sha256']==sha(parent)
        with np.load(path) as data:dense.append(data['observations'].copy().astype(np.float32))
        for p in [path,path.with_suffix('.json')]:bindings[str(p)]=sha(p)
    base=len(dense[1])-12
    records=[]
    for index,pulse in enumerate([0,4,8]):
        protected=np.vstack([*original,dense[0],dense[1][:base+pulse]])
        x=dense[1][base+pulse]
        policy=policy_from(saved,seed).eval()
        with torch.no_grad():policy.actor.correction.weight[:,116]=0.
        before=call(incumbent.actor,protected)
        assert float(abs(call(policy.actor,protected)-before).max())==0.
        nearest=float(np.sqrt((((protected-x)/policy.actor.feature_scale.numpy())**2).sum(1)).min())
        assert nearest>.001
        with torch.no_grad():
            policy.actor.radius[116]=min(float(policy.actor.radius[116]),.8*nearest)
            policy.actor.centers[116]=torch.tensor(x)
        assert float(call(policy.actor.features,protected)[:,116].max())==0.
        y=call(incumbent.actor,x[None])[0]+delta
        b=call(policy.actor.backbone,x[None])[0]
        z=(y-b)/policy.actor.output_scale.numpy()
        assert np.all(abs(z)<.999999),'Unreachable single-feature request'
        f=call(policy.actor.features,x[None])[0]
        old_sum=f@policy.actor.correction.weight.detach().numpy().T
        weight=(np.arctanh(z)-old_sum)/f[116]
        with torch.no_grad():policy.actor.correction.weight[:,116]=torch.tensor(weight)
        target_error=abs(call(policy.actor,x[None])[0]-y)
        assert target_error.max()<1e-5
        protected_error=float(abs(call(policy.actor,protected)-before).max())
        assert protected_error==0.
        payload=dict(saved)
        payload.update(policy=policy.state_dict(),updates=index,new_ppo_updates=0,
            parent_checkpoint=str(parent),parent_checkpoint_sha256=sha(parent),
            training_kind='Offline one-feature actual-state recovery',checkpoint_index_kind='pulse_index_0_4_8')
        checkpoint=stage/f'checkpoint_{index:04d}.pt'
        torch.save(payload,checkpoint)
        records.append(dict(index=index,pulse=pulse,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
            radius=float(policy.actor.radius[116]),nearest_protected_distance=nearest,protected_inputs=len(protected),
            protected_max_raw_output_error=protected_error,target_max_error=target_error.tolist()))
    write(stage/'fit_audit.json',dict(records=records,source_sha256=bindings,initial_original_traces_are_prior_anchor_data=True))
    assert all(sha(p)==h for p,h in bindings.items())
    assert time.perf_counter()-started<120.
    write(stage/'replay_receipt.json',dict(status='finished',source_unchanged=True,new_ppo_updates=0,wall_s=time.perf_counter()-started,local_checks_pending=True))
    print('SINGLE_FEATURE_CANDIDATES',records,flush=True)


if __name__=='__main__':main()
