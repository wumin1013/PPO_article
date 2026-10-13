"""Offline recovery with exact invariance on retained witnessed policy inputs."""
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
    parent_stage=sup/'matched_local_recovery'/f'seed_{seed}'
    parent=parent_stage/'checkpoint_0000.pt'
    probe=sup/'matched_local_second_probe'/f'seed_{seed}'/'half_ms'
    teacher=probe/'trial_10.npz'
    assert read(probe/'trials.json')[10]['robust_local_candidate']
    assert read(probe/'registration.json')['source_sha256'][str(parent)]==sha(parent)
    stage=sup/'strict_prefix_recovery'/f'seed_{seed}'
    stage.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    bindings={str(p):sha(p) for p in [parent,teacher,probe/'trials.json',probe/'registration.json',Path(__file__),ROOT/'variable_radius_recovery_policy.py',ROOT/'fit_matched_local_recovery.py']}
    write(stage/'registration.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),seed=seed,parent_checkpoint=str(parent),parent_checkpoint_sha256=sha(parent),source_sha256=bindings,
        training_kind='Offline local static-feature coefficient fit; zero new PPO',
        evidence_gap='Fixed-radius repair disturbed earlier successful dense visits. Preserve both original traces, nominal dense trace and every half-step prefix input before the last12decisions exactly.',
        procedure='Use24jointfeatures inactive on all protected inputs. Place centers on the first24verified recovery inputs; set each radius to min(oldradius,0.8*nearest protected distance). Freeze all other weights and centers, solve24coefficient columns.',
        architecture_change='The192jointfeatures now store one static radius per center instead of one shared scalar.14inputs,3outputs,zero policy memory,MLP and feature count remain unchanged. This change must be disclosed if adopted; it is not an additional PPO update.',
        adoption='Local closed-loop check then complete bothcases/bothsteps/all34/current strong baselines/time margins. Frozen73 remains unchanged.',
        stop='One fixed candidate or120seconds, any protected-output change, fit failure or changed source.',wall_budget_s=120,new_ppo_updates=0))
    saved=torch.load(parent,map_location='cpu',weights_only=False)
    incumbent=policy_from(saved,seed).eval()
    policy=policy_from(saved,seed).eval()
    protected=[]
    for case in ['original','dense']:
        for key in ['1ms','half_ms']:
            path=parent_stage/f'evaluation/eval_0000_{case}_{key}.npz'
            result=read(path.with_suffix('.json'))
            assert result['trace_sha256']==sha(path) and result['checkpoint_sha256']==sha(parent)
            with np.load(path) as data:x=data['observations'].copy().astype(np.float32)
            if case=='original' or key=='1ms':assert result['success']
            else:
                assert not result['success']
                x=x[:-12]
            protected.append(x)
            for p in [path,path.with_suffix('.json')]:bindings[str(p)]=sha(p)
    protected=np.vstack(protected)
    with np.load(teacher) as data:
        px=data['observations'][:24].copy().astype(np.float32)
        py=data['latents'][:24].copy().astype(np.float32)
    features=call(incumbent.actor.features,protected)
    unused=np.flatnonzero(features.max(0)==0.)
    assert len(unused)>=24
    reuse=unused[:24]
    retained=np.setdiff1d(np.arange(192),reuse)
    scale=incumbent.actor.feature_scale.numpy()
    nearest=np.full(len(px),np.inf)
    for part in np.array_split(protected,100):
        nearest=np.minimum(nearest,np.sqrt((((px[:,None]-part[None])/scale)**2).sum(2)).min(1))
    assert nearest.min()>.01,'Current observation cannot be separated safely at this resolution'
    radii=np.full(192,float(incumbent.actor.radius),dtype=np.float32)
    radii[reuse]=np.minimum(radii[reuse],.8*nearest)
    with torch.no_grad():
        policy.actor.radius=torch.tensor(radii)
        policy.actor.centers[reuse]=torch.tensor(px)
        policy.actor.correction.weight[:,reuse]=0.
    assert call(policy.actor.features,protected)[:,reuse].max()==0.
    f=call(policy.actor.features,px)
    target=np.arctanh(np.clip((py-call(policy.actor.backbone,px))/policy.actor.output_scale.numpy(),-.999999,.999999))
    target-=f@policy.actor.correction.weight.detach().numpy().T
    matrix=f[:,reuse].astype(float)
    coefficients=np.linalg.solve(matrix.T@matrix+1e-10*np.eye(24),matrix.T@target)
    with torch.no_grad():policy.actor.correction.weight[:,reuse]=torch.tensor(coefficients.T,dtype=torch.float32)
    fit_error=abs(call(policy.actor,px)-py).max(0)
    assert np.all(fit_error<np.array([1e-5,1e-4,1e-4])),fit_error
    protected_error=float(abs(call(policy.actor,protected)-call(incumbent.actor,protected)).max())
    assert protected_error==0.,protected_error
    pointwise_error=max(float(abs(policy.act(x,True)[0]-incumbent.act(x,True)[0]).max()) for x in protected[::97])
    assert pointwise_error==0.
    for key,value in policy.state_dict().items():
        if key=='actor.radius':continue
        if key in ['actor.centers','actor.correction.weight']:
            axis=0 if key=='actor.centers' else 1
            assert torch.equal(value.index_select(axis,torch.tensor(retained)),saved['policy'][key].index_select(axis,torch.tensor(retained)))
        else:assert torch.equal(value,saved['policy'][key]),key
    payload=dict(saved)
    payload.update(policy=policy.state_dict(),policy_class='LocalizedJointVariableRadiusPolicy',updates=0,new_ppo_updates=0,
        training_kind='Offline strict-prefix-preserving local fit with per-center static radii',parent_checkpoint=str(parent),parent_checkpoint_sha256=sha(parent),
        checkpoint_index_kind='fixed_candidate',joint_support_radius_kind='per_center',new_static_radius_parameters=191)
    checkpoint=stage/'checkpoint_0000.pt'
    torch.save(payload,checkpoint)
    write(stage/'fit_audit.json',dict(source_sha256=bindings,protected_input_count=len(protected),protected_max_output_error=protected_error,
        sampled_pointwise_output_error=pointwise_error,teacher_max_error=fit_error.tolist(),reused_indices=reuse.tolist(),new_radii=radii[reuse].tolist(),
        minimum_target_distance=float(nearest.min()),checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),no_complete_success_claim=True))
    assert all(sha(p)==h for p,h in bindings.items())
    assert time.perf_counter()-started<120.
    write(stage/'replay_receipt.json',dict(status='finished',wall_s=time.perf_counter()-started,source_unchanged=True,new_ppo_updates=0,complete_evaluation_pending=True))
    print('STRICT_PREFIX_FIT',len(protected),protected_error,fit_error.tolist(),flush=True)


if __name__=='__main__':main()
