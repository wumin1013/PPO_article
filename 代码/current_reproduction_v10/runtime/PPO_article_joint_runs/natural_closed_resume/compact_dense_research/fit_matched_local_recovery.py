"""Fixed-capacity offline repair from a witnessed local recovery, not PPO."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from evaluate_historical_dense import ROOT, write, sha, verify_bound_inputs
from localized_two_action_policy import policy_from


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def call(function, x):
    with torch.no_grad():
        return torch.cat([function(v) for v in torch.tensor(x, dtype=torch.float32).split(512)]).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, choices=[923201, 923203], required=True)
    parser.add_argument('--variant', choices=['support_complete'], required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    verify_bound_inputs()
    sup = ROOT/'experimental_supplement_20260923'
    parent_stage = sup/'matched_direction_initialization'/f'seed_{args.seed}'/'three'
    parent = parent_stage/'fit_12000.pt'
    probe = sup/'matched_dense_prefix_recovery_probe'/f'seed_{args.seed}'/'half_ms'
    trials = read(probe/'trials.json')
    valid = [r for r in trials if r['robust_local_candidate'] and r['kind']=='offset' and r['sampled_margin_mm']>=.1]
    assert valid, 'No suitable witnessed local recovery'
    selected = min(valid, key=lambda r: (float(np.linalg.norm(np.array(r['delta'])/np.array([.002,.02,.05]))), -r['sampled_margin_mm'], r['trial']))
    teacher = Path(selected['trace'])
    assert sha(teacher)==selected['trace_sha256']
    assert read(probe/'registration.json')['source_sha256'][str(parent)]==sha(parent)
    stage = sup/('matched_local_recovery_'+args.variant)/f'seed_{args.seed}'
    stage.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    bindings = {str(p):sha(p) for p in [parent, teacher, probe/'trials.json', probe/'registration.json', Path(__file__)]}
    write(stage/'registration.json', dict(created_utc=datetime.now(timezone.utc).isoformat(), seed=args.seed,
        parent_checkpoint=str(parent), parent_checkpoint_sha256=sha(parent), source_sha256=bindings,
        training_kind='Offline least-squares static feature redistribution; zero PPO updates',
        evidence_gap='Matched12000 retains both complete original executions but fails half-step dense execution. A verified actual-arrival local recovery is available.',
        selection='Require local physical pass and0.1mm sampled margin; minimize normalized request-offset norm, then maximize margin, then earlier trial.',
        selected_trial=selected['trial'], maximum_reallocated_features=32, feature_count=192,
        actor_inputs=14, actor_outputs=3, policy_memory_scalars=0, anchor_weight=4., ridge=1e-5,
        procedure='Reallocate only features exactly inactive on both original-route executions; new centers must be outside their support. Freeze all other weights; fit new coefficients against local recovery and retained witnessed outputs.',
        adoption='Unchanged complete both-case/both-step physical,all34,history,S2,and strongest valid two-output time requirements. Local error is not acceptance.',
        stop='One fixed candidate, source change, loss of original-output equality, or120s; complete evaluation is a separate registered <=30min round.',
        wall_budget_s=120, new_ppo_updates=0, teacher_query_is_not_full_success=True))
    saved = torch.load(parent, map_location='cpu', weights_only=False)
    policy = policy_from(saved,args.seed).eval()
    source = policy_from(saved,args.seed).eval()
    original, other = [], []
    for case in ['original','dense']:
        for key in ['1ms','half_ms']:
            path = parent_stage/f'evaluation/eval_12000_{case}_{key}.npz'
            result = read(path.with_suffix('.json'))
            assert result['trace_sha256']==sha(path) and result['checkpoint_sha256']==sha(parent)
            with np.load(path) as data:
                x = data['observations'].copy().astype(np.float32)
            if case=='original':
                assert result['success']
                original.append(x)
            else:
                other.append(x if result['success'] else x[:-12])
            for p in [path,path.with_suffix('.json')]:bindings[str(p)]=sha(p)
    ox = np.vstack(original)
    ax = np.vstack(other)
    with np.load(teacher) as data:
        px = data['observations'].copy().astype(np.float32)
        py = data['latents'].copy().astype(np.float32)
    of = call(source.actor.features,ox)
    free = np.flatnonzero(of.max(0)==0)
    assert len(free)>=32, 'Insufficient unused fixed-capacity features'
    af = call(source.actor.features,ax)
    old = source.actor.correction.weight.detach().numpy()
    importance = af.max(0)*np.linalg.norm(old*source.actor.output_scale.numpy()[:,None],axis=0)
    reuse = free[np.argsort(importance[free],kind='stable')[:32]]
    retained = np.setdiff1d(np.arange(192),reuse)
    scale = source.actor.feature_scale.numpy()
    radius = float(source.actor.radius)
    nearest = np.full(len(px),np.inf)
    for part in np.array_split(ox,100):
        nearest = np.minimum(nearest,np.sqrt((((px[:,None]-part[None])/scale)**2).sum(2)).min(1))
    assert nearest.min()>1.05*radius, 'New support overlaps protected original states'
    # Retained coefficients are frozen: their support cannot cover a NEW target.
    distance = np.full(len(px),np.inf)
    chosen = []
    for _ in reuse:
        available = distance.copy()
        available[chosen] = -1
        index = int(available.argmax())
        chosen.append(index)
        distance = np.minimum(distance,(((px-px[index])/scale)**2).sum(1))
    with torch.no_grad():
        policy.actor.centers[reuse] = torch.tensor(px[chosen])
        policy.actor.correction.weight[:,reuse] = 0.
    assert call(policy.actor.features,ox)[:,reuse].max()==0.
    x = np.vstack([px,ax])
    y = np.vstack([py,call(source.actor,ax)])
    features = call(policy.actor.features,x)
    assert np.all(features[:len(px),reuse].max(1)>0.), 'New coefficients do not cover every repair target'
    base = call(policy.actor.backbone,x)
    residual = np.arctanh(np.clip((y-base)/policy.actor.output_scale.numpy(),-.999999,.999999))
    residual -= features@policy.actor.correction.weight.detach().numpy().T
    f = features[:,reuse].astype(float)
    weights = np.r_[np.ones(len(px)),np.full(len(ax),4.)]
    coef = np.linalg.solve(f.T@(weights[:,None]*f)+1e-5*np.eye(32),f.T@(weights[:,None]*residual))
    with torch.no_grad():policy.actor.correction.weight[:,reuse] = torch.tensor(coef.T,dtype=torch.float32)
    original_error = float(abs(call(policy.actor,ox)-call(source.actor,ox)).max())
    assert original_error==0., original_error
    pointwise_error = max(float(abs(policy.act(row,True)[0]-source.act(row,True)[0]).max()) for row in ox[::97])
    assert pointwise_error==0., pointwise_error
    for key,value in policy.state_dict().items():
        if key in ['actor.centers','actor.correction.weight']:
            axis=0 if key=='actor.centers' else 1
            assert torch.equal(value.index_select(axis,torch.tensor(retained)),saved['policy'][key].index_select(axis,torch.tensor(retained)))
        else:assert torch.equal(value,saved['policy'][key]),key
    payload=dict(saved)
    payload.update(policy=policy.state_dict(),updates=0,new_ppo_updates=0,training_kind='Offline fixed-capacity local recovery fit',
        parent_checkpoint=str(parent),parent_checkpoint_sha256=sha(parent),checkpoint_index_kind='fixed_candidate',
        selected_local_trace=str(teacher),selected_local_trace_sha256=sha(teacher))
    checkpoint=stage/'checkpoint_0000.pt'
    torch.save(payload,checkpoint)
    errors=call(policy.actor,x)-y
    np.savez_compressed(stage/'fit_data.npz',observations=x,targets=y,positive_count=len(px),reused_indices=reuse)
    write(stage/'fit_audit.json',dict(selected_trial=selected['trial'],reused_indices=reuse.tolist(),original_observation_count=len(ox),
        original_batched_output_max_error=original_error,original_sampled_pointwise_output_max_error=pointwise_error,
        minimum_target_distance_from_original=float(nearest.min()),teacher_max_error=abs(errors[:len(px)]).max(0).tolist(),
        anchor_max_error=abs(errors[len(px):]).max(0).tolist(),source_sha256=bindings,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint)))
    assert all(sha(p)==h for p,h in bindings.items())
    assert time.perf_counter()-started<120.
    write(stage/'replay_receipt.json',dict(status='finished',wall_s=time.perf_counter()-started,new_ppo_updates=0,
        source_unchanged=True,complete_evaluation_pending=True,whole_case_success_demonstrated=False))
    print('LOCAL_FIT',args.seed,'trial',selected['trial'],'original_error',original_error,'teacher_max',abs(errors[:len(px)]).max(0).tolist(),flush=True)


if __name__=='__main__':
    main()
