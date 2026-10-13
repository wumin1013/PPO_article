"""Conditional student refinement against a fixed teacher feedback function.

Synthetic neighbors supply action labels only, never physical-success labels.
Common two-action students are prepared before paired action expansion.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import argparse
import copy
import json
import shutil
import time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from evaluate_historical_dense import ROOT,write,sha,verify_bound_inputs
from replay_lowdim_training_stage import SUPPLEMENT,SEEDS
from localized_two_action_policy import LocalizedTwoPolicy,policy_from
from direct_two_action_control import DirectTwoPolicy
from learning_update_coherent_anchor import equivalent_loss


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,choices=SEEDS,required=True)
    parser.add_argument('--role',choices=['main_repeat','common_two'],required=True)
    args=parser.parse_args()
    torch.set_num_threads(1);torch.manual_seed(args.seed)
    rng=np.random.default_rng(args.seed)
    verify_bound_inputs()
    stage=SUPPLEMENT/'coherent_teacher_refinement'/f'seed_{args.seed}'/args.role
    stage.mkdir(parents=True,exist_ok=False)
    (stage/'source').mkdir()
    shutil.copy2(__file__,stage/'source'/Path(__file__).name)
    if args.role=='main_repeat':
        parent=SUPPLEMENT/'matched_direction_initialization'/f'seed_{args.seed}/three/fit_12000.pt'
        teacher_path=ROOT/'direct14_dictionary02_resume_round73/checkpoint_0000.pt'
        teacher_index=0
    else:
        parent=SUPPLEMENT/'independent_training'/f'seed_{args.seed}/farvector_fit/fit_16000.pt'
        teacher_path=ROOT/'direct14_capacity_two_action_selection_round60/checkpoint_0024.pt'
        teacher_index=24
    saved=torch.load(parent,map_location='cpu',weights_only=False)
    teacher_saved=torch.load(teacher_path,map_location='cpu',weights_only=False)
    teacher=policy_from(teacher_saved,args.seed).eval().requires_grad_(False)
    if args.role=='main_repeat':
        policy=policy_from(saved,args.seed)
        action_dim=3
    else:
        base=DirectTwoPolicy(args.seed)
        base.initialize_common_layers(saved['policy'])
        policy=LocalizedTwoPolicy(args.seed,count=192)
        policy.initialize(base)
        with torch.no_grad():
            for name in ['centers','feature_scale','radius']:
                getattr(policy.actor,name).copy_(getattr(teacher.actor,name))
            assert torch.count_nonzero(policy.actor.correction.weight)==0
        action_dim=2
    policy.actor.requires_grad_(True)
    parts=[];case_ids=[];bindings={str(p):sha(p) for p in [Path(__file__),parent,teacher_path]}
    trace_records=[]
    for case in ['original','dense']:
        for key in ['1ms','half_ms']:
            path=teacher_path.parent/f'eval_{teacher_index:04d}_{case}_{key}.npz'
            result_path=path.with_suffix('.json')
            result=json.loads(result_path.read_text(encoding='utf-8'))
            assert result['success'] and result['trace_sha256']==sha(path)
            for p in [path,result_path]:bindings[str(p)]=sha(p)
            with np.load(path) as data:x=data['observations'].copy()
            ids=np.arange(len(x))
            if case=='original':
                selected=(abs(x[:,6])>.015)|(abs(x[:,13])>.003)|(np.linalg.norm(x[:,1:3],axis=1)<.995)
                ids=np.unique(np.r_[np.flatnonzero(selected),np.arange(0,len(x),32),np.arange(100),np.arange(len(x)-100,len(x))])
            parts.append(x[ids]);case_ids.extend([case]*len(ids))
            trace_records.append(dict(case=case,resolution=key,samples=len(ids),trace=str(path)))
    x=torch.tensor(np.vstack(parts),dtype=torch.float32)
    case_ids=np.array(case_ids)
    groups=[np.flatnonzero(case_ids==case) for case in ['original','dense']]
    weights=torch.zeros(len(x))
    for group in groups:weights[group]=.5/len(group)
    center=(x*weights[:,None]).sum(0)
    scale=torch.sqrt(((x-center).square()*weights[:,None]).sum(0)).clamp_min(.03)
    noise=torch.tensor([.003,.0015,.0015,.004,.004,.003,.003,.0015,.0015,.0015,.0015,.0015,.0015,.003])
    write(stage/'registration.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        seed=args.seed,role=args.role,source_sha256=bindings,parent_checkpoint=str(parent),
        parent_checkpoint_sha256=sha(parent),teacher_checkpoint=str(teacher_path),teacher_sha256=sha(teacher_path),
        trace_records=trace_records,training_kind='Offline coherent-feedback distillation; zero fresh PPO',
        iterations=30000,fixed_checkpoint=30000,batch_size=512,case_fractions=[.5,.5],
        learning_rate=2e-4,decay_iterations=[10000,20000],decay_factor=.3,
        alternating_witnessed_and_synthetic_batches=True,synthetic_std=noise.tolist(),
        shared_teacher_scope='Same fixed teacher and corpus for all three independent student histories; static feature maps shared. Teacher discovery is not independently repeated.',
        no_teacher_output_weights_copied=True,wall_budget_s=1800,worker_deadline_s=1680,
        evidence_gap=('Pointwise fit improved one independent main seed but remaining half-step failures indicate feedback-neighborhood mismatch.' if action_dim==3 else 'A common successful two-action behavior is needed before expanding preview availability for a fair paired continuation.'),
        adoption='Complete both cases at both steps. Main additionally needs unchanged all34/time gates; common-two must be competent before paired expansion. Neither low loss nor synthetic labels imply success.',
        stop='Fixed30000iterations,nonfinite values,or1680s. Retain all seeds/failures; no per-seed checkpoint selection.',new_ppo_updates=0))
    started=time.perf_counter()
    first=(policy.actor.backbone.backbone[0] if action_dim==3 else policy.actor.backbone[0])
    with torch.no_grad():
        before=policy.actor(x[:512]).clone()
        raw=first.weight.clone();first.weight.mul_(scale);first.bias.add_(raw@center)
    def predict(value):
        if action_dim==3:
            raw=policy.actor.backbone.backbone((value-center)/scale)
            correction=.4*torch.tanh(policy.actor.backbone.correction(policy.actor.backbone.features(value)))
            raw=torch.cat([raw[:,:2],raw[:,2:]+correction],dim=-1)
        else:raw=policy.actor.backbone((value-center)/scale)
        return raw+torch.tanh(policy.actor.correction(policy.actor.features(value)))*policy.actor.output_scale
    with torch.no_grad():assert float(abs(predict(x[:512])-before).max())<3e-5
    optimizer=torch.optim.Adam(policy.actor.parameters(),lr=2e-4)
    output_scale=torch.tensor([.015,.15,.3][:action_dim])
    with torch.no_grad():target=torch.cat([teacher.actor(part) for part in x.split(2048)])
    np.savez_compressed(stage/'witnessed_supervision.npz',observations=x.numpy(),targets=target.numpy(),case=case_ids)
    logs=[]
    for iteration in range(1,30001):
        if time.perf_counter()-started>1680:break
        ids=torch.from_numpy(np.concatenate([g[rng.integers(len(g),size=256)] for g in groups]))
        value=x[ids].clone()
        if iteration%2==0:
            value+=torch.randn_like(value)*noise
            value[:,6].clamp_(-.995,.995)
            with torch.no_grad():labels=teacher.actor(value)
        else:labels=target[ids]
        loss=equivalent_loss(predict(value),labels,output_scale)
        assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.actor.parameters(),1.);optimizer.step()
        if iteration in [10000,20000]:
            for group in optimizer.param_groups:group['lr']*=.3
        if iteration%2000==0 or iteration==30000:
            with torch.no_grad():
                prediction=torch.cat([predict(part) for part in x.split(2048)])
                error=torch.clamp(prediction,-1,1)-torch.clamp(target,-1,1)
            record=dict(iteration=iteration,loss=float(loss.detach()),rmse=error.square().mean(0).sqrt().tolist(),
                max_error=error.abs().max(0).values.tolist(),wall_s=time.perf_counter()-started)
            logs.append(record);write(stage/'fit_log.json',logs)
            print('COHERENT_FIT',args.seed,args.role,iteration,record['rmse'],flush=True)
        if iteration in [10000,20000,30000]:
            exported=copy.deepcopy(policy)
            first_out=(exported.actor.backbone.backbone[0] if action_dim==3 else exported.actor.backbone[0])
            with torch.no_grad():
                first_out.weight.div_(scale);first_out.bias.sub_(first_out.weight@center)
                export_error=float(abs(exported.actor(x[:512])-predict(x[:512])).max())
            assert export_error<3e-5
            payload=dict(saved)
            payload.update(policy=exported.state_dict(),interface=[14,action_dim,args.seed],
                policy_class='LocalizedJointPolicy' if action_dim==3 else 'LocalizedTwoPolicy',
                actor_outputs=action_dim,local_basis_count=192,updates=iteration,offline_iterations=iteration,
                new_ppo_updates=0,training_kind='Offline coherent teacher feedback refinement',
                parent_checkpoint=str(parent),parent_checkpoint_sha256=sha(parent),teacher_sha256=sha(teacher_path),
                export_equivalence_error=export_error,preview_rule_mm=3.,fit_optimizer=optimizer.state_dict(),
                normalized_actor=policy.actor.state_dict(),center=center,scale=scale,
                numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state())
            torch.save(payload,stage/f'fit_{iteration:05d}.pt')
    complete=(stage/'fit_30000.pt').exists()
    write(stage/'replay_receipt.json',dict(status='finished' if complete else 'timeout',
        fixed_checkpoint_exists=complete,offline_iterations=iteration,new_ppo_updates=0,
        wall_s=time.perf_counter()-started,sources_unchanged=all(sha(p)==h for p,h in bindings.items()),
        full_pipeline_complete=False))


if __name__=='__main__':main()
