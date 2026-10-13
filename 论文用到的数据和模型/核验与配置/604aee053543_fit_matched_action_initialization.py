"""Equal-budget offline fitting of paired action interfaces on shared inputs."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import argparse
import json
import shutil
import time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from evaluate_historical_dense import ROOT,write,sha,verify_bound_inputs
from replay_lowdim_training_stage import SUPPLEMENT,SEEDS
from localized_two_action_policy import policy_from
from learning_update_coherent_anchor import equivalent_loss


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,choices=SEEDS,required=True)
    parser.add_argument('--arm',choices=['three','two'],required=True)
    args=parser.parse_args()
    torch.set_num_threads(1);torch.manual_seed(args.seed)
    rng=np.random.default_rng(args.seed)
    verify_bound_inputs()
    corpus=SUPPLEMENT/'matched_direction_initialization'
    protocol=json.loads((corpus/'protocol.json').read_text(encoding='utf-8'))
    assert protocol['dataset_sha256']==sha(corpus/'supervision.npz')
    parent=SUPPLEMENT/'paired_action_ablation'/f'seed_{args.seed}'/args.arm/'initial.pt'
    stage=corpus/f'seed_{args.seed}'/args.arm
    assert not (stage/'registration.json').exists(), 'Do not overwrite a registered run'
    stage.mkdir(parents=True,exist_ok=True)
    (stage/'source').mkdir(exist_ok=True)
    shutil.copy2(__file__,stage/'source'/Path(__file__).name)
    hashes={str(p):sha(p) for p in [Path(__file__),parent,corpus/'supervision.npz',corpus/'protocol.json']}
    write(stage/'registration.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        seed=args.seed,arm=args.arm,initial_checkpoint=str(parent),initial_checkpoint_sha256=sha(parent),
        source_sha256=hashes,protocol=protocol,wall_budget_s=1800,worker_deadline_s=1680,
        evidence_gap='Direct removal changes physical direction and short PPO alone left invalid initial trajectories; both arms receive identical offline fitting opportunities before matched fresh PPO.',
        adoption='Fixed12000 checkpoint only; no success-conditioned selection. Offline loss is not physical success.',
        stop='12000iterations,nonfinite loss or1680s; retain incomplete stages.',new_ppo_updates=0))
    started=time.perf_counter()
    saved=torch.load(parent,map_location='cpu',weights_only=False)
    policy=policy_from(saved,args.seed)
    policy.actor.requires_grad_(True)
    with np.load(corpus/'supervision.npz') as data:
        x=torch.tensor(data['observations'])
        y=torch.tensor(data['targets_'+args.arm])
        groups=[np.flatnonzero(data['case']==case) for case in ['original','dense']]
    with torch.no_grad():
        joint=torch.cat([policy.actor.features(part) for part in x.split(2048)])
        preview=(torch.cat([policy.actor.backbone.features(part) for part in x.split(2048)])
                 if args.arm=='three' else None)
    def predict(ids):
        if args.arm=='three':
            raw=policy.actor.backbone.backbone(x[ids])
            p=.4*torch.tanh(policy.actor.backbone.correction(preview[ids]))
            raw=torch.cat([raw[:,:2],raw[:,2:]+p],dim=-1)
        else:raw=policy.actor.backbone(x[ids])
        return raw+torch.tanh(policy.actor.correction(joint[ids]))*policy.actor.output_scale
    with torch.no_grad():
        ids=torch.arange(min(len(x),512))
        assert torch.equal(predict(ids),policy.actor(x[ids]))
    optimizer=torch.optim.Adam(policy.actor.parameters(),lr=2e-4)
    scales=torch.tensor([.015,.15,.3][:saved['interface'][1]])
    log=[]
    for iteration in range(1,12001):
        if time.perf_counter()-started>1680:break
        ids=torch.from_numpy(np.concatenate([g[rng.integers(len(g),size=256)] for g in groups]))
        prediction=predict(ids)
        loss=equivalent_loss(prediction,y[ids],scales)
        assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.actor.parameters(),1.)
        optimizer.step()
        if iteration in [4000,8000]:
            for group in optimizer.param_groups:group['lr']*=.3
        if iteration%500==0:
            with torch.no_grad():
                prediction=torch.cat([predict(ids) for ids in torch.arange(len(x)).split(2048)])
                delta=torch.clamp(prediction,-1,1)-torch.clamp(y,-1,1)
                record=dict(iteration=iteration,loss=float(loss.detach()),
                    clipped_action_rmse=delta.square().mean(0).sqrt().tolist(),
                    maximum_clipped_error=delta.abs().max(0).values.tolist(),wall_s=time.perf_counter()-started)
            log.append(record);write(stage/'fit_log.json',log)
            print('MATCHED_FIT',args.seed,args.arm,iteration,record['clipped_action_rmse'],flush=True)
        if iteration in [4000,8000,12000]:
            payload=dict(saved)
            payload.update(policy=policy.state_dict(),offline_iterations=iteration,new_ppo_updates=0,
                updates=iteration,parent_checkpoint=str(parent),parent_checkpoint_sha256=sha(parent),
                training_kind='Equal-budget shared-input offline fit with action-interface-specific labels',
                dataset_sha256=sha(corpus/'supervision.npz'),optimizer=optimizer.state_dict())
            torch.save(payload,stage/f'fit_{iteration:05d}.pt')
    complete=(stage/'fit_12000.pt').exists()
    write(stage/'replay_receipt.json',dict(status='finished' if complete else 'timeout',
        offline_iterations=iteration,new_ppo_updates=0,wall_s=time.perf_counter()-started,
        sources_unchanged=all(sha(p)==h for p,h in hashes.items()),fixed_checkpoint_exists=complete,
        full_pipeline_complete=False))


if __name__=='__main__':main()
