"""Offline two-output feedback fit with two accumulated generations of witnessed states.

Failed student prefixes provide inputs only. Teacher queries are action labels,
not certificates that the teacher can recover a failed physical arrival.
"""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import copy
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from evaluate_historical_dense import ROOT, write, sha, verify_bound_inputs
from replay_lowdim_training_stage import SUPPLEMENT, SEEDS
from localized_two_action_policy import policy_from
from learning_update_coherent_anchor import equivalent_loss


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    verify_bound_inputs()
    previous = SUPPLEMENT/'two_visited_feedback'/f'seed_{args.seed}'
    assert read(previous/'evaluation/replay_receipt.json')['all_four_evaluations_recorded']
    parent = previous/'fit_30000.pt'
    teacher_path = ROOT/'direct14_capacity_two_action_selection_round60/checkpoint_0024.pt'
    saved = torch.load(parent, map_location='cpu', weights_only=False)
    policy = policy_from(saved, args.seed)
    teacher = policy_from(torch.load(teacher_path, map_location='cpu', weights_only=False), args.seed).eval().requires_grad_(False)
    policy.actor.requires_grad_(True)
    groups, origins = [], []
    bindings = {str(p): sha(p) for p in [parent, teacher_path, Path(__file__), ROOT/'fit_two_visited_feedback.py']}
    for scope, directory, index in [('teacher', teacher_path.parent, 24), ('own', previous/'evaluation', 30000)]:
        for case in ['original', 'dense']:
            parts = []
            inputs = [(directory,index,key,parent) for key in ['1ms','half_ms']]
            if scope == 'own':
                prior = SUPPLEMENT/'coherent_teacher_refinement'/f'seed_{args.seed}'/'common_two'
                inputs += [(prior/'evaluation',30000,key,prior/'fit_30000.pt') for key in ['1ms','half_ms']]
            for origin,index,key,origin_parent in inputs:
                path = origin/f'eval_{index:04d}_{case}_{key}.npz'
                bindings[str(origin_parent)] = sha(origin_parent)
                result = read(path.with_suffix('.json'))
                assert result['trace_sha256'] == sha(path)
                assert result['checkpoint_sha256'] == sha(teacher_path if scope=='teacher' else origin_parent)
                if scope == 'teacher':
                    assert result['success']
                with np.load(path) as data:
                    x = data['observations'].copy()
                ids = np.arange(len(x))
                if case == 'original':
                    mask = (abs(x[:, 6])>.015) | (abs(x[:, 13])>.003) | (np.linalg.norm(x[:, 1:3], axis=1)<.995)
                    ids = np.unique(np.r_[np.flatnonzero(mask), np.arange(0,len(x),32), np.arange(min(100,len(x))), np.arange(max(0,len(x)-100),len(x))])
                parts.append(x[ids])
                origins.append(dict(scope=scope, case=case, resolution=key, success=result['success'], samples=len(ids), trace=str(path)))
                for p in [path, path.with_suffix('.json')]:
                    bindings[str(p)] = sha(p)
            groups.append(torch.tensor(np.vstack(parts), dtype=torch.float32))
    stage = SUPPLEMENT/'two_accumulated_feedback'/f'seed_{args.seed}'
    stage.mkdir(parents=True, exist_ok=False)
    (stage/'source').mkdir()
    shutil.copy2(__file__, stage/'source'/Path(__file__).name)
    write(stage/'registration.json', dict(created_utc=datetime.now(timezone.utc).isoformat(), seed=args.seed,
        source_sha256=bindings, parent_checkpoint=str(parent), teacher_checkpoint=str(teacher_path), origins=origins,
        evidence_gap='The first own-visited fit moved seed2 failure from129/177s to270s without completing. Accumulate both visited generations to cover later drift while retaining earlier failure-state labels; all3seeds receive equal budget.',
        training_kind='Offline queried-teacher fit on own witnessed inputs; zero new PPO',
        iterations=30000, fixed_checkpoint=30000, minibatch=512, group_fractions=[.25]*4,
        group_order=['teacher_original','teacher_dense','own_original','own_dense'], no_synthetic_neighbors=True, accumulated_own_trajectory_generations=2,
        learning_rate=1e-4, decay_iterations=[10000,20000], decay_factor=.3,
        wall_budget_s=1800, worker_deadline_s=1680, no_teacher_output_weights_copied=True,
        adoption='Both cases and both steps must complete unchanged physical requirements before paired initialization. Offer identical fitting budget to all3 seeds, retain earlier valid endpoints and all failed histories.',
        stop='Fixed30000 or1680s; no per-seed checkpoint choice inside this round.',
        label_limit='Same-state teacher output does not establish successful recovery from a failed student arrival.',
        independence='Own student weights with shared teacher and static feature maps, not independent teacher discovery.'))
    started = time.perf_counter()
    mean = sum(x.mean(0) for x in groups)/4
    scale = torch.sqrt(sum(((x-mean).square().mean(0)) for x in groups)/4).clamp_min(.03)
    first = policy.actor.backbone[0]
    with torch.no_grad():
        w = first.weight.clone()
        first.weight.mul_(scale)
        first.bias.add_(w@mean)
        labels = [torch.cat([teacher.actor(x) for x in group.split(1024)]) for group in groups]
    def predict(x):
        return policy.actor.backbone((x-mean)/scale)+torch.tanh(policy.actor.correction(policy.actor.features(x)))*policy.actor.output_scale
    optimizer = torch.optim.Adam(policy.actor.parameters(), lr=1e-4)
    logs = []
    completed = 0
    for iteration in range(1,30001):
        if time.perf_counter()-started>1680:
            break
        indices = [torch.from_numpy(rng.integers(len(g),size=128)) for g in groups]
        x = torch.cat([g[i] for g,i in zip(groups,indices)])
        y = torch.cat([g[i] for g,i in zip(labels,indices)])
        loss = equivalent_loss(predict(x), y, torch.tensor([.015,.15]))
        assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.actor.parameters(),1.)
        optimizer.step()
        completed = iteration
        if iteration in [10000,20000]:
            for group in optimizer.param_groups:
                group['lr'] *= .3
        if iteration%5000 == 0:
            with torch.no_grad():
                errors = [(torch.cat([predict(x) for x in g.split(1024)]).clamp(-1,1)-y.clamp(-1,1)).square().mean(0).sqrt().tolist() for g,y in zip(groups,labels)]
            logs.append(dict(iteration=iteration, group_rmse=errors, wall_s=time.perf_counter()-started))
            write(stage/'fit_log.json',logs)
            print('ACCUMULATED_FIT',args.seed,iteration,errors,flush=True)
    if completed == 30000:
        exported = copy.deepcopy(policy)
        with torch.no_grad():
            exported.actor.backbone[0].weight.div_(scale)
            exported.actor.backbone[0].bias.sub_(exported.actor.backbone[0].weight@mean)
            export_error = float((exported.actor(groups[0][:512])-predict(groups[0][:512])).abs().max())
        assert export_error<3e-5
        payload = dict(saved)
        payload.update(policy=exported.state_dict(), updates=30000, new_ppo_updates=0, offline_iterations_this_round=30000,
            parent_checkpoint=str(parent), parent_checkpoint_sha256=sha(parent),
            training_kind='Offline accumulated-own-visited-feedback fit', export_equivalence_error=export_error,
            fit_optimizer=optimizer.state_dict(), normalized_actor=policy.actor.state_dict(),
            center=mean, scale=scale, numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state())
        torch.save(payload,stage/'fit_30000.pt')
    write(stage/'replay_receipt.json',dict(status='finished' if completed==30000 else 'timeout',
        completed_iterations=completed, wall_s=time.perf_counter()-started,new_ppo_updates=0,
        sources_unchanged=all(sha(p)==h for p,h in bindings.items())))


if __name__ == '__main__':
    main()
