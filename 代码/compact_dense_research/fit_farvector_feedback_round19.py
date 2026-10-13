"""Offline migration of the retained feedback law to fourteen far-vector inputs.

Synthetic neighboring observations are training labels, never physical results.
The exported actor is the unchanged 14->64->64->3 network, with no teacher.
"""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import copy
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime, timezone
from evaluate_historical_dense import ROOT, BASE, sha, write, verify_bound_inputs

STAGE = ROOT / 'direct14_farvector_feedback_fit_round19'
INITIAL = ROOT / 'direct14_guided_batchreset_ppo_seed922302_round17/checkpoint_0024.pt'
DATA = ROOT / 'farvector_round17_data'
SEED = 922302


def worker():
    import numpy as np
    import torch
    from compact_fine_policy import FineCompactPolicy
    from compact_farvector_runtime import FarVectorRuntime
    from compact_research_eval import CompleteChecks
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    started = time.perf_counter()
    deadline = started + 1640.
    checks = CompleteChecks(STAGE, deadline, runtime_class=FarVectorRuntime)
    saved = torch.load(INITIAL, map_location='cpu', weights_only=False)
    teacher = FineCompactPolicy(feed_gain=1.)
    teacher.load_state_dict(saved['policy'])
    teacher.eval().requires_grad_(False)
    policy = FineCompactPolicy(feed_gain=1.)
    policy.load_state_dict(saved['policy'])
    groups = []
    sources = []
    for case in ['original', 'dense']:
        with np.load(DATA / (case + '.npz')) as d:
            new = d['observations'].astype(np.float32)
            old = d['original_observations'].astype(np.float32)
        recovered = np.sqrt(np.maximum(0., 1. - old[:, 6] ** 2))
        error = float(abs(recovered - old[:, 5]).max())
        assert error < 2e-7 and old[:, 5].min() > 0.
        sources.append(dict(case=case, samples=len(new), sha256=sha(DATA / (case + '.npz')),
                            positive_cosine_reconstruction_error=error))
        if case == 'original':
            # Preserve every witnessed turn and endpoint; sample straight cruise
            # uniformly because otherwise it overwhelms the shorter dense case.
            curved = (abs(new[:, 6]) > .008) | (abs(new[:, 13]) > .005)
            ids = np.unique(np.r_[np.flatnonzero(curved), np.arange(100),
                                  np.arange(len(new)-100, len(new)),
                                  np.linspace(0, len(new)-1, 768).astype(int)])
            new = new[ids]
        groups.append(torch.tensor(new))
    x = torch.cat(groups)
    weights = torch.cat([torch.full((len(g),), .5/len(g)) for g in groups])
    center = (x * weights[:, None]).sum(0)
    scale = torch.sqrt(((x-center).square()*weights[:, None]).sum(0)).clamp_min(.03)
    actor = copy.deepcopy(policy.actor)
    with torch.no_grad():
        # Remove only the replaced cosine coefficient; its typical constant
        # contribution initializes the bias. The offline fit learns curvature.
        actor[0].bias.add_(actor[0].weight[:, 5])
        actor[0].weight[:, 5].zero_()
        raw = actor[0].weight.clone()
        actor[0].weight.mul_(scale)
        actor[0].bias.add_(raw @ center)
    optimizer = torch.optim.Adam(actor.parameters(), lr=2e-4)
    action_scale = torch.tensor([.015, .15, .3])
    noise_scale = torch.tensor([.006, .003, .003, .008, .008, .006, .006,
                               .003, .003, .003, .003, .003, .003, .006])

    def labels(value):
        old = value.clone()
        old[:, 5] = torch.sqrt(torch.clamp(1.-old[:, 6].square(), min=1e-8))
        return teacher.actor(old)

    with torch.no_grad():
        target = labels(x)
    np.savez_compressed(STAGE/'supervision.npz', observation=x.numpy(),
                        target=target.numpy(), weight=weights.numpy())
    write(STAGE/'dataset.json', dict(sources=sources, selected_sizes=[len(g) for g in groups],
        seed=SEED, group_weights=[.5, .5], noise_std=noise_scale.tolist(),
        teacher=str(INITIAL), teacher_sha256=sha(INITIAL),
        label_scope='Frozen round17 feedback on witnessed states and synthetic local neighborhoods. Positive cosine is reconstructed for labels only. Neighborhoods are not rollouts or successes.',
        physical_steps_for_data=0, new_ppo_updates=0))
    print('DATA_READY', [len(g) for g in groups], flush=True)
    log = []
    evaluations = []
    for iteration in range(1, 30001):
        if time.perf_counter() > deadline - 250.:
            break
        picks = torch.cat([g[torch.from_numpy(rng.integers(0, len(g), size=256))] for g in groups])
        if iteration % 2 == 0:
            picks = picks + torch.randn_like(picks)*noise_scale
            picks[:, 6].clamp_(-.95, .95)
        with torch.no_grad():
            y = labels(picks)
        error = (actor((picks-center)/scale)-y)/action_scale
        loss = error.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.)
        optimizer.step()
        if iteration in [8000, 16000]:
            for group in optimizer.param_groups:
                group['lr'] *= .3
        if iteration % 500 == 0:
            with torch.no_grad():
                err = actor((x-center)/scale)-target
                record = dict(iteration=iteration, loss=float(loss.detach()),
                    maximum_raw_action_error=err.abs().amax(0).tolist(),
                    rmse_raw_action=err.square().mean(0).sqrt().tolist(),
                    wall_s=time.perf_counter()-started)
            log.append(record)
            write(STAGE/'fit_log.json', log)
            print('FIT', iteration, record['rmse_raw_action'], 'wall', record['wall_s'], flush=True)
        if iteration not in [8000, 16000, 30000]:
            continue
        exported = copy.deepcopy(actor)
        with torch.no_grad():
            exported[0].weight.div_(scale)
            exported[0].bias.sub_(exported[0].weight @ center)
            export_error = float(abs(exported(x)-actor((x-center)/scale)).max())
        assert export_error < 3e-5
        policy.actor.load_state_dict(exported.state_dict())
        checkpoint = STAGE/f'fit_{iteration:05d}.pt'
        torch.save(dict(policy=policy.state_dict(), fit_optimizer=optimizer.state_dict(),
            normalized_actor=actor.state_dict(), center=center, scale=scale,
            interface=[14,3,SEED], policy_class='FineCompactPolicy', feed_gain=1.,
            observation_features=FarVectorRuntime(checks.references['dense']).feature_names(),
            observation_dimension=14, policy_memory_scalars=0, actor_outputs=3,
            source_checkpoint=str(INITIAL), source_checkpoint_sha256=sha(INITIAL),
            new_ppo_updates=0, offline_iterations=iteration, inherited_ppo_updates=88,
            training_kind='Offline supervised feedback migration; no online PPO in this round',
            runtime_class='FarVectorRuntime', export_equivalence_error=export_error), checkpoint)
        record = checks.run(policy, checkpoint, iteration)
        complete = all(r['success'] for r in record['results'].values())
        record['eligible_initialization'] = bool(complete and
            record['results']['original']['complete_time_s'] <= 444.316 and
            record['results']['dense']['complete_time_s'] <= 15.06985 and
            record['results']['original']['corner_comparisons']['historical_main']['not_worse'] >= 21)
        evaluations.append(record)
        write(STAGE/'evaluations.json', evaluations)
    eligible = [r for r in evaluations if r['eligible_initialization']]
    best = min(eligible, key=lambda r:(-r['results']['original']['corner_comparisons']['historical_main']['not_worse'],
        r['results']['original']['complete_time_s'])) if eligible else None
    write(STAGE/'result.json', dict(status='offline_fit_completed' if iteration==30000 else 'wall_budget_completed',
        offline_iterations=iteration, new_ppo_updates=0, inherited_ppo_updates=88,
        evaluations=evaluations, selected_initialization=best, adopted=False,
        retained_checkpoint=str(INITIAL), wall_s=time.perf_counter()-started,
        goal_complete=False, method_frozen=False))
    print('ROUND_FINISHED', iteration, 'eligible', len(eligible), flush=True)


def main():
    if '--worker' in sys.argv:
        worker()
        return
    manifest = verify_bound_inputs()
    assert sha(INITIAL) == '7cc6ee4971f0ae9a0832a3ddf63725de32609527aa4794fb144b99811fcb8ced'
    STAGE.mkdir(exist_ok=False)
    (STAGE/'source').mkdir()
    sources = [Path(__file__), ROOT/'compact_farvector_runtime.py', ROOT/'compact_research_eval.py',
               ROOT/'compact_fine_policy.py', ROOT/'compact_previewcue_runtime.py',
               ROOT/'compact_control.py', ROOT/'direct_compact_runtime.py',
               BASE/'paper_complete_evaluation.py', BASE/'natural_case_audit.py', BASE/'natural_case_metrics.py']
    hashes = {str(p): sha(p) for p in sources}
    for p in sources:
        shutil.copy2(p, STAGE/'source'/p.name)
    write(STAGE/'registration.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        kind='Fourteen-dimensional encoder feedback migration, offline only',
        evidence_gap='Round18 pointwise cost improved26corners but broke dense success. Test whether replacing observed redundant positive reference cosine by30mm tangent geometry preserves the successful feedback before fresh PPO.',
        source_sha256=hashes, historical_binding_manifest=manifest,
        initial_checkpoint=str(INITIAL), initial_checkpoint_sha256=sha(INITIAL),
        data_audit_sha256=sha(DATA/'audit.json'), wall_budget_s=1800,
        worker_deadline_s=1640, child_timeout_s=1740, iterations=30000,
        check_iterations=[8000,16000,30000], new_ppo_updates=0,
        procedure='Frozen round17teacher, equal original/dense supervision, alternate witnessed and synthetic neighborhood minibatches. Initialize replacing cosine coefficient by bias; standardized fitting and exact export back into same14x64x64x3actor. Synthetic labels only, no teacher, cosine reconstruction, or hidden policy memory at execution.',
        adoption='Only training initialization screen: both full successes,original<=444.316,dense<=15.06985,historical nonworse>=21. No final candidate adoption; full joint thresholds unchanged.',
        stop='30000iterations,three scheduled checks,or wall deadline; preserve all failures and retain round17 if no viable new encoder initialization.',
        goal_complete=False, method_frozen=False))
    started = time.perf_counter()
    code = None
    with (STAGE/'stdout.txt').open('w',encoding='utf-8') as out, (STAGE/'stderr.txt').open('w',encoding='utf-8') as err:
        try:
            run = subprocess.run([sys.executable,'-B',str(Path(__file__)),'--worker'], stdout=out,stderr=err,timeout=1740)
            status, code = 'finished', run.returncode
        except subprocess.TimeoutExpired:
            status = 'timeout'
    verify_bound_inputs()
    write(STAGE/'receipt.json', dict(status=status,exit_code=code,wall_s=time.perf_counter()-started,
        sources_unchanged=all(sha(p)==h for p,h in hashes.items()),goal_complete=False))
    print((STAGE/'stdout.txt').read_text(encoding='utf-8')[-4000:])
    if code != 0:
        print((STAGE/'stderr.txt').read_text(encoding='utf-8')[-3000:])


if __name__ == '__main__':
    main()
