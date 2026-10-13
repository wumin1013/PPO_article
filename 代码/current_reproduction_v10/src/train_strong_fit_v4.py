"""Single registered strong-structure pilot, fresh weights and historical data.

No final-test execution or PPO is included. The retained teacher and maps are
development assets; this is not a reconstruction of the historical PPO chain.
"""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'

import argparse
import hashlib
import json
import platform
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from strong_policy_v4 import (STATIC_KEYS, StrongPolicyV4, extract_static_assets,
                              tensor_hash, load_strong_policy)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/strong_fit_v4'
MODEL = ROOT/'baseline_records/supplementary_reproduction/models/main.pt'
DATA = ROOT/'baseline_records/supplementary_reproduction/training/recorded_actions.npz'
MODEL_HASH = 'c16d45e38bab022ce8f289148aac4f3dcb3559f98b18bdfa580e23aea4a99764'
DATA_HASH = '469c8d397d3a9d24b87d6fd399b0c78aa0dd7154dba9edcd8d4b9c5f68b51971'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def equivalent_loss(prediction, target, scale):
    error = prediction-target
    error = torch.where(target >= 1., -torch.relu(1.02-prediction), error)
    error = torch.where(target <= -1., torch.relu(prediction+1.02), error)
    return (error/scale).square().mean()


def prepare_assets():
    if sha(MODEL) != MODEL_HASH or sha(DATA) != DATA_HASH:
        raise ValueError('Historical model or dataset differs from bound source')
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT/'shared_static_assets.pt'
    assets = extract_static_assets(MODEL)
    if path.exists():
        existing = torch.load(path, map_location='cpu', weights_only=False)
        if existing['tensor_sha256'] != assets['tensor_sha256']:
            raise ValueError('Existing static feature assets differ')
    else:
        torch.save(assets, path)
    return assets, path


def numerical_audit(policy, assets, x, preview, joint):
    from frozen_policy import FrozenPolicy
    historical = FrozenPolicy(MODEL)
    checks = []
    with torch.no_grad():
        for ids in torch.arange(len(x)).split(512):
            a = historical._features(x[ids], 'actor.backbone.')
            b = historical._features(x[ids], 'actor.')
            checks.append([float((a-preview[ids]).abs().max()),
                           float((b-joint[ids]).abs().max())])
        ids = torch.arange(min(512, len(x)))
        direct = policy.actor(x[ids])
        cached = policy.actor.from_features(x[ids], preview[ids], joint[ids])
        value = x[ids]
        for index in (0, 2, 4):
            layer = policy.actor.base[index]
            value = F.linear(value, layer.weight, layer.bias)
            if index != 4:
                value = torch.tanh(value)
        delta = .4*torch.tanh(F.linear(preview[ids], policy.actor.preview_correction.weight))
        manual = torch.cat([value[:, :2], value[:, 2:3]+delta], dim=-1)
        manual += torch.tanh(F.linear(joint[ids], policy.actor.joint_correction.weight))*policy.actor.output_scale
        paired = StrongPolicyV4(policy.seed, assets, 'fixed')
        equal = all(torch.equal(value, dict(paired.named_parameters())[key])
                    for key, value in policy.named_parameters())
        fixed_physical = paired.decoded(x[ids])
    maxima = np.max(checks, axis=0)
    result = dict(preview_feature_max_error=float(maxima[0]), joint_feature_max_error=float(maxima[1]),
                  cached_forward_max_error=float((direct-cached).abs().max()),
                  manual_expression_max_error=float((direct-manual).abs().max()),
                  paired_allocated_trainable_tensors_identical=equal,
                  fixed_preview_exactly_3mm=bool(np.all(fixed_physical[:, 2] == 3.)),
                  historical_trainable_parameters_loaded_into_policy=False,
                  reference='FrozenPolicy._features; only reference evaluation reads historical trainable tensors',
                  checked_observations=len(x))
    assert max(result[k] for k in ('preview_feature_max_error', 'joint_feature_max_error',
                                  'cached_forward_max_error', 'manual_expression_max_error')) == 0.
    assert equal and result['fixed_preview_exactly_3mm']
    return result


def fit(seed=101, arm='adaptive'):
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(seed)
    assets, assets_path = prepare_assets()
    stage = OUT/f'pilot_seed{seed}'/arm
    stage.mkdir(parents=True, exist_ok=False)
    with np.load(DATA) as data:
        x = torch.tensor(data['observations'], dtype=torch.float32)
        y = torch.tensor(data['targets_three' if arm == 'adaptive' else 'targets_two'], dtype=torch.float32)
        groups = [np.flatnonzero(data['case'] == case) for case in ('original', 'dense')]
    policy = StrongPolicyV4(seed, assets, arm)
    with torch.no_grad():
        features = [policy.actor.features(part) for part in x.split(512)]
        preview = torch.cat([v[0] for v in features])
        joint = torch.cat([v[1] for v in features])
    audit = numerical_audit(policy, assets, x, preview, joint)
    write(stage/'initialization_audit.json', audit)
    inventory_path = ROOT/'audit/original_strong_recipe_inventory.json'
    inventory = json.loads(inventory_path.read_text('utf-8'))
    source_hashes = {p.name: sha(p) for p in [Path(__file__), ROOT/'src/strong_policy_v4.py']}
    registration = dict(created_utc=datetime.now(timezone.utc).isoformat(), version='v4_strong_fit_pilot_1',
        scope='Historical-development teacher-assisted fitting; no independent-test or PPO result',
        seed=seed, arm=arm, permitted_run='Single seed-101 adaptive pilot only; other arms require separate dispatch',
        model_architecture='14-64-64-3 tanh hidden layers; 96 preview plus 192 joint Wendland features',
        allocated_network_identical_across_arms=True, effective_outputs=policy.action_dim, fixed_preview_mm=3.,
        initialization='Seeded torch Linear reset for every base weight/bias; zero preview/joint correction weights; no pretrained trainable parameter copied',
        initial_trainable_tensor_sha256=tensor_hash(dict(policy.named_parameters())),
        trainable_parameter_count=sum(p.numel() for p in policy.parameters()),
        static_assets=str(assets_path.relative_to(ROOT)), static_assets_sha256=sha(assets_path),
        shared_map_tensor_sha256=assets['tensor_sha256'], static_asset_keys=STATIC_KEYS,
        historical_map_checkpoint_sha256=MODEL_HASH, dataset_sha256=DATA_HASH, dataset=str(DATA.relative_to(ROOT)),
        historical_development_paths=['original', 'dense'], observations=len(x), case_samples=[len(g) for g in groups],
        offline_iterations=12000, minibatch=512, samples_per_case_per_iteration=256,
        optimizer='Adam', learning_rate=2e-4, lr_multipliers_after_iterations={'4000': .3, '8000': .3},
        equivalent_action_loss_scales=[.015, .15, .3][:policy.action_dim], boundary_latent_margin=.02,
        gradient_norm_limit=1., checkpoints=[4000, 8000, 12000], selection='Fixed final 12000 iteration; no outcome-conditioned selection',
        new_ppo_updates=0, new_environment_interactions=0, historical_88_update_pipeline_reproduced=False,
        all_samples_retained=True, fixed_label_conversion='Preserve guide+steering at 3mm when representable; retain all 20 clipped labels',
        fixed_label_clipping_audit=inventory['matched_dataset_audit']['records'],
        shared_teacher_upstream=dict(known_seed923202_stage_wall_s=inventory['historical_seed923202_receipt_total_wall_s'],
            inherited_ppo_updates=88, executed_ppo_updates=96, executed_ppo_decisions=96*2048,
            offline_fit_iteration_and_epoch_total=99400, linear_fit_count=4,
            missing_training_inputs=[p for p in inventory['upstream_inputs'] if not p['exists']],
            excluded_costs='Earlier teacher discovery/search and final historical matched fit are not included in the stage receipt sum; total discovery cost unavailable',
            cost_scope='Historical shared assets, not repeated or charged as new on-policy samples'),
        source_sha256=source_hashes, python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
        cpu=dict(processor=platform.processor(),machine=platform.machine(),threads=torch.get_num_threads()),
        cost_scope='Process CPU and wall time on shared host; no hard-real-time claim',
        no_final_test_execution=True, allow_new_ppo=False)
    write(stage/'registration.json', registration)
    registration_hash = sha(stage/'registration.json')
    optimizer = torch.optim.Adam(policy.actor.parameters(), lr=2e-4)
    scales = torch.tensor([.015, .15, .3][:policy.action_dim])

    def checkpoint(path, step):
        torch.save(dict(format='strong_policy_v4', policy=policy.state_dict(), optimizer=optimizer.state_dict(),
            seed=seed, arm=arm, fixed_preview_mm=3., offline_iterations=step, new_ppo_updates=0,
            version=registration['version'], registration_sha256=registration_hash,
            shared_static_assets_sha256=sha(assets_path), dataset_sha256=DATA_HASH,
            pretrained_trainable_parameters_copied=False,
            trainable_tensor_sha256=tensor_hash(dict(policy.named_parameters()))), path)

    checkpoint(stage/'initial.pt', 0)
    initial_hash = sha(stage/'initial.pt')
    preparation_wall = time.perf_counter()-wall_start
    preparation_cpu = time.process_time()-cpu_start
    fit_wall, fit_cpu = time.perf_counter(), time.process_time()
    summaries = []
    step = 0
    try:
        with (stage/'losses.jsonl').open('x', encoding='utf-8') as log:
            for step in range(1, 12001):
                ids = torch.from_numpy(np.concatenate([g[rng.integers(len(g), size=256)] for g in groups]))
                prediction = policy.actor.from_features(x[ids], preview[ids], joint[ids])[:, :policy.action_dim]
                loss = equivalent_loss(prediction, y[ids], scales)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss')
                lr_used = optimizer.param_groups[0]['lr']
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(policy.actor.parameters(), 1.)
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError('Nonfinite gradient norm')
                optimizer.step()
                if step in (4000, 8000):
                    for group in optimizer.param_groups:
                        group['lr'] *= .3
                log.write(json.dumps(dict(iteration=step, loss=float(loss.detach()),
                    gradient_norm_before_clip=float(gradient_norm), learning_rate=lr_used,
                    fit_wall_s=time.perf_counter()-fit_wall, fit_cpu_s=time.process_time()-fit_cpu))+'\n')
                if step % 500 == 0:
                    log.flush()
                    with torch.no_grad():
                        predictions = torch.cat([policy.actor.from_features(x[idx], preview[idx], joint[idx])[:, :policy.action_dim]
                                                 for idx in torch.arange(len(x)).split(512)])
                        delta = predictions.clamp(-1, 1)-y.clamp(-1, 1)
                        summary = dict(iteration=step, sampled_loss=float(loss.detach()),
                            full_dataset_equivalent_loss=float(equivalent_loss(predictions, y, scales)),
                            clipped_action_rmse=delta.square().mean(0).sqrt().tolist(),
                            maximum_clipped_error=delta.abs().max(0).values.tolist(),
                            fit_wall_s=time.perf_counter()-fit_wall, fit_cpu_s=time.process_time()-fit_cpu)
                    summaries.append(summary)
                    write(stage/'fit_log.json', summaries)
                    print('STRONG_FIT_V4', seed, arm, step, summary['clipped_action_rmse'], flush=True)
                if step in (4000, 8000, 12000):
                    checkpoint(stage/f'fit_{step:05d}.pt', step)
        shutil.copy2(stage/'fit_12000.pt', stage/'final.pt')
        loaded, _ = load_strong_policy(stage/'final.pt')
        with torch.no_grad():
            replay_error = float((loaded.mean(x[:512])-policy.mean(x[:512])).abs().max())
        assert replay_error == 0.
        assert sha(DATA) == DATA_HASH and sha(MODEL) == MODEL_HASH
        assert all(sha(ROOT/'src'/name) == value for name, value in source_hashes.items())
        receipt = dict(status='completed', version=registration['version'], seed=seed, arm=arm,
            offline_iterations=step, supervised_sample_draws=step*512,
            new_ppo_updates=0, new_environment_interactions=0, completion_or_efficiency_not_yet_evaluated=True,
            initial_checkpoint_sha256=initial_hash, model_sha256=sha(stage/'final.pt'),
            final_trainable_tensor_sha256=tensor_hash(dict(policy.named_parameters())),
            registration_sha256=registration_hash, source_unchanged=True,
            reload_max_action_error=replay_error, losses_recorded=step,
            losses_sha256=sha(stage/'losses.jsonl'), final_fit_summary=summaries[-1],
            preparation_wall_s=preparation_wall, preparation_cpu_s=preparation_cpu,
            fit_wall_s=time.perf_counter()-fit_wall, fit_cpu_s=time.process_time()-fit_cpu,
            total_wall_s=time.perf_counter()-wall_start, total_cpu_s=time.process_time()-cpu_start,
            statistical_unit='One training run; loss iterations and teacher rows are not independent seed replicates')
        write(stage/'receipt.json', receipt)
        print(json.dumps(receipt), flush=True)
    except Exception as exc:
        checkpoint(stage/'interrupted.pt', step)
        write(stage/'receipt.json', dict(status='failed', iteration=step,
            error=type(exc).__name__+': '+str(exc), new_ppo_updates=0,
            total_wall_s=time.perf_counter()-wall_start, total_cpu_s=time.process_time()-cpu_start,
            initial_checkpoint_sha256=initial_hash))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=101)
    parser.add_argument('--arm', choices=['adaptive', 'fixed'], default='adaptive')
    args = parser.parse_args()
    if (args.seed, args.arm) != (101, 'adaptive'):
        parser.error('Only seed 101 adaptive is authorized by this pilot registration')
    fit(args.seed, args.arm)
