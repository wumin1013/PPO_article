"""Auditable bounded PPO experiments, with a separate frozen test gate.

All learning variants retain the same 18-input/3-output network. Fixed preview
excludes the inactive output from the likelihood and entropy, and executes two
actions. Training uses only the three designated training geometries.
"""
from __future__ import annotations
import argparse
import contextlib
import copy
import csv
import hashlib
import io
import json
import platform
import random
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
sys.path.insert(0, str(PROJECT))
from src.algorithms.ppo import PPOContinuous
from src.environment.cnc_env import create_environment_from_config
from src.utils.revision_paths import generate_suite, load_manifest, load_path, DEFAULT_DATA_DIR
from src.utils.revision_metrics import evaluate_trajectory, summarize_runs
from main import _build_path

OUT = ROOT / '论文项目/revision_20260908/experiments'
VARIANTS = ('coupled', 'decoupled', 'fixed_short', 'fixed_mid', 'fixed_long')
DISTANCES = dict(fixed_short=1.2, fixed_mid=2.3, fixed_long=4.0)
# The first sealed runner was used for the completed matrix. This one allowed
# upgrade only hardens metadata/cache checks; controller, rewards, metrics and
# selection rules are unchanged. Every other source file must match bytewise.
GUARD_ONLY_PREDECESSOR = '2c48a1969c28b2d3dbf328151adbebd9e796a30054733b96acef6c91df47abce'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def source_state():
    files = sorted(list((PROJECT / 'src').rglob('*.py')) +
                   [Path(__file__), PROJECT / 'main.py'])
    hashes = {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in files}
    code_hash = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'source_sha256': code_hash, 'source_files': hashes,
            'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
            'platform': platform.platform(), 'hardware': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
            'execution_device': 'cpu', 'torch_threads': 1}


def check_source_compatibility(recorded, current):
    if recorded['source_sha256'] == current['source_sha256']:
        return
    runner = 'PPO_project/tools/revision_experiment.py'
    old, new = dict(recorded['source_files']), dict(current['source_files'])
    prior_runner = old.pop(runner, None)
    new.pop(runner, None)
    if prior_runner != GUARD_ONLY_PREDECESSOR or old != new:
        raise ValueError('Source changed since checkpoint; use its archived source or a separately labelled study')


def prepare():
    generate_suite()
    path = OUT / 'protocol.json'
    protocol = {'version': '20260908-v1', 'seeds_screen': [101], 'seeds_main': [11, 22, 33, 44, 55],
                'screen_steps_per_run': 4096, 'main_steps_per_run': 8192,
                'long_run_steps_per_run_pending': 300000, 'rollout_steps': 512,
                'evaluation_max_steps': 12000, 'training_episode_max_steps': 1024, 'fixed_distances_mm': DISTANCES,
                'candidate_change': 'cornerness reward uses a fixed 4.6 mm reference; observation and control preview remain learned',
                'shared_changes': ['latent Gaussian PPO likelihood', 'same actor shape with inactive action masking',
                                   'ordered progress and termination', 'command-consistent KCM',
                                   'remove historical path-specific teacher timing; constant blend 0.08 in train and evaluation',
                                   'terminate out-of-band episodes; 1024-step training truncation ensures path rotation'],
                'selection_rule': 'validation only: strict successes, geometric completions, mean ordered progress, then successful mean time; no speed ranking of failures',
                'time_comparison': 'common successful training-seed/path pairs only; also report all failures',
                'test_gate': 'freeze selection and all model/config hashes before first test; never tune from test results',
                'budget_interpretation': 'bounded five-training-seed replication; 8192 steps does not establish convergence',
                'extra_pretraining_steps': 0, 'constraint_regimes': {
                    'nominal': {'epsilon': 2.5, 'MAX_JERK': 62500.0},
                    'tight': {'epsilon': 1.25, 'MAX_JERK': 62500.0},
                    'low_jerk': {'epsilon': 2.5, 'MAX_JERK': 31250.0}},
                'initial_state': 'reference start, tangent heading, zero feed/acceleration and steering rates',
                'terminal_state': 'ordered endpoint coverage; terminal speed reported; no terminal-rest guarantee',
                'manifest_sha256': sha(DEFAULT_DATA_DIR / 'manifest.json')}
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != protocol:
        raise ValueError('Protocol already frozen with different settings; use a separately versioned study')
    write_json(path, protocol)
    print(path, flush=True)


def config_for(variant, seed):
    cfg = yaml.safe_load((ROOT / '论文项目/generated/configs/full_method_snapshot.yaml').read_text(encoding='utf-8'))
    cfg = copy.deepcopy(cfg)
    cfg['seed'] = seed
    cfg['environment']['max_steps'] = 1024
    cfg['environment']['lookahead_obs_enabled'] = True
    cfg['experiment']['enable_kcm'] = True
    cfg['reward_weights']['teacher_progress']['enabled'] = False
    cfg['reward_weights'].setdefault('boundary', {})['enabled'] = True
    cfg['reward_weights']['control_authority']['blend_schedule'] = 'constant'
    cfg['reward_weights']['control_authority']['warmup_episodes'] = 0
    cfg['reward_weights']['cornerness']['reference_distance'] = None if variant == 'coupled' else 4.6
    cfg['reward_weights']['lookahead_control']['enabled'] = variant not in DISTANCES
    cfg['reward_weights']['lookahead_control']['policy_action'] = variant not in DISTANCES
    if variant in DISTANCES:
        cfg['reward_weights']['lookahead']['distance'] = DISTANCES[variant]
    cfg['revision'] = {'variant': variant, 'seed': seed, 'active_action_dims': [0, 1] if variant in DISTANCES else [0, 1, 2],
                       'network_action_dim': 3, 'initialization': 'independent fresh random training seed',
                       'pretraining_steps': 0, 'training_path_ids': [p['id'] for p in load_manifest()['paths'] if p['split'] == 'train']}
    cfg.pop('autoresearch', None)
    cfg.pop('path_curriculum', None)
    cfg['training'] = {'use_obs_normalizer': False, 'rollout_steps': 512,
                       'path_order': 'round robin; train geometries only'}
    return cfg


def make_env(cfg, path_id, episode=0, reference=None):
    points = load_path(path_id) if reference is None else reference
    local = copy.deepcopy(cfg)
    local['path'] = {'name': path_id, 'type': 'waypoints'}
    with contextlib.redirect_stdout(io.StringIO()):
        env = create_environment_from_config(local, points, device=torch.device('cpu'))
        env._episode_index = episode - 1
        state = env.reset(random_start=False)
    return env, state


def make_agent(cfg, env):
    keys = ('hidden_dim', 'actor_lr', 'critic_lr', 'lmbda', 'epochs', 'eps', 'gamma', 'ent_coef')
    return PPOContinuous(state_dim=env.observation_dim, action_dim=3, device=torch.device('cpu'),
                         active_action_dims=cfg['revision']['active_action_dims'],
                         **{k: cfg['ppo'][k] for k in keys})


def sample_row(env, info, reward, latent):
    row = {'timestamp_s': env.current_step * env.interpolation_period,
           'x_mm': float(env.current_position[0]), 'y_mm': float(env.current_position[1]),
           'v_mm_s': float(env.velocity), 'a_mm_s2': float(env.acceleration), 'j_mm_s3': float(env.jerk),
           'omega_rad_s': float(env.angular_vel), 'alpha_rad_s2': float(env.angular_acc),
           'angular_jerk_rad_s3': float(env.angular_jerk), 'heading_rad': float(env.heading),
           'lookahead_mm': float(env._lookahead_dist_active), 'reward': float(reward),
           'kcm_intervention': float(info.get('kcm_intervention', 0)),
           'progress_env': float(info.get('progress', 0)), 'contour_error_env_mm': float(info.get('contour_error', 0))}
    for i, value in enumerate(latent):
        row[f'latent_{i}'] = float(value)
        row[f'bounded_{i}'] = float(np.clip(value, -1 if i == 0 else 0, 1))
    for group in ('reward_components', 'control_status', 'p4_status'):
        for key, value in info.get(group, {}).items():
            if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
                row[group + '__' + key] = float(value)
    return row


def save_trace(path, rows):
    fields = sorted(set().union(*(r.keys() for r in rows)))
    array = np.array([[r.get(k, np.nan) for k in fields] for r in rows], dtype=np.float64)
    np.savez_compressed(path, columns=np.array(fields), data=array)


def train(variant, seed, stage, total_steps):
    torch.set_num_threads(1)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    cfg = config_for(variant, seed)
    run = OUT / stage / f'{variant}_seed{seed}'
    run.mkdir(parents=True, exist_ok=True)
    cfg_path = run / 'config.yaml'
    state_info = source_state()
    if cfg_path.exists():
        if yaml.safe_load(cfg_path.read_text(encoding='utf-8')) != cfg:
            raise ValueError('Refusing to resume with changed configuration')
        provenance = json.loads((run / 'provenance.json').read_text(encoding='utf-8'))
        check_source_compatibility(provenance, state_info)
        if provenance['path_manifest_sha256'] != sha(DEFAULT_DATA_DIR / 'manifest.json'):
            raise ValueError('Training path manifest changed since checkpoint')
    else:
        cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding='utf-8')
        state_info.update(command='python ' + ' '.join(sys.argv), config_sha256=sha(cfg_path),
                          path_manifest_sha256=sha(DEFAULT_DATA_DIR / 'manifest.json'))
        write_json(run / 'provenance.json', state_info)
        with zipfile.ZipFile(run / 'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for name in state_info['source_files']:
                archive.write(ROOT / name, name)
    ids = cfg['revision']['training_path_ids']
    env, obs = make_env(cfg, ids[0])
    agent = make_agent(cfg, env)
    steps = episode = 0
    elapsed_before = 0.0
    episode_actions = []
    ep_return = 0.0
    checkpoint_path = run / 'checkpoint.pt'
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        agent.actor.load_state_dict(ckpt['actor']); agent.critic.load_state_dict(ckpt['critic'])
        agent.actor_optimizer.load_state_dict(ckpt['actor_optimizer'])
        agent.critic_optimizer.load_state_dict(ckpt['critic_optimizer'])
        steps, episode = ckpt['steps'], ckpt['episode']
        elapsed_before, ep_return = ckpt['training_wall_s'], ckpt['ep_return']
        episode_actions = ckpt['episode_actions']
        env, obs = make_env(cfg, ids[episode % len(ids)], episode)
        for action in episode_actions:
            obs, _, _, _ = env.step(action)
        torch.set_rng_state(ckpt['torch_rng']); np.random.set_state(ckpt['numpy_rng']); random.setstate(ckpt['python_rng'])
    if steps >= total_steps:
        print('Already trained', run, steps, flush=True)
        return run
    started = time.perf_counter()
    buffer = {k: [] for k in ('states', 'actions', 'next_states', 'rewards', 'dones', 'terminated')}
    trace = [sample_row(env, {}, 0, [0, 0, .5])]
    trace_start = steps
    updates = []
    def save_checkpoint():
        torch.save(dict(actor=agent.actor.state_dict(), critic=agent.critic.state_dict(),
                        actor_optimizer=agent.actor_optimizer.state_dict(), critic_optimizer=agent.critic_optimizer.state_dict(),
                        steps=steps, episode=episode, ep_return=ep_return, episode_actions=episode_actions,
                        training_wall_s=elapsed_before + time.perf_counter()-started,
                        torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(), python_rng=random.getstate()), checkpoint_path)
    while steps < total_steps:
        latent = agent.take_action(obs)
        executed = np.clip(latent[:env.action_space_dim], env.action_space.low, env.action_space.high)
        nxt, reward, done, info = env.step(executed)
        episode_actions.append(executed.tolist())
        ep_return += reward
        terminal = done and info.get('done_reason') not in ('max_steps', 'timeout', 'time_limit')
        for key, value in zip(buffer, (obs, latent, nxt, reward, done, terminal)):
            buffer[key].append(value)
        trace.append(sample_row(env, info, reward, latent))
        obs = nxt; steps += 1
        if len(buffer['states']) == 512 or done or steps == total_steps:
            losses = agent.update(buffer)
            updates.append({'step': steps, 'actor_loss': losses[0], 'critic_loss': losses[1]})
            buffer = {k: [] for k in buffer}
        if done or steps == total_steps:
            save_trace(run / f'train_trace_{trace_start:08d}_{steps:08d}.npz', trace)
            with (run / 'episodes.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(dict(episode=episode, path_id=ids[episode % len(ids)], steps=steps,
                                             episode_steps=env.current_step, reward=ep_return,
                                             done=done, done_reason=info.get('done_reason'),
                                             budget_truncation=not done), allow_nan=False) + '\n')
            if done:
                episode += 1; episode_actions = []; ep_return = 0.0
                env, obs = make_env(cfg, ids[episode % len(ids)], episode)
            save_checkpoint()
            trace_start = steps
            trace = [sample_row(env, {}, 0, [0, 0, .5])]
            print(f'{stage} {variant} seed={seed} steps={steps}/{total_steps} episode={episode}', flush=True)
    with (run / 'updates.jsonl').open('a', encoding='utf-8') as handle:
        for row in updates:
            handle.write(json.dumps(row) + '\n')
    write_json(run / 'training_summary.json', dict(variant=variant, seed=seed, steps=steps, episodes=episode,
               training_wall_s=elapsed_before + time.perf_counter()-started,
               pretraining_steps=int(cfg['revision'].get('pretraining_steps', 0)),
               total_learning_steps=steps + int(cfg['revision'].get('pretraining_steps', 0)),
               checkpoint_sha256=sha(checkpoint_path), status='bounded_budget_complete',
               convergence_established=False, stage=stage))
    return run


def evaluate(run, split, regime='nominal', diagnostic_fixed=None):
    torch.set_num_threads(1)
    run = Path(run)
    cfg = yaml.safe_load((run / 'config.yaml').read_text(encoding='utf-8'))
    provenance = json.loads((run / 'provenance.json').read_text(encoding='utf-8'))
    current_state = source_state()
    current_source = current_state['source_sha256']
    check_source_compatibility(provenance, current_state)
    if provenance['config_sha256'] != sha(run / 'config.yaml'):
        raise ValueError('Training configuration was modified; use an explicit transfer diagnostic')
    if provenance['path_manifest_sha256'] != sha(DEFAULT_DATA_DIR / 'manifest.json'):
        raise ValueError('Training path manifest changed since checkpoint')
    training_source = provenance['source_sha256']
    if split == 'test':
        freeze = json.loads((OUT / 'freeze.json').read_text(encoding='utf-8'))
        key = str(run.relative_to(OUT)).replace('\\', '/')
        if freeze['models'].get(key) != sha(run / 'checkpoint.pt'):
            raise ValueError('Model is not the frozen test checkpoint')
        if freeze['configs'].get(key) != sha(run / 'config.yaml') or freeze['source_sha256'] not in (training_source, current_source):
            raise ValueError('Frozen configuration/source changed')
        if freeze['path_manifest_sha256'] != sha(DEFAULT_DATA_DIR / 'manifest.json'):
            raise ValueError('Frozen path manifest changed')
    cfg['environment']['max_steps'] = int(cfg['revision'].get('episode_max_steps', 12000))
    if regime == 'tight': cfg['environment']['epsilon'] = 1.25
    if regime == 'low_jerk': cfg['kinematic_constraints']['MAX_JERK'] = 31250.0
    if diagnostic_fixed is not None:
        cfg['reward_weights']['lookahead_control']['enabled'] = False
        cfg['reward_weights']['lookahead_control']['policy_action'] = False
        cfg['reward_weights']['lookahead']['distance'] = diagnostic_fixed
    out = run / ('evaluation_' + split + '_' + regime + ('' if diagnostic_fixed is None else f'_fixed_{diagnostic_fixed:g}_diagnostic'))
    out.mkdir(exist_ok=True)
    effective_path = out / 'effective_config.yaml'
    if effective_path.exists() and yaml.safe_load(effective_path.read_text(encoding='utf-8')) != cfg:
        raise ValueError('Existing evaluation used a different effective configuration; preserve it')
    effective_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding='utf-8')
    if split == 'regression':
        historical = yaml.safe_load((ROOT / '论文项目/generated/configs/full_method_snapshot.yaml').read_text(encoding='utf-8'))
        specs = [(p['name'], _build_path(p)) for p in historical['training']['path_curriculum']['paths']]
    else:
        specs = [(p['id'], load_path(p['id'])) for p in load_manifest()['paths'] if p['split'] == split]
    all_results = []
    for path_id, points in specs:
        result_path = out / (path_id + '.json')
        if result_path.exists():
            previous = json.loads(result_path.read_text(encoding='utf-8'))
            if previous['checkpoint_sha256'] != sha(run / 'checkpoint.pt') or previous['config_sha256'] != sha(run / 'config.yaml') or previous['code_sha256'] not in (training_source, current_source):
                raise ValueError('Existing result belongs to another model/config/source; preserve it and use a new run')
            all_results.append(previous); continue
        env, obs = make_env(cfg, path_id, reference=points)
        agent = make_agent(cfg, env)
        ckpt = torch.load(run / 'checkpoint.pt', map_location='cpu', weights_only=False)
        agent.actor.load_state_dict(ckpt['actor']); agent.actor.eval()
        rows = [sample_row(env, {}, 0, [0, 0, .5])]
        latencies = []; info = {}; done = False
        start = time.perf_counter()
        with torch.no_grad():
            while not done:
                t0 = time.perf_counter_ns()
                latent = agent.actor(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))[0][0].numpy()
                latencies.append((time.perf_counter_ns()-t0) / 1000)
                executed = np.clip(latent[:env.action_space_dim], env.action_space.low, env.action_space.high)
                obs, reward, done, info = env.step(executed)
                rows.append(sample_row(env, info, reward, latent))
        save_trace(out / (path_id + '.npz'), rows)
        positions = [[r['x_mm'], r['y_mm']] for r in rows]
        limits = cfg['kinematic_constraints']
        metrics = evaluate_trajectory(points, positions, [r['timestamp_s'] for r in rows],
                     tolerance_mm=cfg['environment']['epsilon']/2, limits=limits,
                     termination_reason=info.get('done_reason', 'unknown'), claimed_success=info.get('done_reason') == 'success',
                     initial_velocity=[0., 0.], initial_acceleration=[0., 0.], initial_heading=rows[0]['heading_rad'],
                     initial_angular_velocity=0., initial_angular_acceleration=0.)
        metrics.update(variant=cfg['revision']['variant'], seed=cfg['seed'], path_id=path_id, split=split, regime=regime,
                       training_steps=ckpt['steps'], pretraining_steps=int(cfg['revision'].get('pretraining_steps', 0)),
                       total_learning_steps=ckpt['steps'] + int(cfg['revision'].get('pretraining_steps', 0)),
                       checkpoint_sha256=sha(run / 'checkpoint.pt'),
                       config_sha256=sha(run / 'config.yaml'), code_sha256=training_source,
                       evaluation_source_sha256=current_source, effective_config_sha256=sha(effective_path),
                       inference_median_us=float(np.median(latencies)), inference_p95_us=float(np.percentile(latencies, 95)),
                       evaluation_wall_s=time.perf_counter()-start, terminal_feed_mm_s=rows[-1]['v_mm_s'],
                       comparison_role='independently trained' if diagnostic_fixed is None else 'output replacement only; not independent fixed training')
        write_json(result_path, metrics); all_results.append(metrics)
        print(f'eval {run.name} {path_id} {regime}: success={metrics["success"]} failures={metrics["failure_reasons"]}', flush=True)
    write_json(out / 'summary.json', {'rows': all_results, 'summary': summarize_runs(all_results)})
    return all_results


def freeze():
    if (OUT / 'freeze.json').exists():
        raise FileExistsError('Test freeze already exists; preserve it')
    rankings = []
    for variant in VARIANTS:
        summary = OUT / 'screen' / f'{variant}_seed101/evaluation_validation_nominal/summary.json'
        payload = json.loads(summary.read_text(encoding='utf-8'))
        rows = payload['rows']
        successful_times = [r['elapsed_time_s'] for r in rows if r['success']]
        score = (sum(r['success'] for r in rows), sum(r['path_completed'] for r in rows),
                 float(np.mean([r['ordered_progress'] for r in rows])),
                 -float(np.mean(successful_times)) if successful_times else -1e9)
        rankings.append({'variant': variant, 'score': list(score), 'validation_summary': str(summary)})
    fixed = max((r for r in rankings if r['variant'] in DISTANCES), key=lambda r: r['score'])['variant']
    models = {}; configs = {}
    for variant in ('coupled', 'decoupled', fixed):
        for seed in (11, 22, 33, 44, 55):
            run = OUT / 'main' / f'{variant}_seed{seed}'
            ckpt = torch.load(run / 'checkpoint.pt', map_location='cpu', weights_only=False)
            if ckpt['steps'] != 8192:
                raise ValueError('Every main model must have completed the predefined 8192-step budget')
            models[str(run.relative_to(OUT)).replace('\\', '/')] = sha(run / 'checkpoint.pt')
            configs[str(run.relative_to(OUT)).replace('\\', '/')] = sha(run / 'config.yaml')
    write_json(OUT / 'freeze.json', dict(selected_fixed=fixed, rankings=rankings, models=models, configs=configs,
                  timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  source_sha256=source_state()['source_sha256'], protocol_sha256=sha(OUT / 'protocol.json'),
                  path_manifest_sha256=sha(DEFAULT_DATA_DIR / 'manifest.json'),
                  test_results_used_for_selection=False, candidate='decoupled',
                  interpretation='candidate retained for diagnostic evaluation; no advantage presumed'))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare')
    p = sub.add_parser('train'); p.add_argument('--variant', choices=VARIANTS, required=True)
    p.add_argument('--seed', type=int, required=True); p.add_argument('--stage', default='screen')
    p.add_argument('--steps', type=int, default=4096)
    p = sub.add_parser('evaluate'); p.add_argument('--run', type=Path, required=True)
    p.add_argument('--split', choices=['validation', 'test', 'regression'], required=True)
    p.add_argument('--regime', choices=['nominal', 'tight', 'low_jerk'], default='nominal')
    p.add_argument('--diagnostic-fixed', type=float)
    sub.add_parser('freeze')
    args = parser.parse_args()
    if args.command == 'prepare': prepare()
    elif args.command == 'train': train(args.variant, args.seed, args.stage, args.steps)
    elif args.command == 'evaluate': evaluate(args.run, args.split, args.regime, args.diagnostic_fixed)
    else: freeze()


if __name__ == '__main__':
    main()
