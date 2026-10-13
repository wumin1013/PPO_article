"""Validation-only causal probes of frozen preview control, without training.

Replay saved actions exactly, fork identical environment states, and intervene
only on distance. Short forks retain the original future physical actions to
isolate the preview channel; they are not closed-loop performance comparisons.
The original frozen source/model/config/result files are never written.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPLAY = ROOT / '论文项目/revision_20260908/frozen_replay_explore'
STUDY = Path('论文项目/revision_20260908/preview_direct_explore')
DEFAULT_OUTPUT = ROOT / '论文项目/revision_20260909_cause_analysis/sensitivity'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def clone_environment(env):
    """Explicitly copy forwarding wrappers; generic deepcopy recurses in getattr."""
    # Rtree's deepcopy silently produces an empty native index. The index is
    # populated only during initialization and is read-only during step.
    spatial = env.raw.rtree_idx
    raw = copy.deepcopy(env.raw, {id(spatial): spatial} if spatial is not None else {})
    objective = object.__new__(type(env.env))
    objective.__dict__.update(copy.deepcopy({k: v for k, v in env.env.__dict__.items() if k != 'env'}))
    objective.env = raw
    result = object.__new__(type(env))
    result.__dict__.update(copy.deepcopy({k: v for k, v in env.__dict__.items() if k not in ('env', 'raw')}))
    result.env, result.raw = objective, raw
    return result


def distance_action(action, distance):
    result = np.asarray(action, dtype=float).copy()
    result[2] = (distance - .8) / 3.8
    return result


def record(env, info, reward):
    raw = env.raw
    result = dict(x_mm=float(raw.current_position[0]), y_mm=float(raw.current_position[1]),
        heading_rad=float(raw.heading), v_mm_s=float(raw.velocity), omega_rad_s=float(raw.angular_vel),
        alpha_rad_s2=float(raw.angular_acc), angular_jerk_rad_s3=float(raw.angular_jerk),
        theta_ref_rad=float(raw._heading_intent_status['theta_ref']),
        omega_intent_rad_s=float(raw._heading_intent_status['heading_rate_intent']),
        kcm_intervention=float(raw.kcm_intervention), reward=float(reward),
        error_mm=float(raw.get_contour_error(raw.current_position)),
        covered_s_mm=float(env.env.coverage.covered_s_mm),
        done_reason=info.get('done_reason', ''),
        reward_geometry_delta=float(raw._get_reward_geometry_delta()))
    result.update({f'observation_{i}': float(v) for i, v in enumerate(env.state)})
    result.update({f'reward_{k}': float(v) for k, v in info['reward_components'].items()
                   if k.startswith('r_') or k in ('w_track', 'w_smooth', 'cornerness')})
    return result


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize_pairs(rows, low, high):
    groups = {}
    for row in rows:
        groups.setdefault((row['variant'], row['path_id'], row['source_step']), {})[row['distance_mm']] = row
    pairs = [(group[low], group[high]) for group in groups.values()]
    changes = {key: np.array([abs(b[key] - a[key]) for a, b in pairs]) for key in
               ('theta_ref_rad', 'omega_intent_rad_s', 'omega_rad_s', 'v_mm_s', 'reward', 'reward_geometry_delta')}
    intent_changed = changes['omega_intent_rad_s'] > 1e-9
    unchanged = changes['omega_rad_s'] <= 1e-12
    return dict(distance_pair_mm=[low, high], state_count=len(pairs),
        intent_changed_count=int(intent_changed.sum()),
        identical_omega_despite_changed_intent_count=int((unchanged & intent_changed).sum()),
        identical_omega_given_changed_intent_fraction=float(unchanged[intent_changed].mean()) if intent_changed.any() else None,
        identical_feed_fraction=float((changes['v_mm_s'] <= 1e-12).mean()),
        identical_reward_fraction=float((changes['reward'] <= 1e-12).mean()),
        difference={key: dict(mean=float(value.mean()), maximum=float(value.max())) for key, value in changes.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['direct', 'direct_no_anchor'], default='direct')
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--stride', type=int, default=64)
    parser.add_argument('--branch-count', type=int, default=6)
    parser.add_argument('--horizon', type=int, default=128)
    parser.add_argument('--path', choices=['mixed_lengths_validation', 'dense_corners_validation', 'line_curve_validation'])
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if min(args.stride, args.branch_count, args.horizon) < 1:
        raise ValueError('Positive sampling/branch settings required')
    target = args.output / f'{args.variant}_seed{args.seed}'
    target.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    # This order ensures all numerical study imports use the immutable replay.
    sys.path.insert(0, str(REPLAY / 'PPO_project'))
    sys.path.insert(0, str(REPLAY / 'PPO_project/tools'))
    import torch
    import yaml
    import preview_direct_explore
    study = preview_direct_explore.install()
    study.check_protocol()
    study.activate_backend()
    torch.set_num_threads(1)
    run_key = f'formal/{args.variant}_seed{args.seed}'
    run = REPLAY / STUDY / run_key
    original = ROOT / STUDY / run_key
    cfg = yaml.safe_load((run / 'config.yaml').read_text(encoding='utf-8'))
    cfg['environment']['max_steps'] = cfg['revision']['episode_max_steps']
    paths = [p['id'] for p in study.manifest()['paths'] if p['split'] == 'validation' and (not args.path or p['id'] == args.path)]
    freeze = json.loads((study.OUT / 'freeze.json').read_text(encoding='utf-8'))
    if sha(run / 'checkpoint.pt') != freeze['models'][run_key] or sha(run / 'config.yaml') != freeze['configs'][run_key]:
        raise ValueError('Model/config differs from freeze')
    source = study.source_state()
    input_files = [run / 'checkpoint.pt', run / 'config.yaml', study.OUT / 'protocol.json', study.OUT / 'freeze.json']
    for path_id in paths:
        input_files.extend([original / 'evaluation_validation_nominal' / f'{path_id}.{suffix}' for suffix in ('npz', 'json')])
    hashes = {str(path): sha(path) for path in input_files}
    protocol = dict(command='python ' + ' '.join(sys.argv), cwd=str(Path.cwd()),
        timestamp_utc=datetime.now(timezone.utc).isoformat(), source_sha256=source['source_sha256'],
        script_sha256=sha(__file__), inputs_sha256=hashes, new_training_steps=0, test_used=False,
        args={**vars(args), 'output': str(args.output)},
        one_step_distances_mm=[2.81, 3., 3.19, 2.994, 3.006],
        branch_modes=['constant_3', 'hold_2.81', 'hold_3.19', 'impulse_2.81', 'impulse_3.19'],
        branch_physical_actions='Original deterministic replay actions; same for all forks; not recomputed from changed observations',
        interpretation='Same-state validation probes, not independent training, new test performance, or an optimized controller')
    save_json(target / 'protocol.json', protocol)
    (target / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    one_step, branch_rows, replay_checks = [], [], []
    for path_id in paths:
        saved = original / 'evaluation_validation_nominal' / f'{path_id}.npz'
        with np.load(saved, allow_pickle=False) as trace:
            columns, data = list(trace['columns']), trace['data']
        # Use the saved untransformed latent means, exactly as evaluation did.
        action_names = ['latent_0', 'latent_1', 'latent_2']
        if not all(name in columns for name in action_names):
            action_names = ['action_latent_0', 'action_latent_1', 'action_latent_2']
        if not all(name in columns for name in action_names):
            raise ValueError(f'Locate recorded latent action columns: {columns}')
        actions = data[1:, [columns.index(name) for name in action_names]]
        env, obs = study.make_env(cfg, path_id)
        branch_steps = set(np.linspace(0, max(0, len(actions) - args.horizon - 1), args.branch_count + 2, dtype=int)[1:-1])
        positions = [env.raw.current_position.copy()]
        clone_errors = []
        for index, action in enumerate(actions):
            if index % args.stride == 0:
                original_obs = env.state.copy()
                original_position = env.raw.current_position.copy()
                same = clone_environment(env)
                same_obs, same_reward, same_done, same_info = same.step(action)
                for distance in protocol['one_step_distances_mm']:
                    fork = clone_environment(env)
                    _, reward, _, info = fork.step(distance_action(action, distance))
                    one_step.append(dict(variant=args.variant, seed=args.seed, path_id=path_id,
                        source_step=index, distance_mm=distance, **record(fork, info, reward)))
                if not np.array_equal(env.state, original_obs) or not np.array_equal(env.raw.current_position, original_position):
                    raise AssertionError('A diagnostic fork changed the source environment')
            if index in branch_steps:
                for mode, distance, impulse in [('constant_3', 3., False), ('hold_2.81', 2.81, False),
                        ('hold_3.19', 3.19, False), ('impulse_2.81', 2.81, True), ('impulse_3.19', 3.19, True)]:
                    fork = clone_environment(env)
                    for offset, future_action in enumerate(actions[index:index+args.horizon]):
                        applied = 3. if impulse and offset else distance
                        _, reward, fork_done, info = fork.step(distance_action(future_action, applied))
                        branch_rows.append(dict(variant=args.variant, seed=args.seed, path_id=path_id,
                            source_step=index, branch_mode=mode, offset_step=offset+1,
                            distance_mm=applied, **record(fork, info, reward)))
                        if fork_done:
                            break
            obs, reward, done, info = env.step(action)
            if index % args.stride == 0:
                clone_errors.append(float(np.max(np.abs(obs - same_obs))))
                if not np.array_equal(obs, same_obs) or reward != same_reward or done != same_done:
                    failure = dict(path_id=path_id, step=index, observation=obs.tolist(),
                        clone_observation=same_obs.tolist(), reward=reward, clone_reward=same_reward,
                        done=done, clone_done=same_done,
                        info_reward=info['reward_components'], clone_info_reward=same_info['reward_components'])
                    save_json(target / 'clone_failure.json', failure)
                    raise AssertionError(f'Same-action clone differs at step {index}; inspect clone_failure.json')
            positions.append(env.raw.current_position.copy())
            if done and index != len(actions)-1:
                raise AssertionError('Saved-action replay ended at a different step')
        old_positions = data[:, [columns.index('x_mm'), columns.index('y_mm')]]
        if not done or not np.array_equal(np.asarray(positions), old_positions):
            raise AssertionError('Saved-action replay does not exactly reproduce original positions')
        replay_checks.append(dict(path_id=path_id, steps=len(actions), exact_position_replay=True,
            clone_obs_max_difference=max(clone_errors), environment_done_reason=info['done_reason']))
        write_csv(target / 'one_step.csv', one_step)
        write_csv(target / 'branches.csv', branch_rows)
        print(json.dumps(replay_checks[-1]), flush=True)
    grouped = [dict(variant=args.variant, path_id=path_id,
        pairs=[summarize_pairs([r for r in one_step if r['path_id'] == path_id], low, high)
               for low, high in ((2.81, 3.19), (2.994, 3.006))]) for path_id in paths]
    branch_summary = []
    for path_id in paths:
        for source_step in sorted({r['source_step'] for r in branch_rows if r['path_id'] == path_id}):
            rows = [r for r in branch_rows if r['path_id'] == path_id and r['source_step'] == source_step]
            by_mode = {mode: [r for r in rows if r['branch_mode'] == mode] for mode in protocol['branch_modes']}
            baseline = by_mode['constant_3']
            for mode, values in by_mode.items():
                if mode == 'constant_3':
                    continue
                n = min(len(values), len(baseline))
                delta = np.array([[r['x_mm']-b['x_mm'], r['y_mm']-b['y_mm']] for r, b in zip(values[:n], baseline[:n])])
                branch_summary.append(dict(path_id=path_id, source_step=source_step, mode=mode, matched_steps=n,
                    final_position_difference_mm=float(np.linalg.norm(delta[-1])),
                    max_position_difference_mm=float(np.linalg.norm(delta, axis=1).max()),
                    cumulative_reward_difference=float(sum(r['reward']-b['reward'] for r, b in zip(values[:n], baseline[:n]))),
                    final_covered_s_difference_mm=values[n-1]['covered_s_mm']-baseline[n-1]['covered_s_mm'],
                    early_end=len(values) != args.horizon))
    if hashes != {str(path): sha(path) for path in input_files} or source != study.source_state():
        raise AssertionError('Original evidence or source changed during diagnosis')
    save_json(target / 'summary.json', dict(by_path=grouped, branch_comparisons=branch_summary,
        replay_checks=replay_checks, one_step_count=len(one_step), branch_transition_count=len(branch_rows),
        input_files_unchanged=True, frozen_source_unchanged=True, wall_s=time.perf_counter()-started,
        no_optimizer_called=True, interpretation=protocol['interpretation']))
    save_json(target / 'hash_manifest.json', {path.name: sha(path) for path in target.iterdir() if path.is_file()})
    print(json.dumps(grouped), flush=True)


if __name__ == '__main__':
    main()
