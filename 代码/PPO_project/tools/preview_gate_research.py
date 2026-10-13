"""Isolated PPO study of a geometry prior with a learned preview residual.

Reuses the frozen V2 environment, physical-time objective and audited runner.
Importing this module does not activate adapters or read held-out geometry.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import yaml

import preview_research_v2 as v2

backend = v2.backend
ROOT, DATA = v2.ROOT, v2.DATA
OUT = ROOT / '论文项目/revision_20260908/preview_gate'
BASE_SAMPLE_ROW = backend.sample_row
DISTANCES = dict(fixed_short=1.2, fixed_mid=2.3, fixed_center=3.0, fixed_long=4.0)
PILOT_VARIANTS = (*DISTANCES, 'gate_only', 'gate_residual')
FORMAL_VARIANTS = ('selected_fixed', 'gate_only', 'gate_residual')
SEEDS = (11, 22, 33, 44, 55)
PILOT_STEPS, FORMAL_STEPS = 32768, 131072


def manifest():
    return v2.read_manifest(DATA)


def path_points(path_id):
    return v2.read_path(path_id, DATA)


def source_state():
    state = v2.source_state()
    state['source_files'][str(Path(__file__).relative_to(ROOT)).replace('\\', '/')] = backend.sha(__file__)
    state['source_sha256'] = hashlib.sha256(json.dumps(state['source_files'], sort_keys=True).encode()).hexdigest()
    return state


class GateGeometry:
    """Fixed-window signed turns and their ordered, unwrapped angular range.

    This is algebraically identical to the registered 135-degree probe. Inserting
    collinear vertices or changing absolute orientation does not change it.
    """
    def __init__(self, reference, window_mm=4.6, threshold_deg=135.):
        points = np.asarray(reference, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
            raise ValueError('Need finite Nx2 reference with at least two points')
        vectors = np.diff(points, axis=0)
        lengths = np.linalg.norm(vectors, axis=1)
        if np.any(lengths <= 1e-10):
            raise ValueError('Reference contains adjacent repeated points')
        self.turn_arcs = np.cumsum(lengths)[:-1]
        headings = np.arctan2(vectors[:, 1], vectors[:, 0])
        turns = np.arctan2(np.sin(np.diff(headings)), np.cos(np.diff(headings)))
        self.signed = np.r_[0., np.cumsum(turns)]
        self.absolute = np.r_[0., np.cumsum(np.abs(turns))]
        self.window_mm = float(window_mm)
        self.threshold_rad = math.radians(threshold_deg)

    def at(self, s):
        lo, hi = np.searchsorted(self.turn_arcs, [s, s + self.window_mm], side='right')
        signed = float(self.signed[hi] - self.signed[lo])
        total = float(self.absolute[hi] - self.absolute[lo])
        cancellation = max(0., 1. - abs(signed) / (total + 1e-12)) if total > 1e-8 else 0.
        q = cancellation * min(1., total / math.pi)
        sequence = self.signed[lo:hi + 1]
        angular_range = float(np.max(sequence) - np.min(sequence))
        gate = float(angular_range <= self.threshold_rad + 1e-12)
        return dict(q=float(q), tangent_range_rad=angular_range, range_over_pi=angular_range/math.pi,
                    signed_turn_rad=signed, total_turn_rad=total, gate=gate,
                    prior_mm=2.3 + 1.7*q*gate)


def config_for(variant, seed):
    if variant not in (*PILOT_VARIANTS, 'selected_fixed'):
        raise ValueError(f'Unknown gate-study variant: {variant}')
    cfg = v2.config_for('pretrain_mid', seed)
    distance = DISTANCES.get(variant, 2.3)
    if variant == 'selected_fixed':
        selection = json.loads((OUT / 'selection.json').read_text(encoding='utf-8'))
        distance = float(selection['fixed_distance_mm'])
    # Raw CNCEnv has two active physical actions. The outer adapter alone maps
    # the optional third policy action to a distance; there is no second mapping.
    cfg['reward_weights']['lookahead']['distance'] = distance
    cfg['reward_weights']['lookahead_control'].update(enabled=False, policy_action=False,
        residual_mode=False, straight_dist=distance, corner_dist=distance, action_default=.5)
    cfg['ppo'].update(actor_lr=1e-5, lmbda=.999)
    cfg['revision'].update(variant=variant, study='geometry-gate-preview-v1',
        active_action_dims=[0, 1, 2] if variant == 'gate_residual' else [0, 1],
        fixed_distance_mm=distance, observation_dim=20, base_observation_dim=18,
        extra_observation_features=['q in [0,1]', 'unwrapped tangent range / pi; nonnegative, no clipping'],
        extra_observation_timing='returned obs at s_(t+1); base preview observations use executed D_t',
        initialization='fresh paired-seed network; actor means [0,0.05,0.5], std [0.01,0.01,0.05]',
        pretraining_steps=0, geometry_window_mm=4.6, tangent_range_gate_deg=135.,
        geometry_prior_base_mm=2.3, geometry_prior_amplitude_mm=1.7,
        residual_amplitude_mm=.3, preview_bounds_mm=[.8, 4.6],
        preview_timing='D0 from s_t before action; D_t=D0+0.3*(2*u-1) only for gate_residual',
        training_path_ids=[p['id'] for p in manifest()['paths'] if p['split'] == 'train'])
    if variant == 'selected_fixed':
        cfg['revision']['fixed_selection_sha256'] = backend.sha(OUT / 'selection.json')
    return cfg


class GatePreviewEnvironment:
    """Add shared state features and map preview once, outside frozen CNCEnv."""
    def __init__(self, env, cfg, reference):
        self.env, self.cfg = env, cfg
        self.raw = env.env
        self.variant = cfg['revision']['variant']
        self.geometry = GateGeometry(reference, cfg['revision']['geometry_window_mm'],
                                     cfg['revision']['tangent_range_gate_deg'])
        if env.observation_dim != 18:
            raise ValueError('Gate study requires the frozen 18-dimensional base observation')
        self.observation_dim = 20
        self.action_space_dim = len(cfg['revision']['active_action_dims'])
        low, high = np.array([-1., 0., 0.]), np.ones(3)
        self.action_space = type(env.action_space)(low=low[:self.action_space_dim].astype(np.float32),
            high=high[:self.action_space_dim].astype(np.float32), dtype=np.float32)
        self.observation_space = type(env.observation_space)(
            low=np.r_[env.observation_space.low, [0., 0.]].astype(np.float32),
            high=np.r_[env.observation_space.high, [1., np.inf]].astype(np.float32), dtype=np.float32)
        self.gate_trace_snapshot = {}

    def __getattr__(self, key):
        return getattr(self.env, key)

    @property
    def state(self):
        return self._observation.copy()

    def _features(self):
        s = float(self.raw._project_onto_progress_path(self.raw.current_position)[2])
        return s, self.geometry.at(s)

    def _augment(self, observation):
        began = time.perf_counter_ns()
        s, feature = self._features()
        self._observation = np.r_[np.asarray(observation, dtype=np.float32),
                                 [feature['q'], feature['range_over_pi']]].astype(np.float32)
        self.gate_trace_snapshot.update(observation_arc_mm=s, observation_q=feature['q'],
                                        observation_range_over_pi=feature['range_over_pi'],
                                        observation_features_us=(time.perf_counter_ns()-began)/1000.)
        return self._observation.copy()

    def initialize_observation(self):
        _, feature = self._features()
        initial = feature['prior_mm'] if self.variant.startswith('gate_') else self.cfg['revision']['fixed_distance_mm']
        self.raw.lookahead_dist = float(initial)
        self.raw._resolve_active_lookahead_dist()
        self.raw.state = self.raw._build_state()
        self.gate_trace_snapshot = dict(prior_mm=feature['prior_mm'], selected_mm=float(initial),
                                       residual_mm=0., decision_us=0.)
        return self._augment(self.raw.state)

    def reset(self, *args, **kwargs):
        self.env.reset(*args, **kwargs)
        return self.initialize_observation()

    def step(self, action):
        values = np.asarray(action, dtype=float)
        if values.shape != (self.action_space_dim,) or not np.isfinite(values).all():
            raise ValueError('Action has wrong dimension or nonfinite values')
        values = np.clip(values, self.action_space.low, self.action_space.high)
        began = time.perf_counter_ns()
        s, feature = self._features()
        base = feature['prior_mm'] if self.variant.startswith('gate_') else self.cfg['revision']['fixed_distance_mm']
        residual = self.cfg['revision']['residual_amplitude_mm']*(2.*values[2]-1.) if self.variant == 'gate_residual' else 0.
        selected = float(np.clip(base+residual, *self.cfg['revision']['preview_bounds_mm']))
        self.raw.lookahead_dist = selected
        decision_us = (time.perf_counter_ns()-began)/1000.
        # Raw controller sees exactly two physical actions. Its control and
        # next-state preview query both retain the same selected D_t.
        observation, reward, done, info = self.env.step(values[:2])
        self.gate_trace_snapshot = dict(action_arc_mm=s, action_q=feature['q'],
            action_range_over_pi=feature['range_over_pi'], gate=feature['gate'],
            total_turn_rad=feature['total_turn_rad'], signed_turn_rad=feature['signed_turn_rad'],
            prior_mm=float(base), selected_mm=selected, residual_mm=float(residual), decision_us=decision_us)
        observation = self._augment(observation)
        info.setdefault('control_status', {}).update({'gate_'+k: float(v) for k, v in self.gate_trace_snapshot.items()})
        return observation, reward, done, info


def make_env(cfg, path_id, episode=0, reference=None):
    points = path_points(path_id) if reference is None else reference
    env, _ = v2.make_env(cfg, path_id, episode, reference=points)
    wrapped = GatePreviewEnvironment(env, cfg, points)
    return wrapped, wrapped.initialize_observation()


def make_agent(cfg, env):
    agent = v2.BASE_AGENT(cfg, env)
    with torch.no_grad():
        agent.actor.mu_head.weight.zero_()
        agent.actor.mu_head.bias.copy_(torch.tensor([0., math.log(.05/.95), 0.]))
        agent.actor.std_head.weight.zero_()
        agent.actor.std_head.bias.copy_(torch.tensor([math.log(math.expm1(s-.001)) for s in (.01, .01, .05)]))
    return agent


def sample_row(env, info, reward, latent):
    row = BASE_SAMPLE_ROW(env, info, reward, latent)
    row.update({'control_status__gate_'+k: float(value) for k, value in env.gate_trace_snapshot.items()})
    return row


def activate_backend():
    """CLI-only activation; imports leave all V1/V2 runner globals unchanged."""
    backend.OUT, backend.DEFAULT_DATA_DIR = OUT, DATA
    backend.load_manifest, backend.load_path = manifest, path_points
    backend.config_for, backend.make_env, backend.make_agent = config_for, make_env, make_agent
    backend.source_state, backend.sample_row = source_state, sample_row


def protocol_definition():
    return dict(version='preview-gate-v1', source_sha256=source_state()['source_sha256'],
        manifest_sha256=backend.sha(DATA/'manifest.json'), pilot_variants=list(PILOT_VARIANTS),
        pilot_seed=101, pilot_steps=PILOT_STEPS, formal_variants=list(FORMAL_VARIANTS),
        formal_seeds=list(SEEDS), formal_steps=FORMAL_STEPS, initialization='fresh; no warm start or pretraining',
        total_pilot_steps=len(PILOT_VARIANTS)*PILOT_STEPS, total_formal_steps=len(FORMAL_VARIANTS)*len(SEEDS)*FORMAL_STEPS,
        geometry_window_mm=4.6, threshold_deg=135., threshold_selection='first registered threshold; 135/150 probes identical',
        residual_mm=.3, extra_observations=['q in [0,1], no additional clipping',
        'ordered unwrapped tangent range/pi, >=0 and not clipped'],
        shared_training_changes=dict(actor_lr=1e-5, lmbda=.999, initial_std=[.01,.01,.05],
            attribution='Shared by fixed and adaptive methods; not independent preview benefits'),
        unchanged='PPO MLP depth/width, V2 physical-time objective, KCM, endpoint and ordered coverage checks',
        sample_timing='Geometry prior from s_t; control and base nextobs use D_t; two appended features query s_(t+1)',
        primary_comparison='selected_fixed versus gate_residual; gate_only isolates learned preview residual',
        selection='All registered pilots reported; select one fixed distance by validation success, geometric completion, ordered coverage, successful mean time',
        overhead='Actor inference in result JSON; selector decision_us and returned-state feature timing in trace',
        test_policy='V2 holdout remains unused until this study freezes all 15 formal checkpoints; no automatic test execution',
        success_time_policy='Report all failures; compare times on common successful seed/path pairs')


def prepare():
    value = protocol_definition()
    target = OUT/'protocol.json'
    if target.exists() and json.loads(target.read_text(encoding='utf-8')) != value:
        raise ValueError('Protocol/source changed; preserve this study and choose a new version')
    backend.write_json(target, value)
    return target


def check_protocol():
    value = json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    if value != protocol_definition():
        raise ValueError('Protocol/source changed since preparation')
    return value


def train(variant, seed, stage, steps):
    check_protocol()
    if stage == 'pilot':
        valid = variant in PILOT_VARIANTS and seed == 101 and 0 < steps <= PILOT_STEPS
    elif stage == 'formal':
        valid = variant in FORMAL_VARIANTS and seed in SEEDS and 0 < steps <= FORMAL_STEPS
        if not (OUT/'selection.json').exists():
            raise ValueError('Select the fixed distance from all registered pilots before formal training')
    elif stage == 'smoke':
        valid = variant in PILOT_VARIANTS and 0 < steps <= 512
    else:
        valid = False
    if not valid:
        raise ValueError('Variant, seed, stage or budget violates the registered study')
    return backend.train(variant, seed, stage, steps)


def evaluate(run, split, regime='nominal'):
    check_protocol()
    run = Path(run).resolve()
    run.relative_to(OUT.resolve())
    if split == 'test':
        frozen = json.loads((OUT/'freeze.json').read_text(encoding='utf-8'))
        if frozen['protocol_sha256'] != backend.sha(OUT/'protocol.json') or frozen['selection_sha256'] != backend.sha(OUT/'selection.json'):
            raise ValueError('Frozen protocol or selection changed')
    return backend.evaluate(run, split, regime)


def _validated_run(stage, variant, seed, budget):
    """Validate saved models and validation rows before selection or test freeze."""
    run = OUT/stage/f'{variant}_seed{seed}'
    cfg = yaml.safe_load((run/'config.yaml').read_text(encoding='utf-8'))
    provenance = json.loads((run/'provenance.json').read_text(encoding='utf-8'))
    summary = json.loads((run/'training_summary.json').read_text(encoding='utf-8'))
    current = source_state()['source_sha256']
    checkpoint_hash, config_hash = backend.sha(run/'checkpoint.pt'), backend.sha(run/'config.yaml')
    if cfg != config_for(variant, seed) or provenance['config_sha256'] != config_hash:
        raise ValueError('Run configuration does not match the registered variant')
    if provenance['source_sha256'] != current or provenance['path_manifest_sha256'] != backend.sha(DATA/'manifest.json'):
        raise ValueError('Run source or manifest differs from this study')
    checkpoint = torch.load(run/'checkpoint.pt', map_location='cpu', weights_only=False)
    if summary['steps'] != budget or checkpoint['steps'] != budget or summary['checkpoint_sha256'] != checkpoint_hash:
        raise ValueError('Run has not completed the exact registered budget/checkpoint')
    evaluation = run/'evaluation_validation_nominal'
    rows = json.loads((evaluation/'summary.json').read_text(encoding='utf-8'))['rows']
    expected_ids = {p['id'] for p in manifest()['paths'] if p['split'] == 'validation'}
    if len(rows) != len(expected_ids) or {r['path_id'] for r in rows} != expected_ids:
        raise ValueError('Validation rows are missing or duplicated')
    for row in rows:
        if (row['split'] != 'validation' or row['regime'] != 'nominal'
            or row['checkpoint_sha256'] != checkpoint_hash or row['config_sha256'] != config_hash
            or row['code_sha256'] != current or row['effective_config_sha256'] != backend.sha(evaluation/'effective_config.yaml')
            or row != json.loads((evaluation/(row['path_id']+'.json')).read_text(encoding='utf-8'))):
            raise ValueError('Validation rows are stale or inconsistent with saved results')
    return run, rows


def _rank(variant, rows):
    times = [r['elapsed_time_s'] for r in rows if r['success']]
    return dict(variant=variant, score=[sum(r['success'] for r in rows),
        sum(r['path_completed'] and r['geometry_feasible'] for r in rows),
        float(np.mean([r['ordered_progress'] for r in rows])), -float(np.mean(times)) if times else -1e9])


def select():
    check_protocol()
    target = OUT/'selection.json'
    if target.exists():
        raise FileExistsError('Preserve the existing fixed-distance selection')
    ranking = []
    for variant in PILOT_VARIANTS:
        _, rows = _validated_run('pilot', variant, 101, PILOT_STEPS)
        ranking.append(_rank(variant, rows))
    chosen = max((r for r in ranking if r['variant'] in DISTANCES), key=lambda r: r['score'])['variant']
    backend.write_json(target, dict(selected_fixed=chosen, fixed_distance_mm=DISTANCES[chosen], rankings=ranking,
        source_sha256=source_state()['source_sha256'], protocol_sha256=backend.sha(OUT/'protocol.json'),
        test_results_used_for_selection=False, timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    return target


def freeze():
    check_protocol()
    target = OUT/'freeze.json'
    if target.exists():
        raise FileExistsError('Preserve the existing test freeze')
    selection = json.loads((OUT/'selection.json').read_text(encoding='utf-8'))
    if selection['source_sha256'] != source_state()['source_sha256'] or selection['protocol_sha256'] != backend.sha(OUT/'protocol.json'):
        raise ValueError('Selection belongs to another source/protocol')
    models, configs, rankings = {}, {}, []
    for variant in FORMAL_VARIANTS:
        combined = []
        for seed in SEEDS:
            run, rows = _validated_run('formal', variant, seed, FORMAL_STEPS)
            key = str(run.relative_to(OUT)).replace('\\', '/')
            models[key], configs[key] = backend.sha(run/'checkpoint.pt'), backend.sha(run/'config.yaml')
            combined.extend(rows)
        rankings.append(_rank(variant, combined))
    backend.write_json(target, dict(models=models, configs=configs, validation_ranking=rankings,
        source_sha256=source_state()['source_sha256'], path_manifest_sha256=backend.sha(DATA/'manifest.json'),
        protocol_sha256=backend.sha(OUT/'protocol.json'), selection_sha256=backend.sha(OUT/'selection.json'),
        selection=selection, primary_adaptive_candidate='gate_residual', rule_only_control='gate_only',
        test_results_used_for_selection=False, timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'select', 'freeze'):
        commands.add_parser(name)
    training = commands.add_parser('train')
    training.add_argument('--variant', choices=[*PILOT_VARIANTS, 'selected_fixed'], required=True)
    training.add_argument('--seed', type=int, required=True)
    training.add_argument('--stage', choices=['smoke', 'pilot', 'formal'], default='pilot')
    training.add_argument('--steps', type=int, default=PILOT_STEPS)
    evaluation = commands.add_parser('evaluate')
    evaluation.add_argument('--run', type=Path, required=True)
    evaluation.add_argument('--split', choices=['validation', 'test', 'regression'], required=True)
    evaluation.add_argument('--regime', choices=['nominal', 'tight', 'low_jerk'], default='nominal')
    args = parser.parse_args()
    activate_backend()
    if args.command == 'train':
        print(train(args.variant, args.seed, args.stage, args.steps), flush=True)
    elif args.command == 'evaluate':
        evaluate(args.run, args.split, args.regime)
    else:
        print(globals()[args.command](), flush=True)


if __name__ == '__main__':
    main()
