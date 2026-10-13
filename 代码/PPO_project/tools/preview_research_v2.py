"""Second development round: time-integrated objective and bounded preview.

The first matrix is immutable development evidence. This module reuses its
training/checkpoint/trace implementation with explicit environment and config
adapters. New held-out paths are not evaluated before a separate freeze.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml
import revision_experiment as backend
from src.utils.revision_paths import load_manifest as read_manifest, load_path as read_path
from src.utils.ordered_coverage import OrderedCoverageTracker

ROOT = backend.ROOT
DATA = backend.PROJECT / 'data/preview_v2'
OUT = ROOT / '论文项目/revision_20260908/preview_v2'
BASE_CONFIG = backend.config_for
BASE_ENV = backend.make_env
BASE_AGENT = backend.make_agent
BASE_SOURCE = backend.source_state
DISTANCES = {'pretrain_short': 1.2, 'pretrain_mid': 2.3, 'pretrain_long': 4.0}
SEEDS = (11, 22, 33, 44, 55)


def manifest(): return read_manifest(DATA)
def path_points(path_id): return read_path(path_id, DATA)


def source_state():
    state = BASE_SOURCE()
    state['source_files'][str(Path(__file__).relative_to(ROOT)).replace('\\', '/')] = backend.sha(__file__)
    state['source_sha256'] = hashlib.sha256(json.dumps(state['source_files'], sort_keys=True).encode()).hexdigest()
    return state


def chosen_distance():
    return float(json.loads((OUT / 'selection.json').read_text(encoding='utf-8'))['fixed_distance_mm'])


def config_for(variant, seed):
    cfg = BASE_CONFIG('fixed_mid', seed)
    if variant in DISTANCES:
        distance = DISTANCES[variant]
    else:
        distance = chosen_distance()
    cfg['environment']['max_steps'] = 60000
    cfg['reward_weights']['p4'].update(time_penalty=-1.0, stall_progress_eps=1e-7,
                                      stall_steps=2000, stall_v_eps=.001)
    cfg['reward_weights']['lookahead_reward']['enabled'] = False
    cfg['reward_weights']['p6_1']['w_du'] = 0.0
    cfg['reward_weights']['lookahead']['distance'] = distance
    cfg['reward_weights']['lookahead']['finite_episode_preview'] = True
    cfg['reward_weights']['cornerness']['reference_distance'] = 4.6
    cfg['reward_weights']['lookahead_control'].update(
        enabled=variant in ('residual', 'absolute'), policy_action=variant in ('residual', 'absolute'),
        min_dist=.8, max_dist=4.6, straight_dist=distance, corner_dist=distance,
        mix_gain=1.0, action_default=(distance-.8)/3.8 if variant == 'absolute' else .5,
        residual_mode=variant == 'residual',
        residual_band_ratio=1.0/3.8, residual_corner_only=False, end_lock_progress=1.01)
    cfg['ppo'].update(actor_lr=5e-5, critic_lr=1e-4, gamma=math.exp(-.001/2),
                      lmbda=.99, epochs=4, ent_coef=.0003)
    cfg['revision'].update(variant=variant, objective='arc_length_progress_plus_time_integrated_costs_v2',
                         progress_reward_per_mm=1.0, completion_bonus=10.0, failure_penalty=-20.0,
                         fixed_distance_mm=distance, episode_max_steps=60000,
                         active_action_dims=[0,1,2] if variant in ('residual','absolute') else [0,1],
                         initialization='constant feasible-speed prior: angular mean 0, feed mean 0.05; shared network unchanged',
                         pretraining_steps=0, training_path_ids=[p['id'] for p in manifest()['paths'] if p['split']=='train'])
    if variant in ('fixed', 'residual', 'absolute'):
        prior = OUT / 'pretrain' / f'pretrain_selected_seed{seed}'
        cfg['revision'].update(initialization='paired fixed-preview PPO checkpoint',
                               init_checkpoint=str(prior / 'checkpoint.pt'),
                               init_checkpoint_sha256=backend.sha(prior / 'checkpoint.pt'),
                               pretraining_steps=65536)
    return cfg


class TimeIntegratedObjective:
    """Physical-time costs plus independently verified new ordered coverage.

    Only newly covered reference witnesses yield arc-length reward. The initial
    coverage is a baseline; revisits and skipped branches cannot earn it again.
    A claimed environment success must also pass ordered coverage and all
    sampled-position contour checks before receiving a completion bonus.
    """
    def __init__(self, env, cfg, reference):
        self.env = env
        self.cfg = cfg
        self.reference = np.asarray(reference, dtype=float).copy()
        self._reset_coverage()

    def _reset_coverage(self):
        self.coverage = OrderedCoverageTracker(self.reference, tolerance_mm=self.env.half_epsilon,
                                               initial_position=self.env.current_position)
        self._objective_done = False

    def __getattr__(self, key): return getattr(self.env, key)

    def reset(self, *args, **kwargs):
        observation = self.env.reset(*args, **kwargs)
        self._reset_coverage()
        return observation

    def step(self, action):
        if self._objective_done:
            raise RuntimeError('Reset the completed episode before stepping again')
        obs, raw_reward, done, info = self.env.step(action)
        old = dict(info['reward_components'])
        dt = self.env.interpolation_period
        coverage = self.coverage.update(self.env.current_position)
        progress = coverage['delta_s_mm'] * self.cfg['revision'].get('progress_reward_per_mm', 1.0)
        rates = raw_reward - old['r_progress'] - old.get('r_stall', 0.0)
        terminal = 0.0
        info['environment_done_reason'] = info['done_reason']
        if done:
            success = info['done_reason'] == 'success' and coverage['completion_valid']
            if info['done_reason'] == 'success' and not success:
                info['done_reason'] = 'ordered_incomplete'
            terminal = self.cfg['revision']['completion_bonus'] if success else self.cfg['revision']['failure_penalty']
            self._objective_done = True
        reward = progress + dt * rates + terminal
        scaled = {key: (float(value)*dt if key.startswith('r_') and key not in ('r_progress','r_stall') else value)
                  for key,value in old.items()}
        scaled.update(r_progress=progress, r_stall=0.0, r_terminal=terminal, total=reward,
                       original_per_tick_total=float(raw_reward), running_cost_rate=float(rates),
                       progress_mm=float(coverage['delta_s_mm']), dt_s=dt,
                       ordered_progress=float(coverage['ordered_progress']),
                       ordered_covered_s_mm=float(coverage['covered_s_mm']),
                       ordered_initial_s_mm=float(coverage['initial_covered_s_mm']),
                       ordered_coverage_complete=float(coverage['ordered_coverage_complete']),
                       trajectory_geometry_feasible=float(coverage['geometry_feasible']))
        info['reward_components'] = scaled
        info['ordered_coverage'] = coverage
        return obs, float(reward), done, info


def make_env(cfg, path_id, episode=0, reference=None):
    points = path_points(path_id) if reference is None else reference
    local = copy.deepcopy(cfg)
    local['environment']['max_steps'] = int(cfg['revision']['episode_max_steps'])
    length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    local['reward_weights']['w_s'] = length * cfg['revision']['progress_reward_per_mm']
    env, obs = BASE_ENV(local, path_id, episode, reference=points)
    return TimeIntegratedObjective(env, local, points), obs


def make_agent(cfg, env):
    agent = BASE_AGENT(cfg, env)
    prior = cfg['revision'].get('init_checkpoint')
    if prior:
        ckpt = torch.load(prior, map_location='cpu', weights_only=False)
        if backend.sha(prior) != cfg['revision']['init_checkpoint_sha256']:
            raise ValueError('Fixed-preview initialization checkpoint changed')
        agent.actor.load_state_dict(ckpt['actor']); agent.critic.load_state_dict(ckpt['critic'])
        # Fixed preview did not optimize its inactive third head. Center each
        # adaptive mapping at the identical physical prior distance.
        u0 = (cfg['revision']['fixed_distance_mm']-.8)/3.8 if cfg['revision']['variant']=='absolute' else .5
        with torch.no_grad():
            agent.actor.mu_head.weight[2].zero_()
            agent.actor.mu_head.bias[2] = math.log(u0/(1-u0))
    else:
        with torch.no_grad():
            agent.actor.mu_head.weight.zero_()
            agent.actor.mu_head.bias.copy_(torch.tensor([0., math.log(.05/.95), 0.]))
            agent.actor.std_head.weight.zero_()
            sigmas = (.02, .02, .05)
            agent.actor.std_head.bias.copy_(torch.tensor([math.log(math.expm1(s-.001)) for s in sigmas]))
    return agent


def activate_backend():
    """Install explicit adapters only for this CLI process, never at import time."""
    backend.OUT = OUT
    backend.DEFAULT_DATA_DIR = DATA
    backend.load_manifest = manifest
    backend.load_path = path_points
    backend.config_for = config_for
    backend.make_env = make_env
    backend.make_agent = make_agent
    backend.source_state = source_state


def prepare():
    protocol = dict(version='preview-v2', status='development; no advantage assumed',
        old_test_exposure='All v1 test paths are now development data; never reuse for unseen claims',
        data_manifest_sha256=backend.sha(DATA / 'manifest.json'),
        fixed_screen_distances_mm=[1.2,2.3,4.0], screen_seed=101, screen_steps=32768,
        independent_training_seeds=list(SEEDS), pretrain_steps=65536, paired_finetune_steps=32768,
        total_learning_steps_per_final_method_seed=98304,
        candidate_change='bounded residual preview +/-1.0 mm about selected fixed distance; original absolute-preview and adaptation-off controls',
        adaptive_selection='five-seed validation: strict success count, geometric completion count, mean ordered coverage, successful mean time; preserve both candidates in reports',
        shared_changes=['time-integrated cost rates plus arc-length progress', 'completion/failure objective',
                        'single-lap endpoint preview; ordered coverage gates progress and completion reward',
                        '2 s exponential discount horizon', 'low-variance feasible-speed initialization', 'stall threshold tied to measurable forward motion'],
        main_metrics='completion, tolerance, feed/heading constraints; times on common successful seed/path pairs',
        source_state=source_state(), test_gate='no test use until all models/configurations frozen; otherwise retain as development')
    target = OUT / 'protocol.json'
    if target.exists() and json.loads(target.read_text(encoding='utf-8')) != protocol:
        raise ValueError('Preserve the existing protocol; create a new version for changes')
    backend.write_json(target, protocol)
    print(target, flush=True)


def select():
    ranked = []
    for variant, distance in DISTANCES.items():
        run = OUT / 'screen' / f'{variant}_seed101'
        rows = json.loads((run / 'evaluation_validation_nominal/summary.json').read_text(encoding='utf-8'))['rows']
        successes = [r for r in rows if r['success']]
        mean_time = float(np.mean([r['elapsed_time_s'] for r in successes])) if successes else None
        score = [len(successes), sum(r['path_completed'] and r['geometry_feasible'] for r in rows),
                 float(np.mean([r['ordered_progress'] for r in rows])), -mean_time if mean_time is not None else -1e9]
        ranked.append(dict(variant=variant, distance_mm=distance, score=score,
                           successes=len(successes), successful_mean_time_s=mean_time))
    best = max(ranked, key=lambda r:r['score'])
    target = OUT / 'selection.json'
    if target.exists(): raise FileExistsError('Selection already saved; preserve it')
    backend.write_json(target, dict(fixed_distance_mm=best['distance_mm'], ranked=ranked,
                        selected_on='validation only', viable_on_all_validation=best['successes']==3,
                        timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))


def freeze():
    target = OUT / 'freeze.json'
    if target.exists(): raise FileExistsError(target)
    models, configs = {}, {}
    validation = {name: [] for name in ('fixed', 'residual', 'absolute')}
    for variant in ('fixed','residual','absolute'):
        for seed in SEEDS:
            run = OUT / 'finetune' / f'{variant}_seed{seed}'
            result = json.loads((run / 'training_summary.json').read_text(encoding='utf-8'))
            if result['steps'] != 32768: raise ValueError('Incomplete paired finetuning')
            key = str(run.relative_to(OUT)).replace('\\','/')
            models[key]=backend.sha(run/'checkpoint.pt'); configs[key]=backend.sha(run/'config.yaml')
            validation[variant].extend(json.loads((run/'evaluation_validation_nominal/summary.json').read_text(encoding='utf-8'))['rows'])
    ranking = []
    for variant, rows in validation.items():
        successful_times = [row['elapsed_time_s'] for row in rows if row['success']]
        ranking.append(dict(variant=variant, score=[sum(row['success'] for row in rows),
            sum(row['path_completed'] and row['geometry_feasible'] for row in rows),
            float(np.mean([row['ordered_progress'] for row in rows])),
            -float(np.mean(successful_times)) if successful_times else -1e9]))
    candidate = max((r for r in ranking if r['variant'] != 'fixed'), key=lambda r:r['score'])['variant']
    backend.write_json(target, dict(models=models, configs=configs, source_sha256=source_state()['source_sha256'],
                       path_manifest_sha256=backend.sha(DATA/'manifest.json'),
                       protocol_sha256=backend.sha(OUT/'protocol.json'), selection=json.loads((OUT/'selection.json').read_text(encoding='utf-8')),
                       primary_adaptive_candidate=candidate, validation_ranking=ranking,
                       timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()), test_results_used_for_selection=False))


def main():
    activate_backend()
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest='cmd',required=True)
    sub.add_parser('prepare'); sub.add_parser('select'); sub.add_parser('freeze')
    p=sub.add_parser('train'); p.add_argument('--variant',choices=[*DISTANCES,'pretrain_selected','fixed','residual','absolute'],required=True)
    p.add_argument('--seed',type=int,required=True); p.add_argument('--stage',default='screen'); p.add_argument('--steps',type=int,default=32768)
    p=sub.add_parser('evaluate'); p.add_argument('--run',type=Path,required=True)
    p.add_argument('--split',choices=['validation','test','regression'],required=True)
    p.add_argument('--regime',choices=['nominal','tight','low_jerk'],default='nominal')
    args=parser.parse_args()
    if args.cmd=='prepare': prepare()
    elif args.cmd=='select': select()
    elif args.cmd=='freeze': freeze()
    elif args.cmd=='train': backend.train(args.variant,args.seed,args.stage,args.steps)
    else: backend.evaluate(args.run,args.split,args.regime)


if __name__=='__main__': main()
