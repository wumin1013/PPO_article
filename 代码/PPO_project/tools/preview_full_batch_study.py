"""One scheduling ablation: collect 64 macro decisions across episode ends.

The local-corner environment, positive-feed PPO, reward, optimizer settings,
critic target and initial weights are reused unchanged. A terminal transition
closes GAE, but does not trigger a short update. No holdout/test interface.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time
import zipfile

import numpy as np
import torch
import yaml

import preview_local_corner_study as local

core = local.core
OUT = core.old.ROOT / '论文项目/revision_20260910_continued/full_batch_v1'
METHODS = local.METHODS
MILESTONES = (0, 256, 512, 1024, 2048, 4096)
BATCH_SIZE = 64
BUFFER_KEYS = ('states', 'actions', 'next_states', 'rewards', 'dones',
               'terminated', 'old_log_probs', 'action_masks')


def configuration(candidate, method, seed):
    cfg = local.configuration(candidate, method, seed)
    cfg['revision'].update(study='full-batch-across-episodes-v1',
        candidate_change='Only update scheduling: terminal transitions do not flush a short batch',
        unchanged='Local-corner observations, positive-feed sampling, PPO losses/targets/LR/epochs, reward, KCM and terminal audit',
        validation_milestones=list(MILESTONES),
        zero_step_role='Initial diagnostic only; excluded from trained checkpoint selection',
        training_batch_macro_decisions=BATCH_SIZE,
        terminal_batch_rule='Keep done/terminated flags; reset environment, collect with same actor until 64 decisions',
        checkpoint_buffer_rule='Persist incomplete buffer without optimizing; resume with identical actor, optimizer, RNG and environment replay')
    return cfg


def source_state():
    value = local.source_state()
    path = Path(__file__)
    value['source_files'][path.relative_to(core.old.ROOT).as_posix()] = core.old.sha(path)
    value['source_sha256'] = hashlib.sha256(json.dumps(value['source_files'], sort_keys=True).encode()).hexdigest()
    return value


def prepare():
    # Read/verify the existing local protocol; do not call its install() or
    # rewrite any previously frozen source, configuration or output.
    core.old.check_protocol()
    prior_protocol_path = local.OUT / 'protocol.json'
    prior_protocol = core.old.read(prior_protocol_path)
    if prior_protocol['source'] != local.source_state():
        raise ValueError('The frozen local-corner source changed')
    for method in ('fixed3', 'adaptive'):
        run = local.positive.PRIOR / f'preview_time_{method}_seed101'
        expected = prior_protocol['priors'][method]
        if core.old.sha(run/'checkpoint.pt') != expected['checkpoint_sha256']:
            raise ValueError('The retained sensor initializer changed')
    value = dict(version='full-batch-across-episodes-v1', source=source_state(),
        prior_protocol_sha256=core.old.sha(prior_protocol_path),
        manifest_sha256=core.old.sha(core.old.DATA/'manifest.json'),
        initializers=prior_protocol['priors'], methods=list(METHODS), seed=101,
        pretraining_macro_decisions=4096, validation_milestones=list(MILESTONES),
        selection_milestones=list(MILESTONES[1:]), max_additional_decisions=4096,
        batch_macro_decisions=BATCH_SIZE, fixed_grid_mm=[1.2, 3., 4.],
        main_change='Only stop forcing a short optimizer update at episode termination',
        equal_budget='Same additional decisions and 64-decision batches across fixed/adaptive; log exact optimizer and physical costs',
        selection='Evaluate all three validation paths at each milestone; prioritize complete success count, then time only if all paths succeed; exclude step0',
        fixed_selection='Select one fixed distance from pooled validation, never per path',
        test_access=False, machine_execution_authorized=False)
    target = OUT/'protocol.json'
    if target.exists() and core.old.read(target) != value:
        raise ValueError('Preserve registered full-batch source/protocol')
    if not target.exists():
        core.old.write(target, value)
    return value


def _jsonl(records):
    return ''.join(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n' for record in records).encode('utf-8')


def _atomic_bytes(path, content):
    temp = path.with_name(path.name+'.tmp')
    temp.write_bytes(content)
    temp.replace(path)


class FullBatchTrainer:
    """Checkpoint is authoritative; JSONL files are recoverable projections.

    Saving after each terminal accounting event or full update bounds replay
    work. The checkpoint contains both completed log records and the unfinished
    rollout. A crash between checkpoint/log writes is repaired on resume. Any
    divergent extra log or orphan trace is copied to recovery/ before replacing
    a derived file; no historical bytes are silently discarded.
    """

    def __init__(self, cfg, run, env_factory=local.LocalCornerEnvironment,
                 agent_factory=local.make_agent, source_hash=None):
        self.cfg, self.run = cfg, Path(run)
        self.run.mkdir(parents=True, exist_ok=True)
        self.env_factory, self.source_hash = env_factory, source_hash
        self.ids = cfg['revision']['training_path_ids']
        random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
        self.env = env_factory(cfg, self.ids[0])
        self.agent = agent_factory(cfg, self.env)
        self.decisions = self.episode = self.physics_before = self.commands_before = 0
        self.actions, self.updates, self.episodes, self.trace_files = [], [], [], []
        self.buffer = {key: [] for key in BUFFER_KEYS}
        self.ep_return = self.wall_before = self.replay_wall_s = 0.
        self.started = time.perf_counter()
        if (self.run/'checkpoint.pt').exists():
            self.restore()

    def _preserve(self, path):
        content = path.read_bytes()
        archive = self.run/'recovery'/f'{core.old.sha(path)}_{path.name}'
        archive.parent.mkdir(exist_ok=True)
        if archive.exists() and archive.read_bytes() != content:
            raise ValueError('Recovery artifact collision')
        if not archive.exists():
            archive.write_bytes(content)

    def sync_logs(self):
        for name, records in (('updates.jsonl', self.updates), ('episodes.jsonl', self.episodes)):
            path, content = self.run/name, _jsonl(records)
            if path.exists():
                previous = path.read_bytes()
                if previous == content:
                    continue
                # Normal append and recovery from an interrupted prefix write
                # are projections of committed checkpoint data.
                if not content.startswith(previous):
                    self._preserve(path)
            _atomic_bytes(path, content)

    def restore(self):
        started = time.perf_counter()
        state = torch.load(self.run/'checkpoint.pt', map_location='cpu', weights_only=False)
        if state.get('full_batch_version') != 1 or state['config'] != self.cfg:
            raise ValueError('Full-batch checkpoint/config mismatch')
        if state['source_sha256'] != self.source_hash:
            raise ValueError('Full-batch checkpoint source mismatch')
        for name in ('actor', 'critic', 'actor_optimizer', 'critic_optimizer'):
            getattr(self.agent, name).load_state_dict(state[name])
        for attr in ('decisions', 'episode', 'buffer', 'actions', 'updates', 'episodes',
                     'trace_files', 'ep_return', 'physics_before', 'commands_before'):
            setattr(self, attr, state[attr])
        self.wall_before, self.replay_wall_s = state['training_wall_s'], state['replay_wall_s']
        self.agent.logged_steps = state['optimized_decisions']
        sizes = {len(self.buffer[key]) for key in BUFFER_KEYS}
        if len(sizes) != 1 or not 0 <= next(iter(sizes)) < BATCH_SIZE:
            raise ValueError('Invalid saved unfinished buffer')
        if self.agent.logged_steps + len(self.buffer['states']) != self.decisions:
            raise ValueError('Saved decision/buffer accounting mismatch')
        for entry in self.trace_files:
            if core.old.sha(self.run/entry['file']) != entry['sha256']:
                raise ValueError('Completed episode trace changed')
        self.env = self.env_factory(self.cfg, self.ids[self.episode % len(self.ids)])
        for action in self.actions:
            self.env.step(np.asarray(action, float))
        np.testing.assert_array_equal(self.env.state, state['observation'])
        if self.env.done or self.env.physics_steps != state['current_physics_steps'] or self.env.command_steps != state['current_command_steps']:
            raise ValueError('Current environment replay mismatch')
        torch.set_rng_state(state['torch_rng']); np.random.set_state(state['numpy_rng']); random.setstate(state['python_rng'])
        self.replay_wall_s += time.perf_counter()-started
        self.started = time.perf_counter()
        self.sync_logs()

    def save(self):
        elapsed = self.wall_before+time.perf_counter()-self.started
        state = dict(full_batch_version=1, config=self.cfg, source_sha256=self.source_hash,
            **{name: getattr(self.agent, name).state_dict() for name in
               ('actor', 'critic', 'actor_optimizer', 'critic_optimizer')},
            **{name: getattr(self, name) for name in ('decisions', 'episode', 'buffer', 'actions',
               'updates', 'episodes', 'trace_files', 'ep_return', 'physics_before', 'commands_before')},
            optimized_decisions=self.agent.logged_steps, training_wall_s=elapsed, replay_wall_s=self.replay_wall_s,
            observation=self.env.state.copy(), current_physics_steps=self.env.physics_steps,
            current_command_steps=self.env.command_steps,
            # Keep original core evaluator's expected accounting fields.
            completed_episode_physics_steps=self.physics_before,
            completed_episode_command_steps=self.commands_before, episode_actions=self.actions,
            torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(), python_rng=random.getstate())
        temp = self.run/'checkpoint.tmp'
        torch.save(state, temp)
        temp.replace(self.run/'checkpoint.pt')
        self.sync_logs()
        return state

    def _finish_episode(self):
        path = self.run/f'train_episode_{self.episode:04d}.npz'
        if path.exists():
            # Only possible for an uncommitted event after interruption.
            self._preserve(path)
        temp = self.run/f'train_episode_{self.episode:04d}.tmp.npz'
        core.old.save_trace(temp, self.env.trace)
        temp.replace(path)
        self.trace_files.append(dict(file=path.name, sha256=core.old.sha(path)))
        self.episodes.append(dict(episode=self.episode, path_id=self.ids[self.episode % len(self.ids)],
            decisions=self.decisions, physical_steps=self.env.physics_steps, command_steps=self.env.command_steps,
            reward=self.ep_return, result=copy.deepcopy(self.env.result)))
        self.physics_before += self.env.physics_steps; self.commands_before += self.env.command_steps
        self.episode += 1; self.actions = []; self.ep_return = 0.
        self.env = self.env_factory(self.cfg, self.ids[self.episode % len(self.ids)])

    def advance(self, target_decisions):
        if target_decisions < self.decisions:
            raise ValueError('Requested milestone is older than current run; use saved snapshot')
        while self.decisions < target_decisions:
            observation = self.env.state.copy()
            action, choice = self.agent.take_action(observation, return_info=True)
            nxt, reward, done, info = self.env.step(action)
            self.actions.append(action.tolist()); self.ep_return += reward
            values = (observation, action.copy(), nxt.copy(), reward, done, done,
                      choice['old_log_prob'], choice['action_mask'].copy())
            for key, value in zip(BUFFER_KEYS, values):
                self.buffer[key].append(value)
            self.decisions += 1
            updated = len(self.buffer['states']) == BATCH_SIZE
            if updated:
                self.agent.update(self.buffer)
                self.updates.append(dict(decisions=self.decisions, **self.agent.last_update_diagnostics,
                    batching='64 macro decisions across episode boundaries; terminal flags retained'))
                self.buffer = {key: [] for key in BUFFER_KEYS}
            if done:
                self._finish_episode()
            if updated or done:
                self.save()
            if self.decisions % 256 == 0:
                print(f'{self.cfg["revision"]["variant"]}: decisions={self.decisions}/{target_decisions}, '
                      f'episodes={self.episode}, updates={len(self.updates)}', flush=True)
        return self.save()

    def summary(self, state):
        return dict(candidate='preview_time', method=self.cfg['revision']['variant'], seed=self.cfg['seed'],
            decisions=self.decisions, optimized_decisions=self.agent.logged_steps,
            pending_buffer_decisions=len(self.buffer['states']), episodes=self.episode,
            physics_steps=self.physics_before+self.env.physics_steps,
            command_steps=self.commands_before+self.env.command_steps,
            training_wall_s=state['training_wall_s'], environment_replay_wall_s=self.replay_wall_s,
            optimizer_updates=len(self.updates), adam_steps_per_network=len(self.updates)*self.cfg['ppo']['epochs'],
            checkpoint_sha256=core.old.sha(self.run/'checkpoint.pt'), pretraining_macro_decisions=4096,
            pretraining_included_in_additional_budget=False, convergence_established=False,
            zero_step_initial_diagnostic=self.decisions == 0,
            status='registered_development_milestone_complete')


def _ensure_provenance(run, cfg, protocol):
    config_path = run/'config.yaml'
    if config_path.exists():
        if yaml.safe_load(config_path.read_text(encoding='utf-8')) != cfg:
            raise ValueError('Preserve saved config')
        if core.old.read(run/'provenance.json')['source_sha256'] != protocol['source']['source_sha256']:
            raise ValueError('Preserve saved source')
        return
    config_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding='utf-8')
    core.old.write(run/'provenance.json', dict(**protocol['source'], config_sha256=core.old.sha(config_path),
        protocol_sha256=core.old.sha(OUT/'protocol.json'), command='python '+' '.join(sys.argv)))
    with zipfile.ZipFile(run/'source.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for name in protocol['source']['source_files']:
            archive.write(core.old.ROOT/name, name)


def train(method, seed, decisions, stage='development'):
    if stage == 'development' and (seed != 101 or decisions not in MILESTONES):
        raise ValueError('Use seed101 and a registered development milestone')
    if stage == 'smoke' and (seed != 901 or not 0 <= decisions <= 128):
        raise ValueError('Use smoke seed901 and 0..128 decisions')
    torch.set_num_threads(1)
    protocol = prepare(); cfg = configuration('preview_time', method, seed)
    run = OUT/stage/f'preview_time_{method}_seed{seed}'
    run.mkdir(parents=True, exist_ok=True)
    _ensure_provenance(run, cfg, protocol)
    trainer = FullBatchTrainer(cfg, run, source_hash=protocol['source']['source_sha256'])
    snapshot = run/'checkpoints'/f'step{decisions}'
    if snapshot.exists():
        # A completed snapshot is immutable. Its own model hash establishes
        # the result even if the mutable continuation has advanced further.
        saved = core.old.read(snapshot/'training_summary.json')
        if saved['checkpoint_sha256'] != core.old.sha(snapshot/'checkpoint.pt'):
            raise ValueError('Saved snapshot checkpoint changed')
        manifest = core.old.read(snapshot/'snapshot_manifest.json')
        for name, digest in manifest['files'].items():
            if core.old.sha(snapshot/name) != digest:
                raise ValueError(f'Saved snapshot artifact changed: {name}')
        return snapshot
    state = trainer.advance(decisions)
    core.old.save_trace(run/'training_final_partial.npz', trainer.env.trace)
    core.old.write(run/'training_summary.json', trainer.summary(state))
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    pending = snapshot.parent/f'.step{decisions}.pending-{time.time_ns()}'
    pending.mkdir()
    names = ['checkpoint.pt', 'config.yaml', 'provenance.json', 'source.zip', 'training_summary.json',
             'updates.jsonl', 'episodes.jsonl', 'training_final_partial.npz']
    names += [item['file'] for item in trainer.trace_files]
    for name in names:
        shutil.copyfile(run/name, pending/name)
    core.old.write(pending/'snapshot_manifest.json', dict(decisions=decisions,
        files={name: core.old.sha(pending/name) for name in names},
        recovery='Self-contained checkpoint includes optimizers, RNG, current-episode replay actions, pending rollout and full log records'))
    if pending.resolve().parent != snapshot.resolve().parent:
        raise ValueError('Snapshot publication must stay inside the run checkpoint directory')
    pending.replace(snapshot)
    return snapshot


def install():
    # Reuse the unchanged deterministic validator and its provenance gates.
    core.OUT = OUT; core.source_state = source_state; core.configuration = configuration
    core.MacroEnvironment = local.LocalCornerEnvironment; core.make_agent = local.make_agent; core.prepare = prepare
    return core


def select(budget):
    if budget not in MILESTONES[1:]:
        raise ValueError('Selection requires a trained registered common budget')
    prepare(); records = []
    expected = {p['id'] for p in core.old.manifest()['paths'] if p['split'] == 'validation'}
    key = lambda row: (-row['successes'], row['complete_validation_time_s']
                       if row['complete_validation_time_s'] is not None else math.inf, row['step'])
    for method in METHODS:
        run = OUT/'development'/f'preview_time_{method}_seed101'
        if core.old.read(run/'training_summary.json')['decisions'] < budget:
            raise ValueError('Every method needs the same available training budget')
        candidates = []
        for step in MILESTONES[1:]:
            if step > budget: break
            snapshot = run/'checkpoints'/f'step{step}'
            report = core.old.read(snapshot/'validation/summary.json')['rows']
            model_sha = core.old.sha(snapshot/'checkpoint.pt')
            if len(report) != len(expected) or {row['path_id'] for row in report} != expected:
                raise ValueError('Every checkpoint requires every validation path')
            if any(row['checkpoint_sha256'] != model_sha for row in report):
                raise ValueError('Validation/checkpoint mismatch')
            successes = sum(row['success'] for row in report)
            candidates.append(dict(method=method, step=step, successes=successes, count=len(report),
                complete_validation_time_s=sum(row['elapsed_time_s'] for row in report) if successes == len(report) else None,
                checkpoint=snapshot.relative_to(core.old.ROOT).as_posix(), checkpoint_sha256=model_sha,
                validation_sha256=core.old.sha(snapshot/'validation/summary.json')))
        records.append(dict(selected=min(candidates, key=key), all_checkpoints=candidates))
    result = dict(budget=budget, records=records, zero_step_excluded=True,
        uniform_fixed=min([r['selected'] for r in records if r['selected']['method'] != 'adaptive'], key=key),
        rule='Full success count then time only for complete validation sets; earliest checkpoint breaks other ties',
        data_role='development validation only')
    path = OUT/f'selection_budget{budget}.json'
    if path.exists(): raise FileExistsError('Preserve completed selection')
    core.old.write(path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('prepare')
    train_parser = commands.add_parser('train')
    train_parser.add_argument('--method', choices=METHODS, required=True)
    train_parser.add_argument('--seed', type=int, default=101)
    train_parser.add_argument('--decisions', type=int, default=4096)
    train_parser.add_argument('--stage', choices=('smoke', 'development'), default='development')
    evaluate_parser = commands.add_parser('evaluate')
    evaluate_parser.add_argument('--run', type=Path, required=True)
    evaluate_parser.add_argument('--output', type=Path)
    select_parser = commands.add_parser('select'); select_parser.add_argument('--budget', type=int, required=True)
    args = parser.parse_args(); install()
    if args.command == 'prepare': print(json.dumps(prepare(), ensure_ascii=False, indent=2))
    elif args.command == 'train': print(train(args.method, args.seed, args.decisions, args.stage))
    elif args.command == 'evaluate': core.evaluate(args.run, args.output)
    else: print(json.dumps(select(args.budget), ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
