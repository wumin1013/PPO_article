"""One-state-coordinate continuation with matched observation-on/off controls.

Reuse the frozen full-batch trainer, local-corner controller and validator.
The former constant-zero macro phase coordinate reports ordered coverage lag;
distance remains a PPO action. No holdout interface or execution correction.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

import preview_full_batch_study as full
from src.environment import cnc_env, kinematics

old = full.core.old
ROOT = old.ROOT
BASE_OUT = full.OUT
OUT = ROOT / '论文项目/revision_20260910_continued/coverage_frontier_v1'
METHODS = ('adaptive', 'fixed3', 'fixed1p2', 'fixed4')
OBSERVATIONS = ('on', 'off')
MILESTONES = (0, 256, 512, 1024, 2048, 4096)
SLOT = 21
BASE_CONFIG, BASE_SOURCE, BASE_TRAIN, BASE_TRAINER = (
    full.configuration, full.source_state, full.train, full.FullBatchTrainer)
_active = False


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if old.read(path) != value:
            raise ValueError(f'Preserve registered artifact: {path}')
    else:
        old.write(path, value)
    return value


def frontier_observation(projection_s, covered_s, half_epsilon):
    values = np.asarray([projection_s, covered_s, half_epsilon], dtype=float)
    if not np.isfinite(values).all() or half_epsilon <= 0:
        raise ValueError('Finite arc coordinates and positive one-sided tolerance required')
    return float(np.clip((projection_s - covered_s) / (10. * half_epsilon), -1., 1.))


class CoverageFrontierEnvironment(full.local.LocalCornerEnvironment):
    def _sense(self):
        # SensorEnvironment.__init__ dispatches here after MacroEnvironment has
        # created the complete inner wrapper, initial coverage and trace row.
        super()._sense()
        raw = self.inner.raw
        coverage = self.inner.env.env.coverage
        projected = float(raw._project_onto_progress_path(raw.current_position, update=False)[2])
        covered = float(coverage.covered_s_mm)
        value = frontier_observation(projected, covered, raw.half_epsilon)
        enabled = self.cfg['revision']['frontier_observation'] == 'on'
        self.state[SLOT] = value if enabled else 0.
        self.trace[-1].update(frontier_projected_s_mm=projected,
            frontier_covered_s_mm=covered, frontier_lag_mm=projected-covered,
            frontier_observation=float(self.state[SLOT]), frontier_observation_enabled=float(enabled))


def configuration(candidate, method, seed, observation='on'):
    if candidate != 'preview_time' or method not in METHODS or seed not in (101, 901) or observation not in OBSERVATIONS:
        raise ValueError('Registered methods, on/off, development seed101 or smoke seed901 required')
    cfg = BASE_CONFIG(candidate, method, seed)
    cfg['seed'] = cfg['experiment']['seed'] = cfg['revision']['seed'] = seed
    cfg['revision'].update(study='coverage-frontier-v1', frontier_observation=observation,
        frontier_state_coordinate=SLOT,
        frontier_definition='clip((ordered_projection_s-covered_s)/(10*half_epsilon),-1,1)',
        frontier_read_timing='Initial coverage, then after each20tick macro; pure projection read, never an extra coverage update',
        candidate_change='Only constant-zero phase coordinate21 becomes ordered coverage lag; off keeps zero',
        initialization='Same method-specific sensor seed101 actor/critic for on/off; zero actor.shared_fc1.weight[:,21] and critic.fc1.weight[:,21]; fresh Adam',
        zero_step_role='Eligible for every method and observation mode; identical migrated initialization',
        validation_milestones=list(MILESTONES), registered_additional_budget=4096,
        screening_budget=1024, extension_requires='Passed source-bound all-grid and closed-loop D intervention gate',
        likelihood='Original positive-feed PPO joint latent Gaussian, constant active-action mask, one draw per20ms')
    cfg['revision']['extra_event_state'][1] = (
        'ordered coverage lag/(10*half_epsilon), clipped[-1,1]' if observation == 'on'
        else 'disabled coverage lag, constant0')
    return cfg


def migrate_frontier_inputs(agent):
    if agent.actor.shared_fc1.in_features != 26 or agent.critic.fc1.in_features != 26:
        raise ValueError('The registered migration requires26D actor and critic')
    with torch.no_grad():
        agent.actor.shared_fc1.weight[:, SLOT].zero_()
        agent.critic.fc1.weight[:, SLOT].zero_()
    # The retained factory has loaded only actor/critic, never old Adam state.
    if agent.actor_optimizer.state or agent.critic_optimizer.state:
        raise ValueError('Migration requires fresh actor and critic Adam states')
    return agent


def make_agent(cfg, env):
    return migrate_frontier_inputs(full.local.make_agent(cfg, env))


def source_state():
    value = BASE_SOURCE()
    path = Path(__file__)
    value['source_files'][path.relative_to(ROOT).as_posix()] = old.sha(path)
    value['source_sha256'] = hashlib.sha256(json.dumps(value['source_files'], sort_keys=True).encode()).hexdigest()
    return value


def prepare():
    old.check_protocol()
    parent = old.read(BASE_OUT / 'protocol.json')
    if parent['source'] != BASE_SOURCE():
        raise ValueError('Original frozen full_batch source changed')
    priors = {}
    for method in ('adaptive', 'fixed3'):
        path = full.local.positive.PRIOR / f'preview_time_{method}_seed101'
        summary = old.read(path / 'training_summary.json')
        digest = old.sha(path / 'checkpoint.pt')
        if summary['decisions'] != 4096 or summary['checkpoint_sha256'] != digest or parent['initializers'][method]['checkpoint_sha256'] != digest:
            raise ValueError('Preserve complete method-specific sensor prior')
        priors[method] = dict(checkpoint=(path/'checkpoint.pt').relative_to(ROOT).as_posix(),
            checkpoint_sha256=digest, macro_decisions=4096,
            config_sha256=old.sha(path/'config.yaml'), provenance_sha256=old.sha(path/'provenance.json'))
    value = dict(version='coverage-frontier-v1', source=source_state(),
        parent_protocol_sha256=old.sha(BASE_OUT/'protocol.json'), methods=list(METHODS),
        observations=list(OBSERVATIONS), seed=101, smoke_seed=901, smoke_decisions=32,
        screening_budget=1024, extension_budget=4096, validation_milestones=list(MILESTONES),
        initial_screen_methods=['adaptive', 'fixed3'], fixed_grid_mm=[1.2, 3., 4.],
        selection_rule='Complete successes first; compare total time only for3/3; earliest checkpoint breaks ties; absolute time tolerance1e-9s',
        extension_gate='On adaptive3/3 beats off adaptive and on uniformly selected fixed; learned3/3 also beats best complete locked-D control, or all locked controls have fewer successes; four on/off adaptive/fixed3 chains plus on fixed1p2/fixed4 finish1024 before final gate',
        scheduled_chains=['on_adaptive','off_adaptive','on_fixed3','off_fixed3','on_fixed1p2','on_fixed4'],
        not_scheduled_chains=['off_fixed1p2','off_fixed4'],
        initializer_rule='Adaptive sensor prior for adaptive; fixed3 sensor prior for all fixed distances; identical on/off migration; shared historical priors counted once, not new independent pretraining',
        priors=priors, new_pretraining_decisions=0,
        feature=dict(coordinate=SLOT, dimension=26, definition='clip((projection_s-covered_s)/(10*half_epsilon),-1,1)',
            projection='ordered update=False', coverage='existing online FineTimeObjective tracker',
            initializer='Zero only first-layer input column21 in actor and critic; original optimizers start fresh'),
        unchanged='Other25state entries, reward, D[0.8,4.6]mapping, joint PPO,64decision batch,20ms decisions,1ms original KCM,v2brake and full trajectory audit',
        validation_path_ids=[p['id'] for p in old.manifest()['paths'] if p['split']=='validation'],
        manifest_sha256=old.sha(old.DATA/'manifest.json'),
        exposure='Existing train/validation only; former six holdout paths exposed and never used here',
        test_access=False, machine_execution_authorized=False)
    immutable_json(OUT/'protocol.json', value)
    for observation in OBSERVATIONS:
        immutable_json(OUT/observation/'protocol.json', dict(value, observation=observation))
    return value


def _trainer(cfg, run, source_hash):
    return BASE_TRAINER(cfg, run, CoverageFrontierEnvironment, make_agent, source_hash)


@contextmanager
def installed(observation='on'):
    global _active
    if observation not in OBSERVATIONS:
        raise ValueError('Observation must be on/off')
    if _active:
        raise RuntimeError('Do not nest frontier installation')
    if full.OUT != BASE_OUT or cnc_env.apply_kinematic_constraints is not kinematics.apply_kinematic_constraints or full.core.plan_terminal_brake.__module__ != 'terminal_braking':
        raise RuntimeError('Use a clean process with original KCM and v2brake')
    full_names = ('OUT', 'configuration', 'source_state', 'prepare', 'FullBatchTrainer')
    core_names = ('OUT', 'configuration', 'source_state', 'prepare', 'MacroEnvironment', 'make_agent')
    before_full = {name:getattr(full, name) for name in full_names}
    before_core = {name:getattr(full.core, name) for name in core_names}
    _active = True
    try:
        factory = lambda candidate, method, seed: configuration(candidate, method, seed, observation)
        protocol = lambda: dict(prepare(), observation=observation)
        full.OUT = full.core.OUT = OUT/observation
        full.configuration = full.core.configuration = factory
        full.source_state = full.core.source_state = source_state
        full.prepare = full.core.prepare = protocol
        full.FullBatchTrainer = _trainer
        full.core.MacroEnvironment, full.core.make_agent = CoverageFrontierEnvironment, make_agent
        yield full.core
    finally:
        for name, value in before_full.items(): setattr(full, name, value)
        for name, value in before_core.items(): setattr(full.core, name, value)
        _active = False


def run_directory(observation, method, stage='development'):
    seed = 901 if stage == 'smoke' else 101
    return OUT/observation/stage/f'preview_time_{method}_seed{seed}'


def _request(observation, method, decisions, stage):
    if observation not in OBSERVATIONS or method not in METHODS or stage not in ('smoke', 'development'):
        raise ValueError('Registered observation, method and stage required')
    if decisions not in ((32,) if stage == 'smoke' else MILESTONES):
        raise ValueError('Registered smoke32 or development milestone required')
    if decisions > 1024:
        gate_path = OUT/'advance_gate.json'
        if not gate_path.exists():
            raise ValueError('Extension requires passed1024 selection and D-intervention gate')
        gate = old.read(gate_path)
        if not gate['passed'] or gate['source_sha256'] != source_state()['source_sha256']:
            raise ValueError('Extension gate source or result changed')
        for key in ('selection', 'intervention'):
            entry = gate[key]
            if old.sha(ROOT/entry['path']) != entry['sha256']:
                raise ValueError('Extension gate evidence changed')


def checked_snapshot(snapshot):
    for name, digest in old.read(snapshot/'snapshot_manifest.json')['files'].items():
        if old.sha(snapshot/name) != digest:
            raise ValueError(f'Saved snapshot artifact changed: {name}')
    return old.read(snapshot/'training_summary.json')


def train(observation, method, decisions, stage='development'):
    _request(observation, method, decisions, stage)
    seed = 901 if stage == 'smoke' else 101
    with installed(observation):
        snapshot = BASE_TRAIN(method, seed, decisions, stage)
    state = torch.load(snapshot/'checkpoint.pt', map_location='cpu', weights_only=False)
    summary = checked_snapshot(snapshot)
    immutable_json(snapshot/'frontier_diagnostics.json', dict(
        observation=observation, method=method, seed=seed, decisions=decisions,
        optimized_decisions=summary['optimized_decisions'], optimizer_updates=summary['optimizer_updates'],
        actor_input_column21_l2=float(state['actor']['shared_fc1.weight'][:, SLOT].norm()),
        critic_input_column21_l2=float(state['critic']['fc1.weight'][:, SLOT].norm()),
        checkpoint_sha256=old.sha(snapshot/'checkpoint.pt'), source_sha256=state['source_sha256']))
    return snapshot


def checked_validation(snapshot, output=None):
    checked_snapshot(snapshot)
    folder = snapshot/'validation' if output is None else Path(output)
    rows = old.read(folder/'summary.json')['rows']
    protocol = prepare()
    expected = set(protocol['validation_path_ids'])
    if len(rows) != len(expected) or {r['path_id'] for r in rows} != expected:
        raise ValueError('All three registered validation paths required')
    for row in rows:
        if (row['checkpoint_sha256'] != old.sha(snapshot/'checkpoint.pt') or
            row['source_sha256'] != protocol['source']['source_sha256'] or
            row['config_sha256'] != old.sha(snapshot/'config.yaml') or
            row['trace_sha256'] != old.sha(folder/f'{row["path_id"]}.npz')):
            raise ValueError('Validation model/config/source/trace binding changed')
    return rows


def evaluate(observation, method, decisions, stage='development'):
    _request(observation, method, decisions, stage)
    snapshot = run_directory(observation, method, stage)/'checkpoints'/f'step{decisions}'
    checked_snapshot(snapshot)
    with installed(observation) as core:
        if not (snapshot/'validation/summary.json').exists():
            core.evaluate(snapshot)
        rows = checked_validation(snapshot)
    return rows


def performance(rows):
    success = sum(bool(row['success']) for row in rows)
    return dict(successes=success, count=len(rows),
        complete_time_s=sum(row['elapsed_time_s'] for row in rows) if success == 3 and len(rows) == 3 else None)


def strictly_better(candidate, control):
    if candidate['successes'] != control['successes']:
        return candidate['successes'] > control['successes']
    return bool(candidate['successes'] == 3 and
        candidate['complete_time_s'] < control['complete_time_s'] - 1e-9)


def choose(records):
    selected = records[0]
    for row in records[1:]:
        if strictly_better(row, selected) or (not strictly_better(selected, row) and row['step'] < selected['step']):
            selected = row
    return selected


def intervene(observation, method, decisions, output):
    """Run the ordinary full validator; only replace its sampled D at inference."""
    if method != 'adaptive':
        raise ValueError('D intervention requires the adaptive actor')
    _request(observation, method, decisions, 'development')
    snapshot = run_directory(observation, method)/'checkpoints'/f'step{decisions}'
    checked_validation(snapshot)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve prior intervention; use a new output directory')
    protocol = prepare()
    binding = dict(observation=observation, method=method, decisions=decisions,
        checkpoint_sha256=old.sha(snapshot/'checkpoint.pt'), config_sha256=old.sha(snapshot/'config.yaml'),
        source_sha256=protocol['source']['source_sha256'], manifest_sha256=protocol['manifest_sha256'],
        distances_mm=[None, 1.2, 3., 4.], test_access=False,
        intervention='Recompute original actor feed from each actual intervened state; replace D only; not separately trained fixed methods')
    immutable_json(output/'protocol.json', binding)
    rows = []
    with installed(observation) as core:
        for distance in (None, 1.2, 3., 4.):
            label = 'learned' if distance is None else f'fixed_D{distance:g}'
            def intervened_agent(cfg, env, distance=distance):
                agent = make_agent(cfg, env)
                take = agent.take_action
                def action(state, **kwargs):
                    result = take(state, **kwargs)
                    if kwargs.get('return_info'):
                        raise ValueError('Intervention is deterministic validation only')
                    if distance is not None:
                        result = np.asarray(result, dtype=float).copy()
                        result[2] = (distance-.8)/3.8
                    return result
                agent.take_action = action
                return agent
            core.make_agent = intervened_agent
            core.evaluate(snapshot, output/label)
            part = checked_validation(snapshot, output/label)
            rows.extend(dict(row, intervention=label) for row in part)
    original = {r['path_id']:r for r in checked_validation(snapshot)}
    for row in rows[:3]:
        for key in ('success', 'elapsed_time_s', 'ordered_progress', 'contour_error_max_mm'):
            if row[key] != original[row['path_id']][key]:
                raise ValueError(f'Unmodified closed-loop replay differs: {key}')
    groups = {label:performance([r for r in rows if r['intervention']==label])
              for label in ('learned', 'fixed_D1.2', 'fixed_D3', 'fixed_D4')}
    supported = groups['learned']['successes'] == 3 and all(
        strictly_better(groups['learned'], groups[label]) for label in groups if label != 'learned')
    value = dict(binding=binding, groups=groups, rows=rows, d_effect_supported=supported)
    immutable_json(output/'summary.json', value)
    return value


def select(budget=1024, intervention=None):
    if budget not in (1024, 4096):
        raise ValueError('Select completed1024 or4096 budgets only')
    protocol = prepare()
    groups, missing, not_scheduled = {}, [], []
    for observation in OBSERVATIONS:
        for method in METHODS:
            label = f'{observation}_{method}'
            run = run_directory(observation, method)
            if not (run/'training_summary.json').exists() or old.read(run/'training_summary.json')['decisions'] < budget:
                if observation == 'off' and method in ('fixed1p2','fixed4'):
                    not_scheduled.append(label)
                else:
                    missing.append(label)
                continue
            candidates = []
            for step in MILESTONES:
                if step > budget: continue
                snapshot = run/'checkpoints'/f'step{step}'
                rows = checked_validation(snapshot)
                candidates.append(dict(step=step, observation=observation, method=method,
                    **performance(rows), checkpoint=snapshot.relative_to(ROOT).as_posix(),
                    checkpoint_sha256=old.sha(snapshot/'checkpoint.pt'),
                    validation_sha256=old.sha(snapshot/'validation/summary.json')))
            groups[label] = dict(selected=choose(candidates), all_checkpoints=candidates)
    required_screen = ('on_adaptive', 'off_adaptive', 'on_fixed3', 'off_fixed3')
    screen_ready = all(label in groups for label in required_screen)
    screen_eligible = False
    if screen_ready:
        active = groups['on_adaptive']['selected']
        screen_eligible = active['successes'] == 3 and all(strictly_better(active, groups[label]['selected'])
            for label in ('off_adaptive', 'on_fixed3'))
    complete_grid = not missing
    fixed, candidate_better = {}, False
    if complete_grid:
        fixed['on'] = choose([groups[f'on_{method}']['selected'] for method in METHODS if method != 'adaptive'])
        active = groups['on_adaptive']['selected']
        candidate_better = active['successes'] == 3 and strictly_better(active, groups['off_adaptive']['selected']) and strictly_better(active, fixed['on'])
    intervention_entry, d_effect = None, False
    if intervention is not None:
        path = Path(intervention).resolve()
        if path.is_dir(): path = path/'summary.json'
        evidence = old.read(path)
        binding = evidence['binding']
        if 'on_adaptive' not in groups or binding['checkpoint_sha256'] != groups['on_adaptive']['selected']['checkpoint_sha256'] or binding['source_sha256'] != protocol['source']['source_sha256'] or binding['observation'] != 'on':
            raise ValueError('D intervention must bind the selected on-adaptive checkpoint/source')
        # Verify all12 actual files, not only a favorable summary flag.
        snapshot = ROOT/groups['on_adaptive']['selected']['checkpoint']
        for label in ('learned', 'fixed_D1.2', 'fixed_D3', 'fixed_D4'):
            measured = performance(checked_validation(snapshot, path.parent/label))
            if evidence['groups'][label] != measured:
                raise ValueError('Intervention metrics changed')
        learned = evidence['groups']['learned']
        d_effect = learned['successes'] == 3 and all(strictly_better(learned, control)
            for label, control in evidence['groups'].items() if label != 'learned')
        intervention_entry = dict(path=path.relative_to(ROOT).as_posix(), sha256=old.sha(path))
    passed = bool(complete_grid and candidate_better and d_effect)
    suffix = 'screen' if not complete_grid else ('with_intervention' if intervention is not None else 'full')
    path = OUT/f'selection_budget{budget}_{suffix}.json'
    value = dict(budget=budget, source_sha256=protocol['source']['source_sha256'], groups=groups,
        missing_chains=missing, not_scheduled_chains=not_scheduled, screen_ready=screen_ready, screen_eligible_for_fixed_grid=screen_eligible,
        complete_grid=complete_grid, uniform_fixed=fixed, candidate_better=candidate_better,
        intervention=intervention_entry, d_effect_supported=d_effect, extension_gate_passed=passed,
        interpretation='Development screening only; no test or five-independent-seed conclusion')
    immutable_json(path, value)
    if budget == 1024 and passed:
        immutable_json(OUT/'advance_gate.json', dict(passed=True, source_sha256=value['source_sha256'],
            selection=dict(path=path.relative_to(ROOT).as_posix(), sha256=old.sha(path)), intervention=intervention_entry))
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('prepare')
    for name in ('train', 'evaluate', 'intervene'):
        p = commands.add_parser(name)
        p.add_argument('--observation', choices=OBSERVATIONS, required=True)
        p.add_argument('--method', choices=METHODS, required=True)
        p.add_argument('--decisions', type=int, choices=(*MILESTONES, 32), required=True)
        if name == 'intervene': p.add_argument('--output', type=Path, required=True)
        else: p.add_argument('--stage', choices=('smoke', 'development'), default='development')
    p = commands.add_parser('select')
    p.add_argument('--budget', type=int, choices=(1024, 4096), default=1024)
    p.add_argument('--intervention', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    record = dict(argv=sys.argv, started_utc=datetime.now(timezone.utc).isoformat(), status='running',
        runner_sha256=old.sha(Path(__file__)), torch_threads=1)
    log = OUT/'invocations'/f'{time.time_ns()}_{args.command}.json'
    old.write(log, record)
    start = time.perf_counter()
    try:
        if args.command == 'prepare':
            prepare(); result = dict(protocol=str(OUT/'protocol.json'))
        elif args.command == 'train':
            result = dict(snapshot=str(train(args.observation, args.method, args.decisions, args.stage)))
        elif args.command == 'evaluate':
            rows = evaluate(args.observation, args.method, args.decisions, args.stage)
            result = dict(**performance(rows), snapshot=str(run_directory(args.observation,args.method,args.stage)/'checkpoints'/f'step{args.decisions}'))
        elif args.command == 'intervene':
            value = intervene(args.observation, args.method, args.decisions, args.output)
            result = dict(summary=str(args.output/'summary.json'), d_effect_supported=value['d_effect_supported'])
        else:
            value = select(args.budget, args.intervention)
            result = {key:value[key] for key in ('screen_ready','screen_eligible_for_fixed_grid','missing_chains','complete_grid','d_effect_supported','extension_gate_passed')}
        record.update(status='complete', result=result)
        print(json.dumps(result, ensure_ascii=False))
    except BaseException as error:
        record.update(status='failed_or_interrupted', error=repr(error))
        raise
    finally:
        record.update(wall_s=time.perf_counter()-start, finished_utc=datetime.now(timezone.utc).isoformat())
        old.write(log, record)


if __name__ == '__main__':
    main()
