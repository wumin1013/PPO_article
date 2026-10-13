"""Independent two-phase target-tracking training; no held-out geometry access.

Reuse the frozen sensor loop, FullBatchTrainer, checkpoint and validation code.
Each method/seed really trains its own sensor prior under original execution;
only continuation uses the common scalar feed-target correction. A separate
new holdout gate freezes and evaluates the completed ten-model selection.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess
import time
import zipfile

import torch
import yaml

import preview_formal_continuation as base
import preview_target_tracking_study as target

full, core, sensor, old, ROOT = base.full, base.core, base.sensor, base.old, base.ROOT
BASE = ROOT/'论文项目/revision_20260910_continued'
OUT = BASE/'target_tracking_formal_v1'
VERSION = 'target-tracking-formal-v1'
SEEDS, METHODS = (111, 222, 333, 444, 555), ('fixed3', 'adaptive')
MILESTONES = (0, 256, 512, 1024, 2048, 4096)
PRETRAIN_DECISIONS = ADDITIONAL_DECISIONS = 4096
HOLDOUT_MANIFEST = ROOT/'PPO_project/data/preview_frontier_holdout_v1/manifest.json'
EXTENSION_OUT = BASE/'target_tracking_extension_v1'
EVALUATION_FILES = (
    'PPO_project/tools/preview_formal_holdout.py',
    'PPO_project/tools/preview_formal_statistics.py',
    'PPO_project/tools/preview_formal_two_step.py',
    'PPO_project/tools/preview_two_step_eval.py',
    'PPO_project/tools/preview_target_tracking_holdout.py',
    'PPO_project/tests/test_preview_target_tracking_holdout.py',
    'PPO_project/tools/preview_target_tracking_statistics.py',
    'PPO_project/tests/test_preview_target_tracking_statistics.py',
    'PPO_project/tools/preview_target_tracking_two_step.py',
    'PPO_project/tests/test_preview_target_tracking_two_step.py',
)
DRIVER_FILES = (
    'PPO_project/tools/preview_formal_continuation.py',
    'PPO_project/tools/preview_target_tracking_formal.py',
    'PPO_project/tests/test_preview_target_tracking_formal.py',
)
BASE_SENSOR_AGENT, BASE_TRAIN = base.BASE_SENSOR_AGENT, base.BASE_TRAIN
_TRAIN_PRETRAIN, _TRAIN_CONTINUE = base.train_pretraining, base.train_continuation
_EVALUATE, _SELECT = base.evaluate, base.select
_installed_phase = None
_delegating = False


def artifact(path):
    path = Path(path)
    return dict(file=path.relative_to(ROOT).as_posix(), sha256=old.sha(path))


immutable_json = base.immutable_json
selection_key = base.selection_key
verify_snapshot = base.verify_snapshot


def verify_artifacts(artifacts):
    for item in artifacts.values():
        if old.sha(ROOT/item['file']) != item['sha256']:
            raise ValueError(f'Artifact changed: {item["file"]}')


def _source(files, head=None):
    return dict(source_files=files,
        source_sha256=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
        head=head)


def source_state(phase):
    if phase not in ('pretraining', 'continuation'):
        raise ValueError('Unknown training phase')
    prior = sensor.source_state() if phase == 'pretraining' else target.source_state()
    files = dict(prior['source_files'])
    for name in (*DRIVER_FILES, *EVALUATION_FILES):
        files[name] = old.sha(ROOT/name)
    return _source(files, prior.get('head'))


def evaluation_sources():
    files = {name: old.sha(ROOT/name) for name in EVALUATION_FILES}
    return _source(files)


def development_gate():
    """Verify complete development results; only descriptors enter this study."""
    import preview_target_tracking_extension as extension
    if extension.OUT != EXTENSION_OUT:
        raise ValueError('Use the registered extension directory')
    delivery = old.read(EXTENSION_OUT/'delivery_manifest.json')
    extension.audit.verify_files(EXTENSION_OUT, delivery['files'], {})
    completion = old.read(EXTENSION_OUT/'completion_audit.json')
    if (not completion['passed'] or completion['coordinator_exit_code'] != 0
            or completion['all_validation_rows_verified'] != 72):
        raise ValueError('All extension training and validation must finish before formal registration')
    # report verifies all six nodes for all four methods and preserves its
    # immutable summary. No training or policy rollout occurs here.
    report = extension.report()
    if report['uniform_fixed_method'] != 'fixed3':
        raise ValueError('Uniform distance changed; report to the research coordinator before registration')
    if not report['adaptive_beats_uniform_fixed']:
        raise ValueError('Complete 4096 adaptive/fixed development gate must pass')
    if not report['selected_actor_D_gate']['same_actor_D_gate_passed']:
        raise ValueError('Selected adaptive actor requires its applicable completed D-effect gate')
    adaptive = report['groups']['adaptive']
    if adaptive['success_count'] != 3 or adaptive['complete_total_time_s'] >= 13.931-1e-9:
        raise ValueError('Selected adaptive must beat the original off-adaptive complete comparator')
    files = [EXTENSION_OUT/name for name in
             ('protocol.json', 'summary.json', 'completion_audit.json', 'delivery_manifest.json')]
    return dict(passed=True, uniform_fixed_method='fixed3', uniform_fixed_distance_mm=3.,
        same_actor_D_gate_passed=True,
        descriptors={path.name: artifact(path) for path in files},
        selected_adaptive_checkpoint_sha256=adaptive['checkpoint_sha256'],
        selected_fixed_checkpoint_sha256=report['groups']['fixed3']['checkpoint_sha256'],
        adaptive_complete_time_s=adaptive['complete_total_time_s'],
        fixed_complete_time_s=report['groups']['fixed3']['complete_total_time_s'],
        interpretation='Single-seed development admission only; no repeatability or new test evidence')


def prepare():
    path = OUT/'protocol.json'
    if path.exists():
        protocol = old.read(path)
        if (protocol['version'] != VERSION or protocol['seeds'] != list(SEEDS)
                or protocol['methods'] != list(METHODS)
                or protocol['validation_milestones'] != list(MILESTONES)):
            raise ValueError('Preserve the registered formal version, seeds and milestones')
        for phase in ('pretraining', 'continuation'):
            if protocol['sources'][phase] != source_state(phase):
                raise ValueError('Frozen target formal source changed')
        if protocol['evaluation_sources'] != evaluation_sources():
            raise ValueError('Frozen evaluation source changed')
        verify_artifacts(protocol['development_gate']['descriptors'])
        verify_artifacts({key: protocol[key] for key in ('training_validation_manifest', 'holdout_manifest')})
        return protocol
    old.check_protocol()
    gate = development_gate()
    # Hash the held-out manifest as opaque bytes. Never parse it or load its
    # coordinate arrays in a training/selection/configuration entry point.
    manifest = old.manifest()
    protocol = dict(version=VERSION, seeds=list(SEEDS), methods=list(METHODS),
        pretraining_decisions=PRETRAIN_DECISIONS, additional_decisions=ADDITIONAL_DECISIONS,
        total_available_decisions_per_method_seed=8192, training_budget_total_decisions=81920,
        validation_milestones=list(MILESTONES), selection_milestones=list(MILESTONES),
        selection_rule='Maximize complete success count across all three validation paths; compare total time only at 3/3; ties use earliest checkpoint',
        step0_policy='Equally eligible for both methods after every chain expends the complete continuation budget',
        initialization='Ten newly executed same-method/same-seed sensor chains from random initialization, 4096 decisions each; continuation loads only its own actor/critic and starts fresh Adam',
        prior_reuse=False, new_pretraining_decisions=40960, new_continuation_decisions=40960,
        formal_seeds_previously_used=False,
        phase_execution=dict(pretraining='Original scalar/angular KCM and v2 terminal brake',
            continuation='Common target-tracking scalar feed correction; original angular KCM and v2 terminal brake'),
        sources={phase: source_state(phase) for phase in ('pretraining', 'continuation')},
        evaluation_sources=evaluation_sources(), development_gate=gate,
        uniform_fixed_method='fixed3', uniform_fixed_distance_mm=3.,
        training_validation_manifest=artifact(old.DATA/'manifest.json'),
        holdout_manifest=artifact(HOLDOUT_MANIFEST),
        holdout_descriptor_policy='SHA-only opaque manifest binding; no coordinate or descriptor parsing by training runner',
        training_path_ids=[p['id'] for p in manifest['paths'] if p['split'] == 'train'],
        validation_path_ids=[p['id'] for p in manifest['paths'] if p['split'] == 'validation'],
        pretraining_batch_rule='Frozen sensor loop: flush at 64, episode end or final budget',
        continuation_batch_rule='Frozen FullBatchTrainer: full 64 across episode ends; keep terminal GAE boundaries',
        rng_policy='Reset Python/NumPy/Torch to the same registered independent seed at each phase; resume full checkpoint state',
        terminal_brake='Original v2, full output rest tail included; no v3',
        feed_projection='Original in sensor; target corrected scalar feed in continuation and validation, equally for both methods',
        geometry_policy='Original 1.245 mm certification radius, complete ordered coverage and position-derived six constraints',
        costs='Separate actual sensor, continuation, validations, restore/replay, wall time and macro/physical/command/update counts; step snapshots are cumulative, not independent chains',
        validation_role='Development checkpoint selection only',
        test_access=False, adopted=False, machine_execution_authorized=False)
    immutable_json(path, protocol)
    for phase in ('pretraining', 'continuation'):
        immutable_json(OUT/phase/'protocol.json', dict(source=protocol['sources'][phase],
            phase=phase, formal_protocol=artifact(path)))
    with zipfile.ZipFile(OUT/'registered_source.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        names = set().union(*(value['source_files'] for value in protocol['sources'].values()))
        for name in sorted(names): archive.write(ROOT/name, name)
    return protocol


def validate_request(method, seed, stage):
    if method not in METHODS or stage not in ('formal', 'smoke'):
        raise ValueError('Registered fixed3/adaptive methods and formal/smoke stages only')
    if seed not in (SEEDS if stage == 'formal' else (901,)):
        raise ValueError('Formal seeds111/222/333/444/555; independent smoke901 only')


def run_dir(phase, method, seed, stage='formal'):
    return OUT/phase/stage/f'preview_time_{method}_seed{seed}'


def _own_prior(method, seed, stage):
    prior = run_dir('pretraining', method, seed, stage)
    summary = old.read(prior/'training_summary.json')
    expected = PRETRAIN_DECISIONS if stage == 'formal' else 64
    if (summary['seed'] != seed or summary['method'] != method or summary['decisions'] != expected
            or summary['checkpoint_sha256'] != old.sha(prior/'checkpoint.pt')):
        raise ValueError('Own same-method/same-seed complete freshly trained sensor prior required')
    complete = old.read(prior/'phase_complete.json')
    if complete['seed'] != seed or complete['method'] != method or complete['phase'] != 'pretraining':
        raise ValueError('Own sensor phase completion identity mismatch')
    verify_artifacts(complete['artifacts'])
    cfg = yaml.safe_load((prior/'config.yaml').read_text(encoding='utf-8'))
    if cfg != configuration('pretraining', method, seed, stage):
        raise ValueError('Own sensor config is not the registered original-execution phase')
    provenance = old.read(prior/'provenance.json')
    if (provenance['source_sha256'] != source_state('pretraining')['source_sha256']
            or provenance['config_sha256'] != old.sha(prior/'config.yaml')):
        raise ValueError('Own sensor source/config provenance changed')
    return prior, expected


def configuration(phase, method, seed, stage='formal'):
    validate_request(method, seed, stage)
    if phase == 'pretraining':
        cfg = sensor.configuration('preview_time', method, seed)
        if 'init_checkpoint' in cfg['revision'] or 'feed_tracking' in cfg['revision']:
            raise ValueError('Fresh original sensor initialization only')
    elif phase == 'continuation':
        cfg = target.configuration('preview_time', method, 101)
        prior, expected = _own_prior(method, seed, stage)
        cfg['seed'] = seed
        cfg['revision'].update(init_checkpoint=(prior/'checkpoint.pt').relative_to(ROOT).as_posix(),
            init_checkpoint_sha256=old.sha(prior/'checkpoint.pt'), pretraining_macro_decisions=expected,
            initialization='Own newly trained same-method/same-seed sensor actor/critic only; fresh Adam',
            zero_step_role='Equally eligible after all registered continuation and validation work completes')
    else:
        raise ValueError('Unknown training phase')
    cfg['revision'].update(formal_study=VERSION, formal_phase=phase,
        data_status='All three development validation paths select checkpoints; no holdout access')
    return cfg


@contextmanager
def phase_installed(phase, stage='formal'):
    global _installed_phase
    if phase not in ('pretraining', 'continuation'):
        raise ValueError('Unknown training phase')
    if _installed_phase is not None:
        raise RuntimeError('Do not nest formal phase installations')
    if (base.cnc_env.apply_kinematic_constraints is not base.kinematics.apply_kinematic_constraints
            or core.plan_terminal_brake.__module__ != 'terminal_braking'
            or BASE_SENSOR_AGENT.__module__ != 'preview_macro_study'):
        raise RuntimeError('Start a fresh process with original angular KCM and v2 brake')
    protocol = prepare()
    names = ('OUT', 'source_state', 'configuration', 'MacroEnvironment', 'make_agent', 'prepare')
    saved = {name: getattr(core, name) for name in names}
    original = base.cnc_env.apply_kinematic_constraints
    _installed_phase = phase
    try:
        core.OUT = OUT/phase
        core.source_state = lambda: source_state(phase)
        core.configuration = lambda candidate, method, seed: configuration(phase, method, seed, stage)
        core.MacroEnvironment = sensor.SensorEnvironment if phase == 'pretraining' else full.local.LocalCornerEnvironment
        core.make_agent = BASE_SENSOR_AGENT if phase == 'pretraining' else full.local.make_agent
        core.prepare = lambda: dict(source=protocol['sources'][phase])
        if phase == 'continuation':
            base.cnc_env.apply_kinematic_constraints = target.apply_target_tracking_constraints
        immutable_json(core.OUT/'protocol.json', dict(source=protocol['sources'][phase],
            phase=phase, formal_protocol=artifact(OUT/'protocol.json')))
        yield core
    finally:
        base.cnc_env.apply_kinematic_constraints = original
        for name, value in saved.items(): setattr(core, name, value)
        _installed_phase = None


@contextmanager
def invocation(phase, method, seed, stage):
    path = OUT/'invocations'/f'{time.time_ns()}_{phase}_{method}_{seed}.json'
    record = dict(phase=phase, method=method, seed=seed, stage=stage,
        argv=[sys.executable, *(['-B'] if sys.dont_write_bytecode else []), *sys.argv],
        pid=os.getpid(), parent_pid=os.getppid(), cwd=str(Path.cwd()),
        started_utc=datetime.now(timezone.utc).isoformat(), status='running',
        runner_sha256=old.sha(Path(__file__)), protocol_sha256=old.sha(OUT/'protocol.json'),
        timing_scope='Function wall including checking/replay/compute/storage; parent process wall includes this time')
    old.write(path, record); began = time.perf_counter()
    try:
        yield
    except BaseException as error:
        record.update(status='failed_or_interrupted', exit_code=1, error=repr(error))
        raise
    else:
        record.update(status='complete', exit_code=0)
    finally:
        record.update(wall_s=time.perf_counter()-began, finished_utc=datetime.now(timezone.utc).isoformat())
        old.write(path, record)


@contextmanager
def _delegated():
    """Scope only orchestration bindings; inherited numerical functions stay exact."""
    global _delegating
    if _delegating:
        raise RuntimeError('Do not nest formal delegation')
    values = dict(ROOT=ROOT, OUT=OUT, SEEDS=SEEDS, METHODS=METHODS, MILESTONES=MILESTONES,
        PRETRAIN_DECISIONS=PRETRAIN_DECISIONS, ADDITIONAL_DECISIONS=ADDITIONAL_DECISIONS,
        prepare=prepare, source_state=source_state, configuration=configuration,
        run_dir=run_dir, validate_request=validate_request, phase_installed=phase_installed,
        artifact=artifact, verify_artifacts=verify_artifacts, invocation=invocation)
    saved = {name: getattr(base, name) for name in values}
    _delegating = True
    try:
        for name, value in values.items(): setattr(base, name, value)
        yield
    finally:
        for name, value in saved.items(): setattr(base, name, value)
        _delegating = False


def train_pretraining(method, seed, stage='formal'):
    prepare()
    with _delegated():
        result = _TRAIN_PRETRAIN(method, seed, stage)
    _own_prior(method, seed, stage)
    return result


def train_continuation(method, seed, decisions, stage='formal'):
    prepare()
    with _delegated():
        result = _TRAIN_CONTINUE(method, seed, decisions, stage)
    verify_node(method, seed, decisions, stage, require_validation=False)
    return result


def verify_node(method, seed, decisions, stage='formal', require_validation=True):
    validate_request(method, seed, stage)
    if decisions not in (MILESTONES if stage == 'formal' else (0, 64)):
        raise ValueError('Registered milestone only')
    snapshot = run_dir('continuation', method, seed, stage)/'checkpoints'/f'step{decisions}'
    verify_snapshot(snapshot)
    cfg = configuration('continuation', method, seed, stage)
    if yaml.safe_load((snapshot/'config.yaml').read_text(encoding='utf-8')) != cfg:
        raise ValueError('Snapshot must use the same registered own-prior configuration')
    protocol = prepare()
    provenance = old.read(snapshot/'provenance.json')
    binding = dict(checkpoint_sha256=old.sha(snapshot/'checkpoint.pt'),
        config_sha256=old.sha(snapshot/'config.yaml'),
        source_sha256=protocol['sources']['continuation']['source_sha256'])
    if (provenance['source_sha256'] != binding['source_sha256']
            or provenance['config_sha256'] != binding['config_sha256']):
        raise ValueError('Snapshot source/config provenance mismatch')
    summary = old.read(snapshot/'training_summary.json')
    expected = dict(seed=seed, method=method, decisions=decisions, optimized_decisions=decisions,
        pending_buffer_decisions=0, checkpoint_sha256=binding['checkpoint_sha256'])
    if any(summary.get(key) != value for key, value in expected.items()):
        raise ValueError('Snapshot seed/method/complete optimized budget/model identity mismatch')
    if decisions == 0:
        payload = torch.load(snapshot/'checkpoint.pt', map_location='cpu', weights_only=False)
        if payload['actor_optimizer']['state'] or payload['critic_optimizer']['state']:
            raise ValueError('Both continuation optimizers must start fresh at step0')
    if require_validation:
        rows = old.read(snapshot/'validation/summary.json')['rows']
        if len(rows) != 3 or {row['path_id'] for row in rows} != set(protocol['validation_path_ids']):
            raise ValueError('All three development validation paths are mandatory')
        for row in rows:
            if (any(row.get(key) != value for key, value in binding.items())
                    or row.get('seed') != seed or row.get('method') != method
                    or row.get('training_decisions') != decisions
                    or old.sha(snapshot/'validation'/f'{row["path_id"]}.npz') != row['trace_sha256']):
                raise ValueError('Validation row must identify its exact method/seed/node/source/config/model/trace')
    return snapshot


def evaluate(method, seed, decisions, stage='formal'):
    verify_node(method, seed, decisions, stage, require_validation=False)
    with _delegated():
        rows = _EVALUATE(method, seed, decisions, stage)
    verify_node(method, seed, decisions, stage)
    return rows


def select(method, seed):
    validate_request(method, seed, 'formal')
    for step in MILESTONES:
        verify_node(method, seed, step)
    with _delegated():
        return _SELECT(method, seed)


def chain(method, seed, stage='formal'):
    train_pretraining(method, seed, stage)
    for step in (MILESTONES if stage == 'formal' else (0, 64)):
        train_continuation(method, seed, step, stage)
        evaluate(method, seed, step, stage)
    return select(method, seed) if stage == 'formal' else dict(status='smoke_complete')


def execute_chain(method, seed, stage, directory):
    """One separate process per independent RNG/optimizer chain, fully logged."""
    validate_request(method, seed, stage)
    label = f'{stage}_{method}_seed{seed}'
    path = directory/f'{label}.json'
    if path.exists():
        raise FileExistsError('Preserve prior invocation; resume into a new batch receipt directory')
    argv = [sys.executable, '-B', '-u', str(Path(__file__).resolve()), '--output-root', str(OUT),
            'chain', '--method', method, '--seed', str(seed), '--stage', stage]
    record = dict(argv=argv, command=subprocess.list2cmdline(argv), cwd=str(ROOT),
        parent_pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat(), status='starting')
    old.write(path, record); began = time.perf_counter()
    try:
        with (directory/f'{label}.stdout.log').open('x', encoding='utf-8') as stdout, (
                directory/f'{label}.stderr.log').open('x', encoding='utf-8') as stderr:
            process = subprocess.Popen(argv, cwd=ROOT, stdout=stdout, stderr=stderr,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            record.update(pid=process.pid, status='running'); old.write(path, record)
            code = process.wait()
        record.update(exit_code=code, status='complete' if code == 0 else 'failed')
    except BaseException as error:
        record.update(status='failed_or_interrupted', error=repr(error))
        raise
    finally:
        record.update(wall_s=time.perf_counter()-began, finished_utc=datetime.now(timezone.utc).isoformat(),
            log_sha256={p.name: old.sha(p) for p in directory.glob(f'{label}.*.log')})
        old.write(path, record)
    print(f'{label}: exit={code}, wall={record["wall_s"]:.3f}s', flush=True)
    return record


def batch(workers=3, stage='formal'):
    if not 1 <= workers <= 3 or stage not in ('formal', 'smoke'):
        raise ValueError('One to three workers; formal or smoke stage only')
    protocol = prepare()
    directory = OUT/'batch_invocations'/f'{stage}_{time.time_ns()}'
    directory.mkdir(parents=True)
    record = dict(stage=stage, workers=workers, pid=os.getpid(), argv=sys.argv,
        started_utc=datetime.now(timezone.utc).isoformat(), status='running',
        protocol_sha256=old.sha(OUT/'protocol.json'), commands=[],
        timing_scope='Batch wall is not the sum of concurrent child walls; child wall contains phase/replay/evaluation costs',
        test_access=False, automatic_freeze=False)
    path = directory/'summary.json'; old.write(path, record); began = time.perf_counter()
    try:
        seeds = SEEDS if stage == 'formal' else (901,)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(execute_chain, method, seed, stage, directory) for seed in seeds for method in METHODS]
            for future in as_completed(futures):
                record['commands'].append(future.result())
                old.write(path, record)
        if any(row['exit_code'] != 0 for row in record['commands']):
            raise RuntimeError('Formal chain failure; retain artifacts and resume the exact same chain')
        if prepare() != protocol:
            raise ValueError('Registered protocol changed during batch')
        record.update(status='complete', exit_code=0)
    except BaseException as error:
        record.update(status='failed_or_interrupted', exit_code=1, error=repr(error))
        raise
    finally:
        record.update(wall_s=time.perf_counter()-began, finished_utc=datetime.now(timezone.utc).isoformat(),
            child_process_wall_s=sum(row['wall_s'] for row in record['commands']))
        old.write(path, record)
    return record


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=OUT)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare')
    p = sub.add_parser('batch')
    p.add_argument('--workers', type=int, choices=(1, 2, 3), default=3)
    p.add_argument('--stage', choices=('formal', 'smoke'), default='formal')
    for name in ('pretrain', 'train', 'evaluate', 'select', 'chain'):
        p = sub.add_parser(name)
        p.add_argument('--method', choices=METHODS, required=True)
        p.add_argument('--seed', type=int, required=True)
        if name != 'select': p.add_argument('--stage', choices=('formal', 'smoke'), default='formal')
        if name in ('train', 'evaluate'): p.add_argument('--decisions', type=int, required=True)
    args = parser.parse_args()
    OUT = args.output_root.resolve(); OUT.relative_to(BASE)
    if OUT in (BASE/'formal_v1', BASE/'target_tracking_v1', EXTENSION_OUT):
        raise ValueError('Use a new independent target formal output directory')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.command == 'prepare':
        prepare(); result = dict(protocol=str(OUT/'protocol.json'), training_started=False)
    elif args.command == 'batch': result = batch(args.workers, args.stage)
    elif args.command == 'pretrain': result = str(train_pretraining(args.method, args.seed, args.stage))
    elif args.command == 'train': result = str(train_continuation(args.method, args.seed, args.decisions, args.stage))
    elif args.command == 'evaluate': result = evaluate(args.method, args.seed, args.decisions, args.stage)
    elif args.command == 'select': result = select(args.method, args.seed)
    else: result = chain(args.method, args.seed, args.stage)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
