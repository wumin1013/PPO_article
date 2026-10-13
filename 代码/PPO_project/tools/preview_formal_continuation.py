"""Five independent sensor pretrains followed by the frozen full-batch method.

Only training and complete three-path validation are exposed. Held-out geometry
is never loaded here. A separate evaluator may use the ten-model freeze after
all methods receive the full registered budget and identical selection chances.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time
import zipfile

import torch
import yaml

import preview_full_batch_study as full
from src.environment import cnc_env, kinematics

core, sensor = full.core, full.local.sensor
old, ROOT = core.old, core.old.ROOT
OUT = ROOT/'论文项目/revision_20260910_continued/formal_v1'
SEEDS, METHODS = (11, 22, 33, 44, 55), ('fixed3', 'adaptive')
MILESTONES = (0, 256, 512, 1024, 2048, 4096)
PRETRAIN_DECISIONS = ADDITIONAL_DECISIONS = 4096
HOLDOUT_MANIFEST = ROOT/'PPO_project/data/preview_continuation_v1/manifest.json'
BASE_PROTOCOL = ROOT/'论文项目/revision_20260910_continued/full_batch_v1/protocol.json'
BASE_SELECTION = BASE_PROTOCOL.parent/'selection_budget4096.json'
BASE_TRAIN, BASE_SENSOR_AGENT = core.train, core.make_agent
_installed_phase = None
EVALUATION_FILES = ('PPO_project/tools/preview_formal_holdout.py',
                    'PPO_project/tools/preview_formal_statistics.py')


def artifact(path):
    path = Path(path)
    return dict(file=path.relative_to(ROOT).as_posix(), sha256=old.sha(path))


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if old.read(path) != value:
            raise ValueError(f'Preserve registered artifact: {path}')
    else:
        old.write(path, value)
    return value


def source_state(phase):
    if phase not in ('pretraining', 'continuation'):
        raise ValueError('Unknown training phase')
    value = sensor.source_state() if phase == 'pretraining' else full.source_state()
    path = Path(__file__)
    value['source_files'][path.relative_to(ROOT).as_posix()] = old.sha(path)
    value['source_sha256'] = hashlib.sha256(json.dumps(value['source_files'], sort_keys=True).encode()).hexdigest()
    return value


def evaluation_sources():
    files = {name: old.sha(ROOT/name) for name in EVALUATION_FILES}
    return dict(source_files=files,
        source_sha256=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest())


def prepare():
    old.check_protocol()
    parent = old.read(BASE_PROTOCOL)
    if parent['source'] != full.source_state():
        raise ValueError('Frozen full-batch numerical source changed')
    selected = old.read(BASE_SELECTION)
    if selected['uniform_fixed']['method'] != 'fixed3':
        raise ValueError('Unified fixed3 must follow the retained short/medium/long development selection')
    value = dict(version='formal-preview-continuation-v1',
        seeds=list(SEEDS), methods=list(METHODS),
        pretraining_decisions=PRETRAIN_DECISIONS, additional_decisions=ADDITIONAL_DECISIONS,
        total_available_decisions_per_method_seed=PRETRAIN_DECISIONS+ADDITIONAL_DECISIONS,
        validation_milestones=list(MILESTONES), selection_milestones=list(MILESTONES),
        selection_rule='Maximize full success count over all three paths; compare total time only for 3/3; otherwise earliest checkpoint breaks ties',
        step0_policy='Eligible equally for both methods, following known development evidence; continuation budget is still fully expended',
        initialization='Each seed and method trains its own fresh sensor policy for 4096 decisions; use that actor/critic only, then fresh Adam states as in frozen full_batch',
        rng_policy='Seed Python/NumPy/Torch with each independent seed in each phase, as in the original two-phase development; checkpoint full RNG and optimizer states',
        sources={phase: source_state(phase) for phase in ('pretraining', 'continuation')},
        evaluation_sources=evaluation_sources(),
        development_protocol=artifact(BASE_PROTOCOL), fixed_grid_selection=artifact(BASE_SELECTION),
        training_validation_manifest=artifact(old.DATA/'manifest.json'),
        holdout_manifest=artifact(HOLDOUT_MANIFEST),
        training_path_ids=[p['id'] for p in old.manifest()['paths'] if p['split'] == 'train'],
        validation_path_ids=[p['id'] for p in old.manifest()['paths'] if p['split'] == 'validation'],
        terminal_brake='Original v2; full output audit and rest tail included',
        feed_projection='Original KCM; target-tracking correction and terminal v3 are not installed',
        pretraining_batch_rule='Frozen sensor training: flush at 64, done, or final requested budget',
        continuation_batch_rule='Frozen full_batch: 64 decisions across ends; keep terminal GAE boundaries',
        costs='Record pretraining, continuation, every validation, recovery, invocations, physical/command samples and exact optimizer updates separately',
        validation_role='Checkpoint selection; not independent held-out performance evidence',
        formal_seeds_previously_used='These integer labels also occur in historical algorithms; new chains retrain from scratch under new source/config, and new holdout paths remain untouched',
        test_access=False, machine_execution_authorized=False)
    immutable_json(OUT/'protocol.json', value)
    # The batch coordinator prepares these once before launching workers, so
    # separate seed processes only read shared protocol files while training.
    for phase in ('pretraining','continuation'):
        immutable_json(OUT/phase/'protocol.json',dict(source=value['sources'][phase],
            phase=phase,formal_protocol=artifact(OUT/'protocol.json')))
    return value


def run_dir(phase, method, seed, stage='formal'):
    return OUT/phase/stage/f'preview_time_{method}_seed{seed}'


def validate_request(method, seed, stage):
    if method not in METHODS or stage not in ('formal', 'smoke'):
        raise ValueError('Registered methods and stages only')
    if seed not in (SEEDS if stage == 'formal' else (901,)):
        raise ValueError('Formal independent seeds11/22/33/44/55; smoke seed901 only')


def configuration(phase, method, seed, stage='formal'):
    validate_request(method, seed, stage)
    if phase == 'pretraining':
        cfg = sensor.configuration('preview_time', method, seed)
    elif phase == 'continuation':
        # Reuse all frozen numerical settings. The old seed101 template is read
        # for settings only; no seed101 model is ever loaded by this runner.
        cfg = full.configuration('preview_time', method, 101)
        cfg['seed'] = seed
        prior = run_dir('pretraining', method, seed, stage)
        summary = old.read(prior/'training_summary.json')
        expected = PRETRAIN_DECISIONS if stage == 'formal' else 64
        if summary['seed'] != seed or summary['method'] != method or summary['decisions'] != expected:
            raise ValueError('Each method must use its own same-seed complete sensor pretrain')
        if summary['checkpoint_sha256'] != old.sha(prior/'checkpoint.pt'):
            raise ValueError('Independent pretraining checkpoint changed')
        cfg['revision'].update(init_checkpoint=(prior/'checkpoint.pt').relative_to(ROOT).as_posix(),
            init_checkpoint_sha256=old.sha(prior/'checkpoint.pt'),
            pretraining_macro_decisions=expected,
            initialization='Own same-seed/method sensor actor and critic; fresh Adam; no seed101 weights',
            zero_step_role='Eligible checkpoint under the same rule for fixed3 and adaptive')
    else:
        raise ValueError('Unknown training phase')
    cfg['revision'].update(formal_study='formal-preview-continuation-v1', formal_phase=phase,
        data_status='Independent retraining; all three validation paths select checkpoints; no holdout access')
    return cfg


@contextmanager
def phase_installed(phase, stage='formal'):
    """Scope old module globals; never call another study's install function."""
    global _installed_phase
    if _installed_phase is not None:
        raise RuntimeError('Do not nest formal phase installations')
    if cnc_env.apply_kinematic_constraints is not kinematics.apply_kinematic_constraints or core.plan_terminal_brake.__module__ != 'terminal_braking':
        raise RuntimeError('Start a fresh process without feed correction or alternate terminal brake')
    if BASE_SENSOR_AGENT.__module__ != 'preview_macro_study':
        raise RuntimeError('Import the formal runner in a clean process before other study installations')
    protocol = prepare()
    names = ('OUT', 'source_state', 'configuration', 'MacroEnvironment', 'make_agent', 'prepare')
    saved = {name: getattr(core, name) for name in names}
    _installed_phase = phase
    try:
        core.OUT = OUT/phase
        core.source_state = lambda: source_state(phase)
        core.configuration = lambda candidate, method, seed: configuration(phase, method, seed, stage)
        core.MacroEnvironment = sensor.SensorEnvironment if phase == 'pretraining' else full.local.LocalCornerEnvironment
        core.make_agent = BASE_SENSOR_AGENT if phase == 'pretraining' else full.local.make_agent
        core.prepare = lambda: dict(source=protocol['sources'][phase])
        # core.train stores this protocol SHA in its original provenance.
        immutable_json(core.OUT/'protocol.json', dict(source=protocol['sources'][phase],
            phase=phase, formal_protocol=artifact(OUT/'protocol.json')))
        yield core
    finally:
        for name, value in saved.items(): setattr(core, name, value)
        _installed_phase = None


@contextmanager
def invocation(phase, method, seed, stage):
    path = OUT/'invocations'/f'{time.time_ns()}_{phase}_{method}_{seed}.json'
    value = dict(phase=phase, method=method, seed=seed, stage=stage, argv=sys.argv,
                 started_utc=datetime.now(timezone.utc).isoformat(), status='running',
                 runner_sha256=old.sha(Path(__file__)),
                 protocol_sha256=old.sha(OUT/'protocol.json') if (OUT/'protocol.json').exists() else None,
                 timing_scope='Function call including source/config checks, replay, training/evaluation and file output; interpreter startup excluded')
    old.write(path, value); started = time.perf_counter()
    try:
        yield
    except BaseException as error:
        value.update(status='failed_or_interrupted', error=repr(error))
        raise
    else:
        value['status'] = 'complete'
    finally:
        value.update(wall_s=time.perf_counter()-started, finished_utc=datetime.now(timezone.utc).isoformat())
        old.write(path, value)


def _preserve(path):
    archive = path.parent/'recovery'/f'{old.sha(path)}_{path.name}'
    archive.parent.mkdir(exist_ok=True)
    if not archive.exists(): shutil.copyfile(path, archive)


def repair_sensor_logs(run):
    """Preserve any records written after the last committed old checkpoint."""
    if (run/'checkpoint.tmp').exists(): _preserve(run/'checkpoint.tmp')
    checkpoint = run/'checkpoint.pt'
    if not checkpoint.exists():
        for path in [run/'updates.jsonl', run/'episodes.jsonl']:
            if path.exists():
                _preserve(path); full._atomic_bytes(path, b'')
        for trace in run.glob('train_episode_*.npz'): _preserve(trace)
        return
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    for name in ('updates.jsonl', 'episodes.jsonl'):
        path = run/name
        if not path.exists(): continue
        content = path.read_bytes(); kept = []
        for line in content.splitlines(keepends=True):
            try: row = json.loads(line)
            except (ValueError, UnicodeDecodeError): break
            committed = (row['decisions'] <= saved['decisions'] if name == 'updates.jsonl' else row['episode'] < saved['episode'])
            if not committed: break
            kept.append(line)
        desired = b''.join(kept)
        if desired != content:
            _preserve(path); full._atomic_bytes(path, desired)
    for trace in run.glob('train_episode_*.npz'):
        if int(trace.stem.rsplit('_', 1)[1]) >= saved['episode']:
            _preserve(trace)


def _records(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []


def train_pretraining(method, seed, stage='formal'):
    validate_request(method, seed, stage)
    budget = PRETRAIN_DECISIONS if stage == 'formal' else 64
    run = run_dir('pretraining', method, seed, stage)
    with invocation('pretraining', method, seed, stage), phase_installed('pretraining', stage):
        if (run/'phase_complete.json').exists():
            verify_artifacts(old.read(run/'phase_complete.json')['artifacts'])
            return run
        repair_sensor_logs(run)
        # stage formal has no old pilot-seed guard, while the original loop,
        # terminal short batches, optimizer, and objective remain unchanged.
        run = BASE_TRAIN('preview_time', method, seed, budget, stage)
        summary = old.read(run/'training_summary.json'); updates = _records(run/'updates.jsonl')
        if summary['decisions'] != budget or sum(row['transition_count'] for row in updates) != budget:
            raise ValueError('Sensor update accounting must equal the registered complete budget')
        costs = dict(decisions=budget, optimizer_updates=len(updates),
            adam_steps_per_network=len(updates)*configuration('pretraining', method, seed, stage)['ppo']['epochs'],
            physics_steps=summary['physics_steps'], command_steps=summary['command_steps'],
            training_wall_s=summary['training_wall_s'], additional_pretraining=0)
        old.write(run/'costs.json', costs)
        files = [p for p in run.iterdir() if p.is_file() and p.name not in ('checkpoint.tmp', 'phase_complete.json')]
        immutable_json(run/'phase_complete.json', dict(phase='pretraining',seed=seed,method=method,
            artifacts={p.name: artifact(p) for p in files}, costs=costs))
    return run


def verify_artifacts(artifacts):
    for item in artifacts.values():
        if old.sha(ROOT/item['file']) != item['sha256']:
            raise ValueError(f'Artifact changed: {item["file"]}')


def verify_snapshot(snapshot):
    manifest = old.read(snapshot/'snapshot_manifest.json')
    for name, digest in manifest['files'].items():
        if old.sha(snapshot/name) != digest:
            raise ValueError(f'Immutable checkpoint artifact changed: {name}')


def _ensure_continuation_provenance(run, cfg, protocol):
    run.mkdir(parents=True, exist_ok=True)
    config_path = run/'config.yaml'
    if config_path.exists():
        if yaml.safe_load(config_path.read_text(encoding='utf-8')) != cfg:
            raise ValueError('Preserve continuation configuration')
        provenance = old.read(run/'provenance.json')
        if provenance['source_sha256'] != protocol['sources']['continuation']['source_sha256'] or provenance['config_sha256'] != old.sha(config_path):
            raise ValueError('Preserve continuation provenance')
        return
    config_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding='utf-8')
    old.write(run/'provenance.json', dict(**protocol['sources']['continuation'],
        config_sha256=old.sha(config_path), protocol_sha256=old.sha(OUT/'protocol.json'),
        command='python '+' '.join(sys.argv), phase='continuation',
        own_pretraining_checkpoint=artifact(ROOT/cfg['revision']['init_checkpoint'])))
    with zipfile.ZipFile(run/'source.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for name in protocol['sources']['continuation']['source_files']: archive.write(ROOT/name, name)


def train_continuation(method, seed, decisions, stage='formal'):
    validate_request(method, seed, stage)
    if decisions not in (MILESTONES if stage == 'formal' else (0, 64)):
        raise ValueError('Registered checkpoint milestones only')
    protocol = prepare(); cfg = configuration('continuation', method, seed, stage)
    run = run_dir('continuation', method, seed, stage)
    snapshot = run/'checkpoints'/f'step{decisions}'
    if snapshot.exists():
        verify_snapshot(snapshot)
        if yaml.safe_load((snapshot/'config.yaml').read_text(encoding='utf-8')) != cfg:
            raise ValueError('Cached snapshot config differs from the registered own-seed chain')
        return snapshot
    with invocation('continuation', method, seed, stage), phase_installed('continuation', stage):
        torch.set_num_threads(1)
        _ensure_continuation_provenance(run, cfg, protocol)
        if (run/'checkpoint.tmp').exists(): _preserve(run/'checkpoint.tmp')
        trainer = full.FullBatchTrainer(cfg, run, env_factory=full.local.LocalCornerEnvironment,
            agent_factory=full.local.make_agent, source_hash=protocol['sources']['continuation']['source_sha256'])
        state = trainer.advance(decisions)
        old.save_trace(run/'training_final_partial.npz', trainer.env.trace)
        summary = trainer.summary(state)
        summary.update(pretraining_macro_decisions=cfg['revision']['pretraining_macro_decisions'],
            zero_step_initial_diagnostic=decisions == 0, zero_step_eligible_for_selection=True,
            available_total_decisions=cfg['revision']['pretraining_macro_decisions']+decisions,
            status='registered_independent_formal_milestone_complete' if stage == 'formal' else 'smoke_only')
        old.write(run/'training_summary.json', summary)
        names = ['checkpoint.pt', 'config.yaml', 'provenance.json', 'source.zip', 'training_summary.json',
                 'updates.jsonl', 'episodes.jsonl', 'training_final_partial.npz']
        names += [item['file'] for item in trainer.trace_files]
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        pending = snapshot.parent/f'.step{decisions}.pending-{time.time_ns()}'
        pending.mkdir()
        for name in names: shutil.copyfile(run/name, pending/name)
        old.write(pending/'snapshot_manifest.json', dict(decisions=decisions,
            files={name: old.sha(pending/name) for name in names},
            recovery='FullBatchTrainer: actor/critic/Adam/RNG/pending64-buffer/current-episode actions and complete log records'))
        pending.replace(snapshot)
    return snapshot


def evaluate(method, seed, decisions, stage='formal'):
    validate_request(method, seed, stage)
    if decisions not in (MILESTONES if stage == 'formal' else (0,64)):
        raise ValueError('Registered validation milestones only')
    snapshot = run_dir('continuation', method, seed, stage)/'checkpoints'/f'step{decisions}'
    verify_snapshot(snapshot)
    with invocation('validation', method, seed, stage), phase_installed('continuation', stage):
        summary = snapshot/'validation/summary.json'
        if summary.exists():
            rows = old.read(summary)['rows']
        else:
            rows = core.evaluate(snapshot)
        protocol = prepare()
        if len(rows) != 3 or {row['path_id'] for row in rows} != set(protocol['validation_path_ids']):
            raise ValueError('Keep the complete registered validation set')
        binding = dict(checkpoint_sha256=old.sha(snapshot/'checkpoint.pt'),
            config_sha256=old.sha(snapshot/'config.yaml'),
            source_sha256=protocol['sources']['continuation']['source_sha256'])
        for row in rows:
            if any(row.get(key) != value for key,value in binding.items()) or row['trace_sha256'] != old.sha(summary.parent/f'{row["path_id"]}.npz'):
                raise ValueError('Cached validation source/model/config/trace changed')
        files = {p.name:old.sha(p) for p in summary.parent.iterdir() if p.is_file() and p.name != 'delivery_manifest.json'}
        immutable_json(summary.parent/'delivery_manifest.json',dict(files=files,**binding))
        return rows


def selection_key(row):
    return (-row['successes'], row['complete_validation_time_s'] if row['successes'] == 3 else math.inf, row['step'])


def select(method, seed):
    validate_request(method, seed, 'formal'); protocol = prepare()
    run = run_dir('continuation', method, seed)
    if old.read(run/'training_summary.json')['decisions'] != ADDITIONAL_DECISIONS:
        raise ValueError('Expend the full common continuation budget before selecting, including step0')
    records = []
    for step in MILESTONES:
        snapshot = run/'checkpoints'/f'step{step}'; verify_snapshot(snapshot)
        rows = old.read(snapshot/'validation/summary.json')['rows']
        delivery = old.read(snapshot/'validation/delivery_manifest.json')
        for name,digest in delivery['files'].items():
            if old.sha(snapshot/'validation'/name) != digest:
                raise ValueError('Validation artifact changed after evaluation')
        if len(rows) != 3 or {r['path_id'] for r in rows} != set(protocol['validation_path_ids']):
            raise ValueError('Every checkpoint requires all three validation paths')
        model_hash = old.sha(snapshot/'checkpoint.pt')
        if any(r['checkpoint_sha256'] != model_hash for r in rows): raise ValueError('Model/validation mismatch')
        successes = sum(r['success'] for r in rows)
        records.append(dict(step=step,successes=successes,count=3,
            complete_validation_time_s=sum(r['elapsed_time_s'] for r in rows) if successes == 3 else None,
            checkpoint_dir=snapshot.relative_to(ROOT).as_posix(), checkpoint_sha256=model_hash,
            validation=artifact(snapshot/'validation/summary.json'),
            validation_sha256=old.sha(snapshot/'validation/summary.json'),
            training_summary_sha256=old.sha(snapshot/'training_summary.json'),
            snapshot_manifest_sha256=old.sha(snapshot/'snapshot_manifest.json'),
            validation_evaluation_wall_s=sum(r['evaluation_wall_s'] for r in rows)))
    result = dict(seed=seed,method=method,selected=min(records,key=selection_key),all_checkpoints=records,
        available_pretraining_decisions=PRETRAIN_DECISIONS,available_additional_decisions=ADDITIONAL_DECISIONS,
        rule=protocol['selection_rule'],step0_eligible=True,
        costs=dict(pretraining=old.read(run_dir('pretraining',method,seed)/'costs.json'),
            continuation=old.read(run/'training_summary.json'),
            validation_evaluation_wall_s=sum(r['validation_evaluation_wall_s'] for r in records)))
    return immutable_json(OUT/'selections'/f'{method}_seed{seed}.json', result)


def freeze():
    protocol = prepare(); records = []
    names = ('checkpoint.pt','config.yaml','provenance.json','source.zip','snapshot_manifest.json',
             'training_summary.json','validation/summary.json','validation/delivery_manifest.json')
    for seed in SEEDS:
        for method in METHODS:
            selection = select(method, seed); chosen = selection['selected']; snapshot = ROOT/chosen['checkpoint_dir']
            prior = run_dir('pretraining', method, seed)
            verify_artifacts(old.read(prior/'phase_complete.json')['artifacts'])
            records.append(dict(seed=seed,method=method,selected_step=chosen['step'],
                available_pretraining_decisions=PRETRAIN_DECISIONS,available_additional_decisions=ADDITIONAL_DECISIONS,
                checkpoint_dir=chosen['checkpoint_dir'], artifacts={name:artifact(snapshot/name) for name in names},
                pretraining=dict(checkpoint_dir=prior.relative_to(ROOT).as_posix(),decisions=PRETRAIN_DECISIONS,
                    artifacts={name:artifact(prior/name) for name in ('checkpoint.pt','config.yaml','provenance.json','source.zip','training_summary.json','phase_complete.json','costs.json')}),
                selection=artifact(OUT/'selections'/f'{method}_seed{seed}.json'),
                additional_budget_completion=artifact(run_dir('continuation',method,seed)/'checkpoints/step4096/training_summary.json'),
                additional_budget_snapshot_manifest=artifact(run_dir('continuation',method,seed)/'checkpoints/step4096/snapshot_manifest.json'),
                source_sha256=protocol['sources']['continuation']['source_sha256']))
    result = dict(version='formal-preview-continuation-v1',protocol=artifact(OUT/'protocol.json'),
        holdout_manifest=protocol['holdout_manifest'],seeds=list(SEEDS),methods=list(METHODS),
        source=protocol['sources']['continuation'],records=records,all_training_complete=True,
        selection_rule=protocol['selection_rule'],test_accessed=False,
        training_budget_total_decisions=len(SEEDS)*len(METHODS)*(PRETRAIN_DECISIONS+ADDITIONAL_DECISIONS))
    return immutable_json(OUT/'freeze.json', result)


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=OUT)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare'); sub.add_parser('freeze')
    for name in ('pretrain','train','evaluate','select','chain'):
        p = sub.add_parser(name)
        p.add_argument('--method',choices=METHODS,required=True); p.add_argument('--seed',type=int,required=True)
        if name != 'select': p.add_argument('--stage',choices=('formal','smoke'),default='formal')
        if name in ('train','evaluate'): p.add_argument('--decisions',type=int,required=True)
    args = parser.parse_args(); OUT = args.output_root.resolve()
    OUT.relative_to(ROOT/'论文项目/revision_20260910_continued')
    if args.command == 'prepare': result = prepare()
    elif args.command == 'freeze': result = freeze()
    elif args.command == 'pretrain': result = str(train_pretraining(args.method,args.seed,args.stage))
    elif args.command == 'train': result = str(train_continuation(args.method,args.seed,args.decisions,args.stage))
    elif args.command == 'evaluate': result = evaluate(args.method,args.seed,args.decisions,args.stage)
    elif args.command == 'select': result = select(args.method,args.seed)
    else:
        train_pretraining(args.method,args.seed,args.stage)
        for step in (MILESTONES if args.stage == 'formal' else (0,64)):
            train_continuation(args.method,args.seed,step,args.stage)
            evaluate(args.method,args.seed,step,args.stage)
        result = select(args.method,args.seed) if args.stage == 'formal' else {'status':'smoke_complete'}
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__': main()
