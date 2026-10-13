"""New ten-model target-tracking gate; geometry is opened only after freezing.

The old six-path gate remains unchanged. All policy rollouts use their actual
closed-loop state, the common target wrapper and the original v2 terminal audit.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

import numpy as np
import yaml

import preview_formal_holdout as old

ROOT = Path(__file__).resolve().parents[2]
FORMAL = ROOT/'论文项目/revision_20260910_continued/target_tracking_formal_v1'
VERSION = 'target-tracking-formal-v1'
SEEDS = (111, 222, 333, 444, 555)
METHODS = ('fixed3', 'adaptive')
STEPS = (0, 256, 512, 1024, 2048, 4096)
FAMILIES = ('mixed_lengths', 'dense_corners', 'line_curve')
HOLDOUT_FILE = 'PPO_project/data/preview_frontier_holdout_v1/manifest.json'
HOLDOUT_SHA256 = '21673efe97966cb45e117564df64423014ca4c30b90cb54fc0854bac13e4a927'
EVALUATION_FILES = {
    'PPO_project/tools/preview_target_tracking_holdout.py',
    'PPO_project/tests/test_preview_target_tracking_holdout.py',
    'PPO_project/tools/preview_target_tracking_statistics.py',
    'PPO_project/tests/test_preview_target_tracking_statistics.py',
    'PPO_project/tools/preview_target_tracking_two_step.py',
    'PPO_project/tests/test_preview_target_tracking_two_step.py',
    'PPO_project/tools/preview_formal_two_step.py',
    'PPO_project/tools/preview_two_step_eval.py',
}
CORE_ARTIFACTS = old.CORE_ARTIFACTS
FreezeError, require, sha = old.FreezeError, old.require, old.sha
read_json, relative_path, bound_file = old.read_json, old.relative_path, old.bound_file
bound_artifacts, verify_source, verify_snapshot = old.bound_artifacts, old.verify_source, old.verify_snapshot
write_json_new, write_csv_new, terminal_tail = old.write_json_new, old.write_csv_new, old.terminal_tail


def registered_configuration(phase, method, seed):
    # Configuration construction reads the corresponding training prior only;
    # the formal runner never reads the new holdout manifest or coordinates.
    import preview_target_tracking_formal as formal
    return formal.configuration(phase, method, seed)


def registered_source(phase):
    import preview_target_tracking_formal as formal
    return formal.source_state(phase)


def common_config(cfg, seed, method):
    require(cfg == registered_configuration('continuation', method, seed),
            'Continuation differs from its exact registered target configuration')
    revision = cfg['revision']
    require(cfg['seed'] == seed and bool(revision['adaptive_preview']) == (method == 'adaptive'),
            'Config seed/method mismatch')
    require(cfg['environment']['interpolation_period'] == .001 and revision['macro_period_steps'] == 20,
            'Keep the1ms/20ms sampling model')
    require(cfg['environment']['epsilon'] == 2.5, 'Keep2.5mm full band/1.25mm one-sided tolerance')
    require(revision['fixed_distance_mm'] == revision['initial_preview_mm'] == 3.,
            'Use common initial preview and one pooled fixed3mm distance')
    require(revision['active_action_dims'] == ([1, 2] if method == 'adaptive' else [1]),
            'Unexpected active actions')
    tracking = revision['feed_tracking']
    require(tracking['implementation'] == 'preview_target_tracking.project_feed_target'
            and tracking['sampling_period_s'] == .001 and tracking['not_a_learned_preview_rule'] is True
            and tracking['terminal_brake'] == 'Frozen v2 plan_terminal_brake, unchanged',
            'Unexpected target tracking or terminal implementation')
    limits = cfg['kinematic_constraints']
    require(all(key in limits and np.isfinite(limits[key]) and limits[key] > 0 for key in
                ('MAX_VEL', 'MAX_ACC', 'MAX_JERK', 'MAX_ANG_VEL', 'MAX_ANG_ACC', 'MAX_ANG_JERK')),
            'All six kinematic constraints are required')
    return dict(environment=cfg['environment'], constraints=limits,
                macro_period_steps=20, terminal_definition=revision['terminal_definition'], feed_tracking=tracking)


def verify_delivery(directory):
    manifest = read_json(directory/'delivery_manifest.json')
    require({'summary.json', 'binding.json'} <= set(manifest['files']), 'Incomplete validation delivery')
    for name, digest in manifest['files'].items():
        path = (directory/name).resolve()
        require(path.is_relative_to(directory.resolve()) and path.is_file() and sha(path) == digest,
                f'Validation evidence changed: {path}')


def verify_node(root, study_dir, snapshot, step, seed, method, source, cfg, validation_ids):
    require(snapshot.is_relative_to(study_dir), 'Candidate outside its formal study')
    paths = {name: snapshot/name for name in CORE_ARTIFACTS | {'snapshot_manifest.json'}}
    verify_source(root, paths, source)
    verify_snapshot(snapshot, paths, step)
    require(yaml.safe_load(paths['config.yaml'].read_text(encoding='utf-8')) == cfg,
            'All candidate nodes require the same registered configuration')
    summary = read_json(paths['training_summary.json'])
    model_sha = sha(paths['checkpoint.pt'])
    require(summary['seed'] == seed and summary['method'] == method and summary['decisions'] == step,
            'Candidate training identity changed')
    require(summary['optimized_decisions'] == step and summary['pending_buffer_decisions'] == 0
            and summary['checkpoint_sha256'] == model_sha, 'Incomplete optimized candidate budget')
    updates = [json.loads(line) for line in (snapshot/'updates.jsonl').read_text(encoding='utf-8').splitlines()]
    require(len(updates) == step//64 and all(r['transition_count'] == 64 for r in updates)
            and sum(r['transition_count'] for r in updates) == step, 'Candidate does not contain full64 updates')
    directory = snapshot/'validation'
    verify_delivery(directory)
    rows = read_json(directory/'summary.json')['rows']
    require(len(rows) == 3 and {r['path_id'] for r in rows} == set(validation_ids), 'All three validation paths required')
    binding = dict(checkpoint_sha256=model_sha, config_sha256=sha(paths['config.yaml']), source_sha256=source['source_sha256'])
    require(all(read_json(directory/'binding.json').get(k) == v for k, v in binding.items()), 'Validation binding differs')
    for row in rows:
        require(row['seed'] == seed and row['method'] == method and row['training_decisions'] == step
                and all(row[k] == v for k, v in binding.items()), 'Validation/model/config/source identity mismatch')
        require(read_json(directory/(row['path_id']+'.json')) == row
                and sha(directory/(row['path_id']+'.npz')) == row['trace_sha256'], 'Validation raw trace binding changed')
    return summary, rows


def validate_freeze(freeze_path, root=ROOT):
    """Check all ten chains and all sixty candidates before manifest access."""
    root, freeze_path = Path(root).resolve(), Path(freeze_path).resolve()
    require(freeze_path.is_file() and freeze_path.is_relative_to(root), 'Freeze all ten new models before test access')
    frozen_sha = sha(freeze_path)
    freeze = read_json(freeze_path)
    require(freeze['version'] == VERSION, 'Unknown target-tracking freeze schema')
    require(freeze['seeds'] == list(SEEDS) and freeze['methods'] == list(METHODS), 'Use the five new seeds and both methods')
    records = freeze['records']
    require(len(records) == 10 and {(r['seed'], r['method']) for r in records} == {(s, m) for s in SEEDS for m in METHODS},
            'Missing/duplicate new formal model')
    protocol_path = bound_file(root, freeze['protocol'])
    protocol = read_json(protocol_path)
    require(protocol['version'] == VERSION and protocol['seeds'] == list(SEEDS)
            and protocol['methods'] == list(METHODS), 'Formal protocol identity mismatch')
    require(protocol['sources']['continuation'] == freeze['source'], 'Continuation source mismatch')
    require(all(protocol['sources'][phase] == registered_source(phase) for phase in ('pretraining', 'continuation')),
            'Incomplete or changed registered training source graph')
    require('PPO_project/src/algorithms/two_step_baseline.py' in freeze['source']['source_files'],
            'Freeze the actual traditional baseline implementation too')
    require(freeze['all_training_complete'] is True and freeze['training_budget_total_decisions'] == 81920,
            'All ten actual4096+4096 training chains are required')
    require(protocol['validation_path_ids'] == [f'{family}_validation' for family in FAMILIES],
            'Registered three development validation paths required')
    evaluation = protocol['evaluation_sources']
    require(EVALUATION_FILES <= set(evaluation['source_files']), 'Freeze evaluator, statistics and their tests')
    require(hashlib.sha256(json.dumps(evaluation['source_files'], sort_keys=True).encode()).hexdigest() == evaluation['source_sha256'],
            'Evaluator source digest mismatch')
    for name, digest in evaluation['source_files'].items():
        require(sha(relative_path(root, name)) == digest, f'Frozen evaluation source changed: {name}')
    descriptor = dict(file=HOLDOUT_FILE, sha256=HOLDOUT_SHA256)
    require(freeze['holdout_manifest'] == protocol['holdout_manifest'] == descriptor,
            'Only the pre-reserved twelve-path manifest is allowed')
    # Descriptor-only check: do not hash/open the holdout until every model passes.
    study_dir = freeze_path.parent
    models, prior_hashes, signature = [], set(), None
    for record in records:
        seed, method, step = record['seed'], record['method'], record['selected_step']
        require(step in STEPS, 'Unknown selected node')
        require(record['available_pretraining_decisions'] == record['available_additional_decisions'] == 4096,
                'Unequal available training budget')
        directory, paths = bound_artifacts(root, study_dir, record, CORE_ARTIFACTS | {'snapshot_manifest.json', 'validation/summary.json', 'validation/delivery_manifest.json'})
        require(record['source_sha256'] == freeze['source']['source_sha256'], 'Record source mismatch')
        cfg = yaml.safe_load(paths['config.yaml'].read_text(encoding='utf-8'))
        current = common_config(cfg, seed, method)
        require(signature is None or current == signature, 'Unequal common constraints/target execution')
        signature = current
        prior = record['pretraining']
        prior_dir, prior_paths = bound_artifacts(root, study_dir, prior, CORE_ARTIFACTS | {'phase_complete.json', 'costs.json'})
        verify_source(root, prior_paths, protocol['sources']['pretraining'])
        prior_cfg = yaml.safe_load(prior_paths['config.yaml'].read_text(encoding='utf-8'))
        require(prior_cfg == registered_configuration('pretraining', method, seed), 'Pretraining configuration changed')
        prior_summary = read_json(prior_paths['training_summary.json'])
        require(prior['decisions'] == prior_summary['decisions'] == 4096
                and prior_summary['seed'] == prior_cfg['seed'] == seed and prior_summary['method'] == method,
                'Incomplete or shared pretraining identity')
        require(prior_summary.get('pretraining_steps', 0) == 0 and 'init_checkpoint' not in prior_cfg['revision']
                and 'feed_tracking' not in prior_cfg['revision'], 'Sensor pretraining must be new and use original execution')
        prior_sha = sha(prior_paths['checkpoint.pt'])
        require(prior_summary['checkpoint_sha256'] == prior_sha and prior_sha not in prior_hashes,
                'Pretraining checkpoint reused across chains')
        prior_hashes.add(prior_sha)
        phase = read_json(prior_paths['phase_complete.json'])
        require(phase['phase'] == 'pretraining' and phase['seed'] == seed and phase['method'] == method,
                'Pretraining completion identity mismatch')
        require({'checkpoint.pt', 'config.yaml', 'training_summary.json', 'updates.jsonl', 'episodes.jsonl', 'costs.json'} <= set(phase['artifacts']),
                'Pretraining completion must bind raw update/episode logs')
        for name, item in phase['artifacts'].items():
            require(bound_file(root, item) == (prior_dir/name).resolve(), 'Pretraining artifact belongs to another chain')
        updates = [json.loads(line) for line in (prior_dir/'updates.jsonl').read_text(encoding='utf-8').splitlines()]
        costs = read_json(prior_paths['costs.json'])
        require(sum(r['transition_count'] for r in updates) == costs['decisions'] == 4096
                and len(updates) == costs['optimizer_updates'], 'Actual sensor optimization budget incomplete')
        require(relative_path(root, cfg['revision']['init_checkpoint']) == prior_paths['checkpoint.pt']
                and cfg['revision']['init_checkpoint_sha256'] == prior_sha, 'Continuation does not use its own new sensor prior')
        selection = read_json(bound_file(root, record['selection']))
        require(selection['seed'] == seed and selection['method'] == method, 'Selection seed/method mismatch')
        candidates = selection['all_checkpoints']
        require(len(candidates) == 6 and {c['step'] for c in candidates} == set(STEPS), 'All six common selection opportunities required')
        for candidate in candidates:
            snapshot = relative_path(root, candidate['checkpoint_dir'])
            require(snapshot.parent == directory.parent and snapshot.name == f"step{candidate['step']}", 'Candidate belongs to another chain/node')
            for name, key in (('checkpoint.pt', 'checkpoint_sha256'), ('training_summary.json', 'training_summary_sha256'),
                              ('snapshot_manifest.json', 'snapshot_manifest_sha256'), ('validation/summary.json', 'validation_sha256')):
                require(sha(snapshot/name) == candidate[key], 'Candidate artifact binding changed')
            _, rows = verify_node(root, study_dir, snapshot, candidate['step'], seed, method, freeze['source'], cfg, protocol['validation_path_ids'])
            successes = sum(bool(row['success']) for row in rows)
            elapsed = sum(row['elapsed_time_s'] for row in rows) if successes == 3 else None
            require(candidate['successes'] == successes and candidate['count'] == 3
                    and candidate['complete_validation_time_s'] == elapsed, 'Selection statistics changed')
        selected = min(candidates, key=lambda c: (-c['successes'], c['complete_validation_time_s'] if c['successes'] == 3 else float('inf'), c['step']))
        require(selection['step0_eligible'] is True and selection['selected'] == selected, 'Selection violates the frozen common rule')
        require(selected['step'] == step and selected['checkpoint_sha256'] == sha(paths['checkpoint.pt']), 'Frozen selected model changed')
        completion = bound_file(root, record['additional_budget_completion'])
        final_manifest = bound_file(root, record['additional_budget_snapshot_manifest'])
        require(completion == directory.parent/'step4096/training_summary.json'
                and final_manifest == directory.parent/'step4096/snapshot_manifest.json', 'Final budget belongs to another chain')
        models.append(dict(record=record, cfg=cfg, checkpoint=paths['checkpoint.pt'], directory=directory))
    require(sha(freeze_path) == frozen_sha, 'Freeze changed during validation')
    return dict(freeze=freeze, freeze_path=freeze_path, freeze_sha256=frozen_sha, models=models, common_configuration=signature)


def frozen_manifest(gate, root=ROOT):
    path = bound_file(Path(root), gate['freeze']['holdout_manifest'])
    manifest = read_json(path)
    items = manifest['paths']
    require(manifest['schema_version'] == 'preview-frontier-holdout-v1', 'Wrong reserved manifest schema')
    require(len(items) == 12 and len({p['id'] for p in items}) == 12
            and all(p['split'] == 'test' for p in items), 'Twelve distinct heldout paths required')
    require({family: sum(p['family'] == family for p in items) for family in FAMILIES} == {family: 4 for family in FAMILIES},
            'Require four paths in each of three families')
    require(manifest['tolerance_definition'] == dict(planning_bandwidth_mm=2.5, one_sided_tolerance_mm=1.25,
            source_discretization_bound_mm=.005, source_bound_added_to_planning_tolerance=False), 'Tolerance/source-bound metadata changed')
    return manifest, path.parent


def load_frozen_references(freeze_path, root=ROOT, loader=None):
    gate = validate_freeze(freeze_path, root)
    manifest, data = frozen_manifest(gate, root)
    if loader is None:
        from generate_preview_continuation_paths import load_path
        loader = load_path
    return gate, manifest, {item['id']: loader(item['id'], data) for item in manifest['paths']}


def freeze():
    import preview_target_tracking_formal as formal
    protocol = formal.prepare()
    records = []
    names = CORE_ARTIFACTS | {'snapshot_manifest.json', 'validation/summary.json', 'validation/delivery_manifest.json'}
    for seed in SEEDS:
        for method in METHODS:
            selection = formal.select(method, seed)
            chosen = selection['selected']; snapshot = ROOT/chosen['checkpoint_dir']
            prior = formal.run_dir('pretraining', method, seed)
            final = formal.run_dir('continuation', method, seed)/'checkpoints/step4096'
            records.append(dict(seed=seed, method=method, selected_step=chosen['step'],
                available_pretraining_decisions=4096, available_additional_decisions=4096,
                checkpoint_dir=chosen['checkpoint_dir'], artifacts={name: formal.artifact(snapshot/name) for name in names},
                pretraining=dict(checkpoint_dir=prior.relative_to(ROOT).as_posix(), decisions=4096,
                    artifacts={name: formal.artifact(prior/name) for name in CORE_ARTIFACTS | {'phase_complete.json', 'costs.json'}}),
                selection=formal.artifact(formal.OUT/'selections'/f'{method}_seed{seed}.json'),
                additional_budget_completion=formal.artifact(final/'training_summary.json'),
                additional_budget_snapshot_manifest=formal.artifact(final/'snapshot_manifest.json'),
                source_sha256=protocol['sources']['continuation']['source_sha256']))
    value = dict(version=VERSION, protocol=formal.artifact(formal.OUT/'protocol.json'),
        holdout_manifest=protocol['holdout_manifest'], seeds=list(SEEDS), methods=list(METHODS),
        source=protocol['sources']['continuation'], records=records, all_training_complete=True,
        training_budget_total_decisions=81920, test_accessed=False, adopted=False)
    path = formal.OUT/'freeze.json'
    if path.exists():
        require(read_json(path) == value, 'Preserve previously published freeze')
        validate_freeze(path)
        return path
    pending = path.with_name(f'.freeze.pending-{uuid.uuid4().hex}.json')
    write_json_new(pending, value)
    validate_freeze(pending)  # Still descriptor-only; no test manifest is read.
    require(not path.exists(), 'Another process already published a freeze')
    pending.rename(path)
    return path


def model_identity(gate, chosen):
    record = chosen['record']
    return dict(freeze_sha256=gate['freeze_sha256'], seed=record['seed'], method=record['method'],
        checkpoint_sha256=sha(chosen['checkpoint']), config_sha256=record['artifacts']['config.yaml']['sha256'],
        source_sha256=gate['freeze']['source']['source_sha256'],
        holdout_manifest_sha256=gate['freeze']['holdout_manifest']['sha256'])


def evaluate(freeze_path, seed, method):
    gate = validate_freeze(freeze_path)
    require(seed in SEEDS and method in METHODS, 'Choose a frozen new method/seed')
    chosen = next(m for m in gate['models'] if m['record']['seed'] == seed and m['record']['method'] == method)
    output = gate['freeze_path'].parent/'holdout'/f'{method}_seed{seed}'
    identity = model_identity(gate, chosen)
    if (output/'binding.json').exists():
        require(read_json(output/'binding.json') == identity, 'Cannot resume a different freeze/model')
    else:
        write_json_new(output/'binding.json', identity)
    invocation_id = uuid.uuid4().hex
    invocation = gate['freeze_path'].parent/'holdout_invocations'/invocation_id
    started = time.perf_counter()
    record = dict(**identity, pid=os.getpid(), parent_pid=os.getppid(), cwd=str(Path.cwd()),
        argv=[sys.executable, *(['-B'] if sys.dont_write_bytecode else []), *sys.argv],
        started_utc=datetime.now(timezone.utc).isoformat(), actual_interpreter_flag_B=bool(sys.dont_write_bytecode),
        exposure='First/further frozen holdout access; test-informed changes invalidate independent-test status',
        status='running', machine_execution=False)
    write_json_new(invocation/'started.json', record)
    rows = []
    try:
        # These imports do not load references. The execution context explicitly
        # installs only the already-frozen scalar wrapper, retaining v2 braking.
        import torch
        import preview_full_batch_study as full
        import preview_restart_closed_loop as audit
        from generate_preview_continuation_paths import load_path
        torch.set_num_threads(1)
        manifest, data = frozen_manifest(gate)
        payload = torch.load(chosen['checkpoint'], map_location='cpu', weights_only=False)
        require(payload['config'] == chosen['cfg'] and payload['source_sha256'] == identity['source_sha256'],
                'Evaluation payload config/source mismatch')
        with audit.corrected_execution():
            for item in manifest['paths']:
                path_id = item['id']; result_path = output/f'{path_id}.json'
                if result_path.exists():
                    row = read_json(result_path)
                    require(all(row.get(k) == v for k, v in identity.items()) and row['path_id'] == path_id
                            and row['family'] == item['family'] and row['reference_sha256'] == item['reference_sha256'],
                            'Saved result binding differs')
                    bound_file(ROOT, row['trace']); bound_file(ROOT, row['inference_samples'])
                    rows.append(row)
                    continue
                validate_freeze(freeze_path)
                reference = load_path(path_id, data)
                env = full.local.LocalCornerEnvironment(copy.deepcopy(chosen['cfg']), path_id, reference=reference)
                agent = full.local.make_agent(chosen['cfg'], env)
                agent.actor.load_state_dict(payload['actor']); agent.actor.eval()
                timings = []
                path_started = time.perf_counter()
                attempt = output/'attempts'/invocation_id
                attempt.mkdir(parents=True, exist_ok=True)
                trace_path = attempt/f'{path_id}.npz'
                try:
                    while not env.done:
                        state = env.state
                        inference_start = time.perf_counter()
                        action = agent.take_action(state, deterministic=True)
                        timings.append(time.perf_counter()-inference_start)
                        env.step(action)
                except BaseException as error:
                    full.core.old.save_trace(trace_path, env.trace)
                    write_json_new(attempt/f'{path_id}_interrupted.json', dict(**identity, path_id=path_id,
                        error=repr(error), decisions=len(timings), inference_seconds=timings,
                        trace=dict(file=trace_path.relative_to(ROOT).as_posix(), sha256=sha(trace_path)),
                        completed_outcome=False))
                    raise
                trajectory_wall_s = time.perf_counter()-path_started
                full.core.old.save_trace(trace_path, env.trace)
                samples_path = attempt/f'{path_id}_inference.json'
                write_json_new(samples_path, dict(seconds=timings, unit='s', cpu_threads=1,
                    measured_call='deterministic actor.take_action; no warmup; excludes state retrieval'))
                row = dict(env.result, **identity, path_id=path_id, family=item['family'],
                    selected_step=chosen['record']['selected_step'], terminal_tail_s=terminal_tail(env.trace),
                    policy_decisions=len(timings), inference_wall_s=float(sum(timings)),
                    inference_mean_s=float(np.mean(timings)) if timings else 0.,
                    inference_p50_s=float(np.quantile(timings, .5)) if timings else 0.,
                    inference_p95_s=float(np.quantile(timings, .95)) if timings else 0.,
                    cpu_threads=1, trajectory_wall_s=trajectory_wall_s, evaluation_wall_s=time.perf_counter()-path_started,
                    trace=dict(file=trace_path.relative_to(ROOT).as_posix(), sha256=sha(trace_path)),
                    inference_samples=dict(file=samples_path.relative_to(ROOT).as_posix(), sha256=sha(samples_path)),
                    reference_sha256=item['reference_sha256'], execution='Frozen target-tracking feed projection and original v2 brake',
                    data_role='New frozen independent synthetic holdout; all failures retained; no path-specific configuration or selection',
                    timing_scope='Physical elapsed includes terminal brake and20ms rest; measured CPU times are separate')
                write_json_new(result_path, row); rows.append(row)
                print(json.dumps(dict(seed=seed, method=method, path=path_id, success=row['success'],
                                      elapsed_time_s=row['elapsed_time_s'])), flush=True)
        if (output/'summary.json').exists():
            require(read_json(output/'summary.json') == dict(**identity, rows=rows), 'Existing summary differs')
        else:
            write_json_new(output/'summary.json', dict(**identity, rows=rows))
        record.update(status='complete', exit_code=0, outcomes=len(rows))
    except BaseException as error:
        record.update(status='failed_or_interrupted', exit_code=1, error=repr(error), outcomes=len(rows))
        raise
    finally:
        record.update(wall_s=time.perf_counter()-started, finished_utc=datetime.now(timezone.utc).isoformat())
        write_json_new(invocation/'completed.json', record)
    return rows


def collect_results(freeze_path):
    """Strict120-row collection for the independently frozen statistics tool."""
    gate = validate_freeze(freeze_path)
    manifest, _ = frozen_manifest(gate)
    rows, inputs = [], {gate['freeze_path'].relative_to(ROOT).as_posix(): gate['freeze_sha256']}
    for chosen in gate['models']:
        record = chosen['record']; identity = model_identity(gate, chosen)
        directory = gate['freeze_path'].parent/'holdout'/f"{record['method']}_seed{record['seed']}"
        for item in manifest['paths']:
            path = directory/f"{item['id']}.json"
            require(path.is_file(), f'All120 outcomes including failures are required: {path}')
            row = read_json(path)
            require(all(row.get(k) == v for k, v in identity.items()) and row['path_id'] == item['id']
                    and row['family'] == item['family'] and row['reference_sha256'] == item['reference_sha256'],
                    'Result identity/model/source/reference mismatch')
            for key in ('trace', 'inference_samples'):
                member = bound_file(ROOT, row[key]); inputs[member.relative_to(ROOT).as_posix()] = sha(member)
            inputs[path.relative_to(ROOT).as_posix()] = sha(path); rows.append(row)
    return gate, manifest, rows, inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze', 'check', 'evaluate'))
    parser.add_argument('--freeze', type=Path, default=FORMAL/'freeze.json')
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--method', choices=METHODS)
    args = parser.parse_args()
    if args.command == 'freeze':
        print(str(freeze()))
    elif args.command == 'check':
        gate = validate_freeze(args.freeze)
        print(json.dumps(dict(freeze_sha256=gate['freeze_sha256'], models=len(gate['models']), test_accessed=False)))
    else:
        evaluate(args.freeze, args.seed, args.method)


if __name__ == '__main__':
    main()
