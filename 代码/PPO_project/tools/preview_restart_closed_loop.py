"""Six development-only, closed-loop execution ablations; no training or test API.

Original model/training provenance is referenced unchanged. The evaluation
configuration and source archive are bound separately, including this runner.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import zipfile

import numpy as np
import torch
import yaml

import preview_target_tracking_study as target
import preview_target_tracking_transfer as transfer
from preview_formal_holdout import validate_freeze
from src.environment import cnc_env, kinematics
from src.utils.revision_metrics import independent_kinematics, ordered_projection, _reference_witnesses

ROOT = target.full.core.old.ROOT
BASE = ROOT/'论文项目/revision_20260910_continued'
OUT = BASE/'restart_closed_loop_v1'
PATH_IDS = ('mixed_lengths_validation', 'dense_corners_validation', 'line_curve_validation')
METHODS = ('fixed3', 'adaptive')
old = target.full.core.old


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def artifact(path):
    path = Path(path)
    return dict(file=path.relative_to(ROOT).as_posix(), sha256=sha(path))


def require_paths(rows):
    if len(rows) != 3 or {r['path_id'] for r in rows} != set(PATH_IDS):
        raise ValueError('Exactly the three registered development validation paths are required')


def reference(path_id):
    if path_id not in PATH_IDS:
        raise ValueError('Only the three development validation paths are allowed')
    return old.points(path_id)


def parent_dir(method):
    if method not in METHODS:
        raise ValueError('Only registered seed11 fixed3/adaptive step256 actors')
    return BASE/f'formal_v1/continuation/formal/preview_time_{method}_seed11/checkpoints/step256'


def verify_files(directory, manifest, ledger):
    for name, expected in manifest.items():
        path = (directory/name).resolve()
        if not path.is_relative_to(directory.resolve()) or sha(path) != expected:
            raise ValueError(f'Archived file changed: {path}')
        ledger[path.relative_to(ROOT).as_posix()] = expected


def verify_parent(parent, ledger):
    manifest = read(parent/'snapshot_manifest.json')
    verify_files(parent, manifest['files'], ledger)
    ledger.update({(parent/n).relative_to(ROOT).as_posix(): sha(parent/n)
                   for n in ('snapshot_manifest.json', 'provenance.json', 'source.zip')})
    provenance = read(parent/'provenance.json')
    with zipfile.ZipFile(parent/'source.zip') as archive:
        for name, expected in provenance['source_files'].items():
            if sha(ROOT/name) != expected or hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError(f'Original training source changed: {name}')
    if provenance['config_sha256'] != sha(parent/'config.yaml'):
        raise ValueError('Original config/training binding mismatch')
    return provenance


def cached_rows(directory, ledger, *, archive_manifest=None):
    if (directory/'delivery_manifest.json').exists():
        verify_files(directory, read(directory/'delivery_manifest.json')['files'], ledger)
    elif archive_manifest is not None:
        archive = read(archive_manifest)['inputs']
        for path in directory.iterdir():
            if path.is_file() and archive.get(path.relative_to(ROOT).as_posix()) != sha(path):
                raise ValueError(f'Historical cache does not match archived report: {path}')
        ledger[archive_manifest.relative_to(ROOT).as_posix()] = sha(archive_manifest)
    summary = read(directory/'summary.json')
    rows = summary['rows']
    require_paths(rows)
    binding = read(directory/'binding.json')
    for row in rows:
        trace = directory/(row['path_id']+'.npz')
        saved = read(directory/(row['path_id']+'.json'))
        if saved != row or row['trace_sha256'] != sha(trace):
            raise ValueError('Cached row/summary/trace mismatch')
        if any(row[k] != binding[k] for k in ('checkpoint_sha256', 'config_sha256', 'source_sha256')):
            raise ValueError('Cached source/config/model mismatch')
    for path in directory.iterdir():
        if path.is_file():
            ledger[path.relative_to(ROOT).as_posix()] = sha(path)
    return rows


def evaluation_config(parent_cfg, training):
    cfg = copy.deepcopy(parent_cfg)
    if 'feed_tracking' in cfg['revision']:
        raise ValueError('Expected original uncorrected model configuration')
    cfg['revision']['feed_tracking'] = copy.deepcopy(target.configuration(
        'preview_time', 'adaptive', 101)['revision']['feed_tracking'])
    cfg['revision']['evaluation_transfer'] = copy.deepcopy(training)
    return cfg


@contextmanager
def corrected_execution():
    """Exactly the existing target-tracking scalar wrapper; original v2 retained."""
    original = cnc_env.apply_kinematic_constraints
    if original is not kinematics.apply_kinematic_constraints:
        raise RuntimeError('Start a fresh process without an installed candidate')
    if target.full.core.plan_terminal_brake.__module__ != 'terminal_braking':
        raise RuntimeError('Original v2 terminal brake required')
    cnc_env.apply_kinematic_constraints = target.apply_target_tracking_constraints
    try:
        yield
    finally:
        cnc_env.apply_kinematic_constraints = original


def load_trace(path):
    with np.load(path, allow_pickle=False) as data:
        return {str(k): data['data'][:, i] for i, k in enumerate(data['columns'])}


def diagnostics(columns, points):
    c = columns
    t = c['timestamp_s']
    p = np.column_stack((c['x_mm'], c['y_mm']))
    kin = independent_kinematics(p, t, initial_velocity=[0., 0.], initial_acceleration=[0., 0.],
        initial_heading=float(c['heading_rad'][0]), initial_angular_velocity=0., initial_angular_acceleration=0.)
    coverage = ordered_projection(points, p, tolerance_mm=1.25, witness_spacing_mm=.01)
    tail = np.nan_to_num(c.get('terminal_braking', np.zeros(len(t)))) > .5
    moving = kin['feed_mm_s'] > 1e-9
    interior = (~tail) & (~moving) & np.maximum.accumulate(moving) & np.maximum.accumulate(moving[::-1])[::-1]
    interior[0] = False
    zero = np.flatnonzero(interior)
    fields = ('timestamp_s', 'v_mm_s', 'a_mm_s2', 'heading_rad', 'omega_rad_s',
              'alpha_rad_s2', 'angular_jerk_rad_s3', 'p4_status__v_intent', 'p4_status__omega_intent')
    def value(v):
        return float(v) if np.isfinite(v) else None
    windows = []
    for index in zero:
        rows = []
        for i in range(max(0, index-3), min(len(t), index+4)):
            row = {k: value(c[k][i]) for k in fields if k in c}
            row.update({f'output_{k}': value(kin[k][i]) for k in
                ('feed_mm_s', 'heading_rad', 'angular_velocity_rad_s', 'angular_acceleration_rad_s2', 'angular_jerk_rad_s3')})
            rows.append(row)
        windows.append(dict(zero_time_s=float(t[index]), rows=rows))
    gap = None
    if not coverage['ordered_coverage_complete']:
        witnesses, arcs = _reference_witnesses(np.asarray(points, float), .01)
        frontier = int(coverage['covered_witness_count'])
        point = witnesses[frontier]
        starts, vectors = p[:-1], np.diff(p, axis=0)
        lengths2 = np.sum(vectors*vectors, axis=1)
        fractions = np.clip(np.divide(np.sum((point-starts)*vectors, axis=1), lengths2,
            out=np.zeros(len(vectors)), where=lengths2 > 0), 0., 1.)
        distances = np.linalg.norm(starts+fractions[:, None]*vectors-point, axis=1)
        nearest = int(np.argmin(distances))
        progress = coverage['ordered_progress']
        stall = int(np.flatnonzero(progress >= progress[-1]-1e-14)[0])
        gap = dict(first_uncovered_witness_mm=point.tolist(), first_uncovered_arc_mm=float(arcs[frontier]),
            match_radius_mm=1.245, minimum_distance_to_entire_command_polyline_mm=float(distances[nearest]),
            first_witness_exceeds_physical_tolerance=bool(distances[nearest] > 1.25),
            permanent_stall_time_s=float(t[stall]), post_stall_time_s=float(t[-1]-t[stall]),
            diagnostic_only='First witness distance does not replace ordered certification or bound later omissions')
    target_values = c.get('p4_status__v_intent', np.array([]))
    target_values = target_values[np.isfinite(target_values)]
    return dict(interior_zero_speed_samples=len(zero), interior_zero_speed_times_s=t[zero].tolist(),
        minimum_interior_feed_mm_s=float(np.min(kin['feed_mm_s'][1:][~tail[1:]])) if np.any(~tail[1:]) else None,
        target_feed_min_mm_s=float(target_values.min()) if len(target_values) else None,
        target_feed_max_mm_s=float(target_values.max()) if len(target_values) else None,
        zero_speed_heading_windows=windows, first_coverage_omission=gap,
        ordered_progress=float(coverage['ordered_progress'][-1]), position_heading_audit_preserved=True)


def screening_gate(before, after):
    """Permit a separate training screen only; never adopt from actor transfer."""
    if len(before) != len(after):
        raise ValueError('Paired complete model/path sets required')
    key = lambda row: (row['model_label'], row['path_id'])
    base = {key(r): r for r in before}
    if len(base) != len(before) or {key(r) for r in after} != set(base):
        raise ValueError('Duplicate or mismatched model/path pair')
    regressions = [key(r) for r in after if base[key(r)]['success'] and not r['success']]
    paired = [(base[key(r)], r) for r in after if base[key(r)]['success'] and r['success']]
    before_n, after_n = sum(r['success'] for r in before), sum(r['success'] for r in after)
    before_t = sum(a['elapsed_time_s'] for a, _ in paired)
    after_t = sum(b['elapsed_time_s'] for _, b in paired)
    time_improved = bool(paired and after_t < before_t-1e-9)
    return dict(before_success_count=before_n, after_success_count=after_n, total_pairs=len(before),
        original_success_regressions=[list(k) for k in regressions], common_success_pairs=len(paired),
        common_success_before_time_s=before_t if paired else None,
        common_success_after_time_s=after_t if paired else None,
        common_success_time_change_pct=(after_t/before_t-1)*100 if paired and before_t else None,
        eligible_for_separate_training_screen=bool(not regressions and (after_n > before_n or time_improved)),
        adopted=False, rule='No original successful path becomes failure; additionally success count rises or total time falls on the same jointly fully successful model/path pairs',
        interpretation='Development execution ablation only; no new training, learned-D attribution or holdout claim')


def preflight():
    gate = validate_freeze(BASE/'formal_v1/freeze.json')
    ledger = {str((BASE/'formal_v1/freeze.json').relative_to(ROOT)): gate['freeze_sha256']}
    manifest = old.manifest()
    if {p['id'] for p in manifest['paths'] if p['split'] == 'validation'} != set(PATH_IDS):
        raise ValueError('Development validation whitelist changed')
    ledger[(old.DATA/'manifest.json').relative_to(ROOT).as_posix()] = sha(old.DATA/'manifest.json')
    for path_id in PATH_IDS:
        csv = old.DATA/'paths'/(path_id+'.csv')
        ledger[csv.relative_to(ROOT).as_posix()] = sha(csv)
    models = {}
    for method in METHODS:
        parent = parent_dir(method)
        provenance = verify_parent(parent, ledger)
        baseline = cached_rows(parent/'validation', ledger)
        models[f'seed11_{method}_step256'] = dict(parent=parent.relative_to(ROOT).as_posix(),
            parent_checkpoint=artifact(parent/'checkpoint.pt'), parent_config=artifact(parent/'config.yaml'),
            training_provenance=artifact(parent/'provenance.json'), training_source_archive=artifact(parent/'source.zip'),
            training_source_sha256=provenance['source_sha256'], baseline=baseline,
            baseline_directory=(parent/'validation').relative_to(ROOT).as_posix())
    parent, _, _ = transfer.parent_record()
    verify_parent(parent, ledger)
    baseline = cached_rows(parent/'validation', ledger, archive_manifest=BASE/'reports/full_batch_4096_report_v1/manifest.json')
    transfer.verify()
    verify_files(transfer.OUT, read(transfer.OUT/'transfer_manifest.json')['files'], ledger)
    corrected = cached_rows(transfer.OUT/'raw_validation', ledger)
    models['seed101_adaptive_step256'] = dict(parent=parent.relative_to(ROOT).as_posix(), baseline=baseline,
        cached_target=corrected, baseline_directory=(parent/'validation').relative_to(ROOT).as_posix(),
        cached_target_directory=(transfer.OUT/'raw_validation').relative_to(ROOT).as_posix())
    source = target.prepare()['source']
    files = dict(source['source_files'])
    for name in ('PPO_project/tools/preview_restart_closed_loop.py',
                 'PPO_project/tests/test_preview_restart_closed_loop.py',
                 'PPO_project/tools/preview_target_tracking_transfer.py',
                 'PPO_project/tools/preview_formal_holdout.py'):
        files[name] = sha(ROOT/name)
    for name, digest in files.items():
        ledger[name] = digest
    sources = dict(source_files=files, source_sha256=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
        target_environment_source_sha256=source['source_sha256'])
    return dict(version='restart-closed-loop-v1', frozen_model_gate=dict(passed=True, model_count=len(gate['models']),
        freeze_sha256=gate['freeze_sha256'], holdout_geometry_opened=False), models=models, sources=sources,
        input_sha256=ledger, validation_path_ids=list(PATH_IDS), new_rollouts=6,
        cached_baseline_rows=9, cached_target_rows=3, new_training_decisions=0, test_access=False)


def verify_inputs(protocol):
    for name, expected in protocol['input_sha256'].items():
        if sha(ROOT/name) != expected:
            raise ValueError(f'Bound input changed: {name}')


def worker(method):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    protocol = read(OUT/'protocol.json')
    verify_inputs(protocol)
    model = protocol['models'][f'seed11_{method}_step256']
    parent = ROOT/model['parent']
    cfg = evaluation_config(yaml.safe_load((parent/'config.yaml').read_text(encoding='utf-8')), {
        key: model[key] for key in ('parent_checkpoint', 'parent_config', 'training_provenance',
                                    'training_source_archive', 'training_source_sha256')})
    output = OUT/f'seed11_{method}_step256'
    output.mkdir(exist_ok=False)
    (output/'evaluation_config.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding='utf-8')
    binding = dict(model_checkpoint_sha256=sha(parent/'checkpoint.pt'),
        parent_training_source_sha256=model['training_source_sha256'],
        evaluation_config_sha256=sha(output/'evaluation_config.yaml'),
        evaluation_source_sha256=protocol['sources']['source_sha256'],
        target_environment_source_sha256=protocol['sources']['target_environment_source_sha256'],
        additional_training_decisions=0, policy='Fresh deterministic actor call from current closed-loop state each macro decision',
        model_label=f'seed11_{method}_step256', method=method, seed=11, training_decisions=256)
    write(output/'binding.json', binding)
    checkpoint = torch.load(parent/'checkpoint.pt', map_location='cpu', weights_only=False)
    rows = []
    with corrected_execution():
        for path_id in PATH_IDS:
            started = time.perf_counter()
            started_utc = datetime.now(timezone.utc).isoformat()
            env = None
            error = None
            try:
                env = target.full.local.LocalCornerEnvironment(cfg, path_id, reference=reference(path_id))
                agent = target.full.local.make_agent(cfg, env)
                agent.actor.load_state_dict(checkpoint['actor'])
                agent.actor.eval()
                with torch.no_grad():
                    while not env.done:
                        env.step(agent.take_action(env.state, deterministic=True))
                row = copy.deepcopy(env.result)
            except Exception:
                error = traceback.format_exc()
                row = dict(success=False, elapsed_time_s=float(env.trace[-1]['timestamp_s']) if env else 0.,
                    failure_reasons=['closed_loop_exception'], termination_reason='exception')
            row.update(binding, path_id=path_id, started_utc=started_utc, finished_utc=datetime.now(timezone.utc).isoformat(),
                evaluation_wall_s=time.perf_counter()-started, pid=os.getpid(), error=error,
                full_trace_completed=bool(env and env.done), complete_success_time_s=row['elapsed_time_s'] if row['success'] else None)
            if env and len(env.trace) > 1:
                trace_path = output/(path_id+'.npz')
                old.save_trace(trace_path, env.trace)
                row.update(trace_file=trace_path.relative_to(ROOT).as_posix(), trace_sha256=sha(trace_path))
                row['diagnostics'] = diagnostics(load_trace(trace_path), env.reference)
            write(output/(path_id+'.json'), row)
            rows.append(row)
            print(json.dumps({k: row[k] for k in ('model_label', 'path_id', 'success', 'elapsed_time_s', 'error')}, ensure_ascii=False), flush=True)
    verify_inputs(protocol)
    write(output/'summary.json', dict(rows=rows, **binding))


def run():
    if (OUT/'protocol.json').exists():
        raise FileExistsError('Preserve this registered completed/partial run; no duplicate rollouts')
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = preflight()
    write(OUT/'protocol.json', protocol)
    with zipfile.ZipFile(OUT/'evaluation_source.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for name in protocol['sources']['source_files']:
            archive.write(ROOT/name, name)
    invocations = []
    for method in METHODS:
        cmd = [sys.executable, '-B', str(Path(__file__).resolve()), 'worker', '--method', method]
        logdir = OUT/'invocations'
        logdir.mkdir(exist_ok=True)
        record = dict(argv=cmd, command=subprocess.list2cmdline(cmd), cwd=str(ROOT),
            started_utc=datetime.now(timezone.utc).isoformat(), parent_pid=os.getpid(), method=method,
            status='starting', torch_num_threads=1, additional_training_decisions=0)
        start = time.perf_counter()
        with (logdir/f'{method}.stdout.log').open('w', encoding='utf-8') as stdout, \
                (logdir/f'{method}.stderr.log').open('w', encoding='utf-8') as stderr:
            child = subprocess.Popen(cmd, cwd=ROOT, stdout=stdout, stderr=stderr,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            record.update(pid=child.pid, status='running')
            write(logdir/f'{method}.json', record)
            exit_code = child.wait()
        record.update(exit_code=exit_code, status='complete' if exit_code == 0 else 'failed',
            wall_s=time.perf_counter()-start, finished_utc=datetime.now(timezone.utc).isoformat())
        write(logdir/f'{method}.json', record)
        invocations.append(record)
        print(f'{method}: exit={exit_code}, wall={record["wall_s"]:.3f}s', flush=True)
    before, after = [], []
    for label, model in protocol['models'].items():
        for r in model['baseline']:
            row = dict(r, model_label=label, cache_reused=True)
            path = ROOT/model['baseline_directory']/(r['path_id']+'.npz')
            row['diagnostics'] = diagnostics(load_trace(path), reference(r['path_id']))
            before.append(row)
        if 'cached_target' in model:
            for r in model['cached_target']:
                row = dict(r, model_label=label, cache_reused=True)
                path = ROOT/model['cached_target_directory']/(r['path_id']+'.npz')
                row['diagnostics'] = diagnostics(load_trace(path), reference(r['path_id']))
                after.append(row)
        else:
            summary = OUT/label/'summary.json'
            if not summary.exists():
                for path_id in PATH_IDS:
                    row_path = OUT/label/(path_id+'.json')
                    after.append(read(row_path) if row_path.exists() else dict(model_label=label, path_id=path_id,
                        success=False, elapsed_time_s=0., failure_reasons=['worker_failed_before_output'], full_trace_completed=False))
            else:
                after.extend(read(summary)['rows'])
    verify_inputs(protocol)
    final_gate = validate_freeze(BASE/'formal_v1/freeze.json')
    result = dict(before=before, after=after, gate=screening_gate(before, after), invocations=invocations,
        final_freeze_gate_passed=len(final_gate['models']) == 10, input_sha256_unchanged=True,
        new_rollout_count=6, reused_baseline_count=9, reused_target_count=3,
        additional_training_decisions=0, test_access=False)
    write(OUT/'summary.json', result)
    files = {p.relative_to(OUT).as_posix(): sha(p) for p in OUT.rglob('*') if p.is_file()}
    write(OUT/'delivery_manifest.json', dict(files=files, input_sha256=protocol['input_sha256']))
    print(json.dumps(result['gate'], ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('run', 'worker'))
    parser.add_argument('--method', choices=METHODS)
    args = parser.parse_args()
    if args.command == 'run':
        run()
    elif args.method is None:
        parser.error('Worker requires --method')
    else:
        worker(args.method)


if __name__ == '__main__':
    main()
