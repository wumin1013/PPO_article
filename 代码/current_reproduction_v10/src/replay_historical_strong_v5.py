"""Registered unchanged historical actor on thirteen observed development programs."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'

import argparse
import dis
import json
from pathlib import Path
import numpy as np
import torch
from common import ROOT, sha, utc, write_json
from evaluate import rollout
from replay_strong_checkpoint_v4 import HistoricalActor, MODEL, EXPECTED_MODEL, paths, PROTECTED
from runtime_physical_time_v5 import install, VERSION

V4 = Path(r'D:\PPO_article_work\JNN_efficiency_20261011_v4')
OUT = ROOT / 'results/historical_strong_v5'
SOURCES = sorted(set(PROTECTED + ['replay_historical_strong_v5.py',
    'runtime_physical_time_v5.py', 'runtime_resolution_fixed_v4e.py',
    'terminal_completion_v4d.py', 'runtime_no_target_override_v4b.py',
    'replay_no_target_override_v4b.py', 'unified_training_v3.py', 'paired_learning.py']))


def register():
    file = OUT / 'registration.json'
    if not file.exists():
        rows = paths()
        prior_file = V4 / 'results/strong_fit_v4/pilot_seed101/adaptive/evaluation_v4d/registration.json'
        prior = json.loads(prior_file.read_text('utf-8'))
        for item in prior['cases']:
            if item['id'].startswith('final_'):
                item = dict(item)
                item['reference'] = str(V4 / item['reference'])
                item['historical_split'] = 'observed_development'
                rows.append(item)
        order = ['dense', 'final_dense_fold', 'final_chamfer_chain', 'train_box', 'train_sine',
                 'train_bends', 'train_bezier', 'validation_trapezoid', 'validation_zigzag',
                 'final_rounded_triangle', 'final_quintic_double_bend', 'final_inward_channels', 'original']
        rows = [next(p for p in rows if p['id'] == name) for name in order]
        write_json(file, dict(created_utc=utc(), scope='DEVELOPMENT ONLY; every program previously observed',
            runtime_version=VERSION, model=str(MODEL), model_sha256=EXPECTED_MODEL,
            seed=923202, model_updated=False, dt_s=.001, cases=rows,
            model_identity='Unchanged historical c16d45 checkpoint; distinct from every fresh fit and PPO checkpoint',
            source_sha256={n:sha(ROOT / 'src' / n) for n in SOURCES},
            runtime_manifest_sha256=sha(ROOT / 'runtime/source_manifest.json'),
            observed_case_registration=str(prior_file), observed_case_registration_sha256=sha(prior_file),
            physical_changes=['Terminal loop allowance: 2100 ticks -> round(2.1 / current DT) ticks',
                              'Failure reward budget: max_ticks * .001 -> max_ticks * current DT'],
            order='Dense 1 ms control first, requiring exact equality to archived v4d dense trace before the remaining twelve cases',
            acceptance='Unchanged common planar vector limits, nominal tolerance minus chord bound, every command interval, all ordered gates, full program, initial/final rest',
            retention='All failures and prefixes retained; no successful-only means; no trial-conditioned path changes',
            allowed_additional_resolution_control='Dense at 0.5 ms only; does not add a new program or a new model'))
    r = json.loads(file.read_text('utf-8'))
    if sha(MODEL) != EXPECTED_MODEL:
        raise ValueError('Historical model hash mismatch')
    for name, expected in r['source_sha256'].items():
        if sha(ROOT / 'src' / name) != expected:
            raise ValueError('Registered source changed: ' + name)
    for row in r['cases']:
        if sha(ROOT / row['reference']) != row['reference_sha256']:
            raise ValueError('Registered reference changed: ' + row['id'])
    return r


def patch_audit():
    """Check the two narrow runtime replacements and reinstallation order."""
    from runtime_resolution_fixed_v4e import install as base
    base(dt=.001)
    import joint_runtime as joint
    import direct_compact_runtime as direct
    old_joint = list(dis.Bytecode(joint.JointRuntime.step))
    old_direct = list(dis.Bytecode(direct.DirectCompactRuntime.step))
    checks = []
    for dt in (.001, .0005, .001):
        settings = install(dt=dt)
        checks.append(dict(dt_s=dt, joint_dt_s=joint.DT,
            terminal_allowance_ticks=round(2.1 / joint.DT),
            terminal_allowance_s=round(2.1 / joint.DT) * joint.DT,
            terminal_patch_active=bool(getattr(joint.JointRuntime.step, '_physical_time_v5', False)),
            failure_patch_active=bool(getattr(direct.DirectCompactRuntime.step, '_physical_time_v5', False)),
            settings=settings))
        assert joint.DT == dt
        assert checks[-1]['terminal_allowance_s'] == 2.1
    def instructions(seq):
        return [dict(op=x.opname, arg=x.argrepr) for x in seq]
    output = ROOT / 'audit/runtime_physical_time_independent_check_v5.json'
    if output.exists():
        raise FileExistsError(output)
    write_json(output, dict(created_utc=utc(), checks=checks,
        sources={name:sha(ROOT/'src'/name) for name in SOURCES},
        old_joint_bytecode=instructions(old_joint),
        new_joint_bytecode=instructions(dis.Bytecode(joint.JointRuntime.step)),
        old_direct_bytecode=instructions(old_direct),
        new_direct_bytecode=instructions(dis.Bytecode(direct.DirectCompactRuntime.step)),
        source_edit_scope='Only parent-created runtime_physical_time_v5 installer; archived sources unmodified',
        behavioral_control='Dense 1 ms full-array comparison is mandatory before the remaining programs'))
    print(json.dumps(dict(audit=str(output), checks=checks)), flush=True)


def dense_equality(result_path):
    prior = V4 / 'results/v4d_verified_terminal_preview/strong/dense.npz'
    with np.load(prior) as a, np.load(result_path) as b:
        equality = {key: bool(np.array_equal(a[key], b[key])) for key in a.files}
        same_keys = set(a.files) == set(b.files)
    r = dict(created_utc=utc(), previous_trace=str(prior), previous_sha256=sha(prior),
             current_trace=str(result_path), current_sha256=sha(result_path),
             same_array_keys=same_keys, exact_equal_arrays=equality,
             passed=same_keys and all(equality.values()),
             scope='Every reference, row, position, observation and normalized action; NPZ container bytes need not match')
    write_json(ROOT / 'audit/historical_dense_1ms_equality_v5.json', r)
    if not r['passed']:
        raise AssertionError('Dense 1 ms nominal behavior changed; stop remaining cases')
    return r


def run(case, dt=.001):
    r = register()
    if dt != .001 and case != 'dense':
        raise ValueError('Only dense fine-resolution replay is registered')
    if case != 'dense' or dt != .001:
        check = json.loads((ROOT / 'audit/historical_dense_1ms_equality_v5.json').read_text('utf-8'))
        if not check['passed']:
            raise ValueError('Nominal dense equality control has not passed')
    row = next(x for x in r['cases'] if x['id'] == case)
    setup = install(dt=dt, tolerance=row['evaluation_tolerance_mm'])
    ref = np.loadtxt(ROOT / row['reference'], delimiter=',', skiprows=1)
    file = OUT / (case + f'_dt{round(dt*1e6)}us.npz')
    result = rollout(HistoricalActor(), ref, file, dt=dt, tolerance=row['evaluation_tolerance_mm'],
        wall_s=2400. if case == 'original' else 300., runtime_version='v3')
    if sha(MODEL) != EXPECTED_MODEL:
        raise ValueError('Model changed during replay')
    result.update(scope='DEVELOPMENT ONLY', runtime_version=VERSION, executor_setup=setup,
        method='historical_strong_JNNC_unchanged', path=case, seed=923202,
        model_sha256=EXPECTED_MODEL, model_updated=False, reference_sha256=row['reference_sha256'],
        nominal_tolerance_mm=row['nominal_tolerance_mm'],
        arc_chord_error_bound_mm=row['arc_chord_error_bound_mm'],
        action_source='unchanged_923202_historical_actor; common terminal brake',
        registration_sha256=sha(OUT / 'registration.json'))
    write_json(file.with_suffix('.json'), result)
    if case == 'dense' and dt == .001:
        dense_equality(file)
    print(json.dumps(dict(case=case, dt_s=dt, success=result['success'],
        time_s=result['complete_time_s'], prefix_s=result['elapsed_prefix_s'],
        reason=result['reason'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register', action='store_true')
    parser.add_argument('--patch-audit', action='store_true')
    parser.add_argument('--case')
    parser.add_argument('--remaining-nominal', action='store_true')
    parser.add_argument('--dt', type=float, choices=[.001,.0005], default=.001)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.register:
        print(json.dumps(register(), indent=2))
    elif args.patch_audit:
        patch_audit()
    elif args.case:
        run(args.case, args.dt)
    elif args.remaining_nominal:
        for row in register()['cases']:
            if row['id'] != 'dense':
                run(row['id'])
    else:
        parser.error('Choose a registered operation')
