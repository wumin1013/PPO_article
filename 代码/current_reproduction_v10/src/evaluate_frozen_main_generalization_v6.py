"""Frozen c16 evaluation of all five sealed synthetic programs at both resolutions."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'

import argparse
import json
import runpy
import subprocess
import sys
import time
import numpy as np
import torch
from common import ROOT, sha, utc, write_json
from evaluate import rollout
from replay_strong_checkpoint_v4 import HistoricalActor, MODEL, EXPECTED_MODEL
from runtime_physical_time_v5 import install, VERSION

REGISTRATION = ROOT / 'evidence/frozen_main_generalization_v6.json'
SEAL = ROOT / 'evidence/final_holdout_v6_registration.json'
SEAL_SHA = 'bf0b36ae719084ff8f259b8b607bc44b2493daf9de1b87d984fd2659679db007'
OUT = ROOT / 'results/frozen_main_generalization_v6'
DT_VALUES = (.001, .0005)
WALL_LIMIT_S = 300.


def read(path):
    return json.loads(path.read_text('utf-8'))


def sealed_check(recompute=False):
    if sha(SEAL) != SEAL_SHA:
        raise ValueError('Original sealed registration changed')
    sealed = read(SEAL)
    if sealed['case_count'] != 5 or len(sealed['cases']) != 5:
        raise ValueError('The complete five-case grid is required')
    for relative, expected in sealed['source_and_output_sha256'].items():
        if sha(ROOT / relative) != expected:
            raise ValueError('Sealed source changed: ' + relative)
    if recompute:
        namespace = runpy.run_path(str(ROOT / 'data/final_holdout_v6/register_final_holdout_v6.py'))
        if namespace['verify']() != sealed:
            raise ValueError('Static geometry audit changed')
    return sealed


def existing_controls():
    """Read existing development controls; do not execute them again."""
    controls = []
    for case in ('original', 'dense'):
        physical = ROOT / f'results/historical_strong_v5/{case}_dt1000us.npz'
        original = ROOT / f'evidence/main_model_inputs/{case}_JNN.npz'
        with np.load(physical) as a, np.load(original) as b:
            if set(a.files) != set(b.files):
                raise ValueError('Development control array set differs')
            equal = {key: bool(np.array_equal(a[key], b[key])) for key in a.files}
        if not all(equal.values()):
            raise ValueError('Physical-time entry changes the existing 1 ms control')
        record = read(physical.with_suffix('.json'))
        if not record['success'] or record['model_sha256'] != EXPECTED_MODEL:
            raise ValueError('Incorrect control identity')
        controls.append(dict(case=case, all_arrays_equal=True, arrays=equal,
            physical_time_trace=physical.relative_to(ROOT).as_posix(),
            physical_time_trace_sha256=sha(physical),
            bound_main_trace=original.relative_to(ROOT).as_posix(),
            bound_main_trace_sha256=sha(original)))
    return controls


def register():
    if REGISTRATION.exists():
        return verify()
    sealed = sealed_check(recompute=True)
    previous = read(ROOT / 'results/historical_strong_v5/registration.json')
    source_hashes = dict(previous['source_sha256'])
    source_hashes['evaluate_frozen_main_generalization_v6.py'] = sha(__file__)
    for name, expected in source_hashes.items():
        if sha(ROOT / 'src' / name) != expected:
            raise ValueError('Historical runtime source changed: ' + name)
    cases, grid = [], []
    for item in sealed['cases']:
        reference = np.loadtxt(ROOT / item['reference'], delimiter=',', skiprows=1)
        length = float(np.linalg.norm(np.diff(reference, axis=0), axis=1).sum())
        if abs(length - item['static_audit']['length_mm']) > 1e-9:
            raise ValueError('Sealed length changed')
        row = dict(item, complete_program_length_mm=length,
                   physical_time_budget_s=5. + length / 5.)
        cases.append(row)
        for dt in DT_VALUES:
            name = f"{item['id']}_dt{round(dt * 1e6)}us"
            grid.append(dict(run_id=name, case=item['id'], dt_s=dt,
                physical_tick_budget=round(row['physical_time_budget_s'] / dt),
                trace=f'results/frozen_main_generalization_v6/{name}.npz',
                record=f'results/frozen_main_generalization_v6/{name}.json'))
    if sha(MODEL) != EXPECTED_MODEL:
        raise ValueError('Historical checkpoint changed')
    result = dict(schema=1, created_utc=utc(),
        study_id='frozen_main_generalization_v6',
        scope='Prospectively sealed synthetic complete-program transfer of the frozen historical c16 controller; one checkpoint, no training, no parameter search.',
        authorization='Parent requested all five sealed candidates at both resolutions; execution requires the reviewed registration SHA on the command line.',
        sealed_registration=SEAL.relative_to(ROOT).as_posix(),
        sealed_registration_sha256=SEAL_SHA,
        seal_release='Only this fixed ten-cell study; sealed files remain unchanged. No case can be removed, cropped, replaced or used for tuning.',
        model=MODEL.relative_to(ROOT).as_posix(), model_sha256=EXPECTED_MODEL,
        model_updated=False, training_seed=923202, new_training_runs=0,
        policy_loader='src/frozen_policy.py::FrozenPolicy.mean',
        main_evidence_manifest='evidence/main_model_v5.json',
        main_evidence_manifest_sha256=sha(ROOT / 'evidence/main_model_v5.json'),
        executor=VERSION, execution_entry='src/runtime_physical_time_v5.py::install',
        executor_chain=['runtime_physical_time_v5', 'runtime_resolution_fixed_v4e',
                        'terminal_completion_v4d', 'runtime_no_target_override_v4b'],
        excluded_executors=['recoverable_execution_v5_4', 'geometric_resumption_v5_4',
                            'geometric_efficiency_candidates_v5_5'],
        original_guide=True, target_override=False,
        geometric_recovery_or_feed_tuning=False,
        complete_verified_terminal_brake=True,
        control_period_s=.02, heading_filter_tau_s=.012,
        guide_transition_width_mm=.1, terminal_iteration_allowance_s=2.1,
        planar_vector_limits=[100., 2000., 62500.], nominal_tolerance_mm=1.25,
        arc_handling='Each case uses nominal tolerance minus its sealed analytic chord-error bound; same geometry and tolerance at both resolutions.',
        completion='Initial zero velocity and acceleration; every original directed gate; each complete command interval in the finite-segment capsule union; common bounded terminal brake; at least20ms final rest; independent finite-difference vector acceptance.',
        failure_rule='Every exception and failure retains its full available command prefix, elapsed prefix time, reason and geometry status; completion time is null. Physical wall-limit expiry is a failure.',
        wall_limit_per_run_s=WALL_LIMIT_S,
        wall_limit_scope='Checked during complete execution; trace serialization and independent audit follow so failures can be retained. This is not a hard real-time bound.',
        statistical_unit='One full synthetic program per fixed checkpoint and resolution. Neither interpolation ticks nor the two resolutions are independent training seeds.',
        outcome_rule='Report all ten cells and five paired resolution outcomes, including every incomplete execution. No performance-based selection or unregistered rerun.',
        case_count=5, run_count=10, cases=cases, grid=grid,
        static_seal_recomputed=True, prior_1ms_equivalence=existing_controls(),
        source_sha256=source_hashes,
        runtime_manifest_sha256=sha(ROOT / 'runtime/source_manifest.json'))
    write_json(REGISTRATION, result)
    return verify()


def verify():
    registration = read(REGISTRATION)
    sealed = sealed_check()
    if registration['sealed_registration_sha256'] != SEAL_SHA:
        raise ValueError('Wrong candidate seal')
    if [r['id'] for r in registration['cases']] != [r['id'] for r in sealed['cases']]:
        raise ValueError('Case membership or order changed')
    if registration['run_count'] != 10 or len(registration['grid']) != 10:
        raise ValueError('Incomplete case-resolution grid')
    if sha(MODEL) != EXPECTED_MODEL or registration['model_sha256'] != EXPECTED_MODEL:
        raise ValueError('Registered model changed')
    for name, expected in registration['source_sha256'].items():
        if sha(ROOT / 'src' / name) != expected:
            raise ValueError('Registered source changed: ' + name)
    manifest = ROOT / 'runtime/source_manifest.json'
    if sha(manifest) != registration['runtime_manifest_sha256']:
        raise ValueError('Runtime manifest changed')
    for item in read(manifest):
        if sha(ROOT / 'runtime' / item['relative']) != item['sha256']:
            raise ValueError('Frozen runtime changed: ' + item['relative'])
    if sha(ROOT / registration['main_evidence_manifest']) != registration['main_evidence_manifest_sha256']:
        raise ValueError('Main evidence identity changed')
    return registration


def run(run_id, approved_sha):
    registration = verify()
    if approved_sha != sha(REGISTRATION):
        raise ValueError('Execution requires the exact reviewed registration SHA')
    cell = next(x for x in registration['grid'] if x['run_id'] == run_id)
    case = next(x for x in registration['cases'] if x['id'] == cell['case'])
    output = ROOT / cell['trace']
    if output.exists() or output.with_suffix('.json').exists():
        raise FileExistsError('Do not overwrite a complete or partial formal run: ' + str(output))
    torch.set_num_threads(1)
    started_cpu = time.process_time()
    settings = install(dt=cell['dt_s'], tolerance=case['evaluation_tolerance_mm'])
    reference = np.loadtxt(ROOT / case['reference'], delimiter=',', skiprows=1)
    result = rollout(HistoricalActor(), reference, output, dt=cell['dt_s'],
        tolerance=case['evaluation_tolerance_mm'], wall_s=WALL_LIMIT_S, runtime_version='v3')
    result.setdefault('elapsed_prefix_s', (result['samples'] - 1) * cell['dt_s'])
    result.setdefault('dt_s', cell['dt_s'])
    result.update(study_id=registration['study_id'], run_id=run_id, path=case['id'],
        method='frozen_main_JNNC', source_origin='synthetic', actual_CAM=False,
        model_sha256=EXPECTED_MODEL, model_updated=False, training_seed=923202,
        runtime_version=VERSION, executor_setup=settings,
        reference_sha256=case['reference_sha256'], geometry_sha256=case['geometry_sha256'],
        nominal_tolerance_mm=case['nominal_tolerance_mm'],
        arc_chord_error_bound_mm=case['arc_chord_error_bound_mm'],
        evaluation_tolerance_mm=case['evaluation_tolerance_mm'],
        physical_time_budget_s=case['physical_time_budget_s'],
        physical_tick_budget=cell['physical_tick_budget'],
        cpu_s=time.process_time() - started_cpu,
        cpu_scope='Setup, actor construction, execution, serialization and independent audit in one process.',
        registration_sha256=approved_sha, sealed_registration_sha256=SEAL_SHA,
        action_source='Unchanged c16 deterministic actor; common terminal-brake tail only.',
        no_parameter_or_model_update=True)
    if sha(MODEL) != EXPECTED_MODEL:
        raise ValueError('Historical model changed during execution')
    write_json(output.with_suffix('.json'), result)
    print(json.dumps({key: result[key] for key in ('run_id', 'success', 'complete_time_s',
        'elapsed_prefix_s', 'reason', 'wall_s')}), flush=True)


def run_all(approved_sha):
    registration = verify()
    if approved_sha != sha(REGISTRATION):
        raise ValueError('Wrong registration SHA')
    OUT.mkdir(parents=True, exist_ok=True)
    for cell in registration['grid']:
        record = ROOT / cell['record']
        if record.exists():
            if read(record).get('registration_sha256') != approved_sha:
                raise ValueError('Existing run is incomplete or belongs to another registration')
            continue
        log = OUT / (cell['run_id'] + '.log')
        with log.open('x', encoding='utf-8') as stream:
            completed = subprocess.run([sys.executable, '-B', str(__file__), '--run', cell['run_id'],
                '--approved-registration-sha', approved_sha], stdout=stream, stderr=subprocess.STDOUT,
                check=False)
        if completed.returncode:
            raise RuntimeError('Execution process failed; inspect retained log before continuing: ' + str(log))
        result = read(record)
        print(json.dumps(dict(run_id=cell['run_id'], success=result['success'],
            complete_time_s=result['complete_time_s'], reason=result['reason'])), flush=True)
    records = [read(ROOT / cell['record']) for cell in registration['grid']]
    write_json(OUT / 'completion_receipt.json', dict(created_utc=utc(),
        registration_sha256=approved_sha, executed_cells=len(records),
        successful_cells=sum(r['success'] for r in records),
        all_registered_cells_retained=True, records=[dict(run_id=r['run_id'],
            success=r['success'], complete_time_s=r['complete_time_s'],
            elapsed_prefix_s=r['elapsed_prefix_s'], reason=r['reason']) for r in records]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register', action='store_true')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--run')
    parser.add_argument('--run-all', action='store_true')
    parser.add_argument('--approved-registration-sha')
    args = parser.parse_args()
    if args.register or args.verify:
        value = register() if args.register else verify()
        print(json.dumps(dict(registration=str(REGISTRATION), sha256=sha(REGISTRATION),
            source_count=len(value['source_sha256']), grid=value['grid']), indent=2))
    elif args.run_all:
        run_all(args.approved_registration_sha)
    elif args.run:
        run(args.run, args.approved_registration_sha)
    else:
        parser.error('Choose register, verify, run or run-all')
