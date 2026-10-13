"""Uniform, bounded command-feasibility projection of immutable COPP profiles.

Candidate 0 is the hash-verified v5c execution. Both command periods use one
global time dilation per profile. At most two further candidates may be sampled;
every attempt is retained and fixed SOCP iteration 3 remains canonical.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np
import copp_path_adapter_v4 as core
import copp_adapter_v5b as parent
import copp_geometry_only_v4 as frozen
from copp_cases_v5 import load_case
from metrics_exact_index_v5 import metrics
from runtime_configuration import configure_runtime

ROOT = Path(__file__).resolve().parents[1]
INPUTS = [ROOT/'results/copp_v5c_complete_development', ROOT/'results/copp_v5c_observed_additional']
PERIODS = (.001, .0005)
LIMITS = (100., 2000., 62500.)
CORRECTION_RESERVE = .001
MAX_CANDIDATES = 3


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def common_factor(runs):
    ratios = np.asarray([run['metrics']['vector_limit_ratios'] for run in runs])
    maximum = np.max(ratios, axis=0)
    return float(max(1., maximum[0], np.sqrt(maximum[1]), np.cbrt(maximum[2])))


def case_inputs(record, folder, registration):
    name = record['path_id']
    if folder == INPUTS[0]:
        case = load_case(name)
        return case['reference'], case['source_geometry']
    row = next(row for row in registration['cases'] if row['id'] == name)
    source = ROOT/row['reference']
    if core.sha(source) != row['reference_sha256']:
        raise ValueError('Registered reference changed')
    geometry = None
    if row['converted_geometry']:
        path = Path(row['converted_geometry'])
        if core.sha(path) != row['converted_geometry_sha256']:
            raise ValueError('Registered analytic source changed')
        geometry = read(path)
    return np.loadtxt(source, delimiter=',', skiprows=1), geometry


def profile_file(folder, name, iteration):
    return Path(iteration.get('parent_profile', folder/f"{name}_iteration{iteration['iteration']}_profile.npz"))


def verify_reused_runs(folder, previous, reference, tolerance):
    started = time.perf_counter()
    runs = []
    for old in previous['runs']:
        path = folder/old['trace']
        if core.sha(path) != old['trace_sha256']:
            raise ValueError('Candidate-zero trace hash changed')
        with np.load(path) as trace:
            if not np.array_equal(trace['reference'], reference) or float(trace['dt_s']) != old['dt_s']:
                raise ValueError('Candidate-zero complete reference or command period changed')
            if len(trace['positions']) != old['metrics']['samples']:
                raise ValueError('Candidate-zero metric sample count mismatch')
        if old['metrics']['limits'] != list(LIMITS) or old['metrics']['tolerance_mm'] != tolerance:
            raise ValueError('Candidate-zero evaluation contract mismatch')
        run = deepcopy(old)
        run.update(trace=str(path), trace_reused=True, candidate_index=0,
            inherited_sample_audit_save_wall_s=old['sample_audit_save_wall_s'],
            sample_audit_save_wall_s=0., new_sampling_performed=False,
            independent_acceptance_reused=True)
        runs.append(run)
    if {run['dt_s'] for run in runs} != set(PERIODS):
        raise ValueError('Both registered command periods required')
    if len({run['execution']['time_dilation'] for run in runs}) != 1:
        raise ValueError('Both periods must use one common initial dilation')
    return runs, time.perf_counter()-started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'results/copp_v5d_complete_development')
    args = parser.parse_args()
    if (args.output/'registration.json').exists():
        raise FileExistsError('Preserve all prior candidate and failure evidence')
    prepared = []
    registrations = {}
    for folder in INPUTS:
        reg = read(folder/'registration.json')
        registrations[folder] = reg
        if not all(core.sha(ROOT/'src'/name) == value for name, value in reg['source_sha256'].items()):
            raise ValueError('Source freeze mismatch')
        for record in read(folder/'summary.json'):
            if record != read(folder/(record['path_id']+'.json')):
                raise ValueError('Parent summary differs from immutable per-program record')
            reference, geometry = case_inputs(record, folder, reg)
            prepared.append((folder, record, reference, geometry))
    if len(prepared) != 18 or len({row[1]['path_id'] for row in prepared}) != 18:
        raise ValueError('Exactly eighteen distinct complete development programs required')
    sources = ['copp_command_projection_v5d.py', 'copp_reserve_replay_v5c.py',
        'copp_adapter_v5b.py', 'copp_signed_geometry_v5b.py', 'copp_adapter_v5.py',
        'copp_cases_v5.py', 'copp_path_adapter_v4.py', 'copp_geometry_only_v4.py',
        'copp_time_projection_v4.py', 'metrics.py', 'metrics_exact_index_v5.py',
        'exact_distance_index_v5.py', 'runtime_configuration.py', 'runtime_bootstrap.py']
    args.output.mkdir(parents=True, exist_ok=True)
    registration = dict(created_utc=datetime.now(timezone.utc).isoformat(), version='v5d-command-projection',
        method_identity='Official COPP 0.2.3 TOPP3-SOCP plus project geometry and bounded common command-feasibility projection',
        model_identity='Deterministic planner; no learned model or retraining',
        scope='All eighteen previously observed development programs; no prospective final tests',
        settings=parent.SETTINGS, periods_s=list(PERIODS), limits=list(LIMITS),
        maximum_candidates_including_initial=MAX_CANDIDATES, correction_reserve=CORRECTION_RESERVE,
        selection='Fixed third SOCP iteration for every path; no per-case iterate or version selection',
        rule='Initial v5c candidate unchanged; if joint acceptance fails only for vector limits, multiply the common global dilation by 1.001 * max(1, max(V/Vlim), sqrt(max(A/Alim)), cbrt(max(J/Jlim))) over both periods; resample both and repeat at most twice; all other failures remain failures.',
        reuse='Candidate zero is reused only after original source, complete reference, period, record and trace hashes are verified. Original independent metrics are retained; no computation is claimed for reused sampling/audit.',
        acceptance='Unchanged original gates, each full command interval, reference tolerance minus registered arc chord bound, vector finite differences, initial/final rest and full traversal',
        source_sha256={name:core.sha(ROOT/'src'/name) for name in sources},
        independent_evaluator_review_sha256=core.sha(ROOT/'audit/distance_index_independent_review_v5.json'),
        inputs=[dict(folder=str(folder), registration_sha256=core.sha(folder/'registration.json'),
            summary_sha256=core.sha(folder/'summary.json')) for folder in INPUTS],
        cases=[dict(id=record['path_id'], source_record=str(folder/(record['path_id']+'.json')),
            source_record_sha256=core.sha(folder/(record['path_id']+'.json')),
            reference_tolerance_mm=record['tolerance_mm'], nominal_tolerance_mm=record['nominal_tolerance_mm'],
            chord_error_mm=record['chord_error_mm'],
            profiles=[dict(iteration=row['iteration'], path=str(profile_file(folder, record['path_id'], row)),
                sha256=core.sha(profile_file(folder, record['path_id'], row))) for row in record['iterations']])
            for folder,record,_,_ in prepared])
    core.write(args.output/'registration.json', registration)
    results = []
    for folder, source, reference, geometry in prepared:
        started = time.perf_counter()
        name, tolerance = source['path_id'], source['tolerance_mm']
        record = dict(path_id=name, registered_adapter_version='v5d-command-projection',
            tolerance_mm=tolerance, nominal_tolerance_mm=source['nominal_tolerance_mm'],
            chord_error_mm=source['chord_error_mm'], limits=list(LIMITS),
            parent_record=str(folder/(name+'.json')), parent_record_sha256=core.sha(folder/(name+'.json')),
            optimizer_rerun=False, planner=deepcopy(source.get('planner',{})), iterations=[])
        curve = None
        try:
            for previous in source['iterations']:
                iteration = previous['iteration']
                raw_file = profile_file(folder,name,previous)
                raw = np.load(raw_file)
                if not np.array_equal(raw['reference'], reference):
                    raise ValueError('Immutable raw profile reference differs')
                runs, verify_wall = verify_reused_runs(folder,previous,reference,tolerance)
                dilation = runs[0]['execution']['time_dilation']
                candidates = [dict(candidate_index=0, time_dilation=dilation,
                    trace_and_contract_verification_wall_s=verify_wall,
                    new_sampling_and_audit_wall_s=0., runs=runs)]
                item = dict(iteration=iteration, solver_wall_s=previous['solver_wall_s'], solver_rerun=False,
                    parent_profile=str(raw_file), parent_profile_sha256=core.sha(raw_file),
                    projection_candidates=candidates, runs=runs)
                record['iterations'].append(item)
                for candidate_index in range(1,MAX_CANDIDATES):
                    if all(run['metrics']['success'] for run in runs):
                        break
                    if any(run['metrics']['reason'] not in ('success','vector_limit_violation') for run in runs):
                        item['projection_stop_reason'] = 'Nonkinematic failure is outside time-projection scope'
                        break
                    if curve is None:
                        geometry_started = time.perf_counter()
                        curve,*_ = parent.prepare_geometry(reference,tolerance=tolerance,source_geometry=geometry)
                        sigma,*_ = parent.grid_geometry(curve,geometry,parent.SETTINGS)
                        record['new_geometry_reconstruction_wall_s'] = time.perf_counter()-geometry_started
                    if not np.array_equal(sigma/parent.SETTINGS['spatial_scale_mm'],raw['s']):
                        raise ValueError('Reconstructed immutable station grid differs')
                    factor = (1+CORRECTION_RESERVE)*common_factor(runs)
                    dilation *= factor
                    scaled = core.copp.Profile3rd(raw['a']/dilation**2,raw['b']/dilation**2,
                        tuple(int(x) for x in raw['num_stationary']))
                    new_runs = []
                    candidate_started = time.perf_counter()
                    for dt in PERIODS:
                        clock = time.perf_counter()
                        positions,times,progress,execution = frozen._sample(curve,raw['s'],scaled,dt,parent.SETTINGS)
                        configure_runtime('v3',tolerance=tolerance)
                        result = metrics(reference,positions,dt=dt,tolerance=tolerance,limits=LIMITS)
                        filename = f'{name}_iteration{iteration}_candidate{candidate_index}_dt{round(dt*1e6)}us.npz'
                        np.savez_compressed(args.output/filename,positions=positions,time=times,
                            progress=progress,reference=reference,dt_s=dt)
                        execution.update(time_dilation=dilation, time_projection_version='v5d-command-projection',
                            correction_factor=factor, correction_reserve_fraction=CORRECTION_RESERVE,
                            wrapper_sha256=core.sha(__file__))
                        new_runs.append(dict(dt_s=dt,metrics=result,execution=execution,
                            sample_audit_save_wall_s=time.perf_counter()-clock,
                            trace=filename,trace_sha256=core.sha(args.output/filename),trace_reused=False,
                            new_sampling_performed=True,independent_acceptance_reused=False,
                            candidate_index=candidate_index))
                    runs = new_runs
                    candidates.append(dict(candidate_index=candidate_index,time_dilation=dilation,
                        correction_factor=factor,new_sampling_and_audit_wall_s=time.perf_counter()-candidate_started,
                        runs=runs))
                    item['runs'] = runs
                    core.write(args.output/(name+'.json'),record)
                item['selected_projection_candidate'] = candidates[-1]['candidate_index']
                item['joint_acceptance_pass'] = all(run['metrics']['success'] for run in runs)
                print(json.dumps(dict(path=name,iteration=iteration,candidates=len(candidates),
                    joint_pass=item['joint_acceptance_pass'],times=[run['metrics']['complete_time_s'] for run in runs])),flush=True)
            record['status'] = 'completed' if all(row['joint_acceptance_pass'] for row in record['iterations']) else 'acceptance_failed'
        except Exception as exc:
            record.update(status='failed',error_type=type(exc).__name__,error=str(exc))
            print(json.dumps(dict(path=name,status='failed',error=str(exc))),flush=True)
        record['total_wall_s'] = time.perf_counter()-started
        core.write(args.output/(name+'.json'),record)
        results.append(record)
    core.write(args.output/'summary.json',results)


if __name__ == '__main__':
    main()
