"""Verify the five added COPP development programs and preserve every failure."""
from pathlib import Path
import json
import numpy as np
from copp_path_adapter_v4 import sha, write

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'results/copp_v5c_observed_additional'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    registration = read(FOLDER / 'registration.json')
    original = Path(registration['original_registration'])
    assert sha(original) == registration['original_registration_sha256']
    assert all(sha(ROOT/'src'/name) == value for name, value in registration['source_sha256'].items())
    originals = {row['id']: row for row in registration['cases']}
    records = read(FOLDER / 'summary.json')
    cases = []
    for record in records:
        row = originals[record['path_id']]
        reference_file = ROOT / row['reference']
        assert sha(reference_file) == row['reference_sha256']
        reference = np.loadtxt(reference_file, delimiter=',', skiprows=1)
        assert row['evaluation_tolerance_mm'] == record['tolerance_mm']
        if row['converted_geometry']:
            assert sha(Path(row['converted_geometry'])) == row['converted_geometry_sha256']
        runs = []
        for iteration in record['iterations']:
            for run in iteration['runs']:
                trace = FOLDER / run['trace']
                assert sha(trace) == run['trace_sha256']
                with np.load(trace) as data:
                    assert np.array_equal(reference, data['reference'])
                metric = run['metrics']
                assert metric['tolerance_mm'] == row['evaluation_tolerance_mm']
                assert metric['limits'] == [100., 2000., 62500.]
                assert run['execution']['extra_time_reserve_fraction'] == .001
                runs.append(dict(iteration=iteration['iteration'], dt_s=run['dt_s'],
                    success=metric['success'], reason=metric['reason'],
                    complete_time_s=metric['complete_time_s'], elapsed_prefix_s=metric['elapsed_prefix_s'],
                    vector_peaks=metric['vector_peaks'], vector_limit_ratios=metric['vector_limit_ratios'],
                    rest_pass=metric['rest_pass'], geometry=metric['geometry'],
                    native_node_violations=iteration['native_node_violations'],
                    time_dilation=run['execution']['time_dilation'],
                    trace=str(trace), trace_sha256=run['trace_sha256']))
        final = [run for run in runs if run['iteration'] == 3]
        cases.append(dict(path_id=row['id'], input_reference_hash=row['reference_sha256'],
            source_geometry_sha256=row['converted_geometry_sha256'],
            nominal_tolerance_mm=row['nominal_tolerance_mm'],
            chord_error_mm=row['arc_chord_error_bound_mm'],
            evaluation_tolerance_mm=row['evaluation_tolerance_mm'],
            fixed_third_pass=len(final) == 2 and all(run['success'] for run in final),
            generated_runs=len(runs), successful_runs=sum(run['success'] for run in runs),
            runs=runs))
    result = dict(scope='Five previously observed development paths; historical final_ identifiers do not denote a final holdout',
        additional_program_count=len(cases),
        additional_fixed_third_pass=sum(row['fixed_third_pass'] for row in cases),
        additional_runs=sum(row['generated_runs'] for row in cases),
        additional_successful_runs=sum(row['successful_runs'] for row in cases),
        source_trace_reference_and_tolerance_checks_pass=True,
        matching_prior_cases=registration['prior_matching_cases'],
        original_historical_13_matrix_fixed_third_pass=8+sum(row['fixed_third_pass'] for row in cases),
        original_historical_13_matrix_successful_runs=48+sum(row['successful_runs'] for row in cases),
        original_historical_13_matrix_generated_runs=78,
        all_distinct_18_development_programs_successful_runs=78+sum(row['successful_runs'] for row in cases),
        all_distinct_18_development_programs_generated_runs=108,
        cases=cases)
    write(ROOT / 'audit/copp_v5c_observed_additional_audit.json', result)
    print(json.dumps({key:value for key,value in result.items() if key not in ('cases','matching_prior_cases')}, indent=2))


if __name__ == '__main__':
    main()
