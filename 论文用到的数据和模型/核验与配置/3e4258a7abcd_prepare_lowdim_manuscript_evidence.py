"""Bind the frozen model and current comparisons for the original-format revision."""
import csv
from pathlib import Path
import numpy as np
from evaluate_historical_dense import ROOT, BASE, write, sha
from fit_retained_basis_repair import read


def main():
    selected = ROOT / 'direct14_dictionary02_resume_round73'
    acceptance = selected / 'joint_numeric_and_mechanism_acceptance_20260923.json'
    certificate = read(acceptance)
    assert certificate['method_frozen'] and certificate['core_joint_numeric_goal_achieved']
    output = BASE / 'paper_evidence_review/verified_results/lowdim_original_revision'
    output.mkdir(exist_ok=True)
    assert not (output / 'paper_result_binding.json').exists(), 'Preserve an existing completed binding'
    binding = dict(selected_checkpoint=certificate['checkpoint'],
        selected_checkpoint_sha256=certificate['checkpoint_sha256'],
        acceptance=str(acceptance), acceptance_sha256=sha(acceptance),
        method_frozen=True, final_manuscript_accepted=False,
        observation_dimension=14, actor_outputs=3, policy_memory_scalars=0,
        cases={}, methods={}, source_sha256={str(acceptance): sha(acceptance)})
    locations = {
        'original': BASE / 'paper_evidence_review/natural_closed_case',
        'dense': ROOT / 'dense_closed_case'}
    for case, directory in locations.items():
        geometry = read(directory / 'geometry.json')
        reference = directory / 'reference.csv'
        binding['cases'][case] = dict(reference=str(reference), reference_sha256=sha(reference),
            geometry=str(directory / 'geometry.json'), geometry_sha256=sha(directory / 'geometry.json'),
            length_mm=float(geometry['analytic_length_mm']), corners=len(geometry['sharp_corners']))
    originals = {
        'S2': [locations['original'] / 's2_complete/result.json', ROOT.parent / 's2_resolution_half_ms/result.json'],
        'Two-action': [ROOT / 'direct14_capacity_two_action_selection_round60' / f'eval_0024_original_{key}.json' for key in ['1ms', 'half_ms']],
        'J-NNC': [selected / f'eval_0000_original_{key}.json' for key in ['1ms', 'half_ms']]}
    denses = {
        'S2': [ROOT / 'dense_s2_initial/result.json', ROOT / 'dense_s2_half_ms/result.json'],
        'Two-action': [ROOT / 'direct14_capacity_two_action_selection_round60' / f'eval_0024_dense_{key}.json' for key in ['1ms', 'half_ms']],
        'J-NNC': [selected / f'eval_0000_dense_{key}.json' for key in ['1ms', 'half_ms']]}
    for name in originals:
        binding['methods'][name] = {}
        for case, specifications in [('original', originals), ('dense', denses)]:
            entries = []
            for dt, path in zip([.001, .0005], specifications[name]):
                result = read(path)
                assert result['success'], path
                trace = path.parent / 'trace.npz' if name == 'S2' else path.with_suffix('.npz')
                assert result['trace_sha256'] == sha(trace)
                with np.load(trace) as data:
                    ref, rows = data['reference'], data['rows']
                expected = np.loadtxt(binding['cases'][case]['reference'], delimiter=',', skiprows=1)
                assert np.array_equal(ref, expected)
                speed = np.linalg.norm(rows[:, 2:4], axis=1)
                cruise = speed >= 99.
                edges = np.diff(np.r_[False, cruise, False].astype(int))
                intervals = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
                entry = dict(result=str(path), result_sha256=sha(path), trace=str(trace), trace_sha256=sha(trace),
                    dt_s=dt, complete_time_s=result['complete_time_s'],
                    maximum_distance_mm=result['analytic']['maximum_evaluated_distance_mm'],
                    vector_peaks=result['audit']['peaks'],
                    cruise_ge99_s=float(cruise.sum()*dt), longest_cruise_ge99_s=float(intervals.max()*dt),
                    executed_distance_mm=float(speed.sum()*dt), speed_deficit_mm=float(np.maximum(100-speed, 0).sum()*dt))
                if case == 'original':
                    if name == 'S2':
                        from train_mixed_resolution_ppo import HALF_METRICS
                        metric = locations['original'] / 's2_paper_metrics.json' if dt == .001 else HALF_METRICS[2]
                    else:
                        metric = path.with_name(path.stem + '_metrics.json')
                    payload = read(metric)
                    metrics = payload.get('metrics', payload)
                    assert len(metrics['corners']) == 34
                    entry.update(metrics=str(metric), metrics_sha256=sha(metric),
                                 summed_corner_deficit_mm=float(sum(c['speed_deficit_mm'] for c in metrics['corners'])))
                    binding['source_sha256'][str(metric)] = sha(metric)
                entries.append(entry)
                for item in [path, trace]:
                    binding['source_sha256'][str(item)] = sha(item)
            binding['methods'][name][case] = entries
    matched = ROOT / 'matched14_two_action_evidence_20260923.json'
    audit = read(selected / 'paired_resolution_extended_audit.json')
    binding.update(conservative_dense_two_action_times_s=audit['matched14_two_action_times_s'],
        conservative_dense_source=str(matched), conservative_dense_source_sha256=sha(matched),
        primary_two_action_selection='Capacity-corresponding trained checkpoint24;complete success on both cases and both resolutions.',
        conservative_dense_comparison='Retain the fastest valid dense result at each resolution across earlier14input two-output checkpoints,including candidates that did not succeed on another case.',
        selected_model_count=1, verified_independent_full_pipeline_repetitions=0,
        historical_training_budgets_equal=False,
        efficiency_audit=str(selected / 'complete_efficiency_0000.json'),
        mechanism_audit=str(selected / 'mechanism_0000/audit.json'),
        original_source=r'D:\Nutstore\1\基于强化学习的运动规划方法\论文项目\submission_package_EAAI\06_latex_source_files\latex_source_anonymous\manuscript.tex',
        original_pdf=r'D:\Nutstore\1\基于强化学习的运动规划方法\论文项目\submission_package_EAAI\01_manuscript_without_author_details\manuscript_without_author_details.pdf')
    assert sha(binding['original_source']) == '7e396a937555d8b63a48427213a93198797540767be3ec0cd8fc04ff078f8880'
    assert sha(binding['original_pdf']) == '074724d783318f227720ce09fe07410eb67221c572d661962c4ff2d7db64caa7'
    write(output / 'paper_result_binding.json', binding)
    with (output / 'all34_corner_statistics.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['method', 'dt_s', 'corner_id', 'speed_deficit_mm', 'minimum_speed_mm_s'])
        for name, methods in binding['methods'].items():
            for entry in methods['original']:
                payload = read(entry['metrics'])
                for corner in payload.get('metrics', payload)['corners']:
                    writer.writerow([name, entry['dt_s'], corner['id'], corner['speed_deficit_mm'], corner['minimum_speed_mm_s']])
    print('MANUSCRIPT_EVIDENCE', output, flush=True)


if __name__ == '__main__':
    main()
