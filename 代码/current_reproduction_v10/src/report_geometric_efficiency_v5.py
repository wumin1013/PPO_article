"""Generate the bounded development report only from completed saved evidence."""
import json
import numpy as np
from common import ROOT, sha, utc, write_json
import run_geometric_efficiency_v5 as experiment
from geometric_efficiency_candidates_v5 import CANDIDATES


def read(path):
    return json.loads(path.read_text('utf-8'))


def main():
    reg = experiment.verify()
    review_path = ROOT/'audit/geometric_efficiency_v5_5_independent_review.json'
    review = read(review_path)
    assert review['status'] == 'passed'
    selection = read(experiment.OUT/'selection.json')
    rows, equality = [], []
    for spec in CANDIDATES:
        for case in experiment.CASES:
            folder = experiment.OUT/spec['id']/'dt1000us'/case
            result = read(folder/'actual.json')
            rows.append(dict(candidate=spec['id'], dt_s=.001, case=case,
                success=result['success'], complete_time_s=result['complete_time_s'],
                failed_prefix_s=None if result['success'] else result['elapsed_prefix_s'], reason=result['reason'],
                sources=result['controller']['counts'], cost=result['cost'], trace_sha256=sha(folder/'actual.npz')))
            if case != 'final_dense_fold':
                with np.load(experiment.PRIOR/case/'actual.npz') as original, np.load(folder/'actual.npz') as new:
                    assert set(original.files) == set(new.files)
                    matches = {key: bool(np.array_equal(original[key], new[key])) for key in original.files}
                equality.append(dict(candidate=spec['id'], dt_s=.001, case=case, array_matches=matches,
                    all_arrays_exactly_equal=all(matches.values())))
    if selection['selected'] != 'frozen_v5_4':
        for case in experiment.CASES:
            folder = experiment.OUT/selection['selected']/'dt500us'/case
            result = read(folder/'actual.json')
            rows.append(dict(candidate=selection['selected'], dt_s=.0005, case=case,
                success=result['success'], complete_time_s=result['complete_time_s'],
                failed_prefix_s=None if result['success'] else result['elapsed_prefix_s'], reason=result['reason'],
                sources=result['controller']['counts'], cost=result['cost'], trace_sha256=sha(folder/'actual.npz')))
            if case != 'final_dense_fold':
                original_path = experiment.resolution.OUT/'dt500us'/case/'actual.npz'
                with np.load(original_path) as original, np.load(folder/'actual.npz') as new:
                    assert set(original.files) == set(new.files)
                    matches = {key: bool(np.array_equal(original[key], new[key])) for key in original.files}
                equality.append(dict(candidate=selection['selected'], dt_s=.0005, case=case,
                    array_matches=matches, all_arrays_exactly_equal=all(matches.values())))
    units = []
    for path in experiment.OUT.glob('*/dt*/unit_audit/report.json'):
        row = read(path)
        units.append(dict(path=str(path.relative_to(ROOT)), wall_s=row['wall_s'], cpu_s=row['process_cpu_s']))
    summary = dict(created_utc=utc(), status='completed_bounded_development_experiment',
        registration_sha256=sha(experiment.OUT/'registration.json'), independent_review_sha256=sha(review_path),
        selected=selection['selected'], selection=selection['choices'], results=rows,
        nonintervention_equivalence=equality, unit_costs=units,
        costs=dict(new_complete_program_evaluations=len(rows),
            full_program_wall_s=sum(row['cost']['entry_full_wall_s'] for row in rows),
            full_program_process_cpu_s=sum(row['cost']['full_run_including_archive_and_metrics_process_cpu_s'] for row in rows),
            unit_wall_s=sum(row['wall_s'] for row in units), unit_cpu_s=sum(row['cpu_s'] for row in units),
            scope='Measured whole-case computation, including recording and final metric work; not an online deadline claim'),
        scientific_identity='Unchanged historical c16d45 actor weights with separately versioned v5_5 execution and recovery rules; not the historical Figure 8 controller',
        scope='Previously observed three-program development comparison; no new independent test, original/dense expansion or training run',
        independent_prerun_source_review=dict(reviewer='/root/journal_requirements',
            scope='Read-only review of two new modules and their bound parent helpers; no case execution',
            corrections_before_registration=['Explicit changed geometric-action metadata', 'Parent resolution evidence verification per run', 'Selection registration binding before finer runs'],
            evidence='Reviewer collaboration messages; parent-bound source hashes remain in registration'))
    write_json(experiment.OUT/'comparison.json', summary)
    lines = ['# Bounded geometric-recovery efficiency experiment', '',
        'All parameters and selection rules were registered before running. The frozen v5_4 controller remains available. The three new candidates change only the recovery feed request and its heading threshold; all ordinary candidates, stopping certificates, ordered geometry, limits and completion rules remain unchanged.', '',
        '| Candidate | dt (ms) | Program | Complete | Time (s) | Failed prefix (s) | Whole-case wall (s) | Whole-case CPU (s) |',
        '|---|---:|---|---|---:|---:|---:|---:|']
    for row in rows:
        elapsed = '--' if row['complete_time_s'] is None else f"{row['complete_time_s']:.4f}"
        failed = '--' if row['failed_prefix_s'] is None else f"{row['failed_prefix_s']:.4f}"
        lines.append(f"| {row['candidate']} | {1000*row['dt_s']:.1f} | {row['case']} | {row['success']} | {elapsed} | {failed} | {row['cost']['entry_full_wall_s']:.3f} | {row['cost']['full_run_including_archive_and_metrics_process_cpu_s']:.3f} |")
    lines += ['', f"Selected candidate: `{selection['selected']}`. Selection minimizes the registered mean of three per-path completion-time ratios among fully completing candidates and the frozen reference; finer-resolution results do not change selection.", '',
        'The frozen fold trace contained 0.536 s of stored stopping and 0.820 s of geometric recovery, including 0.540 s of zero-feed turning. This decomposition motivated the bounded requests. The complete comparison retains the slower 20 mm/s, 0.15 rad candidate as well as both faster candidates.', '',
        'The two programs without geometric intervention have exact saved-array comparisons against the frozen v5_4 traces in comparison.json. Every candidate still has an independent complete-program result; none was omitted after inspecting its outcome.', '',
        'This experiment changes the execution policy around unchanged actor weights. It must remain a separate scientific version until that same execution rule is evaluated on any other paths intended for a unified manuscript result. These three development programs do not establish independent-test performance, five-seed effects or hardware timing.', '',
        f"Whole-program evaluation cost: {summary['costs']['full_program_wall_s']:.3f} s measured wall, {summary['costs']['full_program_process_cpu_s']:.3f} s process CPU. Unit checks are recorded separately. Nested branch and stopping costs must not be added again.", '',
        'Independent source review corrected the action-version metadata and strengthened parent-evidence/selection binding before registration. The implementation reuses the frozen step and runner through a narrow parameter entry; no controller copy, source overwrite or path-specific parameters were introduced.', '',
        'Reproduce saved-evidence checks with `python src/audit_geometric_efficiency_v5.py nominal` and `python src/audit_geometric_efficiency_v5.py selected-resolution` in a fresh evidence destination. Existing reports are protected against overwrite. To inspect existing results, read comparison.json and the independent review instead of rerunning the experiment.', '']
    (ROOT/'audit/geometric_efficiency_v5_5.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(dict(selected=selection['selected'], runs=len(rows), successes=sum(row['success'] for row in rows), cost=summary['costs'])), flush=True)


if __name__ == '__main__':
    main()
