"""Register and retain a bounded three-candidate development comparison."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import copy
import json
import time
import types
import torch
from common import ROOT, sha, utc, write_json
import geometric_resolution_check_v5 as resolution
from geometric_efficiency_candidates_v5 import VERSION, CANDIDATES, controller_class

OUT = ROOT/'results/geometric_efficiency_candidates_v5_5'
PRIOR = ROOT/'results/geometric_resumption_candidate_v5_4'
CASES = resolution.CASES


def read(path):
    return json.loads(path.read_text('utf-8'))


def register():
    assert not OUT.exists(), 'Never replace a registered development experiment'
    parent = resolution.verify()
    original = {case: read(PRIOR/case/'actual.json') for case in CASES}
    assert all(row['success'] for row in original.values())
    reg = copy.deepcopy(parent)
    reg.update(created_utc=utc(), version=VERSION,
        scope='Three predeclared candidates on the same three previously observed development programs; not a final test or independent training evidence',
        parent_registration_sha256=sha(PRIOR/'registration.json'),
        resolution_entry_registration_sha256=sha(resolution.OUT/'registration.json'),
        diagnosis_sha256=sha(ROOT/'audit/geometric_efficiency_decomposition_v5.json'),
        candidates=CANDIDATES, candidate_order=[spec['id'] for spec in CANDIDATES],
        baseline=dict(id='frozen_v5_4', feed_mm_s=20., heading_tolerance_rad=.05,
            results={case: dict(complete_time_s=row['complete_time_s'], result_sha256=sha(PRIOR/case/'actual.json')) for case, row in original.items()}),
        rationale='The full fold trace contains 0.536 s stored stopping and 0.820 s geometric recovery, including 0.540 s zero-feed heading alignment. The bounded 2x2 comparison changes only geometric recovery feed (20/40 mm/s) and its heading threshold (0.05/0.15 rad), retaining the frozen baseline cell.',
        changes=['Recovery request feed', 'Recovery heading threshold before requesting nonzero feed'],
        sole_algorithm_change='Only two declared physical parameters in the existing fallback request; no geometry, execution, completion, gate, model or action-grid change',
        resumption_action='Each candidate uses its registered feed/heading threshold; target, 0.8 mm preview, guide compensation and action bounds stay frozen',
        parameter_rule='One parameter pair per candidate, shared by every path and resolution; no outcome-dependent parameter extension',
        run_order=CASES, per_program_wall_limit_s=300.,
        nominal_runs=[dict(candidate=spec['id'], case=case, dt_s=.001) for spec in CANDIDATES for case in CASES],
        full_runs='The nine explicit nominal_runs, followed only by the registered selected-resolution rule',
        control_replay='The original v5_4 nominal equivalence replay and dual-resolution stopping prechecks remain archived in the bound resolution-entry evidence; no duplicated equivalence claim is made for changed candidates',
        stop_precheck='One existing-zigzag raw decision, exact stored stop and resumption unit check at each candidate/resolution before its full programs',
        selection='Include frozen v5_4 and candidates that complete all three 1 ms programs. Minimize the arithmetic mean of per-path completion-time ratios to frozen v5_4. Ties within 1e-12 go to the earlier fixed order: baseline, then registered candidates.',
        resolution_rule='Only the selected new candidate receives the same three full 0.5 ms programs. If frozen v5_4 wins, reuse its existing registered 0.5 ms evidence without another run. No fallback selection after the finer-resolution result.',
        budget='Nine complete 1 ms evaluations at 300 s each; at most three selected 0.5 ms evaluations at 300 s each; one existing-zigzag stopping unit check per candidate/resolution before evaluations',
        outcome_retention='Every candidate, full program, failed prefix, branch, stop and cost is retained. Failed elapsed prefixes are never completion times.',
        no_holdout=True, no_training=True, no_main_or_figure8_replacement=True)
    reg['source_sha256'].update({name: sha(ROOT/'src'/name) for name in
        ('geometric_efficiency_candidates_v5.py', 'run_geometric_efficiency_v5.py')})
    write_json(OUT/'registration.json', reg)
    for spec in CANDIDATES:
        for dt in (.001, .0005):
            sub = copy.deepcopy(reg)
            sub.update(dt_s=dt, candidate=spec)
            write_json(OUT/spec['id']/f'dt{round(dt*1e6)}us/registration.json', sub)
    verify()
    print('EFFICIENCY_REGISTERED', sha(OUT/'registration.json'), flush=True)


def verify():
    resolution.verify()
    reg = read(OUT/'registration.json')
    assert reg['candidates'] == CANDIDATES
    assert all(sha(ROOT/'src'/key) == value for key, value in reg['source_sha256'].items())
    assert sha(PRIOR/'registration.json') == reg['parent_registration_sha256']
    assert sha(resolution.OUT/'registration.json') == reg['resolution_entry_registration_sha256']
    assert all(sha(PRIOR/case/'actual.json') == value['result_sha256'] for case, value in reg['baseline']['results'].items())
    assert all(sha(ROOT/row['reference']) == row['reference_sha256'] for row in reg['cases'])
    assert sha(resolution.runner.MODEL) == reg['model_sha256']
    return reg


def configured(spec, dt):
    _, unit, run = resolution.configure(dt)
    folder = OUT/spec['id']/f'dt{round(dt*1e6)}us'
    namespace = dict(run.__globals__)
    def candidate_install(*, dt, tolerance):
        settings = resolution.install_checked(dt=dt, tolerance=tolerance)
        settings.update(geometric_action_unchanged=False, geometric_recovery_candidate=spec.copy(),
            geometric_target_bounds_and_preview_unchanged=True,
            change_scope='Only the registered recovery feed and heading threshold')
        return settings
    namespace.update(OUT=folder, VERSION=VERSION, RecoverableActionCandidate=controller_class(spec),
                     install=candidate_install)
    def rebind(function):
        return types.FunctionType(function.__code__, namespace, function.__name__,
                                  function.__defaults__, function.__closure__)
    return folder, rebind(unit), rebind(run)


def evaluate_candidate(spec, dt):
    verify()
    folder, unit, run = configured(spec, dt)
    reg = read(folder/'registration.json')
    unit_started, unit_cpu = time.perf_counter(), time.process_time()
    unit(reg)
    unit_report = read(folder/'unit_audit/report.json')
    unit_report.update(candidate=spec, dt_s=dt, wall_s=time.perf_counter()-unit_started,
                       process_cpu_s=time.process_time()-unit_cpu)
    write_json(folder/'unit_audit/report.json', unit_report)
    results = []
    for case in CASES:
        started, cpu = time.perf_counter(), time.process_time()
        result = run(case, reg)
        result['cost'].update(full_run_including_archive_and_metrics_process_cpu_s=time.process_time()-cpu,
                              entry_full_wall_s=time.perf_counter()-started)
        result.update(candidate=spec, resolution_entry=resolution.ENTRY_VERSION,
            action_source='Four logged sources: original model, bounded corrected model, geometric resumption action, stored certified cubic stop',
            action_array_definition='Only actually submitted action requests (original, corrected or geometric); action_decision_ids maps each row to its explicit decision source. Certified stops have submitted_action=null and no action row.')
        write_json(folder/case/'actual.json', result)
        results.append(result)
        verify()
    write_json(folder/'summary.json', dict(created_utc=utc(), candidate=spec, dt_s=dt,
        runs=len(results), successes=sum(row['success'] for row in results), results=results,
        registration_sha256=sha(folder/'registration.json')))


def select():
    reg = verify()
    choices = [dict(id='frozen_v5_4', eligible=True, score=1., tie_order=0,
                    times_s={case: reg['baseline']['results'][case]['complete_time_s'] for case in CASES})]
    for index, spec in enumerate(CANDIDATES, 1):
        folder = OUT/spec['id']/'dt1000us'
        rows = {case: read(folder/case/'actual.json') for case in CASES}
        eligible = all(row['success'] for row in rows.values())
        score = (sum(rows[case]['complete_time_s']/reg['baseline']['results'][case]['complete_time_s']
                     for case in CASES)/len(CASES)) if eligible else None
        choices.append(dict(id=spec['id'], eligible=eligible, score=score, tie_order=index,
            times_s={case: rows[case]['complete_time_s'] for case in CASES},
            failed_prefixes_s={case: row['elapsed_prefix_s'] for case, row in rows.items() if not row['success']},
            result_sha256={case: sha(folder/case/'actual.json') for case in CASES}))
    eligible = [choice for choice in choices if choice['eligible']]
    minimum = min(choice['score'] for choice in eligible)
    winner = min((choice for choice in eligible if choice['score'] <= minimum+1e-12), key=lambda row: row['tie_order'])
    target = OUT/'selection.json'
    assert not target.exists()
    write_json(target, dict(created_utc=utc(), registration_sha256=sha(OUT/'registration.json'),
        rule=reg['selection'], selected=winner['id'], choices=choices,
        half_ms_not_used_in_selection=True))
    print('EFFICIENCY_SELECTION', json.dumps(winner), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['register', 'nominal', 'select', 'selected-resolution'])
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.mode == 'register':
        register()
    elif args.mode == 'nominal':
        verify()
        for candidate in CANDIDATES:
            evaluate_candidate(candidate, .001)
    elif args.mode == 'select':
        select()
    else:
        verify()
        selection = read(OUT/'selection.json')
        assert selection['registration_sha256'] == sha(OUT/'registration.json')
        assert any(row['id'] == selection['selected'] and row['eligible'] for row in selection['choices'])
        if selection['selected'] == 'frozen_v5_4':
            print('REUSE_REGISTERED_FROZEN_V5_4_RESOLUTION', flush=True)
        else:
            evaluate_candidate(next(spec for spec in CANDIDATES if spec['id'] == selection['selected']), .0005)
