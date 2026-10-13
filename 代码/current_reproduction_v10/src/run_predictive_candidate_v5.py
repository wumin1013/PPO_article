"""Fixed three-program development diagnosis for the isolated predictive policy."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
import pickle
import time
from pathlib import Path
import numpy as np
import torch
from common import ROOT, sha, utc, write_json
from evaluate import rollout
from frozen_policy import FrozenPolicy
from replay_strong_checkpoint_v4 import MODEL, EXPECTED_MODEL
from runtime_physical_time_v5 import install
from predictive_action_candidate_v5 import (VERSION, OFFSETS, FACTORS, DECISIONS,
    PredictiveActionCandidate, snapshot, restore, state_bytes, prediction, new_environment,
    retained_step)

OUT = ROOT/'results/predictive_action_candidate_v5'
CASES = ['final_dense_fold', 'final_chamfer_chain', 'validation_zigzag']


def register():
    file = OUT/'registration.json'
    if not file.exists():
        basefile = ROOT/'results/historical_strong_v5/registration.json'
        base = json.loads(basefile.read_text('utf-8'))
        inputs = OUT/'inputs'
        inputs.mkdir(parents=True, exist_ok=True)
        rows = []
        for case in CASES:
            row = next(x.copy() for x in base['cases'] if x['id'] == case)
            original = ROOT/row['reference']
            assert sha(original) == row['reference_sha256']
            target = inputs/(case+'.csv')
            if target.exists():
                assert sha(target) == row['reference_sha256']
            else:
                target.write_bytes(original.read_bytes())
            row['source_reference'] = str(original)
            row['reference'] = target.relative_to(ROOT).as_posix()
            row['scope'] = 'Previously observed development only'
            rows.append(row)
        sources = list(base['source_sha256'])+[
            'predictive_action_candidate_v5.py', 'run_predictive_candidate_v5.py',
            'physical_teacher_v5.py']
        reg = dict(created_utc=utc(), version=VERSION, scope='Limited diagnostic algorithm candidate only',
            model_sha256=EXPECTED_MODEL, unchanged_model=True, new_execution_policy=True,
            interpretation='Independent from historical main and its Figure 8; no replacement or five-seed evidence',
            cases=rows, run_order=CASES, dt_s=.001, per_program_wall_limit_s=300.,
            prediction_decisions=DECISIONS, nominal_prediction_s=.12,
            terminal_rule='A successful common terminal tail is checked fully, including intervals, gates, vectors and final rest; predicted duration can exceed 120 ms. Successful early termination implies absorbing rest.',
            normal_action='Return the original raw array exactly if its complete feedback horizon passes',
            candidate_steering_offsets_rad=OFFSETS, candidate_physical_feed_factors=FACTORS,
            candidate_order='Steering offset outer loop, feed factor inner loop',
            candidates=15, first_action_only=True,
            remaining_actions='Query unchanged original model from each new closed-loop observation for the remaining five decisions',
            preview='Original raw third coordinate, unchanged in every first action',
            selection='Among complete feasible predictions, maximum terminal ordered progress; exact ties minimum squared physical change normalized by [pi/2,100,11.2]; remaining ties enumeration order',
            no_solution='Record no_feasible_candidate then submit original raw action; retain actual failure',
            isolation='Freshly constructed FarVectorRuntime; no source method or recorder closure copied. All noncallable state values, max_ticks/fixed_d, projector counts and pending pulse copied. Check actual state and recorder lists every policy call.',
            evaluation='Same full-program interval geometry, original ordered gates, vector bounds, rest and tolerance contract; no path shortening or changed completion rule',
            retention='Every attempted prediction, all branch rows/positions/actions, state snapshots, actual actions and failed prefixes; no outcome filtering',
            budget='Complete evaluator and prediction share 300 s per program. Check deadline before each predicted decision; archive an interrupted branch and actual prefix.',
            source_sha256={name:sha(ROOT/'src'/name) for name in sorted(set(sources))},
            parent_registration_sha256=sha(basefile),
            runtime_snapshot_manifest_sha256=sha(ROOT/'runtime/source_manifest.json'))
        write_json(file, reg)
    reg = json.loads(file.read_text('utf-8'))
    assert sha(MODEL) == reg['model_sha256'] == EXPECTED_MODEL
    assert all(sha(ROOT/'src'/name) == value for name,value in reg['source_sha256'].items())
    assert all(sha(ROOT/row['reference']) == row['reference_sha256'] for row in reg['cases'])
    return reg


def unit_audit(reg):
    output = OUT/'unit_audit'
    output.mkdir(parents=True, exist_ok=False)
    install(dt=.001, tolerance=1.25)
    teacher = FrozenPolicy(MODEL)
    ref = np.loadtxt(ROOT/'data/train_box.csv', delimiter=',', skiprows=1)
    env = new_environment(ref)
    env.env.max_ticks = 97531
    recorded = []
    original_update = env.tracker.update
    def recorder(point):
        recorded.append(np.asarray(point).copy())
        return original_update(point)
    env.tracker.update = recorder
    raw = teacher.mean(env.observation()).numpy()
    saved = snapshot(env)
    assert 'update' not in saved['tracker'] and saved['core']['max_ticks'] == 97531
    clone = new_environment(ref, saved)
    assert 'update' not in vars(clone.tracker) and clone.max_ticks == env.max_ticks
    assert clone.fixed_d == env.fixed_d and np.array_equal(clone.pending_pulse, env.pending_pulse)
    before = state_bytes(env)
    rows, positions, outcome = prediction(clone, saved, raw, teacher)
    assert before == state_bytes(env) and recorded == []
    direct = new_environment(ref, saved)
    expected = []
    for index in range(DECISIONS):
        chosen = teacher.mean(direct.observation()).numpy()
        one, error = retained_step(direct, chosen)
        assert error is None
        expected.append(one)
        if direct.done:
            break
    assert np.array_equal(rows, np.vstack(expected)) and outcome['feasible']
    policy = PredictiveActionCandidate(MODEL, output/'normal_action')
    selected = policy.action_for(env)
    assert selected.dtype == raw.dtype and np.array_equal(selected, raw)
    assert recorded == [] and before == state_bytes(env)
    actual_rows, error = retained_step(env, selected)
    assert error is None and len(recorded) == len(actual_rows)
    assert np.array_equal(actual_rows, rows[:len(actual_rows)])
    normal = policy.close()
    # Inspect a previously diagnosed failure state; this is development only.
    case = 'validation_zigzag'
    with (ROOT/'results/historical_action_diagnosis_exact_v5'/case/'exact_states.pkl').open('rb') as stream:
        old = pickle.load(stream)[-1]
    row = next(r for r in reg['cases'] if r['id'] == case)
    ref2 = np.loadtxt(ROOT/row['reference'], delimiter=',', skiprows=1)
    failure = new_environment(ref2)
    failure.restore(old['state'])
    failure.env.max_ticks = round((5.+np.linalg.norm(np.diff(ref2,axis=0),axis=1).sum()/5.)/.001)
    failed_recorder = []
    update = failure.tracker.update
    def failed_record(point):
        failed_recorder.append(np.asarray(point).copy())
        return update(point)
    failure.tracker.update = failed_record
    saved_failure = state_bytes(failure)
    failing_policy = PredictiveActionCandidate(MODEL, output/'failure_action')
    failing_policy.action_for(failure)
    assert saved_failure == state_bytes(failure) and failed_recorder == []
    failure_summary = failing_policy.close()
    assert failure_summary['branches'] == 16
    # Configuration restore must also preserve a nondefault fixed preview.
    config = snapshot(failure)
    config['core']['fixed_d'] = 1.7
    config['core']['pending_pulse'] = np.array([.001, -.002, .003])
    restore(clone, config)
    assert clone.fixed_d == 1.7 and np.array_equal(clone.pending_pulse, config['core']['pending_pulse'])
    assert clone.max_ticks == config['core']['max_ticks']
    report = dict(status='passed', created_utc=utc(),
        fresh_environment_step_closure=True, six_step_raw_rows_bitwise_equal=True,
        unmodified_raw_return_bitwise_equal=True, max_ticks_preserved=True,
        fixed_d_and_pending_pulse_preserved=True, source_tracker_update_excluded=True,
        source_state_and_recorder_unchanged=True, actual_recorder_only_actual_samples=True,
        normal_prediction=outcome, normal_policy=normal, failed_state_policy=failure_summary,
        registered_sources_sha256=reg['source_sha256'])
    write_json(output/'report.json', report)
    print('PREDICTIVE_UNIT_AUDIT_PASSED', flush=True)
    return report


def run(case, reg):
    report = json.loads((OUT/'unit_audit/report.json').read_text('utf-8'))
    assert report['status'] == 'passed' and report['registered_sources_sha256'] == reg['source_sha256']
    row = next(r for r in reg['cases'] if r['id'] == case)
    directory = OUT/case
    directory.mkdir(parents=True, exist_ok=False)
    reference = np.loadtxt(ROOT/row['reference'], delimiter=',', skiprows=1)
    setup = install(dt=.001, tolerance=row['evaluation_tolerance_mm'])
    started = time.perf_counter()
    policy = PredictiveActionCandidate(MODEL, directory/'predictions', deadline=started+300.)
    output = directory/'actual.npz'
    try:
        result = rollout(policy, reference, output, dt=.001, wall_s=300.,
            tolerance=row['evaluation_tolerance_mm'], runtime_version='v3')
    finally:
        policy_summary = policy.close()
        write_json(directory/'prediction_summary.json', policy_summary)
    result.update(path=case, scope=reg['scope'], method=VERSION, runtime_version=setup['version'],
        executor_setup=setup, model_sha256=EXPECTED_MODEL, model_updated=False,
        action_source='Original c16d45 raw model plus separately registered six-decision predictive first-action selection; common terminal tail',
        reference_sha256=row['reference_sha256'], nominal_tolerance_mm=row['nominal_tolerance_mm'],
        arc_chord_error_bound_mm=row['arc_chord_error_bound_mm'],
        registration_sha256=sha(OUT/'registration.json'), prediction=policy_summary,
        full_run_wall_s=time.perf_counter()-started)
    assert sha(MODEL) == EXPECTED_MODEL
    write_json(output.with_suffix('.json'), result)
    print('PREDICTIVE_RUN',case,result['success'],result.get('complete_time_s'),
          result.get('elapsed_prefix_s'),result['reason'],policy_summary['counts'],flush=True)
    return result


def summarize(reg):
    results = [json.loads((OUT/case/'actual.json').read_text('utf-8')) for case in CASES]
    report = dict(created_utc=utc(),status='all_three_registered_runs_recorded',scope=reg['scope'],
        version=VERSION,model_sha256=EXPECTED_MODEL,runs=3,successes=sum(r['success'] for r in results),
        results=[dict(path=r['path'],success=r['success'],complete_time_s=r.get('complete_time_s'),
            elapsed_prefix_s=r.get('elapsed_prefix_s'),reason=r['reason'],error=r['error'],
            actual_calls=r['actor_calls'],prediction=r['prediction']) for r in results],
        interpretation='Independent development candidate only; no original main replacement, no independent final-test evidence or five-seed claim')
    write_json(OUT/'summary.json',report)
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',action='store_true')
    p.add_argument('--case',choices=CASES)
    p.add_argument('--all',action='store_true')
    p.add_argument('--summarize',action='store_true')
    args=p.parse_args();torch.set_num_threads(1)
    reg=register()
    if args.audit:unit_audit(reg)
    elif args.case:run(args.case,reg)
    elif args.all:
        for case in CASES:run(case,reg)
        summarize(reg)
    elif args.summarize:summarize(reg)
    else:p.error('Choose audit, case, all or summarize')
