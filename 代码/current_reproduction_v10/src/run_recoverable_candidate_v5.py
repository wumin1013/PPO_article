"""Register and evaluate one stopping-continuation development candidate."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import copy
import json
import time
import numpy as np
import torch
from common import ROOT, sha, utc, write_json
from replay_strong_checkpoint_v4 import MODEL, EXPECTED_MODEL
from runtime_physical_time_v5 import install
from predictive_action_candidate_v5 import new_environment, snapshot, state_bytes, retained_step
from recoverable_action_candidate_v5 import (VERSION, OFFSETS, FACTORS,
    RecoverableActionCandidate, one_decision, execute_stop, stop_start_fingerprint)
from braking_reachability_v5 import stopping_extension
from metrics_exact_index_v5 import metrics

OUT = ROOT/'results/recoverable_action_candidate_v5_3'
CASES = ['final_dense_fold', 'final_chamfer_chain', 'validation_zigzag']


def register():
    path = OUT/'registration.json'
    if not path.exists():
        prior = ROOT/'results/predictive_stop_candidate_v5_2/registration.json'
        reg = copy.deepcopy(json.loads(prior.read_text('utf-8')))
        reg.update(created_utc=utc(), version=VERSION, scope='Previously observed development programs only',
            parent_candidate_registration_sha256=sha(prior), prediction_decisions=1,
            nominal_prediction_s=.02, candidates=25, candidate_physical_feed_factors=FACTORS,
            candidate_steering_offsets_rad=OFFSETS, first_action_only=True,
            remaining_actions='No later model actions predicted; append a complete common cubic stopping continuation',
            terminal_rule='A successful common terminal tail is checked fully, including all intervals, gates, vectors and final rest; it can exceed the 20 ms control period.',
            normal_action='Accept raw model only when the actual 20 ms decision and full subsequent stop pass',
            selection='Maximum ordered progress after the decision; ties minimum normalized physical action change then fixed enumeration',
            no_solution='Execute the entire previously certified stopping continuation, then resume the same controller from rest. No rejected action fallback.',
            sole_algorithm_change=None,
            changes=['Replace six-feedback horizon with one decision plus executable stopping continuation',
                     'Extend fixed feed grid to [0.2,0.4,0.6,0.8,1.0] to include active deceleration',
                     'Retain the accepted stop as backup and execute it when no candidate passes'],
            actual_command_sources=['raw model', 'finite-grid corrected model', 'stored certified cubic stop'],
            guarantee_scope='The represented stopping family is checked numerically with the common contract; completion and efficiency remain full-run experimental outcomes',
            no_recursive_guarantee=None, run_order=CASES, per_program_wall_limit_s=300.,
            parameter_rule='One bounded development candidate; no per-case changes, no additional repeat selected by outcome',
            budget='300 s full program wall budget; common length-based physical-time timeout unchanged',
            stopping_cost='Every attempted full stopping continuation and all intervals retained',
            outcome_rule='All three full programs attempted once, all failed prefixes retained; no new final paths used')
        reg['source_sha256'].update({name: sha(ROOT/'src'/name) for name in
            ['recoverable_action_candidate_v5.py', 'run_recoverable_candidate_v5.py',
             'metrics_exact_index_v5.py', 'exact_distance_index_v5.py']})
        write_json(path, reg)
    reg = json.loads(path.read_text('utf-8'))
    assert sha(MODEL) == reg['model_sha256'] == EXPECTED_MODEL
    assert all(sha(ROOT/'src'/name) == digest for name, digest in reg['source_sha256'].items())
    assert all(sha(ROOT/row['reference']) == row['reference_sha256'] for row in reg['cases'])
    return reg


def unit_audit(reg):
    out = OUT/'unit_audit'
    out.mkdir(parents=True, exist_ok=False)
    install(dt=.001, tolerance=1.25)
    ref = np.loadtxt(ROOT/'data/train_box.csv', delimiter=',', skiprows=1)
    env = new_environment(ref)
    points = []
    original_update = env.tracker.update
    def record(point):
        points.append(np.asarray(point).copy())
        return original_update(point)
    env.tracker.update = record
    before = state_bytes(env)
    sr, sp, stop = stopping_extension(env)
    assert stop['feasible'] and before == state_bytes(env) and points == []
    policy = RecoverableActionCandidate(MODEL, out/'controller')
    raw = policy.teacher.mean(env.observation()).numpy()
    clone = new_environment(ref, snapshot(env))
    expected, error = retained_step(clone, raw)
    action, rows, decision = policy.step(env)
    assert error is None and decision['source'] == 'raw_model'
    assert np.array_equal(raw, action) and np.array_equal(rows, expected)
    assert len(points) == len(rows)
    stop_rows, stop_positions, stop_fingerprint = policy.backup
    start_v, start_a = env.velocity.copy(), env.acceleration.copy()
    unchanged = state_bytes(env)
    mismatched = new_environment(ref, snapshot(env))
    mismatched.env.velocity[0] += .01
    mismatch_before = state_bytes(mismatched)
    try:
        execute_stop(mismatched, stop_positions, stop_rows, stop_fingerprint)
        raise AssertionError('Mismatched stop should have been rejected')
    except AssertionError as exc:
        assert 'different real physical' in str(exc)
    assert state_bytes(mismatched) == mismatch_before and state_bytes(env) == unchanged
    executed = execute_stop(env, stop_positions, stop_rows, stop_fingerprint)
    assert np.array_equal(executed, stop_rows)
    assert len(points) == len(rows)+len(executed)
    assert not env.done and env.rest_ticks*.001 >= .02-1e-10
    # Resumption uses the same controller after the fully executed stop.
    policy.backup = None
    again, resumed, resumed_record = policy.step(env)
    assert len(resumed) > 0 and env.tracker.inside
    summary = policy.close()
    np.savez_compressed(out/'physical_continuation.npz', reference=ref,
        rows=np.vstack([rows, executed, resumed]), positions=np.vstack([ref[0], points]),
        stopping_start_velocity=start_v, stopping_start_acceleration=start_a)
    write_json(out/'report.json', dict(status='passed', created_utc=utc(),
        actual_raw_rows_equal_to_common_executor=True, retained_stop_executed_exactly=True,
        all_positions_recorded=True, intermediate_stop_does_not_finish_program=True,
        same_controller_resumed=True, stale_backup_rejected_before_integration=True, controller=summary,
        registered_sources=reg['source_sha256']))
    print('RECOVERABLE_UNIT_AUDIT_PASSED', flush=True)


def run(case, reg):
    unit = json.loads((OUT/'unit_audit/report.json').read_text('utf-8'))
    assert unit['status'] == 'passed' and unit['registered_sources'] == reg['source_sha256']
    row = next(r for r in reg['cases'] if r['id'] == case)
    out = OUT/case
    out.mkdir(parents=True, exist_ok=False)
    ref = np.loadtxt(ROOT/row['reference'], delimiter=',', skiprows=1)
    settings = install(dt=.001, tolerance=row['evaluation_tolerance_mm'])
    started = time.perf_counter()
    env = new_environment(ref)
    env.env.max_ticks = round((5.+np.linalg.norm(np.diff(ref, axis=0), axis=1).sum()/5.)/.001)
    init_wall = time.perf_counter()-started
    points, chunks, observations, actions, sources, action_decision_ids = [], [], [], [], [], []
    original_update = env.tracker.update
    def record(point):
        points.append(np.asarray(point).copy())
        return original_update(point)
    env.tracker.update = record
    policy = RecoverableActionCandidate(MODEL, out/'predictions', deadline=started+300.)
    error = None
    attempted = 0
    last_report = started
    try:
        while not env.done:
            attempted += 1
            observations.append(env.observation().copy())
            action, rows, decision = policy.step(env)
            if action is not None:
                actions.append(action)
                action_decision_ids.append(decision['decision_id'])
            chunks.append(rows)
            sources.extend([decision['source']]*len(rows))
            if time.perf_counter()-last_report > 30.:
                print('RECOVERABLE_PROGRESS', case, env.ticks, env.tracker.frontier,
                      policy.counts, flush=True)
                last_report = time.perf_counter()
    except Exception as exc:
        error = type(exc).__name__+': '+str(exc)
        if len(policy.actual_partial_rows):
            partial = np.asarray(policy.actual_partial_rows)
            assert sum(len(x) for x in chunks)+len(partial) <= len(points)
            chunks.append(partial)
            sources.extend(['aborted_actual_decision']*len(partial))
    finally:
        summary = policy.close()
    rows = np.vstack(chunks) if chunks else np.empty((0, 13))
    positions = np.vstack([ref[0], np.asarray(points).reshape(-1, 2)])
    np.savez_compressed(out/'actual.npz', reference=ref, positions=positions, rows=rows,
        observations=np.asarray(observations), actions=np.asarray(actions),
        action_decision_ids=np.asarray(action_decision_ids, dtype=np.int64),
        command_sources=np.asarray(sources))
    result = metrics(ref, positions, .001, row['evaluation_tolerance_mm'],
                     declared_success=bool(env.success and error is None), reason=env.reason or error)
    result.update(path=case, version=VERSION, scope=reg['scope'], error=error,
        model_sha256=EXPECTED_MODEL, model_updated=False, executor_setup=settings,
        action_source='Three explicitly recorded sources: original model, bounded corrected model, stored cubic stopping continuation',
        reference_sha256=row['reference_sha256'], nominal_tolerance_mm=row['nominal_tolerance_mm'],
        arc_chord_error_bound_mm=row['arc_chord_error_bound_mm'],
        attempted_decisions=attempted, completed_decisions=summary['decisions'], controller=summary,
        action_array_definition='Only actually submitted model actions; action_decision_ids maps each row to the decision log. Certified stops have submitted_action=null and no actor action row.',
        intervals_accounted=(len(rows) == len(points) == env.ticks),
        registration_sha256=sha(OUT/'registration.json'), trace_sha256=sha(out/'actual.npz'),
        cost=dict(initialization_wall_s=init_wall, full_evaluation_wall_s=time.perf_counter()-started),
        complete_time_includes_all_intermediate_stops=True, created_utc=utc())
    write_json(out/'actual.json', result)
    print('RECOVERABLE_RESULT', case, result['success'], result['complete_time_s'],
          result['elapsed_prefix_s'], result['reason'], summary, flush=True)
    return result


def summarize(reg):
    rows = [json.loads((OUT/case/'actual.json').read_text('utf-8')) for case in CASES]
    write_json(OUT/'summary.json', dict(created_utc=utc(), version=VERSION, scope=reg['scope'],
        runs=len(rows), successes=sum(r['success'] for r in rows), results=rows,
        interpretation='Independent development controller; historical main and Figure 8 unchanged; no new training or final-test evidence'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', action='store_true')
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    reg = register()
    if args.audit:
        unit_audit(reg)
    elif args.all:
        for case in CASES:
            run(case, reg)
        summarize(reg)
    else:
        parser.error('Choose --audit or --all')
