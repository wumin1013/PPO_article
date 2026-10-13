"""Independent saved-trace audit of the registered physical resolution entry."""
import inspect
import json
import textwrap
import time
import types
import numpy as np
import torch
from common import ROOT, sha, utc, write_json
from frozen_policy import FrozenPolicy
from replay_strong_checkpoint_v4 import MODEL
import audit_recoverable_candidate_v5 as base
import audit_geometric_resumption_v5 as geometric
import geometric_resolution_check_v5 as entry

OUT = ROOT/'results/geometric_resolution_check_v5_4'
REPORT = ROOT/'audit/geometric_resolution_v5_4_independent_review.json'


def resolution_audit_case(dt, folder):
    prior = base.read(ROOT/'audit/geometric_resumption_candidate_v5_4_independent_review.json')
    assert sha(geometric.__file__) == prior['audit_source_sha256']
    source = textwrap.dedent(inspect.getsource(geometric.extended_audit_case))
    old = "namespace = dict(base.__dict__)"
    new = """assert source.count('len(rr) == 20') == 1
    source = source.replace('len(rr) == 20', 'len(rr) == round(.02/DT)')
    namespace = dict(base.__dict__)
    namespace.update(DT=RESOLUTION_DT, kinematics=resolution_kinematics, install=entry.install_checked)"""
    assert source.count(old) == 1
    source = source.replace(old, new)
    kg = dict(base.kinematics.__globals__)
    kg['DT'] = dt
    kin = types.FunctionType(base.kinematics.__code__, kg, base.kinematics.__name__,
                             base.kinematics.__defaults__, base.kinematics.__closure__)
    namespace = dict(geometric.__dict__)
    namespace.update(OUT=folder, RESOLUTION_DT=dt, resolution_kinematics=kin, entry=entry)
    output = {}
    exec(compile(source, __file__, 'exec'), namespace, output)
    return output['extended_audit_case']()


def installed_constants(dt):
    entry.configure(dt)
    settings = entry.install_checked(dt=dt, tolerance=1.25)
    import joint_runtime as joint
    import turn_aware_projector as turn
    import continuous_gate_guidance as guide
    assert joint.DT == turn.DT == turn.ORIGINAL.DT == entry.controller.DT == dt
    assert joint.MACRO_TICKS == round(.02/dt)
    assert joint.FILTER_TAU == .012 and guide.GATE_TRANSITION_WIDTH_MM == .1
    assert .021-1e-10 in joint.JointRuntime.step.__code__.co_consts
    assert 2.1 in joint.JointRuntime.step.__code__.co_consts
    assert .003 in joint.terminal.plan_planar_terminal_brake.__code__.co_consts
    assert 'round' in joint.terminal.plan_planar_terminal_brake.__code__.co_names
    assert joint.terminal.PlanarBrakingPolicy().hold_s == .020
    assert settings['stopping_hold_s'] == .023 and settings['internal_terminal_rest_s'] == .021
    return settings


def main():
    started = time.perf_counter()
    torch.set_num_threads(1)
    reg = entry.verify()
    assert all((OUT/'dt500us'/row['id']/'actual.json').exists() for row in reg['cases'])
    constants, units = {}, {}
    for dt in (.001, .0005):
        key = f'dt{round(dt*1e6)}us'
        constants[key] = installed_constants(dt)
        unit = base.read(OUT/key/'unit_audit/report.json')
        sub = base.read(OUT/key/'registration.json')
        assert unit['status'] == 'passed' and unit['registered_sources'] == sub['source_sha256']
        assert unit['dt_s'] == dt and unit['stale_backup_rejected_before_integration']
        with np.load(OUT/key/'unit_audit/physical_continuation.npz') as data:
            pp, rr, ref = data['positions'], data['rows'], data['reference']
        assert len(pp) == len(rr)+1 and np.array_equal(pp[1:], rr[:, :2])
        v = np.diff(pp, axis=0)/dt
        a = np.diff(np.vstack([np.zeros(2), v]), axis=0)/dt
        j = np.diff(np.vstack([np.zeros(2), a]), axis=0)/dt
        peaks = [float(np.linalg.norm(value, axis=1).max()) for value in (v, a, j)]
        assert np.all(np.asarray(peaks)/[100., 2000., 62500.] <= 1.00002)
        from geometry_spec import Corridor, GateTracker
        corridor = Corridor(ref, epsilon=1.25, closed=bool(np.linalg.norm(ref[0]-ref[-1]) < 1e-9))
        tracker = GateTracker(corridor, pp[0])
        for point in pp[1:]:
            tracker.update(point)
        assert tracker.inside
        units[key] = dict(status='passed', intervals=len(rr), vector_peaks=peaks,
            geometry_every_interval_inside=True, source_case=unit['source_case'],
            report_sha256=sha(OUT/key/'unit_audit/report.json'),
            trace_sha256=sha(OUT/key/'unit_audit/physical_continuation.npz'))

    old = entry.PRIOR/'validation_zigzag/actual.npz'
    new = OUT/'dt1000us/validation_zigzag/actual.npz'
    with np.load(old) as a, np.load(new) as b:
        assert set(a.files) == set(b.files)
        equal = {key: bool(np.array_equal(a[key], b[key])) for key in a.files}
    assert all(equal.values())
    assert sha(old) == sha(new)
    control = base.read(OUT/'control_1ms.json')
    assert control['status'] == 'passed' and control['all_arrays_equal'] == equal
    constants['dt500us'] = installed_constants(.0005)
    audit_case = resolution_audit_case(.0005, OUT/'dt500us')
    teacher, cases = FrozenPolicy(MODEL), []
    pulse_states, stop_holds = 0, 0
    for row in reg['cases']:
        print('AUDIT_RESOLUTION_CASE', row['id'], flush=True)
        cases.append(audit_case(row, base.read(OUT/'dt500us/registration.json'), teacher))
        folder = OUT/'dt500us'/row['id']/'predictions'
        for state in base.states_at(folder/'decision_states.pkl'):
            assert not np.any(state['core']['pending_pulse'])
            assert not np.any(state['core'].get('_pulse', np.zeros(3)))
            pulse_states += 1
        for stop in base.jsonl(folder/'stops/branch_records.jsonl'):
            if stop['error'] is None:
                assert abs(stop['total_tail_time_s']-stop['braking_time_s']-.023) < 1e-12
                stop_holds += 1
        actual = base.read(OUT/'dt500us'/row['id']/'actual.json')
        assert actual['cost']['full_run_including_archive_and_metrics_process_cpu_s'] >= 0.
        assert actual['resolution_entry'] == entry.ENTRY_VERSION
        assert actual['dt_s'] == .0005
    # Nominal evidence is already independently audited. Record both sides of
    # each numerical check, keeping failed half-ms prefixes out of time deltas.
    comparisons = []
    for case in cases:
        nominal = base.read(entry.PRIOR/case['case']/'actual.json')
        complete = nominal['success'] and case['success']
        comparisons.append(dict(case=case['case'], one_ms_success=nominal['success'],
            one_ms_complete_time_s=nominal['complete_time_s'], half_ms_success=case['success'],
            half_ms_complete_time_s=case['complete_time_s'], half_ms_failed_prefix_s=case['failed_prefix_s'],
            relative_complete_time_change_percent=(100.*(case['complete_time_s']/nominal['complete_time_s']-1.) if complete else None)))
    report = dict(status='passed', created_utc=utc(), audit_source_sha256=sha(__file__),
        registration_sha256=sha(OUT/'registration.json'), source_hashes_verified=True,
        parent_audit_sources_sha256=dict(recoverable=sha(base.__file__), geometric=sha(geometric.__file__)),
        scope='Read-only evidence and configuration audit; no controller step, rollout, new case or training executed',
        physical_constants_verified=constants, unit_saved_trace_audits=units,
        nominal_replay=dict(array_count=len(equal), every_array_exactly_equal=equal,
            old_and_new_trace_sha256=sha(old), completion_time_s=control['complete_time_s']),
        pulse_branch_scope=dict(saved_states_checked=pulse_states, all_pending_and_active_pulses_zero=True,
            explanation='The inherited /20 pulse fraction has no effect because its guarded pulse vector is zero. The active 14-dimensional encoder does not use the legacy rest/20 observation.'),
        stopping_hold_profiles_checked=stop_holds, stopping_physical_hold_s=.023,
        half_ms_cases=cases, comparisons=comparisons,
        half_ms_successes=sum(c['success'] for c in cases), runs=len(cases),
        actual_intervals=sum(c['actual_intervals'] for c in cases), branches=sum(c['branches'] for c in cases),
        branch_intervals=sum(c['branch_intervals'] for c in cases),
        stopping_assessments=sum(c['stopping_assessments'] for c in cases),
        stop_intervals=sum(c['stop_intervals'] for c in cases),
        audit_wall_s=time.perf_counter()-started,
        qualification='A numerical resolution check on three observed development paths; no training-seed variance, independent generalization or hardware timing claim.')
    assert not REPORT.exists()
    write_json(REPORT, report)
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ('half_ms_cases', 'physical_constants_verified', 'unit_saved_trace_audits')}), flush=True)


if __name__ == '__main__':
    main()
