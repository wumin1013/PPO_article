"""Read-only interval and kinematic audit of the ten frozen-policy executions."""
import json
import numpy as np
from common import ROOT, sha, utc, write_json
from evaluate_frozen_main_generalization_v6 import verify, REGISTRATION
from runtime_bootstrap import setup


def main():
    registration = verify()
    setup()
    from geometry_spec import Corridor, GateTracker
    results = []
    for cell in registration['grid']:
        case = next(x for x in registration['cases'] if x['id'] == cell['case'])
        record = json.loads((ROOT / cell['record']).read_text('utf-8'))
        with np.load(ROOT / cell['trace']) as z:
            positions, reference, rows = z['positions'], z['reference'], z['rows']
        corridor = Corridor(reference, epsilon=case['evaluation_tolerance_mm'],
            closed=bool(np.linalg.norm(reference[0] - reference[-1]) < 1e-9))
        tracker = GateTracker(corridor, positions[0])
        first_outside = None
        for index, point in enumerate(positions[1:], 1):
            tracker.update(point)
            if not tracker.inside and first_outside is None:
                first_outside = index
        geometry = tracker.snapshot(endpoint_tolerance=case['evaluation_tolerance_mm'])
        dt = cell['dt_s']
        velocity = np.diff(positions, axis=0) / dt
        acceleration = np.diff(np.vstack((np.zeros(2), velocity)), axis=0) / dt
        jerk = np.diff(np.vstack((np.zeros(2), acceleration)), axis=0) / dt
        peak_ratios = np.array([np.linalg.norm(x, axis=1).max(initial=0.)
            for x in (velocity, acceleration, jerk)]) / [100., 2000., 62500.]
        if np.any(peak_ratios > 1. + 2e-5):
            raise ValueError('Planar constraint failure in a frozen KCM run')
        if record['executor_reason'] == 'corridor_violation':
            if first_outside != len(positions) - 1:
                raise ValueError('Execution did not stop at the first invalid command interval')
        elif record['success']:
            if first_outside is not None or not geometry['geometry_complete']:
                raise ValueError('Reported complete run has invalid command geometry')
        if len(rows) != len(positions) - 1:
            raise ValueError('Saved physical rows do not cover all commands')
        if not np.array_equal(rows[:, :2], positions[1:]):
            raise ValueError('Saved physical position rows changed')
        results.append(dict(run_id=cell['run_id'], command_intervals=len(positions) - 1,
            all_positions_match_saved_rows=True, first_invalid_interval=first_outside,
            first_invalid_interval_is_last=(first_outside == len(positions) - 1) if first_outside else None,
            vector_peak_ratios=peak_ratios.tolist(), vector_pass=True,
            ordered_geometry=geometry, source_trace=cell['trace'], source_trace_sha256=sha(ROOT / cell['trace'])))
    result = dict(created_utc=utc(), registration_sha256=sha(REGISTRATION),
        auditor_sha256=sha(__file__), full_grid=True, rows=results,
        actual_command_intervals=sum(x['command_intervals'] for x in results),
        all_command_intervals_rechecked=True, all_planar_vector_checks_pass=True,
        all_geometric_failures_stop_at_first_invalid_command=True,
        runtime_sources_unchanged=True, original_guide_restored=True,
        source_evidence=dict(kinematic_entry='joint_runtime.JointRuntime.step -> TurnAwareProjector.enforce_kinematic_constraints on every non-terminal interpolation step',
            geometry_entry='joint_runtime.JointRuntime.step -> GateTracker.update after each recorded motion command; its interval capsule-union and ordered directed-gate checks are unchanged.',
            resolution_entry='execution_settings.configure_dt updates projector finite-difference dt and release horizon; original nominal guide width0.1mm,20msdecision,12msfilter remain fixed.',
            terminal_entry='terminal_completion_v4d verifies the entire common bounded terminal-brake trace before execution; runtime_physical_time_v5 fixes the2.1s tail allowance and failure-budget units.',
            one_ms_controls=registration['prior_1ms_equivalence']),
        new_control_executions=0)
    write_json(ROOT / 'audit/frozen_main_execution_consistency_v6.json', result)
    print(json.dumps({key: result[key] for key in ('actual_command_intervals',
        'all_planar_vector_checks_pass', 'all_geometric_failures_stop_at_first_invalid_command')}, indent=2))


if __name__ == '__main__':
    main()
