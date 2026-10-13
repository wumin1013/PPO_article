"""Read existing true-XY traces without repeating completed physical evaluations."""
import json
import time
import numpy as np
from isolated_corner_shape import corner_shape
from study_io import BASE, register, receipt, sha, write_json


def main():
    started = time.perf_counter()
    bindings = []
    for label, stage, pattern in [
        ('conditional_joint', 'ar_conditional_joint_20260916/final_validation', '*.npz'),
        ('conditional_fixed3', 'ar_conditional_fixed_20260916/final_validation', '*.npz'),
        ('probe_2deg_joint', 'coordinated_suffix_probe_20260916', 'path*_111.npz'),
        ('probe_8deg_joint', 'inner_approach_amplitude_probe_20260916', '*.npz')]:
        bindings.extend((label, p) for p in sorted((BASE/stage).glob(pattern)))
    stage = register('reverse_preswing_xy_audit_20260916',
        [__file__, BASE/'isolated_corner_shape.py']+[p for _, p in bindings],
        dict(trace_loads=len(bindings), physics_ticks=0, model_calls=0, training_updates=0, wall_s=60),
        'Audit first physical turn, outward displacement/duration, heading and curvature from saved XY; include prior interventions, without equating offset medians with shape success.')
    results = []
    for label, path in bindings:
        with np.load(path) as data:
            if len(data['reference']) != 3:
                continue
            row = dict(label=label, file=str(path), sha256=sha(path),
                       **corner_shape(data['reference'], data['rows'], data['actions']))
            # Frame invariance is necessary for consistent left/right reporting.
            mirror = np.diag([1., -1.])
            mirrored_rows = data['rows'].copy()
            for offset in (0, 2, 4, 6, 8):
                mirrored_rows[:, offset:offset+2] = mirrored_rows[:, offset:offset+2]@mirror
            mirrored_actions = data['actions'].copy(); mirrored_actions[:, 0] *= -1
            check = corner_shape(data['reference']@mirror, mirrored_rows, mirrored_actions)
            for key, value in check.items():
                if key == 'turn_deg':
                    assert abs(value+row[key]) < 1e-8
                elif isinstance(value, float):
                    assert abs(value-row[key]) < 1e-8, key
                else:
                    assert value == row[key], key
            results.append(row)
            print(label, path.stem, round(row['turn_deg'], 1), row['first_physical_turn'],
                  round(row['outward_excursion_mm'], 4), round(row['outward_position_duration_s'], 3),
                  round(row['reverse_turning_total_deg'], 2), row['shape_screen_pass'])
    write_json(stage/'results.json', dict(results=results,
        onset_definition='Physical velocity heading exceeds +/-0.5 degree for 20 consecutive 1 ms samples, speed >5 mm/s. Signed normal points into the turn; pre-apex is before closest original vertex approach.',
        screen_definition='Diagnostic only: inward first sustained heading and outward excursion <=0.05 mm. Does not certify smoothness, complete task, or generalization; actual XY/heading/curvature remain required.',
        goal_complete=False))
    receipt(stage, dict(status='completed_saved_true_xy_shape_audit', trace_loads=len(bindings),
        isolated_traces=len(results), mirror_checks=len(results), physics_ticks=0, model_calls=0,
        training_updates=0, wall_s=time.perf_counter()-started))


if __name__ == '__main__':
    main()
