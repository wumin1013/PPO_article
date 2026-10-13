"""Compare retained and current learned policies on identical saved training paths."""
import argparse
import time
import numpy as np
from complete_corner_metrics import complete_corner_metrics
from complete_motion_metrics import complete_motion_metrics
from audit_xy_curvature_resolution import curvature_metrics
from validate_actions import difference_audit
from search_complete_return import read
from study_io import BASE, register, receipt, write_json, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--stage', required=True)
    args = parser.parse_args()
    retained = BASE/'margin_constrained_function_credit_learning_20260917'
    current = BASE/args.candidate
    a = read(retained/'training_selection.json')
    b = read(current/'results.json')
    groups = [('retained', retained/a['model'], a),
              ('current_parent', BASE/'canonical_gate_band_matched_probe_20260917/parent.pt', b['full_parent']),
              ('current_candidate', current/b['chosen']['model'], b['full_candidate'])]
    sources = [__file__, retained/'training_selection.json', retained/'receipt.json',
               current/'results.json', current/'receipt.json'] + [m for _, m, _ in groups]
    stage = register(args.stage, sources, dict(new_physics_ticks=0, new_actor_calls=0, wall_s=60),
        'Recompute original1ms trajectory metrics for all64 identical training paths. '
        'Retain the strong historical model as an explicit comparator. No new rollout, '
        'development or sealed access. Aggregate isolated-corner shape only over16 '
        'isolated paths; mixed-path completion and cruise are evaluated separately. '
        'A current-parent gain does not establish superiority over the retained model.')
    start = time.perf_counter()
    references = {}; records = []; summaries = []
    for label, model, selected in groups:
        values = []
        assert len(selected['results']) == len(selected['traces']) == 64
        for result, trace in zip(selected['results'], selected['traces']):
            path = result['original_path']
            assert result['success']
            with np.load(BASE/trace) as z:
                ref, rows = z['reference'], z['rows']
                if label == 'retained': references[path] = ref.copy()
                else: assert np.array_equal(ref, references[path])
                audit = difference_audit(ref[0], rows)
                assert audit['kinematic_pass'] and audit['rest_pass']
                r = dict(path=path, trace=trace, label=label, motion=complete_motion_metrics(ref, rows),
                         audit=audit, shape=None, curvature=None)
                if len(ref) == 3:
                    r['shape'] = complete_corner_metrics(ref, rows)
                    r['curvature'] = curvature_metrics(ref, rows)
                    u, v = np.diff(ref, axis=0); u /= np.linalg.norm(u)
                    sign = np.sign(u[0]*v[1]-u[1]*v[0]); n = sign*np.array([-u[1], u[0]])
                    speed = np.linalg.norm(rows[:, 2:4], axis=1)
                    heading = np.rad2deg(np.unwrap(np.arctan2(rows[:, 2:4]@n, rows[:, 2:4]@u)))
                    r['onset'] = []
                    for threshold in (.05, .5):
                        ids = np.flatnonzero((speed > 5.) & (abs(heading) > threshold))
                        r['onset'].append(dict(threshold_deg=threshold,
                            first_direction=None if not len(ids) else ('inside' if heading[ids[0]]>0 else 'outside'),
                            first_time_s=None if not len(ids) else float((ids[0]+1)*.001)))
                values.append(r)
        corners = [r for r in values if r['shape'] is not None]
        summary = dict(label=label, model=str(model.relative_to(BASE)), model_sha256=sha(model),
            complete_paths=len(values), isolated_corners=len(corners),
            full_time_s=sum(r['motion']['recorded_time_s'] for r in values),
            cruise_ge99_s=sum(r['motion']['cruise_ge99_s'] for r in values),
            entry_sum_mm=sum(r['shape']['in_edge_outward_mm'] for r in corners),
            exit_sum_mm=sum(r['shape']['out_edge_outward_mm'] for r in corners),
            reverse_sum_deg=sum(r['shape']['full_physical_reverse_deg'] for r in corners))
        summaries.append(summary); records.extend(values); print('SAVED_COMPARISON', summary, flush=True)
    write_json(stage/'results.json', dict(summaries=summaries, records=records,
        identical_reference_geometry=True, goal_complete=False))
    receipt(stage, dict(status='compared_saved_complete_training_policies', new_physics_ticks=0,
        new_actor_calls=0, wall_s=time.perf_counter()-start, goal_complete=False))


if __name__ == '__main__': main()
