"""Read shared joint-response evidence without repeating physical experiments."""
import argparse
import numpy as np
import torch
from audit_early_geometry_outcomes import constraint_names
from localize_joint_shape_residual import characterize
from joint_teacher_state_policy import features
from reflection_policy import reflect_observation
from study_io import BASE, write_json, sha
from search_complete_return import read


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True)
    parser.add_argument('--replay')
    args = parser.parse_args()
    stage = BASE/args.stage
    history = read(stage/'state_joint/results.json')
    candidates = history['candidates']
    baseline = candidates[0]
    paths = [r['path'] for r in baseline['records']]
    names = constraint_names(paths)
    def describe(c):
        values = c['augmented_measured_inequalities']
        assert len(values)==len(names)
        violations = sorted([dict(name=k,margin=v) for k,v in zip(names,values) if v < -1e-8],
                            key=lambda r:r['margin'])
        records = []
        for r in c['records']:
            if not r['success']:
                records.append(dict(path=r['path'],success=False))
                continue
            s = r['shape']
            records.append(dict(path=r['path'],success=True,exit_mm=s['out_edge_outward_mm'],
                reverse_deg=s['full_physical_reverse_deg'],lateral_mm=s['reverse_lateral_significant_mm'],
                lateral_duration_ms=round(1000*s['reverse_lateral_significant_duration_s']),
                curve=s['full_spatial_peak_curvature_mm_inv'],apex_speed=s['apex_speed_mm_s'],
                cruise99_s=r['motion']['cruise_ge99_s'],time_s=r['complete_time_s']))
        return dict(tag=c['tag'],score=c['score'],violations=violations,records=records)
    beneficial = [c for c in candidates if c['score']<baseline['score']-.001]
    nearest = sorted(beneficial,key=lambda c:(max(0.,-min(c['augmented_measured_inequalities'])),c['score']))[:3]
    chosen = next(c for c in candidates if c['tag']==history['best'])
    md = torch.load(stage/'initial_policy.pt',map_location='cpu',weights_only=False)
    with np.load(BASE/args.replay if args.replay else stage/'affected_replay.npz') as z:
        observations, innovations = z['observations'], z['innovations']
    f = features(md,observations,innovations)
    mirror = features(md,reflect_observation(torch.as_tensor(observations)).numpy(),-innovations)
    expected = f*np.array([-1.,1.,1.])
    assert np.array_equal(mirror,expected), 'Shared feature reflection must hold exactly'
    model_file = stage/'measured_joint_credit.json'
    diagnostics = []
    if model_file.exists():
        model = read(model_file)
        jac, gradient, steps = [np.array(model[k]) for k in ('jacobian','loss_gradient','steps')]
        sampled = {tag for pair in model['response_pairs'] for tag in pair}
        for c in candidates[1:]:
            if c['tag'] in sampled:
                continue
            delta = (np.array(c['coefficients'])-baseline['coefficients'])/steps
            predicted = np.array(baseline['augmented_learning_inequalities'])+jac@delta
            actual = np.array(c['augmented_learning_inequalities'])
            error = actual-predicted
            worst = int(np.argmax(abs(error)))
            diagnostics.append(dict(tag=c['tag'],predicted_loss_change=float(gradient@delta),
                actual_loss_change=c['score']-baseline['score'],largest_prediction_error_name=names[worst],
                largest_prediction_error=float(error[worst]),actual_worst_margin=float(actual.min())))
    local = [dict(path=r['path'],**characterize(r['trace'])) for r in baseline['records']]
    out = dict(receipt_sha256=sha(stage/'receipt.json') if (stage/'receipt.json').exists() else None,
        stored_points=len(candidates),baseline=describe(baseline),selected=describe(chosen),
        nearest_beneficial=[describe(c) for c in nearest],response_prediction_errors=diagnostics,
        baseline_local_shape=local,shared_feature_reflection_bitwise=True,new_physical_runs=0,
        final_adopted=False,goal_complete=False)
    write_json(stage/'shared_credit_review.json',out)
    print('SHARED_CREDIT_REVIEW',len(candidates),'selected',chosen['tag'],'feasible',
          sum(min(c['augmented_measured_inequalities'])>=-1e-8 for c in candidates),flush=True)
    for c in nearest:
        d = describe(c)
        print('NEAREST',d['tag'],d['score'],d['violations'][:6],flush=True)
    for row in diagnostics:
        print('MODEL_ERROR',row,flush=True)


if __name__ == '__main__':
    main()
