"""Check model exclusion of measured feasible policies, without physical reruns."""
import numpy as np
from learn_interaction_return_credit import response_features
from audit_early_geometry_outcomes import constraint_names
from study_io import BASE, write_json
from search_complete_return import read


def main():
    stage = BASE/'canonical_gate_band_interaction_return_credit_learning_20260920'
    history = read(stage/'state_joint/results.json')['candidates']
    models = read(stage/'matched_credit_models.json')
    design = read(stage/'factor_design.json')
    baseline = np.r_[history[0]['score'],history[0]['augmented_learning_inequalities']]
    lookup = {c['tag']:c for c in history}
    names = constraint_names([r['path'] for r in history[0]['records']])
    rows = []
    for model in models['models']:
        predicted = baseline+response_features(np.array(design['training']),model['interactions'])@np.array(model['coefficients'])
        actual = [lookup[t] for t in models['training_tags']]
        misses = [dict(tag=c['tag'],predicted_worst=float(v[1:].min()),
            wrong_constraints=[dict(name=names[i],margin=float(v[i+1])) for i in np.flatnonzero(v[1:] < -1e-8)])
            for c,v in zip(actual,predicted) if min(c['augmented_measured_inequalities'])>=-1e-8 and v[1:].min() < -1e-8]
        row = dict(model=model['name'],known_feasible_excluded=misses)
        rows.append(row)
        print('KNOWN_FEASIBLE_CREDIT',row,flush=True)
    write_json(stage/'known_feasible_credit_audit.json',dict(models=rows,new_physical_runs=0))
    write_json(stage/'selection_hold.json',dict(
        reason='No new feasible joint candidate or policy update; best is the reused feed-only control. Prediction accuracy alone is not learning success.',
        formal_adoption=False,goal_complete=False))


if __name__ == '__main__':
    main()
