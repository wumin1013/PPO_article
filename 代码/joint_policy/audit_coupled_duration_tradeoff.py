"""Re-score saved joint responses with omitted duration/speed measures; no new rollout."""
import copy
import numpy as np
from duration_joint_metrics import augment, guards, guard_names
from localize_joint_shape_residual import characterize
from study_io import BASE, write_json, sha
from search_complete_return import read


def main():
    stage = BASE / 'canonical_gate_band_coupled_peak_memory_learning_20260920'
    result = read(stage / 'state_joint/results.json')
    reference = next(r for r in read(stage / 'baseline.json')['baseline']['records'] if r['path'] == 304)
    augment(reference)
    records = []
    for c in result['candidates']:
        r = c['records'][0]
        extra = guards(r, reference)
        old_pass = min(c['measured_inequalities']) >= -1e-8
        records.append(dict(tag=c['tag'],old_guard_pass=old_pass,
            expanded_guard_pass=bool(old_pass and extra.min() >= -1e-8),
            score=c['score'],duration_audit=r.get('duration_audit'),
            extra_guards=extra.tolist(),violations=[k for k,v in zip(guard_names(),extra) if v < -1e-8]))
    selected = next(c for c in result['candidates'] if c['tag'] == result['best'])
    audit = dict(model_sha256=sha(stage/'research_candidate.pt'),baseline=reference,
        selected=selected['tag'],selected_record=augment(copy.deepcopy(selected['records'][0])),
        baseline_localization=characterize(reference['trace']),
        selected_localization=characterize(selected['records'][0]['trace']),
        all_vectors=len(records),new_rollouts=0,guard_names=guard_names(),records=records,
        old_selection_is_not_full_shape_acceptance=True,final_adopted=False)
    good = [c for c in records if c['expanded_guard_pass']]
    audit['best_expanded_guard_candidate'] = min(good,key=lambda c:c['score']) if good else None
    write_json(stage/'duration_tradeoff_audit.json',audit)
    print('DURATION_TRADEOFF',next(c for c in records if c['tag']==selected['tag']),flush=True)
    print('EXPANDED_GUARDS',len(good),'best',audit['best_expanded_guard_candidate'],flush=True)


if __name__ == '__main__':
    main()
