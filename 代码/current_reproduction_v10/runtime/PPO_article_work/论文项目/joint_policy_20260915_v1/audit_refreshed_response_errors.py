"""Attribute saved local-model errors to unchanged measured costs and guards."""
import numpy as np
from audit_coupled_trust_model import labels
from head_counterfactual_rollouts import cost
from learn_information_refreshed_joint import SOURCE
from learn_verified_trust_joint import positive_sum
from study_io import BASE, write_json, sha
from search_complete_return import read


STAGE = BASE/'canonical_gate_band_information_refreshed_joint_learning_20260920'


def main():
    history = read(STAGE/'state_joint/results.json')['candidates']
    by = {c['tag']: c for c in history}
    old = read(SOURCE/'state_joint/results.json')['candidates']
    normalizer = next(c['training_score'] for c in old if c['tag']=='protected_reference')
    data = read(STAGE/'baseline.json')
    baseline = {r['path']: r for r in data['baseline']['records']}
    retained = {r['path']: r['baseline_shape'] for r in data['retained'] if 'baseline_shape' in r}
    with np.load(STAGE/'current_joint_response.npz') as z:
        fresh, stale = z['fresh'], z['stale']
    rows = []
    for block in read(STAGE/'learning_updates.json'):
        if 'measured' not in block:
            continue
        current = by[block.get('anchor','reused_current_actor')]
        tested = by[block['measured']]
        a = np.array(current['subspace_coordinates'])
        values = np.array(current['augmented_learning_inequalities'])
        names = labels(current['records'])
        if block['arm'] in ('stale','fresh'):
            model = stale if block['arm']=='stale' else fresh
        else:
            available = history[:next(i for i,c in enumerate(history) if c['tag']==tested['tag'])]
            near = [c for c in available if c['physical_pass'] and c['tag']!=current['tag'] and
                    np.max(abs(np.array(c['subspace_coordinates'])-a))<=4.]
            x = np.array([np.array(c['subspace_coordinates'])-a for c in near])
            y = np.array([np.r_[np.array(c['augmented_learning_inequalities'])-values,
                              c['training_score']-current['training_score']] for c in near])
            weight = 1./np.maximum(np.linalg.norm(x,axis=1),.25)
            model = np.linalg.solve((x*weight[:,None]).T@x+.01*np.eye(len(a)),
                                    (x*weight[:,None]).T@y+.01*fresh)
        delta = np.array(tested['subspace_coordinates'])-a
        predicted_guards = values+delta@model[:,:-1]
        actual_guards = np.array(tested['augmented_learning_inequalities'])
        predicted_cost_change = float(delta@model[:,-1]/normalizer)
        actual_cost_change = (tested['training_score']-current['training_score'])/normalizer
        predicted_guard_change = .1*(np.maximum(-predicted_guards,0)-np.maximum(-values,0))
        actual_guard_change = .1*(np.maximum(-actual_guards,0)-np.maximum(-values,0))
        assert abs(-predicted_cost_change-predicted_guard_change.sum()-block['predicted_drop'])<1e-8
        assert abs(-actual_cost_change-actual_guard_change.sum()-block['actual_drop'])<1e-8
        guard_error = actual_guard_change-predicted_guard_change
        top = [dict(channel=names[i],error=float(guard_error[i]),
                    predicted_change=float(predicted_guard_change[i]),actual_change=float(actual_guard_change[i]),
                    before=float(values[i]),after=float(actual_guards[i]))
               for i in np.argsort(abs(guard_error))[-12:][::-1]]
        path_rows = []
        for b,r in zip(current['records'],tested['records']):
            if 'shape' not in r:
                continue
            def components(q):
                s=q['shape'];o=retained[q['path']];p=baseline[q['path']]
                return dict(reverse=(s['full_physical_reverse_deg']/.5)**2,
                    exit=(s['out_edge_outward_mm']/.01)**2,
                    energy=(s['full_normalized_energy']/.1)**2,
                    negative_lateral=5*(s['reverse_lateral_total_mm']/.005)**2,
                    negative_duration=(s['reverse_lateral_significant_duration_s']/.05)**2,
                    time_regret=10*max((q['complete_time_s']-p['complete_time_s']-.01)/.02,0.)**2,
                    entry=(s['in_edge_outward_mm']/.002)**2,
                    exit_area=(q['duration_audit']['exit_area_mm_s']/.002)**2)
            bc,rc=components(b),components(r)
            change=(cost(r,baseline[r['path']],retained[r['path']])-cost(b,baseline[b['path']],retained[b['path']]))/normalizer
            path_rows.append(dict(path=r['path'],cost_change=change,
                components_change={k:(rc[k]-bc[k])/normalizer for k in rc},
                time_change_s=r['complete_time_s']-b['complete_time_s'],
                before_trace=b['trace'],after_trace=r['trace']))
        out=dict(tag=tested['tag'],anchor=current['tag'],step_linf=float(max(abs(delta))),
            predicted_cost_change=predicted_cost_change,actual_cost_change=actual_cost_change,
            guard_prediction_error=float(guard_error.sum()),cost_prediction_error=actual_cost_change-predicted_cost_change,
            top_guard_errors=top,paths=sorted(path_rows,key=lambda r:abs(r['cost_change']),reverse=True))
        rows.append(out)
        print('ERROR_CHANNELS',tested['tag'],'cost',out['cost_prediction_error'],'guards',out['guard_prediction_error'],
              'top',top[:4],'paths',[(r['path'],r['cost_change'],r['time_change_s']) for r in out['paths'][:4]],flush=True)
    write_json(STAGE/'response_error_channels.json',dict(normalizer=normalizer,updates=rows,
        results_sha256=sha(STAGE/'state_joint/results.json'),new_physical_runs=0,
        scope='Exact replay and decomposition; causal source of trajectory changes is not established by this accounting.'))


if __name__=='__main__':
    main()
