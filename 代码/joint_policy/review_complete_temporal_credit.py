"""Validate temporal-credit completeness on existing whole-policy responses."""
import numpy as np
from study_io import BASE,write_json,sha
from search_complete_return import read


def main():
    old=BASE/'canonical_gate_band_deterministic_local_credit_20260920'
    full=BASE/'canonical_gate_band_deterministic_full_support_credit_20260920'
    sources=[BASE/n for n in ('canonical_gate_band_deterministic_local_credit_20260920',
        'canonical_gate_band_minimum_norm_complete_credit_20260920','canonical_gate_band_joint_secant_credit_20260920',
        'canonical_gate_band_shape_minimax_training_branch_20260920','canonical_gate_band_smooth_minimax_training_branch_20260920')]
    references=[dict(c,source_stage=s.name) for s in sources for c in read(s/'state_joint/results.json')['candidates']
        if c['tag']!='reused_parent']
    parent=read(old/'state_joint/results.json')['candidates'][0]
    comparisons=[]
    for mode in ('geometry','dynamic'):
        points=[c for c in references if c['context_arm']==mode]
        u=np.array([c['coefficients'] for c in points])
        actual_cost=np.array([c['score']-parent['score'] for c in points])
        actual_learning=np.array([c['augmented_learning_inequalities'] for c in points])
        actual_measured=np.array([c['augmented_measured_inequalities'] for c in points])
        scores={}
        for label,stage in (('sampled_tail',old),('complete_support',full)):
            with np.load(stage/(mode+'_complete_gradient.npz')) as z:
                g,jac,b=[z[k].copy() for k in ('loss_gradient','constraint_jacobian','baseline_margins')]
            predicted_cost=u@g;predicted=b+u@jac.T
            active=abs(actual_learning-b)>1e-7
            actual_bad=actual_measured < -1e-8;predicted_bad=predicted < -1e-8
            scores[label]=dict(cost_rmse=float(np.sqrt(np.mean((predicted_cost-actual_cost)**2))),
                active_constraint_rmse=float(np.sqrt(np.mean((predicted[active]-actual_learning[active])**2))),
                false_safe_policies=int(np.sum(~np.any(predicted_bad,axis=1)&np.any(actual_bad,axis=1))),
                actual_bad_channels=int(np.sum(actual_bad)),
                detected_bad_channels=int(np.sum(actual_bad&predicted_bad)),
                false_alarm_channels=int(np.sum(~actual_bad&predicted_bad)),
                points=[dict(source=c['source_stage'],tag=c['tag'],predicted_loss_change=float(pc),
                    actual_loss_change=float(ac)) for c,pc,ac in zip(points,predicted_cost,actual_cost)])
        comparisons.append(dict(context=mode,policies=len(points),scores=scores))
        print('TEMPORAL_CREDIT_MODEL',mode,{k:{x:y for x,y in v.items() if x!='points'} for k,v in scores.items()},flush=True)
    old_episodes={e['path']:e for e in read(old/'deterministic_episodes.json')['episodes']}
    tail=[]
    for e in read(full/'deterministic_episodes.json')['episodes']:
        before=old_episodes[e['path']]
        dense={s['decision'] for s in before['samples'] if s['query_weight']==1}
        sampled=next(s for s in before['samples'] if s['query_weight']>1)
        by={(b['decision'],b['head'],b['radius'],b['sign']):b for b in e['branches']}
        all_gradient=np.zeros((18,3));tail_gradient=np.zeros_like(all_gradient);estimate=np.zeros_like(all_gradient)
        for s in e['samples']:
            row=np.zeros_like(all_gradient)
            for h in range(3):
                minus,plus=[by[s['decision'],h,.1,sign] for sign in (-1,1)]
                row[:,h]=np.array(s['design'])[:,h]*(plus['score']-minus['score'])/(2*plus['epsilon'])
            all_gradient+=row
            if s['decision'] not in dense:tail_gradient+=row
            if s['decision']==sampled['decision']:estimate=sampled['query_weight']*row
        r=dict(path=e['path'],tail_estimation_error_norm=float(np.linalg.norm(estimate-tail_gradient)),
            error_relative_to_complete_gradient=float(np.linalg.norm(estimate-tail_gradient)/max(np.linalg.norm(all_gradient),1e-12)),
            tail_direction_cosine=float(np.sum(estimate*tail_gradient)/max(np.linalg.norm(estimate)*np.linalg.norm(tail_gradient),1e-12)))
        tail.append(r);print('MEASURED_TAIL_ERROR',r,flush=True)
    write_json(full/'temporal_credit_completeness_audit.json',dict(comparisons=comparisons,tail=tail,
        new_physical_runs=0,receipt_sha256=sha(full/'receipt.json'),
        same_training_paths_not_an_independent_generalization_test=True,
        complete_support_removes_temporal_subsampling_only=True))


if __name__=='__main__':main()
