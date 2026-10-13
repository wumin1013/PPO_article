"""Inspect nominal action-credit coverage before attempting a parameter update."""
import numpy as np
from search_complete_return import read
from study_io import BASE,register,receipt,write_json


def main():
    source=BASE/'single_decision_complete_credit_probe_20260917'
    assert (source/'receipt.json').exists()
    data=read(source/'results.json');sensitivity=read(source/'sensitivities.json')['records']
    stage=register('single_decision_credit_signal_audit_20260917',
        [__file__,source/'receipt.json',source/'results.json',source/'sensitivities.json'],
        dict(new_physics_ticks=0,new_training_decisions=0,wall_s=30),
        'Inspect completed single-decision counterfactual feasibility, action-return signal '
        'coverage and time quantization. Compare two objectives on identical successful '
        'pairs; keep failed pairs separate. This audit determines whether a bounded '
        'matched parameter-direction test is justified, not whether any learned policy '
        'has improved. No failed-prefix shape aggregation or development access.')
    comparisons=[]
    for q in data['queries']:
        for axis in range(3):
            pair=sorted([r for r in data['records'] if r['query']==q['query'] and r['axis']==axis],key=lambda r:r['direction'])
            a,b=pair
            complete=a['success'] and b['success']
            row=dict(query=q['query'],path=q['path'],anchor=q['anchor'],axis=axis,both_complete=complete)
            if complete:
                scale=b['latent_offset']-a['latent_offset']
                row.update(time_difference_ms=(b['full_time_s']-a['full_time_s'])*1000,
                    original_gradient=(b['original_full_return']-a['original_full_return'])/scale,
                    repaired_gradient=(b['independent_zero_return']-a['independent_zero_return'])/scale)
                if a['shape'] is not None:
                    row['shape_gradients']={k:(b['shape'][k]-a['shape'][k])/scale for k in (
                        'in_edge_outward_mm','out_edge_outward_mm','full_physical_reverse_deg','full_normalized_energy')}
            comparisons.append(row)
    by_axis=[]
    for axis in range(3):
        rows=[r for r in comparisons if r['axis']==axis]
        complete=[r for r in rows if r['both_complete']]
        by_axis.append(dict(axis=axis,queries=len(rows),complete_pairs=len(complete),
            same_complete_time_pairs=sum(abs(r['time_difference_ms'])<1e-8 for r in complete),
            nonzero_original_return_pairs=sum(abs(r['original_gradient'])>1e-7 for r in complete),
            nonzero_repaired_return_pairs=sum(abs(r['repaired_gradient'])>1e-7 for r in complete),
            opposite_objective_gradient_sign_pairs=sum(r['original_gradient']*r['repaired_gradient']<-1e-12 for r in complete)))
    failed=[{k:r[k] for k in ('query','path','anchor','axis','direction','reason')} for r in data['records'] if not r['success']]
    modes={}
    for r in sensitivity:
        for mode in r['objectives']['original_full_return']['modes']:modes[mode]=modes.get(mode,0)+1
    decision=dict(proceed_to_bounded_parameter_test=any(x['nonzero_repaired_return_pairs'] for x in by_axis),
        basis='Counterfactual returns provide nonzero complete-pair signal across the declared64-path pool; any candidate still requires full physical evaluation.',
        planned_comparison='Same20ms query data, support, parameterization and step opportunities; only the two explicitly defined shape objectives differ.')
    write_json(stage/'results.json',dict(axes=by_axis,failed_branches=failed,derivative_modes=modes,
        comparisons=comparisons,decision=decision,goal_complete=False))
    receipt(stage,dict(status='completed_single_decision_credit_signal_audit',new_physics_ticks=0,
        new_training_decisions=0,complete_pairs=sum(r['both_complete'] for r in comparisons),
        failed_branches=len(failed),proceed_to_parameter_test=decision['proceed_to_bounded_parameter_test']))
    print(dict(axes=by_axis,failed=failed,modes=modes),flush=True)


if __name__=='__main__':main()
