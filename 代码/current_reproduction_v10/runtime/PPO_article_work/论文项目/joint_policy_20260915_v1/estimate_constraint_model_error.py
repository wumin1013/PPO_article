"""Observed one-sided credit errors for an explicit curvature trust margin."""
import argparse
import time
import numpy as np
from audit_xy_curvature_resolution import curvature_metrics, CURVATURE_NAMES
from search_complete_return import read
from study_io import BASE, register, receipt, write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',default='raw_xy_physical_constraint_credit_20260917')
    parser.add_argument('--stage',default='empirical_curvature_margin_credit_20260917')
    parser.add_argument('--include-fourth',action='store_true')
    parser.add_argument('--shape-margin',action='store_true')
    args=parser.parse_args()
    source=BASE/args.source
    full_space=read(source/'results.json').get('parameter_space')=='full_feedback_with_geometry'
    preparation=BASE/('full_feedback_parameter_space_preparation_20260917' if full_space else 'direct_constraint_parameter_preparation_20260917')
    checks=['direct_constraint_composition_probe_20260917','amplitude_secant_composition_probe_20260917',
            'raw_xy_physical_composition_probe_20260917']
    proposals=['direct_full_subspace_constraint_solve_20260917','amplitude_secant_constraint_solve_20260917',
               'raw_xy_physical_constraint_solve_20260917']
    if args.include_fourth:
        checks+=['empirical_curvature_margin_composition_probe_20260917']
        proposals+=['empirical_curvature_margin_feasible_proposal_20260917']
    stage=register(args.stage,
        [__file__,source/'gradients.npz',source/'results.json',source/'receipt.json',
         preparation/'support.npz',BASE/'audit_xy_curvature_resolution.py']+
         ([source/'baselines.json'] if full_space else [])+
         [BASE/p/f for p in checks for f in ('results.json','receipt.json')]+
         [BASE/p/'direction.npz' for p in proposals],
        dict(new_physics_ticks=0,new_actor_decisions=0,wall_s=60,maximum_metric_step_norm=1.),
        ('Extend the SAME error-margin computation to each corner exit and physical-reverse constraint, using the SAME four measured directions. No new data or gradient-model change. Zero-entry constraints remain unchanged because a positive floor margin on an already zero excursion would demand a negative physical excursion. ' if args.shape_margin else '')+
        'Keep exactly the same three raw-XY/physical credit models. Estimate each '
        'smoothness constraint\'s worst observed one-sided underprediction per unit '
        'action-metric step from the explicitly listed completed positive/negative full-policy '
        'checks. Use that observed error as a fixed conservative margin for radius1 '
        'in the next mathematical solve, then require its actual metric norm<=1. '
        'This addresses previous candidates lying on predicted zero-change constraints. '
        'The observed error is an empirical estimate, NOT a proved error bound outside '
        'measured directions; actual complete validation remains mandatory. No actor, '
        'objective, runtime, physical boundary or existing gradient is changed here.')
    started=time.perf_counter()
    with np.load(source/'gradients.npz') as z:saved={k:z[k].copy() for k in z.files}
    ids=saved['path_id'].tolist();names=saved['metric_names'].tolist();models=saved['gradient_variants']
    smooth=[i for i,name in enumerate(names) if 'energy' in name or 'peak_curvature' in name]
    if args.shape_margin:smooth+=[names.index('out_edge_outward_mm'),names.index('full_physical_reverse_deg')]
    with np.load(preparation/'support.npz') as z:metric=z['extended_metric' if full_space else 'metric'].copy()
    nominal=read(source/'results.json')['nominal_curvature']
    error=np.zeros(models.shape[:3]);observed=[]
    for check,proposal in zip(checks,proposals):
        result=read(BASE/check/'results.json')
        with np.load(BASE/proposal/'direction.npz') as z:
            d=np.r_[z['direction'].ravel(),0.] if full_space else z['reduced_direction'].copy()
        norm=float(np.sqrt(d@metric@d));assert norm>0
        old_names=result['metric_names']
        baselines={r['path']:r['values'] for r in result['baseline_metrics']}
        for j in range(len(result['candidates'][0]['metrics'])):
            path=result['candidates'][0]['metrics'][j]['path'];i=ids.index(path)
            if path==40:continue
            def convert(values,curv):
                row=dict(zip(old_names,values));row.update(curv)
                return np.asarray([row[k] for k in names])
            base=convert(baselines[path],nominal[str(path)])
            for candidate in result['candidates']:
                with np.load(BASE/candidate['traces'][j]) as z:c=curvature_metrics(z['reference'],z['rows'])
                value=convert(candidate['metrics'][j]['values'],c)
                direction_sign=np.sign(candidate['coefficient'])
                # Normalize actual positive and negative parameter changes separately.
                actual=(value-base)/abs(candidate['coefficient'])
                predicted=np.einsum('kmd,d->km',models[:,i],direction_sign*d)
                mismatch=(actual[None]-predicted)/norm
                error[:,i,smooth]=np.maximum(error[:,i,smooth],mismatch[:,smooth])
                observed.append(dict(check=check,path=path,coefficient=candidate['coefficient'],
                    norm_per_unit=norm,largest_smoothness_underprediction=float(max(0.,mismatch[:,smooth].max()))))
    saved['constraint_uncertainty']=error
    saved['uncertainty_radius']=np.asarray(1.)
    np.savez_compressed(stage/'gradients.npz',**saved)
    if full_space:write_json(stage/'baselines.json',read(source/'baselines.json'))
    write_json(stage/'results.json',dict(metric_names=names,observed=observed,
        constraint_uncertainty=error.tolist(),uncertainty_metric_names=[names[i] for i in smooth],
        radius=1.,gradient_models_unchanged=True,
        estimated_from_direction_count=len(checks),parameter_space='full_feedback_with_geometry' if full_space else 'protected_residual',
        goal_complete=False))
    receipt(stage,dict(status='completed_empirical_curvature_margin_credit',
        new_actor_decisions=0,new_physics_ticks=0,wall_s=time.perf_counter()-started))
    print('CURVATURE_ERROR_MARGIN','max',error.max(),'nonzero',np.count_nonzero(error),flush=True)


if __name__=='__main__':main()
