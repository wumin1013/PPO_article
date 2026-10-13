"""Current physical-geometry allowance from both-sign measured prediction errors."""
import time
import numpy as np
from search_complete_return import read
from study_io import BASE,register,receipt,write_json


def main():
    source=BASE/'canonical_gate_band_physical_cruise_endpoint_restore_20260917'
    probe=BASE/'canonical_gate_band_physical_cruise_composition_20260917'
    geometry=BASE/'canonical_gate_band_current_parameter_geometries_20260917'
    layers=BASE/'canonical_gate_band_cruise_prediction_layers_20260917'
    opt=read(source/'optimizer.json');physical=read(probe/'results.json')
    targets={(r['path'],r['metric']) for r in read(probe/'comparison.json')['regressions']}
    assert targets=={(13,'full_physical_peak_curvature_mm_inv')}
    assert read(geometry/'results.json')['parent_sha256']==opt['parent_sha256']==read(layers/'results.json')['parent_sha256']
    stage=register('canonical_gate_band_physical_cruise_error_margin_20260917',
        [__file__,source/'direction.npz',source/'optimizer.json',source/'receipt.json',
         probe/'results.json',probe/'comparison.json',probe/'receipt.json',
         geometry/'geometry.npz',geometry/'results.json',layers/'results.json',layers/'receipt.json'],
        dict(new_actor_decisions=0,new_physics_ticks=0,wall_s=30,subsequent_correction_attempts=1),
        'The both-sign current physical-geometry probe completed58/58 but path13 physical '
        'peak curvature regressed by2.67e-5/mm. Exact geometry of predicted raw states does '
        'not explain the error: the remaining state-response error dominates, and the '
        'positive peak stays at the same tick. Fit a quadratic metric-norm allowance ONLY '
        'for this observed path/metric, using both saved signs. Use the EXACT physical-arm '
        'metric. This is an empirical allowance, not a bound in a new direction. Permit '
        'one minimum-correction solve with identical shape, cruise and physical criteria, '
        'then repeat both-sign physical checks before any amplification. No old-model rates.')
    start=time.perf_counter()
    with np.load(source/'direction.npz') as z:
        d=z['direction'].ravel();names=z['metric_names'].tolist();ids=z['path_id'].tolist()
        predictions={float(s):v for s,v in zip(z['prediction_scales'],z['predictions'])}
    with np.load(geometry/'geometry.npz') as z:norm2=float(d@z['metric_physical']@d)
    baseline={r['path']:r['values'] for r in physical['baseline_metrics']}
    rates={p:0. for p in targets};observations=[]
    for c in physical['candidates']:
        assert c['summary']['successes']==29
        actual={r['path']:r['values'] for r in c['metrics']};step=c['step']
        for p,n in sorted(targets):
            k=names.index(n);pred=float(predictions[step][ids.index(p),k])
            change=actual[p][k]-baseline[p][k];error=change-pred
            rate=max(0.,error)/(norm2*step**2);rates[p,n]=max(rates[p,n],rate)
            observations.append(dict(path=p,metric=n,scale=step,actual_change=change,
                predicted_change=pred,positive_error=max(0.,error),rate=rate))
    result=dict(parent_sha256=opt['parent_sha256'],parameter_geometry=opt['parameter_geometry'],
        observations=observations,coefficients=[dict(path=p,metric=n,rate=r) for (p,n),r in rates.items()],
        guaranteed_bound=False,new_direction_requires_validation=True,goal_complete=False)
    write_json(stage/'error_model.json',result)
    receipt(stage,dict(status='prepared_current_physical_geometry_cruise_margin',new_actor_decisions=0,
        new_physics_ticks=0,wall_s=time.perf_counter()-start))
    print('CRUISE_ERROR_MARGIN',result['coefficients'],flush=True)


if __name__=='__main__':main()
