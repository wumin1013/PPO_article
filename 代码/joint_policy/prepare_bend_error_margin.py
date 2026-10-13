"""Current-center empirical allowances for only the observed shape regressions."""
import time
import numpy as np
from search_complete_return import read
from study_io import BASE,register,receipt,write_json


def main():
    source=BASE/'canonical_gate_band_refined_capsule_energy_restore_20260917'
    reference=BASE/'canonical_gate_band_bend_learning_reference_20260917'
    probes=[BASE/'canonical_gate_band_refined_capsule_energy_composition_20260917',
            BASE/'canonical_gate_band_refined_capsule_energy_learning_20260917']
    stage=register('canonical_gate_band_bend_measured_error_margin_20260917',
        [__file__,source/'direction.npz',source/'optimizer.json',source/'receipt.json',
         reference/'basis.npz',reference/'results.json']+
        [p/n for p in probes for n in ('results.json','receipt.json')],
        dict(new_actor_decisions=0,new_physics_ticks=0,wall_s=30),
        'Both larger learned steps completed29/29; full-step energy improved '
        'about4.07% but three corner metrics regressed. Fit current-center '
        'quadratic prediction-error allowances ONLY for those observed '
        'path/metric pairs, using the existing-.125,+.125,.5,1.0 responses '
        'of this exact direction and its fresh waveform predictions. The '
        'rate is max positive underprediction divided by squared metric norm. '
        'Do not borrow older-center coefficients. This empirical fit is '
        'not a bound in a new direction. One subsequent bounded correction '
        'may use it and must repeat physical composition checks and unchanged '
        '2.5% actual total-energy/shape acceptance. No larger physical budget '
        'or change to the common1.25mm, V/A/J, gate or stop conditions.')
    start=time.perf_counter();optimization=read(source/'optimizer.json')
    baseline=read(reference/'results.json');assert baseline['parent_sha256']==optimization['parent_sha256']
    with np.load(source/'direction.npz') as z:
        direction=z['direction'].ravel();names=z['metric_names'].tolist();ids=z['path_id'].tolist()
        predictions={float(s):a for s,a in zip(z['prediction_scales'],z['predictions'])}
    with np.load(reference/'basis.npz') as z:norm2=float(direction@z['metric']@direction)
    measured=[read(p/'results.json') for p in probes]
    targets={(r['path'],r['metric']) for c in measured[1]['candidates'] for r in c['checks']['regressions']}
    assert targets=={(14,'out_edge_outward_mm'),(14,'heading_overshoot_deg'),(15,'heading_overshoot_deg')}
    base={r['path']:r['values'] for r in baseline['baseline']['metrics']};rates={t:0. for t in targets};observations=[]
    for data in measured:
        for c in data['candidates']:
            assert c['summary']['successes']==29
            actual={r['path']:r['values'] for r in c['metrics']};step=c['step']
            for p,n in sorted(targets):
                i=names.index(n);actual_change=actual[p][i]-base[p][i]
                predicted=float(predictions[step][ids.index(p),i]);error=actual_change-predicted
                rate=max(0.,error)/(norm2*step**2);rates[p,n]=max(rates[p,n],rate)
                observations.append(dict(path=p,metric=n,step=step,actual_change=actual_change,
                    predicted_change=predicted,underprediction=error,quadratic_rate=rate))
    result=dict(parent_sha256=optimization['parent_sha256'],direction_metric_norm_squared=norm2,
        source_proposal=source.name,observations=observations,
        coefficients=[dict(path=p,metric=n,rate=rates[p,n]) for p,n in sorted(targets)],
        guaranteed_bound=False,model_type='empirical_quadratic_metric_norm_underprediction',goal_complete=False)
    write_json(stage/'error_model.json',result)
    receipt(stage,dict(status='prepared_current_center_empirical_shape_margin',targets=len(targets),
        new_actor_decisions=0,new_physics_ticks=0,wall_s=time.perf_counter()-start))
    print('BEND_ERROR_ALLOWANCES',result['coefficients'],flush=True)


if __name__=='__main__':main()
