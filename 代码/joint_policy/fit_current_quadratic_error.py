"""Current, multi-scale empirical remainder model; no new trajectory rollouts."""
import time
import numpy as np
from search_complete_return import read
from study_io import BASE,register,receipt,write_json


def main():
    source=BASE/'canonical_gate_band_waveform_step_solve_20260917'
    small=BASE/'canonical_gate_band_current_composition_20260917'
    large=BASE/'canonical_gate_band_learned_update_20260917'
    known=BASE/'canonical_gate_band_known_direction_error_20260917'
    opt=read(source/'optimizer.json');sd=read(small/'results.json');ld=read(large/'results.json');kd=read(known/'results.json')
    assert opt['parent_sha256']==kd['parent_sha256']
    stage=register('canonical_gate_band_multiscale_error_model_20260917',
        [__file__,source/'optimizer.json',source/'direction.npz',source/'receipt.json',
         small/'results.json',small/'receipt.json',large/'results.json',large/'receipt.json',
         known/'results.json',known/'receipt.json',
         BASE/'canonical_gate_band_energy_operator_audit_20260917/results.json',
         BASE/'canonical_gate_band_energy_operator_audit_20260917/receipt.json'],
        dict(new_actor_decisions=0,new_physics_ticks=0,wall_s=30,
             measured_directions=2,current_direction_scales=[-.125,.125,.5,1.],other_direction_scales=[.125]),
        'The squared-density operator explains the sign of path7 energy '
        'regression, but most path8 error remains; do not claim one operator '
        'explains all dynamics. Keep the original physical metrics. Fit an '
        'empirical upper remainder c1*norm+c2*norm^2 using CURRENT-runtime '
        'already completed parameter contrasts. Set c1 from both small signs '
        'and the earlier current-runtime positive contrast; set nonnegative c2 '
        'from residuals at the two larger scales. Cover every shape metric, '
        'including the two directed-heading regressions, rather than altering '
        'reward weights or lowering acceptance. Verify the fitted remainder '
        'covers its observed samples. This is not a bound for new directions; '
        'a proposal requires fresh both-sign complete29 checks and the same '
        'larger-step gate. One new model hypothesis, no indefinite continuation.')
    start=time.perf_counter();names=opt['metric_names'];ids=opt['path_ids']
    with np.load(source/'direction.npz') as z:pred={float(s):dict(zip(ids,v)) for s,v in zip(z['prediction_scales'],z['predictions'])}
    base={r['path']:np.asarray(r['values']) for r in sd['baseline_metrics']}
    norm=opt['projection']['metric_norm'];samples=[]
    for candidate in sd['candidates']+ld['candidates']:
        step=candidate['step']
        for row in candidate['metrics']:
            path=row['path']
            if path not in ids:continue
            assert row['success']
            error=np.asarray(row['values'])-base[path]-pred[step][path]
            samples.append(dict(path=path,step=step,norm=abs(step)*norm,error=error,
                small=abs(step)<=.125,direction='current'))
    for row in kd['records']:
        samples.append(dict(path=row['path'],step=row['step'],norm=abs(row['step'])*kd['measured_direction_norm'],
            error=np.asarray(row['error']),small=True,direction='previous_current_runtime_contrast'))
    c1=np.zeros((len(ids),len(names)));c2=np.zeros_like(c1)
    for s in samples:
        if s['small']:
            i=ids.index(s['path']);c1[i]=np.maximum(c1[i],s['error']/s['norm'])
    for s in samples:
        i=ids.index(s['path']);c2[i]=np.maximum(c2[i],(s['error']-c1[i]*s['norm'])/s['norm']**2)
    deficit=0.
    for s in samples:
        i=ids.index(s['path']);deficit=max(deficit,float(np.max(s['error']-c1[i]*s['norm']-c2[i]*s['norm']**2)))
    assert deficit<1e-10
    used=names[:names.index('segment_max_distance_mm')]
    zero_cases=[dict(path=p,metric=name) for p in ids[:16] for name in used
        if base[p][names.index(name)]<1e-10 and (c1[ids.index(p),names.index(name)]>1e-12 or c2[ids.index(p),names.index(name)]>1e-12)]
    result=dict(parent_sha256=opt['parent_sha256'],metric_names=names,path_ids=ids,
        one_sided_error_per_norm=c1.tolist(),quadratic_error_per_norm_squared=c2.tolist(),
        proposed_margin_metrics=used,zero_baseline_with_nonzero_margin=zero_cases,
        maximum_observed_upper_model_deficit=deficit,guaranteed_bound=False,goal_complete=False,
        samples=[dict(s,error=s['error'].tolist()) for s in samples])
    write_json(stage/'results.json',result)
    receipt(stage,dict(status='fitted_current_multiscale_empirical_remainder',samples=len(samples),
        new_actor_decisions=0,new_physics_ticks=0,wall_s=time.perf_counter()-start))
    print('CURRENT_REMAINDER',dict(samples=len(samples),deficit=deficit,zero_baseline_cases=zero_cases),flush=True)


if __name__=='__main__':main()
