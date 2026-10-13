"""Put error allowances before max(0, max(wave)) at zero excursion bounds."""
import time
import numpy as np
from prepare_current_waveform_credit import wave_signals
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    original=BASE/'canonical_gate_band_multiscale_error_model_20260917'
    data=read(original/'results.json');wave=BASE/'canonical_gate_band_waveform_credit_20260917'
    current=BASE/'canonical_gate_band_waveform_step_solve_20260917'
    previous=BASE/'canonical_recentered_waveform_feasibility_recovery_20260917'
    measured=BASE/'canonical_gate_band_matched_probe_20260917'
    small=BASE/'canonical_gate_band_current_composition_20260917'
    large=BASE/'canonical_gate_band_learned_update_20260917'
    cases=data['zero_baseline_with_nonzero_margin']
    stage=register('canonical_gate_band_boundary_error_model_20260917',
        [__file__,original/'results.json',original/'receipt.json',wave/'results.json',wave/'receipt.json',
         current/'direction.npz',current/'optimizer.json',previous/'direction.npz',
         measured/'results.json',small/'results.json',large/'results.json',
         BASE/'canonical_gate_band_known_direction_error_20260917/results.json',
         BASE/'prepare_current_waveform_credit.py'],
        dict(new_actor_decisions=0,new_physics_ticks=0,zero_boundary_cases=cases,wall_s=45),
        'A positive isotropic margin AFTER a nonnegative max at a zero excursion '
        'bound makes every nonzero update infeasible, even when the measured '
        'positive direction has no excursion. This is an error-model artifact, '
        'not physical impossibility. For exactly those diagnosed zero-bound '
        'metrics, move the empirically calibrated current error allowance to '
        'each signed waveform sample BEFORE the max/positive-part operator. '
        'Keep every observed error sample and the unchanged zero bound. Use '
        'current credit and the same five completed parameter contrasts; '
        'no new trajectories, reward changes or guarantee for another direction.')
    start=time.perf_counter()
    with np.load(current/'direction.npz') as z:direction=z['direction'].ravel().copy()
    with np.load(previous/'direction.npz') as z:old_direction=z['direction'].ravel().copy()
    cn=read(current/'optimizer.json')['projection']['metric_norm']
    on=read(BASE/'canonical_gate_band_known_direction_error_20260917/results.json')['measured_direction_norm']
    variants=[dict(candidate=c,step=c['step'],direction=direction,norm=abs(c['step'])*cn)
        for c in read(small/'results.json')['candidates']+read(large/'results.json')['candidates']]
    variants.append(dict(candidate=read(measured/'results.json')['candidates'][1],step=.125,
        direction=old_direction,norm=.125*on))
    names=data['metric_names'];ids=data['path_ids'];c1=np.asarray(data['one_sided_error_per_norm']);c2=np.asarray(data['quadratic_error_per_norm_squared'])
    channel_map={'in_edge_outward_mm':(0,-1.),'out_edge_outward_mm':(1,-1.),
                 'negative_signed_heading_deg':(2,-1.),'heading_overshoot_deg':(2,1.)}
    descriptors=[];checks=[]
    for case in cases:
        path=case['path'];metric=case['metric'];channel,sign=channel_map[metric]
        with np.load(wave/f'path{path}_waveform.npz') as z:
            ref=z['reference'].copy();nom=z['nominal'].copy();g=z['gradient'].copy()
            valid=z['nominal_valid'][:,channel].copy();g[~z['common_valid']]=0.
        a,b=np.diff(ref,axis=0);turn=abs(np.rad2deg(np.arctan2(a[0]*b[1]-a[1]*b[0],a@b)))
        offset=-turn if metric=='heading_overshoot_deg' else 0.
        lo=np.zeros(len(nom));quad=np.zeros(len(nom));samples=[]
        for v in variants:
            c=v['candidate'];j=next(i for i,r in enumerate(c['results']) if r['original_path']==path)
            with np.load(BASE/c['traces'][j]) as z:actual,mask=wave_signals(ref,z['rows'],len(nom))
            prediction=sign*(nom[:,channel]+v['step']*g[:,channel]@v['direction'])+offset
            truth=sign*actual[:,channel]+offset;error=truth-prediction
            assert np.all(mask[valid,channel])
            samples.append((v['step'],v['norm'],prediction,truth,error))
            if abs(v['step'])<=.125:lo=np.maximum(lo,error/v['norm'])
        for step,norm,pred,truth,error in samples:quad=np.maximum(quad,(error-lo*norm)/norm**2)
        maximum_deficit=0.
        for step,norm,pred,truth,error in samples:
            maximum_deficit=max(maximum_deficit,float(np.max(error[valid]-lo[valid]*norm-quad[valid]*norm**2)))
        assert maximum_deficit<1e-10
        file=stage/f'path{path}_{metric}_pointwise_error.npz'
        np.savez_compressed(file,linear=lo,quadratic=quad)
        descriptors.append(dict(path=path,metric=metric,channel=channel,sign=sign,offset=offset,
            file=str(file.relative_to(BASE)),sha256=sha(file)))
        checks.append(dict(case,maximum_pointwise_deficit=maximum_deficit,
            old_linear_metric_margin=float(c1[ids.index(path),names.index(metric)]),
            old_quadratic_metric_margin=float(c2[ids.index(path),names.index(metric)])))
        c1[ids.index(path),names.index(metric)]=0.;c2[ids.index(path),names.index(metric)]=0.
    data.update(one_sided_error_per_norm=c1.tolist(),quadratic_error_per_norm_squared=c2.tolist(),
        waveform_error_descriptors=descriptors,zero_boundary_model_checks=checks,
        zero_boundary_handling='pointwise allowances before max, unchanged zero upper bound')
    write_json(stage/'results.json',data)
    receipt(stage,dict(status='prepared_current_pointwise_zero_boundary_error_model',
        cases=len(cases),new_actor_decisions=0,new_physics_ticks=0,wall_s=time.perf_counter()-start))
    print('POINTWISE_BOUNDARY_ERROR',checks,flush=True)


if __name__=='__main__':main()
