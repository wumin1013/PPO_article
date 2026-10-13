"""Reuse all observed current-center shape errors with the corrected metric."""
import time
import argparse
import numpy as np
from waveform_constraint_model import waveform_facets
from prepare_current_waveform_credit import wave_signals
from search_complete_return import read
from study_io import BASE,register,receipt,write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--geometry',default='canonical_gate_band_correct_decoder_parameter_geometries_20260917')
    parser.add_argument('--waveforms',default='canonical_gate_band_refined_energy_current_waveforms_20260917')
    parser.add_argument('--cases-json')
    parser.add_argument('--extra-scales-json')
    parser.add_argument('--stage',default='canonical_gate_band_correct_physical_current_error_model_20260917')
    args=parser.parse_args()
    geometry=BASE/args.geometry
    waves=BASE/args.waveforms
    groups=[('canonical_gate_band_latent_cruise_feasible_restore_20260917',
        ['canonical_gate_band_latent_cruise_composition_20260917','canonical_gate_band_latent_cruise_learning_20260917']),
        ('canonical_gate_band_physical_cruise_endpoint_restore_20260917',
        ['canonical_gate_band_physical_cruise_composition_20260917']),
        ('canonical_gate_band_physical_cruise_margin_solve_20260917',
        ['canonical_gate_band_physical_cruise_margin_composition_20260917','canonical_gate_band_physical_cruise_margin_learning_20260917']),
        ('canonical_gate_band_correct_physical_cruise_endpoint_restore_20260917',
        ['canonical_gate_band_correct_physical_cruise_composition_20260917'])]
    if args.cases_json:groups=[(direction,probes) for _,direction,probes,_ in read(BASE/args.cases_json)]
    sources=([BASE/args.cases_json] if args.cases_json else [])+[__file__,geometry/'geometry.npz',geometry/'results.json',geometry/'receipt.json',
        waves/'results.json',waves/'receipt.json',BASE/'waveform_constraint_model.py']
    loaded=[];targets=set();wd=read(waves/'results.json');parent=wd['parent_sha256']
    assert parent==read(geometry/'results.json')['parent_sha256']
    for solve,probes in groups:
        opt=read(BASE/solve/'optimizer.json');assert opt['parent_sha256']==parent
        sources += [BASE/solve/n for n in ('direction.npz','optimizer.json','receipt.json')]
        with np.load(BASE/solve/'direction.npz') as z:d=z['direction'].ravel().copy()
        for probe in probes:
            data=read(BASE/probe/'results.json');sources += [BASE/probe/n for n in ('results.json','receipt.json')]
            if (BASE/probe/'comparison.json').exists():
                sources.append(BASE/probe/'comparison.json')
                targets.update((r['path'],r['metric']) for r in read(BASE/probe/'comparison.json')['regressions'])
            else:targets.update((r['path'],r['metric']) for c in data['candidates'] for r in c['checks']['regressions'])
            loaded.extend((probe,d,c,data['metric_names']) for c in data['candidates'])
    if args.extra_scales_json:
        sources.append(BASE/args.extra_scales_json)
        for solve,probe in read(BASE/args.extra_scales_json):
            assert read(BASE/solve/'optimizer.json')['parent_sha256']==parent
            with np.load(BASE/solve/'direction.npz') as z:d=z['direction'].ravel().copy()
            data=read(BASE/probe/'results.json');assert data['parent_sha256']==parent
            sources += [BASE/solve/'direction.npz',BASE/probe/'results.json',BASE/probe/'receipt.json']
            for r in data['records']:
                assert r['result']['success']
                c=dict(step=r['scale'],metrics=[r['metrics']],summary=dict(successes=1))
                loaded.append((probe,d,c,data['metric_names']))
    # A nonnegative scalar error added to a zero-bound clipped metric would
    # prohibit every nonzero update. Protect its signed waveform instead.
    waveform_targets=set()
    for _,_,c,names in loaded:
        for r in c['metrics']:
            if r['values'][names.index('in_edge_outward_mm')]>1e-8:
                wr=next((q for q in wd['records'] if q['path']==r['path'] and 'waveform_file' in q),None)
                if wr and wr['shape']['in_edge_outward_mm']<=1e-8:
                    waveform_targets.add((r['path'],'in_edge_outward_mm'))
    targets-=waveform_targets
    stage=register(args.stage,sources,
        dict(new_actor_decisions=0,new_physics_ticks=0,wall_s=60,same_center_directions=len(groups),
             subsequent_correction_attempts=1),
        'After correcting the preview decoder geometry, reuse ALL completed same-center '
        'responses from four available parameter directions. Keep only path/metric pairs '
        'that actually regressed in those responses. Recompute RAW waveform predictions '
        'without earlier empirical inflation, and divide positive errors by the exact '
        'corrected physical-metric norm squared. Earlier mis-scaled full steps are retained '
        'as observed stress responses and explicitly marked outside the current radius; '
        'they are not admissible new updates. Take the maximum measured rate for each '
        'target. This remains an empirical model, not a guaranteed bound in a new '
        'direction. One correction and fresh both-sign validation; no physical relaxation.')
    start=time.perf_counter();observations=[];rates={t:0. for t in targets};cache={}
    with np.load(geometry/'geometry.npz') as z:metric=z['metric_physical'].copy()
    for p,n in targets|waveform_targets:
        if p in cache:continue
        record=next(r for r in wd['records'] if r['path']==p)
        with np.load(waves/record['waveform_file']) as z:cache[p]={k:z[k].copy() for k in z.files}
    waveform_rates={p:np.zeros(len(cache[p]['nominal'])) for p,n in waveform_targets}
    for probe,d,c,names in loaded:
        assert c['summary']['successes']==len(c['metrics'])
        x=c['step']*d;norm2=float(x@metric@x);actual={r['path']:r['values'] for r in c['metrics']}
        predictions={}
        for p,w in cache.items():
            predictions[p]={n:v[0] for n,v in waveform_facets(w['nominal'],w['gradient'],w['valid'],float(w['turn_deg']),x).items()}
            e=w['energy_constant']+w['energy_linear']@x+np.einsum('d,cdk,k->c',x,w['energy_quadratic'],x)
            predictions[p].update(zip(('full_xy_normalized_energy','full_physical_normalized_energy'),e))
        for p,n in sorted(targets):
            if p not in actual:continue
            error=float(actual[p][names.index(n)]-predictions[p][n]);rate=max(0.,error)/norm2
            rates[p,n]=max(rates[p,n],rate)
            observations.append(dict(source=probe,scale=c['step'],path=p,metric=n,underprediction=error,
                metric_norm=float(np.sqrt(norm2)),outside_current_radius=bool(norm2>.1**2+1e-12),rate=rate))
        for p,n in waveform_targets:
            if p not in actual:continue
            j=next(i for i,r in enumerate(c['metrics']) if r['path']==p)
            with np.load(BASE/c['traces'][j]) as z:ref,rows=z['reference'].copy(),z['rows'].copy()
            w=cache[p];truth,_=wave_signals(ref,rows,len(w['nominal']))
            prediction=w['nominal'][:,0]+w['gradient'][:,0]@x
            # Positive overprediction of inward position is unsafe.
            waveform_rates[p]=np.maximum(waveform_rates[p],np.maximum(prediction-truth[:,0],0.)/norm2)
    from cruise_update_geometry import CruiseUpdateGeometry
    identity=CruiseUpdateGeometry(geometry,'physical',parent).identity
    result=dict(parent_sha256=parent,parameter_geometry=identity,observations=observations,
        coefficients=[dict(path=p,metric=n,rate=r) for (p,n),r in sorted(rates.items())],
        guaranteed_bound=False,raw_prediction_recomputed=True,goal_complete=False)
    result['waveform_bounds']=[]
    for p,n in sorted(waveform_targets):
        name=f'path{p}_entry_error.npz';np.savez_compressed(stage/name,rate=waveform_rates[p])
        result['waveform_bounds'].append(dict(path=p,metric=n,channel=0,sign=-1.,file=name,
            maximum_rate=float(waveform_rates[p].max()),signed_waveform_before_clipping=True))
    write_json(stage/'error_model.json',result)
    receipt(stage,dict(status='prepared_current_shape_errors_in_correct_physical_metric',targets=len(targets),
        new_actor_decisions=0,new_physics_ticks=0,wall_s=time.perf_counter()-start))
    print('CURRENT_CORRECT_ERROR_MODEL',result['coefficients'],flush=True)


if __name__=='__main__':main()
