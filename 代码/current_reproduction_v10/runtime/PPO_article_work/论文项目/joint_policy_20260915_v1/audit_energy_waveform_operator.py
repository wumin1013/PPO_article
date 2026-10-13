"""Test omitted squared-response energy using existing current credit only."""
import time
import numpy as np
from energy_waveform_credit import energy_signals
from audit_xy_curvature_resolution import curvature_metrics
from search_complete_return import read
from study_io import BASE,register,receipt,write_json


def main():
    credit=BASE/'canonical_gate_band_complete_credit_20260917'
    source=BASE/'canonical_gate_band_current_parameter_solve_20260917'
    solve=BASE/'canonical_gate_band_waveform_step_solve_20260917'
    small=BASE/'canonical_gate_band_current_composition_20260917'
    large=BASE/'canonical_gate_band_learned_update_20260917'
    stage=register('canonical_gate_band_energy_operator_audit_20260917',
        [__file__,BASE/'energy_waveform_credit.py',BASE/'audit_xy_curvature_resolution.py',
         credit/'results.json',credit/'receipt.json',source/'gradients.npz',source/'receipt.json',
         solve/'direction.npz',small/'results.json',small/'receipt.json',large/'results.json',large/'receipt.json'],
        dict(paths=[7,8],new_actor_decisions=0,new_physics_ticks=0,suffix_files=552,wall_s=90),
        'Half/full parameter steps improve shape totals but regress near-binding '
        'bending energies on paths7/8. Test whether the scalar energy derivative '
        'omits squared waveform-response terms. Encode the EXACT existing energy '
        'as the sum of squared signed curvature times sqrt(spatial sample mass). '
        'For physical curvature the mass is1ms*speed; for rawXY it is the original '
        'mean adjacent segment length. Derive this density response from existing '
        'current paired suffixes, then square AFTER affine prediction. Compare '
        'linear and quadratic terms with every already saved small/large scale. '
        'Identity is checked against independent existing energy metrics. No '
        'position smoothing, changed objective, new physics, fitting or sealed data.')
    start=time.perf_counter();raw=read(credit/'results.json')
    lookup={(r['query'],r['axis'],r['direction']):r for r in raw['records']}
    with np.load(source/'gradients.npz') as z:
        qjac=z['query_jacobian'].copy();gradient=z['gradient'].copy();ids=z['path_id'].tolist();names=z['metric_names'].tolist()
    with np.load(solve/'direction.npz') as z:direction=z['direction'].ravel().copy()
    responses=np.einsum('qhd,d->qh',qjac,direction)
    variants=read(small/'results.json')['candidates']+read(large/'results.json')['candidates']
    energy_names=['full_xy_normalized_energy','full_physical_normalized_energy'];records=[];files=0
    for path in (7,8):
        b=next(b for b in raw['baselines'] if b['path']==path)
        with np.load(BASE/b['trace']) as z:ref,rows=z['reference'].copy(),z['rows'].copy()
        nominal,valid=energy_signals(ref,rows);common=valid.copy();delta=np.zeros_like(nominal)
        baseline=curvature_metrics(ref,rows)
        assert all(abs(np.sum(nominal[:,j]**2)-baseline[name])<1e-10 for j,name in enumerate(energy_names))
        for q in (q for q in raw['queries'] if q['path']==path):
            for head in range(3):
                pair=[lookup[q['query'],head,sign] for sign in (-1,1)];signals=[]
                for r in pair:
                    assert r['success']
                    with np.load(BASE/r['trace']) as z:full=np.vstack([rows[:q['anchor']*20],z['suffix_rows']])
                    value,mask=energy_signals(ref,full,len(rows));common&=mask;signals.append(value);files+=1
                delta+=(signals[1]-signals[0])/(pair[1]['latent_offset']-pair[0]['latent_offset'])*responses[q['query'],head]
        delta[~common]=0.
        np.savez_compressed(stage/f'path{path}_density_response.npz',nominal=nominal,valid=valid,common_valid=common,response=delta)
        for candidate in variants:
            step=candidate['step'];j=next(i for i,r in enumerate(candidate['results']) if r['original_path']==path)
            with np.load(BASE/candidate['traces'][j]) as z:actual=curvature_metrics(ref,z['rows'])
            for channel,name in enumerate(energy_names):
                linear=float(2*step*np.sum(nominal[valid[:,channel],channel]*delta[valid[:,channel],channel]))
                quadratic=float(step**2*np.sum(delta[valid[:,channel],channel]**2))
                scalar=float(step*gradient[ids.index(path),names.index(name)]@direction)
                change=actual[name]-baseline[name]
                record=dict(path=path,step=step,metric=name,actual_change=change,scalar_prediction=scalar,
                    density_linear_term=linear,density_quadratic_term=quadratic,
                    squared_density_prediction=linear+quadratic,
                    scalar_error=change-scalar,squared_density_error=change-linear-quadratic)
                records.append(record)
                if step>=.5:print('ENERGY_OPERATOR',record,flush=True)
    assert files==552 and time.perf_counter()-start<90
    write_json(stage/'results.json',dict(records=records,suffix_files=files,goal_complete=False))
    receipt(stage,dict(status='completed_saved_squared_energy_operator_audit',suffix_files=files,
        new_actor_decisions=0,new_physics_ticks=0,wall_s=time.perf_counter()-start))


if __name__=='__main__':main()
