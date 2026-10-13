"""Calibrate current waveform credit with three complete-policy secants."""
import time
import numpy as np
import torch
from prepare_current_waveform_credit import wave_signals
from waveform_constraint_model import waveform_facets
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    wave=BASE/'canonical_gate_band_half_step_waveform_credit_20260917'
    source=BASE/'canonical_gate_band_half_step_parameter_solve_20260917'
    probe=BASE/'canonical_gate_band_half_step_head_interaction_probe_20260917'
    full=BASE/'canonical_gate_band_half_step_locality_probe_20260917'
    direction_source=BASE/'canonical_gate_band_half_step_locality_direction_20260917'
    model=BASE/'canonical_gate_band_quadratic_error_learned_update_20260917/candidate_0.5.pt'
    md=torch.load(model,map_location='cpu',weights_only=False);weights=md['policy']['residual_coefficients']
    with np.load(source/'support.npz') as z:metric=z['metric'].copy()
    with np.load(direction_source/'direction.npz') as z:direction=z['direction'].copy()
    records=read(probe/'results.json')['records'];full_data=read(full/'results.json')
    stage=register('canonical_gate_band_half_step_coherent_credit_calibration_20260917',
        [__file__,model,source/'support.npz',source/'receipt.json',probe/'results.json',probe/'receipt.json',
         full/'results.json',full/'receipt.json',wave/'results.json',wave/'receipt.json',
         wave/'path14_waveform.npz',wave/'path15_waveform.npz',direction_source/'direction.npz',
         BASE/'prepare_current_waveform_credit.py',BASE/'waveform_constraint_model.py'],
        dict(paths=[14,15],fitted_head_directions=3,fitting_scales=[-.125,.125],
             unused_for_fit_combinations=['011','101','110','111'],new_physics_ticks=0,new_actor_decisions=0,wall_s=60),
        'Single-head finite-difference amplitude reduction did not fix whole-policy '
        'prediction, with feed-direction error dominating. Calibrate three measured '
        'complete-policy head secants by a minimum support-metric low-rank correction '
        'to the current waveform Jacobian. Fit only the six single-head traces for '
        'each path; reserve all pair and all-three traces for internal composition '
        'validation. This is a local learned prediction model, not a new actor or '
        'an exact infinitesimal derivative. No runtime trajectory table or physical '
        'limit change. Do not use these already seen training traces as independent '
        'generalization evidence; any eventual actor direction needs fresh checks.')
    start=time.perf_counter();basis=[]
    def displacement(step,bits):
        moved=weights+torch.tensor(step*direction*np.asarray(bits)[:,None],dtype=torch.float32)
        return (moved.double()-weights.double()).numpy().ravel()
    for head in range(3):
        bits=[int(i==head) for i in range(3)]
        basis.append((displacement(.125,bits)-displacement(-.125,bits))/.25)
    basis=np.asarray(basis);gram=basis@metric@basis.T
    assert np.max(abs(gram-np.diag(np.diag(gram))))<1e-12
    dual=np.linalg.solve(gram,basis@metric)
    assert np.max(abs(dual@basis.T-np.eye(3)))<1e-10
    reports=[];validation=[];details=[]
    for path in (14,15):
        with np.load(wave/f'path{path}_waveform.npz') as z:
            ref=z['reference'].copy();nominal=z['nominal'].copy();old=z['gradient'].copy()
            valid=z['nominal_valid'].copy();common=z['common_valid'].copy();channels=z['channels'].copy()
        old[~common]=0.;values={}
        for r in records:
            if r['path']!=path:continue
            with np.load(BASE/r['trace']) as z:value,mask=wave_signals(ref,z['rows'],len(nominal))
            values[r['step'],tuple(r['bits'])]=value
            if sum(r['bits'])==1:common&=mask
        measured=[]
        for head in range(3):
            bits=tuple(int(i==head) for i in range(3))
            measured.append((values[.125,bits]-values[-.125,bits])/.25)
        measured=np.moveaxis(np.asarray(measured),0,-1)
        error=measured-np.einsum('tcd,hd->tch',old,basis)
        error[~common]=0.
        corrected=old+np.einsum('tch,hd->tcd',error,dual)
        secant_residual=np.einsum('tcd,hd->tch',corrected,basis)-measured
        exact_error=float(np.max(abs(secant_residual[common])))
        assert exact_error<1e-8
        np.savez_compressed(stage/f'path{path}_waveform.npz',reference=ref,nominal=nominal,
            nominal_valid=valid,common_valid=common,gradient=corrected,channels=channels)
        np.savez_compressed(stage/f'path{path}_calibration.npz',basis=basis,dual=dual,
            measured_secants=measured,correction_response=error,common_valid=common)
        reports.append(dict(path=path,maximum_fitted_secant_residual=exact_error))
        variants=[r for r in records if r['path']==path]
        for r in full_data['candidates']:
            j=next(j for j,q in enumerate(r['results']) if q['original_path']==path)
            variants.append(dict(step=r['step'],bits=[1,1,1],trace=r['traces'][j]))
        for r in variants:
            with np.load(BASE/r['trace']) as z:actual,mask=wave_signals(ref,z['rows'],len(nominal))
            x=displacement(r['step'],r['bits']);supported=valid[:,2]&mask[:,2]
            row=dict(path=path,step=r['step'],bits=r['bits'],used_for_fit=sum(r['bits'])==1)
            for label,jac in [('original',old),('calibrated',corrected)]:
                prediction=nominal+np.einsum('tcd,d->tc',jac,x)
                residual=prediction[supported,2]-actual[supported,2]
                facets=waveform_facets(nominal,jac,valid,-155.,x)
                reverse=float(np.maximum(-actual[mask[:,5],5],0.).sum()*.001)
                overshoot=float(max(0.,actual[mask[:,2],2].max()-155.))
                row[label]=dict(heading_max_abs_error_deg=float(abs(residual).max()),
                    heading_rms_error_deg=float(np.sqrt(np.mean(residual**2))),
                    reversal_error_deg=facets['full_physical_reverse_deg'][0]-reverse,
                    overshoot_error_deg=facets['heading_overshoot_deg'][0]-overshoot)
            (details if row['used_for_fit'] else validation).append(row)
            if sum(r['bits'])==3:print('COHERENT_CREDIT',row,flush=True)
    totals={label:sum(r[label]['heading_rms_error_deg'] for r in validation) for label in ('original','calibrated')}
    write_json(stage/'results.json',dict(parent_sha256=sha(model),base_waveform_stage=wave.name,
        records=reports,fitted_single_heads=details,unused_for_fit_validation=validation,
        summed_validation_heading_rms_deg=totals,new_physics_ticks=0,goal_complete=False))
    receipt(stage,dict(status='calibrated_and_checked_current_coherent_secants',new_actor_decisions=0,
        new_physics_ticks=0,wall_s=time.perf_counter()-start,goal_complete=False))
    print('COHERENT_VALIDATION_TOTALS',totals,flush=True)


if __name__=='__main__':main()
