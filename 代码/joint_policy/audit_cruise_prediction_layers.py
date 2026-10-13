"""Separate nonlinear geometry error from affine closed-loop dynamics error."""
import time
import argparse
import numpy as np
from transition_waveform_model import chain_responses
from prepare_current_waveform_credit import wave_signals
from energy_waveform_credit import energy_signals
from waveform_constraint_model import waveform_facets
from search_complete_return import read
from study_io import BASE,register,receipt,write_json


def metrics(reference,rows,turn):
    wave,valid=wave_signals(reference,rows)
    facets=waveform_facets(wave,np.zeros((*wave.shape,1)),valid,turn,np.zeros(1))
    result={n:v[0] for n,v in facets.items()}
    density,_=energy_signals(reference,rows)
    result.update(zip(('full_xy_normalized_energy','full_physical_normalized_energy'),(density**2).sum(0)))
    return result,wave,valid


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--collection',default='canonical_gate_band_refined_energy_current_transition_20260917')
    parser.add_argument('--waveforms',default='canonical_gate_band_refined_energy_current_waveforms_20260917')
    parser.add_argument('--cases-json')
    parser.add_argument('--stage',default='canonical_gate_band_cruise_prediction_layers_20260917')
    args=parser.parse_args()
    collection=BASE/args.collection
    waves=BASE/args.waveforms
    cases=[('physical','canonical_gate_band_physical_cruise_endpoint_restore_20260917',
        ['canonical_gate_band_physical_cruise_composition_20260917'],[13]),
        ('latent','canonical_gate_band_latent_cruise_feasible_restore_20260917',
        ['canonical_gate_band_latent_cruise_composition_20260917','canonical_gate_band_latent_cruise_learning_20260917'],[14,15])]
    if args.cases_json:cases=read(BASE/args.cases_json)
    sources=([BASE/args.cases_json] if args.cases_json else [])+[BASE/__file__,collection/'results.json',collection/'receipt.json',waves/'results.json',waves/'receipt.json',
        BASE/'transition_waveform_model.py',BASE/'prepare_current_waveform_credit.py',BASE/'energy_waveform_credit.py']
    for arm,direction,probes,ids in cases:
        sources += [BASE/direction/n for n in ('direction.npz','optimizer.json','receipt.json')]
        sources += [BASE/p/n for p in probes for n in ('results.json','receipt.json')]
    stage=register(args.stage,sources,
        dict(new_actor_decisions=0,new_physics_ticks=0,wall_s=90,paths=sorted({p for _,_,_,ids in cases for p in ids})),
        'Reuse all saved same-center both-sign and learned-step responses. Compare the original '
        'affine SIGNAL prediction with exact geometric metrics of the affine RAW XY/v/a prediction, '
        'and with actual closed-loop trajectories. This separates geometry linearization from '
        'raw-state prediction error without new physical intervention. Record peak tick/speed '
        'and validity-mask changes to test alternative threshold/peak-exchange explanations. '
        'Do not infer one cause for the historical PPO reversal or change acceptance.')
    start=time.perf_counter();records=[]
    cd=read(collection/'results.json');wd=read(waves/'results.json')
    for arm,direction,probes,ids in cases:
        opt=read(BASE/direction/'optimizer.json');assert opt['parent_sha256']==cd['parent_sha256']==wd['parent_sha256']
        with np.load(BASE/direction/'direction.npz') as z:d=z['direction'].ravel()
        for path in ids:
            record=next(r for r in cd['records'] if r['path']==path)
            with np.load(collection/record['file']) as z:local={k:z[k].copy() for k in z.files}
            delta=chain_responses(local,d[None,:])[:,:,0]
            wr=next(r for r in wd['records'] if r['path']==path)
            with np.load(waves/wr['waveform_file']) as z:w={k:z[k].copy() for k in z.files}
            ref,nom=local['reference'],local['rows'];turn=float(w['turn_deg'])
            baseline,_,_=metrics(ref,nom,turn)
            for probe in probes:
                data=read(BASE/probe/'results.json')
                for c in data['candidates']:
                    scale=c['step'];j=next(j for j,r in enumerate(c['results']) if r['original_path']==path)
                    assert c['results'][j]['success']
                    with np.load(BASE/c['traces'][j]) as z:actual=z['rows'].copy()
                    affine=nom.copy();affine[:,:6]+=scale*delta
                    raw,raw_wave,raw_valid=metrics(ref,affine,turn)
                    truth,actual_wave,actual_valid=metrics(ref,actual,turn)
                    predicted={n:v[0] for n,v in waveform_facets(w['nominal'],w['gradient'],w['valid'],turn,scale*d).items()}
                    energies=w['energy_constant']+w['energy_linear']@(scale*d)+np.einsum('d,cdk,k->c',scale*d,w['energy_quadratic'],scale*d)
                    predicted.update(zip(('full_xy_normalized_energy','full_physical_normalized_energy'),energies))
                    max_tick=lambda a,v:int(np.flatnonzero(v[:,4])[np.argmax(abs(a[v[:,4],4]))])
                    ticks=[max_tick(w['nominal'],w['valid']),max_tick(raw_wave,raw_valid),max_tick(actual_wave,actual_valid)]
                    exit_tick=lambda a,v:int(np.flatnonzero(v[:,1])[np.argmin(a[v[:,1],1])])
                    exit_ticks=[exit_tick(w['nominal'],w['valid']),exit_tick(raw_wave,raw_valid),exit_tick(actual_wave,actual_valid)]
                    n=min(len(actual),len(nom))
                    record=dict(arm=arm,path=path,scale=scale,
                        metrics=[dict(metric=k,baseline=float(baseline[k]),actual=float(truth[k]),
                            affine_signal=float(predicted[k]),exact_geometry_of_affine_state=float(raw[k]),
                            geometry_linearization_error=float(raw[k]-predicted[k]),
                            remaining_state_prediction_error=float(truth[k]-raw[k])) for k in predicted],
                        physical_peak_tick=dict(zip(['baseline','affine_state','actual'],ticks)),
                        out_edge_peak_tick=dict(zip(['baseline','affine_state','actual'],exit_ticks)),
                        physical_peak_speed_mm_s=[float(np.linalg.norm(a[t,2:4])) for a,t in zip([nom,affine,actual],ticks)],
                        moving_mask_changes=int(np.count_nonzero(w['valid'][:n,4]!=actual_valid[:n,4])),
                        maximum_velocity_prediction_error_mm_s=float(np.linalg.norm(actual[:n,2:4]-affine[:n,2:4],axis=1).max()))
                    records.append(record)
                    peak=next(r for r in record['metrics'] if r['metric']=='full_physical_peak_curvature_mm_inv')
                    print('PREDICTION_LAYER',arm,path,scale,'peak',peak,'ticks',ticks,flush=True)
                    assert time.perf_counter()-start<85
    write_json(stage/'results.json',dict(parent_sha256=cd['parent_sha256'],records=records,
        attribution_scope='saved_same_center_directions_only',goal_complete=False))
    receipt(stage,dict(status='completed_cruise_prediction_layer_audit',new_actor_decisions=0,
        new_physics_ticks=0,wall_s=time.perf_counter()-start))


if __name__=='__main__':main()
