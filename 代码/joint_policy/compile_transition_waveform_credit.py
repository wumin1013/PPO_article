"""Compile complete shared-parameter waveform credit and audit existing probes."""
import argparse
import time
import os
import numpy as np
from transition_waveform_model import chain_responses,signal_jacobians
from prepare_current_waveform_credit import wave_signals
from energy_waveform_credit import energy_signals
from waveform_constraint_model import waveform_facets
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--collection',default='canonical_gate_band_complete_transition_credit_20260917')
    parser.add_argument('--reference',default='canonical_gate_band_coherent_parameter_credit_20260917')
    parser.add_argument('--stage',default='canonical_gate_band_transition_waveform_model_20260917')
    parser.add_argument('--reuse-waveforms')
    parser.add_argument('--reuse-collection')
    args=parser.parse_args()
    assert bool(args.reuse_waveforms)==bool(args.reuse_collection)
    source=BASE/args.collection
    probes=BASE/args.reference
    completed=read(source/'receipt.json');assert completed['status']=='completed_current_complete_transition_credit'
    data=read(source/'results.json');physical=read(probes/'results.json')
    assert data['parent_sha256']==physical['parent_sha256']
    reused_sources=[];old_records={};old_locals={}
    if args.reuse_waveforms:
        old_waves=BASE/args.reuse_waveforms;old_collection=BASE/args.reuse_collection
        old=read(old_waves/'results.json');old_data=read(old_collection/'results.json')
        assert old['parent_sha256']==old_data['parent_sha256']==data['parent_sha256']
        bindings=read(old_waves/'registration.json')['source_sha256']
        for p in (probes/'results.json',probes/'basis.npz',BASE/'transition_waveform_model.py',BASE/'waveform_constraint_model.py',BASE/'energy_waveform_credit.py'):
            assert bindings[p.relative_to(BASE.parent.parent).as_posix()]==sha(p)
        old_records={r['path']:r for r in old['records']};old_locals={r['path']:r for r in old_data['records']}
        reused_sources=[old_waves/'results.json',old_waves/'receipt.json',old_collection/'results.json',old_collection/'receipt.json']
    stage=register(args.stage,
        [__file__,source/'results.json',source/'receipt.json',probes/'basis.npz',
         probes/'results.json',probes/'receipt.json',BASE/'transition_waveform_model.py',
         BASE/'waveform_constraint_model.py',BASE/'energy_waveform_credit.py']+reused_sources,
        dict(new_actor_decisions=0,new_physics_ticks=0,wall_s=360,shared_parameters=768,
             current_paths=len(data['records']),existing_complete_variants=len(physical['candidates']),parameter_updates=0),
        'Chain the completed current local state/action response matrices into all '
        '768 shared residual parameters. Differentiate the existing original1ms '
        'edge positions, heading, XY and physical curvature, angular rate and '
        'signed energy density analytically. Check these geometric derivatives '
        'against finite changes of recorded XY/v/a, and verify projection onto '
        'the measured parameter directions matches direct response chaining. Compare '
        'full predicted waves and shape quantities with the already completed '
        'bound complete policy variants, including terminal tails. Save XY response '
        'matrices for conservative local displacement-vs-corridor-margin bounds; '
        'no derivative through discrete completion time. Compile square-density '
        'energies after composition, without changing any shape metric. This '
        'prepares a bounded model-based learning proposal, not an accepted policy.')
    start=time.perf_counter();records=[];checks=[]
    with np.load(probes/'basis.npz') as z:
        directions=z['basis'].copy()*z['amplitudes'][:,None]
        check_direction=int(np.argmax(z['amplitudes']))
    names=physical['metric_names'];base=physical['baseline'];shape={r['path']:r for r in base['summary']['shapes']}
    try:
        for local_record in data['records']:
            path=local_record['path'];j=next(j for j,r in enumerate(base['results']) if r['original_path']==path)
            if path in old_locals and sha(source/local_record['file'])==sha(old_collection/old_locals[path]['file']):
                record=dict(old_records[path],reused_local_model_sha256=sha(source/local_record['file']))
                for key in ('xy_file','waveform_file'):
                    if key in record:record[key]=os.path.relpath(old_waves/record[key],stage)
                records.append(record)
                write_json(stage/'results.json',dict(parent_sha256=data['parent_sha256'],parent_model=data['parent_model'],
                    records=records,metric_names=names,goal_complete=False))
                continue
            with np.load(source/local_record['file']) as z:local={k:z[k].copy() for k in z.files}
            ref,rows=local['reference'],local['rows']
            response=chain_responses(local)
            projected=np.einsum('tcd,jd->tcj',response,directions)
            direct=chain_responses(local,directions)
            projection_error=float(abs(projected-direct).max());assert projection_error<1e-7
            xy_file=f'path{path}_xy.npz'
            np.savez_compressed(stage/xy_file,nominal=rows[:,:2],gradient=response[:,:2])
            record=dict(path=path,xy_file=xy_file,segment_max_distance_mm=base['metrics'][j]['values'][-2],
                        time_s=base['results'][j]['completion_time_s'],projection_identity_error=projection_error)
            if path in shape:
                nominal,valid,gradient,density,dvalid,dgradient=signal_jacobians(ref,rows,response)
                # The independent numerical check perturbs raw trajectory variables,
                # rather than comparing two algebraic forms of the same derivative.
                small=1e-3;delta=direct[:,:,check_direction];plus=rows.copy();minus=rows.copy()
                plus[:,:6]+=small*delta;minus[:,:6]-=small*delta
                ps,pv=wave_signals(ref,plus);ms,mv=wave_signals(ref,minus)
                pe,pev=energy_signals(ref,plus);me,mev=energy_signals(ref,minus)
                signal_fd=(ps-ms)/(2*small);energy_fd=(pe-me)/(2*small)
                signal_exact=np.einsum('tcd,d->tc',gradient,directions[check_direction])
                energy_exact=np.einsum('tcd,d->tc',dgradient,directions[check_direction])
                common=valid&pv&mv;ecommon=dvalid&pev&mev
                se=float(abs(signal_fd-signal_exact)[common].max())
                ee=float(abs(energy_fd-energy_exact)[ecommon].max())
                assert se<1e-3 and ee<1e-5,(path,se,ee)
                constant=(density**2).sum(0);linear=np.einsum('tc,tcd->cd',2*density,dgradient)
                quadratic=np.array([dgradient[:,c].T@dgradient[:,c] for c in range(2)])
                for c,name in enumerate(('full_xy_normalized_energy','full_physical_normalized_energy')):
                    assert abs(constant[c]-base['metrics'][j]['values'][names.index(name)])<1e-9
                wavefile=f'path{path}_waveform.npz'
                np.savez_compressed(stage/wavefile,nominal=nominal,gradient=gradient,valid=valid,
                    energy_constant=constant,energy_linear=linear,energy_quadratic=quadratic,
                    reference=ref,turn_deg=shape[path]['turn_deg'])
                errors=[]
                for candidate in physical['candidates']:
                    direction=candidate['sign']*directions[candidate['direction']]
                    with np.load(BASE/candidate['traces'][j]) as z:actual=z['rows'].copy()
                    truth,tvalid=wave_signals(ref,actual,len(rows))
                    prediction=nominal+np.einsum('tcd,d->tc',gradient,direction)
                    common=valid&tvalid;rmse=[];signal=[]
                    for c in range(6):
                        use=common[:,c]
                        rmse.append(float(np.sqrt(np.mean((prediction[use,c]-truth[use,c])**2))))
                        signal.append(float(np.sqrt(np.mean((truth[use,c]-nominal[use,c])**2))))
                    facets=waveform_facets(nominal,gradient,valid,shape[path]['turn_deg'],direction)
                    predicted={n:float(v[0]) for n,v in facets.items()}
                    e=constant+linear@direction+np.einsum('d,cdk,k->c',direction,quadratic,direction)
                    predicted.update(zip(('full_xy_normalized_energy','full_physical_normalized_energy'),e.tolist()))
                    actual_values=dict(zip(names,candidate['metrics'][j]['values']))
                    errors.append(dict(direction=candidate['direction'],sign=candidate['sign'],
                        rmse=rmse,measured_change_rms=signal,predicted=predicted,
                        actual={n:actual_values[n] for n in predicted}))
                record.update(waveform_file=wavefile,shape=shape[path],errors=errors,
                    derivative_check=dict(signal_max_abs_error=se,energy_max_abs_error=ee))
                print('COMPILE_TRANSITION_WAVE',path,'heading_error',sum(r['rmse'][2] for r in errors),flush=True)
            records.append(record)
            write_json(stage/'results.json',dict(parent_sha256=data['parent_sha256'],parent_model=data['parent_model'],
                records=records,metric_names=names,goal_complete=False))
            assert time.perf_counter()-start<350
        receipt(stage,dict(status='compiled_complete_transition_waveform_model',paths=len(records),
            reused_unchanged_paths=sum('reused_local_model_sha256' in r for r in records),
            new_actor_decisions=0,new_physics_ticks=0,wall_s=time.perf_counter()-start))
    except Exception as exc:
        receipt(stage,dict(status='failed_complete_transition_waveform_compilation',paths=len(records),
            error=str(exc),new_actor_decisions=0,new_physics_ticks=0,wall_s=time.perf_counter()-start))
        raise


if __name__=='__main__':main()
