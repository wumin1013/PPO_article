"""Prepare full isolated-corner waveforms from completed current-policy credit."""
import argparse
import time
import numpy as np
import torch
from canonical_gate_state import CanonicalGatePolicy
from diagnose_local_basis_fit import feature_batches
from audit_current_waveform_credit import signals,CHANNELS
from audit_xy_curvature_resolution import curvature_metrics
from search_complete_return import read
from study_io import BASE,ROOT,register,receipt,write_json,sha

WAVE_CHANNELS=CHANNELS+('signed_angular_rate_deg_s',)


def wave_signals(reference,rows,length=None):
    value,valid=signals(reference,rows,length)
    rows=np.asarray(rows)
    if len(rows)<len(value):rows=np.vstack([rows,np.repeat(rows[-1:],len(value)-len(rows),axis=0)])
    rows=rows[:len(value)]
    u,v=np.diff(reference,axis=0);sign=np.sign(u[0]*v[1]-u[1]*v[0])
    speed2=np.sum(rows[:,2:4]**2,axis=1)
    rate=sign*(rows[:,2]*rows[:,5]-rows[:,3]*rows[:,4])/np.maximum(speed2,1e-30)*180/np.pi
    return np.column_stack([value,rate]),np.column_stack([valid,speed2>25.])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--credit',required=True)
    parser.add_argument('--parent',required=True)
    parser.add_argument('--arm',required=True)
    parser.add_argument('--stage',required=True)
    parser.add_argument('--paths',type=int,nargs='+',default=list(range(16)))
    args=parser.parse_args();torch.set_num_threads(1)
    credit=BASE/args.credit;parent=BASE/args.parent
    assert (credit/'receipt.json').exists(),'Only complete credit may be consumed'
    selected=next(r for r in read(parent/'results.json')['candidates']
        if r.get('arm',r.get('tag'))==args.arm or r.get('model')==args.arm)
    model=parent/selected['model'];metadata=torch.load(model,map_location='cpu',weights_only=False)
    assert metadata['architecture'] in ('canonical_gate_kernel','gate_band_canonical_kernel')
    bindings=read(credit/'registration.json')['source_sha256']
    assert bindings[model.relative_to(ROOT).as_posix()]==sha(model)
    raw=read(credit/'results.json');baseline={r['path']:r for r in raw['baselines']}
    assert all(path in baseline for path in args.paths)
    count_expected=6*sum(1 for q in raw['queries'] if q['path'] in args.paths)
    stage=register(args.stage,[__file__,credit/'results.json',credit/'queries.npz',
        credit/'receipt.json',model,BASE/'audit_current_waveform_credit.py',
        BASE/'diagnose_local_basis_fit.py',BASE/'canonical_gate_state.py'],
        dict(paths=args.paths,new_physics_ticks=0,new_actor_decisions=0,
             suffix_files=count_expected,wall_s=360),
        'Reuse every completed current-policy axial suffix on all explicitly listed '
        'isolated TRAINING corners. Derive shared-parameter sensitivities of raw1ms '
        'inside positions, signed heading, rawXY curvature, physical curvature and '
        'physical angular rate. This extends the confirmed moving-extremum repair '
        'to all isolated cases, without recollecting physics or using previous '
        'parents\' derivatives. A future local model can take maxima/minima or '
        'negative angular-rate sums after waveform prediction. Save common moving '
        'support and its difference from full nominal support explicitly. Stopped '
        'endpoint extension is for timestamp alignment only, never completion '
        'scoring. No model update, new objective, ideal trajectory or test access.')
    start=time.perf_counter();policy=CanonicalGatePolicy(256);policy.load_state_dict(metadata['policy'])
    with np.load(credit/'queries.npz') as z:observations=z['observation'].copy()
    features=feature_batches(policy,observations)*policy.residual_scale.detach().numpy()[None,:,None]
    lookup={(r['query'],r['axis'],r['direction']):r for r in raw['records']}
    records=[];files=0
    for path in args.paths:
        b=baseline[path]
        with np.load(BASE/b['trace']) as z:ref,rows=z['reference'].copy(),z['rows'].copy()
        assert len(ref)==3
        nominal,nominal_valid=wave_signals(ref,rows);common=nominal_valid.copy()
        gradient=np.zeros((len(rows),len(WAVE_CHANNELS),768))
        for q in (q for q in raw['queries'] if q['path']==path):
            for head in range(3):
                pair=[lookup[q['query'],head,sign] for sign in (-1,1)];values=[]
                for r in pair:
                    assert r['success'],'A failed suffix is not a complete waveform derivative'
                    with np.load(BASE/r['trace']) as z:full=np.vstack([rows[:q['anchor']*20],z['suffix_rows']])
                    value,valid=wave_signals(ref,full,len(rows));common&=valid;values.append(value);files+=1
                slope=(values[1]-values[0])/(pair[1]['latent_offset']-pair[0]['latent_offset'])
                # Only this head's residual coefficient block has a direct action Jacobian.
                gradient[:,:,head*256:(head+1)*256]+=slope[:,:,None]*features[q['query'],head]
        full_reverse=float(np.maximum(-nominal[nominal_valid[:,5],5],0.).sum()*.001)
        common_reverse=float(np.maximum(-nominal[common[:,5],5],0.).sum()*.001)
        assert abs(full_reverse-b['shape']['full_physical_reverse_deg'])<1e-9
        peaks=[]
        for channel in (3,4):
            peaks.append(dict(channel=WAVE_CHANNELS[channel],
                full=float(abs(nominal[nominal_valid[:,channel],channel]).max()),
                common=float(abs(nominal[common[:,channel],channel]).max())))
        np.savez_compressed(stage/f'path{path}_waveform.npz',reference=ref,
            nominal=nominal,nominal_valid=nominal_valid,common_valid=common,
            gradient=gradient,channels=np.asarray(WAVE_CHANNELS))
        r=dict(path=path,baseline_trace=b['trace'],shape=b['shape'],curvature=curvature_metrics(ref,rows),
            full_reverse_deg=full_reverse,common_reverse_deg=common_reverse,
            unmodeled_nominal_reverse_deg=full_reverse-common_reverse,
            moving_peak_support=peaks,extra_excluded_samples=(nominal_valid.sum(0)-common.sum(0)).tolist())
        records.append(r)
        write_json(stage/'results.json',dict(records=records,channels=WAVE_CHANNELS,
            parent_model=str(model.relative_to(BASE)),parent_sha256=sha(model),suffix_files=files,goal_complete=False))
        assert time.perf_counter()-start<360
        print('CURRENT_WAVEFORM',path,'files',files,'unmodeled_reverse',r['unmodeled_nominal_reverse_deg'],flush=True)
    assert files==count_expected
    receipt(stage,dict(status='prepared_current_all_corner_waveform_credit',
        paths=len(records),suffix_files=files,new_physics_ticks=0,new_actor_decisions=0,
        wall_s=time.perf_counter()-start,goal_complete=False))


if __name__=='__main__':main()
