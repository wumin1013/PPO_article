"""Locate current policy-composition mismatch on original execution timestamps."""
import time
import numpy as np
from geometry_spec import Corridor
from prepare_current_waveform_credit import wave_signals
from search_complete_return import read
from study_io import BASE,register,receipt,write_json


def main():
    source=BASE/'canonical_gate_band_half_step_waveform_credit_20260917'
    probe=BASE/'canonical_gate_band_half_step_locality_probe_20260917'
    direction_source=BASE/'canonical_gate_band_half_step_locality_direction_20260917'
    parent=BASE/'canonical_gate_band_quadratic_error_learned_update_20260917'
    stage=register('canonical_gate_band_half_step_locality_event_audit_20260917',
        [__file__,source/'results.json',source/'receipt.json',probe/'results.json',probe/'receipt.json',
         direction_source/'direction.npz',parent/'training_selection.json',BASE/'continuous_gate_guidance.py',
         BASE/'prepare_current_waveform_credit.py',BASE/'geometry_spec.py'],
        dict(paths=[8,14,15],new_actor_decisions=0,new_physics_ticks=0,wall_s=60),
        'Locate the current small whole-policy mismatch on original1ms timestamps. '
        'Relate angular-rate and cumulative reversal errors to the guide gate band '
        'and observed V/A/J activity. No causal attribution from correlation alone; '
        'this selects a later bounded mechanism check. No smoothing, limit change, '
        'policy update, new simulation or sealed data.')
    start=time.perf_counter();data=read(probe/'results.json');selection=read(parent/'training_selection.json')
    with np.load(direction_source/'direction.npz') as z:direction=z['direction'].ravel()
    records=[]
    for path in (8,14,15):
        j=next(j for j,r in enumerate(selection['results']) if r['original_path']==path)
        with np.load(BASE/selection['traces'][j]) as z:ref,rows=z['reference'].copy(),z['rows'].copy()
        corridor=Corridor(ref);gate=corridor.gates[0]
        with np.load(source/f'path{path}_waveform.npz') as z:
            nominal=z['nominal'].copy();jac=z['gradient'].copy();valid=z['nominal_valid'].copy()
            jac[~z['common_valid']]=0.
        response=np.einsum('tcd,d->tc',jac,direction)
        def gate_ticks(r):
            h=(np.vstack([ref[0],r[:-1,:2]])-gate.point)@gate.normal
            return h,np.flatnonzero((abs(h)<.05)&(r[:,12]==0)).tolist()
        h0,gate0=gate_ticks(rows)
        for variant in data['candidates']:
            j=next(j for j,r in enumerate(variant['results']) if r['original_path']==path)
            with np.load(BASE/variant['traces'][j]) as z:actual_rows=z['rows'].copy()
            actual,actual_valid=wave_signals(ref,actual_rows,len(nominal))
            predicted=nominal+variant['step']*response
            actual_reverse=np.where(actual_valid[:,5],np.maximum(-actual[:,5],0.)*.001,0.)
            predicted_reverse=np.where(valid[:,5],np.maximum(-predicted[:,5],0.)*.001,0.)
            increments=actual_reverse-predicted_reverse;cumulative=np.cumsum(increments)
            heading_error=actual[:,2]-predicted[:,2]
            h,gate1=gate_ticks(actual_rows)
            thresholds={}
            for threshold in (.001,.01,.05):
                indices=np.flatnonzero(valid[:,2]&actual_valid[:,2]&(abs(heading_error)>threshold))
                thresholds[str(threshold)]=None if not len(indices) else int(indices[0])
            strongest=np.argsort(abs(increments))[-8:][::-1]
            detail=[]
            for tick in strongest:
                r=actual_rows[tick]
                detail.append(dict(tick=int(tick),time_s=float((tick+1)*.001),
                    signed_reversal_error_increment_deg=float(increments[tick]),
                    actual_heading_error_deg=float(heading_error[tick]),
                    gate_distance_mm=float(h[tick]),parent_gate_distance_mm=float(h0[tick]),
                    speed_mm_s=float(np.linalg.norm(r[2:4])),
                    acceleration_mm_s2=float(np.linalg.norm(r[4:6])),
                    jerk_mm_s3=float(np.linalg.norm(r[6:8])),guard=bool(r[11])))
            windows=[]
            for begin in range(0,len(rows),20):
                value=float(increments[begin:begin+20].sum())
                if abs(value)>1e-4:windows.append(dict(begin_tick=begin,reversal_error_deg=value))
            record=dict(path=path,step=variant['step'],total_reversal_prediction_error_deg=float(cumulative[-1]),
                heading_error_first_tick=thresholds,parent_guide_band_ticks=gate0,
                actual_guide_band_ticks=gate1,strongest_increments=detail,windows=windows)
            records.append(record)
            np.savez_compressed(stage/f'p{path}_{variant["step"]:+g}.npz',
                nominal=nominal,predicted=predicted,actual=actual,nominal_valid=valid,
                actual_valid=actual_valid,cumulative_reversal_error=cumulative,
                parent_gate_distance=h0,actual_gate_distance=h)
            print('LOCALITY_EVENT',dict(path=path,step=variant['step'],heading_first=thresholds,
                parent_band=gate0,actual_band=gate1,total_error=record['total_reversal_prediction_error_deg'],
                top_windows=sorted(windows,key=lambda r:abs(r['reversal_error_deg']),reverse=True)[:3]),flush=True)
    write_json(stage/'results.json',dict(records=records,new_physics_ticks=0,causality_established=False,goal_complete=False))
    receipt(stage,dict(status='located_saved_locality_waveform_errors',new_actor_decisions=0,
        new_physics_ticks=0,wall_s=time.perf_counter()-start,goal_complete=False))


if __name__=='__main__':main()
