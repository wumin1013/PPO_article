"""Bind observed small-step shape errors to the current physical update metric."""
import time
import argparse
import numpy as np
from waveform_constraint_model import waveform_facets, SIGNED_SHAPE_CHANNELS
from cruise_update_geometry import CruiseUpdateGeometry
from prepare_current_waveform_credit import wave_signals
from search_complete_return import read
from study_io import BASE,register,receipt,write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--waveforms',default='canonical_gate_band_duration_corrected_waveforms_20260918')
    parser.add_argument('--reference',default='canonical_gate_band_duration_small_reference_20260918')
    parser.add_argument('--geometry',default='canonical_gate_band_retained_center_geometries_20260918')
    parser.add_argument('--stage',default='canonical_gate_band_duration_small_shape_error_20260918')
    parser.add_argument('--radius',type=float,default=.007)
    parser.add_argument('--signed-entry-errors',action='store_true')
    parser.add_argument('--signed-shape-errors',action='store_true')
    args=parser.parse_args()
    waves=BASE/args.waveforms
    reference=BASE/args.reference
    geometry=BASE/args.geometry
    physical=read(reference/'results.json');wd=read(waves/'results.json')
    parent=physical['parent_sha256'];assert wd['parent_sha256']==parent
    current= CruiseUpdateGeometry(geometry,'physical',parent)
    stage=register(args.stage,
        [__file__,waves/'results.json',waves/'receipt.json',reference/'results.json',reference/'receipt.json',
         reference/'basis.npz',geometry/'geometry.npz',geometry/'results.json',geometry/'receipt.json',
         BASE/'waveform_constraint_model.py',BASE/'cruise_update_geometry.py',
         BASE/'prepare_current_waveform_credit.py',BASE/'audit_current_waveform_credit.py'],
        dict(new_actor_decisions=0,new_physics_ticks=0,wall_s=60,
             measured_complete_paths=sum(len(c['results']) for c in physical['candidates']),
             next_learning_metric_radius=args.radius,subsequent_error_inflations=0,
             signed_entry_errors=args.signed_entry_errors,signed_shape_errors=args.signed_shape_errors),
        'Use only the completed signed small responses at the bound current center to '
        'measure local positive shape-prediction errors in the unchanged physical '
        'parameter metric. The corrected duration derivatives and explicit duration '
        'guards are required. Retain broad-step failures separately; no old-center '
        'uncertainty is borrowed. Record observed quadratic rates on existing positive '
        'shape quantities, with zero-bound quantities audited explicitly. A zero-bound '
        'metric actually regressing invalidates this scalar error model. An explicitly '
        'requested signed-waveform model instead bounds the signed position/heading '
        'error at every1ms sample BEFORE clipping to a violation maximum; this retains '
        'zero outward tolerance and records the scalar-model failure separately. These '
        'directions do not guarantee a radius or new direction. One new constrained '
        'solve within the registered local radius must pass fresh signed '
        'complete checks before learned0.5/1 candidates. Register1% minimum actual '
        'positive-shape-total improvement for this small intermediate update, preserving '
        'every corner metric, full64 success, time and measured99mm/s cruise. This '
        'replaces the old large-step15% proposal target only for the smaller new '
        'experiment; no physical/shape limit, previous result or final goal is relaxed. '
        'Budgets: two180s offline solves,120s/6000decisions paired diagnostic, '
        '300s/16000decisions candidate/full64 evaluation; stop direction on any '
        'shape regression and analyze mechanism, with no repeated rate inflation.')
    start=time.perf_counter();names=physical['metric_names'];observations=[];coefficients=[];waveform_bounds=[]
    with np.load(reference/'basis.npz') as z:directions=z['basis']*z['amplitudes'][:,None]
    norm2_by_direction={c['direction']:float(directions[c['direction']]@current.metric@directions[c['direction']])
                        for c in physical['candidates']}
    assert args.radius**2<max(norm2_by_direction.values())
    base={r['path']:r['values'] for r in physical['baseline']['metrics']}
    for record in wd['records']:
        if 'waveform_file' not in record:continue
        path=record['path'];rates={}
        with np.load(waves/record['waveform_file']) as z:w={k:z[k].copy() for k in z.files}
        targets=(SIGNED_SHAPE_CHANNELS if args.signed_shape_errors else
                 {'in_edge_outward_mm':(0,-1.)} if args.signed_entry_errors else {})
        signed_rates={n:np.zeros(len(w['nominal'])) for n in targets}
        signed_valid={n:w['valid'][:,ch].copy() for n,(ch,sgn) in targets.items()}
        for c in physical['candidates']:
            assert c['summary']['successes']==29
            x=c['sign']*directions[c['direction']]
            norm2=norm2_by_direction[c['direction']]
            prediction={n:float(v[0]) for n,v in waveform_facets(w['nominal'],w['gradient'],w['valid'],float(w['turn_deg']),x).items()}
            energy=w['energy_constant']+w['energy_linear']@x+np.einsum('d,cdk,k->c',x,w['energy_quadratic'],x)
            prediction.update(zip(('full_xy_normalized_energy','full_physical_normalized_energy'),energy))
            actual=next(r['values'] for r in c['metrics'] if r['path']==path)
            if targets:
                j=next(j for j,r in enumerate(c['metrics']) if r['path']==path)
                with np.load(BASE/c['traces'][j]) as z:ref,rows=z['reference'].copy(),z['rows'].copy()
                truth,truth_valid=wave_signals(ref,rows,len(w['nominal']))
                for name,(channel,sign) in targets.items():
                    predicted=w['nominal'][:,channel]+w['gradient'][:,channel]@x
                    valid=truth_valid[:,channel]
                    signed_valid[name]|=valid
                    error=np.where(valid,np.maximum(sign*(truth[:,channel]-predicted),0.)/norm2,0.)
                    signed_rates[name]=np.maximum(signed_rates[name],error)
            for index,name in enumerate(names[:names.index('segment_max_distance_mm')]):
                error=float(actual[index]-prediction[name])
                if name in targets:
                    observations.append(dict(path=path,metric=name,direction=c['direction'],sign=c['sign'],
                        actual=actual[index],underprediction=error,signed_waveform_model=True))
                    continue
                if base[path][index]<=1e-8:
                    assert actual[index]<=1e-8,(path,name,c['sign'],actual[index])
                    continue
                rate=max(0.,error)/norm2
                rates[name]=max(rates.get(name,0.),rate)
                observations.append(dict(path=path,metric=name,direction=c['direction'],sign=c['sign'],underprediction=error,
                    measured_metric_norm=float(np.sqrt(norm2)),rate=rate))
        coefficients.extend(dict(path=path,metric=name,rate=rate) for name,rate in rates.items() if rate>0.)
        for name,(channel,sign) in targets.items():
            filename=f'path{path}_{name}_error.npz'
            rate=signed_rates[name];valid=signed_valid[name]
            np.savez_compressed(stage/filename,rate=rate,valid=valid)
            waveform_bounds.append(dict(path=path,metric=name,channel=channel,sign=sign,
                offset=-abs(float(w['turn_deg'])) if name=='heading_overshoot_deg' else 0.,
                file=filename,maximum_rate=float(rate.max()),signed_waveform_before_clipping=True,
                extra_valid_samples=int(np.sum(valid&~w['valid'][:,channel]))))
    write_json(stage/'error_model.json',dict(parent_sha256=parent,parameter_geometry=current.identity,
        coefficients=coefficients,observations=observations,waveform_bounds=waveform_bounds,guaranteed_bound=False,
        raw_prediction_recomputed=True,goal_complete=False))
    receipt(stage,dict(status='prepared_current_local_shape_errors',coefficient_count=len(coefficients),
        new_actor_decisions=0,new_physics_ticks=0,wall_s=time.perf_counter()-start,goal_complete=False))
    print('LOCAL_SHAPE_ERRORS',len(coefficients),'observed_radii',
          {k:float(np.sqrt(v)) for k,v in norm2_by_direction.items()},flush=True)


if __name__=='__main__':main()
