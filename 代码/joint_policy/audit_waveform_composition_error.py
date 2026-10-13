"""Current both-sign whole-policy model error from already saved trajectories."""
import argparse
import time
import numpy as np
from prepare_current_waveform_credit import wave_signals
from search_complete_return import read
from study_io import BASE,register,receipt,write_json


def main():
    parser=argparse.ArgumentParser()
    for name in ('source','probe','waveforms','stage'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    source=BASE/args.source;probe=BASE/args.probe;wave=BASE/args.waveforms
    opt=read(source/'optimizer.json');data=read(probe/'results.json')
    assert (probe/'receipt.json').exists()
    assert read(wave/'results.json')['parent_sha256']==opt['parent_sha256']
    stage=register(args.stage,[__file__,source/'optimizer.json',source/'direction.npz',
        source/'receipt.json',probe/'results.json',probe/'comparison.json',probe/'receipt.json',
        wave/'results.json',wave/'receipt.json',BASE/'prepare_current_waveform_credit.py'],
        dict(new_actor_decisions=0,new_physics_ticks=0,wall_s=45),
        'Analyze BOTH small complete-policy signs for the current parent. Separate '
        'odd secant bias and even response error, compare active curvature times, '
        'and estimate one-sided shape-prediction error per action-metric norm. '
        'A later local solve may tighten curvature constraints using these current '
        'measured errors times the proposed norm. This is an empirical allowance '
        'on one direction, not a proved bound or a physical success. No resampling, '
        'new rollouts, development or sealed access.')
    start=time.perf_counter();names=data['metric_names'];norm=opt['projection']['metric_norm']
    with np.load(source/'direction.npz') as z:
        direction=z['direction'].ravel().copy();ids=z['path_id'].tolist()
        predicted={float(s):dict(zip(ids,p)) for s,p in zip(z['prediction_scales'],z['predictions'])}
    baseline={r['path']:np.asarray(r['values']) for r in data['baseline_metrics']}
    errors=np.zeros((len(ids),len(names)));records=[];peaks=[]
    for path in ids:
        pair=[]
        for candidate in data['candidates']:
            row=next(r for r in candidate['metrics'] if r['path']==path)
            assert row['success']
            step=candidate['step'];change=np.asarray(row['values'])-baseline[path]
            error=change-predicted[step][path]
            errors[ids.index(path)]=np.maximum(errors[ids.index(path)],error/(abs(step)*norm))
            pair.append((step,error))
            records.append(dict(path=path,step=step,actual_change=change.tolist(),
                predicted_change=predicted[step][path].tolist(),error=error.tolist()))
        records[-1]['odd_error_per_unit']=((pair[1][1]-pair[0][1])/(pair[1][0]-pair[0][0])).tolist()
        records[-1]['even_error']=((pair[1][1]+pair[0][1])/2).tolist()
        if not (wave/f'path{path}_waveform.npz').exists():continue
        with np.load(wave/f'path{path}_waveform.npz') as z:
            nominal=z['nominal'].copy();valid=z['nominal_valid'].copy()
            gradient=z['gradient'].copy();gradient[~z['common_valid']]=0.
            ref=z['reference'].copy()
        response=np.einsum('tcd,d->tc',gradient,direction)
        for candidate in data['candidates']:
            j=next(i for i,r in enumerate(candidate['results']) if r['original_path']==path)
            with np.load(BASE/candidate['traces'][j]) as z:actual,av=wave_signals(ref,z['rows'])
            predicted_wave=nominal+candidate['step']*response
            for channel in (3,4):
                def peak(value,mask):
                    ticks=np.flatnonzero(mask[:,channel]);tick=int(ticks[abs(value[ticks,channel]).argmax()])
                    return dict(tick=tick,magnitude=float(abs(value[tick,channel])))
                peaks.append(dict(path=path,step=candidate['step'],channel=channel,
                    parent=peak(nominal,valid),predicted=peak(predicted_wave,valid),actual=peak(actual,av)))
    selected=[n for n in names if 'curvature' in n or 'energy' in n]
    result=dict(parent_sha256=opt['parent_sha256'],source=args.source,probe=args.probe,
        metric_names=names,path_ids=ids,measured_direction_norm=norm,
        one_sided_error_per_norm=errors.tolist(),proposed_margin_metrics=selected,
        records=records,peaks=peaks,guaranteed_bound=False,goal_complete=False)
    write_json(stage/'results.json',result)
    receipt(stage,dict(status='completed_current_waveform_composition_error_audit',
        new_actor_decisions=0,new_physics_ticks=0,wall_s=time.perf_counter()-start))
    print('COMPOSITION_ERROR',dict(path9_peaks=[p for p in peaks if p['path']==9],
        path9_error_per_norm=errors[ids.index(9)].tolist()),flush=True)


if __name__=='__main__':main()
