"""Check max/min active-time changes using existing current-policy suffixes."""
import time
import numpy as np
from search_complete_return import read
from study_io import BASE,register,receipt,write_json

CHANNELS=('inside_in_mm','inside_out_mm','signed_heading_deg',
          'signed_xy_curvature_mm_inv','signed_physical_curvature_mm_inv')


def signals(reference,rows,length=None):
    rows=np.asarray(rows)
    if length is not None:
        if len(rows)<length:
            assert np.linalg.norm(rows[-1,2:4])<.001
            rows=np.vstack([rows,np.repeat(rows[-1:],length-len(rows),axis=0)])
        else:rows=rows[:length]
    u,v=np.diff(reference,axis=0);u/=np.linalg.norm(u);v/=np.linalg.norm(v)
    sign=np.sign(u[0]*v[1]-u[1]*v[0])
    ni=sign*np.array([-u[1],u[0]]);no=sign*np.array([-v[1],v[0]])
    speed=np.linalg.norm(rows[:,2:4],axis=1)
    heading=np.rad2deg(np.unwrap(np.arctan2(rows[:,2:4]@ni,rows[:,2:4]@u)))
    physical=sign*(rows[:,2]*rows[:,5]-rows[:,3]*rows[:,4])/np.maximum(speed,1e-20)**3
    xy=np.vstack([reference[0],rows[:,:2]])
    segment=np.diff(xy,axis=0);distance=np.linalg.norm(segment,axis=1)
    a,b=segment[:-1],segment[1:];chord=np.linalg.norm(a+b,axis=1)
    valid_xy=(distance[:-1]>.005)&(distance[1:]>.005)&(chord>1e-12)
    raw=np.r_[sign*2*(a[:,0]*b[:,1]-a[:,1]*b[:,0])/
        np.maximum(distance[:-1]*distance[1:]*chord,1e-30),0.]
    value=np.column_stack([(rows[:,:2]-reference[1])@ni,(rows[:,:2]-reference[1])@no,
                           heading,raw,physical])
    valid=np.column_stack([np.ones((len(rows),2),bool),speed>5.,np.r_[valid_xy,False],speed>5.])
    return value,valid


def extremal_metrics(value,valid):
    return dict(in_edge_outward_mm=float(max(0.,-value[:,0].min())),
        out_edge_outward_mm=float(max(0.,-value[:,1].min())),
        negative_signed_heading_deg=float(max(0.,-value[valid[:,2],2].min())),
        full_xy_peak_curvature_mm_inv=float(abs(value[valid[:,3],3]).max()),
        full_physical_peak_curvature_mm_inv=float(abs(value[valid[:,4],4]).max()))


def main():
    credit=BASE/'canonical_gate_complete_credit_20260917'
    solve=BASE/'canonical_gate_current_parameter_solve_20260917'
    proposal=BASE/'canonical_gate_measured_margin_solve_20260917'
    learned=BASE/'canonical_gate_learned_shape_update_20260917'
    stage=register('canonical_gate_waveform_credit_audit_20260917',
        [__file__,credit/'results.json',credit/'receipt.json',solve/'gradients.npz',
         proposal/'direction.npz',learned/'results.json',learned/'receipt.json'],
        dict(paths=[2,3,13],new_physics_ticks=0,new_actor_decisions=0,
            wall_s=180,maximum_suffix_files=1200),
        'Larger complete-policy updates still fail despite correct aggregate '
        'credit. On failed training paths2/3/13, derive the actual1ms waveform '
        'Jacobian from already completed paired single-action suffixes and the '
        'same shared-parameter action Jacobian. Compare taking max/min AFTER '
        'waveform prediction with the existing scalar-extremum prediction. This '
        'tests whether moving active peak/extremum locations explain the scale '
        'error, without new physics or policy fitting. Preserve real1ms samples; '
        'no XY smoothing. Exclude heading/curvature samples lacking the existing '
        '>5mm/s support on any branch. If a complete suffix ends before a '
        'nominal timestamp, extend its already stopped endpoint for alignment '
        'only; never assign an earlier completion or change recorded time. '
        'This diagnoses the local credit operator, not all policy root causes.')
    start=time.perf_counter();raw=read(credit/'results.json');data=read(learned/'results.json')
    with np.load(solve/'gradients.npz') as z:jac=z['query_jacobian'].copy()
    with np.load(proposal/'direction.npz') as z:
        direction=z['direction'].ravel().copy();pred=z['predicted_changes'].copy()
        ids=z['path_id'].tolist();names=z['metric_names'].tolist()
    lookup={(r['query'],r['axis'],r['direction']):r for r in raw['records']}
    baselines={r['path']:r for r in raw['baselines']}
    count=0;records=[]
    for path in (2,3,13):
        with np.load(BASE/baselines[path]['trace']) as z:ref,rows=z['reference'].copy(),z['rows'].copy()
        nominal,common=signals(ref,rows);gradient=np.zeros((len(rows),len(CHANNELS),768))
        for q in (q for q in raw['queries'] if q['path']==path):
            for head in range(3):
                pair=[lookup[q['query'],head,sign] for sign in (-1,1)]
                values=[]
                for record in pair:
                    assert record['success']
                    with np.load(BASE/record['trace']) as z:
                        full=np.vstack([rows[:q['anchor']*20],z['suffix_rows']])
                    value,valid=signals(ref,full,len(rows));common&=valid;values.append(value);count+=1
                slope=(values[1]-values[0])/(pair[1]['latent_offset']-pair[0]['latent_offset'])
                gradient+=slope[:,:,None]*jac[q['query'],head]
        nominal_metrics=extremal_metrics(nominal,common)
        full_nominal_metrics=extremal_metrics(*signals(ref,rows))
        np.savez_compressed(stage/f'path{path}_waveform.npz',reference=ref,nominal=nominal,
            common_valid=common,gradient=gradient,channels=np.asarray(CHANNELS))
        delta=np.einsum('tcd,d->tc',gradient,direction)
        variants=[]
        for candidate in data['candidates']:
            step=candidate['step'];j=next(i for i,r in enumerate(candidate['results']) if r['original_path']==path)
            with np.load(BASE/candidate['traces'][j]) as z:actual_rows=z['rows'].copy()
            actual_value,actual_valid=signals(ref,actual_rows)
            actual=extremal_metrics(actual_value,actual_valid)
            waveform=extremal_metrics(nominal+step*delta,common)
            scalar={name:full_nominal_metrics[name]+step*pred[ids.index(path),names.index(name)] for name in actual}
            peaks={}
            for channel in (3,4):
                peak=lambda x,m:int(np.flatnonzero(m[:,channel])[abs(x[m[:,channel],channel]).argmax()])
                peaks[CHANNELS[channel]]=dict(nominal_tick=peak(nominal,common),
                    predicted_tick=peak(nominal+step*delta,common),actual_tick=peak(actual_value,actual_valid))
            variants.append(dict(step=step,actual=actual,waveform_prediction=waveform,
                scalar_prediction=scalar,peak_ticks=peaks))
        row=dict(path=path,nominal=full_nominal_metrics,common_support_nominal=nominal_metrics,
            unsupported_samples=(len(rows)-common.sum(axis=0)).tolist(),variants=variants)
        records.append(row)
        print('WAVEFORM_CREDIT',row,flush=True)
    assert count<=1200 and time.perf_counter()-start<180
    write_json(stage/'results.json',dict(records=records,suffix_files=count,channels=CHANNELS,goal_complete=False))
    receipt(stage,dict(status='completed_saved_waveform_credit_audit',suffix_files=count,
        new_physics_ticks=0,new_actor_decisions=0,wall_s=time.perf_counter()-start))


if __name__=='__main__':main()
