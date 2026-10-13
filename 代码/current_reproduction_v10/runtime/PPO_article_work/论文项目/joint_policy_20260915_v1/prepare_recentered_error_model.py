"""Recenter local model-error measurements using already completed policy pairs."""
import argparse
import time
import numpy as np
import torch
from waveform_constraint_model import waveform_facets
from prepare_current_waveform_credit import wave_signals
from solve_continuous_sensor_credit import vector
from search_complete_return import read
from study_io import BASE, register, receipt, write_json, sha


def main():
    parser = argparse.ArgumentParser()
    for name in ('source', 'waveforms', 'history', 'history-source', 'stage'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--energy')
    args = parser.parse_args(); torch.set_num_threads(1)
    source, wave, history = BASE/args.source, BASE/args.waveforms, BASE/args.history
    opt = read(source/'optimizer.json'); oldopt = read(BASE/args.history_source/'optimizer.json')
    data = read(history/'results.json'); model = BASE/opt['model']
    assert opt['parent_sha256']==sha(model)==read(wave/'results.json')['parent_sha256']
    metadata = torch.load(model, map_location='cpu', weights_only=False)
    plus = next(r for r in data['candidates'] if r['step']==1.)
    models = [BASE/oldopt['model'], history/plus['model']]
    variants = [data['baseline'], plus]
    assert sha(model)==sha(history/data['chosen']['model']) and data['chosen']['step']==.5
    assert all(r['summary']['successes']==len(r['results']) for r in variants)
    with np.load(source/'support.npz') as z: metric = z['metric'].copy()
    with np.load(source/'gradients.npz') as z:
        gradient=z['gradient'].copy(); ids=z['path_id'].tolist(); names=z['metric_names'].tolist()
    baselines=read(source/'baselines.json')['baselines']
    base={r['path']:vector(r,r['curvature']) for r in baselines}
    nominal={}; jac={}; valid={}; turn={}
    for b in baselines:
        p=b['path']
        if b['shape'] is None: continue
        with np.load(wave/f'path{p}_waveform.npz') as z:
            nominal[p]=z['nominal'].copy(); jac[p]=z['gradient'].copy()
            valid[p]=z['nominal_valid'].copy(); jac[p][~z['common_valid']]=0.
        turn[p]=b['shape']['turn_deg']
    energy={}; energy_sources=[]
    if args.energy:
        folder=BASE/args.energy; ed=read(folder/'results.json')
        assert ed['parent_sha256']==sha(model)
        energy_sources=[folder/'results.json',folder/'receipt.json']
        for r in ed['records']:
            file=folder/r['file'];energy_sources.append(file)
            with np.load(file) as z:
                energy[r['path']]=(z['constant'].copy(),z['linear'].copy(),z['quadratic'].copy(),z['metric_names'].tolist())
    directions=[]; norms=[]
    for file in models:
        md=torch.load(file,map_location='cpu',weights_only=False)
        assert md['architecture']==metadata['architecture']
        for k,v in metadata['policy'].items():
            if k!='residual_coefficients': assert torch.equal(v,md['policy'][k]),k
        x=(md['policy']['residual_coefficients'].double()-metadata['policy']['residual_coefficients'].double()).numpy().ravel()
        directions.append(x);norms.append(float(np.sqrt(x@metric@x)))
    assert max(abs(directions[0]+directions[1]))<1e-6
    assert abs(norms[0]-norms[1])<1e-7
    stage=register(args.stage,[__file__,model,*models,source/'optimizer.json',source/'gradients.npz',
        source/'support.npz',source/'baselines.json',source/'receipt.json',wave/'results.json',
        wave/'receipt.json',history/'results.json',history/'receipt.json',
        BASE/args.history_source/'optimizer.json',BASE/'waveform_constraint_model.py']+energy_sources,
        dict(new_actor_decisions=0,new_physics_ticks=0,wall_s=60,measured_directions=1,
             observed_parameter_norms=norms,energy_operator_source=args.energy),
        'The new center is the previously accepted half-step policy. Its old '
        'parent and full-step sibling are an already measured both-sign parameter '
        'pair around this center. Recompute their predictions using NEW current '
        'credit, current support norm and current full waveforms, never old '
        'gradients. Fit an empirical odd linear plus even quadratic upper '
        'remainder from this one paired direction. Cover both signs, including '
        'the rejected full-step sibling. Zero-boundary margins belong to signed '
        'waveforms before max, not after clipping. This is not a guarantee or '
        'new physical evidence; a new direction must receive fresh both-sign '
        '29-route verification. Physical limits and adoption tests are unchanged.')
    started=time.perf_counter();errors=[];field_errors=[];samples=[]
    descriptors={0:(0,-1.,0.),1:(1,-1.,0.),7:(2,-1.,0.),8:(2,1.,None)}
    def prediction(p,x):
        i=ids.index(p); value=base[p]+gradient[i]@x
        if p in nominal:
            for name,(v,_,_) in waveform_facets(nominal[p],jac[p],valid[p],turn[p],x).items():
                value[names.index(name)]=v
        if p in energy:
            constant,linear,q,en=energy[p]
            for c,n in enumerate(en):value[names.index(n)]=constant[c]+linear[c]@x+x@q[c]@x
        return value
    for side,(variant,x,norm) in enumerate(zip(variants,directions,norms)):
        lookup={r['path']:r for r in variant['metrics']}; error=[];fields={}
        for p in ids:
            actual=np.asarray(lookup[p]['values']);predicted=prediction(p,x)
            error.append(actual-predicted)
            samples.append(dict(path=p,side=side,norm=norm,error=(actual-predicted).tolist()))
            if p not in nominal: continue
            j=next(j for j,r in enumerate(variant['results']) if r['original_path']==p)
            with np.load(BASE/variant['traces'][j]) as z:
                actual_wave,mask=wave_signals(z['reference'],z['rows'],len(nominal[p]))
            predicted_wave=nominal[p]+np.einsum('tcd,d->tc',jac[p],x)
            for m,(channel,sign,offset) in descriptors.items():
                if base[p][m]>1e-10:continue
                supported=valid[p][:,channel]&mask[:,channel]
                e=np.zeros(len(nominal[p]));e[supported]=sign*(actual_wave-predicted_wave)[supported,channel]
                fields[p,m]=e
        errors.append(np.asarray(error));field_errors.append(fields)
    def coefficients(minus,plus):
        # Solve the two signed residual equations at their actual measured norms.
        n0,n1=norms
        odd=(plus*n0*n0-minus*n1*n1)/(n1*n0*n0+n0*n1*n1)
        even=(minus+odd*n0)/(n0*n0)
        return abs(odd),np.maximum(even,0.)
    linear,quadratic=coefficients(*errors)
    deficit=max(float(np.max(e-linear*n-quadratic*n*n)) for e,n in zip(errors,norms))
    assert deficit<1e-10
    field_descriptors=[];field_checks=[]
    for (p,m),minus in field_errors[0].items():
        if linear[ids.index(p),m]<=1e-12 and quadratic[ids.index(p),m]<=1e-12:continue
        plus_field=field_errors[1][p,m];c1,c2=coefficients(minus,plus_field)
        channel,sign,offset=descriptors[m]
        offset=-abs(turn[p]) if offset is None else offset
        file=stage/f'path{p}_{names[m]}_pointwise_error.npz'
        np.savez_compressed(file,linear=c1,quadratic=c2)
        field_descriptors.append(dict(path=p,metric=names[m],channel=channel,sign=sign,
            offset=offset,file=str(file.relative_to(BASE)),sha256=sha(file)))
        field_checks.append(dict(path=p,metric=names[m],maximum_pointwise_deficit=max(
            float(np.max(e-c1*n-c2*n*n)) for e,n in zip((minus,plus_field),norms))))
        linear[ids.index(p),m]=0.;quadratic[ids.index(p),m]=0.
    result=dict(parent_sha256=sha(model),metric_names=names,path_ids=ids,
        one_sided_error_per_norm=linear.tolist(),quadratic_error_per_norm_squared=quadratic.tolist(),
        proposed_margin_metrics=names[:names.index('segment_max_distance_mm')],
        waveform_error_descriptors=field_descriptors,zero_boundary_model_checks=field_checks,
        maximum_observed_upper_model_deficit=deficit,samples=samples,
        directions_nearly_symmetric=True,measured_norms=norms,guaranteed_bound=False,
        energy_operator_source=args.energy,goal_complete=False)
    write_json(stage/'results.json',result)
    receipt(stage,dict(status='recentered_saved_paired_policy_error_model',new_actor_decisions=0,
        new_physics_ticks=0,wall_s=time.perf_counter()-started,goal_complete=False))
    print('RECENTERED_ERROR',dict(norms=norms,deficit=deficit,zero_boundaries=field_checks),flush=True)


if __name__=='__main__':main()
