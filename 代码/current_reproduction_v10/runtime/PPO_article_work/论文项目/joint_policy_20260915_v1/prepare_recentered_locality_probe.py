"""Prepare a saved parameter direction for current-center error-locality checks."""
import argparse
import numpy as np
import torch
from waveform_constraint_model import waveform_facets
from solve_continuous_sensor_credit import vector
from search_complete_return import read
from study_io import BASE, register, receipt, write_json, sha


def main():
    parser=argparse.ArgumentParser()
    for name in ('source','waveforms','energy','direction-source','stage'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--last-iterate',action='store_true',help='Diagnostic challenge of an empirical-margin rejection')
    args=parser.parse_args()
    source,wave,energy,previous=[BASE/getattr(args,n) for n in ('source','waveforms','energy','direction_source')]
    opt=read(source/'optimizer.json');model=BASE/opt['model']
    assert sha(model)==opt['parent_sha256']==read(wave/'results.json')['parent_sha256']==read(energy/'results.json')['parent_sha256']
    direction_file=previous/('last_iterate.npz' if args.last_iterate else 'direction.npz')
    stage=register(args.stage,[__file__,model,source/'optimizer.json',source/'gradients.npz',source/'support.npz',
        source/'baselines.json',source/'receipt.json',wave/'results.json',wave/'receipt.json',
        energy/'results.json',energy/'receipt.json',direction_file,previous/'optimizer.json',
        previous/'receipt.json',BASE/'waveform_constraint_model.py'],
        dict(new_actor_decisions=0,new_physics_ticks=0,wall_s=30,probe_scales=[-.125,.125],
             empirical_margin_challenge=args.last_iterate),
        ('Use the saved last iterate of the bounded15% solve as a diagnostic '
         'challenge of the empirical norm-error admission model. Its nominal '
         'predictions and norm are reported, not declared physically feasible. '
         'Only a fresh both-sign29-route physical check is prepared. Actual '
         'shape, constraints, full stopping and every adoption test stay unchanged. '
         'No actor training or automatic larger-step permission is implied. '
         if args.last_iterate else
        'Prepare the previously measured parameter direction around the newly accepted '
        'half-step center. Use current waveforms and energy operators for predictions. '
        'The saved old-parent/full-step pair is at plus/minus0.5 around this center; '
        'fresh plus/minus0.125 checks change only parameter displacement. This tests '
        'whether coarse odd nonlinear response was misidentified as a norm-linear '
        'error floor. This direction is diagnostic only, not a new optimized proposal, '
        'and must not be sent to the larger-step learner. No new physics here.'))
    with np.load(direction_file) as z:direction=z['direction'].ravel().copy()
    with np.load(source/'support.npz') as z:metric=z['metric'].copy()
    with np.load(source/'gradients.npz') as z:
        gradient=z['gradient'].copy();ids=z['path_id'].tolist();names=z['metric_names'].tolist()
    base={r['path']:r for r in read(source/'baselines.json')['baselines']}
    nominal={};jac={};valid={};operators={}
    for p,b in base.items():
        if b['shape'] is None:continue
        with np.load(wave/f'path{p}_waveform.npz') as z:
            nominal[p]=z['nominal'].copy();jac[p]=z['gradient'].copy();valid[p]=z['nominal_valid'].copy()
            jac[p][~z['common_valid']]=0.
        with np.load(energy/f'path{p}_energy.npz') as z:
            operators[p]=(z['constant'].copy(),z['linear'].copy(),z['quadratic'].copy(),z['metric_names'].tolist())
    def prediction(scale):
        x=scale*direction;changes=np.einsum('pmd,d->pm',gradient,x)
        for p in nominal:
            b=vector(base[p],base[p]['curvature']);i=ids.index(p)
            for n,(v,_,_) in waveform_facets(nominal[p],jac[p],valid[p],base[p]['shape']['turn_deg'],x).items():
                changes[i,names.index(n)]=v-b[names.index(n)]
            constant,linear,q,en=operators[p]
            for c,n in enumerate(en):changes[i,names.index(n)]=constant[c]+linear[c]@x+x@q[c]@x-b[names.index(n)]
        return changes
    scales=np.array([-.125,.125])
    np.savez_compressed(stage/'direction.npz',direction=direction.reshape(3,256),
        predicted_changes=prediction(1.),prediction_scales=scales,
        predictions=np.asarray([prediction(s) for s in scales]),path_id=ids,metric_names=np.asarray(names))
    norm=float(np.sqrt(direction@metric@direction))
    assert norm<=.15+1e-10
    write_json(stage/'optimizer.json',dict(parent_sha256=sha(model),model=opt['model'],metric_names=names,
        path_ids=ids,eligible_for_small_probe=True,diagnostic_only=True,
        purpose='empirical_margin_challenge' if args.last_iterate else 'measured_direction_locality_at_new_center',
        projection=dict(metric_norm=norm),
        prediction_kind='affine_waveform_convex_shape_operator',energy_operator_source=args.energy,goal_complete=False))
    receipt(stage,dict(status='prepared_diagnostic_current_center_locality_direction',
        new_actor_decisions=0,new_physics_ticks=0,goal_complete=False))


if __name__=='__main__':main()
