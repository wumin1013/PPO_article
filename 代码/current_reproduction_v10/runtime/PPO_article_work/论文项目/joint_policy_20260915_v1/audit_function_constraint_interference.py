"""Saved corridor-margin prediction and head-factorial tests of a failed update."""
import itertools
import time
import numpy as np
import torch
from kernel_local_feedback_policy import KernelLocalFeedbackPolicy
from train_pilot import evaluate
from joint_runtime_v5 import environment_factory
from search_complete_return import read
from study_io import BASE, register, receipt, write_json, sha


def distances_to_segments(reference, xy):
    starts=reference[:-1]; vectors=np.diff(reference, axis=0)
    offsets=xy[:, None, :]-starts
    t=np.clip(np.sum(offsets*vectors, axis=2)/np.sum(vectors*vectors, axis=1), 0, 1)
    return np.linalg.norm(offsets-t[..., None]*vectors, axis=2)


def distances(reference, xy):
    return distances_to_segments(reference,xy).min(axis=1)


def margin(trace):
    with np.load(trace) as z:
        d=distances(z['reference'], z['rows'][:, :2])
    j=int(d.argmax())
    return dict(max_point_distance_mm=float(d[j]), margin_mm=float(1.25-d[j]),
        maximum_time_s=(j+1)*.001)


def segment_margin(trace, tolerance=1e-6):
    """Bound max distance on every command segment using distance's 1-Lipschitz property."""
    with np.load(trace) as z:
        return command_segment_margin(z['reference'],z['rows'],tolerance)


def command_segment_margin(reference,rows,tolerance=1e-6):
    ref=np.asarray(reference);xy=np.vstack([ref[0],np.asarray(rows)[:, :2]])
    d=distances_to_segments(ref,xy);best=float(d.min(axis=1).max())
    a,b=xy[:-1],xy[1:];da,db=d[:-1],d[1:]
    def interval_upper(a,b,da,db):
        lipschitz=(da.min(axis=1)+db.min(axis=1)+np.linalg.norm(b-a,axis=1))*.5
        # Distance to each closed reference segment is convex. Its endpoint
        # maximum bounds that distance throughout the command interval. The
        # minimum over reference segments is therefore another valid upper
        # bound on distance to their union, exact on a coincident straight.
        convex=np.maximum(da,db).min(axis=1)
        return np.minimum(lipschitz,convex)
    upper=interval_upper(a,b,da,db)
    for iteration in range(50):
        active=upper>best+tolerance
        if not active.any():break
        a,b,da,db=a[active],b[active],da[active],db[active]
        middle=(a+b)*.5;dm=distances_to_segments(ref,middle);best=max(best,float(dm.min(axis=1).max()))
        a,b=np.concatenate([a,middle]),np.concatenate([middle,b])
        da,db=np.concatenate([da,dm]),np.concatenate([dm,db])
        upper=interval_upper(a,b,da,db)
    else:raise RuntimeError('Segment-distance bound did not converge')
    # Previously pruned intervals are bounded by the then-best + tolerance.
    # Keep that enclosure too, not only the final active set's upper bounds.
    certified_upper=best+tolerance
    return dict(max_distance_lower_mm=best,max_distance_upper_mm=certified_upper,
        margin_lower_mm=1.25-certified_upper,iterations=iteration,
        tolerance_mm=tolerance)


def continuous_margin_credit(parent_name='recentered_reverse_shape_credit_learning_20260917',
        source_name='long_pool_residual_function_credit_probe_20260917',
        stage_name='continuous_segment_margin_credit_audit_20260917'):
    """Reproduce the completed saved-only margin audit, in a fresh experiment workspace."""
    parent=BASE/parent_name
    source=BASE/source_name
    selected=read(parent/'training_selection.json');data=read(source/'results.json')
    assert (source/'receipt.json').exists()
    stage=register(stage_name,
        [__file__,parent/'training_selection.json',source/'results.json',source/'receipt.json'],
        dict(new_physics_ticks=0,new_actor_calls=0,wall_s=60),
        'Bound continuous command-segment corridor margins from saved traces. Baseline and ten '
        'direction rollouts; no physical rerun. The Lipschitz bound fixes endpoint-only '
        'margin diagnostics; runtime exact segment acceptance is unchanged.')
    ids=[r['original_path'] for r in data['candidates'][0]['results']]
    base=[segment_margin(BASE/selected['traces'][i]) for i in ids]
    values=[dict(basis=r['basis'],sign=r['sign'],margins=[segment_margin(BASE/t) for t in r['traces']]) for r in data['candidates']]
    dimensions=len(data['candidates'])//2
    assert len(data['candidates'])==2*dimensions and all(r['successes']==len(ids) for r in data['candidates'])
    derivative=np.zeros((len(ids),dimensions));quadratic=derivative.copy()
    nominal=np.array([m['max_distance_upper_mm'] for m in base])
    for k in range(dimensions):
        lo,hi=[np.array([m['max_distance_upper_mm'] for m in next(r for r in values if r['basis']==k and r['sign']==s)['margins']]) for s in (-1,1)]
        derivative[:,k]=hi-lo;quadratic[:,k]=(hi+lo-2*nominal)/.5
    np.savez_compressed(stage/'margin_derivatives.npz',original_path=ids,nominal=nominal,
        derivatives=derivative,diagonal_quadratic=quadratic)
    write_json(stage/'results.json',dict(original_path=ids,baseline=base,candidates=values,goal_complete=False))
    receipt(stage,dict(status='completed_continuous_segment_margin_credit',new_actor_calls=0,new_physics_ticks=0))


def main():
    torch.set_num_threads(1)
    parent=BASE/'recentered_reverse_shape_credit_learning_20260917'
    probe=BASE/'long_pool_residual_function_credit_probe_20260917'
    failed=BASE/'long_pool_reverse_shape_credit_learning_20260917'
    selection=read(parent/'training_selection.json');data=read(probe/'results.json')
    coeff=np.asarray(read(failed/'optimizer.json')['coefficients'])
    stage=register('function_constraint_interference_audit_20260917',
        [__file__, parent/'training_selection.json', parent/selection['model'],
         probe/'results.json', probe/'receipt.json', failed/'results.json',
         failed/'candidate_1.pt', failed/'candidate_1.5.pt', failed/'optimizer.json',
         BASE/'kernel_local_feedback_policy.py', BASE/'geometry_spec.py'],
        dict(new_full_rollouts=12, evaluation_actor_calls=3000, evaluation_physics_ticks=75000,
             wall_s=120, failed_cases=[dict(path=40, step=1), dict(path=8, step=1.5)]),
        'Diagnose new failures after ten individually feasible complete-function directions. '
        'First reuse saved actual XY to compare nominal and finite-difference predicted '
        'point-to-polyline margins for all29 routes with the combined policy. Point margins '
        'are a proposal diagnostic; exact segment corridor and full-stop checks remain unchanged. '
        'Then test all six nontrivial head subsets of each failed residual parameter update '
        'on its failing training route. Reuse the original and full-update outcomes. '
        'This is an explicit2x2x2 parameter-head interaction test, not head-independent credit. '
        'No training, development, sealed tests, final acceptance or failed-prefix timing claim. '
        'Stop after this bounded diagnosis and correct the measured constraint mechanism.')
    started=time.perf_counter();costs=dict(evaluation_actor_calls=0,evaluation_physics_ticks=0)
    indices=[r['original_path'] for r in data['candidates'][0]['results']]
    baseline=[margin(BASE/selection['traces'][i]) for i in indices]
    derivatives=np.zeros((len(indices), 5))
    for k in range(5):
        lo,hi=[next(r for r in data['candidates'] if r['basis']==k and r['sign']==s) for s in (-1,1)]
        derivatives[:, k]=np.array([margin(BASE/t)['max_point_distance_mm'] for t in hi['traces']])-np.array([margin(BASE/t)['max_point_distance_mm'] for t in lo['traces']])
    audits=[]
    for step in (.5, 1., 1.5):
        for j,i in enumerate(indices):
            trace=failed/f'screen_{step:g}/{j}.npz';result=read(trace.with_suffix('.json'))
            audits.append(dict(path=i,step=step,success=result['success'],reason=result['reason'],
                nominal=baseline[j],actual=margin(trace),
                predicted_max_mm=baseline[j]['max_point_distance_mm']+float(step*derivatives[j]@coeff)))
    np.savez_compressed(stage/'margin_derivatives.npz',derivatives=derivatives,
        original_path=np.asarray(indices),nominal=np.asarray([r['max_point_distance_mm'] for r in baseline]))
    write_json(stage/'margin_audit.json',dict(records=audits))
    saved=torch.load(parent/selection['model'],map_location='cpu',weights_only=False)['policy']
    paths=read(BASE/'learning_preparation_v4/paths.json')['training'];records=[]
    p=KernelLocalFeedbackPolicy(centers=256)
    for path,step in ((40,1.),(8,1.5)):
        new=torch.load(failed/f'candidate_{step:g}.pt',map_location='cpu',weights_only=False)['policy']
        delta=new['residual_coefficients']-saved['residual_coefficients']
        for bits in itertools.product((0,1),repeat=3):
            if sum(bits) in (0,3):continue
            p.load_state_dict(saved)
            with torch.no_grad():p.residual_coefficients.add_(delta*torch.tensor(bits)[:,None])
            tag=f'path{path}_'+''.join(map(str,bits))
            rs=evaluate(p,[np.asarray(paths[path])],stage,tag,None,costs,started+120,environment_factory)
            assert costs['evaluation_actor_calls']<=3000 and costs['evaluation_physics_ticks']<=75000
            r=dict(path=path,step=step,bits=list(bits),result=rs[0],trace=f'{tag}/0.npz',
                margin=margin(stage/tag/'0.npz'))
            records.append(r);write_json(stage/'results.json',dict(records=records,costs=costs,goal_complete=False))
            print('HEAD_CONSTRAINT',path,bits,rs[0]['success'],rs[0]['reason'],r['margin'],flush=True)
    receipt(stage,dict(status='completed_function_constraint_interference_audit',costs=costs,wall_s=time.perf_counter()-started))
    print('FAILED_PREDICTIONS',[r for r in audits if not r['success']],flush=True)


if __name__=='__main__':main()
