"""Combine matched amplitude corrections and measured directional secants."""
import time
import numpy as np
from search_complete_return import read
from study_io import BASE, register, receipt, write_json


def main():
    coarse=BASE/'current_direct_constraint_gradient_audit_20260917'
    smaller=BASE/'small_amplitude_credit_composition_audit_20260917'
    proposal=BASE/'direct_full_subspace_constraint_solve_20260917'
    check=BASE/'direct_constraint_composition_probe_20260917'
    preparation=BASE/'direct_constraint_parameter_preparation_20260917'
    stage=register('amplitude_secant_constraint_credit_20260917',
        [__file__,coarse/'gradients.npz',smaller/'gradients.npz',smaller/'receipt.json',
         smaller/'results.json',proposal/'direction.npz',check/'comparison.json',
         check/'results.json',check/'receipt.json',preparation/'support.npz'],
        dict(new_actor_decisions=0,new_physics_ticks=0,wall_s=30),
        'Correct the diagnosed local credit errors before another proposal. Replace '
        'paths3/7/13 with their matched all-time smaller-amplitude gradients. On the '
        'eight already measured routes also construct forward and backward whole-policy '
        'secant-consistent gradients by minimum dual-metric rank-one correction. Keep '
        'all three local models as simultaneous constraints, rather than discarding '
        'the uncalibrated model. The correction is identified in ONLY the measured '
        'parameter direction; other directions still require complete-policy tests. '
        'Amplitude replacement and secant calibration are bundled here; no one-factor '
        'superiority claim. One new bounded proposal/check is authorized by this '
        'mechanism evidence, not repeated unchanged training. Preserve the working actor.')
    started=time.perf_counter()
    with np.load(coarse/'gradients.npz') as z:
        saved={k:z[k].copy() for k in ('path_id','metric_names','gradient','failed_axis_constraints','query_jacobian')}
    ids=saved['path_id'].tolist();g=saved['gradient']
    with np.load(smaller/'gradients.npz') as z:
        for path,value in zip(z['path_id'].tolist(),z['gradient']):g[ids.index(path)]=value
        extra_failed=z['failed_axis_constraints'].copy()
    saved['failed_axis_constraints']=np.concatenate([saved['failed_axis_constraints'],extra_failed])
    with np.load(proposal/'direction.npz') as z:d=z['reduced_direction'].copy()
    with np.load(preparation/'support.npz') as z:metric=z['metric'].copy()
    normal=metric@d/(d@metric@d)
    forward=g.copy();backward=g.copy();errors=[]
    measured=read(check/'results.json')
    low,high=measured['candidates'];alpha=high['coefficient'];assert low['coefficient']==-alpha
    baselines={r['path']:np.asarray(r['values']) for r in measured['baseline_metrics']}
    for lo,hi in zip(low['metrics'],high['metrics']):
        assert lo['path']==hi['path'] and lo['success'] and hi['success']
        path=lo['path'];i=ids.index(path);base=baselines[path]
        slopes=[(np.asarray(hi['values'])-base)/alpha,(base-np.asarray(lo['values']))/alpha]
        for target,slope in zip((forward,backward),slopes):
            target[i]+=(slope-g[i]@d)[:,None]*normal
            errors.append(float(abs(target[i]@d-slope).max()))
    assert max(errors)<1e-10
    saved['gradient_variants']=np.stack([g,forward,backward])
    np.savez_compressed(stage/'gradients.npz',**saved)
    write_json(stage/'results.json',dict(replaced_amplitude_paths=[3,7,13],
        secant_paths=list(baselines),model_names=['local_finite','forward_secant','backward_secant'],
        largest_secant_identity_error=max(errors),calibrated_directions=1,goal_complete=False))
    receipt(stage,dict(status='completed_amplitude_secant_credit_models',new_actor_decisions=0,
        new_physics_ticks=0,wall_s=time.perf_counter()-started))
    print('CALIBRATED_CREDIT',saved['gradient_variants'].shape,max(errors),flush=True)


if __name__=='__main__':main()
