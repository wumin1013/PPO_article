"""Tighten current learned-credit constraints with measured composition errors."""
import argparse
import time
import numpy as np
import torch
from scipy.optimize import linprog
from canonical_gate_state import CanonicalGatePolicy
from diagnose_local_basis_fit import feature_batches
from constrained_parameter_solver import minimum_metric_step
from probe_one_decision_credit import AMPLITUDE
from solve_continuous_sensor_credit import NAMES,constraint_problem
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def load_saved_problem(source,model):
    with np.load(source/'gradients.npz') as z:
        gradient=z['gradient'].copy();ids=z['path_id'].tolist()
        names=z['metric_names'].tolist();exclusions=z['failed_axis_constraints'].copy()
    with np.load(source/'support.npz') as z:
        metric=z['metric'].copy();observations=z['observation'].copy()
    metadata=torch.load(model,map_location='cpu',weights_only=False)
    assert metadata['architecture'] in ('canonical_gate_kernel','gate_band_canonical_kernel')
    policy=CanonicalGatePolicy(256);policy.load_state_dict(metadata['policy'])
    features=feature_batches(policy,observations)*policy.residual_scale.detach().numpy()[None,:,None]/AMPLITUDE[None,:,None]
    support=np.zeros((len(observations),3,768))
    for h in range(3):support[:,h,h*256:(h+1)*256]=features[:,h]
    baselines=read(source/'baselines.json')['baselines']
    return gradient,ids,names,exclusions,metric,support.reshape(-1,768),baselines


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',default='canonical_gate_current_parameter_solve_20260917')
    parser.add_argument('--probe',default='canonical_gate_complete_composition_20260917')
    parser.add_argument('--stage',default='canonical_gate_measured_margin_solve_20260917')
    args=parser.parse_args()
    torch.set_num_threads(1)
    source=BASE/args.source;probe=BASE/args.probe
    old=read(source/'optimizer.json');model=BASE/old['model']
    assert old['parent_sha256']==sha(model)
    data=read(probe/'results.json')
    assert all(r['summary']['successes']==len(r['metrics']) for r in data['candidates'])
    radius=.15
    stage=register(args.stage,[__file__,source/'gradients.npz',source/'support.npz',
        source/'baselines.json',source/'direction.npz',source/'optimizer.json',
        source/'receipt.json',probe/'results.json',probe/'comparison.json',probe/'receipt.json',
        model,BASE/'solve_continuous_sensor_credit.py',BASE/'constrained_parameter_solver.py'],
        dict(new_physics_ticks=0,new_actor_decisions=0,wall_s=180,
             maximum_metric_norm=radius,measured_directions=1,maximum_common_reduction=.2),
        'Reuse the completed current canonical-policy gradients and both complete '
        'whole-policy checks. All five aggregate shape responses agree within1.2% '
        'of the local prediction, but five near-binding curvature constraints regress '
        'by0.001%-0.0044%. Estimate each curvature quantity\'s maximum observed '
        'one-sided underprediction per action-metric norm using BOTH signs. Tighten '
        'each corresponding predicted upper bound by this error times radius0.15. '
        'Do not change the parent, gradient, features, objective, other constraints '
        'or physical limits. The observed margin is not a guaranteed bound for new '
        'directions. Require the solved norm<=0.15 and common gain>=10% before a '
        'separate equal-sign whole-policy check. Failure triggers analysis, not '
        'unbounded margin/step search. This is offline learning, not PPO or a '
        'deployed intervention. No sealed or development data.')
    start=time.perf_counter()
    gradient,ids,names,exclusions,metric,support,baselines=load_saved_problem(source,model)
    assert names==NAMES==data['metric_names']
    with np.load(source/'direction.npz') as z:old_direction=z['direction'].ravel().copy()
    old_norm=float(np.sqrt(old_direction@metric@old_direction))
    assert abs(old_norm-old['projection']['metric_norm'])<1e-10
    original={r['path']:np.asarray(r['values']) for r in data['baseline_metrics']}
    error=np.zeros(gradient.shape[:2]);observed=[]
    smooth=[m for m,name in enumerate(names) if 'energy' in name or 'peak_curvature' in name]
    for candidate in data['candidates']:
        step=candidate['step'];norm=abs(step)*old_norm
        for row in candidate['metrics']:
            if row['path'] not in ids:continue
            i=ids.index(row['path'])
            actual=np.asarray(row['values'])-original[row['path']]
            predicted=gradient[i]@(step*old_direction)
            mismatch=actual-predicted
            error[i,smooth]=np.maximum(error[i,smooth],mismatch[smooth]/norm)
            observed.append(dict(path=row['path'],step=step,norm=norm,
                actual_change=actual.tolist(),predicted_change=predicted.tolist(),
                error=mismatch.tolist()))
    margin=radius*error
    write_json(stage/'model_error.json',dict(metric_names=names,path_ids=ids,
        radius=radius,observed=observed,one_sided_error_per_norm=error.tolist(),
        constraint_margin=margin.tolist(),guaranteed_bound=False))
    a,b,labels=constraint_problem(gradient,baselines,ids,support,exclusions,margin)
    scale=np.maximum(np.linalg.norm(a,axis=1),1e-12)
    lp=linprog(np.r_[np.zeros(768),-1.],A_ub=a/scale[:,None],b_ub=b/scale,
        bounds=[(None,None)]*768+[(0.,.2)],method='highs',options=dict(time_limit=90.))
    result=dict(lp_success=bool(lp.success),message=lp.message,parent_sha256=sha(model),
        model=str(model.relative_to(BASE)),metric_names=names,path_ids=ids,
        eligible_for_small_probe=False,empirical_radius=radius,
        reused_gradient_source=args.source,goal_complete=False)
    if lp.success:
        eta=float(lp.x[-1])
        direction,projection=minimum_metric_step(metric,a[:,:768],b-a[:,-1]*eta,lp.x[:768])
        result.update(predicted_common_reduction=eta,projection=projection,
            maximum_normalized_action_change=float(abs(support@direction).max()),
            eligible_for_small_probe=bool(eta>=.1 and projection['metric_norm']<=radius))
        np.savez_compressed(stage/'direction.npz',direction=direction.reshape(3,256),
            predicted_changes=np.einsum('pmd,d->pm',gradient,direction),path_id=ids,
            metric_names=np.asarray(names))
    assert time.perf_counter()-start<180
    write_json(stage/'optimizer.json',result)
    receipt(stage,dict(status='completed_current_composition_margin_solve',
        eligible_for_small_probe=result['eligible_for_small_probe'],
        new_actor_decisions=0,new_physics_ticks=0,wall_s=time.perf_counter()-start))
    print('CURRENT_MARGIN_SOLVE',result,flush=True)


if __name__=='__main__':main()
