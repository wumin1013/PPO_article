"""Matched learned transfer of parent and improved action functions to continuous sensing."""
import time
import numpy as np
import torch
from continuous_sensor_runtime import environment_factory
from kernel_local_feedback_policy import KernelLocalFeedbackPolicy
from diagnose_local_basis_fit import feature_batches
from learn_single_decision_credit import SCREEN,review,adoption_checks
from audit_xy_curvature_resolution import CURVATURE_NAMES,curvature_metrics
from reflection_policy import reflect_observation
from train_pilot import evaluate,save_torch
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


@torch.no_grad()
def means(policy,observations):
    return np.concatenate([policy.distribution(x).mean.numpy() for x in torch.tensor(observations).split(128)])


def main():
    torch.set_num_threads(1)
    parent=BASE/'recentered_reverse_shape_credit_learning_20260917'
    selected=read(parent/'training_selection.json');parent_model=parent/selected['model']
    source=BASE/'continuous_sensor_preparation_20260917'
    updated=BASE/'full_feedback_geometry_fixed_learning_20260917/candidate_1.pt'
    ids=SCREEN+[38,39,42,43,44,45,46,47]
    stage=register('continuous_sensor_matched_function_transfer_20260917',
        [__file__,BASE/'continuous_sensor_runtime.py',BASE/'kernel_local_feedback_policy.py',
         BASE/'diagnose_local_basis_fit.py',BASE/'learn_single_decision_credit.py',
         parent_model,updated,parent/'training_selection.json',source/'training_pairs.npz',
         source/'receipt.json',BASE/'large_step_single_sensor_causal_probe_20260917/receipt.json'],
        dict(training_states=4167,arms=['parent','updated'],ridge=.001,
            learned_parameters=768,linear_fit_per_arm=1,latent_loss_scales=[.005,.1,.1],
            screen_paths=ids,evaluation_actor_calls=18000,evaluation_physics_ticks=420000,
            wall_s=240,development_evaluations=0),
        'Learn a shared feedback map under the mechanism-corrected continuous2mm '
        'reference sensing. Two matched arms use the same4167 parent training states, '
        'path weights, fixed hidden features and256 kernels, one analytic ridge '
        'fit of768 residual coefficients, and same physical screen. Their sole '
        'difference is the target learned function: retained credit parent versus '
        'its full-step learned update whose regression was causally linked to a '
        'one-decision point-sensor phase shift. Targets are actor outputs on actual '
        'training states, not ideal curves or probe actions. Fit from the target '
        'weights with continuous inputs; compensate representational displacement '
        'using the same shared residual basis. This is supervised policy transfer '
        'of learned functions, not PPO, and not yet new on-policy outcome training. '
        'Check physical completion, signed full XY and curvature, time and cruise; '
        'fit loss or analytic continuity alone never establishes success. One '
        'screen29 per arm; only a candidate passing existing full adoption plus '
        'individual reverse/exit/curvature nonincrease receives remaining35 routes. '
        'No development evaluation or working-model replacement in this stage. '
        'If neither passes, inspect actual errors before changing algorithm; no '
        'automatic ridge/width/checkpoint search.')
    start=time.perf_counter();costs=dict(evaluation_actor_calls=0,evaluation_physics_ticks=0,analytic_fits=0)
    with np.load(source/'training_pairs.npz') as z:
        old=z['old_observation'].copy();new=z['observation'].copy();weights=z['weight'].copy()
    scales=np.array([.005,.1,.1]);training=[np.asarray(p) for p in read(BASE/'learning_preparation_v4/paths.json')['training']]
    def enrich(results,traces):
        v=review(results,traces);curv=[]
        for r,t in zip(results,traces):
            with np.load(t) as z:
                if len(z['reference'])==3 and r['success']:
                    curv.append(dict(path=r['original_path'],**curvature_metrics(z['reference'],z['rows'])))
        v['actual_curvature']=curv;return v
    baseline=enrich([selected['results'][i] for i in ids],[BASE/selected['traces'][i] for i in ids])
    candidates=[]
    for arm,model in (('parent',parent_model),('updated',updated)):
        p=KernelLocalFeedbackPolicy(256)
        p.load_state_dict(torch.load(model,map_location='cpu',weights_only=False)['policy'])
        target=means(p,old);before=means(p,new)
        features=feature_batches(p,new)*p.residual_scale.detach().numpy()[None,:,None]/scales[None,:,None]
        rhs=(target-before)/scales
        delta=[]
        for h in range(3):
            f=features[:,h]
            delta.append(np.linalg.solve(f.T@(weights[:,None]*f)+.001*np.eye(f.shape[1]),f.T@(weights*rhs[:,h])))
        delta=np.asarray(delta)
        predicted=np.einsum('nhf,hf->nh',features,delta)
        with torch.no_grad():p.residual_coefficients.add_(torch.tensor(delta,dtype=torch.float32))
        after=means(p,new);err=(after-target)/scales
        linear_error=float(abs((after-before)/scales-predicted).max())
        assert linear_error<1e-3
        mirrored=means(p,reflect_observation(torch.tensor(new[:256])).numpy())
        assert np.allclose(mirrored,after[:256]*[-1,1,1],atol=1e-6,rtol=0.)
        name=f'{arm}.pt'
        save_torch(stage/name,dict(policy=p.state_dict(),architecture='kernel_local_feedback',centers=256,
            observation_encoding='continuous_centered_reference_mean_tangent_2mm',
            target_model_sha256=sha(model),learning_method='matched_analytic_function_transfer',ridge=.001))
        costs['analytic_fits']+=1
        fit=dict(arm=arm,rms_before=np.sqrt((weights[:,None]*((before-target)/scales)**2).sum(0)).tolist(),
            rms_after=np.sqrt((weights[:,None]*err**2).sum(0)).tolist(),
            maximum_normalized_fit_error=float(abs(err).max()),coefficient_delta_norm=float(np.linalg.norm(delta)),
            linear_superposition_error=linear_error)
        print('CONTINUOUS_TRANSFER_FIT',fit,flush=True)
        rs=evaluate(p,[training[i] for i in ids],stage,arm,None,costs,start+240,environment_factory)
        rs=[dict(r,original_path=i) for r,i in zip(rs,ids)];ts=[stage/arm/f'{i}.npz' for i in range(len(ids))]
        value=enrich(rs,ts);checks=adoption_checks(value,baseline)
        if value['comparable_complete_pool']:
            oldshape={r['path']:r for r in baseline['shapes']}
            for key in ('out_edge_outward_mm','full_physical_reverse_deg'):
                checks['each_'+key+'_nonworse']=all(s[key]<=oldshape[s['path']][key]+1e-6 for s in value['shapes'])
            oldcurv={r['path']:r for r in baseline['actual_curvature']}
            for key in CURVATURE_NAMES:
                checks['each_'+key+'_nonworse']=all(s[key]<=oldcurv[s['path']][key]+1e-6 for s in value['actual_curvature'])
        value.update(arm=arm,model=name,fit=fit,checks=checks,eligible=all(checks.values()),
            results=rs,traces=[str(t.relative_to(BASE)) for t in ts])
        candidates.append(value)
        write_json(stage/'results.json',dict(baseline=baseline,candidates=candidates,costs=costs,goal_complete=False,promoted=False))
        print('CONTINUOUS_TRANSFER_SCREEN',arm,{k:value[k] for k in ('successes','entry_sum_mm','exit_sum_mm','full_reverse_sum_deg','full_time_s','eligible')},flush=True)
    eligible=[r for r in candidates if r['eligible']]
    if eligible:
        chosen=max(eligible,key=lambda r:r['independent_zero_return'])
        p.load_state_dict(torch.load(stage/chosen['model'],map_location='cpu',weights_only=False)['policy'])
        remaining=[i for i in range(64) if i not in ids]
        rs=evaluate(p,[training[i] for i in remaining],stage,'remaining',None,costs,start+240,environment_factory)
        rs=[dict(r,original_path=i) for r,i in zip(rs,remaining)]
        pairs=sorted(zip(chosen['results']+rs,[BASE/t for t in chosen['traces']]+[stage/'remaining'/f'{j}.npz' for j in range(len(remaining))]),key=lambda z:z[0]['original_path'])
        rs,ts=map(list,zip(*pairs));full=enrich(rs,ts)
        write_json(stage/'training_selection.json',dict(model=chosen['model'],training=full,results=rs,traces=[str(t.relative_to(BASE)) for t in ts],promotion_pending=True))
    assert costs['evaluation_actor_calls']<=18000 and costs['evaluation_physics_ticks']<=420000
    receipt(stage,dict(status='completed_matched_continuous_sensor_function_transfer',costs=costs,
        eligible=len(eligible),wall_s=time.perf_counter()-start,goal_complete=False,promoted=False))


if __name__=='__main__':main()
