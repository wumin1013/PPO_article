"""Full-subspace constrained proposal from audited current temporal credit."""
import argparse
import time
import numpy as np
import torch
from scipy.optimize import linprog
from constrained_parameter_solver import minimum_metric_step
from audit_current_constraint_gradients import METRICS
from kernel_local_feedback_policy import KernelLocalFeedbackPolicy
from diagnose_local_basis_fit import feature_batches
from probe_one_decision_credit import AMPLITUDE
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',default='current_direct_constraint_gradient_audit_20260917')
    parser.add_argument('--stage',default='direct_full_subspace_constraint_solve_20260917')
    parser.add_argument('--maximum-reduction',type=float,default=.2)
    parser.add_argument('--per-corner-shape',action='store_true')
    parser.add_argument('--full-feedback',action='store_true')
    parser.add_argument('--geometry-step',type=float,default=0.)
    args=parser.parse_args()
    assert 0<=args.maximum_reduction<=.2
    assert 0<=args.geometry_step<=.1 and (args.full_feedback or args.geometry_step==0)
    torch.set_num_threads(1)
    source=BASE/args.source
    assert (source/'receipt.json').exists(),'Current credit audit must finish first'
    prepared=BASE/('full_feedback_parameter_space_preparation_20260917' if args.full_feedback else 'direct_constraint_parameter_preparation_20260917')
    collection=BASE/'current_dense_constraint_credit_collection_20260917'
    parent=BASE/'recentered_reverse_shape_credit_learning_20260917'
    selected=read(parent/'training_selection.json');model=parent/selected['model']
    stage=register(args.stage,
        [__file__,source/'receipt.json',source/'results.json',source/'gradients.npz',
         prepared/'support.npz',prepared/'receipt.json',collection/'results.json',model,
         BASE/'constrained_parameter_solver.py',BASE/'audit_current_constraint_gradients.py']+
         ([source/'baselines.json'] if args.full_feedback else []),
        dict(new_actor_decisions=0,new_physics_ticks=0,wall_s=240,
             maximum_normalized_saved_action_change=1.,maximum_predicted_common_reduction=args.maximum_reduction,
             each_corner_exit_and_reverse_nonincrease=args.per_corner_shape,
             release_exact_action_freezing=args.full_feedback,maximum_geometry_mix_step=args.geometry_step,
             margin_consumption_fraction=.5),
        ('Paired capacity/representation mechanism: all768 existing feedback coefficients are free, with an optional global geometry blend constrained between0 and the declared maximum. Both arms use identical19-route current credit,15 observed failure exclusions, physical-margin constraints, saved-state output bounds and the same embedded action metric. The geometry-fixed arm sets blend change to0; the other may learn a positive change up to0.1. No claim of exact action preservation on16/31; their complete physical success remains mandatory. ' if args.full_feedback else '')+
        ('Matched shape-allocation change: add each individual corner exit and physical reverse nonincrease constraints. Prior aggregate-only constraints permitted worsening selected corners. Same parent, credit, empirical curvature uncertainty and parameter support; measure feasibility before any new physical candidate. ' if args.per_corner_shape else '')+
        'If the source supplies gradient_variants, satisfy every local model simultaneously; '
        'reported nominal predictions use its first model and do not establish actual safety. '
        'Use all366 free residual parameters and current all-time per-path credit directly, '
        'rather than restricting updates to5/13/17 pooled parameter directions. Preserve '
        'paths16/31 through the original action nullspace. Require each entry nonworse, '
        'nonincreased max exit, common fractional reduction of summed entry/exit/reverse, '
        'each corner energy and peak curvature nonworse, nonincreased total time on the '
        '17 credit routes, and at most half the measured continuous corridor margin '
        'consumed on those17 routes. Other routes have only the saved-state action trust '
        'bound here: no unseen-state or all-route feasibility certificate is claimed. '
        'Include every failed single-axis exclusion. Bound normalized output changes '
        'on all4167 current64-route states and832 current queries. Maximize predicted '
        'common reduction up to20%, then minimize the fixed ridge action-change metric. '
        'This is a mathematical proposal, not training success; no physical evaluation '
        'or checkpoint promotion occurs here. Require separate small full-policy '
        'composition checks before any candidate acceptance.')
    started=time.perf_counter()
    with np.load(source/'gradients.npz') as z:
        ids=z['path_id'].tolist();gradient=z['gradient'].copy();failed=z['failed_axis_constraints'].copy();qjac=z['query_jacobian'].copy()
        variants=z['gradient_variants'].copy() if 'gradient_variants' in z else gradient[None]
        metric_names=z['metric_names'].tolist()
        error_key='constraint_uncertainty' if 'constraint_uncertainty' in z else 'smoothness_uncertainty'
        uncertainty=z[error_key].copy() if error_key in z else np.zeros(variants.shape[:3])
        uncertainty_radius=float(z['uncertainty_radius']) if 'uncertainty_radius' in z else None
    smoothness_names=[name for name in metric_names if 'energy' in name or 'peak_curvature' in name]
    with np.load(prepared/'support.npz') as z:
        x=z['observation'].copy();metric=z['extended_metric' if args.full_feedback else 'metric'].copy()
        if args.full_feedback:
            full_features=z['features'].copy();mix_features=z['normalized_mix_jacobian'].copy()
        else:spaces=[z[f'nullspace_{h}'].copy() for h in range(3)]
    p=KernelLocalFeedbackPolicy(256);p.load_state_dict(torch.load(model,map_location='cpu',weights_only=False)['policy'])
    if args.full_feedback:
        dimension=769;cuts=np.arange(4)*256
    else:
        f=feature_batches(p,x);cuts=np.r_[0,np.cumsum([z.shape[1] for z in spaces])];dimension=int(cuts[-1])
    assert gradient.shape[-1]==dimension
    support=np.zeros((len(x),3,dimension))
    if args.full_feedback:
        for h in range(3):support[:,h,cuts[h]:cuts[h+1]]=full_features[:,h]
        support[:,:,-1]=mix_features
    else:
        for h,z in enumerate(spaces):support[:,h,cuts[h]:cuts[h+1]]=f[:,h]@z
    support=np.concatenate([support.reshape(-1,dimension),(qjac/AMPLITUDE[None,:,None]).reshape(-1,dimension)])
    raw=read(source/'baselines.json') if args.full_feedback else read(collection/'results.json')
    base={r['path']:r for r in raw['baselines']}
    corner=[i for i,path in enumerate(ids) if base[path]['shape'] is not None]
    a=[];b=[];labels=[]
    def add(row,bound,label,eta=0.):
        a.append(np.r_[row,eta]);b.append(float(bound));labels.append(label)
    max_exit=max(base[ids[i]]['shape']['out_edge_outward_mm'] for i in corner)
    for variant_index,variant in enumerate(variants):
        by_metric={name:variant[:,i] for i,name in enumerate(metric_names)}
        for i in corner:
            path=ids[i];old=base[path]['shape']
            add(by_metric['in_edge_outward_mm'][i],0.,f'm{variant_index}_entry_{path}')
            add(by_metric['out_edge_outward_mm'][i],max_exit-old['out_edge_outward_mm'],f'm{variant_index}_max_exit_{path}')
            if args.per_corner_shape:
                for name in ('out_edge_outward_mm','full_physical_reverse_deg'):
                    slack=uncertainty[variant_index,i,metric_names.index(name)]*(uncertainty_radius or 0.)
                    add(by_metric[name][i],-slack,f'm{variant_index}_each_{name}_{path}')
            for name in smoothness_names:
                slack=uncertainty[variant_index,i,metric_names.index(name)]*(uncertainty_radius or 0.)
                add(by_metric[name][i],-slack,f'm{variant_index}_{name}_{path}')
        for name in ('in_edge_outward_mm','out_edge_outward_mm','full_physical_reverse_deg'):
            add(by_metric[name][corner].sum(0),0.,f'm{variant_index}_total_{name}',sum(base[ids[i]]['shape'][name] for i in corner))
        add(by_metric['complete_time_s'].sum(0),0.,f'm{variant_index}_time_on_credit_routes')
        for i,path in enumerate(ids):
            margin=base[path]['segment_distance']['margin_lower_mm'];assert margin>0
            add(by_metric['segment_max_distance_mm'][i],.5*margin,f'm{variant_index}_continuous_margin_{path}')
    for i,row in enumerate(failed):add(row,0.,f'failed_axis_{i}')
    if args.full_feedback:
        row=np.zeros(dimension);row[-1]=1.
        add(row,args.geometry_step,'geometry_mix_upper');add(-row,0.,'geometry_mix_lower')
    for i,row in enumerate(support):
        add(row,1.,f'action_upper_{i}');add(-row,1.,f'action_lower_{i}')
    a=np.asarray(a);b=np.asarray(b)
    scale=np.maximum(np.linalg.norm(a,axis=1),1e-12)
    lp=linprog(np.r_[np.zeros(dimension),-1.],A_ub=a/scale[:,None],b_ub=b/scale,
        bounds=[(None,None)]*dimension+[(0.,args.maximum_reduction)],method='highs',
        options=dict(time_limit=90.,primal_feasibility_tolerance=1e-8,dual_feasibility_tolerance=1e-8))
    if not lp.success:
        write_json(stage/'optimizer.json',dict(lp_success=False,message=lp.message))
        receipt(stage,dict(status='direct_linear_solve_not_successful',goal_complete=False,new_physics_ticks=0));return
    eta=float(lp.x[-1]);bound=b-a[:,-1]*eta;matrix=a[:,:-1]
    direction,projection=minimum_metric_step(metric,matrix,bound,lp.x[:-1])
    parameters=direction[:768].reshape(3,256) if args.full_feedback else np.stack([z@direction[cuts[h]:cuts[h+1]] for h,z in enumerate(spaces)])
    geometry_delta=float(direction[-1]) if args.full_feedback else 0.
    predicted=np.einsum('pmd,d->pm',gradient,direction)
    np.savez_compressed(stage/'direction.npz',reduced_direction=direction,direction=parameters,
        gradient=gradient,path_id=ids,metric_names=np.asarray(metric_names),predicted_changes=predicted,
        geometry_mix_delta=geometry_delta,
        variant_predicted_changes=np.einsum('kpmd,d->kpm',variants,direction))
    duals=[dict(constraint=label,marginal=float(value)) for label,value in zip(labels,lp.ineqlin.marginals/scale) if abs(value)>1e-10]
    result=dict(free_parameters=dimension,simultaneous_gradient_models=len(variants),predicted_joint_fraction=eta,projection=projection,
        actual_free_parameters=dimension-int(args.full_feedback and args.geometry_step==0),
        geometry_mix_delta=geometry_delta,full_feedback_space=args.full_feedback,
        each_corner_exit_and_reverse_nonincrease=args.per_corner_shape,
        smoothness_constraint_metrics=smoothness_names,
        empirical_margin_radius=uncertainty_radius,
        empirical_margin_radius_pass=uncertainty_radius is None or projection['metric_norm']<=uncertainty_radius+1e-8,
        maximum_normalized_action_change=float(abs(support@direction).max()),
        parameter_max=float(abs(parameters).max()),failed_axis_constraints=len(failed),
        largest_duals=sorted(duals,key=lambda r:abs(r['marginal']),reverse=True)[:12],
        parent_sha256=sha(model),requires_small_full_policy_composition_check=True,goal_complete=False)
    write_json(stage/'optimizer.json',result)
    assert time.perf_counter()-started<240
    receipt(stage,dict(status='completed_direct_full_parameter_proposal',new_actor_decisions=0,
        new_physics_ticks=0,predicted_joint_fraction=eta,wall_s=time.perf_counter()-started))
    print('DIRECT_PROPOSAL',result,flush=True)


if __name__=='__main__':main()
