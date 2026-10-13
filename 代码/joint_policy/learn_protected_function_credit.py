"""Coordinate actual complete-function credit under per-path shape constraints."""
import argparse
import time
from datetime import datetime,timezone
import numpy as np
import torch
from scipy.optimize import linprog,minimize,LinearConstraint,Bounds
from local_geometry_feedback_policy import LocalGeometryFeedbackPolicy
from kernel_local_feedback_policy import KernelLocalFeedbackPolicy
from diagnose_local_basis_fit import feature_batches
from learn_exit_local_credit import last_layer_features
from learn_single_decision_credit import SCREEN,review,adoption_checks
from probe_one_decision_credit import AMPLITUDE
from train_pilot import evaluate,save_torch
from joint_runtime_v5 import environment_factory
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--residual',action='store_true');parser.add_argument('--recenter',action='store_true')
    parser.add_argument('--parent');parser.add_argument('--source');parser.add_argument('--stage')
    parser.add_argument('--margin-source');parser.add_argument('--time-priority',action='store_true')
    parser.add_argument('--preserve-smoothness',action='store_true');parser.add_argument('--preparation');args=parser.parse_args()
    if args.time_priority:assert args.margin_source, 'Time priority retains measured continuous-margin constraints'
    if args.parent:
        assert args.source and args.stage, 'Custom learning requires explicit parent, credit source and new stage'
        args.recenter=True
    if args.recenter:args.residual=True
    torch.set_num_threads(1)
    source=BASE/('recentered_residual_function_credit_probe_20260917' if args.recenter else 'protected_residual_function_credit_probe_20260917' if args.residual else 'protected_closed_loop_parameter_credit_probe_20260917')
    if args.source:source=BASE/args.source
    data=read(source/'results.json');baseline=data['baseline'];records=data['candidates']
    screen=[r['original_path'] for r in records[0]['results']]
    parent=BASE/('protected_residual_complete_function_credit_learning_20260917' if args.recenter else 'protected_complete_function_credit_learning_20260917' if args.residual else 'matched_dense_corner_credit_learning_20260917')
    if args.parent:parent=BASE/args.parent
    selection=read(parent/'training_selection.json')
    preparation=BASE/'protected_residual_capacity_initialization_20260917' if args.residual else source
    if args.preparation:preparation=BASE/args.preparation
    with np.load(preparation/'basis.npz') as z:dimensions=len(z['basis'])
    assert len(records)==2*dimensions and all(r['successes']==len(screen) for r in records)
    model=preparation/'initial.pt' if args.residual and not args.recenter else parent/selection['model']
    support_file=preparation/'basis.npz' if args.residual else BASE/'matched_entry_interaction_feedback_20260916/window_training_data.npz'
    stage=register(args.stage or ('recentered_reverse_shape_credit_learning_20260917' if args.recenter else 'protected_residual_complete_function_credit_learning_20260917' if args.residual else 'protected_complete_function_credit_learning_20260917'),
        [__file__,source/'receipt.json',source/'results.json',preparation/'basis.npz',model,
         parent/'training_selection.json',support_file,BASE/'learn_single_decision_credit.py',
         BASE/'learn_exit_local_credit.py',BASE/'joint_runtime_v5.py',BASE/'kernel_local_feedback_policy.py']+
         ([BASE/args.margin_source/'margin_derivatives.npz',BASE/args.margin_source/'receipt.json'] if args.margin_source else []),
        dict(optimizer_variables=dimensions,maximum_predicted_joint_shape_reduction=.2,
             maximum_normalized_saved_action_change=1.,coefficient_bounds=[-1.,1.],
             parameter_steps=[.5,1.,1.5],evaluation_actor_calls=16000,
             evaluation_physics_ticks=360000,wall_s=300,new_single_action_branches=0,screen_paths=screen,
             maximum_predicted_nominal_margin_consumption=.5 if args.margin_source else None,
             time_priority=args.time_priority,predicted_shape_optimum_fraction=.9 if args.time_priority else 1.,
             preserve_per_corner_energy_and_peak=args.preserve_smoothness),
        ('Preserve per-corner curvature energy AND peak curvature in local prediction and actual promotion, after rawXY inspection showed some entry/exit/reverse gains concentrate the turn and lower bend speed. These are two existing shape metrics, not added reward terms. Compare the feasibility of the same measured five-direction basis before new simulation; if the predicted common improvement cannot reach10% at1.5step, end the solve and expand the proposal hypothesis rather than relax the actual shape criteria. ' if args.preserve_smoothness else '')+
        ('Matched optimization-objective control using exactly the same parent and measured consequences as the margin-constrained shape update. Retain90% of the maximum predicted common shape reduction, then minimize predicted FULL-route time, then choose the smallest-action solution retaining95% of that predicted time gain. Actual promotion remains>=10% reduction in all three shape sums and all unchanged completion/entry/max-exit checks. The90% number changes only the local predicted Pareto tradeoff, not actual physical or promotion criteria. Choose fastest eligible training screen candidate before remaining training/development. If predicted time gain is<1ms stop without rollout. This compares an explicitly bundled objective/selection priority, not extra training or new reward terms. ' if args.time_priority else '')+
        ('Single-factor correction of the same failed coefficient solve: additionally bound predicted continuous command-segment distance increase to half each route\'s measured current corridor margin. Same parent, complete-function probes, proposal basis, three steps and shape objective. This conservative LOCAL update trust bound does not change the1.25mm execution criterion and is not a safety certificate; nonlinear interactions still require actual full evaluation. Do not simulate if the optimum cannot predict10% shape reduction even at the largest1.5step. ' if args.margin_source else '')+
        (f'Custom credit source {source.name}, parent {parent.name}. The actual screen pool has {len(screen)} training routes and {dimensions} parameter directions; reuse baseline traces. Missing full-training routes number {64-len(screen)}. These explicit counts supersede the historical21/43/five counts below. ' if args.parent else '')+
        ('Recenter learning at the newly improved residual candidate, using newly measured whole-policy responses. Require the common predicted fractional reduction for THREE quantities: entry excursion,exit excursion,and full physical reverse turning. Actual promotion additionally requires>=10% reduction in full reversal, because reduced outside distance alone has left visible heading corrections. This tightens a previously nonworse-only reversal constraint; no added reward term or relaxed physical bound. Recentered credit and priority constraint both differ from the preceding round, so do not claim a one-factor superiority result. Keep the fixed4174-state proposal metric and two-path protection; it is a local trust metric, not a current-state safety proof. ' if args.recenter else
        'Apply the SAME constrained coefficient solver to the expanded768-coefficient residual, preserving paths16+31. Current4174 training states replace the preceding4177-state metric because the parent trajectory changed. Base policy stays frozen and only residual coefficients learn. ' if args.residual else '')+
        'Learn a coordinated update from the ten complete protected-function probes. '
        'Each finite derivative is measured by an entire closed-loop learned policy, '
        'including temporal composition and completion-time steps. Fit no state/action '
        'teacher and deploy no probe schedule. In five shared-network basis coefficients, '
        'first solve an LP for up to20% predicted reduction in BOTH entry and exit sums, '
        'with every entry nonworse, no higher maximum exit/full reversal, no slower total '
        'time, improved complete objective and saved-state normalized action bound1. '
        'Then minimize state-weighted action change with those same linear constraints. '
        'All constraints here are local empirical predictions, never acceptance certificates. '
        'Evaluate three predeclared scales(.5,1,1.5) of this SINGLE learned direction; '
        'the1.5 scale explicitly explores beyond the fitted support trust bound. Every '
        'candidate retains the trajectory-preserving parameter nullspace. Apply unchanged '
        'actual adoption_checks on21 training paths. At most one selected candidate '
        'receives missing43 training paths and, only after full training passes,12 '
        'development paths. Preserve prior strong models; no sealed tests or final '
        'shape/cruise claim. If no direction or candidate qualifies, stop this solve and '
        'inspect its physical constraint conflict; never extend it unchanged.')
    started=time.perf_counter();costs=dict(evaluation_actor_calls=0,evaluation_physics_ticks=0)
    saved=torch.load(model,map_location='cpu',weights_only=False)
    p=KernelLocalFeedbackPolicy(centers=256) if args.residual else LocalGeometryFeedbackPolicy(mask_endpoint=True)
    p.load_state_dict(saved['policy'])
    with np.load(preparation/'basis.npz') as z:basis=z['basis'].copy();protected=z['protected_features'].copy()
    with np.load(support_file) as z:
        obs=z['observation'].copy() if args.residual else z['observation'][:4177].copy()
        w=z['weight'].copy() if args.residual else z['weight'][:4177].copy()*3
    def features(observations):
        return feature_batches(p,observations)*AMPLITUDE[None,:,None] if args.residual else last_layer_features(p,observations)
    if args.residual:protected=protected*AMPLITUDE[None,:,None]
    f=features(obs);action=np.einsum('nhf,khf->nhk',f,basis)/AMPLITUDE[None,:,None]
    extra=[]
    for name in ('dense_corner_complete_credit_collection_20260917','single_decision_complete_credit_probe_20260917'):
        with np.load(BASE/name/'queries.npz') as z:extra.append(z['observation'].copy())
    ef=features(np.concatenate(extra));eaction=np.einsum('nhf,khf->nhk',ef,basis)/AMPLITUDE[None,:,None]
    support=np.concatenate([action.reshape(-1,dimensions),eaction.reshape(-1,dimensions)])
    derivatives={m:np.zeros((16,dimensions)) for m in ('in_edge_outward_mm','out_edge_outward_mm','full_physical_reverse_deg','full_normalized_energy','full_spatial_peak_curvature_mm_inv')}
    time_gradient=np.zeros(dimensions);return_gradient=np.zeros(dimensions)
    for i in range(dimensions):
        lo,hi=[next(r for r in records if r['basis']==i and r['sign']==s) for s in (-1,1)]
        time_gradient[i]=hi['full_time_s']-lo['full_time_s']
        return_gradient[i]=hi['independent_zero_return']-lo['independent_zero_return']
        for metric in derivatives:
            derivatives[metric][:,i]=np.array([s[metric] for s in hi['shapes']])-np.array([s[metric] for s in lo['shapes']])
    din=derivatives['in_edge_outward_mm'];dout=derivatives['out_edge_outward_mm'];drev=derivatives['full_physical_reverse_deg'].sum(0)
    oldout=np.array([s['out_edge_outward_mm'] for s in baseline['shapes']])
    # The sixth variable is the common fractional improvement; zero is feasible.
    a=[];b=[];labels=[]
    def add(row,bound,label,eta=0.):
        a.append(np.r_[row,eta]);b.append(bound);labels.append(label)
    for i in range(16):
        add(din[i],0.,f'entry_{i}')
        add(dout[i],baseline['max_exit_mm']-oldout[i],f'max_exit_{i}')
    add(din.sum(0),0.,'entry_total',baseline['entry_sum_mm'])
    add(dout.sum(0),0.,'exit_total',baseline['exit_sum_mm'])
    add(drev,0.,'reverse_total',baseline['full_reverse_sum_deg'] if args.recenter else 0.)
    add(time_gradient,0.,'complete_time');add(-return_gradient,0.,'complete_return')
    if args.preserve_smoothness:
        for metric_name in ('full_normalized_energy','full_spatial_peak_curvature_mm_inv'):
            for i in range(16):add(derivatives[metric_name][i],0.,f'{metric_name}_{i}')
    if args.margin_source:
        with np.load(BASE/args.margin_source/'margin_derivatives.npz') as z:
            assert z['original_path'].tolist()==screen
            nominal=z['nominal'];margin_derivative=z['derivatives']
        assert np.all(nominal<1.25)
        for i,path in enumerate(screen):
            add(margin_derivative[i],.5*(1.25-nominal[i]),f'continuous_segment_margin_{path}')
    for row in support:add(row,1.,'positive_action_bound');add(-row,1.,'negative_action_bound')
    a=np.asarray(a);b=np.asarray(b)
    lp=linprog(np.r_[np.zeros(dimensions),-1.],A_ub=a,b_ub=b,bounds=[(-1.,1.)]*dimensions+[(0.,.2)],method='highs')
    assert lp.success,lp.message
    eta=float(lp.x[-1]);c=lp.x[:dimensions];aa=a[:,:dimensions]
    time_optimum=None
    if args.time_priority:eta*=.9
    bb=b-a[:,-1]*eta
    if args.time_priority:
        fastest=linprog(time_gradient,A_ub=aa,b_ub=bb,bounds=[(-1.,1.)]*dimensions,method='highs')
        assert fastest.success,fastest.message
        time_optimum=float(fastest.fun)
        if time_optimum>=-.001:
            receipt(stage,dict(status='no_useful_predicted_time_gain',predicted_time_change=time_optimum,costs=costs,goal_complete=False));return
        c=fastest.x;aa=np.vstack([aa,time_gradient]);bb=np.r_[bb,.95*time_optimum]
    metric=np.einsum('n,nhk,nhl->kl',w,action,action)/3+1e-8*np.eye(dimensions)
    fitted=minimize(lambda x:.5*x@metric@x,c,jac=lambda x:metric@x,
        constraints=[LinearConstraint(aa,-np.inf,bb)],bounds=Bounds(-np.ones(dimensions),np.ones(dimensions)),
        method='SLSQP',options=dict(ftol=1e-12,maxiter=1000))
    feasible=float(np.max(aa@fitted.x-bb))<1e-7
    assert fitted.success and feasible,(fitted.message,float(np.max(aa@fitted.x-bb)))
    c=fitted.x;direction=np.einsum('k,khf->hf',c,basis)
    protected_error=float(abs(np.einsum('nhf,hf->nh',protected,direction)).max());assert protected_error<1e-9
    audit=dict(coefficients=c.tolist(),predicted_joint_fraction=eta,
        predicted_entry_sum=baseline['entry_sum_mm']+float(din.sum(0)@c),
        predicted_exit_sum=baseline['exit_sum_mm']+float(dout.sum(0)@c),
        predicted_reverse_sum=baseline['full_reverse_sum_deg']+float(drev@c),
        predicted_time=baseline['full_time_s']+float(time_gradient@c),
        maximum_normalized_saved_change=float(abs(support@c).max()),constraint_error=float(np.max(aa@c-bb)),
        protected_algebraic_error=protected_error,lp_message=lp.message,qp_message=str(fitted.message),
        time_priority=args.time_priority,predicted_optimal_time_change=time_optimum)
    audit['largest_shape_optimum_duals']=sorted(
        [dict(constraint=label,marginal=float(value)) for label,value in zip(labels,lp.ineqlin.marginals) if abs(value)>1e-10],
        key=lambda row:abs(row['marginal']),reverse=True)[:12]
    write_json(stage/'optimizer.json',audit);np.savez_compressed(stage/'direction.npz',direction=direction,
        coefficients=c,basis=basis,entry_derivative=din,exit_derivative=dout,time_derivative=time_gradient)
    print('PROTECTED_OPTIMUM',audit,flush=True)
    if eta<(1/15 if args.margin_source or args.preserve_smoothness else 1e-4):
        receipt(stage,dict(status='no_useful_predicted_feasible_direction',costs=costs,goal_complete=False));return
    paths=read(BASE/'learning_preparation_v4/paths.json');training=[np.asarray(r) for r in paths['training']]
    def run(indices,tag,geometry=training):
        rs=evaluate(p,[geometry[i] for i in indices],stage,tag,None,costs,started+300,environment_factory)
        assert costs['evaluation_actor_calls']<=16000 and costs['evaluation_physics_ticks']<=360000
        return [dict(r,original_path=i) for r,i in zip(rs,indices)],[stage/tag/f'{i}.npz' for i in range(len(indices))]
    candidates=[];cache={}
    def smoothness_checks(value,previous):
        if not args.preserve_smoothness:return {}
        old={s['path']:s for s in previous['shapes']}
        return {f'each_{m}_nonworse':bool(value['comparable_complete_pool'] and all(
            s[m]<=old[s['path']][m]+1e-6 for s in value['shapes']))
            for m in ('full_normalized_energy','full_spatial_peak_curvature_mm_inv')}
    status_path=BASE/'当前执行状态.json';status=read(status_path)
    status['current'].update(state='protected_function_credit_learning_running',training_running=True,physical_diagnostic_running=True,diagnostic_stage=stage.name)
    status['running']=stage.name;status['active_experiment']=stage.name;write_json(status_path,status)
    for step in (.5,1.,1.5):
        p.load_state_dict(saved['policy']);delta=step*direction
        with torch.no_grad():
            if args.residual:p.residual_coefficients.add_(torch.tensor(delta,dtype=torch.float32))
            else:
                p.actor[-1].weight.add_(torch.tensor(delta[:,:-1],dtype=torch.float32));p.actor[-1].bias.add_(torch.tensor(delta[:,-1],dtype=torch.float32))
        name=f'candidate_{step:g}.pt';save_torch(stage/name,dict(policy=p.state_dict(),architecture='kernel_local_feedback' if args.residual else 'local_geometry_feedback',centers=256 if args.residual else None,
            mask_endpoint=True,observation_dim=72,parent_sha256=sha(model),step=step,
            learning_method='protected_complete_function_constrained_credit',coefficients=(c*step).tolist()))
        rs,traces=run(screen,f'screen_{step:g}');cache[name]=(rs,traces)
        r=dict(review(rs,traces),step=step,model=name);r['checks']=adoption_checks(r,baseline)
        r['checks'].update(smoothness_checks(r,baseline))
        if args.recenter:r['checks']['reverse_sum_reduced_ten_percent']=bool(r['comparable_complete_pool'] and r['full_reverse_sum_deg']<=.9*baseline['full_reverse_sum_deg'])
        r['eligible']=all(r['checks'].values())
        candidates.append(r);write_json(stage/'screen_results.json',dict(baseline=baseline,candidates=candidates,costs=costs))
        print('PROTECTED_LEARNING',step,{k:r[k] for k in ('successes','full_time_s','entry_sum_mm','exit_sum_mm','full_reverse_sum_deg','eligible')},r['checks'],flush=True)
    chosen=None;full=None;dev=None;eligible=[r for r in candidates if r['eligible']]
    if eligible:
        chosen=min(eligible,key=lambda r:(r['full_time_s'],-r['independent_zero_return'])) if args.time_priority else max(eligible,key=lambda r:r['independent_zero_return'])
        write_json(stage/'screen_selection.json',dict(selected=chosen,training_only=True))
        p.load_state_dict(torch.load(stage/chosen['model'],map_location='cpu',weights_only=False)['policy'])
        rs,traces=run([i for i in range(64) if i not in screen],'remaining')
        old_rs,old_traces=cache[chosen['model']];pairs=sorted(zip(old_rs+rs,old_traces+traces),key=lambda v:v[0]['original_path'])
        rs,traces=map(list,zip(*pairs));full=review(rs,traces)
        full['checks']=adoption_checks(full,selection['training'])
        full['checks'].update(smoothness_checks(full,selection['training']))
        if args.recenter:full['checks']['reverse_sum_reduced_ten_percent']=bool(full['comparable_complete_pool'] and full['full_reverse_sum_deg']<=.9*selection['training']['full_reverse_sum_deg'])
        write_json(stage/'training_selection.json',dict(model=chosen['model'],training=full,results=rs,traces=[str(t.relative_to(BASE)) for t in traces]))
        if all(full['checks'].values()):
            rs,traces=run(list(range(12)),'development',[np.asarray(r) for r in paths['validation']]);dev=review(rs,traces)
    write_json(stage/'results.json',dict(baseline=baseline,candidates=candidates,chosen=chosen,full_training=full,development=dev,costs=costs,goal_complete=False))
    receipt(stage,dict(status='completed_protected_complete_function_credit_learning',costs=costs,
        screen_eligible=len(eligible),full_training_eligible=bool(full and all(full['checks'].values())),wall_s=time.perf_counter()-started))
    status=read(status_path);status['current'].update(state='protected_function_credit_learning_complete',training_running=False,physical_diagnostic_running=False)
    status['completed'].append(dict(stage=stage.name,receipt_sha256=sha(stage/'receipt.json')))
    status['running']=None;status['active_experiment']=None;status['updated_utc']=datetime.now(timezone.utc).isoformat();write_json(status_path,status)


if __name__=='__main__':main()
