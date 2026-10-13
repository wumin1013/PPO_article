"""Jointly learn earlier observable action responses and their later compensation."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import copy
import time
import multiprocessing as mp
import numpy as np
import torch
from scipy.optimize import linprog
from joint_teacher_state_policy import features
from observable_memory_joint_policy import dynamic_context,future_bend_context
from duration_joint_metrics import guards,penalty
from cruise_duration_credit import event_learning_guards
from learn_intrinsic_joint_recovery import intrinsic_guards
from learn_complete_joint_teachers import teacher_score,parameters_in_bounds
from learn_early_memory_joint_policy import worker_initialize,worker_evaluate,metric_functions
from learn_direct_observable_joint_policy import finish
from study_io import BASE,register,write_json,receipt
from search_complete_return import read


def main():
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_signed_heading_joint_recovery_20260920'
    corrected=BASE/'canonical_gate_band_centered_metric_joint_recovery_20260920'
    parent_file=BASE/'canonical_gate_band_postpeak_memory_joint_learning_20260920/research_candidate.pt'
    md=torch.load(corrected/'state_joint/candidate.pt',map_location='cpu',weights_only=False)
    supports=read(source/'preemptive_cluster_support_audit.json')['records']
    for key,new_key in [('centers','center'),('per_center_mean','mean'),('per_center_std','std'),
                        ('per_center_context_mean','context_mean'),('per_center_context_scale','context_scale')]:
        md[key]=np.concatenate([np.asarray(md[key]),np.asarray([r[new_key] for r in supports])])
    for key,new_key in [('radii','radius'),('per_center_geometry_mean','geometry_mean'),
                        ('per_center_geometry_scale','geometry_scale'),('observed_bend_metric_centers','bend_center')]:
        md[key]=np.r_[md[key],[r[new_key] for r in supports]]
    md['observed_bend_metric_weights']=np.r_[md['observed_bend_metric_weights'],supports[0]['choices'][1]['weight'],0.]
    md['coefficients']=np.zeros((3*7*7,3));md['learning_method']='preemptive_observable_joint_complete_return'
    md.pop('residual_output_precision',None)
    data=read(source/'baseline.json');baseline=data['baseline']['records']
    with np.load(source/'replayed_states.npz') as z:training={k:z[k].copy() for k in z.files}
    f=features(md,training['observations'],training['innovations'])
    variable_kernels=[0,1,5,6]
    phase_paths=[sorted(np.unique(training['path_ids'][np.any(f[:,k]!=0.,axis=1)]).tolist()) for k in range(7)]
    paths=sorted(set(p for k in variable_kernels for p in phase_paths[k]))
    assert paths==[14,104,105,204,205,304,305]
    selected=[r for r in baseline if r['path'] in paths];n=len(paths)
    context=np.zeros((n,7,4));context[:,0]=1.
    activation=[]
    for i,p in enumerate(paths):
        for k_local,k in enumerate(variable_kernels):
            ids=np.flatnonzero((training['path_ids']==p)&np.any(f[:,k]!=0.,axis=1))
            assert len(ids)<=1,('nonlocal_kernel_support',p,k,ids.tolist())
            if len(ids):
                obs=training['observations'][ids[0]]
                v,a=np.tanh((dynamic_context(obs)-md['per_center_context_mean'][k])/md['per_center_context_scale'][k])
                g=np.tanh((future_bend_context(obs)-md['per_center_geometry_mean'][k])/md['per_center_geometry_scale'][k])
                context[i,1:,k_local]=[v,a,v*a,v*v,a*a,g]
                activation.append(dict(path=p,kernel=k,decision_ms=int(training['decisions'][ids[0]])*20,
                    feature_max=float(abs(f[ids[0],k]).max())))
    stage=register('canonical_gate_band_preemptive_joint_credit_learning_20260920',
        [__file__,BASE/'audit_preemptive_joint_support.py',BASE/'observable_memory_joint_policy.py',
         BASE/'joint_teacher_state_policy.py',BASE/'duration_joint_metrics.py',BASE/'cruise_duration_credit.py',
         BASE/'learn_intrinsic_joint_recovery.py',BASE/'intrinsic_curve_metrics.py',
         BASE/'learn_early_memory_joint_policy.py',BASE/'learn_direct_observable_joint_policy.py',
         parent_file,source/'replayed_states.npz',source/'preemptive_cluster_support_audit.json',
         corrected/'state_joint/candidate.pt',corrected/'intrinsic_curve_audit.json',
         BASE/'canonical_gate_band_positive_margin_joint_recovery_20260920/receipt.json'],
        dict(wall_s=480,workers=4,primary_symmetric_vectors=72,maximum_scaled_replacements=24,
             maximum_joint_controls=4,context_counts=[1,3,7],scaled_radii=[1.,4.,16.],
             required_predicted_lateral_reduction_mm=.001,coefficient_bounds=[-1.,1.]),
        'Near-feasible late corrections still do not address the main75-88ms lateral '
        'reversal. Current frozen approach kernel activates at140ms; add observations '
        'at100ms to allow earlier three-action influence. A single100ms kernel gives '
        '104 only0.00165 activation versus0.2-0.5 elsewhere. Two training-derived '
        'observable centers restore balanced support without time/path inputs: one '
        'for the six neighboring states and one for104, with positive data-distance '
        'radii. The corrected continuous geometry metric excludes unrelated support. '
        'Start from the protected postpeak parent, not any held candidate. Learn '
        'instantaneous and two causal-memory weights jointly with pre-jump and '
        'postpeak kernels. Collect72 fresh +/- constant-context responses; permit '
        'only12 pairs of smaller replacements for excessive continuous responses. '
        'Use the same measured data to compare constant,linear and quadratic-plus-'
        'geometry shared context models. Require predicted lateral improvement of '
        '0.001mm plus ALL original and raw-curvature protections. Test joint,half-step, '
        'early-only and late-only coefficient controls as four complete feedback '
        'functions, never deployed phase switches or feed labels. No sealed testing '
        'or PPO training. End this mechanism if shared credit remains infeasible; '
        'do not extend the same run or weaken acceptance.')
    started=time.perf_counter();deadline=started+460;folder=stage/'state_joint';folder.mkdir()
    choice=dict(paths=paths,variable_kernels=variable_kernels)
    write_json(stage/'baseline.json',data)
    write_json(stage/'support_audit.json',dict(selected=choice,activation=activation,
        phase_paths=phase_paths,context=context.tolist(),parent_paths=len(baseline),
        unaffected_paths=[r['path'] for r in baseline if r['path'] not in paths],
        zero_residual_is_exact_parent=True,no_runtime_time_path_or_angle_lookup=True))
    torch.save(md,stage/'initial_policy.pt')
    _,old_guards=metric_functions(data,selected,stage,md['parent'],deadline)
    retained=[next(r['baseline_shape'] for r in data['retained'] if r['path']==b['path']) for b in selected]
    def losses(records):return np.asarray([teacher_score(r,b,o,barrier=False)+penalty(r) for r,b,o in zip(records,selected,retained)])
    def original_guards(records):return np.concatenate([old_guards(records)]+[guards(r,b) for r,b in zip(records,selected)])
    def lateral(records):return np.asarray([r['shape']['reverse_lateral_significant_mm'] if r['success'] else 1. for r in records])
    def entry(records,u,tag,affected):
        original=original_guards(records);extra=intrinsic_guards(records,selected)
        return dict(tag=tag,coefficients=np.asarray(u).reshape(-1).tolist(),records=records,score=float(losses(records).sum()),
            measured_inequalities=original.tolist(),intrinsic_inequalities=extra.tolist(),
            augmented_measured_inequalities=np.r_[original,extra].tolist(),
            augmented_learning_inequalities=np.r_[event_learning_guards(records,selected,original),extra].tolist(),
            actual_rerun_paths=affected,exact_zero_support_paths=[p for p in paths if p not in affected])
    zero=entry(copy.deepcopy(selected),md['coefficients'],'reused_identity',[])
    history=[zero];md['reused_candidates']=1;cache={tuple(zero['coefficients']):zero}
    def feasible(c):return parameters_in_bounds(c['coefficients'],-1.,1.) and min(c['augmented_measured_inequalities'])>=-1e-8
    def best():return min(filter(feasible,history),key=lambda c:c['score'])
    pool=None;blocks=[];pairs=[]
    try:
        pool=mp.get_context('spawn').Pool(4,worker_initialize,(md,selected,folder,deadline))
        def evaluate(u,kind):
            u=np.asarray(u).reshape(-1);key=tuple(np.round(u,12))
            if key not in cache:
                assert parameters_in_bounds(u,-1.,1.)
                touched=np.any(u.reshape(3,7,7,3)!=0.,axis=(0,1,3))
                affected=sorted(set(p for k in range(7) if touched[k] for p in phase_paths[k]))
                tag=f'preemptive_joint_{len(history)-1}'
                measured=pool.map(worker_evaluate,[(p,u,tag) for p in affected]);by_path={r['path']:r for r in measured}
                records=[by_path.get(r['path'],copy.deepcopy(r)) for r in selected]
                c=entry(records,u,tag,affected);c['kind']=kind;history.append(c);cache[key]=c
                write_json(folder/'results.json',dict(candidates=history,best=best()['tag']))
                print('PREEMPTIVE_JOINT',tag,kind,c['score'],min(c['augmented_measured_inequalities']),float(lateral(records).sum()),flush=True)
            return cache[key]
        shape=(3,4,3);steps=np.empty(shape)
        steps[:,:2]=[.003,.01,.01];steps[:,2:]=[.02,.05,.05]
        jac=np.zeros((len(zero['augmented_learning_inequalities']),*shape))
        gradients=np.zeros((n,*shape));lateral_gradient=np.zeros_like(gradients)
        replacements=0
        for t in range(3):
            for k_local,k in enumerate(variable_kernels):
                for h in range(3):
                    step=steps[t,k_local,h];delta=np.zeros((3,7,7,3));delta[t,0,k,h]=step
                    plus=evaluate(delta,'axis_plus');minus=evaluate(-delta,'axis_minus')
                    size=max(np.max(abs(np.asarray(c['augmented_learning_inequalities'])-zero['augmented_learning_inequalities'])) for c in (plus,minus))
                    if size>1. and replacements<12:
                        factor=max(.05,min(.5,.5/size));delta*=factor;steps[t,k_local,h]*=factor;replacements+=1
                        plus=evaluate(delta,'scaled_axis_plus');minus=evaluate(-delta,'scaled_axis_minus')
                    jac[:,t,k_local,h]=.5*(np.asarray(plus['augmented_learning_inequalities'])-minus['augmented_learning_inequalities'])
                    gradients[:,t,k_local,h]=.5*(losses(plus['records'])-losses(minus['records']))
                    lateral_gradient[:,t,k_local,h]=.5*(lateral(plus['records'])-lateral(minus['records']))
                    pairs.append(dict(time=t,kernel=k,local_kernel=k_local,head=h,step=float(steps[t,k_local,h]),
                        plus=plus['tag'],minus=minus['tag'],initial_response=float(size)))
                    write_json(stage/'response_pairs.json',dict(pairs=pairs,steps=steps.tolist()))
        owners=np.r_[np.repeat(np.arange(n),11),np.repeat(np.arange(n),16),np.repeat(np.arange(n),2)]
        c0=np.asarray(zero['augmented_learning_inequalities']);models=[]
        for count in (1,3,7):
            matrix=np.stack([jac*context[owners,None,c,:,None] for c in range(count)],axis=2).reshape(len(c0),-1)
            g=np.sum(np.stack([gradients*context[:,None,c,:,None] for c in range(count)],axis=2),axis=0).reshape(-1)
            lat=np.sum(np.stack([lateral_gradient*context[:,None,c,:,None] for c in range(count)],axis=2),axis=0).reshape(-1)
            scaled_steps=np.repeat(steps[:,None],count,axis=1).reshape(-1)
            for radius in (1.,4.,16.):
                a=np.vstack([-matrix,lat]);b=np.r_[c0,-.001]
                scale=np.maximum(np.maximum(abs(a).max(1),abs(b)),1e-12)
                bound=np.minimum(radius,1./scaled_steps)
                solved=linprog(g,A_ub=a/scale[:,None],b_ub=b/scale,bounds=list(zip(-bound,bound)),method='highs',
                    options=dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
                row=dict(context_count=count,radius=radius,success=bool(solved.success),status=str(solved.message),
                    predicted_loss_change=float(solved.fun) if solved.success else None,
                    predicted_lateral_change=float(lat@solved.x) if solved.success else None,
                    scaled_proposal=solved.x.tolist() if solved.success else None)
                models.append(row);print('PREEMPTIVE_CAPACITY',{k:v for k,v in row.items() if k!='scaled_proposal'},flush=True)
        write_json(stage/'joint_credit_models.json',dict(models=models,steps=steps.tolist(),pairs=pairs,
            base_constraints=jac.tolist(),base_per_path_loss=gradients.tolist(),base_lateral=lateral_gradient.tolist(),context=context.tolist()))
        useful=[r for r in models if r['success'] and r['predicted_loss_change']<-.001]
        if useful:
            chosen=min(useful,key=lambda r:(r['context_count'],r['radius']))
            count=chosen['context_count'];v=np.asarray(chosen['scaled_proposal']).reshape(3,count,4,3)*steps[:,None]
            joint=np.zeros((3,7,7,3))
            for j,k in enumerate(variable_kernels):joint[:,:count,k]=v[:,:,j]
            early=joint.copy();early[:,:,:5]=0.;late=joint.copy();late[:,:,5:]=0.
            for kind,u in [('joint',joint),('joint_half',joint*.5),('early_only',early),('late_only',late)]:
                c=evaluate(u,kind);blocks.append(dict(method='matched_early_late_joint_control',arm=kind,tag=c['tag'],
                    actual_loss_change=c['score']-zero['score'],lateral_change=float((lateral(c['records'])-lateral(selected)).sum()),
                    actual_guard_minimum=min(c['augmented_measured_inequalities'])))
        pool.close();pool.join();pool=None
        finish(stage,md,baseline,data,best(),zero,history,blocks,choice,parent_file,started)
    except TimeoutError:
        if pool is not None:pool.terminate();pool.join();pool=None
        finish(stage,md,baseline,data,best(),zero,history,blocks,choice,parent_file,started,budget_exhausted=True)
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_preemptive_joint_credit',error=str(exc),wall_s=time.perf_counter()-started,
            adopted=False,goal_complete=False));raise


if __name__=='__main__':mp.freeze_support();main()
