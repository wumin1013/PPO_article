"""Learn pre-reversal and later three-action credit from actual full returns."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import copy
import argparse
import time
import multiprocessing as mp
import numpy as np
import torch
from scipy.optimize import linprog
from audit_context_restoration_scaling import solve_correction
from gate_band_state import GateBandRuntime
from saved_memory_action_identity import SavedMemoryActionIdentity
from duration_joint_metrics import guards,penalty
from shape_event_credit import shape_event_learning_guards
from learn_intrinsic_joint_recovery import intrinsic_guards
from learn_complete_joint_teachers import teacher_score,parameters_in_bounds
from learn_early_memory_joint_policy import worker_initialize,worker_evaluate,metric_functions
from learn_direct_observable_joint_policy import finish
from study_io import BASE,register,write_json,receipt,sha
from search_complete_return import read


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--restore',action='store_true')
    parser.add_argument('--late-recenter',action='store_true')
    parser.add_argument('--active-window',action='store_true')
    args=parser.parse_args()
    if args.late_recenter:args.restore=True
    assert not (args.active_window and args.restore)
    torch.set_num_threads(1)
    source=BASE/('canonical_gate_band_reversal_window_local_recovery_20260920' if args.active_window else
                'canonical_gate_band_early_geometry_credit_recovery_20260920')
    parent_file=source/'research_candidate.pt'
    assert sha(parent_file)==('132e3914433db31aa72c0cf35de80b6d0b010b1a49f82d8ca30ab79776b6f8dd' if args.active_window else
                              '7605e4dd430d529562a430696ea0410d24c0fce839975174ffece414e89febb8')
    stem='active_window' if args.active_window else 'reversal_window_exact'
    md=torch.load(source/(stem+'_initial_policy.pt'),map_location='cpu',weights_only=False)
    dimensions=int(md['coefficients'].size);centers=len(md['centers'])
    data=read(source/'baseline.json');data['baseline']=read(source/'results.json')['candidates'][0]
    baseline=data['baseline']['records'];support=read(source/(stem+'_support_audit.json'))
    assert support['parent_relative_innovations_recomputed']
    paths=support['affected_paths'];selected=[r for r in baseline if r['path'] in paths]
    with np.load(source/(stem+'_affected_replay.npz')) as z:replay={k:z[k].copy() for k in z.files}
    identity=SavedMemoryActionIdentity(md,replay,GateBandRuntime.decode_action)
    assert all(identity.unchanged(p,np.zeros(dimensions)) for p in paths)
    previous=BASE/('canonical_gate_band_reversal_window_secant_recovery_20260920' if args.late_recenter else
                   'canonical_gate_band_reversal_window_joint_credit_20260920')
    if args.restore:assert read(previous/'receipt.json')['status'].startswith('completed')
    stage=register('canonical_gate_band_active_reversal_window_joint_credit_20260920' if args.active_window else
                  'canonical_gate_band_reversal_window_local_recovery_20260920' if args.late_recenter else
                  'canonical_gate_band_reversal_window_secant_recovery_20260920' if args.restore else
                   'canonical_gate_band_reversal_window_joint_credit_20260920',
        [__file__,parent_file,source/'receipt.json',source/'selection_review.json',
         source/'joint_outcome_audit.json',source/'selected_actor_identity_audit.json',
         source/(stem+'_initial_policy.pt'),source/(stem+'_support_audit.json'),
         source/(stem+'_affected_replay.npz'),BASE/'prepare_reversal_window_support.py',
         BASE/'saved_memory_action_identity.py',BASE/'observable_memory_joint_policy.py',
         BASE/'joint_teacher_state_policy.py',BASE/'shape_event_credit.py',BASE/'duration_joint_metrics.py',
         BASE/'learn_intrinsic_joint_recovery.py',BASE/'learn_early_memory_joint_policy.py',
         BASE/'learn_direct_observable_joint_policy.py']+
         ([previous/'receipt.json',previous/'state_joint/results.json',previous/'measured_joint_credit.json',
           previous/('secant_recovery_audit.json' if args.late_recenter else 'joint_proposals.json'),
           BASE/'audit_context_restoration_scaling.py'] if args.restore else []),
        dict(wall_s=90 if args.restore else 240,workers=4,symmetric_responses=12 if args.late_recenter else 0 if args.restore else 2*dimensions,
             maximum_recovery_vectors=3 if args.late_recenter else 8 if args.restore else 0,maximum_proposals=0 if args.restore else 3,
             maximum_factorial_controls=2,response_steps=[.002,.005,.005],
             trust_radii=[.5,2.,8.],coefficient_bounds=[-1.,1.]),
        ('The new132e391 candidate passes all original/raw curvature guards and '
         'all eight spatial peak/energy checks. Matched controls show early weights '
         'alone reduce lateral movement but worsen reverse/cruise; late weights '
         'alone worsen exit. The complete learned joint action passes. Residual '
         'wrong-way movement remains. Add a second early observable-state center '
         'at the measured maximum wrong-way incoming velocity, alongside the '
         'last observed state before reversal and the two later centers. Learn '
         'all36 weights from72 new central complete-return responses. This opens '
         'action support DURING the error instead of only before and after it. '
         'Timing chooses training observations only; actor inputs are unchanged. '
         if args.active_window else '')+
        ('Both four-correction arms leave only the interpolated exit-duration guard '
         'violated (2.61e-6 normalized at secant_63). Successive raw trajectories '
         'differ, so an action-update plateau is not established. Keep the learned '
         'pre-reversal action, and measure12 new local symmetric responses for the '
         'six instantaneous300/320ms steering/feed/preview coefficients at this '
         'actual near-feasible point, steps ten times smaller than the old origin. '
         'At most3 local corrections; same90s budget and every original guard. '
         'This tests stale constraint credit directly without repeating old points. '
         if args.late_recenter else '')+
        ('Reuse all57 completed points and their measured joint response. The radius8 '
         'direction predicts negative lateral movement reduction0.000783mm and '
         'actually achieves0.000769mm, but misses six other guards; accurate reward '
         'response does not imply accurate constraint response. From that exact '
         'joint_56 point compare fixed local Jacobian versus secant-updated Jacobian '
         'recovery, at most4 corrections each. Recenter guard values on each true '
         'complete return, and preserve at least75 percent of its negative-lateral '
         'benefit. LP radii4 then16; positive prediction margin1e-6, with zero-margin '
         'fallback only if infeasible. Real acceptance is unchanged.90s, no old '
         'physical point repeated. ' if args.restore else '')+
        'Use the audited current parent and its saved residual wrong-way motion. '
        'Place the new early center at the OBSERVABLE parent state just before '
        'the residual event (time selects a training observation only), with joint '
        'three-action feedback and causal0.5/0.8 memories plus300/320ms geometry '
        'support. Recompute innovations relative to the current parent, not its '
        'ancestor. Measure the registered symmetric complete-return responses, then minimize '
        'predicted negative lateral movement with every original/current guard '
        'and predicted full loss nonincrease. Test three trust radii, then early '
        'versus later learned-coefficient ablations only if a feasible improvement '
        'exists. Those ablations diagnose credit and are never deployed switching '
        'rules. Reuse exact decoded-action-identical paths only. No PPO, no sealed '
        'tests, no relaxed physics/shape acceptance.240s; if response fails, audit '
        'its nonlinear error and measured action support rather than add iterations.')
    started=time.perf_counter();deadline=started+(80. if args.restore else 220.);pool=None
    folder=stage/'state_joint';folder.mkdir();history=[];cache={};blocks=[]
    choice=dict(paths=paths,variable_kernels=list(range(centers)))
    write_json(stage/'baseline.json',data)
    write_json(stage/'support_audit.json',dict(selected=choice,source_audit=support))
    torch.save(md,stage/'initial_policy.pt')
    _,old_guards=metric_functions(data,selected,stage,md['parent'],deadline)
    retained=[next(o['baseline_shape'] for o in data['retained'] if o['path']==r['path']) for r in selected]
    def entry(records,u,tag,actual,unchanged):
        original=np.concatenate([old_guards(records)]+[guards(r,b) for r,b in zip(records,selected)])
        extra=intrinsic_guards(records,selected)
        cost=float(sum(teacher_score(r,b,o,barrier=False)+penalty(r)+
            (4*(r['shape']['reverse_lateral_total_mm']/.005)**2 if r['success'] else 1e6)
            for r,b,o in zip(records,selected,retained)))
        return dict(tag=tag,coefficients=np.asarray(u).tolist(),records=records,score=cost,
            lateral_mm=sum(r['shape']['reverse_lateral_total_mm'] if r['success'] else 1. for r in records),
            measured_inequalities=original.tolist(),augmented_measured_inequalities=np.r_[original,extra].tolist(),
            augmented_learning_inequalities=np.r_[shape_event_learning_guards(records,selected,original),extra].tolist(),
            actual_rerun_paths=actual,exact_zero_support_paths=unchanged)
    def feasible(c):return parameters_in_bounds(c['coefficients'],-1.,1.) and min(c['augmented_measured_inequalities'])>=-1e-8
    def best():return min(filter(feasible,history),key=lambda c:c['score'])
    zero=entry(copy.deepcopy(selected),np.zeros(dimensions),'reused_parent',[],paths)
    history.append(zero);cache[tuple(zero['coefficients'])]=zero;md['reused_candidates']=1
    if args.restore:
        history=copy.deepcopy(read(previous/'state_joint/results.json')['candidates'])
        assert history[0]['score']==zero['score']
        cache={tuple(np.round(c['coefficients'],12)):c for c in history}
        md['reused_candidates']=len(history)
    try:
        pool=mp.get_context('spawn').Pool(4,worker_initialize,(md,selected,folder,deadline))
        def evaluate(u,kind):
            u=np.asarray(u);key=tuple(np.round(u,12))
            if key not in cache:
                assert parameters_in_bounds(u,-1.,1.)
                if time.perf_counter()>deadline:raise TimeoutError('Reversal-window joint credit budget')
                unchanged=[p for p in paths if identity.unchanged(p,u)]
                actual=[p for p in paths if p not in unchanged];tag=f'{kind}_{len(history)-1}'
                measured=pool.map(worker_evaluate,[(p,u,tag) for p in actual]);by={r['path']:r for r in measured}
                records=[by.get(r['path'],copy.deepcopy(r)) for r in selected]
                c=entry(records,u,tag,actual,unchanged);history.append(c);cache[key]=c
                write_json(folder/'results.json',dict(candidates=history,best=best()['tag']))
                if len(history)%10==0:print('REVERSAL_CREDIT',len(history)-1,c['score'],min(c['augmented_measured_inequalities']),best()['tag'],flush=True)
            return cache[key]
        steps=np.tile([.002,.005,.005],dimensions//3);pairs=[]
        for i in ([] if args.restore else range(dimensions)):
            offset=np.eye(dimensions)[i]*steps
            pairs.append((evaluate(offset,'plus'),evaluate(-offset,'minus')))
        if args.restore:
            if args.late_recenter:
                origin=next(c for c in history if c['tag']=='secant_63')
                indices=np.arange(3,9);steps*=.1
                for i in indices:
                    offset=np.eye(27)[i]*steps
                    pairs.append((evaluate(np.array(origin['coefficients'])+offset,'local_plus'),
                                  evaluate(np.array(origin['coefficients'])-offset,'local_minus')))
                jac=np.array([(np.array(a['augmented_learning_inequalities'])-b['augmented_learning_inequalities'])/2 for a,b in pairs]).T
                loss=np.array([(a['score']-b['score'])/2 for a,b in pairs])
                lateral=np.array([(a['lateral_mm']-b['lateral_mm'])/2 for a,b in pairs])
            else:
                model=read(previous/'measured_joint_credit.json')
                jac=np.array(model['jacobian']);loss=np.array(model['loss_gradient']);lateral=np.array(model['lateral_gradient'])
                indices=np.arange(27)
        else:
            jac=np.array([(np.array(a['augmented_learning_inequalities'])-b['augmented_learning_inequalities'])/2 for a,b in pairs]).T
            loss=np.array([(a['score']-b['score'])/2 for a,b in pairs])
            lateral=np.array([(a['lateral_mm']-b['lateral_mm'])/2 for a,b in pairs])
        values=np.array(zero['augmented_learning_inequalities'])
        write_json(stage/'measured_joint_credit.json',dict(steps=steps.tolist(),jacobian=jac.tolist(),
            loss_gradient=loss.tolist(),lateral_gradient=lateral.tolist(),
            pairs=[[a['tag'],b['tag']] for a,b in pairs]))
        matrix=np.vstack([-jac,loss]);rhs=np.r_[values,0.]
        row_scale=np.maximum(np.maximum(np.max(abs(matrix),axis=1),abs(rhs)),1e-12)
        if args.restore:
            origin=next(c for c in history if c['tag']==('secant_63' if args.late_recenter else 'joint_56'))
            lateral_limit=origin['lateral_mm']+.25*(zero['lateral_mm']-origin['lateral_mm'])
            score_limit=origin['score']+.25*(zero['score']-origin['score'])
            for arm in (('recentered',) if args.late_recenter else ('fixed','secant')):
                current=origin;local_jac=np.vstack([jac,-lateral]);local_loss=loss.copy()
                def measured_values(c):return np.r_[c['augmented_learning_inequalities'],lateral_limit-c['lateral_mm']]
                for iteration in range(3 if args.late_recenter else 4):
                    if feasible(current):break
                    solution=None
                    for margin in (1e-6,0.):
                        for radius in (4.,16.):
                            solution=solve_correction(local_jac,local_loss,measured_values(current),
                                score_limit-current['score'],margin=margin,radius=radius)
                            if solution['success']:break
                        if solution['success']:break
                    row=dict(arm=arm,iteration=iteration,anchor=current['tag'],margin=margin,radius=radius,solution=solution)
                    if not solution['success']:
                        blocks.append(row);break
                    delta=np.array(solution['scaled_parameter_step'])
                    offset=np.zeros(27);offset[indices]=delta*steps[indices]
                    measured=evaluate(np.array(current['coefficients'])+offset,arm)
                    row.update(measured=measured['tag'],actual_guard_min=min(measured['augmented_measured_inequalities']),
                        actual_lateral_change_mm=measured['lateral_mm']-zero['lateral_mm'])
                    blocks.append(row);norm=float(delta@delta)
                    if norm<=1e-18:break
                    if arm in ('secant','recentered'):
                        local_jac+=np.outer(measured_values(measured)-measured_values(current)-local_jac@delta,delta)/norm
                        local_loss+=(measured['score']-current['score']-local_loss@delta)*delta/norm
                    current=measured
                    write_json(stage/'secant_recovery_audit.json',dict(blocks=blocks,lateral_limit=lateral_limit,
                        score_limit=score_limit,acceptance_unchanged=True))
        for radius in ([] if args.restore else (.5,2.,8.)):
            solved=linprog(lateral/max(np.max(abs(lateral)),1e-12),
                A_ub=matrix/row_scale[:,None],b_ub=rhs/row_scale,
                bounds=[(-radius,radius)]*dimensions,method='highs',
                options=dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
            row=dict(radius=radius,solver_success=bool(solved.success),message=str(solved.message))
            if solved.success:
                row.update(predicted_lateral_change_mm=float(lateral@solved.x),
                    predicted_loss_change=float(loss@solved.x),predicted_guard_min=float(np.min(values+jac@solved.x)))
                if lateral@solved.x < -1e-7:
                    measured=evaluate(steps*solved.x,'joint')
                    row.update(measured=measured['tag'],actual_lateral_change_mm=measured['lateral_mm']-zero['lateral_mm'],
                        actual_guard_min=min(measured['augmented_measured_inequalities']))
            blocks.append(row);write_json(stage/'joint_proposals.json',blocks)
        candidate=best()
        if candidate['score']<zero['score']-.001:
            tensor=np.array(candidate['coefficients']).reshape(3,centers,3)
            early=tensor.copy();early[:,centers-2:]=0.
            late=tensor.copy();late[:,:centers-2]=0.
            controls=[evaluate(early.ravel(),'early_control'),evaluate(late.ravel(),'late_control')]
            write_json(stage/'factorial_credit.json',dict(parent=zero['tag'],joint=candidate['tag'],
                controls=[c['tag'] for c in controls],interpretation='Matched learned-coefficient removals in complete state-feedback policies; not phase switching or a deployable rule.'))
        pool.close();pool.join();pool=None
        finish(stage,md,baseline,data,best(),zero,history,blocks,choice,parent_file,started)
    except TimeoutError:
        if pool is not None:pool.terminate();pool.join();pool=None
        finish(stage,md,baseline,data,best(),zero,history,blocks,choice,parent_file,started,budget_exhausted=True)
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_reversal_window_joint_credit',error=repr(exc),
            new_candidates=len(history)-1,adopted=False,goal_complete=False));raise


if __name__=='__main__':mp.freeze_support();main()
