"""Relearn complete joint credit where geometry-supported compensation remains active."""
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
from learn_motion_robust_midturn_joint import metric_weights
from duration_joint_metrics import guards,penalty
from shape_event_credit import shape_event_learning_guards
from learn_intrinsic_joint_recovery import intrinsic_guards
from learn_complete_joint_teachers import teacher_score,parameters_in_bounds
from learn_early_memory_joint_policy import worker_initialize,worker_evaluate,metric_functions
from learn_direct_observable_joint_policy import finish
from study_io import BASE,register,write_json,receipt
from search_complete_return import read


def main():
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_motion_robust_midturn_completion_20260920'
    parent_file=BASE/'canonical_gate_band_decoded_joint_step_learning_20260920/research_candidate.pt'
    md=torch.load(source/'state_joint/candidate.pt',map_location='cpu',weights_only=False)
    md['per_center_distance_weights']=metric_weights(md,0.)
    old=read(source/'state_joint/results.json')['candidates'];zero=copy.deepcopy(old[0])
    anchor=copy.deepcopy(next(c for c in old if c['tag']=='metric_0_joint_0.25'))
    full=copy.deepcopy(next(c for c in old if c['tag']=='metric_0_joint_1'))
    history=[zero,anchor,full];md['reused_candidates']=3
    data=read(source/'baseline.json');baseline=data['baseline']['records'];selected=zero['records']
    assert [r['path'] for r in selected]==[104]
    retained=[next(o['baseline_shape'] for o in data['retained'] if o['path']==104)]
    u0=np.asarray(anchor['coefficients']).reshape(3,7,8,3);kernels=[3,7]
    stage=register('canonical_gate_band_stable_support_joint_credit_learning_20260920',
        [__file__,parent_file,source/'receipt.json',source/'support_audit.json',source/'state_joint/candidate.pt',
         source/'state_joint/results.json',BASE/'learn_motion_robust_midturn_joint.py',
         BASE/'joint_teacher_state_policy.py',BASE/'observable_memory_joint_policy.py',
         BASE/'shape_event_credit.py',BASE/'learn_intrinsic_joint_recovery.py',
         BASE/'learn_early_memory_joint_policy.py',BASE/'learn_direct_observable_joint_policy.py'],
        dict(wall_s=180,workers=1,new_symmetric_vectors=36,maximum_joint_vectors=8,
             derivative_steps=[.002,.005,.005],radii=[1.,4.,16.],
             predicted_exit_reduction_targets_mm=[.005,.0025,.001],coefficient_bounds=[-1.,1.]),
        'Matched identical coefficients at quarter scale improve104 outgoing '
        'excursion with reference-geometry distance but worsen it with full '
        'motion-state distance. Both still violate other guards; full-scale '
        'extrapolation fails for both metrics. Fix the geometry-distance '
        'representation and recompute36 full-loop symmetric responses AROUND '
        'the actual metric0 quarter control. Jointly learn the18 weights of '
        'both observable centers and their causal memories. Use minimum-L1 '
        'steps preserving all original/current speed,shape,physical,completion '
        'and intrinsic-curvature guards. Try predicted exit reductions0.005, '
        '0.0025,0.001mm in that order, each within radii1/4/16; at most8 '
        'joint validations/restorations. Reuse only exact same-function '
        'baseline and the two completed metric0 controls. If no actual '
        'candidate passes, keep decoded16 and change the mechanism based '
        'on these responses. Not PPO, feed-label fitting, or deployed rules.')
    started=time.perf_counter();deadline=started+165;folder=stage/'state_joint';folder.mkdir()
    write_json(stage/'baseline.json',data);choice=dict(paths=[104],variable_kernels=kernels)
    write_json(stage/'support_audit.json',dict(selected=choice,
        exact_zero_support_proof=str(source/'support_audit.json'),other87_paths_unchanged=True))
    _,old_guards=metric_functions(data,selected,stage,md['parent'],deadline)
    def entry(records,u,tag):
        original=np.r_[old_guards(records),guards(records[0],selected[0])];extra=intrinsic_guards(records,selected)
        return dict(tag=tag,coefficients=np.asarray(u).reshape(-1).tolist(),records=records,
            score=float(teacher_score(records[0],selected[0],retained[0],barrier=False)+penalty(records[0])),
            measured_inequalities=original.tolist(),augmented_measured_inequalities=np.r_[original,extra].tolist(),
            augmented_learning_inequalities=np.r_[shape_event_learning_guards(records,selected,original),extra].tolist(),
            actual_rerun_paths=[104],exact_zero_support_paths=[])
    def feasible(c):return parameters_in_bounds(c['coefficients'],-1.,1.) and min(c['augmented_measured_inequalities'])>=-1e-8
    def best():return min(filter(feasible,history),key=lambda c:c['score'])
    def violation(c):return float(np.linalg.norm(np.minimum(c['augmented_measured_inequalities'],0.)))
    def outgoing(c):return c['records'][0]['shape']['out_edge_outward_mm']
    blocks=[];pool=None
    try:
        with np.load(selected[0]['trace']) as z:prefix=z['rows'][:300].copy()
        pool=mp.get_context('spawn').Pool(1,worker_initialize,(md,selected,folder,deadline))
        cache={tuple(np.round(c['coefficients'],12)):c for c in history}
        def evaluate(u,kind):
            u=np.asarray(u).reshape(-1);key=tuple(np.round(u,12))
            if key not in cache:
                assert parameters_in_bounds(u,-1.,1.) and len(history)-3<44
                tag=f'stable_joint_{len(history)-3}'
                records=pool.map(worker_evaluate,[(104,u,tag)])
                with np.load(records[0]['trace']) as z:assert np.array_equal(z['rows'][:300],prefix)
                c=entry(records,u,tag);c['kind']=kind;history.append(c);cache[key]=c
                write_json(folder/'results.json',dict(candidates=history,best=best()['tag']))
                print('STABLE_JOINT',tag,kind,c['score'],violation(c),outgoing(c)-outgoing(zero),flush=True)
            return cache[key]
        steps=np.tile([.002,.005,.005],6);n=len(steps)
        def expand(v):
            delta=np.zeros_like(u0)
            for j,k in enumerate(kernels):delta[:,0,k]=(np.asarray(v)*steps)[9*j:9*j+9].reshape(3,3)
            return delta.reshape(-1)
        size=len(zero['augmented_learning_inequalities']);jac=np.zeros((size,n));g=np.zeros(n);out_grad=np.zeros(n);pairs=[]
        for j in range(n):
            direction=np.eye(n)[j];plus=evaluate(u0.reshape(-1)+expand(direction),'local_plus');minus=evaluate(u0.reshape(-1)-expand(direction),'local_minus')
            jac[:,j]=.5*(np.asarray(plus['augmented_learning_inequalities'])-minus['augmented_learning_inequalities'])
            g[j]=.5*(plus['score']-minus['score']);out_grad[j]=.5*(outgoing(plus)-outgoing(minus))
            pairs.append(dict(coordinate=j,plus=plus['tag'],minus=minus['tag']))
            write_json(stage/'local_response.json',dict(pairs=pairs,steps=steps.tolist(),jacobian=jac.tolist(),
                loss_gradient=g.tolist(),outgoing_gradient=out_grad.tolist()))
        def solve(current,target):
            values=np.asarray(current['augmented_learning_inequalities'])
            a=np.vstack([np.c_[-jac,np.zeros((len(values),n))],np.c_[np.eye(n),-np.eye(n)],
                np.c_[-np.eye(n),-np.eye(n)],np.r_[out_grad,np.zeros(n)],np.r_[g,np.zeros(n)]])
            b=np.r_[values,np.zeros(2*n),outgoing(zero)-target-outgoing(current),zero['score']-.001-current['score']]
            scale=np.maximum(np.maximum(abs(a).max(1),abs(b)),1e-12)
            for radius in (1.,4.,16.):
                result=linprog(np.r_[np.zeros(n),np.ones(n)],A_ub=a/scale[:,None],b_ub=b/scale,
                    bounds=[(-radius,radius)]*n+[(0.,None)]*n,method='highs',
                    options=dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
                if result.success:return result.x[:n],radius
            return None,radius
        step=None;target=None
        for required in (.005,.0025,.001):
            step,radius=solve(anchor,required)
            blocks.append(dict(target=required,radius=radius,success=step is not None))
            if step is not None:target=required;break
        current=anchor
        for iteration in range(8):
            if step is None:break
            c=evaluate(np.asarray(current['coefficients'])+expand(step),'joint_complete_credit')
            block=dict(iteration=iteration,anchor=current['tag'],tag=c['tag'],target=target,violation=violation(c))
            blocks.append(block)
            denom=step@step
            jac+=np.outer(np.asarray(c['augmented_learning_inequalities'])-current['augmented_learning_inequalities']-jac@step,step)/denom
            g+=(c['score']-current['score']-g@step)*step/denom
            out_grad+=(outgoing(c)-outgoing(current)-out_grad@step)*step/denom
            if feasible(c):current=c;break
            if violation(c)<violation(current):current=c
            write_json(stage/'recovery.json',dict(blocks=blocks,nearest=current['tag']))
            step,radius=solve(current,target)
        pool.close();pool.join();pool=None
        finish(stage,md,baseline,data,best(),zero,history,blocks,choice,parent_file,started)
    except TimeoutError:
        if pool is not None:pool.terminate();pool.join();pool=None
        finish(stage,md,baseline,data,best(),zero,history,blocks,choice,parent_file,started,budget_exhausted=True)
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_stable_support_joint',error=str(exc),wall_s=time.perf_counter()-started,
            actual_new_candidates=len(history)-3,adopted=False,goal_complete=False));raise


if __name__=='__main__':mp.freeze_support();main()
