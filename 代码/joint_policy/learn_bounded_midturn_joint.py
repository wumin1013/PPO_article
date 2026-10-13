"""Test coupled exploration over the existing full coefficient bounds."""
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
    source=BASE/'canonical_gate_band_midturn_supported_joint_learning_20260920'
    parent_file=BASE/'canonical_gate_band_decoded_joint_step_learning_20260920/research_candidate.pt'
    md=torch.load(source/'state_joint/candidate.pt',map_location='cpu',weights_only=False)
    data=read(source/'baseline.json');baseline=data['baseline']['records']
    zero=copy.deepcopy(read(source/'state_joint/results.json')['candidates'][0]);history=[zero];selected=zero['records']
    retained=[next(o['baseline_shape'] for o in data['retained'] if o['path']==104)]
    model=read(source/'local_response.json');jac=np.asarray(model['jacobian']);g=np.asarray(model['loss_gradient'])
    out=np.asarray(model['outgoing_gradient']);steps=np.asarray(model['steps']);n=len(steps)
    u0=np.asarray(zero['coefficients']).reshape(3,7,8,3);kernels=[3,7]
    existing=np.concatenate([u0[:,0,k].reshape(-1) for k in kernels])
    stage=register('canonical_gate_band_bounded_midturn_joint_learning_20260920',
        [__file__,parent_file,source/'receipt.json',source/'state_joint/candidate.pt',
         source/'local_response.json',source/'constraint_capacity_attribution.json',
         source/'zero_extension_identity.json',BASE/'observable_memory_joint_policy.py',
         BASE/'shape_event_credit.py',BASE/'learn_intrinsic_joint_recovery.py',
         BASE/'learn_early_memory_joint_policy.py',BASE/'learn_direct_observable_joint_policy.py'],
        dict(wall_s=90,workers=1,maximum_new_vectors=3,coefficient_bounds=[-1.,1.],
             proposal_fractions=[.25,.5,1.],predicted_exit_reduction_mm=.005),
        'Added320ms observable support improves the measured local descent '
        'capacity from0.0000548 to0.000838mm at radius16, still below0.005mm. '
        'An offline speed-guard attribution increases it only to0.00168mm, '
        'so do not remove speed or curvature safeguards. Test the existing '
        'full coefficient bounds[-1,1], which are larger than the prior '
        'finite-difference radius16. Compute one minimum-L1 joint direction '
        'meeting predicted0.005mm outgoing reduction and all guards. Execute '
        'only quarter,half,full parameter steps through complete closed loop. '
        'These learned-parameter controls do not interpolate1ms actions or '
        'deploy rules. If predictions fail, inspect actual state-support '
        'changes rather than treating linear extrapolation as measured credit. '
        'Formal model and decoded16 research candidate stay protected.')
    started=time.perf_counter();deadline=started+75;folder=stage/'state_joint';folder.mkdir()
    write_json(stage/'baseline.json',data)
    choice=dict(paths=[104],variable_kernels=kernels);write_json(stage/'support_audit.json',dict(selected=choice,
        proof_source=str(source/'zero_extension_identity.json'),other87_trajectories_unchanged_by_zero_support=True))
    md['reused_candidates']=1;blocks=[];pool=None
    _,old_guards=metric_functions(data,selected,stage,md['parent'],deadline)
    def feasible(c):return parameters_in_bounds(c['coefficients'],-1.,1.) and min(c['augmented_measured_inequalities'])>=-1e-8
    def best():return min(filter(feasible,history),key=lambda c:c['score'])
    try:
        values=np.asarray(zero['augmented_learning_inequalities'])
        a=np.vstack([np.c_[-jac,np.zeros((len(values),n))],np.c_[np.eye(n),-np.eye(n)],
            np.c_[-np.eye(n),-np.eye(n)],np.r_[out,np.zeros(n)],np.r_[g,np.zeros(n)]])
        b=np.r_[values,np.zeros(2*n),-.005,-.001]
        scale=np.maximum(np.maximum(abs(a).max(1),abs(b)),1e-12)
        bounds=list(zip((-1-existing)/steps,(1-existing)/steps))+[(0.,None)]*n
        solved=linprog(np.r_[np.zeros(n),np.ones(n)],A_ub=a/scale[:,None],b_ub=b/scale,bounds=bounds,method='highs',
            options=dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
        proposal=dict(success=bool(solved.success),status=str(solved.message))
        if solved.success:
            step=solved.x[:n];delta=np.zeros_like(u0)
            for i,k in enumerate(kernels):delta[:,0,k]=(step*steps)[9*i:9*i+9].reshape(3,3)
            proposal.update(scaled_parameters=step.tolist(),coefficient_delta=delta.reshape(-1).tolist(),
                predicted_exit_change_mm=float(out@step),predicted_loss_change=float(g@step),
                predicted_guards=(values+jac@step).tolist())
            pool=mp.get_context('spawn').Pool(1,worker_initialize,(md,selected,folder,deadline))
            with np.load(selected[0]['trace']) as z:prefix=z['rows'][:300].copy()
            for fraction in (.25,.5,1.):
                u=u0+fraction*delta;tag=f'bounded_joint_{fraction:g}'
                assert parameters_in_bounds(u,-1.,1.)
                records=pool.map(worker_evaluate,[(104,u.reshape(-1),tag)])
                with np.load(records[0]['trace']) as z:assert np.array_equal(z['rows'][:300],prefix)
                original=np.r_[old_guards(records),guards(records[0],selected[0])];extra=intrinsic_guards(records,selected)
                c=dict(tag=tag,coefficients=u.reshape(-1).tolist(),records=records,
                    score=float(teacher_score(records[0],selected[0],retained[0],barrier=False)+penalty(records[0])),
                    measured_inequalities=original.tolist(),augmented_measured_inequalities=np.r_[original,extra].tolist(),
                    augmented_learning_inequalities=np.r_[shape_event_learning_guards(records,selected,original),extra].tolist(),
                    actual_rerun_paths=[104],exact_zero_support_paths=[])
                history.append(c);blocks.append(dict(fraction=fraction,tag=tag,feasible=feasible(c)))
                write_json(folder/'results.json',dict(candidates=history,best=best()['tag']))
                print('BOUNDED_MIDTURN',fraction,c['score'],min(c['augmented_measured_inequalities']),
                    records[0]['shape']['out_edge_outward_mm']-selected[0]['shape']['out_edge_outward_mm'],flush=True)
            pool.close();pool.join();pool=None
        write_json(stage/'proposal.json',proposal)
        finish(stage,md,baseline,data,best(),zero,history,blocks,choice,parent_file,started)
    except TimeoutError:
        if pool is not None:pool.terminate();pool.join();pool=None
        finish(stage,md,baseline,data,best(),zero,history,blocks,choice,parent_file,started,budget_exhausted=True)
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_bounded_midturn_learning',error=str(exc),wall_s=time.perf_counter()-started,
            actual_new_candidates=len(history)-1,adopted=False,goal_complete=False));raise


if __name__=='__main__':mp.freeze_support();main()
