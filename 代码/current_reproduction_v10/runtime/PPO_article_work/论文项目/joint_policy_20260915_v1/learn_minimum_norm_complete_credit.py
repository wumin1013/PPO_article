"""Minimum-norm joint updates with matched measured deterministic action credit."""
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
from scipy.optimize import minimize,linprog
from learn_early_memory_joint_policy import worker_initialize,worker_evaluate,metric_functions
from head_counterfactual_rollouts import cost
from duration_joint_metrics import guards
from shape_event_credit import shape_event_learning_guards
from learn_intrinsic_joint_recovery import intrinsic_guards
from learn_direct_observable_joint_policy import finish
from saved_memory_action_identity import SavedMemoryActionIdentity
from gate_band_state import GateBandRuntime
from study_io import BASE,register,write_json,receipt,sha
from search_complete_return import read


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--joint-secants',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1)
    started=time.perf_counter();deadline=started+220.
    source=BASE/'canonical_gate_band_deterministic_local_credit_20260920'
    parent_file=BASE/'canonical_gate_band_reference_supported_joint_learning_20260920/research_candidate.pt'
    assert sha(parent_file)=='ded2a93f85f3503a8c09bf59fd79e48c8f2cd187874f07789c53b8863c936229'
    assert not read(source/'receipt.json')['research_acceptable']
    data=read(source/'baseline.json');baseline=data['baseline']['records']
    support=read(source/'support_audit.json');choice=support['selected'];paths=choice['paths']
    selected=[r for r in baseline if r['path'] in paths]
    retained={r['path']:r['baseline_shape'] for r in data['retained'] if 'baseline_shape' in r}
    old_history=read(source/'state_joint/results.json')['candidates']
    secant_sources=[BASE/n for n in ('canonical_gate_band_deterministic_complete_credit_20260920',
        'canonical_gate_band_deterministic_local_credit_20260920',
        'canonical_gate_band_minimum_norm_complete_credit_20260920')] if args.joint_secants else []
    joint_records=[c for s in secant_sources for c in read(s/'state_joint/results.json')['candidates']
                   if c['tag']!='reused_parent']
    replay_file=BASE/'canonical_gate_band_shared_reference_transfer_joint_learning_20260920/affected_replay.npz'
    with np.load(replay_file) as z:replay={k:z[k].copy() for k in z.files}
    stage=register('canonical_gate_band_joint_secant_credit_20260920' if args.joint_secants
                   else 'canonical_gate_band_minimum_norm_complete_credit_20260920',
        [__file__,parent_file,source/'receipt.json',source/'state_joint/results.json',replay_file]+[
         source/(mode+suffix) for mode in ('geometry','dynamic')
         for suffix in ('_initial_policy.pt','_complete_gradient.npz')]+[BASE/n for n in (
         'observable_memory_joint_policy.py','saved_memory_action_identity.py',
         'learn_early_memory_joint_policy.py','learn_direct_observable_joint_policy.py',
         'head_counterfactual_rollouts.py','duration_joint_metrics.py','shape_event_credit.py',
         'learn_intrinsic_joint_recovery.py')]+[s/'state_joint/results.json' for s in secant_sources],
        dict(wall_s=240,workers=4,new_suffixes=0,new_stochastic_episodes=0,maximum_new_policies=4,
             predicted_gain_fraction=.9,actor_arms=['geometry','dynamic'],trust_bounds=[.05,.15]),
        ('Add a whole-policy response calibration to the prior deterministic-credit '
         'model. For each context actor reuse all SIX measured joint-policy responses '
         'from the full-radius,local-radius and smaller-norm experiments. Apply the '
         'minimum-Frobenius secant correction to reward AND all constraint gradients; '
         'retain only numerically supported SVD directions (rcond1e-6). Add each new '
         'complete response before the next trust proposal. Keep the same actor, '
         'coefficient box, two action-trust bounds and90percent-gain norm regularizer. '
         'A corrected LP provides the feasible QP initialization. This tests feedback '
         'of measured simultaneous/time-coupled parameter responses into the learning '
         'model; it does not assume marginal action effects add exactly. No new suffix '
         'collection, no deployment of a search schedule, no estimator-bias claim. '
         'All actual original guards remain required. ' if args.joint_secants else '')+
        'The same measured deterministic credit leads to LP updates with80-95 '
        'coefficients at the box limit, despite very small parent-state action changes. '
        'Test optimizer regularization, not more data: for each previous LP solution '
        'minimize squared coefficient norm subject to ALL identical linearized '
        'constraint and action-trust inequalities, the same coefficient bounds, and '
        'at least90% of its predicted loss gain. Start the QP from its saved feasible '
        'LP solution. Compare both existing162-parameter context actors and both trust '
        'bounds. Run only genuinely changed smaller-norm proposals; no old rollout '
        'is repeated. Full actual guards remain the selection criterion. This is '
        'deterministic complete-credit joint parameter optimization, not PPO or '
        'supervised action-label fitting. A convex surrogate solution is not a '
        'guarantee for the real nonlinear trajectory.240s then inspect actual errors.')
    folder=stage/'state_joint';folder.mkdir()
    write_json(stage/'baseline.json',data);write_json(stage/'support_audit.json',support)
    history,blocks,models=[],[],{}
    pool=None
    try:
        for mode in ('geometry','dynamic'):
            md=torch.load(source/(mode+'_initial_policy.pt'),map_location='cpu',weights_only=False)
            models[mode]=md;torch.save(md,stage/(mode+'_initial_policy.pt'))
            _,old_guards=metric_functions(data,selected,stage,md['parent'],deadline)
            def entry(records,u,tag):
                first=np.concatenate([old_guards(records)]+[guards(r,b) for r,b in zip(records,selected)])
                extra=intrinsic_guards(records,selected)
                return dict(tag=tag,context_arm=mode,coefficients=np.asarray(u).tolist(),records=records,
                    score=sum(cost(r,b,retained[r['path']]) for r,b in zip(records,selected)),
                    measured_inequalities=first.tolist(),augmented_measured_inequalities=np.r_[first,extra].tolist(),
                    augmented_learning_inequalities=np.r_[shape_event_learning_guards(records,selected,first),extra].tolist(),
                    actual_rerun_paths=[],exact_zero_support_paths=paths)
            if not history:
                zero=entry(copy.deepcopy(selected),np.zeros(md['coefficients'].size),'reused_parent');history.append(zero)
            def best():return min((c for c in history if min(c['augmented_measured_inequalities'])>=-1e-8),key=lambda c:c['score'])
            with np.load(source/(mode+'_complete_gradient.npz')) as z:
                g,jac,margins,parent=[z[k].copy() for k in ('loss_gradient','constraint_jacobian','baseline_margins','parent_matrix')]
            original_gradient=np.vstack([g,jac])
            identity=SavedMemoryActionIdentity(md,replay,GateBandRuntime.decode_action)
            pool=mp.get_context('spawn').Pool(4,worker_initialize,(md,selected,folder,deadline))
            for trust in (.05,.15):
                if time.perf_counter()>deadline:raise TimeoutError('Minimum-norm optimizer budget')
                old=next(c for c in old_history if c['tag']==f'{mode}_trust{trust}')
                initial=np.array(old['coefficients'])
                secant_audit=None
                if args.joint_secants:
                    known=[c for c in joint_records+history[1:] if c['context_arm']==mode]
                    inputs=np.array([c['coefficients'] for c in known])
                    outputs=np.array([[c['score'],*c['augmented_learning_inequalities']] for c in known])
                    base=np.r_[zero['score'],margins]
                    residual=outputs-base-inputs@original_gradient.T
                    correction,_,rank,singular=np.linalg.lstsq(inputs,residual,rcond=1e-6)
                    calibrated=original_gradient+correction.T
                    g,jac=calibrated[0],calibrated[1:]
                    errors=outputs-base-inputs@calibrated.T
                    secant_audit=dict(points=len(known),rank=int(rank),singular_values=singular.tolist(),
                        maximum_fit_error=float(np.max(abs(errors))),new_full_policy_responses_reused=True)
                    np.savez_compressed(stage/f'{mode}_trust{trust}_secant.npz',inputs=inputs,outputs=outputs,
                        original_gradient=original_gradient,corrected_gradient=calibrated,base=base)
                norm=np.maximum(np.linalg.norm(jac,axis=1),1e-12)
                gnorm=max(np.linalg.norm(g),1e-12)
                if args.joint_secants:
                    lp_matrix=np.vstack([-jac/norm[:,None],parent,-parent])
                    lp_rhs=np.r_[margins/norm,np.full(2*len(parent),trust)]
                    lp=linprog(g/gnorm,A_ub=lp_matrix,b_ub=lp_rhs,bounds=[(-.05,.05)]*len(g),
                        method='highs',options=dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
                    if not lp.success or g@lp.x>=-1e-5:
                        block=dict(context_arm=mode,trust=trust,secant=secant_audit,
                            lp_success=bool(lp.success),no_beneficial_joint_direction=True,message=lp.message)
                        blocks.append(block);write_json(stage/'minimum_norm_updates.json',blocks)
                        print('JOINT_SECANT_NO_STEP',block,flush=True)
                        continue
                    initial=lp.x
                matrix=np.vstack([-jac/norm[:,None],parent,-parent,g[None,:]/gnorm])
                rhs=np.r_[margins/norm,np.full(2*len(parent),trust),.9*(g@initial)/gnorm]
                nonzero=np.any(matrix!=0.,axis=1)
                assert np.all(rhs[~nonzero]>=0.)
                matrix,rhs=matrix[nonzero],rhs[nonzero]
                scale=max(float(initial@initial),1e-12)
                def budget(u):
                    if time.perf_counter()>deadline:raise TimeoutError('Minimum-norm QP budget')
                solved=minimize(lambda u:.5*(u@u)/scale,initial,jac=lambda u:u/scale,method='SLSQP',
                    bounds=[(-.05,.05)]*len(initial),constraints=[dict(type='ineq',
                    fun=lambda u:rhs-matrix@u,jac=lambda u:-matrix)],callback=budget,
                    options=dict(maxiter=160,ftol=1e-12))
                u=solved.x
                block=dict(context_arm=mode,trust=trust,success=bool(solved.success),message=solved.message,
                    old_norm=float(np.linalg.norm(initial)),new_norm=float(np.linalg.norm(u)),
                    minimum_optimizer_margin=float(np.min(rhs-matrix@u)),
                    predicted_loss_change=float(g@u),predicted_guard_min=float(np.min(margins+jac@u)))
                if args.joint_secants:block['secant']=secant_audit
                if np.min(rhs-matrix@u)>=-1e-8 and np.linalg.norm(u)<np.linalg.norm(initial)-1e-8:
                    tag=f'{mode}_trust{trust}';actor=copy.deepcopy(md);actor['coefficients']=u.reshape(-1,3)
                    torch.save(actor,stage/(tag+'.pt'))
                    unchanged=[p for p in paths if identity.unchanged(p,u)];actual=[p for p in paths if p not in unchanged]
                    records=pool.map(worker_evaluate,[(p,u,tag) for p in actual]);by={r['path']:r for r in records}
                    c=entry([by.get(r['path'],copy.deepcopy(r)) for r in selected],u,tag)
                    c.update(actual_rerun_paths=actual,exact_zero_support_paths=unchanged);history.append(c)
                    block.update(actual_loss_change=c['score']-zero['score'],actual_guard_min=min(c['augmented_measured_inequalities']))
                    write_json(folder/'results.json',dict(candidates=history,best=best()['tag']))
                blocks.append(block);write_json(stage/'minimum_norm_updates.json',blocks)
                print('MINIMUM_NORM_JOINT',block,flush=True)
            pool.close();pool.join();pool=None
        chosen=best();md=models[chosen['context_arm']];torch.save(md,stage/'initial_policy.pt')
        finish(stage,md,baseline,data,chosen,zero,history,blocks,choice,parent_file,started)
        receipt(stage,dict(status='completed_joint_secant_credit' if args.joint_secants else 'completed_minimum_norm_complete_credit',new_suffixes=0,new_stochastic_episodes=0,
            ppo_updates=0,new_policies=len(history)-1,selected=chosen['tag'],
            research_acceptable=(stage/'research_candidate.pt').exists(),wall_s=time.perf_counter()-started,
            adopted=False,goal_complete=False,source_model_sha256=sha(parent_file)))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_joint_secant_credit' if args.joint_secants else 'failed_minimum_norm_complete_credit',error=str(exc),
            new_policies=max(0,len(history)-1),wall_s=time.perf_counter()-started,adopted=False,goal_complete=False))
        raise


if __name__=='__main__':mp.freeze_support();main()
