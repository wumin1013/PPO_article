"""Measured model-agreement trust region for joint causal-prefix policy updates."""
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
from joint_training_evaluator import build_evaluator
from learn_early_memory_joint_policy import worker_initialize,worker_evaluate
from saved_memory_action_identity import SavedMemoryActionIdentity
from gate_band_state import GateBandRuntime
from learn_full_pool_joint_credit import aggregate
from multi_corner_raw_metrics import augment
from study_io import BASE,register,write_json,receipt,sha
from search_complete_return import read


SOURCE=BASE/'canonical_gate_band_causal_prefix_joint_compensation_20260920'
NAME='canonical_gate_band_verified_trust_joint_learning_20260920'


def positive_sum(values):return float(np.maximum(-np.asarray(values),0.).sum())


def propose(u,values,jac,gradient,steps,radius,normalizer):
    n=len(u);m=len(values)
    bounds=[(max(-radius,(-3-u[i])/steps[i]),min(radius,(3-u[i])/steps[i])) for i in range(n)]+[(0,None)]*m
    a=np.hstack([-jac,-np.eye(m)])
    # Same scalar merit in prediction and actual acceptance; no comparison of
    # event-margin slack against differently scaled measured duration slack.
    objective=np.r_[gradient/normalizer,np.full(m,.1)]
    result=linprog(objective,A_ub=a,b_ub=values,bounds=bounds,method='highs')
    if not result.success:return None
    delta=result.x[:n]
    predicted_drop=.1*(positive_sum(values)-positive_sum(values+jac@delta))-float(gradient@delta)/normalizer
    return delta,predicted_drop


def main():
    torch.set_num_threads(1);started=time.perf_counter();deadline=started+260.
    assert read(SOURCE/'receipt.json')['status']=='completed_causal_prefix_joint_compensation'
    data=read(SOURCE/'baseline.json');baseline=data['baseline']['records']
    support=read(SOURCE/'support_audit.json');paths=support['evaluation_paths']
    selected=[r for r in baseline if r['path'] in paths]
    md=torch.load(SOURCE/'initial_policy.pt',map_location='cpu',weights_only=False)
    md['learning_method']='actual_agreement_trust_region_joint_parameter_learning'
    original=read(SOURCE/'state_joint/results.json')['candidates'];zero=original[0]
    current_records=zero['records']
    for r in selected+current_records:augment(r)
    response=read(SOURCE/'measured_prefix_credit.json')
    initial_jac=np.array(response['jacobian']);initial_gradient=np.array(response['loss_gradient']);steps=np.array(response['steps'])
    stage=register(NAME,[__file__,SOURCE/'receipt.json',SOURCE/'initial_policy.pt',SOURCE/'baseline.json',
        SOURCE/'state_joint/results.json',SOURCE/'measured_prefix_credit.json',SOURCE/'optimization_blocks.json',
        SOURCE/'prefix_credit_audit.json',SOURCE/'affected_replay.npz']+[BASE/f for f in
        ('joint_training_evaluator.py','learn_early_memory_joint_policy.py','observable_memory_joint_policy.py')],
        dict(wall_s=280,workers=4,parameters=18,maximum_new_policies=16,initial_radius=.5,
            radius_bounds=[.03125,4.],shrink=.25,grow=2.,acceptance_ratio=.1,ppo_updates=0),
        'The previous measured-response model was extrapolated to64 response steps and its '
        'joint proposal worsened actual complete outcomes. Reuse all36 coordinate responses '
        'and all41 prior policies, with no new finite-difference collection. Change update '
        'acceptance: bounded joint trust region starts at0.5 response steps; prediction and '
        'actual improvement use the SAME full cost plus0.1 times positive LEARNING guard '
        'slack. Keep measured final guards separate. Require physical success,actual merit '
        'improvement and actual/predicted ratio>=0.1. Shrink by0.25 after poor agreement, '
        'grow by2 only for ratio>0.75; max4. Refit the local secant model from nearby actual '
        'returns, including rejected samples, without pretending a predicted improvement '
        'is realized. Start at the best existing physical merit point and preserve all raw '
        'failures. At most16 new policies/280s. If the smallest radius cannot make verified '
        'progress, stop this optimizer branch and examine joint early/late capacity. '
        'Physical1.25mm/20ms/1ms/vectorV/A/J/ordered completion/rest and protected-model '
        'adoption guards unchanged. Parameter optimization,not PPO,not a final comparison.')
    folder=stage/'state_joint';folder.mkdir()
    write_json(stage/'baseline.json',data);write_json(stage/'support_audit.json',support)
    torch.save(md,stage/'initial_policy.pt')
    write_json(stage/'selection_hold.json',dict(reason='Measured trust-region training; original final guards unchanged.'))
    with np.load(SOURCE/'affected_replay.npz') as z:replay={k:z[k].copy() for k in z.files}
    np.savez_compressed(stage/'affected_replay.npz',**replay)
    identity=SavedMemoryActionIdentity(md,replay,GateBandRuntime.decode_action)
    entry=build_evaluator(data,selected,stage,md['parent'],deadline)
    normalizer=entry(copy.deepcopy(selected),np.zeros(18),'protected_reference')['training_score']
    def merit(c):return c['training_score']/normalizer+.1*positive_sum(c['augmented_learning_inequalities'])
    def acceptable(c):return c['physical_pass'] and min(c['augmented_measured_inequalities'])>=-1e-8 and c['training_score']<normalizer-1e-6
    history=copy.deepcopy(original);cache={tuple(c['coefficients']):c for c in history}
    anchor=min((c for c in history if c['physical_pass']),key=merit)
    start_tag=anchor['tag'];radius=.5;blocks=[];pool=None;new_count=0
    def fit_local(anchor):
        center=np.array(anchor['coefficients']);values=np.array(anchor['augmented_learning_inequalities'])
        samples=sorted(history,key=lambda c:np.linalg.norm((np.array(c['coefficients'])-center)/steps))
        samples=[c for c in samples if c['physical_pass'] and c['tag']!=anchor['tag']
                 and np.max(abs((np.array(c['coefficients'])-center)/steps))<=4.][:48]
        if not samples:return initial_jac.copy(),initial_gradient.copy(),0
        x=np.array([(np.array(c['coefficients'])-center)/steps for c in samples])
        y=np.array([np.r_[np.array(c['augmented_learning_inequalities'])-values,
                                  c['training_score']-anchor['training_score']] for c in samples])
        weight=1./np.maximum(np.linalg.norm(x,axis=1),.25)
        prior=np.column_stack([initial_jac.T,initial_gradient])
        regularizer=.05
        model=np.linalg.solve((x*weight[:,None]).T@x+regularizer*np.eye(18),
            (x*weight[:,None]).T@y+regularizer*prior)
        return model[:,:-1].T,model[:,-1],len(samples)
    try:
        pool=mp.get_context('spawn').Pool(4,worker_initialize,(md,current_records,folder,deadline))
        for iteration in range(16):
            if time.perf_counter()>deadline-15:break
            jac,gradient,count=fit_local(anchor);u=np.array(anchor['coefficients']);values=np.array(anchor['augmented_learning_inequalities'])
            solution=propose(u,values,jac,gradient,steps,radius,normalizer)
            if solution is None:break
            delta,predicted=solution
            if predicted<=1e-9:break
            candidate_u=u+delta*steps;key=tuple(candidate_u.tolist())
            if key in cache:c=cache[key]
            else:
                tag=f'trust_{iteration}';unchanged=[p for p in paths if identity.unchanged(p,candidate_u)]
                actual=[p for p in paths if p not in unchanged]
                records=pool.map(worker_evaluate,[(p,candidate_u,tag) for p in actual]);by={r['path']:r for r in records}
                c=entry([by.get(r['path'],copy.deepcopy(r)) for r in current_records],candidate_u,tag)
                c.update(actual_rerun_paths=actual,exact_zero_support_paths=unchanged)
                history.append(c);cache[key]=c;new_count+=1
            observed=merit(anchor)-merit(c);ratio=observed/predicted
            accepted=c['physical_pass'] and observed>0. and ratio>=.1
            block=dict(iteration=iteration,anchor=anchor['tag'],measured=c['tag'],radius=radius,
                model_samples=count,predicted_drop=predicted,actual_drop=observed,ratio=ratio,
                accepted=accepted,merit=merit(c),measured_guard_min=min(c['augmented_measured_inequalities']),
                training_score=c['training_score'])
            blocks.append(block);print('VERIFIED_TRUST',block,flush=True)
            if accepted:anchor=c
            if ratio<.25 or not c['physical_pass']:radius*=.25
            elif ratio>.75 and accepted:radius=min(4.,radius*2.)
            write_json(folder/'results.json',dict(candidates=history,best=None,training_best=anchor['tag']))
            write_json(stage/'trust_updates.json',blocks)
            if acceptable(anchor) or radius<.03125:break
        pool.close();pool.join();pool=None
        passing=list(filter(acceptable,history));chosen=min(passing,key=lambda c:c['training_score']) if passing else anchor
        actor=copy.deepcopy(md);actor.update(coefficients=np.array(chosen['coefficients']).reshape(-1,3),
            accepted=False,training_branch_only=True)
        torch.save(actor,stage/'shadow_training_policy.pt')
        current=read(SOURCE/'seed.json')['records'];by={r['path']:r for r in chosen['records']}
        full=[copy.deepcopy(by.get(r['path'],r)) for r in current]
        write_json(stage/'shadow_training_result.json',dict(tag=chosen['tag'],records=full,summary=aggregate(full),
            training_score=chosen['training_score'],merit=merit(chosen),original_all_guard_pass=bool(passing),adopted=False))
        receipt(stage,dict(status='completed_verified_trust_joint_learning',new_policies=new_count,
            reused_policies=len(original),accepted_updates=sum(b['accepted'] for b in blocks),
            start_tag=start_tag,training_best=chosen['tag'],original_guard_passing_new_policies=len(passing),
            research_acceptable=False,pending_candidate_audit=bool(passing),ppo_updates=0,
            shadow_sha256=sha(stage/'shadow_training_policy.pt'),wall_s=time.perf_counter()-started,
            adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_verified_trust_joint_learning',error=str(exc),new_policies=new_count,
            wall_s=time.perf_counter()-started,adopted=False,goal_complete=False));raise


if __name__=='__main__':mp.freeze_support();main()
