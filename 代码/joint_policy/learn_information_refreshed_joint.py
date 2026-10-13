"""Remeasure current joint response directions before taking further actor steps."""
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
from learn_coupled_early_late import coupled_worker,worker_initialize
from learn_verified_trust_joint import positive_sum
from joint_response_models import fit_local_response
from joint_training_evaluator import build_evaluator
from multi_corner_raw_metrics import augment
from learn_full_pool_joint_credit import aggregate
from study_io import BASE,register,write_json,receipt,sha
from search_complete_return import read


SOURCE=BASE/'canonical_gate_band_coupled_early_late_joint_completion_20260920'
NAME='canonical_gate_band_information_refreshed_joint_learning_20260920'


def propose_subspace(anchor,transform,model,radius,normalizer,base):
    a=np.array(anchor['subspace_coordinates']);u=base+transform@a
    values=np.array(anchor['augmented_learning_inequalities']);k=len(a);m=len(values)
    jac,gradient=model[:,:-1].T,model[:,-1]
    matrix=np.vstack([np.hstack([-jac,-np.eye(m)]),
        np.hstack([transform,np.zeros((len(u),m))]),
        np.hstack([-transform,np.zeros((len(u),m))])])
    bounds=[(max(-radius,-2-a[i]),min(radius,2-a[i])) for i in range(k)]+[(0,None)]*m
    solved=linprog(np.r_[gradient/normalizer,np.full(m,.1)],A_ub=matrix,
        b_ub=np.r_[values,3-u,3+u],bounds=bounds,method='highs')
    if not solved.success:return None
    delta=solved.x[:k]
    predicted=.1*(positive_sum(values)-positive_sum(values+jac@delta))-gradient@delta/normalizer
    return delta,float(predicted)


def main():
    torch.set_num_threads(1);started=time.perf_counter();deadline=started+460.
    source_receipt=read(SOURCE/'receipt.json')
    assert sha(SOURCE/'shadow_training_policy.pt')==source_receipt['shadow_sha256']
    assert read(SOURCE/'shadow_raw_identity_audit.json')['actual_actor_and_saved_decoded_actions_bitwise']
    old=read(SOURCE/'state_joint/results.json')['candidates'];by={c['tag']:c for c in old}
    seed=by[source_receipt['training_best']];base=np.array(seed['coefficients'])
    data=read(SOURCE/'baseline.json');baseline=data['baseline']['records']
    support=read(SOURCE/'support_audit.json');paths=support['evaluation_paths']
    selected=[r for r in baseline if r['path'] in paths]
    for r in selected+seed['records']:augment(r)
    credit=read(SOURCE/'measured_coupled_credit.json');steps=np.array(credit['steps'])
    prior=np.column_stack([np.array(credit['jacobian']).T,credit['loss_gradient']])
    stale,information=fit_local_response(old,seed,steps,prior)
    normalizer=by['protected_reference']['training_score']
    violated=np.array(seed['augmented_measured_inequalities']) < -1e-8
    rows=np.vstack([stale[:,-1]/normalizer,stale[:,:-1].T[violated]])
    norms=np.linalg.norm(rows,axis=1);rows=rows[norms>1e-10]/norms[norms>1e-10,None]
    _,singular,vh=np.linalg.svd(rows,full_matrices=False)
    k=min(12,int(np.count_nonzero(singular>1e-10)));assert k==12
    basis=vh[:k].T;transform=.5*steps[:,None]*basis
    stale_projected=.5*basis.T@stale
    stage=register(NAME,[__file__,SOURCE/'receipt.json',SOURCE/'shadow_training_policy.pt',
        SOURCE/'shadow_raw_identity_audit.json',SOURCE/'baseline.json',SOURCE/'state_joint/results.json',
        SOURCE/'measured_coupled_credit.json',SOURCE/'local_response_information_audit.json']+
        [BASE/f for f in ('joint_response_models.py','learn_coupled_early_late.py',
            'joint_training_evaluator.py','learn_verified_trust_joint.py')],
        dict(wall_s=480,workers=4,actor_parameters=54,measured_subspace_dimensions=k,
            new_directional_responses=24,maximum_new_proposals=8,maximum_new_policies=32,
            normalized_probe_scale=.5,initial_radius=.5,maximum_radius=1.,
            maximum_coordinate_distance_from_measurement_center=2.,ppo_updates=0),
        'Exact replay found the prior joint fit lost rank54 to2 then1 after accepted '
        'movement, leaving about52-53 effective dimensions dependent on stale response '
        'priors. Do not extend that optimizer blindly. At the audited latest actual actor, '
        'select12 shared parameter directions spanning normalized gradients of the current '
        'violated protections and full-return objective. Remeasure24 symmetric COMPLETE '
        'policy returns at this new center. Compare stale and freshly measured responses '
        'on this SAME subspace,objective,anchor and trust radius. Both early/late layers and '
        'all three heads participate; this is a12-dimensional parameter SEARCH subspace '
        'inside the54-parameter actor,not new runtime rules. Continue at most6 additional '
        'fresh-response proposals, only with full local design rank and no coordinate '
        'farther than2 measured units from the new center. Actual/predicted agreement '
        'controls acceptance and shrink/growth. Reuse the exact protected reference and '
        'current anchor,rerun all15 affected routes including multicorner31 for each '
        'new dense direction.480s,at most32 new policies. All original protected-model '
        'shape,time,cruise and physical adoption guards remain unchanged. If the refreshed '
        'directions still fail actual prediction or plateau, change the measured subspace '
        'or representation using raw responses rather than expanding the radius or budget. '
        'Closed-loop parameter learning,not PPO. No independent-seed or final superiority claim.')
    folder=stage/'state_joint';folder.mkdir()
    md=torch.load(SOURCE/'shadow_training_policy.pt',map_location='cpu',weights_only=False)
    torch.save(md,stage/'initial_policy.pt')
    write_json(stage/'baseline.json',data);write_json(stage/'support_audit.json',support)
    write_json(stage/'selection_hold.json',dict(reason='Information-refresh mechanism comparison; final guards unchanged.'))
    write_json(stage/'subspace.json',dict(base=base.tolist(),transform=transform.tolist(),
        stale_information=information,stale_projected_model=stale_projected.tolist(),
        singular_values=singular.tolist(),heads_squared_norm=[float(np.sum(transform[h::3]**2)) for h in range(3)],
        early_squared_norm=float(np.sum(transform[:18]**2)),late_squared_norm=float(np.sum(transform[18:]**2))))
    entry=build_evaluator(data,selected,stage,md['parent'],deadline)
    zero=entry(copy.deepcopy(seed['records']),base,'reused_current_actor')
    assert zero['training_score']==seed['training_score']
    assert np.array_equal(zero['augmented_measured_inequalities'],seed['augmented_measured_inequalities'])
    zero.update(subspace_coordinates=np.zeros(k).tolist(),actual_rerun_paths=[],exact_zero_support_paths=paths)
    history=[zero];cache={tuple(np.zeros(k)):zero};new_count=0;pool=None;blocks=[]
    def merit(c):return c['training_score']/normalizer+.1*positive_sum(c['augmented_learning_inequalities'])
    def acceptable(c):return c['physical_pass'] and min(c['augmented_measured_inequalities'])>=-1e-8 and c['training_score']<normalizer-1e-6
    def evaluate(a,tag):
        nonlocal new_count
        key=tuple(np.asarray(a).tolist())
        if key in cache:return cache[key]
        if time.perf_counter()>deadline-15:raise TimeoutError('Information refresh wall budget')
        u=base+transform@a;assert np.max(abs(u))<=3.+1e-8
        records=pool.map(coupled_worker,[(p,u,tag) for p in paths]);by_path={r['path']:r for r in records}
        c=entry([by_path[r['path']] for r in selected],u,tag)
        c['subspace_coordinates']=np.asarray(a).tolist();history.append(c);cache[key]=c;new_count+=1
        write_json(folder/'results.json',dict(candidates=history,best=None))
        if new_count%6==0:print('INFORMATION_REFRESH_RESPONSE',new_count,tag,c['training_score'],c['physical_pass'],flush=True)
        return c
    try:
        pool=mp.get_context('spawn').Pool(4,worker_initialize,(md,seed['records'],folder,deadline))
        pairs=[]
        for j in range(k):
            d=np.eye(k)[j]
            pairs.append((evaluate(d,f'response_{j}_plus'),evaluate(-d,f'response_{j}_minus')))
        fresh=np.array([np.r_[(np.array(a['augmented_learning_inequalities'])-b['augmented_learning_inequalities'])/2,
                                   (a['training_score']-b['training_score'])/2] for a,b in pairs])
        np.savez_compressed(stage/'current_joint_response.npz',fresh=fresh,stale=stale_projected,transform=transform,base=base)
        anchor=zero;radius=.5
        for arm,model in (('stale',stale_projected),('fresh',fresh)):
            solution=propose_subspace(zero,transform,model,.5,normalizer,base)
            if solution is None:continue
            delta,predicted=solution;c=evaluate(delta,arm+'_proposal')
            actual=merit(zero)-merit(c);ratio=actual/predicted if predicted>1e-12 else None
            accepted=c['physical_pass'] and actual>0. and predicted>0. and ratio>=.1
            block=dict(arm=arm,measured=c['tag'],predicted_drop=predicted,actual_drop=actual,ratio=ratio,
                accepted=accepted,merit=merit(c),measured_guard_min=min(c['augmented_measured_inequalities']))
            blocks.append(block);print('INFORMATION_REFRESH_CONTROL',block,flush=True)
            if arm=='fresh':
                if accepted:anchor=c
                if ratio is None or ratio<.25:radius*=.25
                elif ratio>.75 and accepted:radius=1.
            write_json(stage/'learning_updates.json',blocks)
        for iteration in range(6):
            if time.perf_counter()>deadline-15 or radius<.03125:break
            a=np.array(anchor['subspace_coordinates']);values=np.array(anchor['augmented_learning_inequalities'])
            near=[c for c in history if c['physical_pass'] and c['tag']!=anchor['tag'] and
                np.max(abs(np.array(c['subspace_coordinates'])-a))<=4.]
            x=np.array([np.array(c['subspace_coordinates'])-a for c in near])
            rank=int(np.linalg.matrix_rank(x,tol=1e-10))
            if rank<k:
                blocks.append(dict(arm='information_gate',rank=rank,required=k,stop=True));break
            y=np.array([np.r_[np.array(c['augmented_learning_inequalities'])-values,
                                    c['training_score']-anchor['training_score']] for c in near])
            weight=1./np.maximum(np.linalg.norm(x,axis=1),.25)
            model=np.linalg.solve((x*weight[:,None]).T@x+.01*np.eye(k),
                                  (x*weight[:,None]).T@y+.01*fresh)
            solution=propose_subspace(anchor,transform,model,radius,normalizer,base)
            if solution is None:break
            delta,predicted=solution
            if predicted<=1e-9:break
            c=evaluate(a+delta,f'refreshed_trust_{iteration}')
            actual=merit(anchor)-merit(c);ratio=actual/predicted
            accepted=c['physical_pass'] and actual>0. and ratio>=.1
            block=dict(arm='refreshed_joint',iteration=iteration,anchor=anchor['tag'],measured=c['tag'],rank=rank,
                radius=radius,predicted_drop=predicted,actual_drop=actual,ratio=ratio,accepted=accepted,
                merit=merit(c),measured_guard_min=min(c['augmented_measured_inequalities']))
            blocks.append(block);print('INFORMATION_REFRESH_UPDATE',block,flush=True)
            if accepted:anchor=c
            if ratio<.25 or not c['physical_pass']:radius*=.25
            elif ratio>.75 and accepted:radius=min(1.,radius*2.)
            write_json(stage/'learning_updates.json',blocks)
            if acceptable(anchor):break
        pool.close();pool.join();pool=None
        passing=list(filter(acceptable,history));chosen=min(passing,key=lambda c:c['training_score']) if passing else anchor
        actor=copy.deepcopy(md);u=np.array(chosen['coefficients'])
        actor.update(coefficients=u[:18].reshape(-1,3),accepted=False,training_branch_only=True)
        actor['parent']['coefficients']=u[18:].reshape(-1,3)
        torch.save(actor,stage/'shadow_training_policy.pt')
        previous_full=read(SOURCE/'shadow_training_result.json')['records'];by_path={r['path']:r for r in chosen['records']}
        full=[copy.deepcopy(by_path.get(r['path'],r)) for r in previous_full]
        write_json(stage/'shadow_training_result.json',dict(tag=chosen['tag'],records=full,summary=aggregate(full),
            training_score=chosen['training_score'],merit=merit(chosen),original_all_guard_pass=bool(passing),adopted=False))
        receipt(stage,dict(status='completed_information_refreshed_joint_learning',new_policies=new_count,
            training_best=chosen['tag'],accepted_updates=sum(b.get('accepted',False) for b in blocks),
            original_guard_passing_new_policies=len(passing),pending_candidate_audit=bool(passing),research_acceptable=False,
            ppo_updates=0,shadow_sha256=sha(stage/'shadow_training_policy.pt'),wall_s=time.perf_counter()-started,
            adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_information_refreshed_joint_learning',error=str(exc),new_policies=new_count,
            wall_s=time.perf_counter()-started,adopted=False,goal_complete=False));raise


if __name__=='__main__':mp.freeze_support();main()
