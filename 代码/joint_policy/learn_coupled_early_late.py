"""Jointly release nested early/late learned layers with measured full-route credit."""
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
from learn_verified_trust_joint import propose,positive_sum
from learn_early_memory_joint_policy import worker_initialize,worker_evaluate,WORK
from joint_training_evaluator import build_evaluator
from consolidate_joint_teacher_policy import collect_data
from multi_corner_raw_metrics import augment
from joint_teacher_state_policy import features,INDICES
from learn_full_pool_joint_credit import aggregate
from study_io import BASE,register,write_json,receipt,sha
from search_complete_return import read


SOURCE=BASE/'canonical_gate_band_causal_prefix_joint_compensation_20260920'
TRUST=BASE/'canonical_gate_band_verified_trust_joint_learning_20260920'
NAME='canonical_gate_band_coupled_early_late_joint_learning_20260920'


def coupled_worker(job):
    path,u,tag=job
    # Training adapter only. The saved actor remains the ordinary nested model,
    # preserving both additions and their original floating-point arithmetic.
    WORK['policy'].base.coefficients=np.asarray(u[18:]).reshape(-1,3)
    return worker_evaluate((path,np.asarray(u[:18]),tag))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume-stage');parser.add_argument('--output-stage')
    args=parser.parse_args();previous=BASE/args.resume_stage if args.resume_stage else None
    torch.set_num_threads(1);started=time.perf_counter();deadline=started+1000.
    assert read(TRUST/'receipt.json')['status']=='completed_verified_trust_joint_learning'
    md=torch.load(SOURCE/'initial_policy.pt',map_location='cpu',weights_only=False)
    later=md['parent']['coefficients'].ravel().copy();assert len(later)==36
    data=read(SOURCE/'baseline.json');baseline=data['baseline']['records']
    support=read(SOURCE/'support_audit.json');paths=support['evaluation_paths']
    selected=[r for r in baseline if r['path'] in paths]
    original=copy.deepcopy(read(TRUST/'state_joint/results.json')['candidates'])
    for c in original:c['coefficients']+=later.tolist()
    history=copy.deepcopy(read(previous/'state_joint/results.json')['candidates']) if previous else copy.deepcopy(original)
    current=original[0]['records']
    if previous:
        assert read(previous/'receipt.json')['status']=='failed_coupled_early_late_joint_learning'
        assert 'budget' in read(previous/'receipt.json')['error'].lower()
    for r in selected+current:augment(r)
    response=read(SOURCE/'measured_prefix_credit.json');early_steps=np.array(response['steps'])
    steps=np.r_[early_steps,np.tile([.02,.04,.04],12)]
    pending_responses=72-len({c['tag'] for c in history if c['tag'].startswith('late_response_')})
    stage=register(args.output_stage or NAME,[__file__,TRUST/'receipt.json',TRUST/'state_joint/results.json',TRUST/'trust_updates.json',
        SOURCE/'initial_policy.pt',SOURCE/'baseline.json',SOURCE/'support_audit.json',
        SOURCE/'measured_prefix_credit.json',SOURCE/'initial_identity.json']+[BASE/f for f in
        ('learn_verified_trust_joint.py','observable_memory_joint_policy.py','joint_training_evaluator.py',
         'learn_early_memory_joint_policy.py','learn_shared_temporal_joint_policy.py')]+
        ([previous/f for f in ('receipt.json','state_joint/results.json','joint_parameter_identity.json','registration.json')]
         if previous else []),
        dict(wall_s=1020,workers=4,parameters=54,early_parameters=18,late_parameters=36,
            new_late_response_policies=pending_responses,maximum_joint_updates=16,maximum_new_policies=pending_responses+17,
            initial_radius=.5,radius_bounds=[.03125,4.],response_steps=[.02,.04,.04],ppo_updates=0,
            resume_stage=args.resume_stage,changed_kernel_zero_support_reuse=True),
        ('Resume only missing parameter vectors from the budget-limited collection. All '
         'completed prior physical vectors are cached exactly; retain the SAME initial '
         'anchor from the47-point original archive, not a newly selected response point. '
         'For late-only perturbations, reuse any initial-anchor route where every CHANGED '
         'kernel is exactly zero at all saved observations. Causal memories then stay zero '
         'for those changed kernels and deterministic induction proves identical complete '
         'actions/trajectory. All other routes run. This changes execution caching only; '
         'the parameter problem and all final guards are unchanged. ' if previous else '')+
        'Early causal support changes incoming negative motion, but fixed-late recovery and '
        'verified trust updates remain infeasible. Release BOTH learned layers jointly: '
        '18 early coefficients and36 late coefficients, each controlling steering/feed/preview. '
        'Reuse all47 prior full-policy points with the exact unchanged nested actor arithmetic '
        'and fixed late vector; do not merge additions or rerun those trajectories. Measure '
        'only72 missing symmetric LATE-coordinate responses around the current verified '
        'training point; fit early credit from nearby existing actual samples. Compare one '
        'early-only proposal against joint proposals at the same anchor and initial trust '
        'radius, then use the actual/predicted agreement rule for at most16 joint steps. '
        'Every changed vector executes all15 affected full routes including multicorner31. '
        'The protected all-zero two-layer actor is also an exact known reference, not a '
        'new experiment or an improved result. No angle/time/route switches or velocity '
        'tables, no interpolation-rule deployment.1020s,at most89 new policies. Final '
        'selection requires original protected-model shape/time/cruise and physical guards; '
        'training-only intermediate regressions cannot replace protected models. If jointly '
        'released parameters still cannot yield reliable accepted progress, inspect the '
        'measured response rank and nonlinear interaction, not more identical generations. '
        'Complete closed-loop parameter optimization,not PPO or generalization evidence.')
    folder=stage/'state_joint';folder.mkdir()
    write_json(stage/'baseline.json',data);write_json(stage/'support_audit.json',support)
    torch.save(md,stage/'initial_policy.pt')
    write_json(stage/'selection_hold.json',dict(reason='Joint temporal parameter learning; no automatic promotion.'))
    entry=build_evaluator(data,selected,stage,md['parent'],deadline)
    protected=entry(copy.deepcopy(selected),np.zeros(54),'protected_reference')
    assert protected['physical_pass'] and min(protected['augmented_measured_inequalities'])>=-1e-8
    normalizer=protected['training_score']
    def merit(c):return c['training_score']/normalizer+.1*positive_sum(c['augmented_learning_inequalities'])
    def acceptable(c):return c['physical_pass'] and min(c['augmented_measured_inequalities'])>=-1e-8 and c['training_score']<normalizer-1e-6
    anchor=min((c for c in original if c['physical_pass']),key=merit);initial_anchor=copy.deepcopy(anchor)
    start_vector=np.array(anchor['coefficients']);start_tag=anchor['tag']
    # Verify adapter layout and exact initial nested action history, without physics.
    checked=copy.deepcopy(md);checked['coefficients']=start_vector[:18].reshape(-1,3)
    checked['parent']['coefficients']=start_vector[18:].reshape(-1,3)
    if previous:
        identity=read(previous/'joint_parameter_identity.json')
        assert identity['original_nested_actor_decoded_actions_bitwise']
        bindings=read(previous/'registration.json')['source_sha256']
        source_key=SOURCE.joinpath('initial_policy.pt').relative_to(BASE.parent.parent).as_posix()
        assert bindings[source_key]==sha(SOURCE/'initial_policy.pt')
        identity.update(reused_from=str(previous/'joint_parameter_identity.json'),reused_policies=len(history))
    else:
        replay=collect_data(checked,checked['parent'],anchor['records'],include_latents=True,deadline=deadline)
        identity=dict(decisions=len(replay['decisions']),original_nested_actor_decoded_actions_bitwise=True,
            early_count=18,late_count=36,reused_policies=len(history),protected_all_zero_identity_by_composition=True)
    write_json(stage/'joint_parameter_identity.json',identity)
    assert not np.any(md['per_center_distance_weights'][:,len(INDICES):])
    assert not np.any(md['parent']['per_center_distance_weights'][:,len(INDICES):])
    active_by_path={}
    for r in initial_anchor['records']:
        with np.load(r['trace']) as z:observations=z['observations']
        f=features(md['parent'],observations,np.zeros((len(observations),3)))
        active_by_path[r['path']]=np.any(f!=0.,axis=(0,2))
    write_json(stage/'changed_kernel_support.json',dict(anchor=start_tag,
        active={str(p):v.tolist() for p,v in active_by_path.items()},
        proof='Only use when every early coefficient equals the anchor and every changed late kernel is identically zero.'))
    zero_values=np.array(anchor['augmented_learning_inequalities'])
    near=[c for c in original if c['physical_pass'] and c['tag']!=anchor['tag'] and
          np.max(abs((np.array(c['coefficients'][:18])-start_vector[:18])/early_steps))<=4.]
    x=np.array([(np.array(c['coefficients'][:18])-start_vector[:18])/early_steps for c in near])
    y=np.array([np.r_[np.array(c['augmented_learning_inequalities'])-zero_values,
                            c['training_score']-anchor['training_score']] for c in near])
    weights=1./np.maximum(np.linalg.norm(x,axis=1),.25)
    prior=np.column_stack([np.array(response['jacobian']).T,np.array(response['loss_gradient'])])
    early_model=np.linalg.solve((x*weights[:,None]).T@x+.05*np.eye(18),
                               (x*weights[:,None]).T@y+.05*prior)
    reused=len(history)
    if not any(c['tag']=='protected_reference' for c in history):history.append(protected)
    cache={tuple(c['coefficients']):c for c in history}
    blocks=[];new_count=0;pool=None
    def evaluate(u,tag):
        nonlocal new_count
        key=tuple(np.asarray(u).tolist())
        if key in cache:return cache[key]
        if time.perf_counter()>deadline-15:raise TimeoutError('Coupled early/late wall budget')
        assert np.max(abs(u))<=3.+1e-8
        unchanged=[]
        if np.array_equal(np.asarray(u[:18]),start_vector[:18]):
            changed=np.any((np.asarray(u[18:])-start_vector[18:]).reshape(3,4,3)!=0.,axis=(0,2))
            unchanged=[p for p in paths if not np.any(active_by_path[p]&changed)]
        actual=[p for p in paths if p not in unchanged]
        records=pool.map(coupled_worker,[(p,u,tag) for p in actual]);by={r['path']:r for r in records}
        by.update({r['path']:copy.deepcopy(r) for r in initial_anchor['records'] if r['path'] in unchanged})
        c=entry([by[r['path']] for r in selected],u,tag)
        c.update(actual_rerun_paths=actual,exact_zero_support_paths=unchanged)
        history.append(c);cache[key]=c;new_count+=1
        write_json(folder/'results.json',dict(candidates=history,best=None,training_best=anchor['tag']))
        if new_count%12==0:print('COUPLED_RESPONSE',new_count,tag,c['training_score'],c['physical_pass'],flush=True)
        return c
    try:
        pool=mp.get_context('spawn').Pool(4,worker_initialize,(md,current,folder,deadline))
        pairs=[]
        for j in range(18,54):
            delta=np.eye(54)[j]*steps
            pairs.append((evaluate(start_vector+delta,f'late_response_{j}_plus'),
                          evaluate(start_vector-delta,f'late_response_{j}_minus')))
        late_jac=np.array([(np.array(a['augmented_learning_inequalities'])-b['augmented_learning_inequalities'])/2 for a,b in pairs]).T
        late_gradient=np.array([(a['training_score']-b['training_score'])/2 for a,b in pairs])
        initial_jac=np.column_stack([early_model[:,:-1].T,late_jac])
        initial_gradient=np.r_[early_model[:,-1],late_gradient]
        write_json(stage/'measured_coupled_credit.json',dict(steps=steps.tolist(),jacobian=initial_jac.tolist(),
            loss_gradient=initial_gradient.tolist(),anchor=start_tag,early_reused_samples=len(near),
            late_response_pairs=[[a['tag'],b['tag']] for a,b in pairs]))
        radius=.5
        for iteration in range(16):
            if time.perf_counter()>deadline-15:break
            u=np.array(anchor['coefficients']);values=np.array(anchor['augmented_learning_inequalities'])
            near=[c for c in history if c['physical_pass'] and c['tag']!=anchor['tag'] and
                  np.max(abs((np.array(c['coefficients'])-u)/steps))<=4.]
            x=np.array([(np.array(c['coefficients'])-u)/steps for c in near])
            y=np.array([np.r_[np.array(c['augmented_learning_inequalities'])-values,
                                    c['training_score']-anchor['training_score']] for c in near])
            weight=1./np.maximum(np.linalg.norm(x,axis=1),.25)
            prior=np.column_stack([initial_jac.T,initial_gradient])
            model=np.linalg.solve((x*weight[:,None]).T@x+.05*np.eye(54),
                                  (x*weight[:,None]).T@y+.05*prior)
            jac,gradient=model[:,:-1].T,model[:,-1]
            if iteration==0:
                # Same local model, anchor, objective and radius as the joint arm.
                control=propose(u[:18],values,jac[:,:18],gradient[:18],early_steps,radius,normalizer)
                if control is not None:
                    delta,predicted=control;c=evaluate(np.r_[u[:18]+delta*early_steps,u[18:]],'early_only_control')
                    blocks.append(dict(arm='early_only_control',anchor=start_tag,predicted_drop=predicted,
                        actual_drop=merit(initial_anchor)-merit(c),measured=c['tag'],merit=merit(c)))
                    write_json(stage/'trust_updates.json',blocks)
            solution=propose(u,values,jac,gradient,steps,radius,normalizer)
            if solution is None:break
            delta,predicted=solution
            if predicted<=1e-9:break
            c=evaluate(u+delta*steps,f'coupled_trust_{iteration}')
            observed=merit(anchor)-merit(c);ratio=observed/predicted
            accepted=c['physical_pass'] and observed>0. and ratio>=.1
            block=dict(arm='joint',iteration=iteration,anchor=anchor['tag'],measured=c['tag'],radius=radius,
                predicted_drop=predicted,actual_drop=observed,ratio=ratio,accepted=accepted,
                training_score=c['training_score'],merit=merit(c),measured_guard_min=min(c['augmented_measured_inequalities']))
            blocks.append(block);print('COUPLED_TRUST',block,flush=True)
            if accepted:anchor=c
            if ratio<.25 or not c['physical_pass']:radius*=.25
            elif ratio>.75 and accepted:radius=min(4.,radius*2.)
            write_json(stage/'trust_updates.json',blocks)
            if acceptable(anchor) or radius<.03125:break
        pool.close();pool.join();pool=None
        passing=list(filter(acceptable,history));chosen=min(passing,key=lambda c:c['training_score']) if passing else anchor
        actor=copy.deepcopy(md);v=np.array(chosen['coefficients'])
        actor.update(coefficients=v[:18].reshape(-1,3),accepted=False,training_branch_only=True)
        actor['parent']['coefficients']=v[18:].reshape(-1,3)
        torch.save(actor,stage/'shadow_training_policy.pt')
        all_current=read(SOURCE/'seed.json')['records'];by={r['path']:r for r in chosen['records']}
        full=[copy.deepcopy(by.get(r['path'],r)) for r in all_current]
        write_json(stage/'shadow_training_result.json',dict(tag=chosen['tag'],records=full,summary=aggregate(full),
            training_score=chosen['training_score'],merit=merit(chosen),original_all_guard_pass=bool(passing),adopted=False))
        receipt(stage,dict(status='completed_coupled_early_late_joint_learning',new_policies=new_count,
            reused_policies=reused,accepted_updates=sum(b.get('accepted',False) for b in blocks),
            start_tag=start_tag,training_best=chosen['tag'],original_guard_passing_new_policies=len(passing),
            research_acceptable=False,pending_candidate_audit=bool(passing),ppo_updates=0,
            shadow_sha256=sha(stage/'shadow_training_policy.pt'),wall_s=time.perf_counter()-started,
            adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_coupled_early_late_joint_learning',error=str(exc),new_policies=new_count,
            wall_s=time.perf_counter()-started,adopted=False,goal_complete=False));raise


if __name__=='__main__':mp.freeze_support();main()
