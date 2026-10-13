"""Whole-policy, temporally coherent joint exploration in a separate training branch."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import copy
import time
import multiprocessing as mp
import numpy as np
import torch
from head_counterfactual_rollouts import cost
from train_factorial_joint_ppo import worker_initialize,worker_evaluate,metric_functions,guards
from train_factorial_joint_ppo import shape_event_learning_guards,intrinsic_guards,finish,parameters_in_bounds
from learn_full_pool_joint_credit import aggregate
from study_io import BASE,register,write_json,receipt,sha
from search_complete_return import read


def main():
    torch.set_num_threads(1);started=time.perf_counter();deadline=started+340.
    source=BASE/'canonical_gate_band_factorial_joint_ppo_20260920'
    cv=BASE/'canonical_gate_band_reference_control_variate_ppo_20260920'
    projected=BASE/'canonical_gate_band_guarded_control_variate_proposal_20260920'
    det=BASE/'canonical_gate_band_deterministic_full_support_credit_20260920'
    parent_file=BASE/'canonical_gate_band_reference_supported_joint_learning_20260920/research_candidate.pt'
    assert sha(parent_file)=='ded2a93f85f3503a8c09bf59fd79e48c8f2cd187874f07789c53b8863c936229'
    assert not read(projected/'receipt.json')['research_acceptable']
    md=torch.load(source/'initial_policy.pt',map_location='cpu',weights_only=False)
    md['learning_method']='complete_closed_loop_covariance_parameter_exploration'
    data=read(source/'baseline.json');baseline=data['baseline']['records']
    support=read(source/'support_audit.json');choice=support['selected'];paths=choice['paths']
    selected=[r for r in baseline if r['path'] in paths]
    with np.load(det/'geometry_complete_gradient.npz') as z:
        ids=np.concatenate([np.arange(18)+54*i for i in range(3)])
        parent=z['parent_matrix'][:,ids].copy()
    stage=register('canonical_gate_band_coherent_parameter_training_20260920',
        [__file__,parent_file,source/'initial_policy.pt',source/'baseline.json',source/'support_audit.json',
         det/'geometry_complete_gradient.npz']+[s/f for s in (source,cv,projected)
         for f in ('receipt.json','state_joint/results.json')]+[BASE/f for f in (
         'train_factorial_joint_ppo.py','observable_memory_joint_policy.py','head_counterfactual_rollouts.py',
         'learn_early_memory_joint_policy.py','learn_direct_observable_joint_policy.py')],
        dict(wall_s=360,workers=4,generations=4,population=12,elite_count=4,parameters=54,
            random_seed=202609202,proposal_action_radius=.25,global_action_bound=1.,
            coefficient_bound=.05,maximum_new_policies=48,ppo_updates=0,
            final_original_guards_unchanged=True,independent_training_branch=True),
        'The corrected PPO proposal lowers total loss but violates55 per-route channels; '
        'projection removes almost all gain and still misses original actual guards. '
        'Test whole-policy feedback credit and coherent joint exploration, not a further '
        'single-action derivative or sparse PPO update. All54 shared observation/history '
        'weights are perturbed together and held for each complete feedback rollout. '
        'Use six antithetic parameter pairs per generation, four generations, update '
        'mean and full covariance from four elites including the incumbent. Normalize '
        'perturbations by saved parent action response, keep fixed coefficient and '
        'global action bounds, and select by ACTUAL complete-route cost. No surrogate '
        'claim, no route/time switches, no angle-speed table. Preserve raw1ms data and '
        'all original203 guards for final adoption. This changes exploration, optimizer '
        'and TRAINING selection jointly; not an isolated PPO superiority comparison. '
        'Intermediate training states may regress soft metrics; hard vectorV/A/J,1.25mm '
        'corridor,ordered completion and20ms rest remain mandatory. Add explicit first-turn '
        'and entry penalties to the existing full-shape/time/cruise objective. Keep '
        'the protected models unchanged, label any useful intermediate as TRAINING ONLY. '
        'Reuse all six known54-parameter policies and parent; do not rerun them. '
        'No sealed tests or final adoption without original guards.360s, at most48 policies. '
        'If full-policy exploration does not lower actual objective, inspect achieved '
        'action directions and stop this search configuration rather than add generations.')
    folder=stage/'state_joint';folder.mkdir()
    write_json(stage/'baseline.json',data);write_json(stage/'support_audit.json',support)
    torch.save(md,stage/'initial_policy.pt')
    write_json(stage/'selection_hold.json',dict(reason='Independent training branch; no automatic research promotion.',
        final_original_guards_unchanged=True,physical_hard_limits_unchanged=True))
    _,old_guards=metric_functions(data,selected,stage,md['parent'],deadline)
    retained={r['path']:r['baseline_shape'] for r in data['retained'] if 'baseline_shape' in r}
    def entry(records,u,tag,rerun):
        first=np.concatenate([old_guards(records)]+[guards(r,b) for r,b in zip(records,selected)])
        extra=intrinsic_guards(records,selected)
        safe=all(r['success'] and r['audit']['kinematic_pass'] and r['audit']['rest_pass']
                 and r['corridor']['margin_lower_mm']>=0. for r in records)
        score=sum(cost(r,b,retained[r['path']]) for r,b in zip(records,selected))
        entry_penalty=sum((r['shape']['in_edge_outward_mm']/.002)**2+
            (0. if r['target_motion']['first_turn_inside'] else 1e5) for r in records) if safe else 1e9
        return dict(tag=tag,coefficients=np.asarray(u).tolist(),records=records,score=score,
            training_score=score+entry_penalty,physical_pass=bool(safe),
            measured_inequalities=first.tolist(),augmented_measured_inequalities=np.r_[first,extra].tolist(),
            augmented_learning_inequalities=np.r_[shape_event_learning_guards(records,selected,first),extra].tolist(),
            actual_rerun_paths=paths if rerun else [],exact_zero_support_paths=[] if rerun else paths)
    zero=entry(copy.deepcopy(selected),np.zeros(54),'reused_parent',False);history=[zero];cache={tuple(zero['coefficients']):zero}
    for s in (source,cv,projected):
        for c in read(s/'state_joint/results.json')['candidates'][1:]:
            key=tuple(c['coefficients'])
            if key in cache:continue
            r=entry(copy.deepcopy(c['records']),c['coefficients'],'reused_'+c['tag'],False)
            r['exact_zero_support_paths']=[];r['reused_whole_policy_result']=str(s/'state_joint/results.json')
            history.append(r);cache[key]=r
    reused=len(history);md['reused_candidates']=reused
    def eligible(c):
        return c['physical_pass'] and parameters_in_bounds(c['coefficients'],-.05,.05)
    def incumbent():return min(filter(eligible,history),key=lambda c:c['training_score'])
    def final_best():return min((c for c in history if eligible(c) and min(c['augmented_measured_inequalities'])>=-1e-8),key=lambda c:c['score'])
    rng=np.random.default_rng(202609202);mean=np.array(incumbent()['coefficients']);covariance=np.eye(54)
    pool=None;blocks=[]
    try:
        pool=mp.get_context('spawn').Pool(4,worker_initialize,(md,selected,folder,deadline))
        for generation in range(4):
            if time.perf_counter()>deadline-15:break
            eig,vectors=np.linalg.eigh(covariance)
            factor=vectors@np.diag(np.sqrt(np.maximum(eig,1e-10)))
            center=mean.copy();batch=[incumbent()]
            for pair in range(6):
                direction=factor@rng.normal(size=54)
                direction*=.25/max(float(np.max(abs(parent@direction))),1e-12)
                for sign in (-1.,1.):
                    u=np.clip(center+sign*direction,-.05,.05)
                    excursion=float(np.max(abs(parent@u)))
                    if excursion>1.:u/=excursion
                    key=tuple(u.tolist())
                    if key in cache:batch.append(cache[key]);continue
                    if time.perf_counter()>deadline-10:raise TimeoutError('Coherent parameter training budget')
                    tag=f'g{generation}_p{pair}_s{int(sign)}'
                    measured=pool.map(worker_evaluate,[(p,u,tag) for p in paths]);by={r['path']:r for r in measured}
                    c=entry([by[r['path']] for r in selected],u,tag,True)
                    history.append(c);cache[key]=c;batch.append(c)
                    write_json(folder/'results.json',dict(candidates=history,best=final_best()['tag'],training_best=incumbent()['tag']))
            elites=sorted(filter(eligible,batch),key=lambda c:c['training_score'])[:4]
            assert elites
            values=np.array([c['coefficients'] for c in elites])
            weights=np.arange(len(elites),0,-1,dtype=float);weights/=weights.sum()
            elite_mean=weights@values;mean=.25*center+.75*elite_mean
            centered=values-elite_mean
            empirical=(centered*weights[:,None]).T@centered
            empirical*=54/max(float(np.trace(empirical)),1e-30)
            covariance=.7*covariance+.3*empirical
            block=dict(generation=generation,new_policies=len(history)-reused,best=incumbent()['tag'],
                training_score=incumbent()['training_score'],original_guard_min=min(incumbent()['augmented_measured_inequalities']),
                physical_passes=sum(c['physical_pass'] for c in batch),batch_entries=len(batch))
            blocks.append(block);write_json(stage/'training_generations.json',blocks)
            print('COHERENT_PARAMETER_GENERATION',block,flush=True)
        pool.close();pool.join();pool=None
    except TimeoutError:
        if pool is not None:pool.terminate();pool.join();pool=None
        write_json(stage/'budget_stop.json',dict(new_policies=len(history)-reused,wall_s=time.perf_counter()-started))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_coherent_parameter_training',error=str(exc),
            new_policies=len(history)-reused,wall_s=time.perf_counter()-started,adopted=False,goal_complete=False));raise
    chosen=incumbent();u=np.array(chosen['coefficients'])
    shadow=chosen['training_score']<zero['training_score']-.001
    if shadow:
        actor=copy.deepcopy(md);actor.update(coefficients=u.reshape(-1,3),accepted=False,training_branch_only=True)
        torch.save(actor,stage/'shadow_training_policy.pt')
        by={r['path']:r for r in chosen['records']}
        full=[copy.deepcopy(by.get(r['path'],r)) for r in baseline]
        write_json(stage/'shadow_training_result.json',dict(tag=chosen['tag'],records=full,summary=aggregate(full),
            original_all_guard_pass=min(chosen['augmented_measured_inequalities'])>=-1e-8,
            adopted=False,protected_candidate_replaced=False,goal_complete=False))
    finish(stage,md,baseline,data,final_best(),zero,history,blocks,choice,parent_file,started)
    write_json(folder/'results.json',dict(candidates=history,best=final_best()['tag'],training_best=chosen['tag']))
    receipt(stage,dict(status='completed_coherent_parameter_training',new_policies=len(history)-reused,
        reused_policies=reused,training_best=chosen['tag'],training_score=chosen['training_score'],
        parent_training_score=zero['training_score'],shadow_saved=shadow,
        shadow_sha256=sha(stage/'shadow_training_policy.pt') if shadow else None,
        final_guard_candidate=final_best()['tag'],research_acceptable=(stage/'research_candidate.pt').exists(),
        ppo_updates=0,source_model_sha256=sha(parent_file),wall_s=time.perf_counter()-started,adopted=False,goal_complete=False))


if __name__=='__main__':
    mp.freeze_support();main()
