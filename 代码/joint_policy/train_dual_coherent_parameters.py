"""Matched measured-return parameter learning with fixed versus adaptive constraint prices."""
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
    torch.set_num_threads(1);started=time.perf_counter();deadline=started+280.
    source=BASE/'canonical_gate_band_coherent_parameter_training_20260920'
    det=BASE/'canonical_gate_band_deterministic_full_support_credit_20260920'
    parent_file=BASE/'canonical_gate_band_reference_supported_joint_learning_20260920/research_candidate.pt'
    assert sha(parent_file)=='ded2a93f85f3503a8c09bf59fd79e48c8f2cd187874f07789c53b8863c936229'
    old_receipt=read(source/'receipt.json');assert old_receipt['shadow_saved'] and not old_receipt['research_acceptable']
    assert sha(source/'shadow_training_policy.pt')==old_receipt['shadow_sha256']
    md=torch.load(source/'initial_policy.pt',map_location='cpu',weights_only=False)
    md['learning_method']='measured_full_policy_fixed_vs_adaptive_dual_cost'
    data=read(source/'baseline.json');baseline=data['baseline']['records']
    support=read(source/'support_audit.json');choice=support['selected'];paths=choice['paths']
    selected=[r for r in baseline if r['path'] in paths]
    original_history=read(source/'state_joint/results.json')['candidates']
    seed=next(c for c in original_history if c['tag']==old_receipt['training_best'])
    n=len(seed['coefficients']);m=len(seed['augmented_measured_inequalities'])
    with np.load(det/'geometry_complete_gradient.npz') as z:
        ids=np.concatenate([np.arange(18)+54*i for i in range(3)]);parent=z['parent_matrix'][:,ids].copy()
    stage=register('canonical_gate_band_dual_coherent_parameter_training_20260920',
        [__file__,parent_file,source/'receipt.json',source/'initial_policy.pt',source/'shadow_training_policy.pt',
         source/'baseline.json',source/'support_audit.json',source/'state_joint/results.json',
         det/'geometry_complete_gradient.npz']+[BASE/f for f in ('train_factorial_joint_ppo.py',
         'head_counterfactual_rollouts.py','observable_memory_joint_policy.py','learn_early_memory_joint_policy.py',
         'learn_direct_observable_joint_policy.py')],
        dict(wall_s=300,workers=4,arms=['fixed_penalty','adaptive_dual'],generations_per_arm=2,
            population=12,maximum_unique_new_policies=36,shared_first_generation=12,
            initial_rho=.1,dual_step=.1,parameters=n,normalized_constraint_channels=m,
            action_perturbation_radius=.25,global_action_bound=1.,coefficient_bound=.05,
            random_seed=202609203,ppo_updates=0),
        'The independent training branch improves reversal,exit and negative lateral '
        'totals but loses3ms cruise and increases curve energy. Reuse its entire55-policy '
        'archive and selected training seed. Compare fixed squared-violation penalty '
        'against adaptive nonnegative per-channel Lagrange multipliers, using ALL203 '
        'original measured guard margins. Same initial actor,covariance,objective scaling, '
        'two generations,twelve proposals and paired random directions. First generation '
        'is identical and physically executed ONCE. After its measured results, adaptive '
        'arm updates lambda=max(0,lambda+0.1*(-margin)); rerank elites before its next '
        'mean/covariance update. Fixed arm keeps lambda=0. No outcome-conditioned '
        'omission of constraints. Both require unchanged physical hard constraints for '
        'training eligibility. Training merit is original full cost divided by frozen '
        'parent cost plus lambda dot positive violations plus0.05 squared violations. '
        'Final adoption still requires every original guard, and protected models remain '
        'untouched. This is measured-return parameter optimization, NOT PPO. Keep raw '
        'trajectories and both final arms, including failures; no independent-seed or '
        'generalization claim.300s, at most36 unique new policies. If adaptive weighting '
        'merely returns the parent or worsens the tradeoff, change policy representation '
        'or action structure based on actual traces rather than extend this search.')
    folder=stage/'state_joint';folder.mkdir();torch.save(md,stage/'initial_policy.pt')
    write_json(stage/'baseline.json',data);write_json(stage/'support_audit.json',support)
    write_json(stage/'selection_hold.json',dict(reason='Matched training mechanism comparison; no automatic promotion.',
        final_original_guards_unchanged=True))
    _,old_guards=metric_functions(data,selected,stage,md['parent'],deadline)
    retained={r['path']:r['baseline_shape'] for r in data['retained'] if 'baseline_shape' in r}
    def entry(records,u,tag):
        first=np.concatenate([old_guards(records)]+[guards(r,b) for r,b in zip(records,selected)])
        extra=intrinsic_guards(records,selected)
        safe=all(r['success'] and r['audit']['kinematic_pass'] and r['audit']['rest_pass']
            and r['corridor']['margin_lower_mm']>=0. for r in records)
        score=sum(cost(r,b,retained[r['path']]) for r,b in zip(records,selected))
        ep=sum((r['shape']['in_edge_outward_mm']/.002)**2+
            (0. if r['target_motion']['first_turn_inside'] else 1e5) for r in records) if safe else 1e9
        return dict(tag=tag,coefficients=np.asarray(u).tolist(),records=records,score=score,training_score=score+ep,
            physical_pass=bool(safe),measured_inequalities=first.tolist(),
            augmented_measured_inequalities=np.r_[first,extra].tolist(),
            augmented_learning_inequalities=np.r_[shape_event_learning_guards(records,selected,first),extra].tolist(),
            actual_rerun_paths=paths,exact_zero_support_paths=[])
    history=copy.deepcopy(original_history);cache={tuple(c['coefficients']):c for c in history}
    zero=history[0];normalizer=zero['training_score'];reused=len(history);md['reused_candidates']=reused
    elites=sorted((c for c in history if c['physical_pass']),key=lambda c:c['training_score'])[:12]
    x=np.array([c['coefficients'] for c in elites]);empirical=np.cov(x.T)
    covariance=.5*np.eye(n)+.5*empirical*(n/max(float(np.trace(empirical)),1e-30))
    def eligible(c):return c['physical_pass'] and parameters_in_bounds(c['coefficients'],-.05,.05)
    def violation(c):return np.maximum(-np.array(c['augmented_measured_inequalities']),0.)
    def merit(c,dual):
        v=violation(c);return c['training_score']/normalizer+dual@v+.05*(v@v)
    def final_best():return min((c for c in history if eligible(c) and min(c['augmented_measured_inequalities'])>=-1e-8),key=lambda c:c['score'])
    pool=None;blocks=[];outcomes=[]
    try:
        pool=mp.get_context('spawn').Pool(4,worker_initialize,(md,selected,folder,deadline))
        for arm in ('fixed_penalty','adaptive_dual'):
            archive=list(history[:reused]);mean=np.array(seed['coefficients']);cov=covariance.copy();dual=np.zeros(m)
            rng=np.random.default_rng(202609203)
            for generation in range(2):
                eig,vectors=np.linalg.eigh(cov);factor=vectors@np.diag(np.sqrt(np.maximum(eig,1e-10)))
                center=mean.copy();before=len(history);tags=[]
                for pair in range(6):
                    direction=factor@rng.normal(size=n);direction*=.25/max(float(np.max(abs(parent@direction))),1e-12)
                    for sign in (-1.,1.):
                        if time.perf_counter()>deadline-10:raise TimeoutError('Dual parameter training budget')
                        u=np.clip(center+sign*direction,-.05,.05);excursion=float(np.max(abs(parent@u)))
                        if excursion>1.:u/=excursion
                        key=tuple(u.tolist())
                        if key in cache:c=cache[key]
                        else:
                            tag=f'{arm}_g{generation}_p{pair}_s{int(sign)}'
                            records=pool.map(worker_evaluate,[(p,u,tag) for p in paths]);by={r['path']:r for r in records}
                            c=entry([by[r['path']] for r in selected],u,tag);history.append(c);cache[key]=c
                        archive.append(c);tags.append(c['tag'])
                        write_json(folder/'results.json',dict(candidates=history,best=final_best()['tag']))
                chosen=min(filter(eligible,archive),key=lambda c:merit(c,dual))
                if arm=='adaptive_dual':dual=np.maximum(dual-.1*np.array(chosen['augmented_measured_inequalities']),0.)
                elite=sorted(filter(eligible,archive),key=lambda c:merit(c,dual))[:4]
                values=np.array([c['coefficients'] for c in elite]);weights=np.arange(len(elite),0,-1,dtype=float);weights/=weights.sum()
                mu=weights@values;mean=.25*center+.75*mu
                centered=values-mu;empirical=(centered*weights[:,None]).T@centered
                empirical*=n/max(float(np.trace(empirical)),1e-30);cov=.7*cov+.3*empirical
                block=dict(arm=arm,generation=generation,new_policies=len(history)-before,
                    proposal_tags=tags,selected_before_dual_update=chosen['tag'],active_prices=int(np.count_nonzero(dual)),
                    maximum_price=float(dual.max()),selected_training_score=chosen['training_score'],
                    measured_guard_min=min(chosen['augmented_measured_inequalities']))
                blocks.append(block);write_json(stage/'dual_generations.json',blocks)
                print('DUAL_COHERENT_GENERATION', {k:v for k,v in block.items() if k!='proposal_tags'},flush=True)
            chosen=min(filter(eligible,archive),key=lambda c:merit(c,dual))
            actor=copy.deepcopy(md);actor.update(coefficients=np.array(chosen['coefficients']).reshape(-1,3),accepted=False,training_branch_only=True)
            torch.save(actor,stage/(arm+'_training_policy.pt'))
            by={r['path']:r for r in chosen['records']};full=[copy.deepcopy(by.get(r['path'],r)) for r in baseline]
            result=dict(arm=arm,tag=chosen['tag'],training_score=chosen['training_score'],merit=merit(chosen,dual),
                summary=aggregate(full),records=full,dual_prices=dual.tolist(),original_all_guard_pass=min(chosen['augmented_measured_inequalities'])>=-1e-8,
                violated_channels=int(np.count_nonzero(np.array(chosen['augmented_measured_inequalities']) < -1e-8)),
                maximum_violation=float(violation(chosen).max()),adopted=False,protected_candidate_replaced=False)
            outcomes.append(result);write_json(stage/'training_arm_results.json',dict(arms=outcomes))
        pool.close();pool.join();pool=None
        common=[b for b in blocks if b['generation']==0]
        assert common[0]['proposal_tags']==common[1]['proposal_tags'] and common[1]['new_policies']==0
        finish(stage,md,baseline,data,final_best(),zero,history,blocks,choice,parent_file,started)
        receipt(stage,dict(status='completed_dual_coherent_parameter_training',new_policies=len(history)-reused,
            reused_prior_policies=reused,first_generation_shared_bitwise=True,arms=2,ppo_updates=0,
            final_guard_candidate=final_best()['tag'],research_acceptable=(stage/'research_candidate.pt').exists(),
            source_model_sha256=sha(parent_file),wall_s=time.perf_counter()-started,adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_dual_coherent_parameter_training',error=str(exc),
            new_policies=len(history)-reused,completed_arms=len(outcomes),wall_s=time.perf_counter()-started,
            adopted=False,goal_complete=False));raise


if __name__=='__main__':
    mp.freeze_support();main()
