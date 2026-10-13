"""Equal-capacity geometry/dynamics context under measured whole-policy learning."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import copy
import time
import multiprocessing as mp
import numpy as np
import torch
from train_matched_context_ppo import contextualize
from train_constraint_counterfactual_ppo import design_matrix
from head_counterfactual_rollouts import exploration_sigma
from saved_memory_action_identity import SavedMemoryActionIdentity
from gate_band_state import GateBandRuntime
from joint_teacher_state_policy import OUTPUT_SCALE
from joint_training_evaluator import build_evaluator
from learn_early_memory_joint_policy import worker_initialize,worker_evaluate
from learn_full_pool_joint_credit import aggregate
from study_io import BASE,register,write_json,receipt,sha
from search_complete_return import read


def main():
    torch.set_num_threads(1);started=time.perf_counter();deadline=started+340.
    source=BASE/'canonical_gate_band_dual_coherent_parameter_training_20260920'
    parent_file=source/'adaptive_dual_training_policy.pt'
    protected=BASE/'canonical_gate_band_reference_supported_joint_learning_20260920/research_candidate.pt'
    assert sha(protected)=='ded2a93f85f3503a8c09bf59fd79e48c8f2cd187874f07789c53b8863c936229'
    audit=read(source/'adaptive_dual_raw_identity_audit.json');assert audit['model_sha256']==sha(parent_file)
    seed=next(a for a in read(source/'training_arm_results.json')['arms'] if a['arm']=='adaptive_dual')
    data=read(source/'baseline.json');baseline=data['baseline']['records']
    support=read(source/'support_audit.json');choice=support['selected'];paths=choice['paths']
    selected=[r for r in baseline if r['path'] in paths];seed_records=[r for r in seed['records'] if r['path'] in paths]
    original=torch.load(source/'initial_policy.pt',map_location='cpu',weights_only=False)
    original['parent']=torch.load(parent_file,map_location='cpu',weights_only=False)
    original['coefficients']=np.zeros_like(original['coefficients'])
    with np.load(source/'adaptive_dual_actor_replay.npz') as z:replay={k:z[k].copy() for k in z.files}
    replay['parent_latents']=replay['current_latents'].copy()
    episodes=[dict(path=r['path'],seed=0,record=r,samples=[]) for r in seed_records]
    dual=np.array(seed['dual_prices']);normalizer=read(source/'state_joint/results.json')['candidates'][0]['training_score']
    stage=register('canonical_gate_band_context_coherent_parameter_training_20260920',
        [__file__,parent_file,protected,source/'receipt.json',source/'initial_policy.pt',source/'baseline.json',
         source/'support_audit.json',source/'training_arm_results.json',source/'adaptive_dual_raw_identity_audit.json',
         source/'adaptive_dual_actor_replay.npz']+[BASE/f for f in ('joint_training_evaluator.py',
         'train_matched_context_ppo.py','observable_memory_joint_policy.py','saved_memory_action_identity.py',
         'head_counterfactual_rollouts.py','learn_early_memory_joint_policy.py')],
        dict(wall_s=360,workers=4,arms=['geometry','dynamic'],parameters_per_arm=162,
            generations_per_arm=2,population=12,maximum_new_policies=48,ppo_updates=0,
            action_perturbation_radius=.25,global_action_bound=.7,coefficient_bound=.03,random_seed=202609204),
        'The fixed/adaptive penalty comparison keeps selecting tradeoffs, not a new feasible '
        'policy. Test whether state-conditioned compensation can improve the SAME training '
        'seed under measured complete-policy learning. Add a ZERO residual layer to the '
        'audited adaptive-dual training actor. Equal162 parameters in two arms: geometric '
        'range/side context versus observed speed/tangential acceleration context. Both '
        'retain identical compact geometric support and causal memories. Same frozen '
        'dual prices,full-return objective,two generations,twelve proposals,antithetic '
        'seed and action trust. Prior context-PPO/local-gradient experiments are NOT rerun: '
        'this tests context in the new full-policy training branch and cannot attribute '
        'all historical failures to representation. Verify ZERO-layer decoded identity '
        'on every saved decision. Other81 routes remain exact zero-support by induction. '
        'No route/angle IDs or trajectory templates in the actor. Training soft metrics '
        'may vary,physical hard constraints and original203 final guards are unchanged. '
        'Parameter optimization,not PPO. At most48 new policies/360s. If no arm improves '
        'the actual shape/cruise tradeoff,inspect the changed physical target and state '
        'response; do not extend the same search or claim superiority/generalization.')
    folder=stage/'state_joint';folder.mkdir();write_json(stage/'baseline.json',data);write_json(stage/'support_audit.json',support)
    write_json(stage/'selection_hold.json',dict(reason='Training-only representation comparison; protected model unchanged.'))
    entry=build_evaluator(data,selected,stage,original['parent'],deadline)
    # Refactored evaluator must reproduce the previous frozen objective/guards exactly.
    previous=next(c for c in read(source/'state_joint/results.json')['candidates'] if c['tag']==seed['tag'])
    checked=entry(copy.deepcopy(seed_records),np.zeros(162),'seed')
    assert checked['training_score']==previous['training_score']
    assert np.array_equal(checked['augmented_measured_inequalities'],previous['augmented_measured_inequalities'])
    history=[];blocks=[];outcomes=[];pool=None
    def merit(c):
        v=np.maximum(-np.array(c['augmented_measured_inequalities']),0.)
        return c['training_score']/normalizer+dual@v+.05*(v@v)
    try:
        for mode in ('geometry','dynamic'):
            md,_,context_audit=contextualize(original,episodes,mode)
            md['learning_method']='contextual_complete_policy_covariance_learning'
            proof=SavedMemoryActionIdentity(md,replay,GateBandRuntime.decode_action)
            assert all(proof.unchanged(p,np.zeros(162)) for p in paths)
            matrix=[]
            for f,memory,_,_ in proof.paths.values():
                design=np.concatenate([f,memory[:,0],memory[:,1]],axis=1)*OUTPUT_SCALE
                matrix.extend(design_matrix(design,exploration_sigma(md,design)).reshape(-1,162))
            matrix=np.array(matrix);matrix=matrix[np.any(matrix!=0.,axis=1)]
            torch.save(md,stage/(mode+'_initial_policy.pt'))
            if mode=='geometry':torch.save(md,stage/'initial_policy.pt')
            write_json(stage/(mode+'_identity.json'),dict(context_audit,zero_layer_decoded_identity=True,
                parent_sha256=sha(parent_file),decisions=len(replay['decisions'])))
            zero=entry(copy.deepcopy(seed_records),np.zeros(162),mode+'_seed');zero['context_arm']=mode
            zero.update(actual_rerun_paths=[],exact_zero_support_paths=paths)
            archive=[zero];history.append(zero);mean=np.zeros(162);cov=np.eye(162);rng=np.random.default_rng(202609204)
            pool=mp.get_context('spawn').Pool(4,worker_initialize,(md,seed_records,folder,deadline))
            for generation in range(2):
                eig,vectors=np.linalg.eigh(cov);factor=vectors@np.diag(np.sqrt(np.maximum(eig,1e-10)))
                center=mean.copy();batch=[min((c for c in archive if c['physical_pass']),key=merit)]
                for pair in range(6):
                    direction=factor@rng.normal(size=162);direction*=.25/max(float(np.max(abs(matrix@direction))),1e-12)
                    for sign in (-1.,1.):
                        if time.perf_counter()>deadline-10:raise TimeoutError('Context parameter budget')
                        u=np.clip(center+sign*direction,-.03,.03);excursion=float(np.max(abs(matrix@u)))
                        if excursion>.7:u*=.7/excursion
                        tag=f'{mode}_g{generation}_p{pair}_s{int(sign)}'
                        records=pool.map(worker_evaluate,[(p,u,tag) for p in paths]);by={r['path']:r for r in records}
                        c=entry([by[r['path']] for r in selected],u,tag);c['context_arm']=mode
                        archive.append(c);history.append(c);batch.append(c)
                        write_json(folder/'results.json',dict(candidates=history,training_best=min(
                            (r for r in history if r['physical_pass']),key=merit)['tag'],best=None))
                elite=sorted((c for c in batch if c['physical_pass']),key=merit)[:4]
                values=np.array([c['coefficients'] for c in elite]);weights=np.arange(len(elite),0,-1,dtype=float);weights/=weights.sum()
                mu=weights@values;mean=.25*center+.75*mu
                centered=values-mu;empirical=(centered*weights[:,None]).T@centered
                empirical*=162/max(float(np.trace(empirical)),1e-30);cov=.7*cov+.3*empirical
                chosen=min((c for c in archive if c['physical_pass']),key=merit)
                block=dict(mode=mode,generation=generation,training_best=chosen['tag'],training_score=chosen['training_score'],
                    merit=merit(chosen),guard_min=min(chosen['augmented_measured_inequalities']))
                blocks.append(block);write_json(stage/'context_generations.json',blocks)
                print('CONTEXT_COHERENT_GENERATION',block,flush=True)
            pool.close();pool.join();pool=None
            chosen=min((c for c in archive if c['physical_pass']),key=merit)
            actor=copy.deepcopy(md);actor.update(coefficients=np.array(chosen['coefficients']).reshape(-1,3),accepted=False,training_branch_only=True)
            torch.save(actor,stage/(mode+'_training_policy.pt'))
            by={r['path']:r for r in chosen['records']};full=[copy.deepcopy(by.get(r['path'],r)) for r in seed['records']]
            v=np.maximum(-np.array(chosen['augmented_measured_inequalities']),0.)
            outcomes.append(dict(arm=mode,tag=chosen['tag'],training_score=chosen['training_score'],merit=merit(chosen),
                summary=aggregate(full),records=full,maximum_violation=float(v.max()),violated_channels=int(np.count_nonzero(v>1e-8)),
                original_all_guard_pass=bool(np.all(v<=1e-8)),adopted=False,protected_candidate_replaced=False))
            write_json(stage/'training_arm_results.json',dict(arms=outcomes))
        receipt(stage,dict(status='completed_context_coherent_parameter_training',new_policies=len(history)-2,
            arms=2,ppo_updates=0,original_guard_passing_new_policies=sum(c['physical_pass'] and
                min(c['augmented_measured_inequalities'])>=-1e-8 for c in history if not c['tag'].endswith('_seed')),
            research_acceptable=False,source_model_sha256=sha(parent_file),protected_model_sha256=sha(protected),
            wall_s=time.perf_counter()-started,adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_context_coherent_parameter_training',error=str(exc),
            stored_policies=len(history),completed_arms=len(outcomes),wall_s=time.perf_counter()-started,
            adopted=False,goal_complete=False));raise


if __name__=='__main__':
    mp.freeze_support();main()
