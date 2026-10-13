"""Matched action-capacity learning from measured complete policy returns."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse
import copy
import time
import multiprocessing as mp
import numpy as np
import torch
from joint_teacher_state_policy import features,state_pair,reference_oriented_observation,INDICES,OUTPUT_SCALE
from consolidate_joint_teacher_policy import collect_data
from saved_memory_action_identity import SavedMemoryActionIdentity
from gate_band_state import GateBandRuntime
from learn_early_memory_joint_policy import worker_initialize,worker_evaluate
from joint_training_evaluator import build_evaluator
from learn_full_pool_joint_credit import aggregate
from audit_current_target_changes import target_changes
from study_io import BASE,register,write_json,receipt,sha
from search_complete_return import read


SOURCE=BASE/'canonical_gate_band_shared_reference_transfer_joint_learning_20260920'
PARENT=BASE/'canonical_gate_band_reference_supported_joint_learning_20260920/research_candidate.pt'
NAME='canonical_gate_band_action_capacity_joint_training_20260920'


def prepare():
    assert sha(PARENT)=='ded2a93f85f3503a8c09bf59fd79e48c8f2cd187874f07789c53b8863c936229'
    data=read(SOURCE/'baseline.json');baseline=data['baseline']['records']
    parent=torch.load(PARENT,map_location='cpu',weights_only=False)
    observations=[];path_ids=[];decisions=[];selectors=[]
    for r in baseline:
        with np.load(r['trace']) as z:obs=z['observations'].copy()
        observations.extend(obs);path_ids.extend([r['path']]*len(obs));decisions.extend(range(len(obs)))
        if r['path'] in (105,204):
            events=target_changes(r['trace'])['largest_changes']
            event=max((e for e in events if e['decision_boundary']),key=lambda e:abs(e['target_change_deg']))
            j=event['tick']//20
            selectors.extend(dict(path=r['path'],decision=d) for d in (j-1,j))
    obs=np.array(observations);ids=np.array(path_ids);decision=np.array(decisions)
    innovation=np.zeros((len(obs),3));raw,_=state_pair(reference_oriented_observation(obs),innovation)
    k=len(selectors)
    mean,std,weights=[np.repeat(parent[key][-1:],k,axis=0) for key in
                      ('per_center_mean','per_center_std','per_center_distance_weights')]
    assert not np.any(weights[:,len(INDICES):])
    centers=np.array([(raw[np.flatnonzero((ids==s['path'])&(decision==s['decision'])).item()]-mean[i])/std[i]
                      for i,s in enumerate(selectors)])
    md=dict(architecture='observable_memory_joint_radial',parent=parent,centers=centers,
        mean=mean[0],std=std[0],per_center_mean=mean,per_center_std=std,
        per_center_distance_weights=weights,per_center_reference_frame=np.ones(k,dtype=bool),
        memory_decays=[.5,.8],coefficients=np.zeros((3*k,3)),accepted=False,
        action_interval_ms=20,physical_tick_ms=1,
        learning_method='paired_action_capacity_complete_policy_parameter_learning')
    choices=[]
    for width in (.05,.1,.2,.3,.6):
        md['radii']=np.full(k,width);active=np.any(features(md,obs,innovation)!=0.,axis=(1,2))
        paths=sorted(np.unique(ids[active]).tolist())
        choices.append(dict(width=width,paths=paths,decisions=int(active.sum()),
                            nonisolated=[r['path'] for r in baseline if r['path'] in paths and 'shape' not in r]))
    md['radii']=np.full(k,.2)
    support=dict(selected=next(c for c in choices if c['width']==.2),choices=choices,
        selectors_for_training_only=selectors,all_training_paths=len(baseline),
        outside_support_exact_zero=True,no_runtime_time_or_path_inputs=True)
    return data,md,support


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1)
    data,md,support=prepare()
    print('CAPACITY_SUPPORT',support,flush=True)
    if args.prepare:return
    started=time.perf_counter();deadline=started+700.
    baseline=data['baseline']['records'];paths=support['selected']['paths']
    selected=[r for r in baseline if r['path'] in paths]
    assert not support['selected']['nonisolated'],'Use a full multicorner evaluator if scope changes'
    n=md['coefficients'].size
    evidence=BASE/'canonical_gate_band_context_coherent_parameter_training_20260920'
    stage=register(NAME,[__file__,PARENT,SOURCE/'baseline.json',SOURCE/'receipt.json',
        evidence/'target_event_capacity_audit.json',evidence/'current_target_change_audit.json']+
        [BASE/f for f in ('joint_training_evaluator.py','observable_memory_joint_policy.py',
            'joint_teacher_state_policy.py','saved_memory_action_identity.py','learn_early_memory_joint_policy.py',
            'head_counterfactual_rollouts.py','learn_shared_temporal_joint_policy.py')],
        dict(wall_s=720,workers=4,arms=['small','wide'],parameters_per_arm=n,
            coefficient_bounds=[.03,3.],generations_per_arm=3,population=12,maximum_new_policies=72,
            radius=.2,normalized_search_step=.25,random_seed=202609205,ppo_updates=0),
        'The latest saved-state capacity audit shows tiny available corrections at the actual '
        'large target-change states. Return to the protected ded2 parent, with a ZERO joint '
        'residual on four observed states immediately before/at the largest decision-boundary '
        'target changes on105/204. State geometry and two causal memories are shared; no '
        'path,time,angle-table or target-trajectory input enters the actor. Radius0.2 includes '
        'all13 affected isolated corners, including104; all88 training routes are scanned. '
        'Compare identical36-parameter actors with coefficient bounds0.03 versus3, same '
        'normalized antithetic directions, same population,three mean/covariance generations '
        'and full measured return. Do not normalize by the previous tiny exploration sigma. '
        'This tests action capacity at a matched new support; it does not isolate support '
        'from the prior162-parameter experiment. Training uses adaptive violation prices '
        'and may traverse soft regressions; all physical constraints and ALL original final '
        'guards remain unchanged. Reuse zero-support routes by decoded-identity induction, '
        'never count failure prefixes as successful time.720s,at most72 new policies. '
        'Adoption requires a strict full-return improvement and every original guard, then '
        'raw actor/XY audit. If wide search only trades errors, inspect measured three-head '
        'and temporal responses and change the learning mechanism instead of extending '
        'the search. Complete closed-loop parameter optimization,not PPO. No sealed tests.')
    folder=stage/'state_joint';folder.mkdir()
    write_json(stage/'baseline.json',data);write_json(stage/'support_audit.json',support)
    torch.save(md,stage/'initial_policy.pt')
    write_json(stage/'selection_hold.json',dict(reason='Training mechanism comparison; raw acceptance audit required.'))
    replay=collect_data(md,md['parent'],selected,include_latents=True,deadline=deadline)
    replay['held_actions']=np.vstack([np.load(r['trace'])['actions'] for r in selected])
    np.savez_compressed(stage/'affected_replay.npz',**replay)
    identity=SavedMemoryActionIdentity(md,replay,GateBandRuntime.decode_action)
    assert all(identity.unchanged(p,np.zeros(n)) for p in paths)
    capacity=[]
    for p,(f,memory,latents,held) in identity.paths.items():
        event=max((e for e in target_changes(next(r['trace'] for r in selected if r['path']==p))['largest_changes']
                   if e['decision_boundary']),key=lambda e:abs(e['target_change_deg']))
        j=event['tick']//20;design=np.concatenate([f,memory[:,0],memory[:,1]],axis=1)*OUTPUT_SCALE
        for bound in (.03,3.):
            delta=bound*np.sum(abs(design[j]),axis=0)
            width=GateBandRuntime.decode_action(latents[j]+delta)-GateBandRuntime.decode_action(latents[j]-delta)
            capacity.append(dict(path=p,decision=j,bound=bound,decoded_box_width=
                [float(np.degrees(width[0])),float(width[1]),float(width[2])]))
    write_json(stage/'preflight_identity_capacity.json',dict(zero_decoded_identity=True,
        affected_decisions=len(replay['decisions']),all_training_routes_scanned=88,
        zero_support_paths=[r['path'] for r in baseline if r['path'] not in paths],capacity=capacity,
        protected_model_sha256=sha(PARENT),capacity_scope='Saved-state box widths, not simultaneous achievable target correction.'))
    print('CAPACITY_PREFLIGHT',[(c['path'],c['bound'],c['decoded_box_width']) for c in capacity if c['path'] in (104,105,204)],flush=True)
    entry=build_evaluator(data,selected,stage,md['parent'],deadline)
    zero=entry(copy.deepcopy(selected),np.zeros(n),'protected_parent')
    zero.update(actual_rerun_paths=[],exact_zero_support_paths=paths)
    assert zero['physical_pass'] and min(zero['augmented_measured_inequalities'])>=-1e-8
    normalizer=zero['training_score'];history=[zero];outcomes=[];blocks=[];pool=None
    def eligible(c):return c['physical_pass']
    def acceptable(c):return eligible(c) and min(c['augmented_measured_inequalities'])>=-1e-8 and c['training_score']<normalizer-1e-6
    try:
        pool=mp.get_context('spawn').Pool(4,worker_initialize,(md,selected,folder,deadline))
        for arm,bound in (('small',.03),('wide',3.)):
            archive=[zero];mean=np.zeros(n);cov=np.eye(n);dual=np.zeros(len(zero['augmented_measured_inequalities']))
            rng=np.random.default_rng(202609205)
            def merit(c):
                v=np.maximum(-np.array(c['augmented_measured_inequalities']),0.)
                return c['training_score']/normalizer+dual@v+.05*(v@v)
            for generation in range(3):
                eig,vectors=np.linalg.eigh(cov);factor=vectors@np.diag(np.sqrt(np.maximum(eig,1e-10)))
                center=mean.copy();batch=[min(filter(eligible,archive),key=merit)]
                for pair in range(6):
                    direction=factor@rng.normal(size=n);direction*=.25/max(float(np.max(abs(direction))),1e-12)
                    for sign in (-1.,1.):
                        if time.perf_counter()>deadline-15:raise TimeoutError('Joint capacity wall budget')
                        u=bound*np.clip(center+sign*direction,-1.,1.)
                        tag=f'{arm}_g{generation}_p{pair}_s{int(sign)}'
                        unchanged=[p for p in paths if identity.unchanged(p,u)]
                        actual=[p for p in paths if p not in unchanged]
                        records=pool.map(worker_evaluate,[(p,u,tag) for p in actual]);by={r['path']:r for r in records}
                        c=entry([by.get(r['path'],copy.deepcopy(r)) for r in selected],u,tag)
                        c.update(arm=arm,actual_rerun_paths=actual,exact_zero_support_paths=unchanged)
                        archive.append(c);history.append(c);batch.append(c)
                        write_json(folder/'results.json',dict(candidates=history,best=None,
                            training_best=min(filter(eligible,archive),key=merit)['tag']))
                chosen=min(filter(eligible,batch),key=merit)
                dual=np.maximum(0.,dual-.1*np.array(chosen['augmented_measured_inequalities']))
                elite=sorted(filter(eligible,batch),key=merit)[:4]
                values=np.array([c['coefficients'] for c in elite])/bound
                weights=np.arange(len(elite),0,-1,dtype=float);weights/=weights.sum()
                mu=weights@values;mean=.25*center+.75*mu
                centered=values-mu;empirical=(centered*weights[:,None]).T@centered
                empirical*=n/max(float(np.trace(empirical)),1e-30);cov=.7*cov+.3*empirical
                chosen=min(filter(eligible,archive),key=merit)
                block=dict(arm=arm,generation=generation,training_best=chosen['tag'],training_score=chosen['training_score'],
                    merit=merit(chosen),guard_min=min(chosen['augmented_measured_inequalities']),
                    physical_passing=sum(c['physical_pass'] for c in batch[1:]),admissible=sum(acceptable(c) for c in archive))
                blocks.append(block);write_json(stage/'capacity_generations.json',blocks)
                print('CAPACITY_GENERATION',block,flush=True)
            chosen=min(filter(eligible,archive),key=merit)
            actor=copy.deepcopy(md);actor.update(coefficients=np.array(chosen['coefficients']).reshape(-1,3),
                accepted=False,training_branch_only=True)
            torch.save(actor,stage/(arm+'_training_policy.pt'))
            by={r['path']:r for r in chosen['records']};full=[copy.deepcopy(by.get(r['path'],r)) for r in baseline]
            v=np.maximum(-np.array(chosen['augmented_measured_inequalities']),0.)
            passing=list(filter(acceptable,archive))
            outcomes.append(dict(arm=arm,tag=chosen['tag'],training_score=chosen['training_score'],merit=merit(chosen),
                summary=aggregate(full),records=full,maximum_violation=float(v.max()),
                violated_channels=int(np.count_nonzero(v>1e-8)),original_all_guard_pass=bool(np.all(v<=1e-8)),
                admissible_new_policies=[c['tag'] for c in passing],dual_prices=dual.tolist(),adopted=False))
            write_json(stage/'training_arm_results.json',dict(arms=outcomes))
        pool.close();pool.join();pool=None
        passing=list(filter(acceptable,history))
        if passing:
            best=min(passing,key=lambda c:c['training_score']);actor=copy.deepcopy(md)
            actor['coefficients']=np.array(best['coefficients']).reshape(-1,3)
            torch.save(actor,stage/'guard_passing_candidate.pt')
            write_json(stage/'pending_candidate_audit.json',dict(tag=best['tag'],model_sha256=sha(stage/'guard_passing_candidate.pt')))
        receipt(stage,dict(status='completed_action_capacity_joint_training',new_policies=len(history)-1,
            ppo_updates=0,original_guard_passing_new_policies=len(passing),research_acceptable=False,
            pending_candidate_audit=bool(passing),source_model_sha256=sha(PARENT),
            wall_s=time.perf_counter()-started,adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_action_capacity_joint_training',error=str(exc),
            new_policies=len(history)-1,completed_arms=len(outcomes),wall_s=time.perf_counter()-started,
            adopted=False,goal_complete=False));raise


if __name__=='__main__':
    mp.freeze_support();main()
