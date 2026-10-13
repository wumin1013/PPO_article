"""Deterministic complete-suffix action gradients and joint constrained actor updates."""
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
from scipy.optimize import linprog
from observable_memory_joint_policy import ObservableMemoryJointPolicy
from head_counterfactual_rollouts import action_distribution,cost
from saved_memory_action_identity import SavedMemoryActionIdentity
from gate_band_state import GateBandRuntime,environment_factory
from analyze_nonlinear_joint_response import measure
from joint_teacher_state_policy import OUTPUT_SCALE
from train_matched_context_ppo import contextualize
from train_constraint_counterfactual_ppo import design_matrix
from head_counterfactual_rollouts import exploration_sigma
from learn_early_memory_joint_policy import metric_functions,worker_initialize,worker_evaluate
from duration_joint_metrics import guards
from shape_event_credit import shape_event_learning_guards
from learn_intrinsic_joint_recovery import intrinsic_guards
from learn_direct_observable_joint_policy import finish
from study_io import BASE,register,write_json,receipt,sha
from search_complete_return import read


WORK={}


def initialize(md,records,retained,folder,deadline,radii):
    torch.set_num_threads(1)
    WORK.update(md=md,records={r['path']:r for r in records},
        retained={r['path']:r['baseline_shape'] for r in retained if 'baseline_shape' in r},
        folder=folder,deadline=deadline,radii=radii)


def collect(job):
    path,queries,peak,weights,*extra=job
    cached=extra[0] if extra else None
    cached_branches={} if cached is None else {(b['decision'],b['head'],b['radius'],b['sign']):b for b in cached['branches']}
    baseline=WORK['records'][path]
    with np.load(baseline['trace']) as z:
        ref,full,held,observed=[z[k].copy() for k in ('reference','rows','actions','observations')]
    policy=ObservableMemoryJointPolicy(copy.deepcopy(WORK['md']))
    env=policy.bind(environment_factory(ref))
    snapshots,samples={},[]
    while not env.done and env.decisions<=max(queries):
        if time.perf_counter()>WORK['deadline']:raise TimeoutError('Deterministic prefix budget')
        j=env.decisions
        x=env.observation()
        mean,design,sigma=action_distribution(policy,x)
        assert np.array_equal(x,observed[j]),('parent_observation_identity',path,j)
        if j in queries:
            assert np.all(sigma>0.)
            snapshots[j]=copy.deepcopy((env,policy))
            samples.append(dict(decision=j,design=design.tolist(),sigma=sigma.tolist(),
                observation=x.tolist(),mean=mean.tolist(),query_weight=weights[j]))
        _,_,_,info=env.step(mean)
        assert np.array_equal(info['held_action'],held[j]),('parent_action_identity',path,j)
        assert np.array_equal(info['rows'],full[j*20:j*20+len(info['rows'])]),('parent_xy_identity',path,j)
    branches=[];new_suffixes=reused_suffixes=0
    for qi,sample in enumerate(samples):
        index=sample['decision']
        for head in range(3):
            for radius in (WORK['radii'] if index==peak else WORK['radii'][:1]):
                for sign in (-1,1):
                    if (index,head,radius,sign) in cached_branches:
                        previous=cached_branches[index,head,radius,sign]
                        assert previous['epsilon']==radius*sample['sigma'][head]
                        branches.append(copy.deepcopy(previous));reused_suffixes+=1
                        continue
                    branch_env,branch_policy=copy.deepcopy(snapshots[index])
                    chunks,actions,observations=[],[],[]
                    epsilon=radius*sample['sigma'][head]
                    while not branch_env.done:
                        if time.perf_counter()>WORK['deadline']:raise TimeoutError('Deterministic suffix budget')
                        j=branch_env.decisions
                        x=branch_env.observation()
                        latent=branch_policy.act(x,True)[0].astype(float)
                        if j==index:latent[head]+=sign*epsilon
                        _,_,_,info=branch_env.step(latent)
                        chunks.append(info['rows']);actions.append(info['held_action']);observations.append(x)
                    rows=np.vstack([full[:index*20],*chunks])
                    file=WORK['folder']/f'p{path}_d{index}_h{head}_e{radius}_s{sign}.npz'
                    np.savez_compressed(file,reference=ref,rows=rows,
                        actions=np.vstack([held[:index],actions]),observations=np.vstack([observed[:index],observations]))
                    record=measure(ref,rows,branch_env)
                    record.update(path=path,trace=str(file),curved_path=True)
                    branches.append(dict(decision=index,head=head,radius=radius,sign=sign,epsilon=epsilon,
                        score=cost(record,baseline,WORK['retained'][path]),record=record))
                    new_suffixes+=1
        if qi%5==0:print('DETERMINISTIC_SUFFIX_PROGRESS',path,qi+1,len(samples),len(branches),flush=True)
    result=dict(path=path,seed=0,record=baseline,samples=samples,branches=branches,
        prefix_xy_actions_bitwise=True,future_policy_deterministic=True,peak=peak,
        new_suffixes=new_suffixes,reused_suffixes=reused_suffixes)
    write_json(WORK['folder']/f'path{path}_credit.json',result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--local-radius',action='store_true')
    parser.add_argument('--complete-support',action='store_true')
    args=parser.parse_args()
    if args.complete_support:args.local_radius=True
    radii=(.1,.05) if args.local_radius else (1.,.5)
    torch.set_num_threads(1)
    started=time.perf_counter()
    source=BASE/'canonical_gate_band_head_counterfactual_ppo_collection_20260920'
    previous=BASE/'canonical_gate_band_stratified_constraint_credit_ppo_20260920'
    parent_file=BASE/'canonical_gate_band_reference_supported_joint_learning_20260920/research_candidate.pt'
    assert sha(parent_file)=='ded2a93f85f3503a8c09bf59fd79e48c8f2cd187874f07789c53b8863c936229'
    assert not read(previous/'receipt.json')['research_acceptable']
    original=torch.load(source/'initial_policy.pt',map_location='cpu',weights_only=False)
    data=read(source/'baseline.json');baseline=data['baseline']['records']
    support=read(source/'support_audit.json');choice=support['selected'];paths=choice['paths']
    selected=[r for r in baseline if r['path'] in paths]
    replay_file=BASE/'canonical_gate_band_shared_reference_transfer_joint_learning_20260920/affected_replay.npz'
    with np.load(replay_file) as z:replay={k:z[k].copy() for k in z.files}
    identity=SavedMemoryActionIdentity(original,replay,GateBandRuntime.decode_action)
    cached_source=BASE/'canonical_gate_band_deterministic_local_credit_20260920'
    cached_episodes=read(cached_source/'deterministic_episodes.json')['episodes'] if args.complete_support else []
    jobs,sampling=[],[]
    for p,(f,memory,_,_) in identity.paths.items():
        design=np.concatenate([f,memory[:,0],memory[:,1]],axis=1)*OUTPUT_SCALE
        energy=np.sum(design*design,axis=(1,2));nonzero=np.flatnonzero(energy)
        ordered=nonzero[np.argsort(energy[nonzero])[::-1]]
        number=int(np.searchsorted(np.cumsum(energy[ordered])/energy.sum(),.99)+1)
        dense=ordered[:number];tail=ordered[number:]
        weights={int(q):1 for q in dense}
        if len(tail):weights[int(np.random.default_rng(20260920+p).choice(tail))]=len(tail)
        if args.complete_support:weights={int(q):1 for q in nonzero}
        queries=sorted(weights);peak=int(energy.argmax())
        jobs.append((p,queries,peak,weights,next(e for e in cached_episodes if e['path']==p)) if args.complete_support
                    else (p,queries,peak,weights))
        sampling.append(dict(path=p,dense_queries=dense.tolist(),tail=tail.tolist(),queries=queries,
            inverse_inclusion_weights=weights,peak=peak))
    stage=register('canonical_gate_band_deterministic_full_support_credit_20260920' if args.complete_support else
                   'canonical_gate_band_deterministic_local_credit_20260920' if args.local_radius
                   else 'canonical_gate_band_deterministic_complete_credit_20260920',
        [__file__,parent_file,source/'initial_policy.pt',source/'baseline.json',source/'receipt.json',
         previous/'receipt.json',replay_file]+[BASE/n for n in ('observable_memory_joint_policy.py',
         'head_counterfactual_rollouts.py','saved_memory_action_identity.py','train_matched_context_ppo.py',
         'train_constraint_counterfactual_ppo.py','learn_early_memory_joint_policy.py',
         'shape_event_credit.py','duration_joint_metrics.py','learn_intrinsic_joint_recovery.py')]+
        ([cached_source/'receipt.json',cached_source/'deterministic_episodes.json',
          cached_source/'tail_credit_influence_audit.json'] if args.complete_support else []),
        dict(wall_s=900 if args.complete_support else 660,workers=4,paths=paths,queries=sum(len(j[1]) for j in jobs),
            suffixes=sum(6*len(j[1])+6 for j in jobs),new_stochastic_episodes=0,
            deterministic_parameter_learning=True,actor_arms=['geometry','dynamic'],parameters_per_arm=162,
            maximum_standardized_action_changes=[.05,.15],maximum_new_policies=4,measurement_radii=radii),
        ('FULL TEMPORAL CREDIT CONTROL: the sparse tail sample contributes15% and11% '
         'of the sum of dense/tail gradient norms on105/304 despite the tail containing '
         'only1% of design energy. Replace the sampled tail estimate by the actual '
         'sum over EVERY supported decision on all7 deterministic parent routes. '
         'Keep the SAME local .1/.05 perturbation radii, actor functions, two trust '
         'bounds, LP objective and ALL final guards. Reuse every existing474 conditional '
         'branch by identity; collect only missing decision/head/sign combinations. '
         'All inclusion weights become1. This removes temporal subsampling on these '
         'routes, not finite-difference, nonsmoothness or nonlinear joint-update error. '
         'Budget900s including learning. No repeated old physical experiment. ' if args.complete_support else '')+
        ('Matched radius control: the previous full/half-radius response audit showed '
         'large constraint-gradient changes, while actual visited-state residual changes '
         'closely followed parent-state trust limits. Keep exactly the same72 queries, '
         'paths, tail seed/weights, actors and optimization settings. Change ONLY the '
         'measurement perturbations from1/.5 to.1/.05 times exploration sigma. '
         'No old branch is repeated. This tests the local validity of measured credit '
         'and directly uses the corrected credit for joint updates. ' if args.local_radius else '')+
        'Mechanism control aligning action credit with deterministic deployment after '
        'stochastic PPO credit updates failed. On all7 supported complete parent routes, '
        'replay original states/actions/XY exactly and replace a single head by symmetric '
        'perturbations, with all future actions from the deterministic closed-loop policy. '
        'Collect all decisions covering99% of parent design energy and ONE uniformly '
        'sampled remaining-tail decision with its exact inverse inclusion weight. Energy '
        'is a sampling proxy, not proof of causal importance. Use a second half-radius '
        'pair at every route peak to audit finite-difference sensitivity. Learn paired '
        'equal162-parameter geometry/dynamic actors using complete-return action '
        'differences chained through their state/history features, with linearized '
        'per-metric constraint gradients and bounded policy changes. All7 supported '
        'routes contribute credit; no unsampled route equality freezes them. Recheck '
        'all203 actual guards, complete stop, original physics, time and cruise for '
        'each new policy. This is sampled finite-difference deterministic policy-gradient '
        'parameter learning, NOT PPO, critic fitting or a claim of exact unbiased gradients '
        'in a nonsmooth simulator. Sampling and perturbations are never deployed. This '
        'changes data and estimator together relative to PPO, not a single-factor proof. '
        '660s, no sealed tests, no final adoption without full audits.')
    deadline=started+(880. if args.complete_support else 640.)
    folder=stage/'state_joint';folder.mkdir();raw=stage/'suffixes';raw.mkdir()
    write_json(stage/'baseline.json',data);write_json(stage/'support_audit.json',support)
    write_json(stage/'sampling_design.json',sampling)
    _,old_guards=metric_functions(data,selected,stage,original['parent'],deadline)
    def metrics(records):
        first=np.concatenate([old_guards(records)]+[guards(r,b) for r,b in zip(records,selected)])
        extra=intrinsic_guards(records,selected)
        return first,np.r_[first,extra],np.r_[shape_event_learning_guards(records,selected,first),extra]
    retained={r['path']:r['baseline_shape'] for r in data['retained'] if 'baseline_shape' in r}
    def score(records):return sum(cost(r,b,retained[r['path']]) for r,b in zip(records,selected))
    zero_first,zero_measured,zero_learning=metrics(selected)
    episodes,history,blocks,models=[],[],[],{}
    pool=None
    try:
        pool=mp.get_context('spawn').Pool(4,initialize,(original,selected,data['retained'],raw,deadline,radii))
        for result in pool.imap_unordered(collect,jobs):
            episodes.append(result)
            write_json(stage/'deterministic_episodes.json',dict(episodes=episodes))
            print('DETERMINISTIC_CREDIT_COLLECTED',result['path'],len(result['branches']),flush=True)
        pool.close();pool.join();pool=None
        episodes.sort(key=lambda e:e['path'])
        response={};radius_audit=[]
        for e in episodes:
            by={(r['decision'],r['head'],r['radius'],r['sign']):r for r in e['branches']}
            for s in e['samples']:
                j=s['decision']
                for h in range(3):
                    derivatives={}
                    for radius in (radii if j==e['peak'] else radii[:1]):
                        minus,plus=[by[j,h,radius,sign] for sign in (-1,1)]
                        values=[]
                        for r in (minus,plus):
                            records=[r['record'] if b['path']==e['path'] else b for b in selected]
                            values.append(np.r_[r['score'],metrics(records)[2]])
                        derivatives[radius]=(values[1]-values[0])/(2*plus['epsilon'])
                    response[e['path'],j,h]=derivatives[radii[0]]
                    if len(derivatives)>1:
                        a,b=derivatives[radii[0]],derivatives[radii[1]]
                        radius_audit.append(dict(path=e['path'],decision=j,head=h,
                            score_derivative_full=float(a[0]),score_derivative_half=float(b[0]),
                            constraint_relative_difference=float(np.linalg.norm(a[1:]-b[1:])/max(np.linalg.norm(a[1:]),1e-12))))
        write_json(stage/'finite_difference_radius_audit.json',radius_audit)
        for mode in ('geometry','dynamic'):
            md,transformed,audit=contextualize(original,episodes,mode)
            md['learning_method']='sampled_deterministic_complete_suffix_policy_gradient'
            models[mode]=md;torch.save(md,stage/(mode+'_initial_policy.pt'))
            write_json(stage/(mode+'_feature_identity.json'),audit)
            gradient=np.zeros(md['coefficients'].size);jacobian=np.zeros((len(zero_learning),len(gradient)))
            for e in transformed:
                for s in e['samples']:
                    d=np.array(s['design'])
                    for h in range(3):
                        derivative=response[e['path'],s['decision'],h]
                        contribution=s['query_weight']*d[:,h]
                        gradient[h::3]+=derivative[0]*contribution
                        jacobian[:,h::3]+=derivative[1:,None]*contribution[None,:]
            identity=SavedMemoryActionIdentity(md,replay,GateBandRuntime.decode_action)
            parent_matrix=[]
            for f,memory,_,_ in identity.paths.values():
                d=np.concatenate([f,memory[:,0],memory[:,1]],axis=1)*OUTPUT_SCALE
                parent_matrix.extend(design_matrix(d,exploration_sigma(md,d)).reshape(-1,len(gradient)))
            parent_matrix=np.array(parent_matrix);parent_matrix=parent_matrix[np.any(parent_matrix!=0.,axis=1)]
            np.savez_compressed(stage/(mode+'_complete_gradient.npz'),loss_gradient=gradient,
                constraint_jacobian=jacobian,baseline_margins=zero_learning,parent_matrix=parent_matrix)
            def entry(records,u,tag):
                first,measured,learning=metrics(records)
                return dict(tag=tag,context_arm=mode,coefficients=np.asarray(u).tolist(),records=records,score=score(records),
                    measured_inequalities=first.tolist(),augmented_measured_inequalities=measured.tolist(),
                    augmented_learning_inequalities=learning.tolist(),actual_rerun_paths=[],exact_zero_support_paths=paths)
            if not history:
                zero=entry(copy.deepcopy(selected),np.zeros(len(gradient)),'reused_parent');history.append(zero)
            def best():return min((c for c in history if min(c['augmented_measured_inequalities'])>=-1e-8),key=lambda c:c['score'])
            pool=mp.get_context('spawn').Pool(4,worker_initialize,(md,selected,folder,deadline))
            previous_vectors=[]
            for trust in (.05,.15):
                if time.perf_counter()>deadline:raise TimeoutError('Deterministic gradient optimization budget')
                norms=np.maximum(np.linalg.norm(jacobian,axis=1),1e-12)
                matrix=np.vstack([-jacobian/norms[:,None],parent_matrix,-parent_matrix])
                rhs=np.r_[zero_learning/norms,np.full(2*len(parent_matrix),trust)]
                solved=linprog(gradient/max(np.linalg.norm(gradient),1e-12),A_ub=matrix,b_ub=rhs,
                    bounds=[(-.05,.05)]*len(gradient),method='highs',
                    options=dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
                block=dict(context_arm=mode,trust=trust,solver_success=bool(solved.success),message=solved.message)
                if solved.success:
                    u=solved.x
                    block.update(predicted_loss_change=float(gradient@u),predicted_guard_min=float(np.min(zero_learning+jacobian@u)),
                        maximum_standardized_change=float(np.max(abs(parent_matrix@u))))
                    if gradient@u < -1e-5 and not any(np.array_equal(u,v) for v in previous_vectors):
                        previous_vectors.append(u.copy());tag=f'{mode}_trust{trust}'
                        actor=copy.deepcopy(md);actor['coefficients']=u.reshape(-1,3);torch.save(actor,stage/(tag+'.pt'))
                        unchanged=[p for p in paths if identity.unchanged(p,u)];actual=[p for p in paths if p not in unchanged]
                        records=pool.map(worker_evaluate,[(p,u,tag) for p in actual]);by={r['path']:r for r in records}
                        candidate=entry([by.get(r['path'],copy.deepcopy(r)) for r in selected],u,tag)
                        candidate.update(actual_rerun_paths=actual,exact_zero_support_paths=unchanged)
                        history.append(candidate)
                        block.update(tag=tag,actual_loss_change=candidate['score']-zero['score'],
                            actual_guard_min=min(candidate['augmented_measured_inequalities']))
                        write_json(folder/'results.json',dict(candidates=history,best=best()['tag']))
                blocks.append(block);write_json(stage/'deterministic_updates.json',blocks)
                print('DETERMINISTIC_JOINT_UPDATE',block,flush=True)
            pool.close();pool.join();pool=None
        chosen=best();md=models[chosen['context_arm']];torch.save(md,stage/'initial_policy.pt')
        finish(stage,md,baseline,data,chosen,zero,history,blocks,choice,parent_file,started)
        receipt(stage,dict(status='completed_deterministic_complete_credit_learning',
            deterministic_parent_replays=len(episodes),new_suffixes=sum(e['new_suffixes'] for e in episodes),
            reused_suffixes=sum(e['reused_suffixes'] for e in episodes),
            complete_temporal_support=args.complete_support,
            new_stochastic_episodes=0,ppo_updates=0,new_policies=len(history)-1,selected=chosen['tag'],
            research_acceptable=(stage/'research_candidate.pt').exists(),wall_s=time.perf_counter()-started,
            adopted=False,goal_complete=False,source_model_sha256=sha(parent_file)))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_deterministic_complete_credit_learning',error=str(exc),
            completed_paths=len(episodes),new_policies=max(0,len(history)-1),
            wall_s=time.perf_counter()-started,adopted=False,goal_complete=False))
        raise


if __name__=='__main__':
    mp.freeze_support();main()
