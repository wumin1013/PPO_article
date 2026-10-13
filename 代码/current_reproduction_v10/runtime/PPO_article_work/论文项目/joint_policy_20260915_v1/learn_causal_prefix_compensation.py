"""Learn three-action causal-prefix compensation from complete policy responses."""
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
from scipy.optimize import linprog
from joint_teacher_state_policy import features,state_pair,reference_oriented_observation,INDICES
from audit_saved_lateral_motion import lateral_motion
from consolidate_joint_teacher_policy import collect_data
from saved_memory_action_identity import SavedMemoryActionIdentity
from gate_band_state import GateBandRuntime
from joint_training_evaluator import build_evaluator
from learn_early_memory_joint_policy import worker_initialize,worker_evaluate
from learn_full_pool_joint_credit import aggregate
from multi_corner_raw_metrics import augment
from study_io import BASE,register,write_json,receipt,sha
from search_complete_return import read


SOURCE=BASE/'canonical_gate_band_action_capacity_joint_training_20260920'
NAME='canonical_gate_band_causal_prefix_joint_compensation_20260920'


def prepare():
    data=read(SOURCE/'baseline.json');baseline=data['baseline']['records']
    history=read(SOURCE/'state_joint/results.json')['candidates']
    seed=min((c for c in history if c.get('arm')=='wide' and c['physical_pass']),key=lambda c:c['training_score'])
    parent=torch.load(SOURCE/'initial_policy.pt',map_location='cpu',weights_only=False)
    parent['coefficients']=np.array(seed['coefficients']).reshape(-1,3)
    by={r['path']:r for r in seed['records']};current=[copy.deepcopy(by.get(r['path'],r)) for r in baseline]
    observations=[];path_ids=[];decisions=[];selectors=[]
    for r in current:
        with np.load(r['trace']) as z:obs=z['observations'].copy()
        observations.extend(obs);path_ids.extend([r['path']]*len(obs));decisions.extend(range(len(obs)))
        if r['path'] in (105,204):
            first=lateral_motion(r['trace'])['intervals'][0]['start_ms']
            selectors.append(dict(path=r['path'],decision=max(0,first//20-1)))
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
        action_interval_ms=20,physical_tick_ms=1,learning_method='causal_prefix_full_response_joint_compensation')
    choices=[]
    for width in (.05,.1,.2,.3):
        md['radii']=np.full(k,width);active=np.any(features(md,obs,innovation)!=0.,axis=(1,2))
        paths=sorted(np.unique(ids[active]).tolist())
        choices.append(dict(width=width,paths=paths,decisions=int(active.sum()),
                            nonisolated=[r['path'] for r in baseline if r['path'] in paths and 'shape' not in r]))
    md['radii']=np.full(k,.1)
    support=dict(selected=next(c for c in choices if c['width']==.1),choices=choices,
        selectors_for_training_only=selectors,all_training_paths=len(baseline),
        outside_support_exact_zero=True,no_runtime_time_or_path_inputs=True,
        inherited_changed_paths=[r['path'] for r in seed['records']])
    return data,md,support,current,seed


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1);data,md,support,current,seed=prepare()
    print('CAUSAL_PREFIX_SUPPORT',support,flush=True)
    if args.prepare:return
    started=time.perf_counter();deadline=started+760.
    baseline=data['baseline']['records'];paths=sorted(set(support['selected']['paths'])|set(support['inherited_changed_paths']))
    selected=[r for r in baseline if r['path'] in paths];current_selected=[r for r in current if r['path'] in paths]
    for r in selected+current_selected:augment(r)
    stage=register(NAME,[__file__,SOURCE/'receipt.json',SOURCE/'initial_policy.pt',SOURCE/'baseline.json',
        SOURCE/'state_joint/results.json',SOURCE/'causal_support_audit.json',SOURCE/'capacity_result_audit.json']+
        [BASE/f for f in ('joint_training_evaluator.py','multi_corner_raw_metrics.py',
            'observable_memory_joint_policy.py','saved_memory_action_identity.py','learn_early_memory_joint_policy.py')],
        dict(wall_s=780,workers=4,trainable_parameters=18,new_response_policies=36,
            maximum_new_proposals=8,response_steps=[.02,.04,.04],coefficient_bound=3.,
            response_radii=[4.,16.,64.],early_support_radius=.1,ppo_updates=0),
        'The completed capacity experiment can reduce exit excursion but loses curvature/speed; '
        'its entire incoming wrong-way prefix occurs BEFORE the first nonzero new feature. '
        'Add two observed-state centers one decision before the FIRST significant reverse '
        'lateral event on105/204, with the same causal memories and three learned action heads. '
        'Start a training-only recovery branch at the measured lowest-cost physically valid '
        'wide candidate; keep its later learned layer fixed in this bounded experiment. '
        'This isolates whether newly enabled upstream joint control can repair that actual '
        'tradeoff. Scan all88 actual seed trajectories. Radius0.1 also affects multicorner31: '
        'retain its existing raw ordered-window guards and add nondegrading complete time and '
        'cruise guards; never omit it to narrow evaluation. Joint18-parameter symmetric complete '
        'responses produce measured credit for every constraint, then compare single-head '
        'and joint slack-restoration proposals using the SAME data. Slack is a training '
        'objective only: final adoption still requires every original measured guard against '
        'the protected ded2 model, full physical success and a strict cost improvement. '
        'Allow four measured secant corrections only when the actual violation decreases. '
        '780s,36 response policies plus at most8 proposals. No route/time inputs or fixed '
        'trajectory templates enter deployment. Parameter optimization,not PPO. If upstream '
        'control fails to repair the tradeoff, use measured responses to decide on jointly '
        'releasing early/late parameters or a new representation; do not extend this branch.')
    folder=stage/'state_joint';folder.mkdir()
    support['evaluation_paths']=paths
    write_json(stage/'baseline.json',data);write_json(stage/'support_audit.json',support)
    write_json(stage/'seed.json',dict(tag=seed['tag'],records=current,source_results_sha256=sha(SOURCE/'state_joint/results.json')))
    torch.save(md,stage/'initial_policy.pt')
    write_json(stage/'selection_hold.json',dict(reason='Training-only causal-support recovery; protected actors remain unchanged.'))
    replay=collect_data(md,md['parent'],current_selected,include_latents=True,deadline=deadline)
    replay['held_actions']=np.vstack([np.load(r['trace'])['actions'] for r in current_selected])
    np.savez_compressed(stage/'affected_replay.npz',**replay)
    identity=SavedMemoryActionIdentity(md,replay,GateBandRuntime.decode_action)
    assert all(identity.unchanged(p,np.zeros(18)) for p in paths)
    write_json(stage/'initial_identity.json',dict(zero_decoded_identity=True,decisions=len(replay['decisions']),
        all_training_paths_scanned=88,evaluation_paths=paths,
        exact_inherited_paths=[r['path'] for r in current if r['path'] not in paths]))
    entry=build_evaluator(data,selected,stage,md['parent'],deadline)
    protected=entry(copy.deepcopy(selected),np.zeros(18),'protected_reference')
    assert protected['physical_pass'] and min(protected['augmented_measured_inequalities'])>=-1e-8
    zero=entry(copy.deepcopy(current_selected),np.zeros(18),'reused_wide_seed')
    zero.update(actual_rerun_paths=[],exact_zero_support_paths=paths)
    normalizer=protected['training_score'];history=[zero];cache={tuple(np.zeros(18)):zero};blocks=[];pool=None
    def violation(c):
        v=np.maximum(-np.array(c['augmented_measured_inequalities']),0.)
        return float(v.sum()),float(v.max())
    def acceptable(c):
        return c['physical_pass'] and violation(c)[1]<=1e-8 and c['training_score']<normalizer-1e-6
    def evaluate(u,tag):
        key=tuple(np.asarray(u,dtype=float).tolist())
        if key in cache:return cache[key]
        assert np.max(abs(u))<=3.+1e-8
        if time.perf_counter()>deadline-15:raise TimeoutError('Causal-prefix response budget')
        unchanged=[p for p in paths if identity.unchanged(p,u)];actual=[p for p in paths if p not in unchanged]
        rows=pool.map(worker_evaluate,[(p,u,tag) for p in actual]);by={r['path']:r for r in rows}
        c=entry([by.get(r['path'],copy.deepcopy(r)) for r in current_selected],u,tag)
        c.update(actual_rerun_paths=actual,exact_zero_support_paths=unchanged)
        history.append(c);cache[key]=c
        write_json(folder/'results.json',dict(candidates=history,best=None))
        if len(history)%6==1 or not tag.startswith('response'):
            print('PREFIX_JOINT',tag,c['training_score'],violation(c),c['physical_pass'],flush=True)
        return c
    def solve(anchor,jac,gradient,indices,radius):
        values=np.array(anchor['augmented_learning_inequalities']);u=np.array(anchor['coefficients'])
        m=len(values);d=len(indices);a=jac[:,indices]
        bounds=[(max(-radius,(-3-u[i])/steps[i]),min(radius,(3-u[i])/steps[i])) for i in indices]+[(0,None)]*m
        constraints=np.hstack([-a,-np.eye(m)])
        first=linprog(np.r_[np.zeros(d),np.ones(m)],A_ub=constraints,b_ub=values,bounds=bounds,method='highs')
        if not first.success:return dict(success=False,message=first.message)
        # Secondary full-return descent may not increase the minimum total model slack.
        second=linprog(np.r_[gradient[indices]/normalizer,np.zeros(m)],
            A_ub=np.vstack([constraints,np.r_[np.zeros(d),np.ones(m)]]),
            b_ub=np.r_[values,first.fun+1e-8],bounds=bounds,method='highs')
        point=second.x if second.success else first.x;delta=np.zeros(18);delta[indices]=point[:d]
        return dict(success=True,scaled_step=delta.tolist(),predicted_slack=float(first.fun),
            predicted_cost_change=float(gradient@delta),radius=radius)
    try:
        pool=mp.get_context('spawn').Pool(4,worker_initialize,(md,current_selected,folder,deadline))
        steps=np.tile([.02,.04,.04],6);pairs=[]
        for j in range(18):
            delta=np.eye(18)[j]*steps
            pairs.append((evaluate(delta,f'response_{j}_plus'),evaluate(-delta,f'response_{j}_minus')))
        jac=np.array([(np.array(a['augmented_learning_inequalities'])-b['augmented_learning_inequalities'])/2 for a,b in pairs]).T
        gradient=np.array([(a['training_score']-b['training_score'])/2 for a,b in pairs])
        write_json(stage/'measured_prefix_credit.json',dict(steps=steps.tolist(),jacobian=jac.tolist(),
            loss_gradient=gradient.tolist(),response_pairs=[[a['tag'],b['tag']] for a,b in pairs],
            constraint_count=len(zero['augmented_measured_inequalities'])))
        joint=[]
        for head,name in ((0,'steering'),(1,'feed'),(2,'preview'),(None,'joint')):
            ids=np.arange(18) if head is None else np.arange(head,18,3)
            solutions=[solve(zero,jac,gradient,ids,r) for r in (4.,16.,64.)]
            valid=[s for s in solutions if s['success']]
            block=dict(arm=name,solutions=solutions)
            if valid:
                s=min(valid,key=lambda s:(s['predicted_slack'],s['predicted_cost_change']))
                c=evaluate(np.array(s['scaled_step'])*steps,name+'_proposal')
                block.update(measured=c['tag'],actual_violation=violation(c),actual_score=c['training_score'])
                if head is None:joint.append(c)
            blocks.append(block);write_json(stage/'optimization_blocks.json',blocks)
        if joint:
            anchor=joint[0]
            for iteration in range(4):
                if acceptable(anchor):break
                solutions=[solve(anchor,jac,gradient,np.arange(18),r) for r in (1.,4.,16.)]
                valid=[s for s in solutions if s['success']]
                if not valid:break
                s=min(valid,key=lambda s:(s['predicted_slack'],s['predicted_cost_change']))
                delta=np.array(s['scaled_step']);norm=float(delta@delta)
                if norm<1e-18:break
                c=evaluate(np.array(anchor['coefficients'])+delta*steps,f'joint_secant_{iteration}')
                observed=np.array(c['augmented_learning_inequalities'])-anchor['augmented_learning_inequalities']
                jac+=np.outer(observed-jac@delta,delta)/norm
                gradient+=(c['training_score']-anchor['training_score']-gradient@delta)*delta/norm
                blocks.append(dict(arm='joint_secant',iteration=iteration,anchor=anchor['tag'],solution=s,
                    measured=c['tag'],actual_violation=violation(c),actual_score=c['training_score']))
                write_json(stage/'optimization_blocks.json',blocks)
                if not c['physical_pass'] or violation(c)>=violation(anchor):break
                anchor=c
        pool.close();pool.join();pool=None
        passing=list(filter(acceptable,history));safe=[c for c in history if c['physical_pass']]
        chosen=min(passing,key=lambda c:c['training_score']) if passing else min(safe,key=lambda c:(violation(c),c['training_score']))
        actor=copy.deepcopy(md);actor.update(coefficients=np.array(chosen['coefficients']).reshape(-1,3),
            accepted=False,training_branch_only=True)
        torch.save(actor,stage/'shadow_training_policy.pt')
        by={r['path']:r for r in chosen['records']};full=[copy.deepcopy(by.get(r['path'],r)) for r in current]
        write_json(stage/'shadow_training_result.json',dict(tag=chosen['tag'],records=full,summary=aggregate(full),
            training_score=chosen['training_score'],original_all_guard_pass=bool(passing),
            violation=violation(chosen),adopted=False))
        receipt(stage,dict(status='completed_causal_prefix_joint_compensation',new_policies=len(history)-1,
            ppo_updates=0,research_acceptable=False,pending_candidate_audit=bool(passing),
            original_guard_passing_new_policies=len(passing),training_best=chosen['tag'],
            shadow_sha256=sha(stage/'shadow_training_policy.pt'),wall_s=time.perf_counter()-started,
            adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_causal_prefix_joint_compensation',error=str(exc),
            new_policies=len(history)-1,wall_s=time.perf_counter()-started,adopted=False,goal_complete=False));raise


if __name__=='__main__':mp.freeze_support();main()
