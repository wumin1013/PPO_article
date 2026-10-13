"""Matched temporal state representation learned from complete physical routes."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import copy
import time
import multiprocessing as mp
import numpy as np
import torch
from scipy.optimize import minimize
from observable_memory_joint_policy import ObservableMemoryJointPolicy
from joint_teacher_state_policy import features
from consolidate_joint_teacher_policy import collect_data
from learn_direct_observable_joint_policy import finish
from learn_shared_temporal_joint_policy import initialize,inequalities,evaluate_complete_parameters
from learn_complete_joint_teachers import teacher_score,parameters_in_bounds
from multi_corner_raw_metrics import augment,guards,objective
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


WORK={}


def worker_initialize(md,records,folder,deadline):
    torch.set_num_threads(1)
    WORK.update(policy=ObservableMemoryJointPolicy(md),records={r['path']:r for r in records},folder=folder,deadline=deadline)


def worker_evaluate(job):
    path,coefficients,tag=job
    return evaluate_complete_parameters(WORK['policy'],coefficients,[WORK['records'][path]],
        WORK['folder'],tag,WORK['deadline'])[0]


def metric_functions(data,selected,stage,parent,deadline):
    corners=[r for r in selected if 'shape' in r]
    retained=[next(o['baseline_shape'] for o in data['retained'] if o['path']==r['path']) for r in corners]
    initialize(stage,parent,corners,retained,deadline,False,None,None,False,{'early_state':True},[])
    def score(records):
        regular=[r for r,b in zip(records,selected) if 'shape' in b]
        return float(sum(teacher_score(r,b,o,barrier=False) for r,b,o in zip(regular,corners,retained))+
                     sum(objective(r,b) for r,b in zip(records,selected) if 'shape' not in b))
    def constraints(records):
        values=inequalities([r for r,b in zip(records,selected) if 'shape' in b]).tolist()
        for r,b in zip(records,selected):
            if 'shape' not in b:values.extend(guards(r,b))
        return np.asarray(values)
    return score,constraints


def main():
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_early_local_affine_joint_learning_20260920'
    coupled=BASE/'canonical_gate_band_early_affine_coupled_joint_learning_20260920'
    cache_stage=BASE/'canonical_gate_band_early_joint_exploration_restoration_20260920'
    assert read(coupled/'receipt.json')['status'].startswith('completed')
    assert not read(coupled/'receipt.json')['research_acceptable'],'Audit a successful coupled model first'
    parent_file=BASE/'canonical_gate_band_direct_observable_joint_learning_20260920/research_candidate.pt'
    assert sha(parent_file)=='9e37da543426532b1b6dd5c86f012c86ac8b304ba5dd76ca57f8f3fe8e64ca64'
    md=torch.load(source/'state_joint/candidate.pt',map_location='cpu',weights_only=False)
    for key in ('local_affine_context','context_scale','center_context'):md.pop(key,None)
    md.update(architecture='observable_memory_joint_radial',memory_decays=[.5,.8],coefficients=np.zeros((6,3)),
        learning_method='causal_observation_memory_complete_return_joint_optimization',accepted=False)
    data=read(source/'baseline.json');baseline=data['baseline']['records']
    support=read(source/'support_audit.json');choice=support['selected']
    selected=[r for r in baseline if r['path'] in choice['paths']]
    for r in selected:augment(r)
    stage=register('canonical_gate_band_early_observable_memory_joint_learning_20260920',
        [__file__,BASE/'observable_memory_joint_policy.py',BASE/'joint_teacher_state_policy.py',
         BASE/'deep_joint_credit_policy.py',BASE/'consolidate_joint_teacher_policy.py',
         BASE/'learn_direct_observable_joint_policy.py',BASE/'learn_shared_temporal_joint_policy.py',
         BASE/'learn_complete_joint_teachers.py',BASE/'multi_corner_raw_metrics.py',parent_file,
         source/'receipt.json',source/'early_support_response_audit.json',source/'replayed_states.npz',
         source/'state_joint/candidate.pt',cache_stage/'state_joint/results.json',coupled/'receipt.json'],
        dict(wall_s=600,workers=3,maximum_optimizer_calls=48,parameters=18,coefficient_bounds=[-1.,1.],
             fixed_memory_decays=[.5,.8],decision_ms=20,execution_ms=1,training_paths=choice['paths']),
        'Saved exact replay shows the instantaneous early kernel has vanished while wrong-way '
        'lateral movement continues, including the two lowest-loss measured state-affine candidates. '
        'Keep the same parent,two centers,width0.15,18 trainable coefficients,full24 affected routes, '
        'COBYLA and every original final protection. Replace the12 instantaneous speed/range '
        'slopes with12 learned three-action weights on TWO strictly causal observation traces. '
        'The traces use fixed decays0.5/0.8 per20ms step and only past kernel observations; '
        'these generic recurrence bases are not timed route switches,angle/speed tables or '
        'trajectory templates. Each route resets memory. Physical actions remain constant '
        'over20ms with1ms execution. Reuse all70 constant-policy vectors with zero memory weights '
        'after exact decoded-action identity on all88 parent routes and the saved initial point. '
        'Start from the measured low-violation constant point used by coupled affine learning. '
        'Jointly learn all18 weights for48 calls/600s including preparation; choose only a '
        'strictly improved complete candidate satisfying the existing shape/time/cruise/physical '
        'guards. If unsuccessful, analyze measured temporal action responses instead of extending '
        'the same run. This is closed-loop parameter learning, not PPO or proof of generalization.')
    started=time.perf_counter();deadline=started+580;folder=stage/'state_joint';folder.mkdir()
    write_json(stage/'baseline.json',data);write_json(stage/'support_audit.json',support)
    with np.load(source/'replayed_states.npz') as z:training={k:z[k].copy() for k in z.files}
    np.savez_compressed(stage/'replayed_states.npz',**training)
    f=features(md,training['observations'],training['innovations'])
    score,constraints=metric_functions(data,selected,stage,md['parent'],deadline)
    history=[];cache={};pool=None
    try:
        old=read(cache_stage/'state_joint/results.json')['candidates']
        for c in old:
            c=copy.deepcopy(c);c['coefficients']+= [0.]*12;c['tag']='reused_'+c['tag']
            for r in c['records']:augment(r)
            c['score']=score(c['records']);c['measured_inequalities']=constraints(c['records']).tolist()
            c['reused_full_physical_vector']=True
            history.append(c);cache[tuple(np.round(c['coefficients'],12))]=c
        zero=history[0];assert not np.any(zero['coefficients'])
        md['reused_candidates']=len(history)
        seed=read(coupled/'coupled_initial_point.json')['coefficients']
        assert not np.any(seed[6:]),'Reused seed must be the same instantaneous constant function'
        anchor=cache[tuple(np.round(seed,12))]
        replay=collect_data(md,md['parent'],baseline)
        assert not np.any(replay['labels'])
        extended=copy.deepcopy(md);extended['coefficients']=np.asarray(seed).reshape(-1,3)
        replay_seed=collect_data(extended,md['parent'],anchor['records'])
        write_json(stage/'memory_zero_extension_identity.json',dict(parent_paths=len(baseline),
            parent_decisions=len(replay['path_ids']),initial_point_paths=len(anchor['records']),
            initial_point_decisions=len(replay_seed['path_ids']),decoded_actions_bitwise=True,
            memory_reset_each_route=True,zero_memory_weights_reuse_constant_function=True,physical_reruns=0))
        print('MEMORY_IDENTITY',len(replay['path_ids']),len(replay_seed['path_ids']),'REUSED',len(history),'START',anchor['score'],flush=True)
        def feasible(c):return parameters_in_bounds(c['coefficients'],-1.,1.) and min(c['measured_inequalities'])>=-1e-8
        pool=mp.get_context('spawn').Pool(3,worker_initialize,(md,selected,folder,deadline))
        def evaluate(u):
            key=tuple(np.round(u,12))
            if key not in cache:
                tag=f'memory_{len(history)}';active=np.any(np.asarray(u).reshape(-1,2,3)!=0.,axis=(0,2))
                zero_paths=[r['path'] for r in selected if not np.any(f[training['path_ids']==r['path']][:,active,:])]
                measured=pool.map(worker_evaluate,[(r['path'],u,tag) for r in selected if r['path'] not in zero_paths])
                by_path={r['path']:r for r in measured}
                records=[by_path.get(r['path'],dict(copy.deepcopy(r),reused_exact_zero_support=True)) for r in selected]
                for r in records:augment(r)
                c=dict(tag=tag,coefficients=np.asarray(u).tolist(),records=records,score=score(records),
                    measured_inequalities=constraints(records).tolist(),exact_zero_support_paths=zero_paths,
                    actual_rerun_paths=[r['path'] for r in selected if r['path'] not in zero_paths])
                history.append(c);cache[key]=c
                best=min(filter(feasible,history),key=lambda c:c['score'])
                write_json(folder/'results.json',dict(candidates=history,best=best['tag']))
                if len(history)%5==0:print('MEMORY_JOINT',len(history),'best',best['score'],'latest',c['score'],min(c['measured_inequalities']),flush=True)
            return cache[key]
        solved=minimize(lambda u:evaluate(u)['score'],np.asarray(seed),method='COBYLA',bounds=[(-1.,1.)]*18,
            constraints=[dict(type='ineq',fun=lambda u:np.asarray(evaluate(u)['measured_inequalities']))],
            options=dict(maxiter=48,rhobeg=.05,tol=1e-6,catol=1e-8))
        pool.close();pool.join();pool=None
        best=min(filter(feasible,history),key=lambda c:c['score'])
        finish(stage,md,baseline,data,best,zero,history,[dict(solver=str(solved.message))],choice,parent_file,started)
    except TimeoutError:
        if pool is not None:pool.terminate();pool.join();pool=None
        best=min((c for c in history if parameters_in_bounds(c['coefficients'],-1.,1.) and min(c['measured_inequalities'])>=-1e-8),key=lambda c:c['score'])
        finish(stage,md,baseline,data,best,zero,history,[],choice,parent_file,started,budget_exhausted=True)
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_observable_memory_learning',error=str(exc),wall_s=time.perf_counter()-started,
            actual_new_candidates=max(0,len(history)-md.get('reused_candidates',0)),adopted=False,goal_complete=False));raise


if __name__=='__main__':mp.freeze_support();main()
