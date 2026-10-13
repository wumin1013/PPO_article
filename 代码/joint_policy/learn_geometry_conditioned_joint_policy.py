"""Physically validate learned geometry/motion/history coordination on full routes."""
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
from consolidate_joint_teacher_policy import collect_data
from duration_joint_metrics import guards, penalty
from learn_complete_joint_teachers import teacher_score, parameters_in_bounds
from learn_early_memory_joint_policy import worker_initialize, worker_evaluate, metric_functions
from learn_direct_observable_joint_policy import finish
from search_complete_return import read
from study_io import BASE, register, receipt, write_json


def main():
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_conditioned_main_turn_joint_learning_20260920'
    assert not read(source/'receipt.json')['research_acceptable']
    audit=read(source/'geometry_context_capacity.json')
    model=next(r for r in audit['models'] if r['context_count']==7 and r['radius']==4.)
    assert model['success'] and model['predicted_guard_minimum']>=-1e-8
    parent_file=BASE/'canonical_gate_band_postpeak_memory_joint_learning_20260920/research_candidate.pt'
    md=torch.load(source/'state_joint/candidate.pt',map_location='cpu',weights_only=False)
    md.update(geometry_context_terms=1,per_center_geometry_mean=np.asarray(audit['per_center_geometry_mean']),
        per_center_geometry_scale=np.asarray(audit['per_center_geometry_scale']),coefficients=np.zeros((105,3)),
        learning_method='geometry_motion_context_complete_three_action_learning')
    old=read(source/'state_joint/results.json')['candidates'];old_anchor=old[1]
    early=np.zeros((3,7,5,3));early[:,:6]=np.asarray(old_anchor['coefficients']).reshape(3,6,5,3)
    early=early.reshape(-1)
    steps=np.asarray(read(source/'joint_main_turn_credit_models.json')['steps'])
    proposed=np.asarray(model['scaled_proposal']).reshape(3,7,4,3)*steps[:,None]
    joint=np.zeros((3,7,5,3))
    for j,k in enumerate((0,1,3,4)):joint[:,:,k]=proposed[:,:,j]
    stage=register('canonical_gate_band_geometry_conditioned_joint_learning_20260920',
        [__file__,BASE/'observable_memory_joint_policy.py',BASE/'joint_teacher_state_policy.py',
         BASE/'duration_joint_metrics.py',BASE/'learn_early_memory_joint_policy.py',
         BASE/'learn_shared_temporal_joint_policy.py',BASE/'learn_direct_observable_joint_policy.py',
         BASE/'consolidate_joint_teacher_policy.py',parent_file,source/'receipt.json',
         source/'geometry_context_capacity.json',source/'shared_trust_region_audit.json',
         source/'conditioned_route_capacity.json',source/'joint_main_turn_credit_models.json',
         source/'state_joint/candidate.pt',source/'state_joint/results.json'],
        dict(wall_s=600,workers=4,geometry_context_terms=1,parameters=315,maximum_control_vectors=5,
             maximum_amplitude_optimizer_calls=60,amplitude_dimensions=24,coefficient_bounds=[-1.,1.]),
        'Main-turn response restores individual-route local capacity, but shared '
        'motion-only models remain infeasible even at fourfold radius. Using the '
        'SAME measured responses, one continuous visible-reference-bend input makes '
        'the shared model feasible. Append this reflection-even observation to '
        'the causal context memory; jointly learn all three action weights. No '
        'angle-speed lookup, path ID, time switch or trajectory label is introduced. '
        'Verify exact old-function embedding, then test the full,half,quarter '
        'learned correction, the matched geometry-input removal, and main-turn '
        'component removal. All are full13-route feedback; ablations are attribution '
        'controls only. If measured score and violation norm both improve over '
        'the early-gain anchor, learn24 phase/action/context amplitude groups for '
        'at most60 complete-loop calls. Only original fully protected actual '
        'outcomes can be selected. Preserve the early improvement prefix and '
        'every physical/shape/duration/speed/cruise criterion. Parameter learning, '
        'not PPO, final shape acceptance or generalization proof.')
    started=time.perf_counter();deadline=started+580;folder=stage/'state_joint';folder.mkdir()
    data=read(source/'baseline.json');baseline=data['baseline']['records']
    support=read(source/'support_audit.json');choice=support['selected'];paths=choice['paths']
    selected=[r for r in baseline if r['path'] in paths]
    write_json(stage/'baseline.json',data);write_json(stage/'support_audit.json',support)
    with np.load(source/'replayed_states.npz') as z:np.savez_compressed(stage/'replayed_states.npz',**{k:z[k] for k in z.files})
    _,old_guards=metric_functions(data,selected,stage,md['parent'],deadline)
    retained=[next(r['baseline_shape'] for r in data['retained'] if r['path']==b['path']) for b in selected]
    def score(records):return float(sum(teacher_score(r,b,o,barrier=False)+penalty(r) for r,b,o in zip(records,selected,retained)))
    def constraints(records):return np.concatenate([old_guards(records)]+[guards(r,b) for r,b in zip(records,selected)])
    zero=dict(tag='reused_identity',coefficients=[0.]*315,records=selected,score=score(selected),measured_inequalities=constraints(selected).tolist())
    anchor=copy.deepcopy(old_anchor);anchor['coefficients']=early.tolist()
    history=[zero,anchor];md['reused_candidates']=2;cache={tuple(np.round(c['coefficients'],12)):c for c in history}
    def feasible(c):return parameters_in_bounds(c['coefficients'],-1.,1.) and min(c['measured_inequalities'])>=-1e-8
    def best():return min(filter(feasible,history),key=lambda c:c['score'])
    def violation(c):return float(np.linalg.norm(np.minimum(c['measured_inequalities'],0.)))
    pool=None;blocks=[]
    try:
        checked=copy.deepcopy(md);checked['coefficients']=early.reshape(-1,3)
        replay=collect_data(checked,md['parent'],anchor['records'])
        write_json(stage/'geometry_zero_extension_identity.json',dict(decoded_action_identity_decisions=len(replay['decisions']),
            paths=paths,old_coefficients_preserved=True,new_physical_runs=0))
        print('GEOMETRY_JOINT_START','predicted',model['predicted_loss'],'anchor_violation',violation(anchor),flush=True)
        pool=mp.get_context('spawn').Pool(4,worker_initialize,(md,selected,folder,deadline))
        prefix_stage=BASE/'canonical_gate_band_approach_conditioned_recovery_learning_20260920'
        prefix={r['path']:r['first_later_kernel_ms'] for r in read(prefix_stage/'conditioned_support_audit.json')['records']}
        def evaluate(u):
            u=np.asarray(u).reshape(-1);key=tuple(np.round(u,12))
            if key not in cache:
                assert parameters_in_bounds(u,-1.,1.)
                assert np.array_equal(u.reshape(3,7,5,3)[:,:,2],early.reshape(3,7,5,3)[:,:,2])
                tag=f'geometry_joint_{len(history)}';records=pool.map(worker_evaluate,[(p,u,tag) for p in paths])
                for r,b in zip(records,anchor['records']):
                    with np.load(r['trace']) as z:actual=z['rows'][:prefix[r['path']]].copy()
                    with np.load(b['trace']) as z:reference=z['rows'][:prefix[r['path']]].copy()
                    assert np.array_equal(actual,reference),('early_prefix_changed',tag,r['path'])
                c=dict(tag=tag,coefficients=u.tolist(),records=records,score=score(records),measured_inequalities=constraints(records).tolist(),
                    actual_rerun_paths=paths,exact_zero_support_paths=[],early_prefix_bitwise=True)
                history.append(c);cache[key]=c;write_json(folder/'results.json',dict(candidates=history,best=best()['tag']))
                print('GEOMETRY_JOINT',tag,c['score'],min(c['measured_inequalities']),violation(c),flush=True)
            return cache[key]
        no_geometry=joint.copy();no_geometry[:,6]=0.
        no_main=joint.copy();no_main[:,:,3:]=0.
        for label,delta in [('joint',joint),('joint_half',joint*.5),('joint_quarter',joint*.25),
                            ('without_geometry',no_geometry),('without_main_turn',no_main)]:
            c=evaluate(early+delta.reshape(-1))
            blocks.append(dict(method='matched_geometry_state_control',arm=label,tag=c['tag'],score=c['score'],
                guard_minimum=min(c['measured_inequalities']),violation_norm=violation(c),early_prefix_bitwise=True))
        useful=[c for c in history[2:] if c['score']<anchor['score'] and violation(c)<violation(anchor)]
        if useful:
            start=min(useful,key=violation);u0=np.asarray(start['coefficients']);vectors=[]
            for k in (0,1,3,4):
                for group in (slice(0,6),slice(6,7)):
                    for h in range(3):
                        v=np.zeros_like(joint);v[:,group,k,h]=joint[:,group,k,h];vectors.append(v.reshape(-1))
            basis=np.asarray(vectors).T;basis=basis[:,np.any(basis!=0.,axis=0)]
            result=minimize(lambda z:evaluate(u0+basis@z)['score'],np.zeros(basis.shape[1]),method='COBYLA',
                bounds=[(-1.,1.)]*basis.shape[1],constraints=[dict(type='ineq',fun=lambda z:np.asarray(evaluate(u0+basis@z)['measured_inequalities']))],
                options=dict(maxiter=60,rhobeg=.25,tol=1e-6,catol=1e-8))
            blocks.append(dict(method='actual_geometry_motion_amplitude_learning',start=start['tag'],dimensions=basis.shape[1],solver=str(result.message)))
        pool.close();pool.join();pool=None
        finish(stage,md,baseline,data,best(),zero,history,blocks,choice,parent_file,started)
    except TimeoutError:
        if pool is not None:pool.terminate();pool.join();pool=None
        finish(stage,md,baseline,data,best(),zero,history,blocks,choice,parent_file,started,budget_exhausted=True)
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_geometry_conditioned_joint_learning',error=str(exc),wall_s=time.perf_counter()-started,
            adopted=False,goal_complete=False));raise


if __name__=='__main__':mp.freeze_support();main()
