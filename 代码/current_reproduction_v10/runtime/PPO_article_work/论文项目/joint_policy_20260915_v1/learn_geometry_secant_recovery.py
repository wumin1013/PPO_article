"""Correct measured joint geometry credit with bounded complete-loop secants."""
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
from audit_context_restoration_scaling import solve_correction
from duration_joint_metrics import guards, penalty
from learn_complete_joint_teachers import teacher_score, parameters_in_bounds
from learn_early_memory_joint_policy import worker_initialize, worker_evaluate, metric_functions
from learn_direct_observable_joint_policy import finish
from study_io import BASE, register, receipt, write_json
from search_complete_return import read
from heading_duration_credit import learning_guards


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--signed-heading',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1)
    source=BASE/('canonical_gate_band_geometry_secant_recovery_learning_20260920' if args.signed_heading else
                 'canonical_gate_band_geometry_conditioned_joint_learning_20260920')
    response=BASE/'canonical_gate_band_conditioned_main_turn_joint_learning_20260920'
    assert not read(source/'receipt.json')['research_acceptable']
    model_file=source/('signed_heading_credit_audit.json' if args.signed_heading else 'measured_intercept_restoration_audit.json')
    model=read(model_file)
    assert model['models'][-1]['success']
    md=torch.load(source/'state_joint/candidate.pt',map_location='cpu',weights_only=False)
    parent_file=BASE/'canonical_gate_band_postpeak_memory_joint_learning_20260920/research_candidate.pt'
    history=copy.deepcopy(read(source/'state_joint/results.json')['candidates'])
    zero,early=history[:2];current=next(c for c in history if c['tag']==model['anchor'])
    md['reused_candidates']=len(history);radius=1. if args.signed_heading else 4.
    stage=register('canonical_gate_band_signed_heading_joint_recovery_20260920' if args.signed_heading else
                   'canonical_gate_band_geometry_secant_recovery_learning_20260920',
        [__file__,BASE/'observable_memory_joint_policy.py',BASE/'duration_joint_metrics.py',
         BASE/'learn_early_memory_joint_policy.py',BASE/'learn_shared_temporal_joint_policy.py',
         BASE/'learn_direct_observable_joint_policy.py',BASE/'audit_context_restoration_scaling.py',
         parent_file,source/'receipt.json',source/'state_joint/candidate.pt',source/'state_joint/results.json',
         model_file,response/'joint_main_turn_credit_models.json']+
         ([BASE/'heading_duration_credit.py',BASE/'audit_signed_heading_credit.py'] if args.signed_heading else [source/'joint_guard_failures.json']),
        dict(wall_s=180,workers=4,maximum_measured_corrections=6,maximum_new_vectors=8,
             maximum_backtracks=2,local_scaled_radius=radius,coefficient_bounds=[-1.,1.],signed_heading_credit=args.signed_heading),
        ('The8 measured secants eliminated the extra104 heading millisecond and '
         'reduced actual violation norm from0.0503 to0.00209. Replace ONLY the '
         'optimizer heading-duration representation by its EXACT order-statistic '
         'slack: count(heading>limit)<=N iff the(N+1)-th largest heading<=limit. '
         'The equivalence was checked in1118 stored record comparisons. Fit heading '
         'credit with existing symmetric responses and actual joint secants, using '
         'no new axis trajectories. Keep original discrete counts in final candidate '
         'selection. Start at the latest actual candidate and use radius1, the '
         'smallest audited feasible correction radius. At most8 full-loop vectors '
         'and180s; no acceptance tolerance is relaxed. '
         if args.signed_heading else '')+
        'The earlier geometry-conditioned full candidate reduced aggregate reversal,excursion '
        'and lateral motion, but28 protected rows failed, including one extra '
        'millisecond of104 heading overshoot. Its original zero-intercept credit '
        'does not predict these actual errors. Use the audited row-scaled '
        'MEASURED-intercept correction and reuse all existing vectors with the '
        '252-dimensional response model; '
        'validate up to6 corrections and2 half-step backtracks. Update the '
        'constraint and loss Jacobians by the actual joint secant after each new '
        'measurement. This changes the local credit approximation, not any real '
        'acceptance bound. Keep early coefficients and actual prefix fixed. '
        'Only fully protected complete outcomes are selectable; otherwise retain '
        'the original research parent. Parameter learning, not PPO or bias proof.')
    started=time.perf_counter();deadline=started+160;folder=stage/'state_joint';folder.mkdir()
    data=read(source/'baseline.json');baseline=data['baseline']['records']
    support=read(source/'support_audit.json');choice=support['selected'];paths=choice['paths']
    selected=[r for r in baseline if r['path'] in paths]
    write_json(stage/'baseline.json',data);write_json(stage/'support_audit.json',support)
    with np.load(source/'replayed_states.npz') as z:np.savez_compressed(stage/'replayed_states.npz',**{k:z[k] for k in z.files})
    _,old_guards=metric_functions(data,selected,stage,md['parent'],deadline)
    retained=[next(r['baseline_shape'] for r in data['retained'] if r['path']==b['path']) for b in selected]
    def score(records):return float(sum(teacher_score(r,b,o,barrier=False)+penalty(r) for r,b,o in zip(records,selected,retained)))
    def constraints(records):return np.concatenate([old_guards(records)]+[guards(r,b) for r,b in zip(records,selected)])
    for c in history:
        assert abs(c['score']-score(c['records']))<1e-10
        assert np.array_equal(c['measured_inequalities'],constraints(c['records']))
        c['learning_inequalities']=(learning_guards(c['records'],selected,c['measured_inequalities']).tolist()
                                    if args.signed_heading else c['measured_inequalities'])
    cache={tuple(np.round(c['coefficients'],12)):c for c in history}
    def feasible(c):return parameters_in_bounds(c['coefficients'],-1.,1.) and min(c['measured_inequalities'])>=-1e-8
    def best():return min(filter(feasible,history),key=lambda c:c['score'])
    def violation(c):return float(np.linalg.norm(np.minimum(c['measured_inequalities'],0.)))
    steps=np.asarray(read(response/'joint_main_turn_credit_models.json')['steps'])
    steps=np.repeat(steps[:,None],7,axis=1).reshape(-1)
    def expand(z):
        v=np.asarray(z).reshape(3,7,4,3);out=np.zeros((3,7,5,3))
        for j,k in enumerate((0,1,3,4)):out[:,:,k]=v[:,:,j]
        return out.reshape(-1)
    jac=np.asarray(model['jacobian']);gradient=np.asarray(model['loss_gradient'])
    prefix_stage=BASE/'canonical_gate_band_approach_conditioned_recovery_learning_20260920'
    prefix={r['path']:r['first_later_kernel_ms'] for r in read(prefix_stage/'conditioned_support_audit.json')['records']}
    pool=None;blocks=[];backtracks=0
    try:
        pool=mp.get_context('spawn').Pool(4,worker_initialize,(md,selected,folder,deadline))
        def evaluate(u):
            key=tuple(np.round(u,12))
            if key not in cache:
                assert len(history)-md['reused_candidates']<8
                assert parameters_in_bounds(u,-1.,1.)
                tag=f'geometry_secant_{len(history)}';records=pool.map(worker_evaluate,[(p,u,tag) for p in paths])
                for r,b in zip(records,early['records']):
                    with np.load(r['trace']) as z:actual=z['rows'][:prefix[r['path']]].copy()
                    with np.load(b['trace']) as z:reference=z['rows'][:prefix[r['path']]].copy()
                    assert np.array_equal(actual,reference),('early_prefix_changed',tag,r['path'])
                c=dict(tag=tag,coefficients=np.asarray(u).tolist(),records=records,score=score(records),
                    measured_inequalities=constraints(records).tolist(),actual_rerun_paths=paths,
                    exact_zero_support_paths=[],early_prefix_bitwise=True)
                c['learning_inequalities']=(learning_guards(records,selected,c['measured_inequalities']).tolist()
                                            if args.signed_heading else c['measured_inequalities'])
                history.append(c);cache[key]=c;write_json(folder/'results.json',dict(candidates=history,best=best()['tag']))
                print('GEOMETRY_SECANT',tag,c['score'],min(c['measured_inequalities']),violation(c),flush=True)
            return cache[key]
        print('GEOMETRY_SECANT_START',current['tag'],violation(current),flush=True)
        for iteration in range(6):
            if len(history)-md['reused_candidates']>=8:break
            values=np.asarray(current['learning_inequalities'])
            solved=solve_correction(jac,gradient,values,early['score']-current['score'],radius=radius)
            row=dict(iteration=iteration,anchor=current['tag'],**solved)
            if not solved['success']:
                blocks.append(row);break
            delta=np.asarray(solved['scaled_parameter_step']);u=np.asarray(current['coefficients'])
            proposal=evaluate(u+expand(steps*delta))
            row.update(tag=proposal['tag'],actual_minimum=min(proposal['measured_inequalities']),actual_violation=violation(proposal))
            actual=np.asarray(proposal['learning_inequalities']);denom=float(delta@delta)
            assert denom>1e-20
            jac+=np.outer(actual-values-jac@delta,delta)/denom
            gradient+=(proposal['score']-current['score']-gradient@delta)*delta/denom
            row['joint_secant_constraint_residual']=float(np.max(abs(jac@delta-(actual-values))))
            blocks.append(row);write_json(stage/'optimization_blocks.json',blocks)
            if feasible(proposal):break
            if violation(proposal)>=violation(current) and backtracks<2 and len(history)-md['reused_candidates']<8:
                half=evaluate(u+expand(steps*delta*.5));backtracks+=1
                blocks.append(dict(method='half_step_control',tag=half['tag'],actual_violation=violation(half)))
                if violation(half)<violation(proposal):proposal=half
                if feasible(proposal):break
            # The new measured secant may restore a direction even if the first proposal failed.
            if violation(proposal)<violation(current):current=proposal
        write_json(stage/'final_secant_credit.json',dict(jacobian=jac.tolist(),loss_gradient=gradient.tolist(),
            corrections=blocks,actual_new_vectors=len(history)-md['reused_candidates']))
        pool.close();pool.join();pool=None
        finish(stage,md,baseline,data,best(),zero,history,blocks,choice,parent_file,started)
    except TimeoutError:
        if pool is not None:pool.terminate();pool.join();pool=None
        finish(stage,md,baseline,data,best(),zero,history,blocks,choice,parent_file,started,budget_exhausted=True)
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_geometry_secant_recovery',error=str(exc),wall_s=time.perf_counter()-started,
            adopted=False,goal_complete=False));raise


if __name__=='__main__':mp.freeze_support();main()
