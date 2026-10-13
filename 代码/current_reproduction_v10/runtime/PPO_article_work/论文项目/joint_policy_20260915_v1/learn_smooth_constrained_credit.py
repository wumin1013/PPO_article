"""Matched continuous-objective credit with identical hard feasibility guards."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import copy
import time
import numpy as np
import torch
from scipy.optimize import linprog
from learn_fine_innovation_credit import setup,evaluate,EPSILON
from learn_complete_joint_teachers import teacher_score,parameters_in_bounds
from study_io import BASE,register,receipt,write_json,sha
from search_complete_return import read


def main():
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_independent_command_credit_learning_20260920'
    data=read(source/'baseline.json');original=read(source/'innovation_coordinates/results.json')['candidates']
    outcomes=copy.deepcopy(original)
    zero=next(c for c in outcomes if not np.any(c['coefficients']))
    baseline=zero['records'];paths=[r['path'] for r in baseline]
    retained=[next(o['baseline_shape'] for o in data['retained'] if o['path']==p) for p in paths]
    def score(records):
        return sum(teacher_score(r,b,o,barrier=False) for r,b,o in zip(records,baseline,retained))
    clipped=[]
    for c in outcomes:
        c['original_barrier_score']=c['score'];c['score']=float(score(c['records']))
        if abs(c['score']-c['original_barrier_score'])>1e-8:clipped.append(c['tag'])
        if min(c['measured_inequalities'])>=-1e-8:assert c['score']==c['original_barrier_score']
    nominal=np.r_[zero['score'],zero['measured_inequalities']]
    def vector(c):return np.r_[c['score'],c['measured_inequalities']]
    jacobian=np.asarray([(vector(next(c for c in outcomes if c['tag']==f'axis_{i}_+1'))-
                          vector(next(c for c in outcomes if c['tag']==f'axis_{i}_-1')))/(2*EPSILON) for i in range(36)])
    previous=read(source/'local_response_audit.json');old_j=np.asarray(previous['jacobian'])
    assert np.array_equal(jacobian[:,1:],old_j[:,1:])
    margin=np.asarray(previous['inner_prediction_margin'])
    solve=linprog(jacobian[:,0],A_ub=-jacobian[:,1:].T,b_ub=nominal[1:]-margin+1e-10,
                  bounds=[(-EPSILON,EPSILON)]*36,method='highs')
    proposal=solve.x if solve.success else np.zeros(36)
    historical=[]
    for c in outcomes:
        if c['tag'].startswith('joint_'):
            u=np.asarray(c['coefficients'])
            historical.append(dict(tag=c['tag'],old_prediction=float(nominal[0]+u@old_j[:,0]),
                smooth_prediction=float(nominal[0]+u@jacobian[:,0]),actual_score=c['score']))
    parent_file=BASE/'canonical_gate_band_bounded_causal_block_state_weight_learning_20260920/research_candidate.pt'
    expert_file=BASE/'canonical_gate_band_shape_cruise_completion_feedback_20260918/candidate_0.5.pt'
    assert sha(parent_file)==read(source/'receipt.json')['source_model_sha256']
    md=torch.load(source/'innovation_coordinates/candidate.pt',map_location='cpu',weights_only=False)
    stage=register('canonical_gate_band_smooth_constrained_credit_learning_20260920',
        [__file__,BASE/'learn_complete_joint_teachers.py',BASE/'learn_fine_innovation_credit.py',
         BASE/'learn_shared_temporal_joint_policy.py',BASE/'learned_action_pulse_policy.py',
         BASE/'joint_innovation_coordinates.py',source/'receipt.json',source/'local_response_audit.json',
         source/'innovation_coordinates/results.json',source/'innovation_coordinates/candidate.pt',parent_file,expert_file],
        dict(wall_s=90,workers=1,reused_paired_vectors=72,maximum_new_joint_vectors=3,
             dimensions=36,trust_box=EPSILON,identical_constraint_jacobian=True,
             same_prediction_buffer=True,acceptance_bounds_unchanged=True),
        'The36-parameter geometry-action extension creates a buffered linear feasible '
        'cone, but fixed100000 penalties on some bad-inward samples make its central '
        'return gradient predict a29775 improvement where actual improvement is0.42. '
        'Matched loss control: reuse all72 exact paired physical responses, same '
        'parent,features,solver,step box,guard Jacobian and prediction buffer. Only '
        'remove the redundant hard inward/first-turn constant from the LEARNING '
        'objective for successful complete trajectories, retaining every continuous '
        'shape/efficiency term. All actual inbound,first-turn,outbound,reverse,lateral, '
        'energy,curvature,speed,time,cruise and physical/completion guards remain '
        'IDENTICAL. Verify feasible old candidates have exactly the same scores. '
        'First rescore old joint validation trajectories without rerunning them; '
        'then evaluate at most three new feasible-direction scales. No failed '
        'prefix can pass. This isolates a local parameter-learning loss saturation '
        'problem, not PPO estimator bias or a universal root cause.')
    (stage/'innovation_coordinates').mkdir();write_json(stage/'baseline.json',data)
    write_json(stage/'loss_credit_audit.json',dict(clipped_samples=clipped,reused_physical_vectors=len(outcomes),
        identical_constraint_jacobian=True,old_feasible_scores_exactly_equal=True,
        historical_validations=historical,jacobian=jacobian.tolist(),
        linear_feasible=bool(solve.success),predicted_score_change=float(proposal@jacobian[:,0]),
        proposal=proposal.tolist(),inner_prediction_margin=margin.tolist()))
    started=time.perf_counter();new=[]
    try:
        parent=torch.load(parent_file,map_location='cpu',weights_only=False)
        expert=torch.load(expert_file,map_location='cpu',weights_only=False)['policy']
        setup(stage,parent,baseline,retained,expert,md['feature_spec'],started+85,36)
        if solve.success and proposal@jacobian[:,0]<-.001:
            for scale in (1.,.5,.25):
                c=evaluate((f'smooth_joint_{scale}',proposal*scale))
                c['original_barrier_score']=c['score'];c['score']=float(score(c['records']))
                c['predicted_score']=float(nominal[0]+scale*proposal@jacobian[:,0])
                new.append(c)
                print('SMOOTH_CONSTRAINED_JOINT',scale,c['predicted_score'],c['score'],
                      min(c['measured_inequalities']),flush=True)
        candidates=[zero,*new]
        good=[c for c in candidates if parameters_in_bounds(c['coefficients'],-3.,3.)
              and min(c['measured_inequalities'])>=-1e-8]
        best=min(good,key=lambda c:c['score'])
        md['coefficients']=np.asarray(best['coefficients']);md['learning_method']='continuous_objective_full_return_parameter_learning'
        torch.save(md,stage/'innovation_coordinates/candidate.pt')
        write_json(stage/'innovation_coordinates/results.json',dict(candidates=candidates,best=best['tag']))
        outcome=dict(mode='innovation_coordinates',best=best,baseline_score=zero['score'],new_candidates=len(new),
            reused_candidates=len(outcomes),improved=best['score']<zero['score']-.001,full_pool_pass=False)
        write_json(stage/'results.json',dict(arms=[outcome],training_only=True,adopted=False))
        receipt(stage,dict(status='completed_smooth_constrained_credit_learning',new_candidates=len(new),
            improved_arms=['innovation_coordinates'] if outcome['improved'] else [],
            source_model_sha256=sha(parent_file),wall_s=time.perf_counter()-started,
            full_pool_pass=False,adopted=False,goal_complete=False))
        print('SMOOTH_CONSTRAINED_COMPLETE',best['tag'],best['score'],historical,flush=True)
    except Exception as exc:
        receipt(stage,dict(status='failed_smooth_constrained_credit_learning',error=str(exc),
            completed_vectors=len(new),wall_s=time.perf_counter()-started,goal_complete=False));raise


if __name__=='__main__':main()
