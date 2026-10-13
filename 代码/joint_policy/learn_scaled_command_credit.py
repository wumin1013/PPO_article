"""Calibrate independent action coordinates at their measured smaller trust scale."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import copy
import multiprocessing as mp
import time
import numpy as np
import torch
from scipy.optimize import linprog
from learn_fine_innovation_credit import setup,evaluate
from learn_complete_joint_teachers import teacher_score,parameters_in_bounds
from study_io import BASE,register,receipt,write_json,sha
from search_complete_return import read


def main():
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_independent_command_credit_learning_20260920'
    smooth=BASE/'canonical_gate_band_smooth_constrained_credit_learning_20260920'
    data=read(source/'baseline.json');history=read(source/'innovation_coordinates/results.json')['candidates']
    zero=copy.deepcopy(next(c for c in history if not np.any(c['coefficients'])))
    baseline=zero['records'];paths=[r['path'] for r in baseline]
    retained=[next(o['baseline_shape'] for o in data['retained'] if o['path']==p) for p in paths]
    cached=[]
    for c in history:
        if c['tag'].startswith('axis_') and int(c['tag'].split('_')[1])<27:
            c=copy.deepcopy(c);c['original_barrier_score']=c['score']
            c['score']=sum(teacher_score(r,b,o,barrier=False) for r,b,o in zip(c['records'],baseline,retained))
            c['reused_response']=True;cached.append(c)
    assert len(cached)==54
    steps=np.r_[np.full(27,.0025),np.full(9,.00025)]
    margin=np.asarray(read(smooth/'loss_credit_audit.json')['inner_prediction_margin'])
    parent_file=BASE/'canonical_gate_band_bounded_causal_block_state_weight_learning_20260920/research_candidate.pt'
    expert_file=BASE/'canonical_gate_band_shape_cruise_completion_feedback_20260918/candidate_0.5.pt'
    assert sha(parent_file)==read(source/'receipt.json')['source_model_sha256']
    md=torch.load(source/'innovation_coordinates/candidate.pt',map_location='cpu',weights_only=False)
    stage=register('canonical_gate_band_scaled_command_credit_learning_20260920',
        [__file__,BASE/'learn_fine_innovation_credit.py',BASE/'learn_shared_temporal_joint_policy.py',
         BASE/'learn_complete_joint_teachers.py',BASE/'learned_action_pulse_policy.py',BASE/'joint_innovation_coordinates.py',
         source/'receipt.json',source/'innovation_coordinates/results.json',source/'innovation_coordinates/candidate.pt',
         smooth/'receipt.json',smooth/'loss_credit_audit.json',smooth/'innovation_coordinates/results.json',parent_file,expert_file],
        dict(wall_s=240,workers=2,dimensions=36,reused_axis_vectors=54,new_axis_vectors=18,
             maximum_joint_vectors=3,per_coordinate_step=steps.tolist(),
             continuous_learning_objective=True,original_acceptance_unchanged=True),
        'Removing the fixed inward loss barrier corrected the old held-out direction '
        'prediction, but newly optimized directions still overpredict return gains '
        'at all three measured scales and miss shape guards. Recalibrate only the '
        'nine newly independent physical-command directions at0.00025 instead of '
        '0.0025. Reuse all54 paired responses for the earlier27 weights exactly. '
        'Keep the same continuous objective,parent,features,training paths,hard '
        'guards and measured inner prediction margin. Solve the joint36-parameter '
        'direction within the now anisotropic measured trust box and actually '
        'validate1,1/2,1/4 steps. Neither a small prediction residual nor a lower '
        'loss substitutes for actual per-path feasibility and the later full88 '
        'check. This is a response-scale correction,not unchanged extra training '
        'or a PPO estimator claim. No physical limit or saved strong model changes.')
    (stage/'innovation_coordinates').mkdir();write_json(stage/'baseline.json',data)
    started=time.perf_counter();pool=None;outcomes=[zero,*cached];validations=[]
    try:
        parent=torch.load(parent_file,map_location='cpu',weights_only=False)
        expert=torch.load(expert_file,map_location='cpu',weights_only=False)['policy']
        pool=mp.get_context('spawn').Pool(2,setup,(stage,parent,baseline,retained,expert,md['feature_spec'],started+225,36,False))
        jobs=[(f'axis_{i}_{s:+d}',steps[i]*s*np.eye(36)[i]) for i in range(27,36) for s in (-1,1)]
        for c in pool.imap_unordered(evaluate,jobs):
            outcomes.append(c)
            write_json(stage/'innovation_coordinates/results.json',dict(candidates=outcomes,best='reused_identity'))
            if len(outcomes)%5==0:print('SCALED_COMMAND_RESPONSE',len(outcomes)-55,flush=True)
        def vector(c):return np.r_[c['score'],c['measured_inequalities']]
        nominal=vector(zero);jacobian=[];paired_even=[]
        for i in range(36):
            plus=vector(next(c for c in outcomes if c['tag']==f'axis_{i}_+1'))
            minus=vector(next(c for c in outcomes if c['tag']==f'axis_{i}_-1'))
            jacobian.append((plus-minus)/(2*steps[i]));paired_even.append(.5*(plus+minus)-nominal)
        jacobian=np.asarray(jacobian)
        solve=linprog(jacobian[:,0],A_ub=-jacobian[:,1:].T,b_ub=nominal[1:]-margin+1e-10,
                      bounds=list(zip(-steps,steps)),method='highs')
        proposal=solve.x if solve.success else np.zeros(36)
        if solve.success and proposal@jacobian[:,0]<-.001:
            jobs=[(f'scaled_joint_{s}',proposal*s) for s in (1.,.5,.25)]
            for c in pool.imap_unordered(evaluate,jobs):
                outcomes.append(c);predicted=nominal+np.asarray(c['coefficients'])@jacobian
                validations.append(dict(tag=c['tag'],predicted_score=float(predicted[0]),actual_score=c['score'],
                    predicted_minimum_guard=float(predicted[1:].min()),actual_minimum_guard=float(min(c['measured_inequalities']))))
                print('SCALED_COMMAND_JOINT',validations[-1],flush=True)
        pool.close();pool.join();pool=None
        good=[c for c in outcomes if parameters_in_bounds(c['coefficients'],-3.,3.) and min(c['measured_inequalities'])>=-1e-8]
        best=min(good,key=lambda c:c['score'])
        write_json(stage/'local_response_audit.json',dict(steps=steps.tolist(),jacobian=jacobian.tolist(),
            paired_even_residual=np.asarray(paired_even).tolist(),inner_prediction_margin=margin.tolist(),
            proposal=proposal.tolist(),linear_program_success=bool(solve.success),
            predicted_score_change=float(proposal@jacobian[:,0]),validations=validations))
        write_json(stage/'innovation_coordinates/results.json',dict(candidates=outcomes,best=best['tag']))
        md['coefficients']=np.asarray(best['coefficients']);md['learning_method']='anisotropic_local_complete_return_parameter_learning'
        torch.save(md,stage/'innovation_coordinates/candidate.pt')
        outcome=dict(mode='innovation_coordinates',best=best,baseline_score=zero['score'],new_candidates=len(outcomes)-55,
            reused_candidates=55,improved=best['score']<zero['score']-.001,full_pool_pass=False)
        write_json(stage/'results.json',dict(arms=[outcome],training_only=True,adopted=False))
        receipt(stage,dict(status='completed_scaled_command_credit_learning',new_candidates=len(outcomes)-55,
            improved_arms=['innovation_coordinates'] if outcome['improved'] else [],
            source_model_sha256=sha(parent_file),wall_s=time.perf_counter()-started,
            full_pool_pass=False,adopted=False,goal_complete=False))
        print('SCALED_COMMAND_COMPLETE',best['tag'],best['score'],flush=True)
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_scaled_command_credit_learning',error=str(exc),
            completed_vectors=len(outcomes)-55,wall_s=time.perf_counter()-started,goal_complete=False));raise


if __name__=='__main__':mp.freeze_support();main()
