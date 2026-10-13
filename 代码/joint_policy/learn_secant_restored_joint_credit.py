"""Joint complete-return learning with measured multi-secant constraint correction."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import time
import copy
import numpy as np
import torch
from scipy.optimize import linprog
from learn_fine_innovation_credit import setup,evaluate
from learn_signed_slack_credit import signed_guards
from learn_complete_joint_teachers import parameters_in_bounds
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def corrected_model(initial,nominal,points,steps):
    """Minimum-change secant fit in dimensionless trust coordinates."""
    x=np.array([np.asarray(p['coefficients'])/steps for p,v in points])
    y=np.array([v-nominal for p,v in points])
    # Near-collinear measured directions are combined, not inverted as independent.
    correction=np.linalg.lstsq(x,y-x@initial,rcond=1e-3)[0]
    return initial+correction,dict(rank=int(np.linalg.matrix_rank(x,tol=1e-3)),
        maximum_fitted_residual=float(np.abs(y-x@(initial+correction)).max()))


def main():
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_signed_slack_credit_learning_20260920'
    scaled=BASE/'canonical_gate_band_scaled_command_credit_learning_20260920'
    saved=read(source/'innovation_coordinates/results.json')['candidates']
    zero=copy.deepcopy(saved[0]);data=read(source/'baseline.json')
    steps=np.array(read(scaled/'local_response_audit.json')['steps'])
    initial=np.array(read(source/'signed_slack_audit.json')['jacobian'])*steps[:,None]
    cache={}
    def vector(c):return np.r_[c['score'],signed_guards(c,cache)]
    nominal=vector(zero)
    # Quarter/half-scale secants calibrate direction; full scale remains a saved check.
    calibration=[(c,vector(c)) for c in saved if c['tag'] in ('signed_joint_0.25','signed_joint_0.5')]
    calibrated,fit=corrected_model(initial,nominal,calibration,steps)
    heldout=next(c for c in saved if c['tag']=='signed_joint_1.0')
    u=np.asarray(heldout['coefficients'])/steps;truth=vector(heldout)
    validation=dict(tag=heldout['tag'],score_actual=heldout['score'],
        old_maximum_guard_error=float(np.abs(truth[1:]-(nominal+u@initial)[1:]).max()),
        corrected_maximum_guard_error=float(np.abs(truth[1:]-(nominal+u@calibrated)[1:]).max()),
        curvature14_old_error=float(truth[5]-(nominal+u@initial)[5]),
        curvature14_corrected_error=float(truth[5]-(nominal+u@calibrated)[5]))
    parent_file=BASE/'canonical_gate_band_bounded_causal_block_state_weight_learning_20260920/research_candidate.pt'
    expert_file=BASE/'canonical_gate_band_shape_cruise_completion_feedback_20260918/candidate_0.5.pt'
    assert sha(parent_file)==read(source/'receipt.json')['source_model_sha256']
    stage=register('canonical_gate_band_secant_restored_joint_credit_learning_20260920',
        [__file__,BASE/'learn_fine_innovation_credit.py',BASE/'learn_signed_slack_credit.py',
         BASE/'learn_shared_temporal_joint_policy.py',BASE/'learned_action_pulse_policy.py',
         BASE/'joint_innovation_coordinates.py',BASE/'learn_complete_joint_teachers.py',
         source/'receipt.json',source/'signed_slack_audit.json',source/'constraint_prediction_audit.json',
         source/'innovation_coordinates/results.json',source/'innovation_coordinates/candidate.pt',
         scaled/'local_response_audit.json',parent_file,expert_file],
        dict(wall_s=150,maximum_new_vectors=12,paths=[r['path'] for r in zero['records']],
             dimensions=36,maximum_normalized_trust_radius=.25,minimum_radius=.015625,
             reused_axis_vectors=72,reused_joint_vectors=3,unchanged_actual_acceptance=True),
        'Signed-slack proposals repeatedly violate peak curvature at the SAME raw XY spatial '
        'location. The central responses almost cancel with grossly underestimated residual. '
        'Use saved quarter/half joint directions for minimum-change multi-secant correction '
        'and assess the saved full direction without rerunning it. Then learn new shared '
        '36D parameter vectors by sequential constrained linear solves in a smaller trust '
        'box. Feed each ACTUAL complete five-path outcome back into the derivative fit; '
        'never accept predicted feasibility. Inner positive margins protect sensitive '
        'active constraints; all original physical and shape guards remain unchanged. '
        'Stop on an actually feasible improvement, an empty feasible cone, budget, or '
        'twelve new vectors. No PPO interaction, new feature, temporal pulse or rule.')
    folder=stage/'innovation_coordinates';folder.mkdir();write_json(stage/'baseline.json',data)
    write_json(stage/'saved_prediction_validation.json',validation)
    print('SAVED_SECANT_VALIDATION',validation,flush=True)
    md=torch.load(source/'innovation_coordinates/candidate.pt',map_location='cpu',weights_only=False)
    baseline=zero['records'];paths=[r['path'] for r in baseline]
    retained=[next(o['baseline_shape'] for o in data['retained'] if o['path']==p) for p in paths]
    started=time.perf_counter();history=[zero];updates=[];radius=.25
    # Keep the smaller secants near the optimization scale; full-scale data stays held out.
    points=list(calibration)
    try:
        setup(stage,md['parent'],baseline,retained,md['expert_state'],md['feature_spec'],started+145,36,False)
        for iteration in range(12):
            if time.perf_counter()-started>135:break
            model,fit=corrected_model(initial,nominal,points,steps)
            sensitive=np.linalg.norm(model[:,1:],axis=0)>1e-8
            near=nominal[1:]<1e-5
            # Deliberate strict inward movement, not the old almost-zero cancellation.
            margin=np.where(sensitive&near,1e-5*radius/.25,0.)
            solved=linprog(model[:,0],A_ub=-model[:,1:].T,
                b_ub=nominal[1:]-margin,bounds=[(-radius,radius)]*36,method='highs')
            if not solved.success:
                updates.append(dict(iteration=iteration,status='no_strict_feasible_direction',radius=radius));break
            proposal=solved.x*steps
            if solved.fun>-.001:
                updates.append(dict(iteration=iteration,status='no_predicted_gain',radius=radius));break
            if any(np.allclose(proposal,c['coefficients'],rtol=0.,atol=1e-12) for c in history):break
            c=evaluate((f'secant_{iteration}',proposal));history.append(c)
            actual=vector(c);predicted=nominal+solved.x@model
            detail=dict(iteration=iteration,tag=c['tag'],radius=radius,fit=fit,
                predicted_score=float(predicted[0]),actual_score=c['score'],
                original_minimum=float(min(c['measured_inequalities'])),
                guard_error=float(np.abs(actual[1:]-predicted[1:]).max()),
                failed_constraints=np.flatnonzero(np.asarray(c['measured_inequalities']) < -1e-8).tolist())
            updates.append(detail);print('MEASURED_SECANT_UPDATE',detail,flush=True)
            write_json(folder/'results.json',dict(candidates=history,best='reused_identity'))
            write_json(stage/'updates.json',updates)
            if min(c['measured_inequalities'])>=-1e-8 and c['score']<zero['score']-.001:break
            points.append((c,actual))
            if len(points)%4==0:radius=max(radius*.5,.015625)
        good=[c for c in history if parameters_in_bounds(c['coefficients'],-3.,3.)
              and min(c['measured_inequalities'])>=-1e-8]
        best=min(good,key=lambda c:c['score']);improved=best['score']<zero['score']-.001
        md['coefficients']=np.array(best['coefficients']);md['learning_method']='measured_multi_secant_complete_return_parameter_learning'
        torch.save(md,folder/'candidate.pt')
        write_json(folder/'results.json',dict(candidates=history,best=best['tag']))
        write_json(stage/'updates.json',updates)
        write_json(stage/'results.json',dict(arms=[dict(mode='innovation_coordinates',best=best,
            baseline_score=zero['score'],new_candidates=len(history)-1,improved=improved,full_pool_pass=False)],
            training_only=True,adopted=False))
        receipt(stage,dict(status='completed_secant_restored_joint_credit_learning',
            new_candidates=len(history)-1,improved_arms=['innovation_coordinates'] if improved else [],
            source_model_sha256=sha(parent_file),wall_s=time.perf_counter()-started,
            full_pool_pass=False,adopted=False,goal_complete=False))
        print('SECANT_RESTORED_COMPLETE',best['tag'],best['score'],flush=True)
    except Exception as exc:
        receipt(stage,dict(status='failed_secant_restored_joint_credit_learning',error=str(exc),
            new_candidates=len(history)-1,wall_s=time.perf_counter()-started,goal_complete=False));raise


if __name__=='__main__':main()
