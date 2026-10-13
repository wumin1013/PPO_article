"""Calibrate local constraint prediction with measured full-return residuals."""
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
from learn_fine_innovation_credit import setup,evaluate,EPSILON
from learn_complete_joint_teachers import parameters_in_bounds
from study_io import BASE,register,receipt,write_json,sha
from search_complete_return import read


def main():
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_fine_motion_state_credit_learning_20260920'
    audit=read(source/'local_response_audit.json');jacobian=np.asarray(audit['jacobian'])
    old=read(source/'innovation_coordinates/results.json')['candidates']
    zero=copy.deepcopy(next(c for c in old if not np.any(c['coefficients'])))
    nominal=np.r_[zero['score'],zero['measured_inequalities']]
    errors=[]
    for c in old:
        if not c['tag'].startswith('joint_'):continue
        predicted=nominal+np.asarray(c['coefficients'])@jacobian
        errors.append(np.asarray(c['measured_inequalities'])-predicted[1:])
    assert len(errors)==3
    active=(nominal[1:]<1e-7)&(np.linalg.norm(jacobian[:,1:],axis=0)>1e-8)
    buffer=np.where(active,np.maximum(-np.min(errors,axis=0),0.)+1e-7,0.)
    proposals=[]
    for factor in (2.,5.,10.):
        margin=buffer*factor
        solve=linprog(jacobian[:,0],A_ub=-jacobian[:,1:].T,b_ub=nominal[1:]-margin,
                      bounds=[(-EPSILON,EPSILON)]*18,method='highs')
        proposal=solve.x if solve.success else np.zeros(18)
        gain=float(proposal@jacobian[:,0])
        proposals.append(dict(tag=f'buffer_{factor}',factor=factor,linear_feasible=bool(solve.success),
            coefficients=proposal.tolist(),predicted_score_change=gain,
            simulate=bool(solve.success and gain<-.001)))
    parent_file=BASE/'canonical_gate_band_bounded_causal_block_state_weight_learning_20260920/research_candidate.pt'
    expert_file=BASE/'canonical_gate_band_shape_cruise_completion_feedback_20260918/candidate_0.5.pt'
    assert sha(parent_file)==read(source/'receipt.json')['source_model_sha256']
    data=read(source/'baseline.json');md=torch.load(source/'innovation_coordinates/candidate.pt',map_location='cpu',weights_only=False)
    stage=register('canonical_gate_band_margin_restored_innovation_learning_20260920',
        [__file__,BASE/'learn_fine_innovation_credit.py',BASE/'learn_shared_temporal_joint_policy.py',
         BASE/'learned_action_pulse_policy.py',BASE/'joint_innovation_coordinates.py',
         source/'receipt.json',source/'local_response_audit.json',source/'innovation_coordinates/results.json',
         source/'innovation_coordinates/candidate.pt',parent_file,expert_file],
        dict(wall_s=90,workers=1,maximum_new_vectors=3,buffer_factors=[2.,5.,10.],
             trust_box=EPSILON,original_acceptance_unchanged=True),
        'The fine18-parameter model predicts objective improvement accurately, but '
        'all three real joint steps miss some active shape guards by small amounts. '
        'Calibrate a per-guard INNER prediction buffer from the worst observed '
        'negative actual-minus-predicted residual, plus1e-7 in the existing scaled '
        'guard units. Solve the same local linear direction with2,5,10 times that '
        'buffer; simulate only linearly feasible materially improving directions. '
        'This changes prediction margin, never a physical or acceptance bound. '
        'Reuse all old trajectories and the measured Jacobian. Every proposed '
        'shared three-action update must pass the unchanged actual full-feedback '
        'guards and objective before full88 validation. If the buffered cone is '
        'empty, do not claim exact physical infeasibility or repeat simulation. '
        'Parameter optimization,not PPO or final adoption.')
    (stage/'innovation_coordinates').mkdir();write_json(stage/'baseline.json',data)
    write_json(stage/'buffer_model.json',dict(buffer=buffer.tolist(),active=active.tolist(),
        measured_prediction_residuals=np.asarray(errors).tolist(),proposals=proposals))
    started=time.perf_counter();outcomes=[zero];validations=[]
    try:
        baseline=zero['records'];paths=[b['path'] for b in baseline]
        retained=[next(o['baseline_shape'] for o in data['retained'] if o['path']==p) for p in paths]
        parent=torch.load(parent_file,map_location='cpu',weights_only=False)
        expert=torch.load(expert_file,map_location='cpu',weights_only=False)['policy']
        setup(stage,parent,baseline,retained,expert,md['feature_spec'],started+85,18)
        seen=set()
        for p in proposals:
            if not p['simulate']:continue
            key=tuple(np.round(p['coefficients'],14))
            if key in seen:continue
            seen.add(key);c=evaluate((p['tag'],p['coefficients']));outcomes.append(c)
            validations.append(dict(tag=c['tag'],predicted_score=zero['score']+p['predicted_score_change'],
                actual_score=c['score'],actual_minimum_guard=float(min(c['measured_inequalities']))))
            write_json(stage/'innovation_coordinates/results.json',dict(candidates=outcomes,best='reused_identity'))
        good=[c for c in outcomes if parameters_in_bounds(c['coefficients'],-3.,3.) and min(c['measured_inequalities'])>=-1e-8]
        best=min(good,key=lambda c:c['score'])
        md['coefficients']=np.asarray(best['coefficients']);md['learning_method']='measured_constraint_margin_restored_parameter_learning'
        torch.save(md,stage/'innovation_coordinates/candidate.pt')
        write_json(stage/'innovation_coordinates/results.json',dict(candidates=outcomes,best=best['tag']))
        outcome=dict(mode='innovation_coordinates',best=best,baseline_score=zero['score'],
            new_candidates=len(outcomes)-1,reused_candidates=1,improved=best['score']<zero['score']-.001,full_pool_pass=False)
        write_json(stage/'results.json',dict(arms=[outcome],validations=validations,training_only=True,adopted=False))
        receipt(stage,dict(status='completed_margin_restored_innovation_learning',new_candidates=len(outcomes)-1,
            linearly_feasible_proposals=sum(p['linear_feasible'] for p in proposals),
            improved_arms=['innovation_coordinates'] if outcome['improved'] else [],
            source_model_sha256=sha(parent_file),wall_s=time.perf_counter()-started,
            full_pool_pass=False,adopted=False,goal_complete=False))
        print('MARGIN_RESTORED_INNOVATION',best['tag'],best['score'],validations,flush=True)
    except Exception as exc:
        receipt(stage,dict(status='failed_margin_restored_innovation_learning',error=str(exc),
            completed_vectors=len(outcomes)-1,wall_s=time.perf_counter()-started,goal_complete=False));raise


if __name__=='__main__':mp.freeze_support();main()
