"""Use equivalent signed XY slack for local learning of existing hard constraints."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import copy
import time
import numpy as np
import torch
from scipy.optimize import linprog
from learn_fine_innovation_credit import setup,evaluate
from learn_complete_joint_teachers import teacher_score,parameters_in_bounds
from study_io import BASE,register,receipt,write_json,sha
from search_complete_return import read


def signed_guards(candidate,cache):
    guards=np.asarray(candidate['measured_inequalities'],dtype=float).copy()
    assert len(guards)==11*len(candidate['records'])
    for index,r in enumerate(candidate['records']):
        file=r['trace']
        if file not in cache:
            with np.load(file) as z:ref,rows=z['reference'].copy(),z['rows'].copy()
            u,v=np.diff(ref,axis=0);u/=np.linalg.norm(u);v/=np.linalg.norm(v)
            sign=np.sign(u[0]*v[1]-u[1]*v[0]);xy=rows[:,:2]-ref[1]
            cache[file]=[float((xy@(sign*np.array([-d[1],d[0]]))).min()) for d in (u,v)]
        for j,(key,scale) in enumerate((('in_edge_outward_mm',.002),('out_edge_outward_mm',.005))):
            signed=cache[file][j];outward=r['shape'][key]
            assert abs(max(0.,-signed)-outward)<1e-12,(r['path'],key,signed,outward)
            guards[11*index+j]+=(outward+signed)/scale
    original=np.asarray(candidate['measured_inequalities'])
    assert np.array_equal(original>=-1e-8,guards>=-1e-8),'Signed reformulation changed actual feasibility'
    return guards


def main():
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_scaled_command_credit_learning_20260920'
    calibration=BASE/'canonical_gate_band_fine_motion_state_credit_learning_20260920'
    data=read(source/'baseline.json');history=read(source/'innovation_coordinates/results.json')['candidates']
    audit=read(source/'local_response_audit.json');steps=np.asarray(audit['steps'])
    zero=copy.deepcopy(next(c for c in history if not np.any(c['coefficients'])))
    baseline=zero['records'];paths=[r['path'] for r in baseline]
    retained=[next(o['baseline_shape'] for o in data['retained'] if o['path']==p) for p in paths]
    cache={};transformed={c['tag']:signed_guards(c,cache) for c in history}
    nominal=np.r_[zero['score'],transformed[zero['tag']]]
    def vector(c):return np.r_[c['score'],transformed[c['tag']]]
    jacobian=np.asarray([(vector(next(c for c in history if c['tag']==f'axis_{i}_+1'))-
                          vector(next(c for c in history if c['tag']==f'axis_{i}_-1')))/(2*steps[i]) for i in range(36)])
    errors=[]
    for c in read(calibration/'innovation_coordinates/results.json')['candidates']:
        if c['tag'].startswith('joint_'):
            actual=signed_guards(c,cache)
            errors.append(actual-(nominal[1:]+np.asarray(c['coefficients'])@jacobian[:18,1:]))
    active=(nominal[1:]<1e-7)&(np.linalg.norm(jacobian[:,1:],axis=0)>1e-8)
    margin=np.where(active,2.*(np.maximum(-np.min(errors,axis=0),0.)+1e-7),0.)
    solve=linprog(jacobian[:,0],A_ub=-jacobian[:,1:].T,b_ub=nominal[1:]-margin+1e-10,
                  bounds=list(zip(-steps,steps)),method='highs')
    proposal=solve.x if solve.success else np.zeros(36)
    parent_file=BASE/'canonical_gate_band_bounded_causal_block_state_weight_learning_20260920/research_candidate.pt'
    expert_file=BASE/'canonical_gate_band_shape_cruise_completion_feedback_20260918/candidate_0.5.pt'
    assert sha(parent_file)==read(source/'receipt.json')['source_model_sha256']
    md=torch.load(source/'innovation_coordinates/candidate.pt',map_location='cpu',weights_only=False)
    stage=register('canonical_gate_band_signed_slack_credit_learning_20260920',
        [__file__,BASE/'learn_complete_joint_teachers.py',BASE/'learn_fine_innovation_credit.py',
         BASE/'learn_shared_temporal_joint_policy.py',BASE/'learned_action_pulse_policy.py',BASE/'joint_innovation_coordinates.py',
         source/'receipt.json',source/'local_response_audit.json',source/'innovation_coordinates/results.json',
         source/'innovation_coordinates/candidate.pt',calibration/'local_response_audit.json',
         calibration/'innovation_coordinates/results.json',parent_file,expert_file],
        dict(wall_s=120,workers=1,reused_axis_vectors=72,maximum_new_joint_vectors=3,
             dimensions=36,per_coordinate_step=steps.tolist(),original_acceptance_unchanged=True),
        'Hinge-clipped outward excursion gives zero constraint value everywhere '
        'inside an edge, concealing actual geometric slack. Read the saved raw1ms '
        'XY and reformulate ONLY the local surrogate edge guards as bound+minimum '
        'signed edge distance instead of bound-max(0,-distance). Verify identical '
        'actual pass/fail for every existing recorded guard and exact reproduction '
        'of every recorded outward excursion. Reuse72 paired outcomes and the '
        'continuous objective, anisotropic trust box, parent and state/action basis. '
        'Recompute the2x inner prediction buffer from the SAME three18D calibration '
        'trajectories under the equivalent signed coordinates. Validate up to three '
        'new shared joint directions with the ORIGINAL actual acceptance guards. '
        'No corridor,edge,first-turn,shape,physics,time/cruise or rest bound changes. '
        'A better constraint representation is tested, not presumed physically feasible.')
    (stage/'innovation_coordinates').mkdir();write_json(stage/'baseline.json',data)
    write_json(stage/'signed_slack_audit.json',dict(reused_vectors=len(history),read_raw_traces=len(cache),
        all_existing_constraint_decisions_identical=True,all_outward_metrics_exact=True,
        nominal_old_guards=zero['measured_inequalities'],nominal_signed_guards=nominal[1:].tolist(),
        jacobian=jacobian.tolist(),inner_prediction_margin=margin.tolist(),proposal=proposal.tolist(),
        linear_program_success=bool(solve.success),predicted_score_change=float(proposal@jacobian[:,0])))
    started=time.perf_counter();outcomes=[zero];validations=[]
    try:
        parent=torch.load(parent_file,map_location='cpu',weights_only=False)
        expert=torch.load(expert_file,map_location='cpu',weights_only=False)['policy']
        setup(stage,parent,baseline,retained,expert,md['feature_spec'],started+115,36,False)
        if solve.success and proposal@jacobian[:,0]<-.001:
            for scale in (1.,.5,.25):
                c=evaluate((f'signed_joint_{scale}',proposal*scale));outcomes.append(c)
                actual=signed_guards(c,cache);predicted=nominal+np.asarray(c['coefficients'])@jacobian
                validations.append(dict(tag=c['tag'],predicted_score=float(predicted[0]),actual_score=c['score'],
                    predicted_signed_minimum=float(predicted[1:].min()),actual_signed_minimum=float(actual.min()),
                    original_actual_minimum=float(min(c['measured_inequalities']))))
                print('SIGNED_SLACK_JOINT',validations[-1],flush=True)
        good=[c for c in outcomes if parameters_in_bounds(c['coefficients'],-3.,3.) and min(c['measured_inequalities'])>=-1e-8]
        best=min(good,key=lambda c:c['score'])
        md['coefficients']=np.asarray(best['coefficients']);md['learning_method']='signed_slack_complete_return_parameter_learning'
        torch.save(md,stage/'innovation_coordinates/candidate.pt')
        write_json(stage/'innovation_coordinates/results.json',dict(candidates=outcomes,best=best['tag']))
        outcome=dict(mode='innovation_coordinates',best=best,baseline_score=zero['score'],new_candidates=len(outcomes)-1,
            reused_candidates=len(history),improved=best['score']<zero['score']-.001,full_pool_pass=False)
        write_json(stage/'results.json',dict(arms=[outcome],validations=validations,training_only=True,adopted=False))
        receipt(stage,dict(status='completed_signed_slack_credit_learning',new_candidates=len(outcomes)-1,
            improved_arms=['innovation_coordinates'] if outcome['improved'] else [],
            source_model_sha256=sha(parent_file),wall_s=time.perf_counter()-started,
            full_pool_pass=False,adopted=False,goal_complete=False))
        print('SIGNED_SLACK_COMPLETE',best['tag'],best['score'],flush=True)
    except Exception as exc:
        receipt(stage,dict(status='failed_signed_slack_credit_learning',error=str(exc),
            completed_vectors=len(outcomes)-1,wall_s=time.perf_counter()-started,goal_complete=False));raise


if __name__=='__main__':main()
