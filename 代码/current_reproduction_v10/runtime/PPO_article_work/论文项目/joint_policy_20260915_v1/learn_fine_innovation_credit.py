"""Small paired full-return responses and measured feasible joint updates."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import copy
import argparse
import multiprocessing as mp
import time
import numpy as np
import torch
from scipy.optimize import linprog
from learned_action_pulse_policy import LearnedActionPulsePolicy
from learn_shared_temporal_joint_policy import initialize,inequalities,evaluate_complete_parameters
from learn_complete_joint_teachers import teacher_score,parameters_in_bounds
from study_io import BASE,register,receipt,write_json,sha
from search_complete_return import read


WORK={}
EPSILON=.0025


def setup(stage,parent,baseline,retained,expert,spec,deadline,dimensions,barrier=True):
    initialize(stage,parent,baseline,retained,deadline,dimensions>=18,None,expert,True,spec,[104,204,206,304])
    WORK.update(stage=stage,parent=parent,baseline=baseline,retained=retained,deadline=deadline,barrier=barrier,
        policy=LearnedActionPulsePolicy(parent,np.zeros(dimensions),'innovation_coordinates',expert,spec))


def evaluate(job):
    tag,u=job
    records=evaluate_complete_parameters(WORK['policy'],u,WORK['baseline'],
                                        WORK['stage']/'innovation_coordinates',tag,WORK['deadline'])
    score=sum(teacher_score(r,b,o,barrier=WORK['barrier']) for r,b,o in zip(records,WORK['baseline'],WORK['retained']))
    return dict(tag=tag,coefficients=np.asarray(u).tolist(),records=records,score=float(score),
                measured_inequalities=inequalities(records).tolist())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--motion-state',action='store_true')
    parser.add_argument('--geometry-state',action='store_true')
    parser.add_argument('--anticipation-state',action='store_true')
    args=parser.parse_args()
    if args.anticipation_state:args.geometry_state=True
    if args.geometry_state:args.motion_state=True
    torch.set_num_threads(1)
    source=BASE/('canonical_gate_band_geometry_motion_credit_learning_20260920' if args.anticipation_state
                 else 'canonical_gate_band_fine_motion_state_credit_learning_20260920' if args.geometry_state
                 else 'canonical_gate_band_motion_state_innovation_learning_20260920' if args.motion_state
                 else 'canonical_gate_band_shared_innovation_coordinate_learning_20260920')
    data=read(source/'baseline.json');history=read(source/'innovation_coordinates/results.json')['candidates']
    zero=copy.deepcopy(next(c for c in history if not np.any(c['coefficients'])))
    parent_file=BASE/'canonical_gate_band_bounded_causal_block_state_weight_learning_20260920/research_candidate.pt'
    expert_file=BASE/'canonical_gate_band_shape_cruise_completion_feedback_20260918/candidate_0.5.pt'
    assert sha(parent_file)==read(source/'receipt.json')['source_model_sha256']
    assert not any(a['improved'] for a in read(source/'results.json')['arms'])
    state_source=BASE/'canonical_gate_band_shared_innovation_coordinate_learning_20260920'
    if args.geometry_state:
        identity=(source/'anticipation_feature_identity_audit.json' if args.anticipation_state else
                  BASE/'canonical_gate_band_motion_state_innovation_learning_20260920/geometry_feature_identity_audit.json')
        assert read(identity)['parent_sha256']==sha(parent_file)
        spec=torch.load(source/'innovation_coordinates/candidate.pt',map_location='cpu',weights_only=False)['feature_spec']
        spec=dict(spec,gate_forward_time_scale_s=read(state_source/'innovation_feedback_state_audit.json')['gate_forward_time_median_s'])
        zero['coefficients']+= [0.]*9
        if args.anticipation_state:spec['independent_command_features']='Signed future-direction averages in three fixed spatial sensor groups; odd angular-steering residual and even squared feed/preview latent residuals.'
    else:spec=read(source/'registration.json')['budget']['feature_spec']
    dimensions=len(zero['coefficients'])
    prior=source if args.geometry_state else BASE/'canonical_gate_band_fine_innovation_credit_learning_20260920'
    cached=[]
    if args.motion_state:
        assert sha(parent_file)==read(prior/'receipt.json')['source_model_sha256']
        assert not any(a['improved'] for a in read(prior/'results.json')['arms'])
        for c in read(prior/'innovation_coordinates/results.json')['candidates']:
            if not c['tag'].startswith('axis_'):continue
            c=copy.deepcopy(c);c['coefficients']=list(c['coefficients'])+[0.]*9
            c['reused_fine_response']=True;cached.append(c)
        assert len(cached)==2*(dimensions-9)
    paths=[r['path'] for r in zero['records']]
    margin_source=BASE/'canonical_gate_band_fine_motion_state_credit_learning_20260920'
    stage=register('canonical_gate_band_independent_command_credit_learning_20260920' if args.anticipation_state
                   else 'canonical_gate_band_geometry_motion_credit_learning_20260920' if args.geometry_state
                   else 'canonical_gate_band_fine_motion_state_credit_learning_20260920' if args.motion_state
                   else 'canonical_gate_band_fine_innovation_credit_learning_20260920',
        [__file__,BASE/'learn_shared_temporal_joint_policy.py',BASE/'learned_action_pulse_policy.py',
         BASE/'joint_innovation_coordinates.py',BASE/'learn_complete_joint_teachers.py',
         source/'receipt.json',source/'results.json',source/'innovation_coordinates/results.json',
         (prior if args.motion_state else source)/('local_response_audit.json' if args.motion_state else 'shared_constraint_credit_audit.json'),
         parent_file,expert_file]+
         ([prior/'receipt.json',prior/'results.json',prior/'innovation_coordinates/results.json'] if args.motion_state else [])+
         ([state_source/'innovation_feedback_state_audit.json',
           identity,margin_source/'local_response_audit.json',margin_source/'innovation_coordinates/results.json',
           BASE/'canonical_gate_band_margin_restored_innovation_learning_20260920/receipt.json',
           BASE/'canonical_gate_band_margin_restored_innovation_learning_20260920/buffer_model.json'] if args.geometry_state else []),
        dict(wall_s=240,workers=2,dimensions=dimensions,paired_axis_vectors=2*dimensions,
             new_axis_vectors=18,reused_axis_vectors=len(cached),feature_spec=spec,
             maximum_joint_validation_vectors=3,epsilon=EPSILON,paths=paths,
             physical_limits_unchanged=True,zero_rollouts_repeated=0),
        ('The27-parameter state extension still has no buffered feasible local direction. '
         'Its steering correction vanishes when the parent target jump is zero, and '
         'feed/preview corrections vanish where the two learned experts agree. Add '
         'nine jointly learned INDEPENDENT command coefficients on three observed '
         'future-heading sensor groups. Signed averaged headings generate an odd '
         'angular steering residual; squared headings generate even feed/preview '
         'latent residuals. Combine all three before target-coordinate decoding, so '
         'preview and steering remain physically coupled. Sensor groups are fixed '
         'spatial observation encoding, never corner templates, time windows or '
         'angle-speed tables. Reuse all54 existing small paired responses exactly '
         'with nine zero appended coefficients. Measure only18 new directions and '
         'up to three real joint updates under the SAME measured prediction buffer '
         'and original acceptance guards. This tests action-direction capacity; '
         'it is not a claim that state/action aliasing is the sole cause. '
         if args.anticipation_state else '')+
        ('The18-dimensional local model has a decreasing boundary direction, but '
         'its real steps miss shape guards and its residual-buffered feasible cone '
         'is empty. Extend only the shared observable state: multiply the three '
         'even innovation features by tanh of heading-frame signed gate-forward '
         'range/(speed+5), scaled by the median absolute value on all88 saved '
         'training trajectories. This continuous relative geometry coordinate '
         'contains no path, absolute time or angle lookup. Reuse all36 old paired '
         'responses with nine appended zeros. Measure only18 new geometry responses. '
         'Use the already measured18D negative constraint-prediction errors as a2x '
         'INNER buffer for the27D proposal. Every new physical result must satisfy '
         'the unchanged acceptance constraints; the buffered surrogate is not proof. '
         if args.geometry_state and not args.anticipation_state else '')+
        ('The coarse dynamic-state extension also has no feasible update. Test its '
         'local credit with the same0.0025 perturbation scale. Reuse all18 earlier '
         'nine-parameter fine responses EXACTLY by appending zero dynamic coefficients; '
         'measure only the18 new +/- dynamic-coordinate responses, then at most '
         'three actual joint proposal vectors. Keep all parent/paths/guards unchanged. '
         if args.motion_state and not args.geometry_state else '')+
        'The failed79-vector nine-parameter run never sampled a nonzero parameter norm '
        'below0.25, and its weighted linear full-return fit has42percent held-out '
        'normalized error. Do not attribute that failure solely to missing state. '
        'Keep the EXACT nine-parameter state/action function,parent,paths,objective '
        'and measured guards. Collect paired +/-0.0025 complete-feedback responses '
        'for each coefficient, compute the local return and constraint Jacobians, '
        'then solve a box-bounded linear feasible direction and measure it in the '
        'real full feedback loop at1,1/2,1/4 amplitude. Reuse zero paths and prior '
        'identity evidence. Prediction alone never passes: require actual parameter '
        'box,all trajectory constraints and improved complete-return objective before '
        'a full88 check. Save negative responses and surrogate validation errors. '
        'This is local full-return parameter learning,not PPO,not a deployed rule '
        'and not a proof that the linear model is globally valid.')
    (stage/'innovation_coordinates').mkdir()
    write_json(stage/'baseline.json',data)
    started=time.perf_counter();pool=None;outcomes=[zero,*cached];reused=len(outcomes)
    try:
        parent=torch.load(parent_file,map_location='cpu',weights_only=False)
        expert=torch.load(expert_file,map_location='cpu',weights_only=False)['policy']
        baseline=zero['records']
        retained=[next(o['baseline_shape'] for o in data['retained'] if o['path']==p) for p in paths]
        pool=mp.get_context('spawn').Pool(2,setup,(stage,parent,baseline,retained,expert,spec,started+225,dimensions))
        first=dimensions-9 if args.motion_state else 0
        jobs=[(f'axis_{i}_{sign:+d}',EPSILON*sign*np.eye(dimensions)[i]) for i in range(first,dimensions) for sign in (-1,1)]
        for c in pool.imap_unordered(evaluate,jobs):
            outcomes.append(c)
            write_json(stage/'innovation_coordinates/results.json',dict(candidates=outcomes,best='reused_identity'))
            if len(outcomes)%5==0:print('FINE_INNOVATION_RESPONSE',len(outcomes)-1,flush=True)
        def vector(c):return np.r_[c['score'],c['measured_inequalities']]
        nominal=vector(zero);jacobian=[];even_residual=[]
        for i in range(dimensions):
            minus=vector(next(c for c in outcomes if c['tag']==f'axis_{i}_-1'))
            plus=vector(next(c for c in outcomes if c['tag']==f'axis_{i}_+1'))
            jacobian.append((plus-minus)/(2*EPSILON))
            even_residual.append(.5*(plus+minus)-nominal)
        jacobian=np.asarray(jacobian)
        margin=np.zeros_like(nominal[1:])
        if args.geometry_state:
            old_j=np.asarray(read(margin_source/'local_response_audit.json')['jacobian'])
            errors=[]
            for c in read(margin_source/'innovation_coordinates/results.json')['candidates']:
                if c['tag'].startswith('joint_'):
                    errors.append(np.asarray(c['measured_inequalities'])-(nominal+np.asarray(c['coefficients'])@old_j)[1:])
            active=(nominal[1:]<1e-7)&(np.linalg.norm(jacobian[:,1:],axis=0)>1e-8)
            margin=np.where(active,2.*(np.maximum(-np.min(errors,axis=0),0.)+1e-7),0.)
        result=linprog(jacobian[:,0],A_ub=-jacobian[:,1:].T,b_ub=nominal[1:]-margin+1e-10,
                       bounds=[(-EPSILON,EPSILON)]*dimensions,method='highs')
        proposal=result.x if result.success else np.zeros(dimensions)
        validations=[]
        if np.max(abs(proposal))>1e-8 and proposal@jacobian[:,0]<-.001:
            jobs=[(f'joint_{scale}',proposal*scale) for scale in (1.,.5,.25)]
            for c in pool.imap_unordered(evaluate,jobs):
                outcomes.append(c);predicted=nominal+np.asarray(c['coefficients'])@jacobian
                validations.append(dict(tag=c['tag'],predicted_score=float(predicted[0]),actual_score=c['score'],
                    predicted_minimum_guard=float(predicted[1:].min()),
                    actual_minimum_guard=float(min(c['measured_inequalities']))))
                write_json(stage/'innovation_coordinates/results.json',dict(candidates=outcomes,best='reused_identity'))
        pool.close();pool.join();pool=None
        good=[c for c in outcomes if parameters_in_bounds(c['coefficients'],-3.,3.)
              and min(c['measured_inequalities'])>=-1e-8]
        best=min(good,key=lambda c:c['score'])
        write_json(stage/'local_response_audit.json',dict(epsilon=EPSILON,jacobian=jacobian.tolist(),
            inner_prediction_margin=margin.tolist(),linear_program_success=bool(result.success),
            paired_even_residual=np.asarray(even_residual).tolist(),proposal=proposal.tolist(),
            predicted_score_change=float(proposal@jacobian[:,0]),validations=validations,
            surrogate_is_not_physical_acceptance=True))
        write_json(stage/'innovation_coordinates/results.json',dict(candidates=outcomes,best=best['tag']))
        md=dict(architecture='joint_expert_residual',parent=parent,coefficients=np.asarray(best['coefficients']),
            mode='innovation_coordinates',expert_state=expert,feature_spec=spec,
            reference_envelope_paths=[104,204,206,304],protect_current_lateral_energy_exit=True,
            learning_method='paired_local_complete_return_parameter_learning',accepted=False,
            motion_conditioned=args.motion_state,
            relative_geometry_conditioned=args.geometry_state,
            independent_command_residual=args.anticipation_state,
            action_interval_ms=20,physical_tick_ms=1)
        torch.save(md,stage/'innovation_coordinates/candidate.pt')
        outcome=dict(mode='innovation_coordinates',best=best,baseline_score=zero['score'],
            new_candidates=len(outcomes)-reused,reused_candidates=reused,
            improved=best['score']<zero['score']-.001,full_pool_pass=False)
        write_json(stage/'results.json',dict(arms=[outcome],training_only=True,adopted=False))
        receipt(stage,dict(status='completed_independent_command_credit_learning' if args.anticipation_state
            else 'completed_geometry_motion_credit_learning' if args.geometry_state
            else 'completed_fine_motion_state_credit_learning' if args.motion_state else 'completed_fine_innovation_credit_learning',
            new_candidates=len(outcomes)-reused,reused_candidates=reused,
            improved_arms=['innovation_coordinates'] if outcome['improved'] else [],
            source_model_sha256=sha(parent_file),wall_s=time.perf_counter()-started,
            full_pool_pass=False,adopted=False,goal_complete=False))
        print('FINE_INNOVATION_COMPLETE',best['tag'],best['score'],validations,flush=True)
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_independent_command_credit_learning' if args.anticipation_state
            else 'failed_geometry_motion_credit_learning' if args.geometry_state
            else 'failed_fine_motion_state_credit_learning' if args.motion_state else 'failed_fine_innovation_credit_learning',error=str(exc),
            completed_vectors=len(outcomes)-reused,wall_s=time.perf_counter()-started,goal_complete=False));raise


if __name__=='__main__':mp.freeze_support();main()
