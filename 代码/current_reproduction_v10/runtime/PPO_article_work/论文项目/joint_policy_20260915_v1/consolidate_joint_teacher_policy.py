"""Fit shared joint-action state functions, then test real interpolation closure."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import copy
import argparse
import math
import multiprocessing as mp
import time
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.sparse.linalg import lsmr,splu
from scipy.sparse import eye
from shared_teacher_kernel_policy import SharedTeacherKernelPolicy,kernel
from joint_teacher_state_policy import state_pair
from decision_innovation_policy import innovation_features
from deep_joint_credit_policy import load_credit_policy
from canonical_gate_state import checkpoint_environment_factory
from learn_nonlinear_joint_feedback import rollout
from learn_full_pool_joint_credit import aggregate
from learn_retained_joint_growth import smooth_cruise
from study_io import BASE,register,receipt,write_json,sha
from search_complete_return import read
from joint_runtime import DT,FILTER_TAU,wrap


WORK={}


def collect_data(current,parent,records,teachers=None,teacher_stage=None,expert_state=None,include_latents=False,deadline=None):
    same_parent=current is parent
    current=None if same_parent else load_credit_policy(current)
    parent=load_credit_policy(parent)
    expert=None
    if expert_state is not None:
        from canonical_gate_state import CanonicalGatePolicy
        expert=CanonicalGatePolicy(256);expert.load_state_dict(expert_state)
    factory=checkpoint_environment_factory('gate_band_canonical_kernel')
    observations=[];inputs=[];labels=[];path_ids=[];decisions=[]
    teacher_weights=[];directions=[]
    replayed_parent_latents=[];replayed_current_latents=[]
    teachers=teachers or {}
    for r in records:
        with np.load(r['trace']) as z:
            ref,rows,actions,obs=[z[k].copy() for k in ('reference','rows','actions','observations')]
        teacher=teachers.get(r['path'])
        if teacher:
            from scipy.interpolate import CubicSpline
            from learn_complete_joint_teachers import SCALE
            with np.load(teacher_stage/teacher['best']['trace']) as z:
                prefix=int(z['start_tick'])//20
                stop=int(z['stop_tick'])//20
                obs=np.vstack([obs[:prefix],z['observations']])
                ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
                parent_latents=z['parent_latents'].copy()
                saved_labels=z['residual_labels'].copy()
                latent_actions=z['latent_actions'].copy() if 'latent_actions' in z else None
                basis=z['basis'].copy() if 'basis' in z else None
            if basis is None:
                basis=CubicSpline(np.linspace(0,1,5),np.vstack([np.eye(4),np.zeros((1,4))]),
                                  bc_type='clamped')(np.linspace(0,1,stop-prefix+1))
            parameters=np.asarray(teacher['best']['parameters']).reshape(-1,3)
        env=factory(ref);tick=0
        if hasattr(current,'bind'):current.bind(env)
        if hasattr(parent,'bind'):parent.bind(env)
        for index,x in enumerate(obs):
            if deadline is not None and time.perf_counter()>=deadline:
                raise TimeoutError('Saved state replay wall budget')
            frame=index*20
            while tick<frame:
                # Reproduce the actual filtered heading, not an atan2 round trip.
                # New state kernels can detect tiny innovations even if XY is equal.
                if tick%20==0:env.held=actions[tick//20].copy()
                assert not rows[tick,12]
                target=env.guidance_target();delta=target-env.position
                beta=math.atan2(delta[1],delta[0]) if np.linalg.norm(delta)>1e-12 else env.heading
                increment=(1-math.exp(-DT/FILTER_TAU))*wrap(beta+env.held[0]-env.heading)
                env.heading=wrap(env.heading+increment)
                desired_velocity=env.held[1]*np.array([math.cos(env.heading),math.sin(env.heading)])
                assert np.array_equal(desired_velocity,rows[tick,8:10]),('heading_replay',r['path'],tick)
                env.position=rows[tick,:2].copy();env.velocity=rows[tick,2:4].copy()
                env.acceleration=rows[tick,4:6].copy()
                env.tracker.update(env.position);tick+=1
            if index:env.held=actions[index-1].copy()
            env.ticks=frame;env.decisions=index
            original=parent.act(x,True)[0]
            if teacher:
                j=index-prefix
                desired=original
                if j>=0:
                    assert np.array_equal(original,parent_latents[j]),('parent_identity',r['path'],index)
                    offset=basis[j]@parameters*SCALE if j<len(basis) else np.zeros(3)
                    desired=latent_actions[j] if latent_actions is not None else (original+offset).astype(original.dtype)
                    assert np.array_equal(desired-original,saved_labels[j])
            else:
                desired=original if same_parent else current.act(x,True)[0]
            assert np.array_equal(env.decode_action(desired),actions[index]),('teacher_action',r['path'],index)
            if expert is not None:
                direction=expert.act(x,True)[0]-original
                weights=basis[j]@parameters if teacher and 0<=j<len(basis) else np.zeros(3)
                assert np.array_equal((original+weights*direction).astype(original.dtype),desired),('expert_weight_identity',r['path'],index)
                weights[direction==0]=0.
                teacher_weights.append(weights);directions.append(direction)
            _,innovation=innovation_features(env,original)
            observations.append(x);inputs.append(innovation);labels.append(desired-original)
            path_ids.append(r['path']);decisions.append(index)
            if include_latents:
                replayed_parent_latents.append(original.copy());replayed_current_latents.append(desired.copy())
    data=dict(observations=np.asarray(observations),innovations=np.asarray(inputs),labels=np.asarray(labels),
              path_ids=np.asarray(path_ids),decisions=np.asarray(decisions))
    if expert is not None:data.update(teacher_weights=np.asarray(teacher_weights),directions=np.asarray(directions))
    if include_latents:data.update(parent_latents=np.asarray(replayed_parent_latents),current_latents=np.asarray(replayed_current_latents))
    return data


def initialize(stage,parent,baseline,data,deadline,direct=False,teachers=None,expert_state=None):
    torch.set_num_threads(1)
    WORK.update(stage=stage,parent=parent,baseline=baseline,data=data,deadline=deadline,direct=direct,
                teachers=teachers or {},expert_state=expert_state)


def fit_and_evaluate(width):
    tag=f'width_{width}'
    folder=WORK['stage']/tag;folder.mkdir()
    data=WORK['data'];a,b=state_pair(data['observations'],data['innovations'])
    both=np.vstack([a,b]);mean=both.mean(0);std=np.maximum(both.std(0),.05)
    a,b=(a-mean)/std,(b-mean)/std
    expert_mode=WORK['expert_state'] is not None
    labels=data['teacher_weights' if expert_mode else 'labels'].astype(np.float64)
    tree=cKDTree(a);radius=width*np.sqrt(a.shape[1])
    def matrix(points):
        distances=cKDTree(points).sparse_distance_matrix(tree,radius,output_type='coo_matrix')
        distances.data=kernel(distances.data/np.sqrt(a.shape[1]),width,a.shape[1])
        distances.data[distances.data<1e-14]=0.
        distances.eliminate_zeros()
        return distances.tocsr()
    first,second=matrix(a),matrix(b)
    odd,even=.5*(first-second),.5*(first+second)
    coefficients=[];errors=[];solvers=[];factor=None
    predictions=[]
    for head,features in enumerate((even,even,even) if expert_mode else (odd,even,even)):
        if time.perf_counter()>WORK['deadline']:raise TimeoutError('Shared kernel fitting budget')
        if WORK['direct']:
            if head==0 or (head==1 and not expert_mode):
                factor=splu((features+1e-12*eye(len(a),format='csr')).tocsc())
            coefficient=factor.solve(labels[:,head])
            residual=features@coefficient-labels[:,head]
            solvers.append(dict(method='float64_sparse_lu',diagonal_regularizer=1e-12,
                                residual=float(np.linalg.norm(residual))))
        else:
            solution=lsmr(features,labels[:,head],damp=1e-11,atol=1e-11,btol=1e-11,
                          conlim=1e14,maxiter=1800)
            coefficient=solution[0]
            solvers.append(dict(stop=int(solution[1]),iterations=int(solution[2]),residual=float(solution[3])))
        coefficients.append(coefficient);errors.append(float(abs(features@coefficient-labels[:,head]).max()))
        predictions.append(features@coefficient)
    md=dict(architecture='shared_joint_teacher_kernel',parent=WORK['parent'],mean=mean,std=std,
            centers=a,width=width,coefficients=np.column_stack(coefficients),accepted=False,
            learning_method='offline_joint_state_collocation_fitting',decision_ms=20,execution_ms=1)
    if expert_mode:md.update(output_mode='expert_difference_weights',expert_state=WORK['expert_state'])
    torch.save(md,folder/'candidate.pt')
    audit=dict(width=width,states=len(a),maximum_action_fit_error=errors,solvers=solvers,
               nonzero_kernel_entries=[int(odd.nnz),int(even.nnz)],time_or_path_inputs=False)
    if expert_mode:
        audit.update(output_mode='expert_difference_weights',fit_target='even_expert_weights',
            maximum_latent_action_fit_error=np.max(abs(np.column_stack(predictions)*data['directions']-data['labels']),axis=0).tolist())
    write_json(folder/'fitting_audit.json',audit)
    print('SHARED_KERNEL_FIT',audit,flush=True)
    # Large training errors are already decisive; do not spend complete rollouts on them.
    if max(errors)>1e-5 or max(audit.get('maximum_latent_action_fit_error',[0.]))>1e-5:
        return dict(tag=tag,fit_pass=False,fit=audit,research_acceptable=False)
    policy=SharedTeacherKernelPolicy(md)
    def factory(ref):return policy.bind(checkpoint_environment_factory('gate_band_canonical_kernel')(ref))
    records=[];guards=[]
    costs=dict(actor_calls=0,physics_ticks=0)
    for baseline in WORK['baseline']['records']:
        with np.load(baseline['trace']) as z:ref=z['reference'].copy()
        r,rows,actions,obs=rollout(policy,factory,ref,costs,WORK['deadline'])
        file=folder/f"path{baseline['path']}.npz"
        np.savez_compressed(file,reference=ref,rows=rows,actions=actions,observations=obs)
        r.update(path=baseline['path'],trace=file.relative_to(WORK['stage']).as_posix(),curved_path=baseline['curved_path'])
        if r['success']:r['smooth_cruise99_s']=smooth_cruise(rows)
        good=r['success'];checks=dict(complete=bool(good))
        if 'shape' in baseline:
            teacher=WORK['teachers'].get(r['path'])
            bound=teacher['shape_guard_bounds'] if teacher else baseline['shape']
            keys=('in_edge_outward_mm','out_edge_outward_mm','full_physical_reverse_deg',
                  'full_normalized_energy','full_spatial_peak_curvature_mm_inv')
            if 'reverse_lateral_total_mm' in bound:keys+=('reverse_lateral_total_mm',)
            checks.update({k:r['shape'][k]<=bound[k]+1e-7 for k in keys})
            checks.update(first_turn=bool(r['target_motion']['first_turn_inside']),
                          apex_speed=r['shape']['apex_speed_mm_s']>=bound['apex_speed_mm_s']-1e-4)
            if teacher:
                checks.update(time=r['complete_time_s']<=baseline['complete_time_s']+.02,
                    cruise=r['motion']['cruise_ge99_s']>=baseline['motion']['cruise_ge99_s']-.02)
        good=all(checks.values())
        records.append(r);guards.append(dict(path=r['path'],pass_all=bool(good),checks=checks))
        if len(records)%16==0:print('SHARED_KERNEL_CLOSURE',tag,len(records),sum(g['pass_all'] for g in guards),flush=True)
    summary=aggregate(records);old=WORK['baseline']['summary']
    accepted=bool(all(g['pass_all'] for g in guards) and summary['complete_time_s']<=old['complete_time_s']+.02
        and summary['curved_cruise99_s']>=old['curved_cruise99_s']-.05
        and summary['reverse_sum_deg']<old['reverse_sum_deg']
        and summary['exit_sum_mm']<=old['exit_sum_mm'] and summary['energy_sum']<=old['energy_sum'])
    result=dict(tag=tag,fit_pass=True,fit=audit,records=records,summary=summary,
        original64_summary=aggregate([r for r in records if r['path']<64]),
        added_training_summary=aggregate([r for r in records if r['path']>=64]),
        guards=guards,research_acceptable=accepted,
        aggregate_checks=dict(time=summary['complete_time_s']<=old['complete_time_s']+.02,
            cruise=summary['curved_cruise99_s']>=old['curved_cruise99_s']-.05,
            reverse=summary['reverse_sum_deg']<old['reverse_sum_deg'],
            exit=summary['exit_sum_mm']<=old['exit_sum_mm'],energy=summary['energy_sum']<=old['energy_sum'])
            if summary['successes']==summary['paths'] else dict(all_complete=False))
    write_json(WORK['stage']/(tag+'.json'),result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--direct-solve',action='store_true')
    parser.add_argument('--adaptive-teachers',action='store_true')
    parser.add_argument('--visible-teachers',action='store_true')
    parser.add_argument('--event-teachers',action='store_true')
    parser.add_argument('--expert-teachers',action='store_true')
    parser.add_argument('--exact-state',action='store_true')
    parser.add_argument('--lateral-teachers',action='store_true')
    parser.add_argument('--expert-weights',action='store_true')
    parser.add_argument('--causal-teachers',action='store_true')
    parser.add_argument('--bounded-causal-teachers',action='store_true')
    args=parser.parse_args()
    if args.bounded_causal_teachers:args.causal_teachers=True
    assert not args.expert_weights or args.lateral_teachers,'Weight comparison reuses lateral-aware data'
    if args.visible_teachers:args.adaptive_teachers=True
    if args.event_teachers:args.adaptive_teachers=True
    if args.expert_teachers:args.adaptive_teachers=True
    if args.lateral_teachers:args.adaptive_teachers=True
    if args.causal_teachers:args.adaptive_teachers=True
    weight_mode=args.expert_weights or args.causal_teachers
    source=BASE/'canonical_gate_band_target_jump_teacher_state_distillation_20260919'
    pool_source=BASE/'canonical_gate_band_shared_temporal_joint_learning_20260919'
    if args.event_teachers:
        source=BASE/'canonical_gate_band_visible_credit_shared_kernel_learning_20260920'
        pool_source=BASE/'canonical_gate_band_event_resolved_joint_teacher_learning_20260920'
    if args.expert_teachers:
        source=BASE/'canonical_gate_band_visible_credit_shared_kernel_learning_20260920'
        pool_source=BASE/'canonical_gate_band_expert_direction_joint_teacher_learning_20260920'
    if args.lateral_teachers:
        source=BASE/'canonical_gate_band_expert_direction_exact_state_shared_kernel_learning_20260920'
        pool_source=BASE/'canonical_gate_band_lateral_aware_joint_teacher_learning_20260920'
    if args.causal_teachers:
        source=BASE/'canonical_gate_band_lateral_aware_expert_weight_learning_20260920'
        pool_source=BASE/('canonical_gate_band_bounded_causal_block_joint_teacher_learning_20260920'
                         if args.bounded_causal_teachers else 'canonical_gate_band_causal_block_joint_teacher_learning_20260920')
    current_file=source/('width_1.2/candidate.pt' if args.causal_teachers else 'research_candidate.pt')
    source_audit=source/('raw_result_audit_width_1.2.json' if args.causal_teachers else 'raw_result_audit.json')
    parent_file=BASE/'canonical_gate_band_iterative_motion_constraint_learning_20260919/research_candidate.pt'
    teacher_stage=BASE/'canonical_gate_band_adaptive_credit_joint_teacher_learning_20260920'
    if args.visible_teachers:teacher_stage=BASE/'canonical_gate_band_visible_credit_joint_teacher_learning_20260920'
    if args.event_teachers:teacher_stage=pool_source
    if args.expert_teachers:teacher_stage=pool_source
    if args.lateral_teachers:teacher_stage=pool_source
    if args.causal_teachers:teacher_stage=pool_source
    teachers={}
    if args.adaptive_teachers:
        args.direct_solve=True
        parent_file=current_file
        teachers={r['path']:r for r in read(teacher_stage/'results.json')['teachers'] if r['score_improved']}
        assert sha(parent_file)==read(teacher_stage/'receipt.json')['source_model_sha256']
        assert read(teacher_stage/'receipt.json')['status'].startswith('completed_')
    data=read(pool_source/'baseline.json');baseline=data['baseline']
    path_count=len(baseline['records'])
    widths=[1.2] if args.causal_teachers else [.6,1.2]
    prior=BASE/'canonical_gate_band_shared_teacher_kernel_learning_20260919'
    if args.expert_weights:prior=BASE/'canonical_gate_band_lateral_aware_shared_kernel_learning_20260920'
    expert_state=None
    expert_file=BASE/'canonical_gate_band_shape_cruise_completion_feedback_20260918/candidate_0.5.pt'
    if args.lateral_teachers or args.causal_teachers:
        assert sha(expert_file)==read(teacher_stage/'receipt.json')['expert_model_sha256']
        expert_state=torch.load(expert_file,map_location='cpu',weights_only=False)['policy']
    stage=register('canonical_gate_band_bounded_causal_block_state_weight_learning_20260920' if args.bounded_causal_teachers
                   else 'canonical_gate_band_causal_block_state_weight_learning_20260920' if args.causal_teachers
                   else 'canonical_gate_band_lateral_aware_expert_weight_learning_20260920' if args.expert_weights
                   else 'canonical_gate_band_lateral_aware_shared_kernel_learning_20260920' if args.lateral_teachers
                   else 'canonical_gate_band_expert_direction_exact_state_shared_kernel_learning_20260920'
                   if args.expert_teachers and args.exact_state
                   else 'canonical_gate_band_expert_direction_shared_kernel_learning_20260920' if args.expert_teachers
                   else 'canonical_gate_band_event_resolved_shared_kernel_learning_20260920' if args.event_teachers
                   else 'canonical_gate_band_visible_credit_shared_kernel_learning_20260920' if args.visible_teachers
                   else 'canonical_gate_band_adaptive_credit_shared_kernel_learning_20260920' if args.adaptive_teachers
                   else 'canonical_gate_band_shared_teacher_kernel_direct_learning_20260920' if args.direct_solve
                   else 'canonical_gate_band_shared_teacher_kernel_learning_20260919',
        [__file__,BASE/'shared_teacher_kernel_policy.py',BASE/'joint_teacher_state_policy.py',BASE/'joint_runtime.py',
         current_file,parent_file,source/'results.json',source/'receipt.json',source_audit,
         pool_source/'baseline.json']+([expert_file] if args.lateral_teachers or args.causal_teachers else [])+
         ([prior/'joint_state_training_data.npz',prior/'receipt.json'] if args.expert_weights else [])+
         ([teacher_stage/'results.json',teacher_stage/'receipt.json',
             BASE/'learn_complete_joint_teachers.py',BASE/'deep_joint_credit_policy.py'] if args.adaptive_teachers else
             [prior/'joint_state_training_data.npz',prior/'receipt.json',
                                      prior/'results.json'] if args.direct_solve else []),
        dict(wall_s=720,workers=len(widths),fit_paths=path_count if args.adaptive_teachers else 72,
             full_training_check_paths=path_count,shared_widths=widths,
             maximum_lsmr_iterations=1800,action_fit_error_screen=1e-5,
             direct_float64_solve=args.direct_solve,diagonal_regularizer=1e-12 if args.direct_solve else None,
             exact_filtered_heading_replay=True),
        ('Distill compact20ms complete-return teachers into continuous OBSERVABLE-STATE '
         'expert weights; no offline block support,time or path index is a policy input. '
         'Use the1.2 width selected by the completed training-neighborhood comparison, '
         'then evaluate all88 training paths and current lateral displacement protection. '
         'Teacher parameters may have any finite number of basis columns; their exact '
         'stored latent actions,basis and actual parent outputs must agree. ' if args.causal_teachers else '')+
        ('Matched representation control: reuse EXACTLY the saved lateral-aware states, '
         'parent,teachers,widths and full80 screens. Learn three even state-dependent '
         'expert weights, then multiply by both learned policies differences at each '
         'actual feedback state. Steering difference is odd; all weights are even. '
         'The state mapper never receives time,path ID or offline teacher knots. '
         'Compare to additive residual fitting without recollecting data or changing '
         'a teacher. Full80 reproduction is required but cannot alone establish '
         'off-trajectory robustness; use new training-neighborhood checks next. ' if args.expert_weights else '')+
        (f'Freeze the hash-verified current model as parent, fit ONLY new three-action residuals '
         f'from teachers{sorted(teachers)}; other{path_count-len(teachers)} complete trajectories have zero targets. '
         'Reconstruct teacher prefix/suffix observations and exact latent actions, checking '
         'saved parent latents and physical action decoding. Reuse the declared current/retained '
         'shape envelopes only on those unresolved training cases. Preserve every other '
         'current path, full-pool time/cruise and total reverse/exit/energy screens. A teacher '
         f'tradeoff or fitted labels alone cannot pass the aggregate screen. Run fixed widths{widths} '
         'by float64 LU; no PPO, no deployed time/path inputs, no final adoption. ' if args.adaptive_teachers else
        ('Prior fits both reached the1800-step solver cap, with numerical condition warnings. '
         'Those failures do not identify a representation limit. Reuse exactly the saved '
         'three-action data and two widths; test float64 sparse direct collocation solve '
         'with a declared1e-12 diagonal regularizer and identical fit/physical screens. '
         'Do not recollect states or repeat any previous rollout. ' if args.direct_solve else '')+
        'Training midpoint trajectories showed narrow local corrections do not reach all '
        'nearby states. Reuse the COMPLETE current72-path learned policy as three-action '
        'state labels relative to pre-teacher5ab; reconstruct each saved state and verify '
        'actual decoded actions exactly. Learn shared compact collocation kernels whose '
        'coefficients jointly fit active and zero targets; zeros are fitted by the shared '
        'function, not by shrinking every center to the nearest protected sample. Use '
        'two preregistered broader widths and a dimension-appropriate Wendland kernel. '
        'Inputs remain observable geometry,physics,previous action and target innovation, '
        'never path/angle IDs,time or teacher schedule. This is OFFLINE joint fitting, '
        'not PPO. Reject inaccurate fits before simulation; otherwise execute all80 '
        'training paths including midpoint cases. Preserve every parent corner and '
        'the original physical/completion constraints before saving any research model.'))
    write_json(stage/'baseline.json',data)
    started=time.perf_counter();pool=None;outcomes=[]
    try:
        parent=torch.load(parent_file,map_location='cpu',weights_only=False)
        current=parent if args.adaptive_teachers else torch.load(current_file,map_location='cpu',weights_only=False)
        if args.expert_weights:
            with np.load(prior/'joint_state_training_data.npz') as z:training={k:z[k].copy() for k in z.files}
        elif args.adaptive_teachers:
            training=collect_data(current,parent,baseline['records'],teachers,teacher_stage,expert_state)
            np.savez_compressed(stage/'joint_state_training_data.npz',**training)
            a,b=state_pair(training['observations'],training['innovations'])
            symmetric=np.max(abs(a-b),axis=1)<1e-12
            assert not np.any(abs(training['labels'][symmetric,0])>1e-8),'Unobservable signed teacher labels'
            write_json(stage/'data_audit.json',dict(states=len(training['labels']),
                changed_paths=sorted(teachers),zero_target_paths=path_count-len(teachers),
                parent_sha256=sha(parent_file),all_decoded_actions_exact=True,
                time_and_path_metadata_used_as_inputs=False))
        elif args.direct_solve:
            with np.load(prior/'joint_state_training_data.npz') as z:training={k:z[k].copy() for k in z.files}
        else:
            original=[r for r in baseline['records'] if r['path']<200]
            training=collect_data(current,parent,original)
            np.savez_compressed(stage/'joint_state_training_data.npz',**training)
        pool=mp.get_context('spawn').Pool(len(widths),initialize,(stage,parent,baseline,training,started+695,args.direct_solve,teachers,
                                                       expert_state if weight_mode else None))
        for r in pool.imap_unordered(fit_and_evaluate,widths):
            outcomes.append(r);write_json(stage/'results.json',dict(candidates=outcomes,adopted=False))
        pool.close();pool.join();pool=None
        good=[r for r in outcomes if r['research_acceptable']]
        chosen=min(good,key=lambda r:r['summary']['reverse_sum_deg']) if good else None
        result=dict(candidates=outcomes,adopted=False,goal_complete=False)
        if chosen:
            md=torch.load(stage/chosen['tag']/'candidate.pt',map_location='cpu',weights_only=False)
            torch.save(md,stage/'research_candidate.pt');result['best']=chosen['tag']
        write_json(stage/'results.json',result)
        receipt(stage,dict(status='completed_shared_teacher_kernel_learning',candidates=len(outcomes),
            fitting_passes=sum(r['fit_pass'] for r in outcomes),best=chosen['tag'] if chosen else None,
            wall_s=time.perf_counter()-started,adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_shared_teacher_kernel_learning',error=str(exc),
            completed=len(outcomes),wall_s=time.perf_counter()-started,goal_complete=False));raise


if __name__=='__main__':mp.freeze_support();main()
