"""Matched full-loop precision controls at the last three guarded joint iterates."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import copy
import time
import multiprocessing as mp
import numpy as np
import torch
from consolidate_joint_teacher_policy import collect_data
from observable_joint_design import design,correction
from duration_joint_metrics import guards,penalty
from cruise_duration_credit import event_learning_guards
from learn_intrinsic_joint_recovery import intrinsic_guards
from learn_complete_joint_teachers import teacher_score,parameters_in_bounds
from learn_early_memory_joint_policy import worker_initialize,worker_evaluate,metric_functions
from learn_direct_observable_joint_policy import finish
from study_io import BASE,register,write_json,receipt
from search_complete_return import read
from canonical_gate_state import checkpoint_environment_factory


def main():
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_intrinsic_curvature_joint_recovery_20260920'
    source_history=read(source/'state_joint/results.json')['candidates']
    controls=[next(c for c in source_history if c['tag']==f'intrinsic_joint_{i}') for i in (3,4,5)]
    md=torch.load(source/'state_joint/candidate.pt',map_location='cpu',weights_only=False)
    parent_file=BASE/'canonical_gate_band_postpeak_memory_joint_learning_20260920/research_candidate.pt'
    data=read(source/'baseline.json');baseline=data['baseline']['records']
    support=read(source/'support_audit.json');choice=support['selected'];paths=choice['paths']
    selected=[r for r in baseline if r['path'] in paths]
    active=read(source/'metric_zero_support_audit.json')['affected_paths']
    stage=register('canonical_gate_band_bounded_output_precision_probe_20260920',
        [__file__,BASE/'observable_memory_joint_policy.py',BASE/'observable_joint_design.py',
         BASE/'learn_intrinsic_joint_recovery.py',BASE/'intrinsic_curve_metrics.py',
         BASE/'joint_runtime.py',BASE/'joint_runtime_v2.py',BASE/'cruise_duration_credit.py',
         source/'receipt.json',source/'state_joint/candidate.pt',source/'state_joint/results.json',
         source/'remaining_guard_failures.json',parent_file,BASE/'consolidate_joint_teacher_policy.py',
         BASE/'canonical_gate_band_joint_output_precision_probe_20260920/receipt.json'],
        dict(wall_s=120,workers=4,matched_parameter_points=[3,4,5],maximum_new_vectors=3,
             unchanged_parameter_values=True,changed_component='outer learned residual sum dtype only'),
        'The previous preparation was stopped for a wall-budget overrun before any '
        'new physical control. Avoid deep replay of all88 parents: the decoder explicitly '
        'converts both input precisions to float64, so adding a zero float64 residual '
        'preserves every finite float32 latent exactly, globally. Check this identity '
        'on512 fixed numerical controls. Replay only the two remaining failing paths '
        '105/304 with a per-decision40s preparation deadline. The three matched full '
        'physical controls are still pending and are not repeats. '
        'Six raw-curvature guarded corrections reduced violation norm to6.03e-7, '
        'leaving four tiny original guards. Test, do not assume, final float32 action '
        'rounding at the same last three parameter vectors. Keep all parent networks, '
        'state dtype, guide, actions20ms, physics1ms, all constraints and coefficients '
        'unchanged. Return only the OUTER learned residual sum in float64; decoder '
        'already promotes inputs to float64. First verify zero-residual decoded identity '
        'on all88 stored parent paths and reconstruct the latest float32 actions '
        'exactly on saved states. Run three changed complete-loop functions on the '
        'seven supported routes; use existing float32 trajectories as matched controls. '
        'This is a physical numerical-representation diagnostic, not PPO or proof that '
        'rounding explains other reverse phenomena. Select only all-original-plus-raw '
        'curvature guarded outcomes; otherwise retain the protected parent.')
    started=time.perf_counter();deadline=started+105;folder=stage/'state_joint';folder.mkdir()
    write_json(stage/'baseline.json',data);write_json(stage/'support_audit.json',support)
    write_json(stage/'metric_zero_support_audit.json',read(source/'metric_zero_support_audit.json'))
    replay_md=copy.deepcopy(md);replay_md['coefficients']=np.asarray(controls[-1]['coefficients']).reshape(-1,3)
    replay_records=[r for r in controls[-1]['records'] if r['path'] in (105,304)]
    try:
        replay=collect_data(replay_md,replay_md['parent'],replay_records,include_latents=True,deadline=started+40.)
    except TimeoutError:
        receipt(stage,dict(status='budget_stopped_precision_replay',wall_s=time.perf_counter()-started,
            new_physical_vectors=0,adopted=False,goal_complete=False));return
    intended=replay['parent_latents'].astype(float)+correction(replay_md,design(replay_md,replay))
    assert np.array_equal(intended.astype(replay['current_latents'].dtype),replay['current_latents'])
    error=intended-replay['current_latents']
    md['residual_output_precision']='float64';md['coefficients']=np.zeros_like(md['coefficients'])
    decoder=checkpoint_environment_factory('gate_band_canonical_kernel')(np.array([[0.,0.],[50.,0.]])).decode_action
    inputs=np.random.default_rng(20260920).normal(size=(512,3)).astype(np.float32)
    inputs[0]=0.;inputs[1]=np.finfo(np.float32).eps
    assert all(np.array_equal(decoder(x),decoder(x.astype(np.float64)+np.zeros(3))) for x in inputs)
    write_json(stage/'matched_precision_audit.json',dict(controls=[c['tag'] for c in controls],
        zero_residual_identity_applies_to_paths=len(baseline),decoder_identity_controls=len(inputs),
        zero_residual_identity='The decoder casts to float64 before all nonlinear operations; zero residual preserves each finite float32 value exactly.',
        roundoff_replay_paths=[105,304],roundoff_replay_decisions=len(replay['path_ids']),
        exact_float32_action_reconstruction=True,maximum_outer_roundoff_per_head=np.max(abs(error),axis=0).tolist(),
        nonzero_roundoff_components=int(np.count_nonzero(error)),new_physical_runs_for_replay=0))
    _,old_guards=metric_functions(data,selected,stage,md['parent'],deadline)
    retained=[next(r['baseline_shape'] for r in data['retained'] if r['path']==b['path']) for b in selected]
    def score(records):return float(sum(teacher_score(r,b,o,barrier=False)+penalty(r) for r,b,o in zip(records,selected,retained)))
    def constraints(records):return np.concatenate([old_guards(records)]+[guards(r,b) for r,b in zip(records,selected)])
    zero=copy.deepcopy(source_history[0]);history=[zero];md['reused_candidates']=1
    def feasible(c):return parameters_in_bounds(c['coefficients'],-1.,1.) and min(c['augmented_measured_inequalities'])>=-1e-8
    def best():return min(filter(feasible,history),key=lambda c:c['score'])
    pool=None;comparisons=[]
    try:
        pool=mp.get_context('spawn').Pool(4,worker_initialize,(md,selected,folder,deadline))
        for control in controls:
            u=np.asarray(control['coefficients']);tag='float64_'+control['tag']
            measured=pool.map(worker_evaluate,[(p,u,tag) for p in active])
            by_path={r['path']:r for r in measured}
            records=[by_path.get(r['path'],dict(copy.deepcopy(r),reused_exact_zero_support=True)) for r in selected]
            original=constraints(records);extra=intrinsic_guards(records,selected)
            c=dict(tag=tag,coefficients=u.tolist(),records=records,score=score(records),
                measured_inequalities=original.tolist(),intrinsic_inequalities=extra.tolist(),
                augmented_measured_inequalities=np.r_[original,extra].tolist(),
                augmented_learning_inequalities=np.r_[event_learning_guards(records,selected,original),extra].tolist(),
                actual_rerun_paths=active,exact_zero_support_paths=[p for p in paths if p not in active])
            history.append(c)
            row=dict(control=control['tag'],candidate=tag,float32_score=control['score'],float64_score=c['score'],
                float32_violation_norm=float(np.linalg.norm(np.minimum(control['augmented_measured_inequalities'],0.))),
                float64_violation_norm=float(np.linalg.norm(np.minimum(c['augmented_measured_inequalities'],0.))),
                float64_minimum_guard=float(min(c['augmented_measured_inequalities'])),all_guards_pass=feasible(c))
            comparisons.append(row);write_json(stage/'matched_results.json',comparisons)
            write_json(folder/'results.json',dict(candidates=history,best=best()['tag']))
            print('OUTPUT_PRECISION',row,flush=True)
        pool.close();pool.join();pool=None
        finish(stage,md,baseline,data,best(),zero,history,comparisons,choice,parent_file,started)
    except TimeoutError:
        if pool is not None:pool.terminate();pool.join();pool=None
        finish(stage,md,baseline,data,best(),zero,history,comparisons,choice,parent_file,started,budget_exhausted=True)
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_joint_output_precision_probe',error=str(exc),wall_s=time.perf_counter()-started,
            adopted=False,goal_complete=False));raise


if __name__=='__main__':mp.freeze_support();main()
