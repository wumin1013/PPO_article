"""Transfer a joint function into the new observation metric, then verify real closure."""
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
from scipy.optimize import lsq_linear
from observable_joint_design import design,correction
from joint_teacher_state_policy import OUTPUT_SCALE
from joint_teacher_state_policy import features
from duration_joint_metrics import guards,penalty
from cruise_duration_credit import event_learning_guards
from learn_complete_joint_teachers import teacher_score,parameters_in_bounds
from learn_early_memory_joint_policy import worker_initialize,worker_evaluate,metric_functions
from learn_direct_observable_joint_policy import finish
from study_io import BASE,register,write_json,receipt
from search_complete_return import read


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--centered-bend',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_signed_heading_joint_recovery_20260920'
    metric_stage=BASE/'canonical_gate_band_bend_metric_joint_learning_20260920'
    old_history=read(source/'state_joint/results.json')['candidates']
    target=next(c for c in old_history if c['tag']=='geometry_secant_17')
    old_md=torch.load(source/'state_joint/candidate.pt',map_location='cpu',weights_only=False)
    old_md['coefficients']=np.asarray(target['coefficients']).reshape(-1,3)
    md=torch.load(metric_stage/'state_joint/candidate.pt',map_location='cpu',weights_only=False)
    md['coefficients']=old_md['coefficients'].copy()
    if args.centered_bend:
        capacity=read(source/'centered_bend_metric_capacity.json')
        md.update(observed_bend_metric_centers=np.asarray(capacity['centers']),
                  observed_bend_metric_weights=np.asarray(capacity['choices'][0]['weights']))
    data=read(source/'baseline.json');baseline=data['baseline']['records']
    support=read(source/'support_audit.json');choice=support['selected'];paths=choice['paths']
    selected=[r for r in baseline if r['path'] in paths]
    changed_paths=read(metric_stage/'metric_zero_support_audit.json')['affected_paths']
    zero_paths=[p for p in paths if p not in changed_paths]
    parent_file=BASE/'canonical_gate_band_postpeak_memory_joint_learning_20260920/research_candidate.pt'
    stage=register('canonical_gate_band_centered_bend_function_transfer_20260920' if args.centered_bend else
                   'canonical_gate_band_metric_function_transfer_20260920',
        [__file__,BASE/'observable_joint_design.py',BASE/'observable_memory_joint_policy.py',
         BASE/'joint_teacher_state_policy.py',BASE/'cruise_duration_credit.py',
         source/'state_joint/results.json',source/'state_joint/candidate.pt',source/'geometry_secant_17_replay.npz',
         metric_stage/'receipt.json',metric_stage/'state_joint/candidate.pt',
         metric_stage/'cruise_event_credit_audit.json',metric_stage/'event_model_conflicts.json',parent_file]+
         ([source/'centered_bend_metric_capacity.json',BASE/'audit_bend_metric_capacity.py',
           BASE/'canonical_gate_band_metric_function_transfer_20260920/unbounded_function_capacity.json'] if args.centered_bend else []),
        dict(wall_s=180,workers=4,ridge_controls=[0.,1e-10,1e-7],coefficient_bounds=[-1.,1.],
             maximum_complete_parameter_vectors=3),
        ('Correct an identified representation construction error: the prior distance '
         'centers reused context-normalization means from mixed support, not the desired '
         'radial-kernel states. This suppressed useful pre-jump support by91-97% and '
         'removed it entirely on104. Recenter the new metric on actual target-support '
         'observations and recompute the smallest separating weight. No angle/speed '
         'table or hard physical bound changes. Rerun only the changed function; reuse '
         'old physical traces if all decoded actions are exactly reproduced. '
         if args.centered_bend else '')+
        'The new geometry metric removes six conflicting support paths, but reusing old '
        'coefficients changes the intended joint function: path14 loses two cruise samples '
        'and post-apex speed. Exact cruise-event margins expose response, yet the measured '
        '20-direction model stays infeasible. Test representation transfer rather than '
        'extending that model: fit ALL THREE residual outputs on the actual saved '
        'geometry_secant_17 states with bounded coefficients in the NEW metric. Freeze '
        'the early kernel, verify exact old-function latent reconstruction, and compare '
        'three registered ridge levels. Do not fit feed alone or adopt labels as a policy. '
        'Run each distinct fitted function through the seven actual affected complete '
        'routes; other81 routes have certified zero support. Apply every original '
        'shape/time/cruise/physics guard. This is offline joint function fitting followed '
        'by full closed-loop validation, not PPO, and cannot establish generalization.')
    started=time.perf_counter();deadline=started+165;folder=stage/'state_joint';folder.mkdir()
    write_json(stage/'baseline.json',data);write_json(stage/'support_audit.json',support)
    with np.load(source/'geometry_secant_17_replay.npz') as z:replay={k:z[k].copy() for k in z.files}
    old_design=design(old_md,replay);new_design=design(md,replay)
    if args.centered_bend:
        with np.load(source/'replayed_states.npz') as z:
            pf=features(md,z['observations'],z['innovations'])
            live=sorted(np.unique(z['path_ids'][np.any(pf!=0.,axis=(1,2))]).tolist())
        assert live==changed_paths
        write_json(stage/'metric_zero_support_audit.json',dict(parent_paths=len(baseline),affected_paths=live,
            all_unaffected_paths=[r['path'] for r in baseline if r['path'] not in live],
            exact_zero_support=True,corrected_centers=md['observed_bend_metric_centers'].tolist()))
    target_delta=correction(old_md,old_design)
    reconstructed=(replay['parent_latents']+target_delta).astype(replay['current_latents'].dtype)
    assert np.array_equal(reconstructed,replay['current_latents']),'Exact old function design mismatch'
    fit_mask=np.isin(replay['path_ids'],changed_paths)
    index=np.arange(len(md['coefficients'])).reshape(3,7,5)
    variable=index[:,:,[0,1,3,4]].reshape(-1);fixed=index[:,:,2].reshape(-1)
    original=md['coefficients'].copy();seed=original.copy();seed[variable]=0.
    md['coefficients']=seed
    fixed_delta=correction(md,new_design)
    retained=[next(r['baseline_shape'] for r in data['retained'] if r['path']==b['path']) for b in selected]
    _,old_guards=metric_functions(data,selected,stage,md['parent'],deadline)
    def score(records):return float(sum(teacher_score(r,b,o,barrier=False)+penalty(r) for r,b,o in zip(records,selected,retained)))
    def constraints(records):return np.concatenate([old_guards(records)]+[guards(r,b) for r,b in zip(records,selected)])
    history=copy.deepcopy(old_history[:2]);zero,early=history
    md['reused_candidates']=len(history)
    def feasible(c):return parameters_in_bounds(c['coefficients'],-1.,1.) and min(c['measured_inequalities'])>=-1e-8
    def best():return min(filter(feasible,history),key=lambda c:c['score'])
    pool=None;fits=[];cache=set()
    try:
        pool=mp.get_context('spawn').Pool(4,worker_initialize,(md,selected,folder,deadline))
        for ridge in (0.,1e-10,1e-7):
            coefficient=seed.copy();heads=[]
            for h in range(3):
                a=new_design[fit_mask][:,variable,h]*OUTPUT_SCALE[h]
                y=(target_delta-fixed_delta)[fit_mask,h]
                aa=np.vstack([a,np.sqrt(ridge)*np.eye(len(variable))])
                yy=np.r_[y,np.sqrt(ridge)*original[variable,h]]
                solved=lsq_linear(aa,yy,bounds=(-1.,1.),tol=1e-13,lsmr_tol=1e-13,max_iter=200)
                coefficient[variable,h]=solved.x
                heads.append(dict(head=h,success=bool(solved.success),max_error=float(abs(a@solved.x-y).max()),
                    error_norm=float(np.linalg.norm(a@solved.x-y)),rank=int(np.linalg.matrix_rank(a))))
            assert np.array_equal(coefficient[fixed],original[fixed])
            md['coefficients']=coefficient
            latent=(replay['parent_latents']+correction(md,new_design)).astype(replay['current_latents'].dtype)
            row=dict(ridge=ridge,heads=heads,fit_decisions=int(fit_mask.sum()),
                changed_float32_components=int((latent[fit_mask]!=replay['current_latents'][fit_mask]).sum()),
                maximum_latent_error=float(abs(latent[fit_mask]-replay['current_latents'][fit_mask]).max()))
            fits.append(row);write_json(stage/'fit_audit.json',dict(old_design_exact=True,fits=fits))
            u=coefficient.reshape(-1);key=tuple(np.round(u,12))
            if key in cache:continue
            cache.add(key);tag=f'joint_transfer_{len(history)-2}'
            if row['changed_float32_components']==0:
                measured=[copy.deepcopy(r) for r in target['records'] if r['path'] in changed_paths]
                row['exact_action_identity_reused']=True
            else:
                measured=pool.map(worker_evaluate,[(p,u,tag) for p in changed_paths])
            by_path={r['path']:r for r in measured}
            records=[by_path.get(r['path'],dict(copy.deepcopy(r),reused_exact_zero_support=True)) for r in selected]
            original_guards=constraints(records)
            c=dict(tag=tag,coefficients=u.tolist(),records=records,score=score(records),
                measured_inequalities=original_guards.tolist(),
                learning_inequalities=event_learning_guards(records,selected,original_guards).tolist(),
                actual_rerun_paths=[] if row.get('exact_action_identity_reused') else changed_paths,
                exact_action_identity_reused_paths=changed_paths if row.get('exact_action_identity_reused') else [],
                exact_zero_support_paths=zero_paths)
            history.append(c);row.update(tag=tag,score=c['score'],minimum_guard=float(original_guards.min()),
                violation_norm=float(np.linalg.norm(np.minimum(original_guards,0.))))
            write_json(stage/'fit_audit.json',dict(old_design_exact=True,fits=fits))
            write_json(folder/'results.json',dict(candidates=history,best=best()['tag']))
            print('JOINT_FUNCTION_TRANSFER',row,flush=True)
        pool.close();pool.join();pool=None
        finish(stage,md,baseline,data,best(),zero,history,fits,choice,parent_file,started)
    except TimeoutError:
        if pool is not None:pool.terminate();pool.join();pool=None
        finish(stage,md,baseline,data,best(),zero,history,fits,choice,parent_file,started,budget_exhausted=True)
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_metric_function_transfer',error=str(exc),wall_s=time.perf_counter()-started,
            adopted=False,goal_complete=False));raise


if __name__=='__main__':mp.freeze_support();main()
