"""Recenter late joint-action credit at two measured early-window improvements."""
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
from gate_band_state import GateBandRuntime
from saved_memory_action_identity import SavedMemoryActionIdentity
from audit_context_restoration_scaling import solve_correction
from duration_joint_metrics import guards,penalty
from shape_event_credit import shape_event_learning_guards
from learn_intrinsic_joint_recovery import intrinsic_guards
from learn_complete_joint_teachers import teacher_score,parameters_in_bounds
from learn_early_memory_joint_policy import worker_initialize,worker_evaluate,metric_functions
from learn_direct_observable_joint_policy import finish
from study_io import BASE,register,write_json,receipt,sha
from search_complete_return import read
from consolidate_joint_teacher_policy import collect_data
from joint_teacher_state_policy import features,state_pair
from intrinsic_curve_metrics import intrinsic_metrics


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preturn',action='store_true')
    parser.add_argument('--postpeak',action='store_true')
    parser.add_argument('--bounded-postpeak',action='store_true')
    parser.add_argument('--reference-recenter',action='store_true')
    args=parser.parse_args();started=time.perf_counter()
    assert sum([args.preturn,args.postpeak,args.bounded_postpeak,args.reference_recenter])<=1
    torch.set_num_threads(1)
    source=BASE/('canonical_gate_band_reference_support_closed_loop_probe_20260920' if args.reference_recenter else
                'canonical_gate_band_postpeak_separated_joint_recovery_20260920' if args.bounded_postpeak else
                'canonical_gate_band_late_support_aliasing_probe_20260920' if args.postpeak else
                'canonical_gate_band_active_window_local_joint_recovery_20260920' if args.preturn else
                'canonical_gate_band_active_reversal_window_joint_credit_20260920')
    parent_stage=BASE/'canonical_gate_band_reversal_window_local_recovery_20260920'
    parent_file=parent_stage/'research_candidate.pt'
    assert sha(parent_file)==('132e3914433db31aa72c0cf35de80b6d0b010b1a49f82d8ca30ab79776b6f8dd'
        if args.reference_recenter else read(source/'receipt.json')['protected_parent_sha256' if args.postpeak else 'source_model_sha256'])
    model_file=source/('bounded_51.pt' if args.reference_recenter else 'radius_0.97.pt' if args.postpeak else 'initial_policy.pt')
    md=torch.load(model_file,map_location='cpu',weights_only=False)
    data=read(source/'baseline.json');baseline=data['baseline']['records']
    support_source=(BASE/'canonical_gate_band_postpeak_bounded_joint_recovery_20260920' if args.reference_recenter else
                    BASE/'canonical_gate_band_preturn_supported_joint_recovery_20260920' if args.postpeak else source)
    support=read(support_source/'support_audit.json');choice=support['selected'];paths=choice['paths']
    selected=[r for r in baseline if r['path'] in paths]
    with np.load(parent_stage/'active_window_affected_replay.npz') as z:replay={k:z[k].copy() for k in z.files}
    history=copy.deepcopy(read(support_source/'state_joint/results.json')['candidates'])
    if args.reference_recenter:
        history=history[:1]
        warm=read(source/'results.json')['controls'][-1]
        warm['exact_decoded_identity_paths']=warm['exact_decoded_identity_paths']
        support['postpeak_kernel']['distance_frame']='observed_reference_orientation_with_geometry_weights'
    if args.postpeak:
        history=history[:1]
        warm=read(source/'results.json')['controls'][0]
        current=collect_data(md,md['parent'],warm['records'],include_latents=True,deadline=started+220.)
        target=next(r for r in warm['records'] if r['path']==104)
        with np.load(target['trace']) as z:ref,rows=z['reference'],z['rows']
        u,v=np.diff(ref,axis=0);u/=np.linalg.norm(u);v/=np.linalg.norm(v)
        cross=u[0]*v[1]-u[1]*v[0];sign=np.sign(cross)
        heading=np.unwrap(np.arctan2(rows[:,2:4]@(sign*np.array([-u[1],u[0]])),rows[:,2:4]@u))
        angle=abs(np.arctan2(cross,u@v));valid=np.linalg.norm(rows[:,2:4],axis=1)>5.
        first=int(np.flatnonzero((heading>angle+np.deg2rad(.5))&valid)[0])
        decision=first//20-1
        assert decision*20>intrinsic_metrics(rows)['peak_ms']
        raw,_=state_pair(current['observations'],current['innovations'])
        index=np.flatnonzero((current['path_ids']==104)&(current['decisions']==decision)).item()
        center=(raw[index]-md['per_center_mean'][0])/md['per_center_std'][0]
        md['centers']=np.vstack([md['centers'],center])
        for key in ('per_center_mean','per_center_std','per_center_distance_weights'):
            md[key]=np.vstack([md[key],md[key][0]])
        md['radii']=np.r_[md['radii'],md['radii'][0]]
        old=np.array(md['coefficients']).reshape(3,5,3);expanded=np.zeros((3,6,3));expanded[:,:5]=old
        warm['coefficients']=expanded.ravel().tolist();md['coefficients']=np.zeros((18,3))
        history[0]['coefficients']=[0.]*54
        with np.load(parent_stage/'active_window_training_states.npz') as z:
            f=features(md,z['observations'],z['innovations'])
            new_paths=sorted(np.unique(z['path_ids'][np.any(f[:,-1]!=0.,axis=1)]).tolist())
        assert set(new_paths)<=set(paths),('Postpeak support needs expanded evaluator',new_paths)
        support['postpeak_kernel']=dict(training_selector_ms=decision*20,
            physical_curvature_peak_ms=intrinsic_metrics(rows)['peak_ms'],first_overshoot_ms=first,
            baseline_support_paths=new_paths,no_runtime_time_or_path_input=True)
        choice['variable_kernels']=list(range(6))
    if args.preturn:
        model=read(source/'joint_74_local_credit.json')
        reuse_tags=['reused_parent','joint_74']+[t for pair in model['pairs'] for t in pair]
        history=[c for c in history if c['tag'] in reuse_tags]
        origin=next(c for c in history if c['tag']=='joint_74')
        actor=copy.copy(md);actor['coefficients']=np.array(origin['coefficients']).reshape(-1,3)
        current=collect_data(actor,md['parent'],origin['records'],include_latents=True,deadline=started+280.)
        raw,_=state_pair(current['observations'],current['innovations'])
        index=np.flatnonzero((current['path_ids']==104)&(current['decisions']==14)).item()
        center=(raw[index]-md['per_center_mean'][0])/md['per_center_std'][0]
        md['centers']=np.vstack([md['centers'],center])
        for key in ('per_center_mean','per_center_std','per_center_distance_weights'):
            md[key]=np.vstack([md[key],md[key][0]])
        md['radii']=np.r_[md['radii'],md['radii'][0]]
        md['coefficients']=np.zeros((15,3))
        for c in history:
            old=np.array(c['coefficients']).reshape(3,4,3);expanded=np.zeros((3,5,3));expanded[:,:4]=old
            c['coefficients']=expanded.ravel().tolist()
        with np.load(parent_stage/'active_window_training_states.npz') as z:
            f=features(md,z['observations'],z['innovations'])
            new_paths=sorted(np.unique(z['path_ids'][np.any(f[:,-1]!=0.,axis=1)]).tolist())
        assert set(new_paths)<=set(paths),('New support requires expanded evaluator',new_paths)
        support['preturn_kernel']=dict(training_selector_ms=280,baseline_support_paths=new_paths,
            feature_on_measured_anchor=features(md,current['observations'][index],current['innovations'][index])[-1].tolist(),
            no_runtime_time_or_path_input=True)
        choice['variable_kernels']=list(range(5))
    identity=SavedMemoryActionIdentity(md,replay,GateBandRuntime.decode_action)
    md['reused_candidates']=len(history);zero=history[0]
    dimensions=len(zero['coefficients']);assert dimensions==(54 if args.postpeak or args.bounded_postpeak or args.reference_recenter else 45 if args.preturn else 36)
    assert all(identity.unchanged(p,np.zeros(dimensions)) for p in paths)
    stage=register('canonical_gate_band_reference_supported_joint_learning_20260920' if args.reference_recenter else
                  'canonical_gate_band_postpeak_bounded_joint_recovery_20260920' if args.bounded_postpeak else
                  'canonical_gate_band_postpeak_separated_joint_recovery_20260920' if args.postpeak else
                  'canonical_gate_band_preturn_supported_joint_recovery_20260920' if args.preturn else
                   'canonical_gate_band_active_window_local_joint_recovery_20260920',
        [__file__,parent_file,source/'receipt.json',model_file,source/'baseline.json',
         support_source/'support_audit.json',support_source/'state_joint/results.json',
         (source/'results.json' if args.reference_recenter else support_source/'local_constraint_conflict_audit.json' if args.postpeak else
          source/('local_recovery_audit.json' if args.preturn or args.bounded_postpeak else 'joint_proposals.json')),
         parent_stage/'active_window_affected_replay.npz',BASE/'audit_context_restoration_scaling.py',
         BASE/'saved_memory_action_identity.py',BASE/'observable_memory_joint_policy.py',
         BASE/'joint_teacher_state_policy.py',BASE/'duration_joint_metrics.py',BASE/'shape_event_credit.py',
         BASE/'learn_intrinsic_joint_recovery.py',BASE/'learn_early_memory_joint_policy.py',
         BASE/'learn_direct_observable_joint_policy.py']+
         ([source/'radius_0.97_local_credit.json'] if args.bounded_postpeak else []),
        dict(wall_s=240 if args.reference_recenter else 90 if args.bounded_postpeak else 240 if args.postpeak else 300 if args.preturn else 180,workers=4,
             anchors=['reference_full'] if args.reference_recenter else ['radius_0.97'] if args.postpeak or args.bounded_postpeak else ['joint_74'] if args.preturn else ['joint_73','joint_74'],
             late_parameter_indices=list(range(6,12)),new_preturn_parameter_indices=[12,13,14,27,28,29,42,43,44] if args.preturn else [],
             response_steps=[.001,.0025,.0025] if args.reference_recenter else [.0002,.0005,.0005],symmetric_responses=48 if args.reference_recenter else 0 if args.bounded_postpeak else 48 if args.postpeak else 18 if args.preturn else 24,maximum_corrections_per_anchor=3,
             bounded_capacity_radii=[64.,128.,256.] if args.bounded_postpeak else [],
             matched_step_fractions=[.25,.5,1.] if args.bounded_postpeak else [],
             correction_radii=[4.,16.,64.],minimum_retained_lateral_fraction=.75),
        ('Matched physical controls show the observed-reference geometry representation '
         'reduces the full-step worst violation0.701 to0.165, but reverse, spatial '
         'peak curvature and1ms cruise still fail. Do not adopt that representation '
         'control as a learned solution. Recenter all24 pre-turn/late/postpeak '
         'three-action partial responses at its ACTUAL complete trajectory using '
         '48 new symmetric samples, then at most3 constrained joint corrections. '
         'Steps0.001/0.0025/0.0025 supply real decoded-action resolution; all '
         'coefficient bounds and true original acceptance remain unchanged. '
         'Keep the early learned benefit and at least75 percent negative-lateral '
         'gain.240s. This is complete closed-loop parameter learning with the '
         'existing guide/projector/brake, not PPO or a claimed rule-free policy.' if args.reference_recenter else
        'The postpeak24-column local model is infeasible inside the previous scaled '
         'radius64 but feasible at256. Many parameter bounds are active in the '
         'minimum-slack solution. Thus the last failure was partly the correction '
         'trust box, not proof that this representation or physics cannot work. '
         'Reuse every measured response without new sampling. Choose the first '
         'feasible radius64/128/256, validate its coefficients inside[-1,1], and '
         'measure quarter/half/full steps as three matched complete policies. '
         'Then at most3 true-return secant corrections around the closest full '
         'result.90s; all original physical/shape/cruise constraints and75-percent '
         'lateral-gain protection remain unchanged. Do not adopt a model prediction '
         'or repeat old physical points.' if args.bounded_postpeak else
        'The matched3/5-percent radius controls restore exact parent actions on15/207/307 '
         'without removing104 early lateral benefit; the104 shape conflict remains. '
         'Use the less changed3-percent training support radius, not a deployed probe. '
         'The sampled local conflict binds spatial-XY peak curvature and overshoot duration. '
         'Add an observed-state kernel after the raw physical curvature peak332ms and '
         'before first heading overshoot376ms (340ms selects the training observation). '
         'Learn all three actions and two causal memories of this kernel jointly with '
         'the nine pre-turn parameters and six late instantaneous parameters:48 new '
         'central responses and at most3 measured corrections,240s. Full saved-state '
         'action replay must validate zero appended weights before reuse. Retain all '
         'current physical/shape/time/cruise guards and at least75 percent lateral gain. '
         'No runtime time/path input, phase switch, template, PPO, sealed testing or '
         'automatic adoption of a diagnostic. Only a fully audited feasible learned '
         'function can become a protected research candidate.' if args.postpeak else
        ('The measured six-parameter late subspaces were infeasible at BOTH actual '
         'improved points. Add a compact observable-state center selected before '
         'the main turn (280ms in the failed joint_74 training trajectory), with '
         'three instantaneous and six causal-memory coefficients. Preserve the '
         'previous four state kernels/parameters. Reuse the12 measured late '
         'responses only after replaying their FULL saved policy actions through '
         'the augmented actor with exactly zero added weights. Measure18 new '
         'pre-turn responses, then solve the combined15-parameter recovery. '
         'Adding a center is a state/history representation change, not a '
         'deployed timed phase or shape template.300s including exact reuse '
         'audits, at most3 new corrections. ' if args.preturn else '')+
        'The active-window actor learns during wrong-way motion. Actual joint_73 '
        'and joint_74 reduce negative lateral movement0.000598/0.002457mm, but '
        'curve/speed guards regress; the larger point also slightly affects307. '
        'Retain the audited132e391 parent. At each actual improved point measure '
        'six late instantaneous steering/feed/preview partial responses symmetrically '
        '(24 total), keeping earlier learned action fixed. Recenter all guard values '
        'and gradients, including307 rather than dismissing tiny effects. Solve at '
        'most3 minimal joint corrections per point and update from real secants. '
        'Keep at least75 percent of each negative-lateral improvement. Predictive '
        'positive margin1e-6 with exact-zero-margin fallback if infeasible; original '
        'acceptance unchanged. Reuse all old coefficient points exactly.180s. '
        'Failure means this late subspace is insufficient under measured conditions, '
        'not physical impossibility, and triggers support/credit review rather than '
        'more identical correction iterations. Closed-loop parameter learning, not PPO.'))
    deadline=started+(225. if args.reference_recenter else 80. if args.bounded_postpeak else 225. if args.postpeak else 285. if args.preturn else 165.);pool=None;blocks=[]
    folder=stage/'state_joint';folder.mkdir()
    write_json(stage/'baseline.json',data);write_json(stage/'support_audit.json',support)
    torch.save(md,stage/'initial_policy.pt')
    _,old_guards=metric_functions(data,selected,stage,md['parent'],deadline)
    retained=[next(o['baseline_shape'] for o in data['retained'] if o['path']==r['path']) for r in selected]
    def entry(records,u,tag,actual,unchanged):
        original=np.concatenate([old_guards(records)]+[guards(r,b) for r,b in zip(records,selected)])
        extra=intrinsic_guards(records,selected)
        score=float(sum(teacher_score(r,b,o,barrier=False)+penalty(r)+
            (4*(r['shape']['reverse_lateral_total_mm']/.005)**2 if r['success'] else 1e6)
            for r,b,o in zip(records,selected,retained)))
        return dict(tag=tag,coefficients=np.asarray(u).tolist(),records=records,score=score,
            lateral_mm=sum(r['shape']['reverse_lateral_total_mm'] if r['success'] else 1. for r in records),
            measured_inequalities=original.tolist(),augmented_measured_inequalities=np.r_[original,extra].tolist(),
            augmented_learning_inequalities=np.r_[shape_event_learning_guards(records,selected,original),extra].tolist(),
            actual_rerun_paths=actual,exact_zero_support_paths=unchanged)
    def feasible(c):return parameters_in_bounds(c['coefficients'],-1.,1.) and min(c['augmented_measured_inequalities'])>=-1e-8
    def best():return min(filter(feasible,history),key=lambda c:c['score'])
    assert entry(selected,np.zeros(dimensions),'check',[],paths)['score']==zero['score']
    if args.postpeak or args.reference_recenter:
        c=entry(warm['records'],np.array(warm['coefficients']),'reference_full' if args.reference_recenter else 'radius_0.97',warm['actual_rerun_paths'],warm['exact_decoded_identity_paths'])
        history.append(c);md['reused_candidates']=len(history)
    cache={tuple(np.round(c['coefficients'],12)):c for c in history}
    try:
        if args.preturn or args.postpeak or args.reference_recenter:
            for number,c in enumerate(history[1:]):
                actor=copy.copy(md);actor['coefficients']=np.array(c['coefficients']).reshape(-1,3)
                collect_data(actor,md['parent'],c['records'],include_latents=False,deadline=deadline)
                if number%4==0:print('PRETURN_REUSE_REPLAY',number+1,len(history)-1,flush=True)
            write_json(stage/'augmented_actor_reuse_audit.json',dict(points=len(history)-1,
                all_saved_decoded_actions_exact=True,new_physical_runs=0,added_kernel_zero=True))
        pool=mp.get_context('spawn').Pool(4,worker_initialize,(md,selected,folder,deadline))
        def evaluate(u,kind):
            u=np.asarray(u);key=tuple(np.round(u,12))
            if key not in cache:
                assert parameters_in_bounds(u,-1.,1.)
                if time.perf_counter()>deadline:raise TimeoutError('Actual-state joint recovery budget')
                unchanged=[p for p in paths if identity.unchanged(p,u)]
                actual=[p for p in paths if p not in unchanged];tag=f'{kind}_{len(history)-1}'
                measured=pool.map(worker_evaluate,[(p,u,tag) for p in actual]);by={r['path']:r for r in measured}
                records=[by.get(r['path'],copy.deepcopy(r)) for r in selected]
                c=entry(records,u,tag,actual,unchanged);history.append(c);cache[key]=c
                write_json(folder/'results.json',dict(candidates=history,best=best()['tag']))
                if len(history)%5==0:print('ACTIVE_RECOVERY',len(history)-1,c['score'],min(c['augmented_measured_inequalities']),best()['tag'],flush=True)
            return cache[key]
        indices=(np.r_[np.arange(6,15),np.arange(30,33),np.arange(48,51),np.arange(15,18),np.arange(33,36),np.arange(51,54)]
                 if args.postpeak or args.bounded_postpeak or args.reference_recenter else np.r_[np.arange(6,12),[12,13,14,27,28,29,42,43,44]] if args.preturn else np.arange(6,12))
        steps=np.tile([.0002,.0005,.0005],len(indices)//3)
        if args.reference_recenter:steps*=5.
        for anchor in (('reference_full',) if args.reference_recenter else ('radius_0.97',) if args.postpeak or args.bounded_postpeak else ('joint_74',) if args.preturn else ('joint_73','joint_74')):
            origin=next(c for c in history if c['tag']==anchor);u=np.array(origin['coefficients']);pairs=[]
            if args.preturn:
                lookup={c['tag']:c for c in history}
                pairs=[(lookup[a],lookup[b]) for a,b in model['pairs']]
            for i,step in ([] if args.bounded_postpeak else zip(indices[6:] if args.preturn else indices,steps[6:] if args.preturn else steps)):
                offset=np.eye(dimensions)[i]*step
                pairs.append((evaluate(u+offset,anchor+'_plus'),evaluate(u-offset,anchor+'_minus')))
            lateral_limit=origin['lateral_mm']+.25*(zero['lateral_mm']-origin['lateral_mm'])
            score_limit=origin['score']+.25*(zero['score']-origin['score'])
            def values(c):return np.r_[c['augmented_learning_inequalities'],lateral_limit-c['lateral_mm']]
            if args.bounded_postpeak:
                local_model=read(source/'radius_0.97_local_credit.json')
                jac=np.array(local_model['jacobian']);gradient=np.array(local_model['loss_gradient'])
            else:
                jac=np.array([(values(a)-values(b))/2 for a,b in pairs]).T
                gradient=np.array([(a['score']-b['score'])/2 for a,b in pairs])
            write_json(stage/(anchor+'_local_credit.json'),dict(indices=indices.tolist(),steps=steps.tolist(),
                jacobian=jac.tolist(),loss_gradient=gradient.tolist(),pairs=[[a['tag'],b['tag']] for a,b in pairs]))
            current=origin
            if args.bounded_postpeak:
                capacities=[]
                for radius in (64.,128.,256.):
                    solution=solve_correction(jac,gradient,values(origin),score_limit-origin['score'],margin=0.,radius=radius)
                    capacities.append(dict(radius=radius,solution=solution))
                    if solution['success']:break
                write_json(stage/'bounded_capacity.json',dict(capacities=capacities))
                if solution['success']:
                    delta=np.array(solution['scaled_parameter_step']);offset=np.zeros(dimensions);offset[indices]=delta*steps
                    assert parameters_in_bounds(u+offset,-1.,1.)
                    measured_steps=[]
                    for fraction in (.25,.5,1.):
                        c=evaluate(u+fraction*offset,'bounded')
                        measured_steps.append(c)
                        blocks.append(dict(fraction=fraction,tag=c['tag'],score=c['score'],actual_guard_min=min(c['augmented_measured_inequalities'])))
                    current=min(measured_steps,key=lambda c:max(0.,-min(c['augmented_measured_inequalities'])))
            for iteration in range(3):
                if feasible(current):break
                for margin in (1e-6,0.):
                    for radius in ((4.,16.,64.,256.) if args.bounded_postpeak else (4.,16.,64.)):
                        solution=solve_correction(jac,gradient,values(current),score_limit-current['score'],margin=margin,radius=radius)
                        if solution['success']:break
                    if solution['success']:break
                row=dict(anchor=anchor,iteration=iteration,current=current['tag'],margin=margin,radius=radius,solution=solution)
                if not solution['success']:
                    blocks.append(row);break
                delta=np.array(solution['scaled_parameter_step']);offset=np.zeros(dimensions);offset[indices]=delta*steps
                measured=evaluate(np.array(current['coefficients'])+offset,anchor+'_correction')
                row.update(measured=measured['tag'],actual_guard_min=min(measured['augmented_measured_inequalities']))
                blocks.append(row);norm=float(delta@delta)
                if norm<=1e-18:break
                jac+=np.outer(values(measured)-values(current)-jac@delta,delta)/norm
                gradient+=(measured['score']-current['score']-gradient@delta)*delta/norm
                current=measured
            write_json(stage/'local_recovery_audit.json',dict(blocks=blocks,acceptance_unchanged=True))
        pool.close();pool.join();pool=None
        finish(stage,md,baseline,data,best(),zero,history,blocks,choice,parent_file,started)
    except TimeoutError:
        if pool is not None:pool.terminate();pool.join();pool=None
        finish(stage,md,baseline,data,best(),zero,history,blocks,choice,parent_file,started,budget_exhausted=True)
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_actual_state_joint_recovery',error=repr(exc),
            new_candidates=len(history)-md['reused_candidates'],adopted=False,goal_complete=False));raise


if __name__=='__main__':mp.freeze_support();main()
