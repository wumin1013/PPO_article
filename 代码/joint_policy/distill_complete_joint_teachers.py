"""Joint state-function fitting, followed by actual full64 feedback validation."""
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
from joint_teacher_state_policy import JointTeacherStatePolicy,features,state_pair,OUTPUT_SCALE
from decision_innovation_policy import innovation_features
from deep_joint_credit_policy import load_credit_policy
from canonical_gate_state import checkpoint_environment_factory
from learn_nonlinear_joint_feedback import rollout,promotion
from learn_full_pool_joint_credit import aggregate
from learn_retained_joint_growth import smooth_cruise
from learn_absolute_shape_joint_credit import absolute_actual
from learn_pathwise_full_credit import actual_selection
from learn_full_pool_compensation import progress
from study_io import BASE,register,receipt,write_json,sha
from search_complete_return import read


def transfer_teacher(stage,parent,baseline,selected,source,deadline):
    """Measure an old oracle with the new feedback parent before fitting it."""
    from scipy.interpolate import CubicSpline
    from probe_coordinated_suffix import reconstruct
    from analyze_nonlinear_joint_response import measure
    policy=load_credit_policy(parent)
    factory=checkpoint_environment_factory('gate_band_canonical_kernel')
    costs=dict(saved_prefix_geometry_ticks=0,actor_calls=0,physics_ticks=0)
    for item in selected:
        b=next(r for r in baseline['records'] if r['path']==item['path'])
        with np.load(b['trace']) as z:ref,nominal,old_actions=[z[k].copy() for k in ('reference','rows','actions')]
        start,stop=item['start_tick'],item['stop_tick']
        env=reconstruct(ref,nominal,old_actions,start,costs,env_factory=factory)
        if hasattr(policy,'bind'):policy.bind(env)
        count=(stop-start)//20+1
        basis=CubicSpline(np.linspace(0,1,6),np.vstack([np.zeros((1,4)),np.eye(4),np.zeros((1,4))]),
                          bc_type='clamped')(np.linspace(0,1,count))
        if item.get('temporal_scope') in ('full_window','target_jump_window','observed_approach'):
            basis=CubicSpline(np.linspace(0,1,5),np.vstack([np.eye(4),np.zeros((1,4))]),
                              bc_type='clamped')(np.linspace(0,1,count))
        u=np.asarray(item['candidate']['parameters']).reshape(4,3)
        chunks=[];held=[];obs=[];inputs=[];labels=[];base_actions=[]
        while not env.done:
            if time.perf_counter()>deadline:raise TimeoutError('Teacher transfer wall budget')
            x=env.observation();latent=policy.act(x,True)[0]
            _,innovation=innovation_features(env,latent)
            index=(env.ticks-start)//20
            offset=basis[index]@u*OUTPUT_SCALE if index<count else np.zeros(3)
            action=(latent+offset).astype(latent.dtype)
            _,_,_,info=env.step(action)
            chunks.append(info['rows']);held.append(info['held_action']);obs.append(x)
            inputs.append(innovation);labels.append(action-latent);base_actions.append(latent)
        rows=np.vstack([nominal[:start],*chunks]);actions=np.vstack([old_actions[:start//20],held])
        record=measure(ref,rows,env)
        assert record['success'],'Transferred teacher failed with the new feedback parent'
        s=record['shape'];bs=b['shape']
        for key in ('in_edge_outward_mm','out_edge_outward_mm','full_physical_reverse_deg',
                    'full_normalized_energy','full_spatial_peak_curvature_mm_inv'):
            assert s[key]<=bs[key]+1e-7,('Transferred teacher lost shape protection',key)
        assert s['apex_speed_mm_s']>=bs['apex_speed_mm_s']-1e-4
        assert record['target_motion']['first_turn_inside']
        assert record['complete_time_s']<=b['complete_time_s']+.02
        assert record['motion']['cruise_ge99_s']>=b['motion']['cruise_ge99_s']-.02
        file=stage/f"transferred_teacher_path{item['path']}.npz"
        np.savez_compressed(file,reference=ref,rows=rows,actions=actions,observations=np.asarray(obs),
            innovations=np.asarray(inputs),residual_labels=np.asarray(labels),parent_latents=np.asarray(base_actions),
            parameters=u.ravel(),start_tick=start,stop_tick=stop)
        item['original_teacher_trace']=str(source/item['candidate']['trace'])
        item['candidate']['trace']=str(file)
        item['transferred_record']=record
    write_json(stage/'teacher_transfer.json',dict(selected=selected,offline_teacher_only=True,
        deployable_policy=False,not_a_repeated_parent_experiment=True))


WORK={}


def select_teachers(source,baseline):
    selected=[]
    for item in read(source/'results.json')['teachers']:
        b=next(r for r in baseline['records'] if r['path']==item['path'])
        choices=[]
        for c in read(source/f"path{item['path']}"/'results.json')['candidates']:
            r=c['record'];s=r.get('shape',{})
            if c['tag']=='identity' or not r['success'] or not r['target_motion']['first_turn_inside']:continue
            keys=['in_edge_outward_mm','out_edge_outward_mm','full_physical_reverse_deg',
                  'full_normalized_energy','full_spatial_peak_curvature_mm_inv']
            if any(s[k]>b['shape'][k]+1e-7 for k in keys):continue
            if s['apex_speed_mm_s']<b['shape']['apex_speed_mm_s']-1e-4:continue
            if r['complete_time_s']>b['complete_time_s']+.02:continue
            if r['motion']['cruise_ge99_s']<b['motion']['cruise_ge99_s']-.02:continue
            if c['score']>=item['baseline_score']-.001:continue
            choices.append(c)
        if choices:
            best=min(choices,key=lambda c:c['score'])
            selected.append(dict(path=item['path'],candidate=best,start_tick=item['start_tick'],
                                 stop_tick=item['stop_tick'],temporal_scope=item.get('temporal_scope','apex')))
    return selected


def anchor_data(parent,baseline,selected):
    policy=load_credit_policy(parent)
    factory=checkpoint_environment_factory('gate_band_canonical_kernel')
    windows={c['path']:(c['start_tick'],c['stop_tick']) for c in selected}
    observations=[];innovations=[]
    for r in baseline['records']:
        with np.load(r['trace']) as z:ref,rows,actions,obs=[z[k].copy() for k in ('reference','rows','actions','observations')]
        env=factory(ref);tick=0
        if hasattr(policy,'bind'):policy.bind(env)
        for index,x in enumerate(obs):
            frame=index*20
            while tick<frame:
                env.position=rows[tick,:2].copy();env.velocity=rows[tick,2:4].copy();env.acceleration=rows[tick,4:6].copy()
                env.heading=math.atan2(rows[tick,9],rows[tick,8]);env.tracker.update(env.position);tick+=1
            if index:env.held=actions[index-1].copy()
            env.ticks=frame;env.decisions=index
            if r['path'] in windows and windows[r['path']][0]<=frame<windows[r['path']][1]:continue
            latent=policy.act(x,True)[0]
            assert np.array_equal(env.decode_action(latent),actions[index])
            _,inputs=innovation_features(env,latent)
            observations.append(x);innovations.append(inputs)
    return np.asarray(observations),np.asarray(innovations)


def initialize(stage,references,curved,retained,deadline):
    torch.set_num_threads(1)
    WORK.update(stage=stage,references=references,curved=curved,retained=retained,deadline=deadline,
                factory=checkpoint_environment_factory('gate_band_canonical_kernel'))


def evaluate(job):
    tag,model=job
    policy=JointTeacherStatePolicy(model)
    def factory(ref):return policy.bind(WORK['factory'](ref))
    folder=WORK['stage']/tag;folder.mkdir()
    records=[];costs=dict(actor_calls=0,physics_ticks=0)
    for path,ref in WORK['references'].items():
        r,rows,actions,observations=rollout(policy,factory,ref,costs,WORK['deadline'])
        file=folder/f'path{path}.npz'
        np.savez_compressed(file,reference=ref,rows=rows,actions=actions,observations=observations)
        r.update(path=path,trace=file.relative_to(WORK['stage']).as_posix(),curved_path=WORK['curved'][path])
        if r['success']:r['smooth_cruise99_s']=smooth_cruise(rows)
        records.append(r)
    summary=aggregate(records);checks=promotion(records,WORK['retained'])
    c=dict(tag=tag,records=records,summary=summary,costs=costs,promotion=checks,
        primary_preserved=all(checks[k] for k in ('all_complete','inward_first_turns','entry_nonworse')),
        absolute_objective=None)
    if summary['successes']==len(records):c['absolute_objective']=absolute_actual(c,WORK['retained'])
    if len(records)>64:
        c['original64_summary']=aggregate([r for r in records if r['path']<64])
        c['added_training_summary']=aggregate([r for r in records if r['path']>=64])
    write_json(WORK['stage']/(tag+'.json'),c)
    return c


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--continue-from')
    parser.add_argument('--constrained-teachers',action='store_true')
    parser.add_argument('--teacher-source')
    parser.add_argument('--output-stage')
    parser.add_argument('--augmented-training',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1)
    source=BASE/('canonical_gate_band_constrained_joint_teacher_learning_20260919' if args.constrained_teachers else
                 'canonical_gate_band_bounded_joint_teacher_learning_20260919' if args.continue_from
                 else 'canonical_gate_band_complete_joint_teacher_learning_20260919')
    if args.teacher_source:source=BASE/args.teacher_source
    data=read(BASE/'canonical_gate_band_current_bend_capacity_learning_20260919/baseline.json')
    baseline,retained=data['baseline'],data['retained']
    parent_file=BASE/'canonical_gate_band_iterative_motion_constraint_learning_20260919/research_candidate.pt'
    extra_sources=[]
    if args.continue_from:
        previous=BASE/args.continue_from
        previous_result=read(previous/'results.json')
        baseline=copy.deepcopy(next(c for c in previous_result['candidates'] if c['tag']==previous_result['best']))
        for r in baseline['records']:r['trace']=str(previous/r['trace'])
        parent_file=previous/'research_candidate.pt'
        extra_sources=[previous/'results.json',previous/'receipt.json']
    if args.augmented_training:
        augmented=read(source/'baseline.json')
        baseline,retained=augmented['baseline'],augmented['retained']
        extra_sources.append(source/'baseline.json')
    parent=torch.load(parent_file,map_location='cpu',weights_only=False)
    transfer_needed=bool(args.continue_from and read(source/'receipt.json').get('source_model_sha256')!=sha(parent_file))
    selected=select_teachers(source,baseline)
    assert selected,'No nondegrading joint teacher: do not regress unhelpful labels'
    stage_name=args.output_stage or ('canonical_gate_band_constrained_teacher_state_distillation_20260919' if args.constrained_teachers else
                   'canonical_gate_band_continued_joint_teacher_distillation_20260919' if args.continue_from
                   else 'canonical_gate_band_joint_teacher_state_distillation_20260919')
    stage=register(stage_name,
        [__file__,BASE/'joint_teacher_state_policy.py',BASE/'decision_innovation_policy.py',
         BASE/'learn_complete_joint_teachers.py',BASE/'learn_pathwise_full_credit.py',parent_file,
         source/'results.json',source/'receipt.json',
         BASE/'canonical_gate_band_current_bend_capacity_learning_20260919/baseline.json']+
        [source/c['candidate']['trace'] for c in selected]+extra_sources,
        dict(wall_s=420,workers=2,teacher_paths=[c['path'] for c in selected],
             output_heads=['steering','feed','preview'],ridge_controls=[1e-10,1e-5],full_pool_candidates=2,
             training_paths=len(baseline['records']),augmented_training=args.augmented_training,
             transferred_teacher_checks=len(selected) if transfer_needed else 0),
        ('Preserve the prior learned state-policy as a frozen parent. Revalidate each selected '
         'old offline teacher only if the parent hash differs; otherwise reuse its actual '
         'completed trajectory. On a NEW parent its feedback may differ '
         'on perturbed states even if nominal paths match. The new compact residual must '
         'be exactly zero on all protected current-parent observations, including the '
         'previously improved+140 path. ' if args.continue_from else '')+
        'Select actual complete teachers that improve score without degrading ANY current '
        'corner entry/exit/reversal/energy/peak/apex metric; preserve time/cruise too. '
        'Jointly fit their three action residuals using a shared compact observable-state '
        'radial policy. Inputs are local geometry/physical state, actually held prior '
        'actions and observed target innovation/lag, never path,angle ID,decision time '
        'or teacher schedule. Compact support radii derive from distance to saved '
        'protected state observations and zero-label teacher states, giving exact zero '
        'correction there. This is offline joint-action fitting, not PPO or evidence '
        'of closed-loop success. Fit two declared ridge controls, verify protected '
        'states exactly and reflection symmetry, then execute each on all registered training routes. '
        'Any added training paths also require explicit per-path nondegradation against their parent. '
        'Only unchanged actual_selection can save a research checkpoint. Oracle curves '
        'and temporal bases are not imported by the deployed policy. Formal86cd and '
        'current5ab remain preserved; independent/fair/generalization checks are not done.')
    started=time.perf_counter();pool=None;outcomes=[]
    progress(stage,True,'将有效教师的三动作联合拟合为无时间/路径输入的状态函数，并实际验完整64路径。')
    try:
        write_json(stage/'baseline.json',dict(baseline=baseline,retained=retained))
        if transfer_needed:transfer_teacher(stage,parent,baseline,selected,source,started+60)
        write_json(stage/'teacher_selection.json',selected)
        obs=[];inputs=[];labels=[]
        for c in selected:
            with np.load(source/c['candidate']['trace']) as z:
                obs.append(z['observations'].copy());inputs.append(z['innovations'].copy());labels.append(z['residual_labels'].copy())
        obs,inputs,labels=np.vstack(obs),np.vstack(inputs),np.vstack(labels)
        protected_obs,protected_inputs=anchor_data(parent,baseline,selected)
        active=np.max(abs(labels),axis=1)>1e-9
        protected_obs=np.vstack([protected_obs,obs[~active]])
        protected_inputs=np.vstack([protected_inputs,inputs[~active]])
        a,b=state_pair(obs,inputs);pa,pb=state_pair(protected_obs,protected_inputs)
        raw=np.vstack([a,b,pa,pb]);mean=raw.mean(0);std=np.maximum(raw.std(0),.05)
        centers=(a[active]-mean)/std
        protected=np.vstack([pa,pb]);protected=(protected-mean)/std
        nearest=np.full(len(centers),np.inf)
        for start in range(0,len(protected),128):
            d=np.sqrt(np.mean((centers[:,None,:]-protected[None,start:start+128])**2,axis=-1))
            nearest=np.minimum(nearest,d.min(axis=1))
        assert nearest.min()>1e-7,'Nonzero teacher label conflicts with protected observed state'
        model=dict(architecture='joint_teacher_state_radial',parent=parent,mean=mean,std=std,
            centers=centers,radii=.95*nearest,output_scale=OUTPUT_SCALE.copy(),
            learning_method='offline_three_action_teacher_state_distillation',accepted=False)
        f=features(model,obs,inputs);anchor=features(model,protected_obs,protected_inputs)
        assert np.max(abs(anchor))==0.
        identity=features(model,obs,inputs)
        mirrored_obs=torch.as_tensor(obs)
        from reflection_policy import reflect_observation
        mirrored=features(model,reflect_observation(mirrored_obs).numpy(),-inputs)
        assert np.max(abs(mirrored-identity*np.array([-1.,1.,1.])))<1e-14
        np.savez_compressed(stage/'joint_training_data.npz',observations=obs,innovations=inputs,
            residual_labels=labels,protected_observations=protected_obs,protected_innovations=protected_inputs)
        audits=[];jobs=[]
        for ridge in (1e-10,1e-5):
            coefficients=[]
            for head in range(3):
                matrix=np.vstack([f[:,:,head],np.sqrt(ridge)*np.eye(len(centers))])
                target=np.r_[labels[:,head]/OUTPUT_SCALE[head],np.zeros(len(centers))]
                coefficients.append(np.linalg.lstsq(matrix,target,rcond=1e-12)[0])
            md=copy.deepcopy(model);md['coefficients']=np.column_stack(coefficients);md['ridge']=ridge
            error=(f*md['coefficients']).sum(1)*OUTPUT_SCALE-labels
            tag=f'joint_ridge_{ridge}'
            torch.save(md,stage/(tag+'.pt'));jobs.append((tag,md))
            audits.append(dict(tag=tag,teacher_states=len(obs),active_centers=len(centers),
                protected_states=len(protected_obs),protected_feature_max=float(abs(anchor).max()),
                maximum_label_error_by_head=np.max(abs(error),axis=0).tolist(),
                residual_label_rms_by_head=np.sqrt(np.mean(labels**2,axis=0)).tolist(),
                minimum_radius=float(model['radii'].min()),maximum_radius=float(model['radii'].max())))
        write_json(stage/'fitting_audit.json',audits)
        print('JOINT_DISTILLATION_FIT',audits,flush=True)
        refs,curved={},{}
        for r in baseline['records']:
            with np.load(r['trace']) as z:refs[r['path']]=z['reference'].copy()
            curved[r['path']]=r['curved_path']
        pool=mp.get_context('spawn').Pool(2,initialize,(stage,refs,curved,retained,started+405))
        for c in pool.imap_unordered(evaluate,jobs):
            outcomes.append(c);print('JOINT_DISTILLATION_ACTUAL',c['tag'],c['summary'],flush=True)
            write_json(stage/'results.json',dict(candidates=outcomes,adopted=False,goal_complete=False))
        pool.close();pool.join();pool=None
        eligible=outcomes
        if args.augmented_training:
            old={r['path']:r for r in baseline['records']}
            for c in outcomes:
                guards=[]
                for r in c['records']:
                    if r['path']<64:continue
                    b=old[r['path']]
                    keys=('in_edge_outward_mm','out_edge_outward_mm','full_physical_reverse_deg',
                          'full_normalized_energy','full_spatial_peak_curvature_mm_inv')
                    good=(r['success'] and r['target_motion']['first_turn_inside']
                        and all(r['shape'][k]<=b['shape'][k]+1e-7 for k in keys)
                        and r['shape']['apex_speed_mm_s']>=b['shape']['apex_speed_mm_s']-1e-4
                        and r['complete_time_s']<=b['complete_time_s']+.02
                        and r['motion']['cruise_ge99_s']>=b['motion']['cruise_ge99_s']-.02)
                    guards.append(dict(path=r['path'],pass_all=bool(good)))
                c['added_training_guards']=guards
            eligible=[c for c in outcomes if all(r['pass_all'] for r in c['added_training_guards'])]
        chosen=actual_selection(eligible,baseline,retained)
        result=dict(candidates=outcomes,adopted=False,goal_complete=False)
        if chosen:
            result['best']=chosen['tag'];md=next(m for tag,m in jobs if tag==chosen['tag'])
            torch.save(md,stage/'research_candidate.pt')
        write_json(stage/'results.json',result)
        receipt(stage,dict(status='completed_joint_teacher_state_distillation',teacher_paths=[c['path'] for c in selected],
            full64_candidates=len(outcomes),best=chosen['tag'] if chosen else None,
            training_paths=len(baseline['records']),augmented_training=args.augmented_training,
            research_model_sha256=sha(stage/'research_candidate.pt') if chosen else None,
            wall_s=time.perf_counter()-started,adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_joint_teacher_state_distillation',error=str(exc),full64_candidates=len(outcomes),
            wall_s=time.perf_counter()-started,adopted=False,goal_complete=False));raise
    finally:
        progress(stage,False,'据状态策略真实闭环结果继续修复困难角，不将离线教师或拟合误差作为策略验收。')


if __name__=='__main__':mp.freeze_support();main()
