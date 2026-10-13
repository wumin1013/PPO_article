"""Learn three-action feedback before the measured wrong-way lateral interval."""
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
from scipy.optimize import minimize
from consolidate_joint_teacher_policy import collect_data
from joint_teacher_state_policy import state_pair,features,motion_geometry_context
from audit_saved_lateral_motion import lateral_motion
from learn_shared_temporal_joint_policy import initialize,inequalities
from learn_complete_joint_teachers import teacher_score,parameters_in_bounds
from learn_direct_observable_joint_policy import worker_initialize,worker_evaluate,finish
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha
from multi_corner_raw_metrics import augment,guards as multi_guards,objective as multi_objective


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--full-route',action='store_true')
    parser.add_argument('--pulse',action='store_true');parser.add_argument('--recovery',action='store_true')
    parser.add_argument('--local-affine',action='store_true');parser.add_argument('--affine-coupled',action='store_true');args=parser.parse_args()
    if args.affine_coupled:args.local_affine=True
    assert sum((args.pulse,args.recovery,args.local_affine))<=1
    if args.pulse or args.recovery or args.local_affine:args.full_route=True
    dimensions=18 if args.local_affine else 6
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_direct_observable_joint_learning_20260920'
    parent_file=source/'research_candidate.pt'
    audit=read(source/'raw_result_audit.json')
    assert audit['model_sha256']==sha(parent_file) and audit['unchanged_other_paths_bitwise']
    old=read(source/'results.json');chosen=next(c for c in old['candidates'] if c['tag']==old['best'])
    data=read(source/'baseline.json');data['baseline']=copy.deepcopy(chosen)
    baseline=data['baseline']['records'];parent=torch.load(parent_file,map_location='cpu',weights_only=False)
    prepared=BASE/'canonical_gate_band_early_observable_joint_learning_20260920'
    wall=480 if args.full_route else 300
    constant_stage=BASE/'canonical_gate_band_early_fullroute_joint_learning_20260920'
    recovery_stage=BASE/'canonical_gate_band_early_joint_exploration_restoration_20260920'
    affine_stage=BASE/'canonical_gate_band_early_local_affine_joint_learning_20260920'
    cache_stage=affine_stage if args.affine_coupled else recovery_stage if args.local_affine else constant_stage
    if args.pulse or args.recovery:assert read(constant_stage/'receipt.json')['status'].startswith('completed')
    if args.local_affine:assert read(recovery_stage/'receipt.json')['status'].startswith('completed')
    if args.affine_coupled:assert read(affine_stage/'receipt.json')['status'].startswith('completed')
    stage=register('canonical_gate_band_early_affine_coupled_joint_learning_20260920' if args.affine_coupled
                   else 'canonical_gate_band_early_local_affine_joint_learning_20260920' if args.local_affine
                   else 'canonical_gate_band_early_joint_exploration_restoration_20260920' if args.recovery
                   else 'canonical_gate_band_early_learned_pulse_joint_learning_20260920' if args.pulse
                   else 'canonical_gate_band_early_fullroute_joint_learning_20260920' if args.full_route
                   else 'canonical_gate_band_early_observable_joint_learning_20260920',
        [__file__,BASE/'learn_direct_observable_joint_policy.py',BASE/'joint_teacher_state_policy.py',
         BASE/'consolidate_joint_teacher_policy.py',BASE/'learn_shared_temporal_joint_policy.py',
         BASE/'learn_complete_joint_teachers.py',BASE/'learned_action_pulse_policy.py',
         parent_file,source/'receipt.json',source/'raw_result_audit.json',source/'temporal_support_audit.json',
         source/'replayed_states.npz',source/'results.json',source/'saved_lateral_motion_audit.json',
         BASE/'canonical_gate_band_direct_observable_exit_learning_20260920/receipt.json']+
         ([BASE/'multi_corner_raw_metrics.py',BASE/'complete_corner_metrics.py',BASE/'geometry_spec.py',
           prepared/'receipt.json',prepared/'support_audit.json',prepared/'replayed_states.npz'] if args.full_route else [])+
         ([constant_stage/'receipt.json',constant_stage/'state_joint/results.json',
           constant_stage/'full_endpoint_coverage_audit.json'] if args.pulse or args.recovery else [])+
         ([recovery_stage/'receipt.json',recovery_stage/'state_joint/results.json',recovery_stage/'exploration_result.json',
           recovery_stage/'unaccepted_exploration_candidate.pt',recovery_stage/'motion_response_context_audit.json'] if args.local_affine else [])+
         ([affine_stage/'receipt.json',affine_stage/'state_joint/results.json',affine_stage/'local_context_identity.json',
           affine_stage/'state_joint/candidate.pt'] if args.affine_coupled else []),
        dict(wall_s=wall,workers=3,maximum_new_vectors=120 if args.full_route else 90,centers=2,outputs=3,
             coefficient_bounds=[-1.,1.],kernel_width_choices=[.15,.3,.6],
             maximum_affected_corner_paths=24 if args.full_route else 12,strict_current_parent_shape_guards=True,
             fixed_width=.15 if args.full_route else None,include_complete_multicorner_paths=args.full_route,
             temporal_mode='quadratic_decay_pulse' if args.pulse else '20ms_constant',
             dimensions=dimensions,local_affine_context=args.local_affine,
             maximum_affine_calls=48 if args.affine_coupled else 32 if args.local_affine else None,
             optimize_bias_and_state_slopes_together=args.affine_coupled,
             exploration_then_original_guard_restoration=args.recovery,
             maximum_exploration_calls=18 if args.recovery else None,
             maximum_restoration_calls=20 if args.recovery else None,
             saved_state_replay_reused_paths=88-len(audit['motion_changed_paths'])),
        ('Matched joint-parameter control after fixed-bias slope restoration: retain the '
         'EXACT18-parameter state/action function and every source physical response, '
         'but release the six fixed bias coefficients together with the12 state slopes. '
         'State slopes are zero at their own centers by definition, so fixing the '
         'bias prevents changing the initial center action; slope failure alone cannot '
         'test the full function capacity. Start from the measured training-safe point '
         'with smallest squared final-guard violation while retaining at least half '
         'the saved exploration loss improvement. Optimize all18 coefficients jointly '
         'for at most48 calls, with unchanged actual final constraints and no repeated '
         'parameter vector. This compares coupled learning to a frozen-bias subspace; '
         'do not attribute any outcome to a PPO bias or universal root cause. '
         if args.affine_coupled else '')+
        ('Saved physical responses reduce206 curvature but increase14 curvature; other '
         'routes near83mm/s differ in observed forward gate range and have conflicting '
         'responses. Test local state-to-action capacity, not a speed-only causal claim. '
         'Keep the verified parent,centers,width0.15,constant20ms actions,full24-route '
         'executor and exact final guards. Add learned affine coefficients on TWO '
         'continuous OBSERVED coordinates: speed and heading-frame forward gate distance, '
         'centered at each kernel center and scaled by supported training-state90th '
         'absolute deviations with numeric floors1mm/s and0.1mm. Apply tanh normalization; '
         'these are feature scales, not speed commands or angle rules. Three action '
         'heads each get independent learned context slopes. Reuse every previously '
         'completed constant-policy vector exactly with12 zero slope coefficients. '
         'Verify zero-extension feature/action identity. '+
         ('Jointly optimize all18 coefficients for at most48 calls as specified above, '
          if args.affine_coupled else
          'Start from the saved unaccepted exploration point and optimize ONLY its12 added slopes for at most32 calls, ')+
         'requiring every unchanged measured final constraint before a research model '
         'can be selected. The anchor is training-only and cannot be adopted by itself. '
         if args.local_affine else '')+
        ('Matched optimizer mechanism change: the saved complete constant-action outcomes '
         'include lower true shape losses, but every such point fails the simultaneous '
         'per-metric nonincrease screen. Keep the SAME six constant-action policy '
         'parameters,parent,states,width,raw full24-route feedback and final acceptance '
         'guards. Re-score and reuse all36 saved physical vectors under the corrected '
         'endpoint-inclusive objective/guards. Start from the lowest-loss recorded '
         'point satisfying unchanged full physical completion/corridor, inward first '
         'turn, incoming-side, full time and cruise requirements. Allow intermediate '
         'shape tradeoffs ONLY in a separate training exploration phase (18 calls). '
         'Then restore the ORIGINAL complete shape guards from that nonzero point '
         '(20 calls). No exploration model can replace a retained/research checkpoint: '
         'save it distinctly as unaccepted training evidence. Only an actually measured '
         'improvement passing ALL original guards and full-pool checks may produce a '
         'research candidate. No physical tolerance, time base, actuator limit, failed '
         'prefix handling, final shape requirement or deployment rule is relaxed. '
         if args.recovery else '')+
        ('Matched temporal representation comparison after the actual early constant-action '
         'responses. Keep parent, two observable-state centers,width0.15,three jointly '
         'learned coefficients per center,actual24-route feedback and full endpoint-inclusive '
         'guards. Change only how each learned residual coefficient is executed: it '
         'multiplies the existing generic quadratic decay (1-t/20ms)^2 within the decision '
         'interval. All coefficients are learned from full physical trajectories; no '
         'trajectory template,angle table,phase switch,or hand-chosen speed is deployed. '
         'Decisions stay20ms,execution1ms,all V/A/J/geometry/braking/rest constraints stay '
         'unchanged. This is NOT the rejected linear interpolation between old/new '
         'actions. Require a bitwise zero-pulse20ms identity check, save every actually '
         'executed1ms action, and reject use through a constant-action runtime. Re-score '
         'saved constant-arm outcomes with the same full endpoint-inclusive metric; do '
         'not rerun them or equate this comparison with PPO-estimator evidence. '
         if args.pulse else '')+
        ('The prepared early basis reaches24 training paths including multi-corner18 and31. '
         'Its isolated-corner scope check correctly stopped BEFORE any physical rollout. '
         'Reuse the prepared exact states, fix the smallest already-audited width0.15, '
         'and expand actual evaluation to ALL24 affected complete routes. For multi-corner '
         'routes add rawXY metrics for each original vertex between adjacent leg-midpoint '
         'ordered-progress boundaries, initialized at the actual preceding1ms sample. '
         'Guard both0.1mm and0.2mm curvature/energy, signed edge excursions, reverse '
         'turning, wrong-way displacement/duration, apex speed, whole-route time/cruise, '
         'physical limits and completion. These declared local windows are research '
         'diagnostics, not an ideal-tangency acceptance claim. '+
         ('Use the joint optimization scope and call budget specified above; no prior '
          if args.recovery or args.local_affine else
          'Compare two three-action blocks in order304 then206, at most60 new complete vectors each; no prior ')+
         'physical experiment is rerun. No selective exclusion of affected full routes. '
         if args.full_route else '')+
        'The compact late-state basis is exactly zero before640ms on206 and300ms on304, '
        'whereas significant wrong-way lateral motion has already finished by620ms and249ms. '
        'Its coefficients therefore cannot repair those past errors. Freeze the verified '
        '9e37da state-feedback candidate with the meaningful206 improvement; the subsequent '
        'exit-only block gains just0.0013 objective and does not change this limitation. '
        'Add two compact OBSERVABLE state centers selected one decision before the first '
        'saved significant lateral interval. Those times select training observations only, '
        'never deployed switches. Learn the declared coefficients by actual complete closed-loop '
        'COBYLA under STRICT current-parent shape guards (no reference envelope relaxation). '
        'Reuse exact saved state replay for87 bitwise unchanged routes; replay only206 with '
        'the new parent and assert every decoded action. Audit support over all88. '+
        ('Use the fixed0.15 radius with all24 affected full routes. ' if args.full_route else
         'Choose the widest registered radius touching at most12 isolated corners. ')+
        'Reuse routes '
        'with exactly zero active basis and all previous vectors, and execute every affected '
        'route through full ordered coverage, braking and rest. No PPO, fixed speed rule, '
        'teacher schedule, new sealed data, or inferred causal sufficiency claim.')
    folder=stage/'state_joint';folder.mkdir();write_json(stage/'baseline.json',data)
    started=time.perf_counter();deadline=started+wall-20;pool=None;history=[];blocks=[];exploration=None
    try:
        changed=audit['motion_changed_paths']
        if args.full_route:
            with np.load(prepared/'replayed_states.npz') as z:training={k:z[k].copy() for k in z.files}
        else:
            with np.load(source/'replayed_states.npz') as z:training={k:z[k].copy() for k in z.files}
            keep=~np.isin(training['path_ids'],changed)
            update=collect_data(parent,parent,[r for r in baseline if r['path'] in changed])
            training={k:np.concatenate([v[keep],update[k]],axis=0) for k,v in training.items()}
        assert not np.any(training['labels'])
        np.savez_compressed(stage/'replayed_states.npz',**training)
        a,b=state_pair(training['observations'],training['innovations'])
        both=np.vstack([a,b]);mean=both.mean(0);std=np.maximum(both.std(0),.05);a=(a-mean)/std
        centers=[];anchors=[];anchor_indices=[]
        for path in (206,304):
            r=next(r for r in baseline if r['path']==path);lateral=lateral_motion(r['trace'])
            first=lateral['intervals'][0];decision=max(0,first['start_ms']//20-1)
            index=np.flatnonzero((training['path_ids']==path)&(training['decisions']==decision)).item()
            anchor_indices.append(index)
            centers.append(a[index]);anchors.append(dict(path=path,decision=decision,
                first_lateral_interval=first,anchor_used_only_for_training=True))
        md=dict(architecture='joint_teacher_state_radial',parent=parent,mean=mean,std=std,
            centers=np.array(centers),coefficients=np.zeros((dimensions//3,3)),accepted=False,
            learning_method='early_observable_state_complete_return_joint_optimization',
            action_interval_ms=20,physical_tick_ms=1,reused_candidates=1)
        if args.pulse:md.update(temporal_mode='quadratic_decay_pulse',required_runtime='LearnedActionPulseRuntime',
            learning_method='early_observable_state_learned_pulse_complete_return_optimization',
            require_executed_1ms_actions_for_replay=True)
        choices=[];corner_paths={r['path'] for r in baseline if 'shape' in r}
        for width in (.15,.3,.6):
            md['radii']=np.full(2,width);f=features(md,training['observations'],training['innovations'])
            active=np.any(f!=0.,axis=(1,2));paths=sorted(np.unique(training['path_ids'][active]).tolist())
            choices.append(dict(width=width,paths=paths,decisions=int(active.sum()),all_corners=set(paths)<=corner_paths))
        eligible=[choices[0]] if args.full_route else [c for c in choices if len(c['paths'])<=12 and c['all_corners']]
        write_json(stage/'preparation_support_audit.json',dict(choices=choices,anchors=anchors,new_physical_vectors=0))
        assert eligible,'Early observed states need a broader full-route evaluator before optimization'
        choice=max(eligible,key=lambda c:c['width']);md['radii']=np.full(2,choice['width'])
        f=features(md,training['observations'],training['innovations'])
        if args.local_affine:
            context=motion_geometry_context(training['observations']);center_context=context[anchor_indices]
            scales=np.array([np.maximum(np.quantile(abs(context[np.any(f[:,i,:]!=0.,axis=1)]-center_context[i]),.9,axis=0),[1.,.1]) for i in range(2)])
            md.update(local_affine_context=True,center_context=center_context,context_scale=scales,
                learning_method='local_affine_motion_geometry_complete_return_joint_optimization')
            previous_md=torch.load(recovery_stage/'unaccepted_exploration_candidate.pt',map_location='cpu',weights_only=False)
            for name in ('centers','mean','std','radii'):assert np.array_equal(md[name],previous_md[name])
            old_features=features(previous_md,training['observations'],training['innovations'])
            assert np.array_equal(f,old_features)
            old_correction=(f*previous_md['coefficients']).sum(axis=-2)
            expanded=np.vstack([previous_md['coefficients'],np.zeros((4,3))])
            assert np.array_equal(old_correction,(f*expanded[:2]).sum(axis=-2))
            write_json(stage/'local_context_identity.json',dict(center_context=center_context.tolist(),
                context_scale=scales.tolist(),all_saved_state_features_bitwise_equal=True,
                zero_extended_action_corrections_bitwise_equal=True,training_states=len(context),
                new_physical_baseline_rollouts=0))
        write_json(stage/'support_audit.json',dict(anchors_for_training_only=anchors,width_choices=choices,
            selected=choice,reused_replay_paths=88-len(changed),newly_replayed_paths=changed,
            parent_decoded_action_identity=True,unaffected_paths=[r['path'] for r in baseline if r['path'] not in choice['paths']]))
        print('EARLY_STATE_SUPPORT',choice,anchors,flush=True)
        if args.pulse:
            from joint_teacher_state_policy import JointTeacherStatePolicy
            from learned_action_pulse_policy import environment_factory
            from probe_coordinated_suffix import reconstruct
            r=next(r for r in baseline if r['path']==206)
            with np.load(r['trace']) as z:ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
            frame=anchors[0]['decision']*20;costs=dict(actor_calls=0,physics_ticks=0,saved_prefix_geometry_ticks=0)
            env=reconstruct(ref,rows,actions,frame,costs,env_factory=environment_factory)
            policy=JointTeacherStatePolicy(md);policy.bind(env)
            _,_,_,info=env.step(policy.act(env.observation(),True)[0])
            assert np.array_equal(info['rows'],rows[frame:frame+20])
            assert not np.any(env.pending_pulse)
            write_json(stage/'zero_pulse_identity.json',dict(path=206,start_tick=frame,physical_ticks=20,
                raw_rows_bitwise_equal=True,original_actions_unchanged=True,full_baseline_rollouts_repeated=0))
        selected=[r for r in baseline if r['path'] in choice['paths']]
        for r in selected:augment(r)
        corner_selected=[r for r in selected if 'shape' in r]
        retained=[next(o['baseline_shape'] for o in data['retained'] if o['path']==r['path']) for r in corner_selected]
        initialize(stage,parent,corner_selected,retained,deadline,False,None,None,False,{'early_state':True},[])
        write_json(stage/'baseline.json',data)
        def score(records):
            regular=[r for r in records if r['path'] in {b['path'] for b in corner_selected}]
            total=sum(teacher_score(r,b,o,barrier=False) for r,b,o in zip(regular,corner_selected,retained))
            return float(total+sum(multi_objective(r,b) for r,b in zip(records,selected) if 'shape' not in b))
        def constraints(records):
            values=inequalities([r for r,b in zip(records,selected) if 'shape' in b]).tolist()
            for r,b in zip(records,selected):
                if 'shape' not in b:values.extend(multi_guards(r,b))
            return np.asarray(values)
        def training_guards(records):
            values=[]
            for r,b in zip(records,selected):
                size=6 if 'shape' in b else 4
                if not r['success']:values.extend([-100.]*size);continue
                values.extend([1.,r['corridor']['margin_lower_mm']/.01,
                    (b['complete_time_s']+.02-r['complete_time_s'])/.02,
                    (r['motion']['cruise_ge99_s']-b['motion']['cruise_ge99_s']+.02)/.05])
                if 'shape' in b:values.extend([1. if r['target_motion']['first_turn_inside'] else -1.,
                    (b['shape']['in_edge_outward_mm']-r['shape']['in_edge_outward_mm'])/.002])
            return np.asarray(values)
        zero=dict(tag='reused_identity',coefficients=[0.]*dimensions,records=selected,score=score(selected),
                  measured_inequalities=constraints(selected).tolist())
        assert min(zero['measured_inequalities'])>=0.
        if args.full_route:
            write_json(stage/'baseline_multicorner_audit.json',dict(records=[r for r in selected if 'shape' not in r],
                zero_constraint_minimum=min(zero['measured_inequalities']),original_constraints_unchanged=True))
        history.append(zero);cache={tuple(np.zeros(dimensions)):zero}
        if args.recovery or args.local_affine:
            for saved in read(cache_stage/'state_joint/results.json')['candidates'][1:]:
                c=copy.deepcopy(saved);c['tag']='reused_'+c['tag']
                if args.local_affine and len(c['coefficients'])==6:c['coefficients']+= [0.]*12
                assert len(c['coefficients'])==dimensions
                for r in c['records']:
                    r.pop('multi_corner_windows',None);r.pop('multi_corner_definition',None);augment(r)
                c['score']=score(c['records']);c['measured_inequalities']=constraints(c['records']).tolist()
                c['reused_full_physical_vector']=True
                history.append(c);cache[tuple(np.round(c['coefficients'],12))]=c
            md['reused_candidates']=len(history)
            write_json(stage/'reused_metric_rescoring.json',dict(candidates=[dict(tag=c['tag'],score=c['score'],
                original_guard_minimum=min(c['measured_inequalities']),training_guard_minimum=float(training_guards(c['records']).min())) for c in history],
                physical_reruns=0))
        pool=mp.get_context('spawn').Pool(3,worker_initialize,(md,selected,folder,deadline))
        def feasible(c):return parameters_in_bounds(c['coefficients'],-1.,1.) and min(c['measured_inequalities'])>=-1e-8
        def evaluate(u):
            key=tuple(np.round(u,12))
            if key not in cache:
                tag=f'early_{len(history)}';active=np.any(np.asarray(u).reshape(-1,2,3)!=0.,axis=(0,2))
                zero_paths=[r['path'] for r in selected if not np.any(f[training['path_ids']==r['path']][:,active,:])]
                measured=pool.map(worker_evaluate,[(r['path'],u,tag) for r in selected if r['path'] not in zero_paths])
                by_path={r['path']:r for r in measured}
                records=[by_path.get(r['path'],dict(copy.deepcopy(r),reused_exact_zero_support=True)) for r in selected]
                for r in records:augment(r)
                c=dict(tag=tag,coefficients=np.asarray(u).tolist(),records=records,score=score(records),
                    measured_inequalities=constraints(records).tolist(),exact_zero_support_paths=zero_paths,
                    actual_rerun_paths=[r['path'] for r in selected if r['path'] not in zero_paths])
                history.append(c);cache[key]=c
                best=min(filter(feasible,history),key=lambda c:c['score'])
                write_json(folder/'results.json',dict(candidates=history,best=best['tag']))
                if len(history)%10==0:print('EARLY_JOINT_EVALUATED',len(history)-1,'best',best['score'],'minimum',min(c['measured_inequalities']),flush=True)
            return cache[key]
        if args.recovery:
            good_training=lambda c:parameters_in_bounds(c['coefficients'],-1.,1.) and training_guards(c['records']).min()>=-1e-8
            exploration=min(filter(good_training,history),key=lambda c:c['score'])
            print('EXPLORATION_REUSED_START',exploration['tag'],exploration['score'],flush=True)
            solved=minimize(lambda u:evaluate(u)['score'],np.asarray(exploration['coefficients']),method='COBYLA',bounds=[(-1.,1.)]*6,
                constraints=[dict(type='ineq',fun=lambda u:training_guards(evaluate(u)['records']))],
                options=dict(maxiter=18,rhobeg=.08,tol=1e-6,catol=1e-8))
            exploration=min(filter(good_training,history),key=lambda c:c['score'])
            saved_md=copy.deepcopy(md);saved_md['coefficients']=np.asarray(exploration['coefficients']).reshape(-1,3)
            saved_md.update(accepted=False,final_shape_guards_pass=False,training_exploration_only=True)
            torch.save(saved_md,stage/'unaccepted_exploration_candidate.pt')
            write_json(stage/'exploration_result.json',dict(best=exploration,original_guard_pass=feasible(exploration),
                original_baseline_score=zero['score'],adopted=False))
            print('EXPLORATION_MEASURED_BEST',exploration['tag'],exploration['score'],'original_guard_min',min(exploration['measured_inequalities']),flush=True)
        if args.local_affine:
            exploration=read(recovery_stage/'exploration_result.json')['best']
            exploration=dict(exploration,coefficients=exploration['coefficients']+[0.]*12)
            assert tuple(np.round(exploration['coefficients'],12)) in cache
            if args.affine_coupled:
                limit=zero['score']-.5*(zero['score']-exploration['score'])
                eligible=[c for c in history if c['score']<=limit and parameters_in_bounds(c['coefficients'],-1.,1.)
                          and training_guards(c['records']).min()>=-1e-8]
                assert eligible
                exploration=min(eligible,key=lambda c:np.square(np.minimum(c['measured_inequalities'],0.)).sum())
                write_json(stage/'coupled_initial_point.json',dict(tag=exploration['tag'],score=exploration['score'],
                    squared_final_guard_violation=float(np.square(np.minimum(exploration['measured_inequalities'],0.)).sum()),
                    coefficients=exploration['coefficients'],all_previous_physical_vectors_reused=True))
                print('AFFINE_COUPLED_START',exploration['tag'],exploration['score'],min(exploration['measured_inequalities']),flush=True)
        groups=[np.arange(18)] if args.affine_coupled else [np.arange(6,18)] if args.local_affine else [np.arange(6)] if args.recovery else [np.arange(3,6),np.arange(3)] if args.full_route else [np.arange(6)]
        for indices in groups:
            anchor=np.asarray((exploration if args.recovery or args.local_affine else min(filter(feasible,history),key=lambda c:c['score']))['coefficients']).copy()
            def expand(v):
                full=anchor.copy();full[indices]=v;return full
            solved=minimize(lambda u:evaluate(expand(u))['score'],anchor[indices],method='COBYLA',bounds=[(-1.,1.)]*len(indices),
                constraints=[dict(type='ineq',fun=lambda u:np.asarray(evaluate(expand(u))['measured_inequalities']))],
                options=dict(maxiter=48 if args.affine_coupled else 32 if args.local_affine else 20 if args.recovery else 60 if args.full_route else 90,
                    rhobeg=.05 if args.local_affine else .015 if args.recovery else .03,tol=1e-6,catol=1e-8))
            best=min(filter(feasible,history),key=lambda c:c['score'])
            blocks.append(dict(indices=indices.tolist(),best=best['tag'],score=best['score'],solver=str(solved.message)))
            print('EARLY_FULLROUTE_BLOCK',blocks[-1],flush=True)
        best=min(filter(feasible,history),key=lambda c:c['score'])
        blocks.append(dict(block='early_state_joint',new_candidates=len(history)-1,best=best['tag'],score=best['score'],solver=str(solved.message)))
        pool.close();pool.join();pool=None
        finish(stage,md,baseline,data,best,zero,history,blocks,choice,parent_file,started)
    except TimeoutError:
        if pool is not None:pool.terminate();pool.join();pool=None
        if not history:raise
        best=min((c for c in history if parameters_in_bounds(c['coefficients'],-1.,1.) and min(c['measured_inequalities'])>=-1e-8),key=lambda c:c['score'])
        finish(stage,md,baseline,data,best,zero,history,blocks,choice,parent_file,started,budget_exhausted=True)
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join();pool=None
        receipt(stage,dict(status='failed_early_observable_joint_learning',error=str(exc),
            new_candidates=max(0,len(history)-1),wall_s=time.perf_counter()-started,adopted=False,goal_complete=False));raise


if __name__=='__main__':mp.freeze_support();main()
