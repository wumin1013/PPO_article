"""PPO on two isolated physical resolutions, with one memoryless14input actor."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import json
import multiprocessing as mp
import random
import shutil
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime,timezone
from evaluate_historical_dense import ROOT,BASE,CASE,write,sha,verify_bound_inputs

MAIN=ROOT/'direct14_farvector_local_feedback_fit_round21'
TWO=ROOT/'direct14_two_action_initialization_round27/d_3/fit_12000.pt'
HALF_METRICS=[ROOT.parent/'all34_dominance_research'/name/'half_ms/corner_metrics.json'
              for name in ['joint_transit_retained_seed922222_full','fixed_actual_failure_recovery_full']]
HALF_METRICS.append(MAIN/'S2_half_ms_metrics_from_bound_trace.json')


def policy_from(saved,seed):
    if saved.get('policy_class') in ['LocalizedJointPolicy','LocalizedPreviewPolicy']:
        from research_policy_loader import from_saved
        return from_saved(saved,seed)
    from direct_two_action_control import DirectTwoPolicy
    from compact_fine_policy import FineCompactPolicy
    if saved.get('policy_class')=='GatedHeadPolicy':
        from gated_head_policy import GatedHeadPolicy
        p=GatedHeadPolicy(seed,preview_only=saved.get('preview_only_correction',False))
    else:
        p=DirectTwoPolicy(seed) if saved['interface'][1]==2 else FineCompactPolicy(seed=seed,feed_gain=saved.get('feed_gain',1.))
    p.load_state_dict(saved['policy'])
    return p


def physics_worker(connection,dt,args,stage,deadline):
    try:
        import numpy as np
        import torch
        import joint_runtime
        from run_refined_turn_candidate import configure_resolution,corner_stats
        from run_natural_resolution import physical_audit
        from natural_case_audit import analytic_command_audit
        from compact_farvector_runtime import FarVectorRuntime
        from direct_two_action_control import DirectTwoRuntime
        from saved_local_contexts import from_complete_trace
        from corner_deficit_objective import CornerDeficitCost
        from training_rejection_handling import sampled_step
        torch.set_num_threads(1)
        if dt!=.001:configure_resolution(dt)
        assert joint_runtime.DT==dt and joint_runtime.MACRO_TICKS*dt==.02
        key='1ms' if dt==.001 else 'half_ms'
        np.random.seed(args.seed+(dt==.0005));random.seed(args.seed+(dt==.0005));torch.manual_seed(args.seed+(dt==.0005))
        initial=torch.load(args.initial,map_location='cpu',weights_only=False)
        policy=policy_from(initial,args.seed);teacher=policy_from(initial,args.seed).eval().requires_grad_(False)
        action_dim=initial['interface'][1]
        def new_env(reference):
            return DirectTwoRuntime(reference,preview_rule_mm=initial['preview_rule_mm'],potential_speed=100.) if action_dim==2 else FarVectorRuntime(reference,potential_speed=100.)
        folders={'original':BASE/'paper_evidence_review/natural_closed_case','dense':CASE}
        refs={k:np.loadtxt(p/'reference.csv',delimiter=',',skiprows=1) for k,p in folders.items()}
        geometries={k:json.loads((p/'geometry.json').read_text(encoding='utf-8')) for k,p in folders.items()}
        full=MAIN/'eval_8000_original.npz' if dt==.001 else ROOT/'direct14_farvector_local_feedback_fit_round21_original_half_ms/trace.npz'
        gates=MAIN/'eval_8000_original_gates.json' if dt==.001 else full.parent/'gate_events.json'
        reference_checkpoint=MAIN/'fit_08000.pt'
        metric_path=MAIN/'eval_8000_original_metrics.json' if dt==.001 else full.parent/'corner_metrics.json'
        if getattr(args,'context_source',None):
            full=args.context_source/f'eval_{args.context_index:04d}_original_{key}.npz'
            gates=full.with_name(full.stem+'_gates.json');metric_path=full.with_name(full.stem+'_metrics.json')
            reference_checkpoint=args.context_source/f'checkpoint_{args.context_index:04d}.pt'
            bound=json.loads(full.with_suffix('.json').read_text(encoding='utf-8'))
            assert bound['success'] and bound['trace_sha256']==sha(full) and bound['checkpoint_sha256']==sha(reference_checkpoint)
        ref,expected,contexts,audit=from_complete_trace(full,gates,geometries['original'],dt=dt)
        reference_policy=policy_from(torch.load(reference_checkpoint,map_location='cpu',weights_only=False),args.seed)
        errors=[]
        replay_decisions=8 if getattr(args,'context_source',None) else 4
        for context in contexts:
            env=FarVectorRuntime(ref);env.restore(context['state']);chunks=[]
            for _ in range(replay_decisions):
                _,_,done,info=env.step(reference_policy.act(env.observation(),True)[0]);chunks.append(info['rows']);assert not done
            rows=np.vstack(chunks)
            error=float(abs(rows[:,:4]-expected[context['tick']:context['tick']+len(rows),:4]).max())
            assert error<2e-5,(key,context['id'],error)
            errors.append(error)
        budget_paths=None if dt==.001 else list(HALF_METRICS)
        if getattr(args,'extra_comparators',None):
            extra=json.loads(args.extra_comparators.read_text(encoding='utf-8'))
            assert all(sha(path)==digest for path,digest in extra['source_sha256'].items())
            if budget_paths is None:
                protocol=json.loads((ROOT/'joint_lowdim_acceptance_20260922.json').read_text(encoding='utf-8'))
                budget_paths=[Path(v['path']) for v in protocol['original']['comparators'].values()]
            budget_paths+=list(map(Path,extra[key]))
        cost=CornerDeficitCost(ref,geometries['original'],dt=dt,metric_paths=budget_paths)
        # The incremental cost must integrate the actual physical dt, not1ms.
        metric=json.loads(metric_path.read_text(encoding='utf-8'))
        cost_errors=[]
        for c in metric['corners']:
            cost.reset(c['id']);start=max(0,c['first_index']-40);stop=c['last_index']+41
            for i in range(start,stop,round(.02/dt)):cost.consume(expected[i:min(i+round(.02/dt),stop)])
            cost_errors.append(abs(cost.accumulated-c['speed_deficit_mm']))
        assert max(cost_errors)<1e-9
        audit.update(maximum_replay_xyv_error=max(errors),replayed_decisions_per_corner=replay_decisions,
                     maximum_incremental_deficit_error_mm=max(cost_errors),budget_source_sha256=cost.source_sha256,
                     reset_scope='Training-only physical states from complete incumbent trajectories;not policy memory or final evaluation resets.')
        write(stage/f'context_audit_{key}.json',audit)
        envs={k:new_env(r) for k,r in refs.items()};cursor=0;episodes=[];rejection_count=0
        schedule=list(range(len(contexts)))
        critical_indices=[5,29]
        if getattr(args,'critical_corners',None):
            mapping={c['id']:i for i,c in enumerate(contexts)}
            critical_indices=[mapping[key] for key in args.critical_corners]
        for _ in range(getattr(args,'critical_repeat',1)-1):schedule.extend(critical_indices)
        phase_values=getattr(args,'reset_phases_ms',[0]);current_phase=0
        audit.update(critical_corner_ids=[contexts[i]['id'] for i in critical_indices],
            reset_phases_ms=phase_values,reference_checkpoint_sha256=sha(reference_checkpoint))
        write(stage/f'context_audit_{key}.json',audit)
        dense_decisions=getattr(args,'dense_decisions',256)
        imported={str(Path(m.__file__).resolve()):sha(m.__file__) for m in list(sys.modules.values())
                  if getattr(m,'__file__',None) and Path(m.__file__).suffix=='.py' and Path(m.__file__).is_file()
                  and any(Path(m.__file__).resolve().is_relative_to(p) for p in [ROOT.parent,BASE])}
        write(stage/f'physical_sources_{key}.json',imported)
        connection.send(dict(ready=True,dt=dt,audit=audit))
        while time.perf_counter()<deadline:
            if not connection.poll(min(1.,max(0.,deadline-time.perf_counter()))):continue
            command=connection.recv()
            if command['kind']=='stop':break
            saved=torch.load(command['checkpoint'],map_location='cpu',weights_only=False)
            if set(policy.state_dict())!=set(saved['policy']):policy=policy_from(saved,args.seed)
            else:policy.load_state_dict(saved['policy'])
            if command['kind']=='collect':
                fields=['observation','next_observation','latent','log_probability','value','reward','next_value',
                        'terminal','segment_end','teacher_target','physical_reward','corner_penalty_s']
                batch={k:[] for k in fields};case_ids=[];corner_ids=[];phase_ids=[];start=time.perf_counter()
                for case,size in [('original',1024-dense_decisions),('dense',dense_decisions)]:
                    for j in range(size):
                        if time.perf_counter()>deadline:raise TimeoutError('Registered mixed-resolution deadline')
                        if case=='original' and (j%64==0 or envs[case].done):
                            context=contexts[schedule[cursor%len(schedule)]];cursor+=1
                            envs[case].restore(context['state'])
                            current_phase=phase_values[((cursor-1)//len(schedule))%len(phase_values)]
                            if current_phase:
                                env=envs[case];held=env.held.copy()/np.array([np.pi/2,50.,5.6])-np.array([0.,1.,6.4/5.6])
                                old_ticks=joint_runtime.MACRO_TICKS
                                try:
                                    joint_runtime.MACRO_TICKS=round(current_phase*.001/dt)
                                    _,_,phase_done,phase_info=env.step(held[:action_dim]);assert not phase_done
                                    assert env.progress()[0]<cost.corners[context['id']]['reference_s_mm']-10.
                                finally:joint_runtime.MACRO_TICKS=old_ticks
                            cost.reset(context['id'])
                        env=envs[case];x=env.observation();z,logp,value=policy.act(x)
                        def record_rejection(payload):
                            nonlocal rejection_count
                            rejection_count+=1
                            payload.update(case=case,checkpoint=command['checkpoint'],checkpoint_sha256=sha(command['checkpoint']),
                                           observation=x.copy(),sampled_log_probability=logp)
                            torch.save(payload,stage/f'rejection_{key}_{rejection_count:04d}.pt')
                        before=env.potential();x2,_,done,info=sampled_step(env,z,dt,record_rejection)
                        after=0. if env.success else env.potential()
                        physical=-info['elapsed_s']+after-before-(5. if done and not env.success else 0.)
                        penalty=cost.consume(info['rows'])[0] if case=='original' and len(info['rows']) else 0.
                        with torch.no_grad():
                            next_value=0. if done else float(policy.value(torch.as_tensor(x2)))
                            target=teacher.actor(torch.as_tensor(x)).numpy()
                        values=[x,x2,z,logp,value,physical-penalty,next_value,done,
                                j==size-1 or(case=='original' and j%64==63),target,physical,penalty]
                        for k,v in zip(fields,values):batch[k].append(v)
                        case_ids.append(case);corner_ids.append(cost.current if case=='original' else '')
                        phase_ids.append(current_phase if case=='original' else 0)
                        if done:
                            episodes.append(dict(update=saved['updates']+1,case=case,dt_s=dt,success=env.success,
                                                 reason=env.reason,prefix_s=env.ticks*dt))
                            if case=='dense':envs[case]=new_env(refs[case])
                path=stage/f'batch_{saved["updates"]+1:04d}_{key}.npz'
                np.savez_compressed(path,**{k:np.asarray(v) for k,v in batch.items()},case=np.array(case_ids),
                                    corner_id=np.array(corner_ids),reset_phase_ms=np.array(phase_ids),dt_s=dt)
                torch.save(dict(environments={k:e.state() for k,e in envs.items()},cursor=cursor,torch_rng=torch.get_rng_state(),
                                numpy_rng=np.random.get_state(),python_rng=random.getstate(),dt_s=dt),stage/f'physical_state_{saved["updates"]+1:04d}_{key}.pt')
                write(stage/f'episodes_{key}.json',episodes)
                connection.send(dict(batch=str(path),wall_s=time.perf_counter()-start))
            elif command['kind']=='local_check':
                from natural_case_metrics import ordered_projection
                records=[]
                for context,corner in zip(contexts,geometries['original']['sharp_corners']):
                    if time.perf_counter()>deadline-10:raise TimeoutError('Registered local-screen deadline')
                    env=new_env(refs['original']);env.restore(context['state']);start=env.env.state();chunks=[];error=None
                    try:
                        for _ in range(64):
                            _,_,done,info=env.step(policy.act(env.observation(),True)[0]);chunks.append(info['rows'])
                            if done:break
                    except Exception as exc:error=f'{type(exc).__name__}: {exc}'
                    rows=np.vstack(chunks) if chunks else np.empty((0,13))
                    valid=False;deficit=None;peaks=[];complete=False;analytic={}
                    if len(rows):
                        progress=ordered_projection(refs['original'],rows[:,:2]);lo=corner['reference_s_mm']-10.;hi=corner['reference_s_mm']+10.
                        complete=bool(progress.min()<lo and progress.max()>hi);mask=(progress>=lo)&(progress<=hi)
                        a=np.diff(np.vstack([start['velocity'],rows[:,2:4]]),axis=0)/dt
                        j=np.diff(np.vstack([start['acceleration'],a]),axis=0)/dt
                        peaks=[float(np.linalg.norm(rows[:,2:4],axis=1).max()),float(np.linalg.norm(a,axis=1).max()),float(np.linalg.norm(j,axis=1).max())]
                        analytic=analytic_command_audit(start['position'],rows,geometries['original'])
                        valid=bool(not error and not env.done and complete and analytic['analytic_corridor_pass'] and np.all(np.array(peaks)<=[100.000001,2000.001,62500.02]))
                        deficit=float(np.maximum(100-np.linalg.norm(rows[mask,2:4],axis=1),0.).sum()*dt)
                    trace=stage/f'local_{saved["updates"]:04d}_{key}_{context["id"]}.npz'
                    np.savez_compressed(trace,rows=rows,start_position=start['position'],initial_velocity=start['velocity'],initial_acceleration=start['acceleration'],dt_s=dt)
                    records.append(dict(id=context['id'],valid=valid,window_complete=complete,deficit_mm=deficit,budget_mm=cost.budgets[context['id']],
                        budget_pass=bool(valid and deficit<=cost.budgets[context['id']]),peaks=peaks,error=error,reason=env.reason,
                        trace_sha256=sha(trace),analytic=analytic))
                report=dict(dt_s=dt,records=records,valid_count=sum(r['valid'] for r in records),budget_count=sum(r['budget_pass'] for r in records),
                    scope='Local screen from incumbent arrival states;not whole-path success.',checkpoint_sha256=sha(command['checkpoint']))
                write(stage/f'local_check_{saved["updates"]:04d}_{key}.json',report);connection.send(report)
            elif command['kind']=='evaluate':
                case=command['case'];env=new_env(refs[case]);chunks=[];observations=[];actions=[];latents=[];times=[]
                error=None
                try:
                    while not env.done:
                        if time.perf_counter()>deadline-5:raise TimeoutError('Registered complete-check deadline')
                        x=env.observation();z=policy.act(x,True)[0]
                        observations.append(x);latents.append(z);times.append(env.ticks*dt)
                        _,_,_,info=env.step(z);chunks.append(info['rows']);actions.append(info['held_action'])
                except Exception as exc:error=f'{type(exc).__name__}: {exc}'
                rows=np.vstack(chunks) if chunks else np.empty((0,13))
                trace=stage/f'eval_{saved["updates"]:04d}_{case}_{key}.npz'
                np.savez_compressed(trace,reference=refs[case],rows=rows,observations=np.array(observations),
                                    actions=np.array(actions),latents=np.array(latents),decision_times_s=np.array(times),dt_s=dt)
                physical=physical_audit(refs[case],rows,dt) if len(rows) else {}
                analytic=analytic_command_audit(refs[case][0],rows,geometries[case]) if len(rows) else {}
                ordered=env.tracker.snapshot(1.25)
                success=bool(not error and env.success and ordered['geometry_complete'] and physical.get('kinematic_pass')
                             and physical.get('rest_pass') and analytic.get('analytic_corridor_pass'))
                result=dict(success=success,complete_time_s=len(rows)*dt if success else None,elapsed_prefix_s=len(rows)*dt,
                            reason=env.reason,error=error,audit=physical,analytic=analytic,geometry=ordered,dt_s=dt,
                            checkpoint=command['checkpoint'],checkpoint_sha256=sha(command['checkpoint']),trace_sha256=sha(trace),
                            observation_dimension=14,actor_outputs=action_dim,policy_memory_scalars=0,
                            observation_features=env.feature_names(),decision_dt_s=.02,goal_complete=False,method_frozen=False)
                write(trace.with_suffix('.json'),result);write(trace.with_name(trace.stem+'_gates.json'),env.tracker.events)
                if success and case=='original':write(trace.with_name(trace.stem+'_metrics.json'),corner_stats(refs[case],rows,geometries[case],dt))
                connection.send(result)
    except BaseException:
        error=traceback.format_exc()
        write(stage/f'worker_error_{dt}.json',dict(error=error))
        connection.send(dict(error=error))
    finally:
        connection.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--name',required=True);parser.add_argument('--seed',type=int,default=922302)
    parser.add_argument('--initial',type=Path,default=TWO);parser.add_argument('--updates',type=int,default=12)
    parser.add_argument('--heads-only',action='store_true')
    parser.add_argument('--feed-std',type=float,default=.008)
    parser.add_argument('--preview-std',type=float,default=.006)
    parser.add_argument('--steer-std',type=float,default=.0007)
    parser.add_argument('--correction-only',action='store_true')
    parser.add_argument('--local-screen',action='store_true')
    parser.add_argument('--check-every',type=int,default=4)
    parser.add_argument('--dense-decisions',type=int,default=256)
    parser.add_argument('--critical-repeat',type=int,default=1)
    parser.add_argument('--critical-corners',nargs='+')
    parser.add_argument('--context-source',type=Path);parser.add_argument('--context-index',type=int,default=1)
    parser.add_argument('--extra-comparators',type=Path)
    parser.add_argument('--reset-phases-ms',type=int,nargs='+',default=[0])
    parser.add_argument('--all-actor',action='store_true')
    args=parser.parse_args();assert Path(args.name).name==args.name
    assert args.dense_decisions in [256,512,768] and 1<=args.critical_repeat<=16
    assert args.reset_phases_ms and all(v in [0,4,8,12,16] for v in args.reset_phases_ms)
    assert not args.context_source or args.context_source.is_dir()
    import numpy as np
    import torch
    from learning_update_coherent_anchor import optimizers,update
    verify_bound_inputs();torch.set_num_threads(1);torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    stage=ROOT/args.name;stage.mkdir(exist_ok=False);(stage/'source').mkdir()
    started=time.perf_counter();deadline=started+1680.
    saved=torch.load(args.initial,map_location='cpu',weights_only=False);action_dim=saved['interface'][1]
    policy=policy_from(saved,args.seed);teacher=policy_from(saved,args.seed).eval().requires_grad_(False)
    if args.all_actor:
        assert not args.heads_only and not args.correction_only
        policy.actor.requires_grad_(True)
    std=[args.steer_std,args.feed_std,args.preview_std][:action_dim]
    assert all(0.<s<=.1 for s in std)
    if args.heads_only:
        assert action_dim==3
        for p in policy.actor.parameters():p.requires_grad_(False)
        for p in policy.actor[4].parameters():p.requires_grad_(True)
        weight_mask=torch.ones_like(policy.actor[4].weight);weight_mask[0]=0.
        bias_mask=torch.tensor([0.,1.,1.])
        policy.actor[4].weight.register_hook(lambda g:g*weight_mask)
        policy.actor[4].bias.register_hook(lambda g:g*bias_mask)
    if args.correction_only:
        assert not args.heads_only and saved.get('policy_class')=='GatedHeadPolicy'
        policy.actor.backbone.requires_grad_(False);policy.actor.correction.requires_grad_(True)
    with torch.no_grad():policy.log_std.copy_(torch.log(torch.tensor(std)))
    policy.log_std.requires_grad_(False);optimizer=optimizers(policy)
    sources=[Path(__file__)]+[ROOT/n for n in ['saved_local_contexts.py','corner_deficit_objective.py',
        'learning_update_coherent_anchor.py','direct_two_action_control.py','compact_fine_policy.py',
        'training_rejection_handling.py','gated_head_policy.py']]
    if saved.get('policy_class') in ['LocalizedJointPolicy','LocalizedPreviewPolicy']:
        sources.extend(ROOT/n for n in ['research_policy_loader.py','localized_joint_correction.py','localized_preview_head.py'])
    hashes={str(p):sha(p) for p in sources}
    for p in sources:shutil.copy2(p,stage/'source'/p.name)
    write(stage/'registration.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),kind='On-policy PPO with two physical integration resolutions',
        wall_budget_s=1800,worker_deadline_s=1680,initial_checkpoint=str(args.initial),initial_checkpoint_sha256=sha(args.initial),
        source_sha256=hashes,seed=args.seed,updates=args.updates,batch_size=2048,
        per_resolution_cases={'original':1024-args.dense_decisions,'dense':args.dense_decisions},
        critical_corner_reset_repeats=args.critical_repeat,
        critical_corner_ids=args.critical_corners,training_reset_phases_ms=args.reset_phases_ms,
        actual_context_source=str(args.context_source) if args.context_source else None,
        actual_context_index=args.context_index if args.context_source else None,
        extra_comparators=str(args.extra_comparators) if args.extra_comparators else None,
        extra_comparators_sha256=sha(args.extra_comparators) if args.extra_comparators else None,
        explicitly_unfreeze_entire_actor=args.all_actor,
        certification_rejections='Known physical rejection terminates training episode with unchanged failure penalty;accepted partial XY/V witnessed and saved. Full evaluation remains strict and unwrapped.',
        dt_s=[.001,.0005],ordinary_decision_s=.02,observation_dimension=14,actor_outputs=action_dim,policy_memory_scalars=0,
        exploration_std=std,kl_limit=.01,scheduled_dense_checks=list(range(args.check_every,args.updates+1,args.check_every)),original_checks='Both dense checks plus all34local gates at both dt' if args.local_screen else 'Only if both dense checks succeed',
        trainable_actor='Nonlinear correction only;entire original14input backbone frozen' if args.correction_only else ('Feed and preview output rows only;shared features and steering frozen' if args.heads_only else 'All actor layers'),
        trainable_actor_parameters=sum(p.numel() for p in policy.actor.parameters() if p.requires_grad),
        evidence_gap=('Static16feature updates cannot affect146required repair observations; use fresh on-policy updates of the entire14input actor,actual incumbent contexts,phase variation,and newly validated two-action corner budgets.' if args.all_actor and args.context_source else 'Previous narrow exploration and nominal-only PPO did not close the joint gap;preserve steering if requested and train on both physical resolutions.'),
        procedure='One current actor collects fresh transitions in two isolated dt processes;all34original local reset states plus continuous dense episodes. Same time reward and dt-correct corner cost. Own initializer anchors. Step size not observed.',
        adoption='A checkpoint needs full success at both resolutions,without discarding faster valid previous baselines. Main method additionally needs joint all34 and numerical time gates. No final acceptance from training alone.',
        stop=f'At{args.updates}updates or registered wall deadline;retain all sampled batches and failures;do not repeat unchanged if joint evidence does not improve.',
        independence='Conditional PPO seed from shared historical initialization;not an independent from-scratch repetition.',goal_complete=False,method_frozen=False))
    ax=[]
    anchor_paths=[MAIN/'eval_8000_original.npz',MAIN/'eval_8000_dense.npz',
                  ROOT/'direct14_farvector_local_feedback_fit_round21_dense_half_ms/trace.npz']
    if args.context_source:
        anchor_paths=[args.context_source/f'eval_{args.context_index:04d}_{case}_{key}.npz'
                      for case in ['original','dense'] for key in ['1ms','half_ms']]
    for key in ['1ms','half_ms']:
        path=args.initial.parent/f'eval_{saved.get("updates",0):04d}_dense_{key}.npz'
        if path.exists():anchor_paths.append(path)
    anchor_paths=list(dict.fromkeys(anchor_paths))
    write(stage/'anchor_source_binding.json',{str(p):sha(p) for p in anchor_paths})
    for path in anchor_paths:
        with np.load(path) as data:ax.append(data['observations'][::8 if 'original' in path.stem else 1].copy())
    ax=torch.tensor(np.vstack(ax),dtype=torch.float32)
    with torch.no_grad():ay=teacher.actor(ax)
    anchors=(ax,ay);np.savez_compressed(stage/'own_initializer_anchors.npz',observations=ax.numpy(),targets=ay.numpy())
    def save(count):
        path=stage/f'checkpoint_{count:04d}.pt'
        if args.heads_only:
            for k,v in saved['policy'].items():
                if not k.startswith('actor.'):continue
                assert torch.equal(policy.state_dict()[k][0],v[0]) if k.startswith('actor.4.') else torch.equal(policy.state_dict()[k],v)
        if args.correction_only:
            assert all(torch.equal(v,saved['policy']['actor.backbone.'+k]) for k,v in policy.actor.backbone.state_dict().items())
        data=dict(saved);data.update(policy=policy.state_dict(),optimizers={k:v.state_dict() for k,v in optimizer.items()},
            interface=[14,action_dim,args.seed],updates=count,steps=count*2048,mixed_resolution_ppo_updates=count,
            ppo_updates_this_round=count,new_ppo_updates=count,
            parent_mixed_resolution_updates=saved.get('mixed_resolution_ppo_updates',0),
            frozen_steering_and_shared_features=bool(args.heads_only or args.correction_only),
            frozen_complete_base_actor=bool(args.correction_only),
            observation_dimension=14,actor_outputs=action_dim,policy_memory_scalars=0,initial_checkpoint=str(args.initial),
            initial_checkpoint_sha256=sha(args.initial),torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),
            python_rng=random.getstate(),training_kind='Fresh mixed-resolution PPO with own initializer auxiliary anchors')
        torch.save(data,path);return path
    checkpoint=save(0);processes=[];connections=[];count=0;stats=[];evaluations=[];error=None
    def receive(connection):
        while time.perf_counter()<deadline:
            if connection.poll(1.):
                message=connection.recv()
                if 'success' not in message and isinstance(message.get('error'),str):raise RuntimeError(message['error'])
                return message
        raise TimeoutError('Registered30minute round limit')
    try:
        ctx=mp.get_context('spawn')
        for dt in [.001,.0005]:
            parent,child=ctx.Pipe();process=ctx.Process(target=physics_worker,args=(child,dt,args,stage,deadline))
            process.start();child.close();processes.append(process);connections.append(parent)
        for connection in connections:receive(connection)
        print('MIXED_CONTEXTS_VERIFIED',flush=True)
        while count<args.updates and time.perf_counter()<deadline-240.:
            for connection in connections:connection.send(dict(kind='collect',checkpoint=str(checkpoint)))
            responses=[receive(c) for c in connections];parts=[]
            for response in responses:
                with np.load(response['batch']) as data:parts.append({k:data[k].copy() for k in data.files if k not in ['case','corner_id','reset_phase_ms','dt_s']})
            batch={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
            assert len(batch['reward'])==2048 and np.allclose(batch['reward']+batch['corner_penalty_s'],batch['physical_reward'],atol=1e-12,rtol=0)
            result=update(policy,optimizer,batch,anchors,kl_limit=.01);count+=1
            result.update(update=count,wall_s=time.perf_counter()-started,collection_wall_s=[r['wall_s'] for r in responses])
            stats.append(result);write(stage/'updates.json',stats);checkpoint=save(count)
            print('MIXED_PPO',count,'KL',result['post_update_exact_kl'],'wall',round(time.perf_counter()-started,1),flush=True)
            if count%args.check_every==0:
                for c in connections:c.send(dict(kind='evaluate',case='dense',checkpoint=str(checkpoint)))
                dense=[receive(c) for c in connections]
                record=dict(update=count,dense=dense)
                print('DENSE_CHECK',count,[(r['success'],r['complete_time_s'],r['elapsed_prefix_s']) for r in dense],flush=True)
                eligible=all(r['success'] for r in dense)
                if eligible and args.local_screen:
                    for c in connections:c.send(dict(kind='local_check',checkpoint=str(checkpoint)))
                    record['local']=[receive(c) for c in connections]
                    eligible=all(r['valid_count']==34 and r['budget_count']==34 for r in record['local'])
                    print('LOCAL_SCREEN',count,[(r['valid_count'],r['budget_count']) for r in record['local']],flush=True)
                if eligible and time.perf_counter()<deadline-400.:
                    for c in connections:c.send(dict(kind='evaluate',case='original',checkpoint=str(checkpoint)))
                    record['original']=[receive(c) for c in connections]
                    print('ORIGINAL_CHECK',count,[(r['success'],r['complete_time_s']) for r in record['original']],flush=True)
                evaluations.append(record);write(stage/'evaluations.json',evaluations)
    except BaseException:error=traceback.format_exc();print(error,flush=True)
    finally:
        for c in connections:
            try:c.send(dict(kind='stop'))
            except (BrokenPipeError,EOFError,OSError):pass
        for p in processes:
            p.join(timeout=5.)
            if p.is_alive():p.terminate();p.join(timeout=5.)
        for c in connections:c.close()
        write(stage/'result.json',dict(ppo_updates=count,new_decisions=count*2048,error=error,evaluations=evaluations,
            initial_checkpoint=str(args.initial),latest_checkpoint=str(checkpoint),wall_s=time.perf_counter()-started,
            goal_complete=False,method_frozen=False))
        write(stage/'receipt.json',dict(status='finished' if error is None else 'error',wall_s=time.perf_counter()-started,
            sources_unchanged=all(sha(p)==h for p,h in hashes.items()),goal_complete=False))


if __name__=='__main__':main()
