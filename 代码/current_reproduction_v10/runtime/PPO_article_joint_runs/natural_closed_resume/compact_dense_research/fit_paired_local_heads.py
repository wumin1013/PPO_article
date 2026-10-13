"""Fit only feed/preview rows to paired physical teacher data;zero new PPO."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import argparse
import json
import multiprocessing as mp
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime,timezone
from evaluate_historical_dense import ROOT,BASE,write,sha,verify_bound_inputs
from probe_paired_local_repair import INITIAL,SOURCE
from train_mixed_resolution_ppo import HALF_METRICS,physics_worker,policy_from

PROBES=ROOT/'direct14_paired_local_repair_round35'


def collect(dt,stage,deadline,guidance_labels=False,probe_root=PROBES):
    import numpy as np
    import torch
    from run_refined_turn_candidate import configure_resolution
    from compact_farvector_runtime import FarVectorRuntime
    from saved_local_contexts import from_complete_trace
    from corner_deficit_objective import CornerDeficitCost
    from natural_case_audit import analytic_command_audit
    from natural_case_metrics import ordered_projection
    import joint_runtime
    from guidance_action_activity import observe_preview_activity
    from observable_feedback_gate import modify
    observable=bool(json.loads((probe_root/'registration.json').read_text(encoding='utf-8')).get('observable_feedback_gate',False))
    torch.set_num_threads(1)
    if dt!=.001:configure_resolution(dt)
    key='1ms' if dt==.001 else 'half_ms'
    geometry=json.loads((BASE/'paper_evidence_review/natural_closed_case/geometry.json').read_text(encoding='utf-8'))
    full=SOURCE/'eval_8000_original.npz' if dt==.001 else ROOT/'direct14_farvector_local_feedback_fit_round21_original_half_ms/trace.npz'
    gates=SOURCE/'eval_8000_original_gates.json' if dt==.001 else full.parent/'gate_events.json'
    ref,_,contexts,audit=from_complete_trace(full,gates,geometry,dt=dt)
    verified=json.loads((probe_root/key/'context_audit.json').read_text(encoding='utf-8'))
    assert verified['trace_sha256']==audit['trace_sha256'] and verified['four_decision_replays_all34_pass']
    policy=policy_from(torch.load(INITIAL,map_location='cpu',weights_only=False),922302)
    selected=json.loads((probe_root/'result.json').read_text(encoding='utf-8'))['selected']
    responses={s['id']:s['selected']['response'] for s in selected};assert len(responses)==2
    cost=CornerDeficitCost(ref,geometry,dt=dt,metric_paths=HALF_METRICS if dt==.0005 else None)
    records=[];xs=[];ys=[];ids=[];activities=[]
    schedule=[(context,corner,phase) for context,corner in zip(contexts,geometry['sharp_corners'])
              for phase in ([0,4,8,12,16] if guidance_labels and context['id'] in responses else [0])]
    replay_errors=[]
    for context,corner,phase in schedule:
        if time.perf_counter()>deadline-10:raise TimeoutError('Registered teacher collection deadline')
        env=FarVectorRuntime(ref);env.restore(context['state'])
        if phase:
            held=env.held.copy();latent=held/np.array([np.pi/2,50.,5.6])-np.array([0.,1.,6.4/5.6])
            previous=joint_runtime.MACRO_TICKS
            try:
                joint_runtime.MACRO_TICKS=round(phase*.001/dt)
                _,_,done,_=env.step(latent);assert not done
            finally:joint_runtime.MACRO_TICKS=previous
        start=env.env.state();chunks=[]
        reset_activity,get_activity=observe_preview_activity(env) if guidance_labels else (lambda:None,lambda:float('nan'))
        response=responses.get(context['id'],[0.,1.])
        local_x=[];local_y=[];local_activity=[]
        for _ in range(64):
            x=env.observation();z=policy.act(x,True)[0].copy()
            if observable:z=modify(z,x,*response)
            else:
                z[1]+=response[0];z[2]=((6.4+5.6*float(z[2]))*response[1]-6.4)/5.6
            reset_activity()
            _,_,done,info=env.step(z);chunks.append(info['rows']);local_x.append(x);local_y.append(z);local_activity.append(get_activity())
            if done:break
        rows=np.vstack(chunks);progress=ordered_projection(ref,rows[:,:2]);lo=corner['reference_s_mm']-10.;hi=corner['reference_s_mm']+10.
        complete=bool(progress.min()<lo and progress.max()>hi);mask=(progress>=lo)&(progress<=hi)
        a=np.diff(np.vstack([start['velocity'],rows[:,2:4]]),axis=0)/dt;j=np.diff(np.vstack([start['acceleration'],a]),axis=0)/dt
        peaks=[float(np.linalg.norm(rows[:,2:4],axis=1).max()),float(np.linalg.norm(a,axis=1).max()),float(np.linalg.norm(j,axis=1).max())]
        corridor=analytic_command_audit(start['position'],rows,geometry)
        valid=bool(not env.done and complete and corridor['analytic_corridor_pass'] and np.all(np.array(peaks)<=[100.000001,2000.001,62500.02]))
        deficit=float(np.maximum(100-np.linalg.norm(rows[mask,2:4],axis=1),0).sum()*dt)
        item=dict(id=context['id'],phase_ms=phase,valid=valid,budget_pass=bool(valid and deficit<=cost.budgets[context['id']]),
            deficit_mm=deficit,budget_mm=cost.budgets[context['id']],response=response,
            maximum_distance_mm=corridor['maximum_evaluated_distance_mm'],peaks=peaks,reason=env.reason)
        if guidance_labels:
            if phase==0:
                previous_trace=ROOT/'direct14_paired_local_heads_round36'/f'teacher_{key}_{context["id"]}.npz'
            else:
                chosen=next(s['selected'] for s in selected if s['id']==context['id'])
                previous_trace=probe_root/key/f'{context["id"]}_r{chosen["index"]:02d}_p{phase:02d}.npz'
            with np.load(previous_trace) as previous_data:
                error=float(abs(previous_data['rows'][:,:4]-rows[:,:4]).max())
            assert error==0.,('Diagnostic queries changed physical rollout',key,context['id'],phase,error)
            replay_errors.append(error)
        records.append(item);xs.extend(local_x);ys.extend(local_y);ids.extend([context['id']]*len(local_x));activities.extend(local_activity)
        suffix=f'_p{phase:02d}' if phase else ''
        np.savez_compressed(stage/f'teacher_{key}_{context["id"]}{suffix}.npz',observations=np.array(local_x),latents=np.array(local_y),rows=rows,
                            preview_activity_rad=np.array(local_activity),
                            start_position=start['position'],initial_velocity=start['velocity'],initial_acceleration=start['acceleration'],dt_s=dt)
    np.savez_compressed(stage/f'teacher_dataset_{key}.npz',observations=np.array(xs),targets=np.array(ys),corner_id=np.array(ids),preview_activity_rad=np.array(activities))
    report=dict(dt_s=dt,records=records,valid_count=sum(r['valid'] for r in records if r['phase_ms']==0),
        budget_count=sum(r['budget_pass'] for r in records if r['phase_ms']==0),all_registered_fragments_pass=all(r['budget_pass'] for r in records),
        maximum_replay_xyv_error=max(replay_errors) if replay_errors else None,new_ppo_updates=0)
    write(stage/f'teacher_audit_{key}.json',report)
    print('PAIRED_TEACHERS',key,report['valid_count'],report['budget_count'],flush=True)
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--name',required=True);parser.add_argument('--guidance-labels',action='store_true')
    parser.add_argument('--probes',type=Path,default=PROBES);parser.add_argument('--nonlinear-head',action='store_true')
    parser.add_argument('--reuse-teachers',type=Path);parser.add_argument('--preview-only',action='store_true')
    parser.add_argument('--dense-weight',type=float,default=1.);parser.add_argument('--verify-replay',type=Path);args=parser.parse_args()
    assert not args.preview_only or args.nonlinear_head
    assert 1.<=args.dense_weight<=8.
    probe_root=args.probes.resolve()
    assert Path(args.name).name==args.name
    import numpy as np
    import torch
    from gated_head_policy import GatedHeadPolicy
    from concurrent.futures import ProcessPoolExecutor
    verify_bound_inputs();stage=ROOT/args.name;stage.mkdir(exist_ok=False);(stage/'source').mkdir()
    started=time.perf_counter();deadline=started+1680.;torch.set_num_threads(1);torch.manual_seed(922302)
    sources=[Path(__file__),ROOT/'train_mixed_resolution_ppo.py',ROOT/'probe_paired_local_repair.py',
             ROOT/'saved_local_contexts.py',ROOT/'corner_deficit_objective.py',ROOT/'guidance_action_activity.py',ROOT/'observable_feedback_gate.py',ROOT/'gated_head_policy.py']
    checkpoints=[4000,12000,30000] if args.nonlinear_head else [1000,4000,12000]
    hashes={str(p):sha(p) for p in sources}
    for p in sources:shutil.copy2(p,stage/'source'/p.name)
    write(stage/'registration.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),kind='Offline nonlinear feed/preview correction;zero newPPO' if args.nonlinear_head else 'Offline feed/preview-row fitting;zero newPPO',
        wall_budget_s=1800,worker_deadline_s=1680,checkpoint=str(INITIAL),checkpoint_sha256=sha(INITIAL),source_sha256=hashes,
        probe_directory=str(probe_root),probe_result_sha256=sha(probe_root/'result.json'),fit_checkpoints=checkpoints,observation_dimension=14,policy_memory_scalars=0,
        actor_outputs=3,trainable_parameters=(2113 if args.preview_only else 2146) if args.nonlinear_head else 130,
        nonlinear_feed_preview_head=args.nonlinear_head,
        preview_only_correction=args.preview_only,dense_anchor_sampling_weight=args.dense_weight,
        reused_teacher_source=str(args.reuse_teachers.resolve()) if args.reuse_teachers else None,
        replay_verification_checkpoint=str(args.verify_replay.resolve()) if args.verify_replay else None,
        actor_architecture=f'Frozen14-64-64-3backbone plus64-32-{1 if args.preview_only else 2}bounded correction activated by current observed geometry;one memoryless14input policy' if args.nonlinear_head else '14-64-64-3,only final2rows trainable',
        physical_guidance_activity_labels=args.guidance_labels,
        preview_inactive_threshold_rad=1e-6 if args.guidance_labels else None,
        preview_activity_probe_distances_mm=[.8,6.,12.] if args.guidance_labels else None,
        evidence_gap='Paired robust local06/30feedback targets now exist;need one deployable network while preserving dense and other corners.',
        procedure=('First validate all34local teachers at each dt. Stop before fitting if any teacher misses a physical or deficit gate. '
                   +((('Fit a64-32-1bounded preview-only correction;steering and feed functions remain exact. ' if args.preview_only else 'Fit a64-32-2bounded nonlinear correction to feed/preview. ')
                      +'Correction activates only by observed geometry;the original14input backbone is bitwise frozen. ') if args.nonlinear_head else 'Fit only feed/preview final-layer rows;steering and shared features bitwise fixed. ')
                   +'Mask preview labels on nearly exact straight states or explicitly registered guidance inactivity. Use both-dt dense feedback anchors.'),
        adoption='Full dense and original at both dt;all34and registered time margin,then matched two-action and mechanism/repetition evidence. Local teachers are never final-policy results.',
        stop=f'Failed local teacher gate,{checkpoints[-1]}fitsteps,or wall deadline. No physical bound,task,or comparison changes.',goal_complete=False,method_frozen=False))
    ctx=mp.get_context('spawn')
    if args.reuse_teachers:
        previous=args.reuse_teachers.resolve();registration=json.loads((previous/'registration.json').read_text(encoding='utf-8'))
        assert registration['checkpoint_sha256']==sha(INITIAL) and registration['probe_result_sha256']==sha(probe_root/'result.json')
        assert registration['physical_guidance_activity_labels']==args.guidance_labels
        bindings={}
        for key in ['1ms','half_ms']:
            for filename in [f'teacher_dataset_{key}.npz',f'teacher_audit_{key}.json']:
                path=previous/filename;shutil.copy2(path,stage/filename);bindings[str(path)]=sha(path)
        local=[json.loads((stage/f'teacher_audit_{key}.json').read_text(encoding='utf-8')) for key in ['1ms','half_ms']]
        write(stage/'reused_teacher_audit.json',dict(source_sha256=bindings,new_preparation_physics_ticks=0))
    else:
        with ProcessPoolExecutor(max_workers=2,mp_context=ctx) as pool:
            futures=[pool.submit(collect,dt,stage,deadline,args.guidance_labels,probe_root) for dt in [.001,.0005]]
            local=[f.result() for f in futures]
    if not all(r['valid_count']==34 and r['budget_count']==34 and r['all_registered_fragments_pass'] for r in local):
        write(stage/'result.json',dict(status='teacher_gate_failed_before_fitting',local_audits=local,new_ppo_updates=0,offline_fit_steps=0,
            wall_s=time.perf_counter()-started,goal_complete=False,method_frozen=False))
        write(stage/'receipt.json',dict(status='finished',wall_s=time.perf_counter()-started,sources_unchanged=all(sha(p)==h for p,h in hashes.items()),goal_complete=False))
        return
    saved=torch.load(INITIAL,map_location='cpu',weights_only=False);policy=policy_from(saved,922302)
    teacher=policy_from(saved,922302).eval().requires_grad_(False)
    if args.nonlinear_head:
        policy=GatedHeadPolicy(922302,preview_only=args.preview_only);policy.initialize(teacher)
    datasets=[];bindings={}
    def add(path,x,y,critical=False,activity=None,sampling_weight=None):
        bindings[str(path)]=sha(path);x=torch.tensor(x,dtype=torch.float32);y=torch.tensor(y,dtype=torch.float32)
        mask=torch.ones((len(x),2));straight=(x[:,0].abs()<.005)&(x[:,6].abs()<.003)&(x[:,13].abs()<.003)
        if activity is None:mask[straight,1]=0.
        else:mask[torch.as_tensor(activity)<=1e-6,1]=0.
        with torch.no_grad():h=teacher.actor[:4](x);base=teacher.actor(x)
        datasets.append(dict(h=h,x=x,base=base,target=y[:,1:],mask=mask,critical=critical,
                             sampling_weight=sampling_weight if sampling_weight is not None else (4. if critical else 1.)))
    for key in ['1ms','half_ms']:
        path=stage/f'teacher_dataset_{key}.npz'
        with np.load(path) as d:
            if args.guidance_labels:
                critical=np.isin(d['corner_id'],['corner_06','corner_30'])
                for chosen,flag in [(critical,True),(~critical,False)]:
                    add(path,d['observations'][chosen],d['targets'][chosen],critical=flag,activity=d['preview_activity_rad'][chosen])
            else:add(path,d['observations'],d['targets'])
    selection=json.loads((probe_root/'result.json').read_text(encoding='utf-8'))['selected']
    for selected in ([] if args.guidance_labels else selection):
        for item in selected['selected']['traces']:
            path=Path(item)
            with np.load(path) as d:add(path,d['observations'],d['latents'],critical=True)
    dense_paths=[INITIAL.parent/'dense.npz',ROOT/'direct14_head_bias_search_round29_dense_half_08/trace.npz']
    for path in dense_paths:
        with np.load(path) as d:x=torch.tensor(d['observations'],dtype=torch.float32)
        with torch.no_grad():y=teacher.actor(x).numpy()
        add(path,x.numpy(),y,sampling_weight=args.dense_weight)
    path=SOURCE/'eval_8000_original.npz'
    with np.load(path) as d:
        x=d['observations'];indices=np.unique(np.r_[np.arange(100),np.arange(len(x)-100,len(x)),np.arange(0,len(x),16)])
        x=torch.tensor(x[indices],dtype=torch.float32)
    with torch.no_grad():y=teacher.actor(x).numpy()
    add(path,x.numpy(),y)
    h=torch.cat([d['h'] for d in datasets]);target=torch.cat([d['target'] for d in datasets]);mask=torch.cat([d['mask'] for d in datasets])
    observations=torch.cat([d['x'] for d in datasets]);base=torch.cat([d['base'] for d in datasets])
    weights=torch.cat([torch.full((len(d['h']),),d['sampling_weight']) for d in datasets])
    write(stage/'dataset.json',dict(source_sha256=bindings,samples=len(h),critical_weight=4.,masked_preview_samples=int((mask[:,1]==0).sum()),
        scope='Physical local teacher rollouts plus explicit incumbent feedback anchors;these are offline targets,not PPO samples.'))
    w=torch.nn.Parameter(saved['policy']['actor.4.weight'][1:].clone());b=torch.nn.Parameter(saved['policy']['actor.4.bias'][1:].clone())
    w0=w.detach().clone();b0=b.detach().clone();optimizer=torch.optim.Adam([w,b],lr=2e-4)
    if args.nonlinear_head:
        with torch.no_grad():
            reference=teacher.actor(observations)
            initial_error=float(abs(policy.actor(observations)-reference).max())
            cached_roundoff=float(abs(reference-base).max())
        assert initial_error==0.,initial_error
        assert cached_roundoff<2e-6,cached_roundoff
        optimizer=torch.optim.Adam(policy.actor.correction.parameters(),lr=5e-4)
        write(stage/'initialization_audit.json',dict(maximum_actor_output_error=initial_error,
            cached_group_vs_full_batch_float32_roundoff=cached_roundoff,
            additional_trainable_parameters=sum(p.numel() for p in policy.actor.correction.parameters()),
            observation_dimension=14,policy_memory_scalars=0,initial_behavior_unchanged=True))
    scale=torch.tensor([.04,.1]);connections=[];processes=[];evaluations=[];logs=[];iteration=0;error=None
    def receive(c):
        while time.perf_counter()<deadline:
            if c.poll(1.):
                r=c.recv()
                if 'success' not in r and isinstance(r.get('error'),str):raise RuntimeError(r['error'])
                return r
        raise TimeoutError('Registered offline-fit deadline')
    try:
        worker_args=SimpleNamespace(initial=INITIAL,seed=922302)
        for dt in [.001,.0005]:
            parent,child=ctx.Pipe();p=ctx.Process(target=physics_worker,args=(child,dt,worker_args,stage,deadline));p.start();child.close()
            connections.append(parent);processes.append(p)
        for c in connections:receive(c)
        for iteration in range(1,checkpoints[-1]+1):
            if time.perf_counter()>deadline-250.:break
            ids=torch.multinomial(weights,512,replacement=True)
            prediction=policy.actor.corrected(h[ids],base[ids],observations[ids]) if args.nonlinear_head else torch.nn.functional.linear(h[ids],w,b)
            difference=prediction-target[ids]
            difference=torch.where(target[ids]>=1.,-torch.relu(1.02-prediction),difference)
            difference=torch.where(target[ids]<=-1.,torch.relu(prediction+1.02),difference)
            loss=((difference/scale).square()*mask[ids]).mean()
            if not args.nonlinear_head:loss=loss+1e-3*((w-w0).square().mean()+(b-b0).square().mean())
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            if iteration%500==0:
                logs.append(dict(iteration=iteration,loss=float(loss.detach()),wall_s=time.perf_counter()-started));write(stage/'fit_log.json',logs)
            if iteration not in checkpoints:continue
            if args.nonlinear_head:
                assert all(torch.equal(v,teacher.actor.state_dict()[k]) for k,v in policy.actor.backbone.state_dict().items())
            else:
                with torch.no_grad():policy.actor[4].weight[1:].copy_(w);policy.actor[4].bias[1:].copy_(b)
                for k,v in saved['policy'].items():
                    if k.startswith('actor.'):
                        assert torch.equal(policy.state_dict()[k][0],v[0]) if k.startswith('actor.4.') else torch.equal(policy.state_dict()[k],v)
            checkpoint=stage/f'checkpoint_{iteration:04d}.pt'
            if args.verify_replay:
                expected=torch.load(args.verify_replay,map_location='cpu',weights_only=False)
                if iteration==expected['offline_fit_steps']:
                    equal={k:torch.equal(v,expected['policy'][k]) for k,v in policy.state_dict().items()}
                    assert all(equal.values()),equal
                    write(stage/'replayed_fit_audit.json',dict(source=str(args.verify_replay),source_sha256=sha(args.verify_replay),
                        repeated_offline_compute_steps=iteration,all_policy_tensors_bitwise_equal=True,
                        new_ppo_updates=0,scope='Repeated offline computation for exact continuation;not additional unique ancestral updates or a repetition experiment.'))
            torch.save(dict(policy=policy.state_dict(),interface=[14,3,922302],runtime_class='FarVectorRuntime',policy_class='GatedHeadPolicy' if args.nonlinear_head else 'FineCompactPolicy',
                observation_features=saved['observation_features'],feed_gain=1.,updates=iteration,checkpoint_index_kind='offline_fit_iteration',
                offline_fit_steps=iteration,new_ppo_updates=0,ppo_updates_this_round=0,inherited_ppo_updates=88,policy_memory_scalars=0,
                frozen_steering_and_shared_features=True,initial_checkpoint=str(INITIAL),initial_checkpoint_sha256=sha(INITIAL),
                preview_only_correction=args.preview_only,
                training_kind='Offline observable-gated nonlinear feed/preview correction;notPPO' if args.nonlinear_head else 'Offline final-two-row fitting of paired local feedback;notPPO'),checkpoint)
            for c in connections:c.send(dict(kind='evaluate',case='dense',checkpoint=str(checkpoint)))
            dense=[receive(c) for c in connections];record=dict(update=iteration,update_kind='offline_fit',checkpoint=str(checkpoint),dense=dense)
            print('FIT_DENSE',iteration,[(r['success'],r['complete_time_s']) for r in dense],flush=True)
            if all(r['success'] for r in dense) and time.perf_counter()<deadline-420.:
                for c in connections:c.send(dict(kind='evaluate',case='original',checkpoint=str(checkpoint)))
                record['original']=[receive(c) for c in connections]
                print('FIT_ORIGINAL',iteration,[(r['success'],r['complete_time_s']) for r in record['original']],flush=True)
            evaluations.append(record);write(stage/'evaluations.json',evaluations)
    except BaseException:
        import traceback
        error=traceback.format_exc();print(error,flush=True)
    finally:
        for c in connections:
            try:c.send(dict(kind='stop'))
            except (BrokenPipeError,OSError):pass
        for p in processes:
            p.join(timeout=5.)
            if p.is_alive():p.terminate();p.join(timeout=5.)
        write(stage/'result.json',dict(offline_fit_steps=iteration,new_ppo_updates=0,error=error,evaluations=evaluations,
            wall_s=time.perf_counter()-started,goal_complete=False,method_frozen=False))
        write(stage/'receipt.json',dict(status='finished' if error is None else 'error',wall_s=time.perf_counter()-started,
            sources_unchanged=all(sha(p)==h for p,h in hashes.items()),goal_complete=False))


if __name__=='__main__':main()
