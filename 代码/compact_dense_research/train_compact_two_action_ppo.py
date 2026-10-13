"""Independent PPO seed runs for the same14input, genuinely two-output control."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime,timezone
from evaluate_historical_dense import ROOT,BASE,CASE,write,sha,verify_bound_inputs

SOURCE=ROOT/'direct14_two_action_initialization_round27'
INITIAL=SOURCE/'d_3/fit_12000.pt'
BATCHES={'original':1536,'dense':512}


def worker(args):
    import numpy as np
    import torch
    from collections import Counter
    from direct_two_action_control import DirectTwoPolicy,DirectTwoRuntime,DirectTwoTimeRuntime
    from saved_local_contexts import from_complete_trace
    from learning_update_coherent_anchor import optimizers,update
    from corner_deficit_objective import CornerDeficitCost
    from paper_complete_evaluation import evaluate
    from natural_case_audit import analytic_command_audit
    from natural_case_metrics import metrics
    stage=ROOT/args.name;started=time.perf_counter();deadline=started+1640.
    torch.set_num_threads(1);torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    folders={'original':BASE/'paper_evidence_review/natural_closed_case','dense':CASE}
    refs={k:np.loadtxt(p/'reference.csv',delimiter=',',skiprows=1) for k,p in folders.items()}
    geometries={k:json.loads((p/'geometry.json').read_text(encoding='utf-8')) for k,p in folders.items()}
    saved=torch.load(INITIAL,map_location='cpu',weights_only=False);distance=float(saved['preview_rule_mm'])
    policy=DirectTwoPolicy(args.seed);policy.load_state_dict(saved['policy'])
    teacher=DirectTwoPolicy(args.seed);teacher.load_state_dict(saved['policy']);teacher.eval().requires_grad_(False)
    with torch.no_grad():policy.log_std.copy_(torch.log(torch.tensor([.0007,.008])))
    policy.log_std.requires_grad_(False)
    optimizer=optimizers(policy)
    ref,expected,contexts,audit=from_complete_trace(SOURCE/'selected_original.npz',
        SOURCE/'selected_original_gates.json',geometries['original'])
    local=[];errors=[];validation_ticks=0
    for context in contexts:
        env=DirectTwoRuntime(ref,preview_rule_mm=distance);env.restore(context['state']);chunks=[]
        for _ in range(64):
            x=env.observation();local.append(x);_,_,_,info=env.step(teacher.act(x,True)[0]);chunks.append(info['rows'])
            assert not env.done
        rows=np.vstack(chunks);validation_ticks+=len(rows)
        error=float(np.max(abs(rows[:,:4]-expected[context['tick']:context['tick']+len(rows),:4])))
        assert error<2e-5,(context['id'],error);errors.append(error)
    torch.save(contexts,stage/'all34_contexts.pt')
    audit.update(maximum_replay_xyv_error=max(errors),new_validation_ticks=validation_ticks,
        contexts_sha256=sha(stage/'all34_contexts.pt'),all34_unmodified_local_checks_pass=True)
    write(stage/'context_audit.json',audit)
    trace=SOURCE/'d_3/dense_12000.npz'
    with np.load(trace) as data:dense_x=data['observations'].copy()
    ax=torch.tensor(np.vstack([local,dense_x]),dtype=torch.float32)
    with torch.no_grad():ay=teacher.actor(ax)
    anchors=(ax,ay)
    np.savez_compressed(stage/'auxiliary_anchors.npz',observations=ax.numpy(),targets=ay.numpy())
    envs={k:DirectTwoTimeRuntime(r,preview_rule_mm=distance,potential_speed=100.) for k,r in refs.items()}
    cost=CornerDeficitCost(ref,geometries['original'])
    count=steps=cursor=0;current_corner='';updates=[];evaluations=[];episodes=[]
    resumed_count=resumed_steps=0
    if args.resume:
        resumed=torch.load(args.resume,map_location='cpu',weights_only=False)
        assert resumed['interface']==[14,2,args.seed] and resumed['preview_rule_mm']==distance
        assert resumed['initial_checkpoint_sha256']==sha(INITIAL)
        policy.load_state_dict(resumed['policy'])
        for key,item in optimizer.items():item.load_state_dict(resumed['optimizers'][key])
        for case,env in envs.items():env.restore(resumed['environments'][case])
        count=int(resumed['updates']);steps=int(resumed['steps']);cursor=int(resumed['context_cursor'])
        resumed_count,resumed_steps=count,steps
        torch.set_rng_state(resumed['torch_rng']);np.random.set_state(resumed['numpy_rng']);random.setstate(resumed['python_rng'])
        write(stage/'resume_audit.json',dict(checkpoint=str(args.resume),checkpoint_sha256=sha(args.resume),
            retained_prior_ppo_updates=count,retained_prior_decisions=steps,
            optimizer_and_all_physical_states_and_rng_restored=True))

    def save(label):
        path=stage/(label+'.pt')
        torch.save(dict(policy=policy.state_dict(),optimizers={k:v.state_dict() for k,v in optimizer.items()},
            environments={k:v.state() for k,v in envs.items()},steps=steps,updates=count,context_cursor=cursor,
            torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),
            interface=[14,2,args.seed],policy_class='DirectTwoPolicy',runtime_class='DirectTwoRuntime',
            observation_dimension=14,actor_outputs=2,policy_memory_scalars=0,preview_rule_mm=distance,
            observation_features=envs['dense'].feature_names(),initial_checkpoint=str(INITIAL),
            initial_checkpoint_sha256=sha(INITIAL),inherited_ppo_updates=88,new_ppo_updates=count,
            ppo_updates_this_stage=count-resumed_count,resumed_prior_ppo_updates=resumed_count,
            resume_source=str(args.resume) if args.resume else None,
            training_kind='Fresh two-output PPO with own incumbent feedback anchors',
            independent_scope='Independent fine-tuning RNG from a shared pretrained and fitted initialization'),path)
        return path

    def checks(checkpoint):
        results={}
        for case in ['dense','original']:
            env=DirectTwoRuntime(refs[case],preview_rule_mm=distance)
            trace=stage/f'eval_{count:04d}_{case}.npz'
            result=evaluate(policy,env,trace,deadline)
            with np.load(trace) as data:
                rows=data['rows'];assert data['latents'].shape[1]==2
                assert np.all(data['actions'][:,2]==distance)
            result['analytic']=analytic_command_audit(refs[case][0],rows,geometries[case])
            if not result['analytic']['analytic_corridor_pass']:
                result.update(success=False,complete_time_s=None,motion_scope='failure_prefix_only')
            result.update(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
                observation_dimension=14,actor_outputs=2,policy_memory_scalars=0,global_preview_mm=distance,
                observation_features=env.feature_names(),ppo_updates_this_stage=count-resumed_count,
                independent_full_method_training=False,goal_complete=False,method_frozen=False)
            write(trace.with_suffix('.json'),result);write(stage/f'eval_{count:04d}_{case}_gates.json',env.tracker.events)
            if case=='original' and result['success']:
                write(stage/f'eval_{count:04d}_{case}_metrics.json',metrics(refs[case],rows,geometries[case]))
            results[case]=result
            print('FULL_CHECK',count,case,result['success'],result['complete_time_s'],flush=True)
        return dict(update=count,checkpoint=str(checkpoint),results=results)

    save('initial')
    while count<24 and time.perf_counter()<deadline-260.:
        batch={k:[] for k in ['observation','next_observation','latent','log_probability','value','reward','next_value',
            'terminal','segment_end','teacher_target','physical_reward','corner_penalty_s']}
        case_ids=[];corner_ids=[];collected=time.perf_counter()
        for case,size in BATCHES.items():
            for j in range(size):
                if case=='original' and (j%64==0 or envs[case].done):
                    context=contexts[cursor%len(contexts)];cursor+=1;envs[case].restore(context['state'])
                    current_corner=context['id'];cost.reset(current_corner)
                env=envs[case];x=env.observation();z,logp,value=policy.act(x)
                assert z.shape==(2,)
                x2,reward,done,info=env.step(z);physical=reward;penalty=0.
                if case=='original':penalty,_=cost.consume(info['rows']);reward-=penalty
                with torch.no_grad():
                    next_value=0. if done else float(policy.value(torch.as_tensor(x2)))
                    target=teacher.actor(torch.as_tensor(x)).numpy()
                values=[x,x2,z,logp,value,reward,next_value,done,j==size-1 or(case=='original' and j%64==63),target,physical,penalty]
                for key,item in zip(batch,values):batch[key].append(item)
                case_ids.append(case);corner_ids.append(current_corner if case=='original' else '')
                if done:
                    episodes.append(dict(case=case,update=count+1,success=env.success,reason=env.reason,
                        prefix_s=env.ticks*.001,complete_time_s=env.ticks*.001 if env.success else None))
                    if case=='dense':envs[case]=DirectTwoTimeRuntime(refs[case],preview_rule_mm=distance,potential_speed=100.)
        assert Counter(case_ids)==BATCHES
        assert np.allclose(np.asarray(batch['reward'])+np.asarray(batch['corner_penalty_s']),batch['physical_reward'],atol=1e-12,rtol=0)
        np.savez_compressed(stage/f'batch_{count+1:04d}.npz',**{k:np.asarray(v) for k,v in batch.items()},
            case=np.asarray(case_ids),corner_id=np.asarray(corner_ids))
        if args.verify_replayed_batch and count==resumed_count:
            with np.load(args.verify_replayed_batch) as expected_batch:
                equal={key:bool(np.array_equal(np.asarray(value),expected_batch[key])) for key,value in batch.items()}
            assert all(equal.values()),equal
            write(stage/'blocked_batch_replay_audit.json',dict(source=str(args.verify_replayed_batch),
                source_sha256=sha(args.verify_replayed_batch),bitwise_equal_fields=equal,
                new_physics_replayed_for_resume=True,duplicate_ppo_updates=0))
        stats=update(policy,optimizer,batch,anchors,kl_limit=.01)
        count+=1;steps+=sum(BATCHES.values());stats.update(update=count,steps=steps,collect_wall_s=time.perf_counter()-collected)
        updates.append(stats);write(stage/'updates.json',updates);write(stage/'episodes.json',episodes)
        checkpoint=save(f'checkpoint_{count:04d}')
        print('TWO_ACTION_PPO',count,'KL',stats['post_update_exact_kl'],'wall',time.perf_counter()-started,flush=True)
        if count in [8,16,24]:
            evaluations.append(checks(checkpoint));write(stage/'evaluations.json',evaluations)
    save('final')
    valid=[r for r in evaluations if r['results']['dense']['success']]
    best=min(valid,key=lambda r:r['results']['dense']['complete_time_s']) if valid else None
    write(stage/'result.json',dict(ppo_updates_this_round=count-resumed_count,new_decisions=steps-resumed_steps,inherited_ppo_updates=88,
        total_two_action_refinement_updates=count,resumed_prior_ppo_updates=resumed_count,
        evaluations=evaluations,fastest_complete_dense_ppo_checkpoint=best,
        initial_dense_time_s=15.879,initial_original_time_s=444.476,wall_s=time.perf_counter()-started,
        initial_checkpoint_retained=True,goal_complete=False,method_frozen=False,
        independent_scope='This seed is an independently sampled conditional PPO refinement;shared historical teacher and offline initialization are not independently trained from scratch.'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--name',required=True)
    parser.add_argument('--seed',type=int,required=True);parser.add_argument('--worker',action='store_true')
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--verify-replayed-batch',type=Path)
    args=parser.parse_args()
    assert Path(args.name).name==args.name
    if args.worker:
        worker(args);return
    verify_bound_inputs();stage=ROOT/args.name;stage.mkdir(exist_ok=False);(stage/'source').mkdir()
    sources=[Path(__file__)]+[ROOT/n for n in ['direct_two_action_control.py','saved_local_contexts.py',
        'learning_update_coherent_anchor.py','learning_update_guarded_mc.py','learning_update_compact_mc.py',
        'corner_deficit_objective.py','compact_control.py','compact_farvector_runtime.py']]
    hashes={str(p):sha(p) for p in sources}
    for p in sources:shutil.copy2(p,stage/'source'/p.name)
    resumed_updates=resumed_decisions=0
    if args.resume:
        import torch
        resume_metadata=torch.load(args.resume,map_location='cpu',weights_only=False)
        resumed_updates=int(resume_metadata['updates']);resumed_decisions=int(resume_metadata['steps'])
    write(stage/'registration.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),kind='Genuine14input two-output on-policy PPO',
        wall_budget_s=1800,worker_deadline_s=1640,child_timeout_s=1740,source_sha256=hashes,
        initial_checkpoint=str(INITIAL),initial_checkpoint_sha256=sha(INITIAL),seed=args.seed,
        updates=24-resumed_updates,total_refinement_update_target=24,new_decisions=49152-resumed_decisions,
        resume_checkpoint=str(args.resume) if args.resume else None,
        resume_checkpoint_sha256=sha(args.resume) if args.resume else None,
        replayed_blocked_batch=str(args.verify_replayed_batch) if args.verify_replayed_batch else None,
        replayed_blocked_batch_sha256=sha(args.verify_replayed_batch) if args.verify_replayed_batch else None,
        batches=BATCHES,full_checks=[8,16,24],exploration_std=[.0007,.008],kl_limit=.01,
        evidence_gap='A complete two-output offline initializer needs an independent PPO and checkpoint-selection opportunity.',
        procedure='Train all two-output actor weights and separate critic with fresh sampled physical transitions,same time reward and original34corner cost. Coherent own-initializer action anchors;fixed3mm preview selected from six registered distances. No inherited controller at execution.',
        adoption='Retain the fastest full valid dense checkpoint including the initial model;report original completion separately. Require independent paired-resolution checks before final comparison. Never discard a faster credible baseline to favor the main method.',
        stop='Twenty-four updates or wall deadline;keep all failures,initial model,and historical budget disclosures.',
        planned_seeds=[922302,922303,922304],independent_scope='Independent conditional PPO seeds from shared initialization,not independent full historical training.',
        goal_complete=False,method_frozen=False))
    started=time.perf_counter();code=None
    with (stage/'stdout.txt').open('w',encoding='utf-8') as out,(stage/'stderr.txt').open('w',encoding='utf-8') as err:
        try:
            run=subprocess.run([sys.executable,'-B',str(Path(__file__)),*sys.argv[1:],'--worker'],stdout=out,stderr=err,timeout=1740)
            status,code='finished',run.returncode
        except subprocess.TimeoutExpired:status='timeout'
    write(stage/'receipt.json',dict(status=status,exit_code=code,wall_s=time.perf_counter()-started,
        sources_unchanged=all(sha(p)==h for p,h in hashes.items()),goal_complete=False))
    print((stage/'stdout.txt').read_text(encoding='utf-8')[-4000:])
    if code!=0:print((stage/'stderr.txt').read_text(encoding='utf-8')[-4000:])


if __name__=='__main__':main()
