"""Bounded new PPO training with rolling restart state and complete cost records."""
from __future__ import annotations
import argparse
import copy
import gzip
import json
import math
import os
import shutil
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from study_io import BASE, ROOT, register, receipt, write_json, record_progress, sha, atomic_replace
from joint_runtime import JointRuntime
from joint_policy import JointPolicy, ppo_update

BATCH_KEYS = ['observation','latent','log_probability','value','reward','next_value','terminal']


def make_paths():
    rng = np.random.default_rng(915031)
    training = []
    for angle in [35.,60.,90.,115.,135.,155.]:
        for sign in [-1.,1.]:
            for replicate in range(2):
                first,last = rng.uniform(12.,28.,size=2)
                theta = math.radians(sign*angle)
                training.append(np.array([[-first,0.],[0.,0.],[last*math.cos(theta),last*math.sin(theta)]]))
    validation = []
    for angle in [45.,90.,130.,155.]:
        theta = math.radians(angle)
        validation.append(np.array([[-18.,0.],[0.,0.],[23*math.cos(theta),23*math.sin(theta)]]))
    return training,validation


def save_torch(path, value):
    temporary = path.with_suffix('.tmp')
    torch.save(value,temporary)
    atomic_replace(temporary,path)


def evaluate(policy, paths, stage, tag, fixed_d, costs, deadline, env_factory=JointRuntime):
    from validate_actions import difference_audit
    directory = stage/tag
    directory.mkdir()
    results = []
    for index,ref in enumerate(paths):
        env = env_factory(ref,fixed_d=fixed_d,max_time=5.)
        chunks,actions = [],[]
        while not env.done:
            if time.perf_counter()>deadline: raise TimeoutError('Registered wall-clock limit')
            action,_,_ = policy.act(env.observation(),deterministic=True)
            costs['evaluation_actor_calls'] += 1
            _,_,_,info = env.step(action)
            costs['evaluation_physics_ticks'] += len(info['rows'])
            chunks.append(info['rows'])
            actions.append(info['held_action'])
        rows = np.vstack(chunks)
        audit = difference_audit(ref[0],rows)
        success = bool(env.success and audit['kinematic_pass'] and audit['rest_pass'])
        results.append(dict(path=index,success=success,reason=env.reason,elapsed_s=env.ticks*.001,
            completion_time_s=env.ticks*.001 if success else None,geometry=env.tracker.snapshot(1.25),
            audit=audit,decisions=env.decisions))
        np.savez_compressed(directory/(str(index)+'.npz'),reference=ref,rows=rows,actions=actions)
        write_json(directory/(str(index)+'.json'),results[-1])
    write_json(directory/'summary.json',dict(results=results,successes=sum(r['success'] for r in results),
        successful_time_sum=sum(r['elapsed_s'] for r in results if r['success'])))
    print(dict(validation=tag,successes=sum(r['success'] for r in results),
               seconds=[r['completion_time_s'] for r in results]),flush=True)
    return results


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage',required=True)
    parser.add_argument('--mode',choices=['joint','fixed','d_only'],default='joint')
    parser.add_argument('--steps',type=int,default=4096)
    parser.add_argument('--seed',type=int,default=713)
    parser.add_argument('--fixed-d',type=float,default=3.)
    parser.add_argument('--wall-s',type=float,default=1800.)
    parser.add_argument('--resume-from',type=str)
    parser.add_argument('--curriculum',choices=['single','mixed_v2','mixed_v3'],default='single')
    parser.add_argument('--init-model',type=str)
    parser.add_argument('--recover-from',type=str)
    args = parser.parse_args()
    maximum = 30000 if args.curriculum.startswith('mixed') else 10000
    if args.steps<=0 or args.steps>maximum: raise ValueError('Registered curriculum decision budget exceeded')
    if args.curriculum.startswith('mixed') and args.resume_from:
        raise ValueError('Mixed curriculum begins a new episode; old on-policy buffers are not transferred')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    env_factory = JointRuntime
    if args.curriculum.startswith('mixed'):
        from mixed_curriculum import environment_factory
        if args.curriculum=='mixed_v3':
            from conditioned_curriculum_v3 import environment_factory
        curriculum_stage = BASE/'mixed_curriculum_preparation_v2'
        if not (curriculum_stage/'receipt.json').exists():
            raise ValueError('Shared curriculum must be prepared first')
        prepared_paths = json.loads((curriculum_stage/'paths.json').read_text(encoding='utf-8'))
        train_paths = [np.array(row,float) for row in prepared_paths['training']]
        valid_paths = [np.array(row,float) for row in prepared_paths['validation']]
        env_factory = environment_factory
    else:
        train_paths,valid_paths = make_paths()
    sources = [__file__,BASE/'study_io.py',BASE/'geometry_spec.py',BASE/'joint_runtime.py',
               BASE/'joint_policy.py',BASE/'task_contract.md',BASE/'validate_actions.py']
    if args.curriculum.startswith('mixed'):
        sources += [BASE/'mixed_curriculum.py',BASE/'joint_runtime_v2.py',BASE/'mixed_curriculum_protocol_v2.md',
            curriculum_stage/'paths.json',curriculum_stage/'receipt.json']
        if args.curriculum=='mixed_v3':
            sources += [BASE/'conditioned_curriculum_v3.py',BASE/'joint_runtime_v3.py',
                BASE/'conditioned_training_protocol_v3.md',BASE/'time_potential_validation_v3/receipt.json']
    if args.init_model:
        sources += [BASE/args.init_model]
    resume_source = None
    recovery = bool(args.recover_from)
    if recovery and args.resume_from:
        raise ValueError('Choose one source of continuation')
    if recovery:
        previous = BASE/args.recover_from
        failed = json.loads((previous/'failure.json').read_text(encoding='utf-8'))
        prior_registration=json.loads((previous/'registration.json').read_text(encoding='utf-8'))
        for name in ['joint_runtime.py','joint_runtime_v2.py','geometry_spec.py','joint_policy.py','mixed_curriculum.py']:
            source=BASE/name
            if sha(source)!=prior_registration['source_sha256'][source.relative_to(ROOT).as_posix()]:
                raise ValueError('Scientific implementation changed; cannot exact-resume '+name)
        previous_decisions=failed['costs']['training_actor_calls']
        if previous_decisions+args.steps>30000:
            raise ValueError('Recovery exceeds the original training decision limit')
        resume_source=previous/('resume_'+str(previous_decisions%2)+'.pt')
        sources += [previous/'failure.json',previous/'registration.json',resume_source]
    if args.resume_from:
        previous = BASE/args.resume_from
        previous_receipt = json.loads((previous/'receipt.json').read_text(encoding='utf-8'))
        previous_pointer = json.loads((previous/'resume_pointer.json').read_text(encoding='utf-8'))
        if previous_receipt['status']!='bounded_pilot_completed' or previous_receipt['mode']!=args.mode:
            raise ValueError('Continuation requires completed compatible training evidence')
        previous_decisions = previous_pointer['decision']
        if previous_decisions+args.steps>10000:
            raise ValueError('Cumulative first-pilot budget is 10000 decisions per arm')
        resume_source = previous/('resume_'+str(previous_decisions%2)+'.pt')
        sources += [previous/'receipt.json',previous/'resume_pointer.json',resume_source]
    stage = register(args.stage,sources,dict(training_decisions=args.steps,physics_main_ticks=args.steps*20,
        terminal_tail_ticks=args.steps*2100,validation_runs=2*len(valid_paths),max_episode_ticks=25000,wall_s=args.wall_s,
        checkpoint_interval=512,seed=args.seed,mode=args.mode,fixed_d=args.fixed_d if args.mode=='fixed' else None),
        'Cost calibration and bounded single-seed feasibility pilot; no claim of superiority or unseen-path generalization.')
    write_json(stage/'paths.json',dict(training=[r.tolist() for r in train_paths],validation=[r.tolist() for r in valid_paths],
        split='Development only; calibration/selection paths are not sealed test evidence.'))
    costs = dict(training_actor_calls=0,training_physics_ticks=0,evaluation_actor_calls=0,
                 evaluation_physics_ticks=0,ppo_updates=0,optimizer_minibatches=0)
    started = time.perf_counter()
    deadline = started+args.wall_s
    volatile = stage
    if args.curriculum.startswith('mixed'):
        volatile = Path('D:/PPO_article_joint_runs/rolling_states')/args.stage
        volatile.mkdir(parents=True,exist_ok=False)
        write_json(stage/'storage.json',dict(rolling_state_directory=str(volatile),
            purpose='Frequently replaced checkpoints outside synchronization directory'))
    try:
        fixed_d = args.fixed_d if args.mode=='fixed' else None
        env = env_factory(train_paths[0],fixed_d=fixed_d,max_time=5.)
        reference_policy = JointPolicy(len(env.observation()))
        if args.init_model:
            incoming = torch.load(BASE/args.init_model,map_location='cpu',weights_only=False)
            reference_policy.load_state_dict({key:value for key,value in incoming['policy'].items()
                if not key.startswith('reference_actor.')})
            reference_policy.active_mask.copy_(torch.tensor([1.,1.,1.]))
            if args.curriculum=='mixed_v2':
                with torch.no_grad():
                    reference_policy.actor[-1].weight[2].zero_()
                    reference_policy.actor[-1].bias[2] = float(np.arctanh((3.-6.4)/5.6))
                    reference_policy.log_std[2] = math.log(.10)
        policy = reference_policy if args.mode=='joint' else JointPolicy(len(env.observation()),mode=args.mode,reference_policy=reference_policy)
        if args.mode!='joint':
            policy.actor.load_state_dict(reference_policy.actor.state_dict())
            policy.critic.load_state_dict(reference_policy.critic.state_dict())
            policy.log_std.data.copy_(reference_policy.log_std.data)
        learning_rate = 7e-4 if args.curriculum.startswith('mixed') else 3e-4
        optimizer = torch.optim.Adam([p for p in policy.parameters() if p.requires_grad],lr=learning_rate,eps=1e-5)
        continuation = None
        if resume_source:
            continuation = torch.load(resume_source,map_location='cpu',weights_only=False)
            if len(continuation['batch']['reward']) and not recovery:
                raise ValueError('This continuation supports completed update boundaries only')
            policy.load_state_dict(continuation['policy'])
            optimizer.load_state_dict(continuation['optimizer'])
            torch.set_rng_state(continuation['torch_rng'])
            np.random.set_state(continuation['numpy_rng'])
            rng.bit_generator.state = continuation['generator_rng']
            if recovery and continuation['decision']!=previous_decisions:
                raise ValueError('Saved full macro state does not match failed-run cost ledger')
            if recovery and continuation['batch']['reward']:
                pending_observation=torch.tensor(np.asarray(continuation['batch']['observation']),dtype=torch.float32)
                pending_action=torch.tensor(np.asarray(continuation['batch']['latent']),dtype=torch.float32)
                with torch.no_grad():
                    checked_logp=policy.evaluate(pending_observation,pending_action)[0].numpy()
                if not np.allclose(checked_logp,continuation['batch']['log_probability'],atol=2e-5,rtol=0):
                    raise ValueError('Recovered on-policy likelihood mismatch')
            write_json(stage/'continuation.json',dict(source=args.recover_from or args.resume_from,
                previous_decisions=continuation['decision'],new_decisions=args.steps,
                source_snapshot_sha256=sha(resume_source),old_completed_experiments_replayed=0,
                exact_partial_buffer_restored=recovery))
        save_torch(stage/'initial.pt',dict(policy=policy.state_dict(),observation_dim=len(env.observation()),mode=args.mode))
        if continuation:
            validation_source=previous/('initial_validation' if recovery else 'final_validation')
            shutil.copytree(validation_source,stage/'initial_validation')
            initial = json.loads((stage/'initial_validation/summary.json').read_text(encoding='utf-8'))['results']
            write_json(stage/'initial_validation_reuse.json',dict(source=str(validation_source),
                purpose='Reuse original study initial validation' if recovery else 'Reuse identical continuation checkpoint validation',
                new_actor_calls=0,new_physics_ticks=0))
        else:
            initial = evaluate(policy,valid_paths,stage,'initial_validation',fixed_d,costs,deadline,env_factory)
        batch = {key:[] for key in BATCH_KEYS}
        episode_index = 0
        path_index = 0
        episode_return = 0.
        episode_rows = []
        episodes = []
        updates = []
        if continuation:
            path_index = continuation['path_index']
            episode_index = continuation['episode_index']
            env = env_factory(train_paths[path_index],fixed_d=fixed_d,max_time=5.)
            env.restore(continuation['environment'])
            episode_return = continuation['episode_return']
            episode_rows = continuation['episode_rows']
            if recovery:
                batch=continuation['batch']
        block_started = time.perf_counter()
        training_started = block_started
        with gzip.open(stage/'decisions.jsonl.gz','wt',encoding='utf-8') as journal:
            for decision in range(args.steps):
                study_decision=decision+(previous_decisions if recovery else 0)
                if time.perf_counter()>deadline: raise TimeoutError('Registered wall-clock limit')
                observation = env.observation()
                action,logp,value = policy.act(observation)
                costs['training_actor_calls'] += 1
                next_observation,reward,done,info = env.step(action)
                costs['training_physics_ticks'] += len(info['rows'])
                next_value = 0. if done else float(policy.value(torch.tensor(next_observation)).detach())
                for key,item in zip(BATCH_KEYS,[observation,action,logp,value,reward,next_value,done]):
                    batch[key].append(item)
                episode_return += reward
                episode_rows.append(info['rows'])
                journal.write(json.dumps(dict(decision=decision,episode=episode_index,path=path_index,
                    latent=action.tolist(),held=info['held_action'].tolist(),log_probability=logp,
                    reward=reward,elapsed_s=info['elapsed_s'],terminal=done,reason=info['reason']))+'\n')
                journal.flush()
                if done:
                    episodes.append(dict(episode=episode_index,path=path_index,success=env.success,
                        reason=env.reason,elapsed_s=env.ticks*.001,return_sum=episode_return,
                        decisions=env.decisions))
                    np.savez_compressed(stage/('episode_'+str(episode_index)+'.npz'),rows=np.vstack(episode_rows),
                        reference=train_paths[path_index])
                    episode_index += 1
                    curriculum_count = 16 if args.curriculum.startswith('mixed') and study_decision<5000 else len(train_paths)
                    path_index = int(rng.integers(curriculum_count))
                    env = env_factory(train_paths[path_index],fixed_d=fixed_d,max_time=5.)
                    episode_return,episode_rows = 0.,[]
                if len(batch['reward'])==512 or decision==args.steps-1:
                    statistics = ppo_update(policy,optimizer,batch,lam=1. if args.curriculum.startswith('mixed') else .98)
                    costs['ppo_updates'] += 1
                    costs['optimizer_minibatches'] += 4*math.ceil(len(batch['reward'])/128)
                    updates.append(dict(decisions=decision+1,statistics=statistics,
                        block_wall_s=time.perf_counter()-block_started,
                        episode_successes=sum(row['success'] for row in episodes),episodes=len(episodes)))
                    print(updates[-1],flush=True)
                    batch = {key:[] for key in BATCH_KEYS}
                    block_started = time.perf_counter()
                    save_torch(stage/('checkpoint_'+str(decision+1)+'.pt'),dict(policy=policy.state_dict(),
                        optimizer=optimizer.state_dict(),mode=args.mode,observation_dim=len(observation)))
                    write_json(stage/'training_progress.json',dict(costs=costs,updates=updates,episodes=episodes))
                # A replaceable two-slot snapshot avoids accumulating whole histories.
                save_torch(volatile/('resume_'+str((study_decision+1)%2)+'.pt'),dict(
                    decision=study_decision+1,policy=policy.state_dict(),optimizer=optimizer.state_dict(),
                    environment=env.state(),path_index=path_index,episode_index=episode_index,
                    episode_return=episode_return,episode_rows=episode_rows,batch=batch,
                    torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),generator_rng=rng.bit_generator.state,
                    costs=costs,episodes=episodes,updates=updates))
                write_json(volatile/'resume_pointer.json',dict(decision=study_decision+1,slot=(study_decision+1)%2,
                    phase='training',training_complete=False))
        training_wall = time.perf_counter()-training_started
        save_torch(stage/'final.pt',dict(policy=policy.state_dict(),optimizer=optimizer.state_dict(),
            observation_dim=len(env.observation()),mode=args.mode))
        final = evaluate(policy,valid_paths,stage,'final_validation',fixed_d,costs,deadline,env_factory)
        write_json(stage/'results.json',dict(initial_validation=initial,final_validation=final,
            costs=costs,training_wall_s=training_wall,wall_s=time.perf_counter()-started,
            training_seconds_per_decision=training_wall/args.steps,updates=updates,episodes=episodes,
            source_attribution=('Common pilot checkpoint initialization: '+args.init_model) if args.init_model else 'New from-scratch latent policy; no old actor or old performance inherited.',
            joint_head_bias_delta=(policy.actor[-1].bias-reference_policy.actor[-1].bias).detach().tolist()
                if args.mode!='joint' else None))
        receipt(stage,dict(status='bounded_pilot_completed',mode=args.mode,costs=costs,
            initial_successes=sum(r['success'] for r in initial),final_successes=sum(r['success'] for r in final),
            final_sha256=sha(stage/'final.pt'),wall_s=time.perf_counter()-started,
            superiority_established=False,unseen_geometry_evaluated=False))
        write_json(volatile/'resume_pointer.json',dict(decision=args.steps+(previous_decisions if recovery else 0),phase='complete',training_complete=True,
            final='final.pt',automatic_resume_forbidden=True))
        record_progress(args.stage,[stage/'receipt.json'],
            '新联合策略已完成有限单种子训练/成本阶段，结果以joint_policy_20260915_v1下对应收据为准；不把先导当成优于S2或固定D的证据。下一步读取成功率和完整时间，基于失败原因决定有界后继，并进行同权限固定D和仅D对照。论文保持指定原稿英文风格，旧数据失败保留，目标active，20–25页/逐图报告/全PDF验收未完成。')
        print(dict(status='pilot_completed',stage=args.stage,costs=costs,training_wall_s=training_wall),flush=True)
    except Exception:
        write_json(stage/'failure.json',dict(traceback=traceback.format_exc(),costs=costs,
            wall_s=time.perf_counter()-started,automatic_retry=False))
        raise
