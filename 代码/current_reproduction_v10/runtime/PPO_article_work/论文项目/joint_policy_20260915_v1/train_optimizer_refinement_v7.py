"""Bounded lower-step PPO refinement with externally stored training traces."""
import argparse
import gzip
import json
import math
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from study_io import BASE,register,receipt,write_json,record_progress,sha
from joint_policy import JointPolicy
from smooth_policy_candidate import SmoothExplorationPolicy
from learning_update_v4 import optimizers,update
from joint_runtime_v5 import environment_factory
from resume_codec_v5 import save_resume
from train_pilot import BATCH_KEYS,evaluate,save_torch


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',required=True)
    parser.add_argument('--mode',choices=['joint','fixed'],required=True)
    parser.add_argument('--sampler',choices=['independent','gsde8'],required=True)
    parser.add_argument('--initial-reuse',type=str)
    args=parser.parse_args()
    parent=BASE/'fixed3_independent_exploration_learning_v1'
    if not (parent/'receipt.json').exists():
        raise RuntimeError('Completed common fixed3 source receipt is required')
    selected=json.loads((parent/'selection.json').read_text(encoding='utf-8'))['selected']
    if selected['successes']!=12:
        raise RuntimeError('Common source must have 12 complete development paths')
    source=parent/('initial.pt' if selected['decisions']==0 else 'final.pt' if selected['tag']=='final_validation' else f"checkpoint_{selected['decisions']}.pt")
    original_validation=parent/selected['tag']
    preparation=BASE/'learning_preparation_v4'
    if args.mode=='joint' and args.sampler=='gsde8' and not args.initial_reuse:
        raise ValueError('Reuse the identical completed joint-independent initial validation')
    sources=[__file__,source,parent/'receipt.json',parent/'selection.json',
        original_validation/'summary.json',preparation/'paths.json',
        BASE/'refinement_later_analysis_v6/receipt.json',
        BASE/'ppo_update_size_diagnosis_v2/receipt.json',BASE/'fixed_distance_search_results_v1/receipt.json',
        BASE/'smooth_exploration_candidate_validation_v1/receipt.json',
        BASE/'smooth_exploration_physical_probe_v1/receipt.json',
        BASE/'temporal_exploration_fulltext_v1/smooth_exploration.pdf']
    sources += [BASE/name for name in ['learning_design_v7.md','smooth_policy_candidate.py',
        'learning_update_v4.py','study_io.py','joint_policy.py','joint_runtime.py',
        'joint_runtime_v2.py','joint_runtime_v3.py','joint_runtime_v5.py',
        'terminal_candidate_runtime_v1.py','resume_codec_v5.py','geometry_spec.py',
        'train_pilot.py','validate_actions.py','task_contract.md']]
    if args.initial_reuse:
        reuse=BASE/args.initial_reuse
        sources += [reuse/'initial.pt',reuse/'initial_validation/summary.json']
    stage=register(args.stage,sources,dict(training_decisions=360000,
        validation_runs=84 if args.mode=='joint' and not args.initial_reuse else 72,
        validation_every_decisions=60000,physics_main_ticks=7200000,
        terminal_tail_ticks_upper_bound=756000000,max_episode_ticks=25000,wall_s=18000,actor_learning_rate=3e-5,
        seed=713,mode=args.mode,sampler=args.sampler,fixed_d=3. if args.mode=='fixed' else None,
        batch=2048,minibatch=256,epochs=6),
        'Four-arm lower-step PPO refinement after saved-buffer diagnosis; paired exploratory study, not independent seeds or final superiority evidence.')
    volatile=Path('D:/PPO_article_joint_runs/rolling_states')/args.stage
    volatile.mkdir(parents=True,exist_ok=False)
    episode_directory=volatile/'episodes'
    episode_directory.mkdir(exist_ok=False)
    write_json(stage/'storage.json',dict(rolling_state_directory=str(volatile),training_episode_directory=str(episode_directory),decision_journal=str(volatile/'decisions.jsonl.gz'),format='joint_compact_v5'))
    torch.set_num_threads(1)
    torch.manual_seed(713)
    np.random.seed(713)
    rng=np.random.default_rng(713)
    costs=dict(training_actor_calls=0,training_physics_ticks=0,evaluation_actor_calls=0,
        evaluation_physics_ticks=0,ppo_updates=0,actor_minibatches=0,critic_minibatches=0)
    started=time.perf_counter()
    deadline=started+18000.
    phase='initialization'
    try:
        paths=json.loads((preparation/'paths.json').read_text(encoding='utf-8'))
        train_paths=[np.asarray(ref,float) for ref in paths['training']]
        valid_paths=[np.asarray(ref,float) for ref in paths['validation']]
        incoming=torch.load(source,map_location='cpu',weights_only=False)
        policy=(SmoothExplorationPolicy(incoming['observation_dim'],mode=args.mode,noise_interval=8)
                if args.sampler=='gsde8' else JointPolicy(incoming['observation_dim'],mode=args.mode))
        if args.sampler=='gsde8':
            policy.load_base_policy(incoming['policy'])
        else:
            policy.load_state_dict(incoming['policy'])
        with torch.no_grad():
            policy.active_mask.copy_(torch.tensor([1.,1.,1.] if args.mode=='joint' else [1.,1.,0.]))
            policy.actor[-1].weight[2].zero_()
            policy.actor[-1].bias[2]=math.atanh((3.-6.4)/5.6)
            policy.log_std.copy_(torch.log(torch.tensor([.03,.10,.20])))
        optimizer=optimizers(policy)
        for group in optimizer['actor'].param_groups:
            group['lr']=3e-5
        fixed_d=3. if args.mode=='fixed' else None
        def model_state():
            return dict(policy=policy.state_dict(),optimizer={k:v.state_dict() for k,v in optimizer.items()},
                        observation_dim=incoming['observation_dim'],mode=args.mode,sampler=args.sampler,noise_interval=8 if args.sampler=='gsde8' else 1)
        save_torch(stage/'initial.pt',model_state())
        validations=[]
        phase='initial_validation'
        if args.mode=='fixed' and fixed_d==3.:
            # Inactive D head and std do not affect deterministic fixed3 execution.
            for key,value in incoming['policy'].items():
                if key in ('log_std','actor.4.weight','actor.4.bias'):
                    continue
                assert torch.equal(value,policy.state_dict()[key]),key
            assert torch.equal(incoming['policy']['actor.4.weight'][:2],policy.actor[-1].weight[:2])
            assert torch.equal(incoming['policy']['actor.4.bias'][:2],policy.actor[-1].bias[:2])
            previous=original_validation
            result=json.loads((previous/'summary.json').read_text(encoding='utf-8'))['results']
            write_json(stage/'initial_reuse.json',dict(source=str(previous),summary_sha256=sha(previous/'summary.json'),
                exact_deterministic_execution_equivalence=True,new_evaluation_runs=0))
        elif args.initial_reuse:
            reused=torch.load(reuse/'initial.pt',map_location='cpu',weights_only=False)
            for key,value in reused['policy'].items():
                assert torch.equal(value,policy.state_dict()[key]),key
            result=json.loads((reuse/'initial_validation/summary.json').read_text(encoding='utf-8'))['results']
            write_json(stage/'initial_reuse.json',dict(source=str(reuse/'initial_validation'),
                summary_sha256=sha(reuse/'initial_validation/summary.json'),
                exact_deterministic_execution_equivalence=True,new_evaluation_runs=0))
        else:
            result=evaluate(policy,valid_paths,stage,phase,fixed_d,costs,deadline,environment_factory)
        validations.append(dict(decisions=0,tag=phase,results=result))
        write_json(stage/'validations.json',dict(validations=validations,costs=costs))
        batch={key:[] for key in BATCH_KEYS}
        episode_index=0
        path_index=int(rng.integers(len(train_paths)))
        env=environment_factory(train_paths[path_index],fixed_d=fixed_d)
        episode_return=0.
        episode_rows=[]
        episodes=[]
        updates=[]
        phase='training'
        def checkpoint(decision,pending_validation=False):
            save_resume(volatile/f'resume_{decision%2}.pt',dict(model_state(),decision=decision,
                environment=env.state(),path_index=path_index,episode_index=episode_index,
                episode_return=episode_return,episode_rows=episode_rows,batch=batch,
                torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),generator_rng=rng.bit_generator.state,
                costs=costs,episodes=episodes,updates=updates,validations=validations,
                phase='pending_validation' if pending_validation else 'training'))
            write_json(volatile/'resume_pointer.json',dict(decision=decision,slot=decision%2,
                phase='pending_validation' if pending_validation else 'training',training_complete=False))
        with gzip.open(volatile/'decisions.jsonl.gz','wt',encoding='utf-8') as journal:
            for decision in range(1,360001):
                if time.perf_counter()>deadline:
                    raise TimeoutError('Registered wall-clock limit')
                observation=env.observation()
                action,logp,value=policy.act(observation)
                costs['training_actor_calls']+=1
                next_observation,reward,done,info=env.step(action)
                costs['training_physics_ticks']+=len(info['rows'])
                next_value=0. if done else float(policy.value(torch.tensor(next_observation)).detach())
                for key,item in zip(BATCH_KEYS,[observation,action,logp,value,reward,next_value,done]):
                    batch[key].append(item)
                episode_return+=reward
                episode_rows.append(info['rows'])
                journal.write(json.dumps(dict(decision=decision,episode=episode_index,path=path_index,
                    latent=action.tolist(),held=info['held_action'].tolist(),log_probability=logp,
                    reward=reward,elapsed_s=info['elapsed_s'],terminal=done,reason=info['reason'],
                    failure_remaining_penalty_s=info['failure_remaining_penalty_s']))+'\n')
                journal.flush()
                if done:
                    episodes.append(dict(episode=episode_index,path=path_index,success=env.success,
                        reason=env.reason,elapsed_s=env.ticks*.001,return_sum=episode_return,decisions=env.decisions))
                    np.savez_compressed(episode_directory/f'episode_{episode_index}.npz',reference=train_paths[path_index],rows=np.vstack(episode_rows))
                    if args.sampler=='gsde8':
                        policy.exploration_age.fill_(policy.noise_interval)
                    episode_index+=1
                    path_index=int(rng.integers(len(train_paths)))
                    env=environment_factory(train_paths[path_index],fixed_d=fixed_d)
                    episode_return,episode_rows=0.,[]
                evaluate_now=decision%60000==0
                if len(batch['reward'])==2048 or evaluate_now:
                    statistics=update(policy,optimizer,batch)
                    if args.sampler=='gsde8':
                        policy.exploration_age.fill_(policy.noise_interval)
                    costs['ppo_updates']+=1
                    for key in ['actor_minibatches','critic_minibatches']:
                        costs[key]+=statistics[key]
                    updates.append(dict(decisions=decision,statistics=statistics,
                        episode_successes=sum(r['success'] for r in episodes),episodes=len(episodes)))
                    batch={key:[] for key in BATCH_KEYS}
                    save_torch(stage/f'checkpoint_{decision}.pt',model_state())
                    write_json(stage/'training_progress.json',dict(costs=costs,updates=updates,episodes=episodes))
                    print(updates[-1],flush=True)
                checkpoint(decision,evaluate_now)
                if evaluate_now:
                    phase='final_validation' if decision==360000 else f'validation_{decision}'
                    result=evaluate(policy,valid_paths,stage,phase,fixed_d,costs,deadline,environment_factory)
                    validations.append(dict(decisions=decision,tag=phase,results=result))
                    write_json(stage/'validations.json',dict(validations=validations,costs=costs))
                    phase='training'
                    checkpoint(decision)
        save_torch(stage/'final.pt',model_state())
        choices=[]
        for v in validations:
            n=sum(r['success'] for r in v['results'])
            total=sum(r['completion_time_s'] for r in v['results']) if n==12 else None
            choices.append(dict(decisions=v['decisions'],tag=v['tag'],successes=n,complete_time_sum_s=total))
        best=max(choices,key=lambda r:(r['successes'],-r['complete_time_sum_s'] if r['complete_time_sum_s'] is not None else float('-inf')))
        write_json(stage/'selection.json',dict(candidates=choices,selected=best,development_selection=True))
        receipt(stage,dict(status='lower_step_paired_refinement_completed',mode=args.mode,sampler=args.sampler,costs=costs,
            selected=best,final_sha256=sha(stage/'final.pt'),wall_s=time.perf_counter()-started,
            superiority_established=False,goal_complete=False))
        write_json(volatile/'resume_pointer.json',dict(decision=360000,phase='complete',training_complete=True,
            automatic_resume_forbidden=True))
        record_progress(args.stage,[stage/'receipt.json'],
            'v7较小actor步长配对细化一组完成360000新决策；按四组全部检查点判定，旧v6、60k探索及七值固定搜索全部保留。无稳定优势预断，仍需充分训练/多种子、长路径、匹配消融、封存测试及20–25页全PDF验收。')
    except Exception:
        write_json(stage/'failure.json',dict(traceback=traceback.format_exc(),costs=costs,phase=phase,
            wall_s=time.perf_counter()-started,automatic_retry=False))
        raise


