"""Paired lambda-one pilot; preserves the completed base training loop and task."""
import argparse
from datetime import datetime, timezone
from functools import partial
import gzip
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from study_io import BASE, register, receipt, write_json, sha, record_progress
from spatial_learning_candidate import environment_factory, fresh_policy
from joint_policy import advantages
from learning_update_v4 import optimizers, update
from resume_codec_v5 import save_resume
from train_pilot import BATCH_KEYS, evaluate, save_torch

V7=('joint_independent_refinement_v7','joint_gsde_refinement_v7',
    'fixed3_independent_refinement_v7','fixed3_gsde_refinement_v7')


def main():
    parser=argparse.ArgumentParser()
    parser.set_defaults(arm='base')
    parser.add_argument('--mode',choices=('joint','fixed'),required=True)
    args=parser.parse_args()
    lam=1.
    extra_sources=[BASE/'credit_horizon_design_v1.md',
        BASE/'minimal_first_batch_credit_v1/receipt.json',
        BASE/'minimal_late_credit_audit_v1/receipt.json',
        BASE/'minimal_late_credit_audit_v1/results.json',
        BASE/f'minimal_base_{args.mode}_pilot_v1/receipt.json']
    prerequisites=[BASE/name/'receipt.json' for name in V7]+[
        BASE/'v7_checkpoint_359999_analysis_v1/receipt.json',
        BASE/'v7_complete_curve_audit_v1/receipt.json',BASE/'spatial_learning_contract_v1/receipt.json']
    if not all(p.exists() for p in prerequisites):
        raise RuntimeError('Complete v7, audit its actual final checkpoint and full curve before new single-factor training')
    representation='spatial22' if args.arm=='state' else 'original72'
    reward_mode='unshaped' if args.arm=='reward' else 'original'
    fixed_d=3. if args.mode=='fixed' else None
    factory=partial(environment_factory,representation=representation,reward_mode=reward_mode)
    dim=22 if args.arm=='state' else 72
    prepared=BASE/'learning_preparation_v4/paths.json'
    paths=json.loads(prepared.read_text(encoding='utf-8'))
    train_paths=[np.asarray(ref) for ref in paths['training']]
    valid_paths=[np.asarray(ref) for ref in paths['validation']]
    source_names=('minimal_factorial_design_v1.md','spatial_learning_candidate.py',
        'compact_observation_candidate.py','event_observation_candidate.py',
        'joint_runtime.py','joint_runtime_v2.py','joint_runtime_v3.py','joint_runtime_v5.py',
        'terminal_candidate_runtime_v1.py','geometry_spec.py','joint_policy.py',
        'learning_update_v4.py','train_pilot.py','resume_codec_v5.py','study_io.py','validate_actions.py')
    sources=[Path(__file__),prepared,*prerequisites,*extra_sources]+[BASE/name for name in source_names]
    reuse=BASE/f'minimal_base_{args.mode}_pilot_v1'
    if not (reuse/'receipt.json').exists():
        raise RuntimeError('Matched base must be complete')
    sources += [reuse/'initial.pt',reuse/'initial_validation/summary.json',reuse/'initial_completed.json']
    if not all(p.exists() for p in sources):
        raise RuntimeError('Completed same-mode base initial validation is required for exact within-mode reuse')
    stage=register(f'credit_horizon_{args.mode}_pilot_v1',sources,
        dict(training_decisions=60000,wall_s=7200,seed=917,arm=args.arm,mode=args.mode,gae_lambda=lam,
            representation=representation,observation_dim=dim,reward_mode=reward_mode,fixed_d=fixed_d,
            validation_runs=24,validation_every_decisions=30000,
            physics_main_ticks=1200000,terminal_tail_ticks_upper_bound=126000000,
            batch=2048,minibatch=256,epochs=6,actor_lr=3e-4,critic_lr=1e-3),
        'Single-factor lambda-one credit comparison against completed matched base, with equal fixed-D opportunity; not independent replications or final superiority evidence.')
    volatile=Path('D:/PPO_article_joint_runs/rolling_states')/stage.name
    volatile.mkdir(parents=True,exist_ok=False)
    episode_directory=volatile/'episodes'
    episode_directory.mkdir()
    write_json(stage/'storage.json',dict(rolling_state_directory=str(volatile),
        training_episode_directory=str(episode_directory),decision_journal=str(volatile/'decisions.jsonl.gz'),
        format='joint_compact_v5'))
    torch.set_num_threads(1)
    torch.manual_seed(917)
    np.random.seed(917)
    rng=np.random.default_rng(917)
    policy=fresh_policy(dim,mode=args.mode,seed=917)
    optimizer=optimizers(policy)
    costs=dict(training_actor_calls=0,training_physics_ticks=0,evaluation_actor_calls=0,
        evaluation_physics_ticks=0,ppo_updates=0,actor_minibatches=0,critic_minibatches=0)
    component_seconds=dict(observation=0.,policy=0.,physics=0.,optimization=0.,checkpoint=0.)
    started=time.perf_counter()
    deadline=started+7200.
    phase='initial_validation'
    write_json(stage/'started.json',dict(pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat(),
        arm=args.arm,mode=args.mode,representation=representation,reward_mode=reward_mode,
        observation_dim=dim,model_constructions=1,training_decisions=0,phase=phase))
    def model_state():
        return dict(policy=policy.state_dict(),optimizer={k:v.state_dict() for k,v in optimizer.items()},
            observation_dim=dim,mode=args.mode,sampler='independent',
            representation=representation,reward_mode=reward_mode,seed=917,gae_lambda=lam)
    try:
        save_torch(stage/'initial.pt',model_state())
        if (reuse/'initial.pt').exists():
            other=torch.load(reuse/'initial.pt',map_location='cpu',weights_only=False)['policy']
            current=policy.state_dict()
            assert torch.count_nonzero(other['actor.4.weight'])==0
            assert torch.count_nonzero(current['actor.4.weight'])==0
            assert torch.equal(other['actor.4.bias'],current['actor.4.bias'])
            assert torch.equal(other['active_mask'],current['active_mask'])
            assert torch.equal(other['log_std'],current['log_std'])
            assert other.keys()==current.keys()
            assert all(torch.equal(other[k],current[k]) for k in current)
            result=json.loads((reuse/'initial_validation/summary.json').read_text(encoding='utf-8'))['results']
            write_json(stage/'initial_reuse.json',dict(source=str(reuse/'initial_validation'),
                source_summary_sha256=sha(reuse/'initial_validation/summary.json'),
                constant_finite_input_policy_equivalence=True,within_same_execution_mode=True,
                new_evaluation_runs=0))
        write_json(stage/'initial_completed.json',dict(results_source=str(reuse),
            successes=sum(r['success'] for r in result),parameters=sum(p.numel() for p in policy.parameters()),
            observation_dim=dim,common_initial_value_zero=True))
        validations=[dict(decisions=0,tag=phase,results=result)]
        write_json(stage/'validations.json',dict(validations=validations,costs=costs))
        batch={key:[] for key in BATCH_KEYS}
        episode_index=0
        path_index=int(rng.integers(len(train_paths)))
        env=factory(train_paths[path_index],fixed_d=fixed_d)
        episode_return=0.
        episode_rows=[]
        episodes=[]
        updates=[]
        phase='training'
        def checkpoint(decision,pending_validation=False):
            tick=time.perf_counter()
            save_resume(volatile/f'resume_{decision%2}.pt',dict(model_state(),decision=decision,
                environment=env.state(),path_index=path_index,episode_index=episode_index,
                episode_return=episode_return,episode_rows=episode_rows,batch=batch,
                torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),generator_rng=rng.bit_generator.state,
                costs=costs,component_seconds=component_seconds,episodes=episodes,updates=updates,
                validations=validations,elapsed_active_s=time.perf_counter()-started,
                phase='pending_validation' if pending_validation else 'training'))
            write_json(volatile/'resume_pointer.json',dict(decision=decision,slot=decision%2,
                phase='pending_validation' if pending_validation else 'training',training_complete=False))
            component_seconds['checkpoint']+=time.perf_counter()-tick
        checkpoint(0)
        with gzip.open(volatile/'decisions.jsonl.gz','wt',encoding='utf-8') as journal:
            for decision in range(1,60001):
                if time.perf_counter()>deadline:
                    raise TimeoutError('Registered wall-clock limit')
                tick=time.perf_counter()
                observation=env.observation()
                component_seconds['observation']+=time.perf_counter()-tick
                tick=time.perf_counter()
                action,logp,value=policy.act(observation)
                component_seconds['policy']+=time.perf_counter()-tick
                costs['training_actor_calls']+=1
                tick=time.perf_counter()
                next_observation,reward,done,info=env.step(action)
                component_seconds['physics']+=time.perf_counter()-tick
                costs['training_physics_ticks']+=len(info['rows'])
                tick=time.perf_counter()
                with torch.no_grad():
                    next_value=0. if done else float(policy.value(torch.tensor(next_observation)))
                component_seconds['policy']+=time.perf_counter()-tick
                for key,item in zip(BATCH_KEYS,[observation,action,logp,value,reward,next_value,done]):
                    batch[key].append(item)
                episode_return+=reward
                episode_rows.append(info['rows'])
                journal.write(json.dumps(dict(decision=decision,episode=episode_index,path=path_index,
                    latent=action.tolist(),held=info['held_action'].tolist(),log_probability=logp,
                    value=value,next_value=next_value,reward=reward,elapsed_s=info['elapsed_s'],
                    terminal=done,reason=info['reason'],
                    failure_remaining_penalty_s=info['failure_remaining_penalty_s']))+'\n')
                journal.flush()
                if done:
                    episodes.append(dict(episode=episode_index,path=path_index,success=env.success,
                        reason=env.reason,elapsed_s=env.ticks*.001,return_sum=episode_return,decisions=env.decisions))
                    np.savez_compressed(episode_directory/f'episode_{episode_index}.npz',
                        reference=train_paths[path_index],rows=np.vstack(episode_rows))
                    episode_index+=1
                    path_index=int(rng.integers(len(train_paths)))
                    env=factory(train_paths[path_index],fixed_d=fixed_d)
                    episode_return,episode_rows=0.,[]
                evaluate_now=decision%30000==0
                if len(batch['reward'])==2048 or evaluate_now:
                    tick=time.perf_counter()
                    advantage,_=advantages(batch['reward'],batch['value'],batch['next_value'],batch['terminal'],lam=lam)
                    statistics=update(policy,optimizer,batch,lam=lam)
                    statistics.update(gae_mean=float(advantage.mean()),gae_std=float(advantage.std()),
                        terminal_rows=int(sum(batch['terminal'])))
                    component_seconds['optimization']+=time.perf_counter()-tick
                    costs['ppo_updates']+=1
                    for key in ('actor_minibatches','critic_minibatches'):
                        costs[key]+=statistics[key]
                    updates.append(dict(decisions=decision,statistics=statistics,
                        episode_successes=sum(r['success'] for r in episodes),episodes=len(episodes)))
                    batch={key:[] for key in BATCH_KEYS}
                    save_torch(stage/f'checkpoint_{decision}.pt',model_state())
                    write_json(stage/'training_progress.json',dict(costs=costs,updates=updates,episodes=episodes,
                        component_seconds=component_seconds,active_wall_s=time.perf_counter()-started))
                    print(updates[-1],flush=True)
                checkpoint(decision,evaluate_now)
                if evaluate_now:
                    phase='final_validation' if decision==60000 else f'validation_{decision}'
                    result=evaluate(policy,valid_paths,stage,phase,fixed_d,costs,deadline,factory)
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
        best=max(choices,key=lambda r:(r['successes'],
            -r['complete_time_sum_s'] if r['complete_time_sum_s'] is not None else -r['decisions']))
        write_json(stage/'selection.json',dict(candidates=choices,selected=best,development_selection=True,
            incomplete_tie_rule='earliest_checkpoint_no_failed_time_ranking'))
        receipt(stage,dict(status='credit_horizon_pilot_completed',arm=args.arm,mode=args.mode,
            costs=costs,component_seconds=component_seconds,selected=best,final_sha256=sha(stage/'final.pt'),
            wall_s=time.perf_counter()-started,representation=representation,reward_mode=reward_mode,
            parameters=sum(p.numel() for p in policy.parameters()),superiority_established=False,goal_complete=False))
        write_json(volatile/'resume_pointer.json',dict(decision=60000,phase='complete',training_complete=True,
            automatic_resume_forbidden=True))
        # Keep the training completion independent of any tracker lock failure.
        try:
            record_progress(stage.name,[stage/'receipt.json'],
                'λ=1信用先导一组已完成60k，须与已完成base的λ=.98同模式比较并核另一组；保留六组简化负结果与强v7，不自动组合或扩预算。合理选距、独立种子、封存测试与20–25页全文仍未完成。')
        except Exception:
            write_json(stage/'progress_recording_failure.json',dict(traceback=traceback.format_exc(),
                training_completed=True,training_must_not_repeat=True))
    except Exception:
        write_json(stage/'failure.json',dict(traceback=traceback.format_exc(),costs=costs,phase=phase,
            wall_s=time.perf_counter()-started,automatic_retry=False))
        raise


if __name__=='__main__':
    main()
