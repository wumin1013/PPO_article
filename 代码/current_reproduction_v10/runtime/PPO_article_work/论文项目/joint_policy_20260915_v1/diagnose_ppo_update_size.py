"""Compare step sizes on four frozen real PPO buffers; no environment is run."""
import copy
import json
from pathlib import Path
import numpy as np
import torch
from study_io import BASE, register, receipt, write_json, sha
from resume_codec_v5 import load_resume
from joint_policy import JointPolicy, advantages
from smooth_policy_candidate import SmoothExplorationPolicy
from learning_update_v4 import optimizers, update


def main():
    groups = ['joint_independent_exploration_learning_v1', 'joint_gsde_exploration_learning_v1',
              'fixed3_independent_exploration_learning_v1', 'fixed3_gsde_exploration_learning_v1']
    locations, pointers = {}, {}
    sources = [Path(__file__), BASE/'optimizer_diagnosis_plan_v1.md', BASE/'learning_update_v4.py',
               BASE/'joint_policy.py', BASE/'smooth_policy_candidate.py', BASE/'resume_codec_v5.py']
    for name in groups:
        folder = BASE/name
        if not (folder/'receipt.json').exists():
            raise RuntimeError('Original four-arm learning is still incomplete')
        root = Path('D:/PPO_article_joint_runs/rolling_states')/name
        pointer = json.loads((root/'resume_pointer.json').read_text(encoding='utf-8'))
        if pointer.get('phase') != 'complete' or pointer.get('decision') != 60000:
            raise RuntimeError('Only completed immutable source states may be used')
        locations[name] = root/'resume_1.pt'
        pointers[name] = dict(path=str(root/'resume_pointer.json'), sha256=sha(root/'resume_pointer.json'), value=pointer)
        sources += [folder/'receipt.json', folder/'training_progress.json']
    sources += [BASE/'ppo_update_size_diagnosis_v1/failure.json']
    stage = register('ppo_update_size_diagnosis_v2', sources,
        dict(real_buffer_loads=4, offline_ppo_updates=12, epochs_per_update=6,
             minibatch=256, training_decisions=0, physics_ticks=0),
        'Test three actor step sizes on each frozen penultimate PPO buffer, without restoring or rerunning physical training.')
    write_json(stage/'external_source_pointers.json', pointers)
    torch.set_num_threads(1)
    results, bindings = [], []
    total_actor = total_critic = 0
    for name in groups:
        source = locations[name]
        source_sha = sha(source)
        state = load_resume(source)
        assert state['decision'] == 59999
        batch = state['batch']
        n = len(batch['reward'])
        assert 1024 <= n <= 2048
        bindings.append(dict(stage=name, path=str(source), sha256=source_sha,
                             decision=state['decision'], batch_rows=n))
        mode, sampler = state['mode'], state['sampler']
        obs = torch.as_tensor(np.asarray(batch['observation']), dtype=torch.float32)
        latent = torch.as_tensor(np.asarray(batch['latent']), dtype=torch.float32)
        oldlog = torch.as_tensor(batch['log_probability'], dtype=torch.float32)
        advantage, returns = advantages(batch['reward'], batch['value'], batch['next_value'], batch['terminal'])
        advantage = torch.as_tensor(advantage)
        advantage = (advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-8)
        returns = torch.as_tensor(returns)
        for learning_rate in [3e-4, 1e-4, 3e-5]:
            policy = (SmoothExplorationPolicy(state['observation_dim'], mode=mode, noise_interval=8)
                      if sampler == 'gsde8' else JointPolicy(state['observation_dim'], mode=mode))
            policy.load_state_dict(state['policy'])
            optimizer = optimizers(policy)
            for key, value in state['optimizer'].items():
                optimizer[key].load_state_dict(copy.deepcopy(value))
            for group in optimizer['actor'].param_groups:
                group['lr'] = learning_rate
            with torch.no_grad():
                old = policy.distribution(obs)
                old_mean, old_std = old.mean.clone(), old.stddev.clone()
                actual_log, _, oldvalue = policy.evaluate(obs, latent)
                log_error = float(torch.max(torch.abs(actual_log-oldlog)))
                assert log_error < 1e-3, 'Saved observations and policy likelihood do not match'
            torch.set_rng_state(state['torch_rng'].clone())
            statistics = update(policy, optimizer, batch)
            total_actor += statistics['actor_minibatches']
            total_critic += statistics['critic_minibatches']
            with torch.no_grad():
                new = policy.distribution(obs)
                kl = torch.log(new.stddev/old_std)+(old_std.square()+(old_mean-new.mean).square())/(2*new.stddev.square())-.5
                kl *= policy.active_mask
                logp, _, value = policy.evaluate(obs, latent)
                ratio = torch.exp(logp-oldlog)
                objective = torch.minimum(ratio*advantage, ratio.clamp(.8, 1.2)*advantage).mean()
                result = dict(stage=name, sampler=sampler, mode=mode, actor_learning_rate=learning_rate,
                    batch_rows=n, statistics=statistics, saved_logprob_max_difference=log_error,
                    analytic_marginal_kl_by_action=kl.mean(0).tolist(),
                    analytic_marginal_total_kl=float(kl.sum(-1).mean()),
                    sample_ratio_clip_fraction=float(((ratio<.8)|(ratio>1.2)).float().mean()),
                    clipped_surrogate_on_saved_batch=float(objective),
                    critic_mse_before=float((oldvalue-returns).square().mean()),
                    critic_mse_after=float((value-returns).square().mean()))
            results.append(result)
            print(name, learning_rate, statistics['actor_minibatches'], result['analytic_marginal_total_kl'], flush=True)
        assert sha(source) == source_sha
    write_json(stage/'results.json', dict(results=results, source_buffers=bindings,
        physical_performance_evaluated=False, production_model_modified=False,
        counts=dict(real_buffer_loads=4, model_constructions=12, offline_ppo_updates=12,
            actor_minibatch_steps=total_actor, critic_minibatch_steps=total_critic,
            explicit_full_batch_distribution_calls=24, explicit_full_batch_evaluate_calls=24,
            update_internal_evaluate_calls=total_critic, physics_ticks=0, training_decisions=0)))
    lines = ['# PPO更新幅度诊断', '',
        '| 组 | Actor学习率 | 实际actor小批数 | 解析边际KL | 原批裁剪代理目标 |',
        '|---|---:|---:|---:|---:|']
    for r in results:
        lines.append(f"| {r['stage']} | {r['actor_learning_rate']:g} | {r['statistics']['actor_minibatches']} | {r['analytic_marginal_total_kl']:.5f} | {r['clipped_surrogate_on_saved_batch']:.5f} |")
    lines += ['',
        '每组使用自身59,999步的同一保存批、原Adam状态和相同随机小批顺序。没有采集新动作或恢复环境。表中代理目标来自训练样本，不能作为新轨迹性能或泛化证据；gSDE数字是边际高斯KL，不是给定完整历史的联合KL。较小学习率的作用应结合四组观察，不能只按联合组挑选。']
    (stage/'更新幅度诊断.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    receipt(stage, dict(status='completed_frozen_buffer_update_diagnosis',
        real_buffer_loads=4, offline_ppo_updates=12, physics_ticks=0, training_decisions=0,
        actor_minibatch_steps=total_actor, critic_minibatch_steps=total_critic))


if __name__ == '__main__':
    main()
