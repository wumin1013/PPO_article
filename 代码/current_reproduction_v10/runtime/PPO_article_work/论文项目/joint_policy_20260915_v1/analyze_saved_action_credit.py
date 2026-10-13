"""Read frozen sampled batches; describe action credit without optimization."""
import json
from pathlib import Path
import numpy as np
import torch
from joint_policy import JointPolicy, advantages
from smooth_policy_candidate import SmoothExplorationPolicy
from resume_codec_v5 import load_resume
from study_io import BASE, register, receipt, write_json, sha


GROUPS = ['joint_independent_exploration_learning_v1', 'joint_gsde_exploration_learning_v1',
          'fixed3_independent_exploration_learning_v1', 'fixed3_gsde_exploration_learning_v1']


def quantiles(values):
    return np.quantile(values, [.1, .5, .9]).tolist() if len(values) else None


def main():
    sources = [Path(__file__), BASE/'saved_action_credit_plan_v1.md', BASE/'joint_policy.py',
               BASE/'smooth_policy_candidate.py', BASE/'resume_codec_v5.py',
               BASE/'ppo_update_size_diagnosis_v2/results.json']
    paths = {}
    bindings = []
    for name in GROUPS:
        sources += [BASE/name/'receipt.json']
        root = Path('D:/PPO_article_joint_runs/rolling_states')/name
        pointer = json.loads((root/'resume_pointer.json').read_text(encoding='utf-8'))
        assert pointer['training_complete'] and pointer['decision'] == 60000
        path = root/'resume_1.pt'
        paths[name] = path
        bindings.append(dict(stage=name, path=str(path), sha256=sha(path),
                             pointer=pointer, pointer_sha256=sha(root/'resume_pointer.json')))
    stage = register('saved_action_credit_v1', sources,
        dict(buffer_loads=4, model_constructions=4, actor_batch_forward_calls=4,
             critic_forward_calls=0, optimization_steps=0, physics_ticks=0),
        'Describe sampled action credit and critic residuals in four completed frozen batches.')
    write_json(stage/'external_sources.json', bindings)
    torch.set_num_threads(1)
    records = []
    for name in GROUPS:
        state = load_resume(paths[name])
        assert state['decision'] == 59999
        batch = state['batch']
        obs = np.asarray(batch['observation'], dtype=np.float32)
        latent = np.asarray(batch['latent'], dtype=np.float32)
        mode, sampler = state['mode'], state['sampler']
        policy = (SmoothExplorationPolicy(state['observation_dim'], mode=mode, noise_interval=8)
                  if sampler == 'gsde8' else JointPolicy(state['observation_dim'], mode=mode))
        policy.load_state_dict(state['policy'])
        with torch.no_grad():
            distribution = policy.distribution(torch.from_numpy(obs))
            mean = distribution.mean.numpy()
            std = distribution.stddev.numpy()
            logp = (distribution.log_prob(torch.from_numpy(latent))*policy.active_mask).sum(-1).numpy()
        error = float(np.max(np.abs(logp-np.asarray(batch['log_probability']))))
        assert error < 1e-3
        active = policy.active_mask.numpy()
        raw_adv, _ = advantages(batch['reward'], batch['value'], batch['next_value'], batch['terminal'])
        scaled_adv = (raw_adv-raw_adv.mean())/(raw_adv.std()+1e-8)
        noise = (latent-mean)/std*active
        signal = scaled_adv[:, None]*noise
        directions = obs[:, 42:62].reshape(-1, 10, 2)
        valid = obs[:, 62:72] > .5
        angles = np.abs(np.arctan2(directions[:, 1:8, 1], directions[:, 1:8, 0]))
        forward_valid = valid[:, 1:8]
        straight = np.all((angles <= np.deg2rad(1.)) | ~forward_valid, axis=1) & forward_valid.any(axis=1)
        feed_mean = 50*(1+np.tanh(mean[:, 1]))
        feed_sample = 50*(1+np.tanh(latent[:, 1]))
        speed = 100*np.linalg.norm(obs[:, 2:4], axis=1)
        terminal = np.asarray(batch['terminal'], dtype=bool)
        rewards = np.asarray(batch['reward'], dtype=float)
        values = np.asarray(batch['value'], dtype=float)
        # Actual remaining returns exist only before a terminal observed in this batch.
        actual_returns = np.full(len(rewards), np.nan)
        ended_suffix = 0.
        have_terminal = False
        for i in range(len(rewards)-1, -1, -1):
            if terminal[i]:
                ended_suffix, have_terminal = 0., True
            if have_terminal:
                ended_suffix += rewards[i]
                actual_returns[i] = ended_suffix
        observed = np.isfinite(actual_returns)
        partitions = []
        for label, mask in [('all', np.ones(len(obs), dtype=bool)), ('local_straight', straight),
                            ('turn_or_terminal', ~straight)]:
            completed = mask & observed
            partitions.append(dict(group=label, samples=int(mask.sum()),
                raw_advantage_quantiles=quantiles(raw_adv[mask]),
                mean_feed_intent_quantiles_mm_s=quantiles(feed_mean[mask]),
                sampled_feed_intent_quantiles_mm_s=quantiles(feed_sample[mask]),
                physical_speed_quantiles_mm_s=quantiles(speed[mask]),
                normalized_noise_credit_mean=signal[mask].mean(0).tolist() if mask.any() else None,
                completed_suffix_rows=int(completed.sum()),
                saved_value_minus_actual_return_quantiles=quantiles((values-actual_returns)[completed])))
        records.append(dict(stage=name, sampler=sampler, active_mask=active.tolist(),
            batch_rows=len(obs), observed_terminal_rows=int(terminal.sum()),
            censored_tail_rows=int((~observed).sum()), saved_log_probability_max_difference=error,
            raw_advantage_mean=float(raw_adv.mean()), raw_advantage_std=float(raw_adv.std()),
            partitions=partitions))
        assert sha(paths[name]) == next(b['sha256'] for b in bindings if b['stage'] == name)
    write_json(stage/'results.json', dict(groups=records, independent_samples_assumed=False,
        policy_gradient_unbiasedness_claimed=False, physics_ticks=0, training_decisions=0))
    lines = ['# 保存批中的动作学习信号', '',
        '| 方法 | 状态组 | 样本 | 均值速度意图中位数 | 实际速度中位数 | 速度噪声信用均值 | 前瞻噪声信用均值 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in records:
        for p in r['partitions']:
            if p['samples']:
                lines.append(f"| {r['stage']} | {p['group']} | {p['samples']} | {p['mean_feed_intent_quantiles_mm_s'][1]:.3f} | {p['physical_speed_quantiles_mm_s'][1]:.3f} | {p['normalized_noise_credit_mean'][1]:.5f} | {p['normalized_noise_credit_mean'][2]:.5f} |")
    lines += ['', '正负号分别表示在这一保存样本和原GAE估计下，较大或较小潜变量噪声与标准化优势的乘积均值；它不是净参数更新或因果效果。gSDE8仅描述边际伪目标，不能视作无偏条件策略梯度。局部直向仅指15mm有效未来采样方向，包含加减速及终端邻近状态。', '',
        'value减实际后缀回报只在批内能看到终止的位置计算；批尾未结束部分记为删失，不以bootstrap伪充真实回报。四份来源属于已完成探索阶段，未使用v7正在写入的状态，也未修改其设置。']
    (stage/'动作学习信号.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    receipt(stage, dict(status='completed_saved_action_credit_analysis', buffer_loads=4,
        model_constructions=4, actor_batch_forward_calls=4, critic_forward_calls=0,
        optimization_steps=0, physics_ticks=0, training_decisions=0))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
