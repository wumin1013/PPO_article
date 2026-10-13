"""Measure critic dropout noise on one fixed, reproducible training rollout.

This is an implementation diagnostic, not a held-out performance experiment.
Both modes use the same trained critic, states, rewards and GAE definition.
"""
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

import preview_research_v2 as study
from src.utils.rl_utils import compute_advantage


def main():
    torch.set_num_threads(1)
    torch.manual_seed(809)
    run = study.OUT / 'finetune/fixed_seed11'
    cfg = yaml.safe_load((run / 'config.yaml').read_text(encoding='utf-8'))
    env, obs = study.make_env(cfg, cfg['revision']['training_path_ids'][0])
    agent = study.make_agent(cfg, env)
    checkpoint = torch.load(run / 'checkpoint.pt', map_location='cpu', weights_only=False)
    agent.actor.load_state_dict(checkpoint['actor'])
    agent.critic.load_state_dict(checkpoint['critic'])
    states, next_states, rewards, dones = [], [], [], []
    for _ in range(512):
        with torch.no_grad():
            action = agent.actor(torch.tensor(obs[None], dtype=torch.float32))[0][0].numpy()
        nxt, reward, done, _ = env.step(np.clip(action[:env.action_space_dim], env.action_space.low, env.action_space.high))
        states.append(obs); next_states.append(nxt); rewards.append(reward); dones.append(done)
        obs = nxt
        if done:
            break
    state_tensor = torch.tensor(np.asarray(states), dtype=torch.float32)
    next_tensor = torch.tensor(np.asarray(next_states), dtype=torch.float32)
    reward_tensor = torch.tensor(rewards, dtype=torch.float32).reshape(-1, 1)
    done_tensor = torch.tensor(dones, dtype=torch.float32).reshape(-1, 1)
    gamma = float(np.exp(-.001 / 10.))
    repeated_values, repeated_td, repeated_advantages = [], [], []
    with torch.no_grad():
        agent.critic.train()
        for _ in range(64):
            next_value = agent.critic(next_tensor)
            value = agent.critic(state_tensor)
            td = reward_tensor + gamma * next_value * (1 - done_tensor) - value
            advantage = compute_advantage(gamma, 1., td, done_tensor)
            repeated_values.append(value.numpy().ravel())
            repeated_td.append(td.numpy().ravel())
            repeated_advantages.append(advantage.numpy().ravel())
        agent.critic.eval()
        deterministic_value = agent.critic(state_tensor)
        deterministic_td = reward_tensor + gamma * agent.critic(next_tensor) * (1 - done_tensor) - deterministic_value
        deterministic_advantage = compute_advantage(gamma, 1., deterministic_td, done_tensor).numpy().ravel()
        eval_repeat_max = float(torch.max(torch.abs(agent.critic(state_tensor) - deterministic_value)))
    repeated_values, repeated_td, repeated_advantages = map(np.asarray, (repeated_values, repeated_td, repeated_advantages))
    centered = repeated_advantages - repeated_advantages.mean(axis=1, keepdims=True)
    centered_eval = deterministic_advantage - deterministic_advantage.mean()
    out = study.ROOT / '论文项目/revision_20260908/value_noise_diagnostics'
    out.mkdir(exist_ok=True)
    np.savez_compressed(out / 'fixed_rollout_value_noise.npz', states=state_tensor.numpy(),
        next_states=next_tensor.numpy(), rewards=np.asarray(rewards), dones=np.asarray(dones),
        repeated_values=repeated_values, repeated_td=repeated_td, repeated_advantages=repeated_advantages,
        deterministic_value=deterministic_value.numpy(), deterministic_td=deterministic_td.numpy(),
        deterministic_advantage=deterministic_advantage)
    record = dict(command='python ' + ' '.join(sys.argv), path_id=cfg['revision']['training_path_ids'][0],
        checkpoint=str(run / 'checkpoint.pt'), checkpoint_sha256=study.backend.sha(run / 'checkpoint.pt'),
        config_sha256=study.backend.sha(run / 'config.yaml'), source_state=study.source_state(),
        script_sha256=study.backend.sha(__file__), raw_sha256=study.backend.sha(out / 'fixed_rollout_value_noise.npz'),
        gamma=gamma, gae_lambda=1., steps=len(states), dropout_repeats=64, rng_seed=809,
        mean_abs_tick_reward=float(np.mean(np.abs(rewards))),
        critic_dropout_value_std_mean=float(repeated_values.std(axis=0, ddof=1).mean()),
        critic_dropout_td_std_mean=float(repeated_td.std(axis=0, ddof=1).mean()),
        deterministic_td_abs_mean=float(deterministic_td.abs().mean()),
        critic_dropout_gae_std_mean=float(repeated_advantages.std(axis=0, ddof=1).mean()),
        deterministic_centered_gae_rms=float(np.sqrt(np.mean(centered_eval**2))),
        mean_centered_advantage_sign_disagreement=float(np.mean(np.sign(centered) != np.sign(centered_eval)[None])),
        eval_repeated_forward_max_difference=eval_repeat_max,
        interpretation='Same fixed batch; Monte Carlo dropout noise diagnostic only. Not independent training repeats or policy performance.')
    study.backend.write_json(out / 'summary.json', record)
    print(json.dumps({k:v for k,v in record.items() if k not in ('source_state', 'checkpoint')}, indent=2))


if __name__ == '__main__':
    main()
