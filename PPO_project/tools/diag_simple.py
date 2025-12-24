# -*- coding: utf-8 -*-
"""简单诊断脚本"""
import sys
sys.path.insert(0, 'PPO_project')
from src.environment import Env
import yaml
import numpy as np
import math

with open('PPO_project/configs/train_square.yaml', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
env = Env(cfg)
obs, _ = env.reset()

half_eps = float(getattr(env, 'half_epsilon', 0.75))
print(f'half_eps: {half_eps}, total_len: {env._progress_total_length}')

done = False
steps = 0
max_steps = 80
e_n_max = 0
corner_steps = 0

CORNER_OMEGA_SCALE = 0.55

while not done and steps < max_steps:
    pos = np.array(env.position)
    s_now = float(getattr(env, '_progress_s', 0.0))
    heading = float(getattr(env, 'heading', 0.0))
    
    proj = np.asarray(env._interpolate_progress_point_at_s(s_now), dtype=float)
    t_hat, n_hat = env._get_frenet_frame_at_s(s_now)
    e_n = float(np.dot(pos - proj, n_hat))
    e_n_max = max(e_n_max, abs(e_n))
    
    scan = env._scan_for_next_turn(s_now)
    dist_to_turn = float(scan.get('dist_to_turn', float('inf')))
    turn_angle = float(scan.get('turn_angle', 0.0))
    
    corner_mode = False
    if math.isfinite(turn_angle) and abs(turn_angle) > 1e-6:
        sin_half = math.sin(min(abs(turn_angle) * 0.5, 0.5 * math.pi))
        sin_half = max(sin_half, 0.2)
        r_allow = half_eps / max(sin_half, 1e-9)
        d_fillet = r_allow * math.tan(abs(turn_angle) * 0.5)
        if math.isfinite(dist_to_turn) and dist_to_turn <= d_fillet:
            corner_mode = True
            corner_steps += 1
    
    if corner_mode:
        omega_ratio = CORNER_OMEGA_SCALE * math.copysign(1.0, turn_angle)
    else:
        L = np.clip(dist_to_turn + 1.5 * half_eps, 1.5 * half_eps, 4.0 * half_eps)
        p_target = np.asarray(env._interpolate_progress_point_at_s(s_now + L), dtype=float)
        theta_des = math.atan2(p_target[1] - pos[1], p_target[0] - pos[0])
        heading_err = env._wrap_angle(theta_des - heading)
        omega_ratio = np.clip(2.0 * heading_err - 1.5 * e_n / half_eps, -0.6, 0.6)
    
    action = np.array([float(omega_ratio), 0.05])
    obs, rew, term, trunc, info = env.step(action)
    done = term or trunc
    steps += 1
    
    if steps <= 5 or corner_mode or done:
        print(f'Step {steps:3d}: s={s_now:.2f} e_n={e_n:+.4f} dist={dist_to_turn:.2f} corner={corner_mode} omega={omega_ratio:+.3f}')

print()
print(f'Result: steps={steps}, e_n_max={e_n_max:.4f}, corner_steps={corner_steps}')
print(f'reached={info.get("reached_target", False)}, term={term}, trunc={trunc}')
