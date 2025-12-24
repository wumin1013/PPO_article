# -*- coding: utf-8 -*-
"""诊断脚本 - 检查 corner exit 后的速度变化"""
import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import yaml
import numpy as np

cfg = yaml.safe_load(Path("configs/train_square.yaml").open("r", encoding="utf-8"))
from tools.accept_p8_1_observation_and_corner_phase import (
    build_open_square,
    _build_env,
    _expert_policy,
    RECOVERY_E_ON_RATIO,
)

env = _build_env(cfg, build_open_square(side=10.0, num_points=200))
obs = env.reset()
half_eps = float(getattr(env, "half_epsilon", 0.75))

done = False
step = 0
vcmd_prev = None
rs = {}

print(f"half_eps={half_eps}, RECOVERY_E_ON_RATIO={RECOVERY_E_ON_RATIO}")
print(f"recovery trigger at e_n > {RECOVERY_E_ON_RATIO * half_eps:.4f}")
print("--- Checking corner exit and heading ---")
print("step | e_n    | corner | heading_err | vcap_ang | vcap | v_exec | omega")

while not done and step < 200:
    action, vcmd, vcap_pre, cm = _expert_policy(
        env=env, v_ratio_cmd_prev=vcmd_prev, recenter_state=rs
    )
    vcmd_prev = vcmd
    obs, _, done, info = env.step(action)
    proj, _, s, _, nh = env._project_onto_progress_path(env.current_position)
    en = float(np.dot(env.current_position - proj, nh))
    
    p4 = info.get("p4_status", {})
    corner_mode_p4 = float(p4.get("corner_mode", 0.0))
    v_cap_ang = float(p4.get("v_ratio_cap_ang", 0.0))
    v_cap = float(p4.get("v_ratio_cap", 0.0))
    v_exec = float(getattr(env, "velocity", 0.0)) / float(env.MAX_VEL)
    
    # 计算 heading error (与切向方向的偏差)
    heading = float(getattr(env, "heading", 0.0))
    theta_path = float(env._tangent_angle_at_s(s))
    heading_err_rad = abs(float(env._wrap_angle(theta_path - heading)))
    heading_err_deg = math.degrees(heading_err_rad)
    
    omega = float(action[0])
    
    if step >= 40 and step <= 60:
        v_cmd = float(action[1])  # v_ratio_cmd from action
        v_cap_geom = float(p4.get("v_ratio_cap_geom", 0.0))
        turn_angle_deg = float(p4.get("turn_angle", 0.0)) * 180 / 3.14159
        print(f"{step:4d} | e_n={en:+.4f} | cm={corner_mode_p4:.0f} | psi={heading_err_deg:+.1f}° | turn={turn_angle_deg:+.1f}° | v_cap={v_cap:.4f} | w={omega:+.4f}", flush=True)
    step += 1

print(f"\nFinal: reached={env.reached_target}, steps={step}")
