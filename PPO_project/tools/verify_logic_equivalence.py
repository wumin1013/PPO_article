#!/usr/bin/env python3
import argparse
import sys
import yaml
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.environment import Env
from src.utils.path_generator import get_path_by_name
import torch

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_env_from_config(cfg):
    """构建环境，与 acceptance_suite.py 保持一致"""
    env_cfg = cfg["environment"]
    kcm_cfg = cfg["kinematic_constraints"]
    path_cfg = cfg["path"]
    reward_weights = cfg.get("reward_weights", {})

    scale = float(path_cfg.get("scale", 10.0))
    num_points = int(path_cfg.get("num_points", 200))
    extra_kwargs = {k: v for k, v in path_cfg.items() if k not in {"type", "scale", "num_points"}}
    path_points = get_path_by_name(str(path_cfg["type"]), scale=scale, num_points=num_points, **extra_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    env = Env(
        device=device,
        epsilon=env_cfg["epsilon"],
        interpolation_period=env_cfg["interpolation_period"],
        MAX_VEL=kcm_cfg["MAX_VEL"],
        MAX_ACC=kcm_cfg["MAX_ACC"],
        MAX_JERK=kcm_cfg["MAX_JERK"],
        MAX_ANG_VEL=kcm_cfg["MAX_ANG_VEL"],
        MAX_ANG_ACC=kcm_cfg["MAX_ANG_ACC"],
        MAX_ANG_JERK=kcm_cfg["MAX_ANG_JERK"],
        Pm=path_points,
        max_steps=env_cfg["max_steps"],
        lookahead_points=env_cfg.get("lookahead_points", 5),
        lookahead_obs_enabled=env_cfg.get("lookahead_obs_enabled", True),
        lookahead_obs_scales=env_cfg.get("lookahead_obs_scales", [1.0]),
        reward_weights=reward_weights,
        curvature_observation=env_cfg.get("curvature_observation"),
        return_normalized_obs=True,
    )
    return env

def run_episode(env, max_steps=1000, seed=42):
    env.reset()
    # Set seed for reproducibility if supported, otherwise rely on numpy seed
    np.random.seed(seed)
    
    trace = []
    
    obs = env.reset()
    done = False
    step = 0
    
    # Initial state
    trace.append({
        "step": step,
        "x": float(env.current_position[0]),
        "y": float(env.current_position[1]),
        "velocity": float(env.velocity),
        "reward": 0.0,
        "done": False,
        "progress": float(env.state[4]) if len(env.state) > 4 else 0.0,
        "corner_phase": "unknown", # Will be updated
        "turn_sign": 0
    })
    
    while not done and step < max_steps:
        # Fixed action for deterministic behavior testing: 
        # Alternating slightly or just constant to traverse path
        # Using a simple policy: try to maintain speed, small steering
        action = np.array([0.01 * np.sin(step/10.0), 0.8]) 
        
        obs, reward, done, info = env.step(action)
        step += 1
        
        # Extract solidified fields
        # Note: "corner_phase" might be in info['turn_info'] or P4 status depending on version
        corner_phase = False
        turn_sign = 0
        
        if "turn_info" in info:
            corner_phase = info["turn_info"].get("corner_phase", False)
            turn_sign = info["turn_info"].get("turn_sign", 0)
        elif "p4_status" in info:
             # Legacy fallback
             pass
        
        # P0/Pre-Phase20 might store corner_phase differently (e.g. self.in_corner_phase)
        # We can accept getting it from info or env attribute if public
        if hasattr(env, "in_corner_phase"):
             corner_phase = bool(env.in_corner_phase)
             
        trace.append({
            "step": step,
            "x": float(env.current_position[0]),
            "y": float(env.current_position[1]),
            "velocity": float(env.velocity),
            "reward": float(reward),
            "done": bool(done),
            "progress": float(env.state[4]) if len(env.state) > 4 else 0.0,
            "corner_phase": bool(corner_phase),
            "turn_sign": int(turn_sign)
        })
        
    return trace

def generate_trace(config_path, output_path, seed=42):
    cfg = load_config(config_path)
    # Ensure deterministic
    np.random.seed(seed)
    
    env = build_env_from_config(cfg)
    trace = run_episode(env, max_steps=2000, seed=seed)
    
    df = pd.DataFrame(trace)
    df.to_csv(output_path, index=False)
    print(f"Trace saved to {output_path} with {len(df)} steps.")

def compare_traces(path_before, path_after, report_path):
    df_b = pd.read_csv(path_before)
    df_a = pd.read_csv(path_after)
    
    report = {
        "status": "pass",
        "diffs": [],
        "max_errors": {}
    }
    
    if len(df_b) != len(df_a):
        report["status"] = "fail"
        report["diffs"].append(f"Length mismatch: {len(df_b)} vs {len(df_a)}")
        # Truncate to min len for further comparison
        min_len = min(len(df_b), len(df_a))
        df_b = df_b.iloc[:min_len]
        df_a = df_a.iloc[:min_len]
    
    # Compare columns
    columns = ["x", "y", "velocity", "reward", "progress"]
    tolerances = {
        "x": 1e-9,
        "y": 1e-9,
        "velocity": 1e-9,
        "reward": 1e-9,
        "progress": 1e-9
    }
    
    for col in columns:
        if col not in df_b.columns or col not in df_a.columns:
            continue
            
        diff = np.abs(df_b[col] - df_a[col])
        max_diff = diff.max()
        report["max_errors"][col] = float(max_diff)
        
        if max_diff > tolerances[col]:
            report["status"] = "fail"
            report["diffs"].append(f"Column {col} mismatch. Max diff: {max_diff}")
            
    # Boolean cols
    for col in ["done", "corner_phase", "turn_sign"]:
        if col not in df_b.columns or col not in df_a.columns:
            continue
        # Check strict equality
        mismatch = (df_b[col] != df_a[col]).sum()
        if mismatch > 0:
             report["status"] = "fail"
             report["diffs"].append(f"Column {col} mismatch count: {mismatch}")
             report["max_errors"][col] = float(mismatch)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    print(f"Comparison report saved to {report_path}")
    if report["status"] == "pass":
        print("✅ LOGIC EQUIVALENCE CHECK PASSED")
        return 0
    else:
        print("❌ LOGIC EQUIVALENCE CHECK FAILED")
        return 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["before", "after", "compare"], required=True)
    parser.add_argument("--config", default="configs/p0_l2_gold.yaml")
    parser.add_argument("--out", default="out/phase20_gate")
    parser.add_argument("--trace_before", default="trace_before.csv")
    parser.add_argument("--trace_after", default="trace_after.csv")
    
    args = parser.parse_args()
    
    os.makedirs(args.out, exist_ok=True)
    
    if args.mode == "before":
        generate_trace(args.config, os.path.join(args.out, args.trace_before))
    elif args.mode == "after":
        generate_trace(args.config, os.path.join(args.out, args.trace_after))
    elif args.mode == "compare":
        ret = compare_traces(
            os.path.join(args.out, args.trace_before),
            os.path.join(args.out, args.trace_after),
            os.path.join(args.out, "diff_report.json")
        )
        sys.exit(ret)

if __name__ == "__main__":
    main()
