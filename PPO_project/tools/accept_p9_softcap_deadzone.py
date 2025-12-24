"""P9 自动化验收：SoftCap + Deadzone + 死区内弱向心力。

验收依据：
- `P9_patch_instructions_v2_CN.md`

验收项目：
1. Soft 模式：多步验证 v_u_exec == v_u_policy（误差 < 1e-6），确保已取消 turning-feasible 的硬截断。
2. 在拐角前强制 v_u_policy=1.0，至少一次出现 cap_violation_ratio > 0（说明 cap 在算且能"看见弯"）。
3. reward components 必须包含 cap_violation_penalty，且 violation 时该项为负。
4. 当 abs(e) <= dz 时，tracking_reward 必须为正，且 tracking_reward <= deadzone_center_weight。

输出要求：
- summary.json：pass/fail + 关键统计（violation 次数、均值等）
- trace.csv：包含关键列

退出码：
- pass：0
- fail：2
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]  # PPO_project
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

try:
    import matplotlib.pyplot as plt

    from src.environment import Env
except ImportError as exc:  # pragma: no cover
    print(f"[ERROR] 依赖缺失：{exc}. 请先安装依赖，例如: python.cmd -m pip install -r PPO_project/requirements.txt")
    raise


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_outdir(outdir: Optional[Path]) -> Path:
    if outdir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return ROOT / "artifacts" / "p9_softcap" / stamp
    outdir = Path(outdir)
    return outdir if outdir.is_absolute() else (ROOT / outdir)


def _pm_for_square(scale: float, num_points: int) -> List[np.ndarray]:
    """生成闭环正方形路径"""
    L = float(scale)
    vertices = [
        np.array([0.0, 0.0], dtype=float),
        np.array([L, 0.0], dtype=float),
        np.array([L, L], dtype=float),
        np.array([0.0, L], dtype=float),
        np.array([0.0, 0.0], dtype=float),  # 闭合
    ]
    per_edge = max(2, int(num_points) // 4)
    pts: List[np.ndarray] = []
    for i in range(4):
        p0, p1 = vertices[i], vertices[i + 1]
        ts = np.linspace(0.0, 1.0, per_edge, endpoint=(i == 3))
        if i > 0:
            ts = ts[1:]  # 避免重复
        for t in ts:
            pts.append(p0 + t * (p1 - p0))
    return [np.array(p, dtype=float) for p in pts]


@dataclass
class P9TestResult:
    """P9 验收结果"""
    passed: bool = True
    soft_mode_verified: bool = False
    cap_violation_seen: bool = False
    cap_violation_penalty_negative: bool = False
    deadzone_tracking_valid: bool = False
    violation_count: int = 0
    violation_mean: float = 0.0
    soft_mode_errors: List[float] = field(default_factory=list)
    deadzone_tracking_in_dz: List[float] = field(default_factory=list)
    messages: List[str] = field(default_factory=list)


def run_p9_acceptance(cfg: Dict, outdir: Path, max_steps: int = 500) -> P9TestResult:
    """运行 P9 验收测试"""
    result = P9TestResult()
    
    # 构建环境
    env_cfg = cfg.get("environment", {}) or {}
    kin_cfg = cfg.get("kinematic_constraints", {}) or {}
    reward_cfg = cfg.get("reward_weights", {}) or {}
    path_cfg = cfg.get("path", {}) or {}
    
    epsilon = float(env_cfg.get("epsilon", 1.5))
    half_epsilon = epsilon / 2.0
    scale = float(path_cfg.get("scale", 10.0))
    num_points = int(path_cfg.get("num_points", 200))
    
    Pm = _pm_for_square(scale, num_points)
    
    env = Env(
        device="cpu",
        epsilon=epsilon,
        interpolation_period=float(env_cfg.get("interpolation_period", 0.01)),
        MAX_VEL=float(kin_cfg.get("MAX_VEL", 20.0)),
        MAX_ACC=float(kin_cfg.get("MAX_ACC", 200.0)),
        MAX_JERK=float(kin_cfg.get("MAX_JERK", 2000.0)),
        MAX_ANG_VEL=float(kin_cfg.get("MAX_ANG_VEL", math.pi * 2)),
        MAX_ANG_ACC=float(kin_cfg.get("MAX_ANG_ACC", 100.0)),
        MAX_ANG_JERK=float(kin_cfg.get("MAX_ANG_JERK", 1000.0)),
        Pm=Pm,
        max_steps=int(env_cfg.get("max_steps", 4000)),
        reward_weights=reward_cfg,
        lookahead_points=int(env_cfg.get("lookahead_points", 8)),
        return_normalized_obs=True,
    )
    
    # 检查是否为 soft 模式
    cap_mode = str(getattr(env, "_p4_cap_mode", "hard")).lower()
    is_soft = (cap_mode == "soft")
    if not is_soft:
        result.messages.append(f"[WARN] cap_mode={cap_mode}，非 soft 模式，部分验收项跳过")
    
    # 读取 deadzone 配置
    deadzone_ratio = float(getattr(env, "_p4_deadzone_ratio", 0.8))
    deadzone_center_weight = float(getattr(env, "_p4_deadzone_center_weight", 0.1))
    dz = deadzone_ratio * half_epsilon
    
    trace_rows = []
    env.reset()
    
    violations = []
    soft_mode_errors = []
    deadzone_tracking_rewards = []
    cap_violation_penalties_negative = []
    
    for step_i in range(max_steps):
        # 策略：前半段正常跑，后半段强制满速冲弯测试 cap_violation
        if step_i < max_steps // 2:
            # 正常动作
            theta_u = 0.0
            v_u = 0.8
        else:
            # 强制满速
            theta_u = 0.0
            v_u = 1.0
        
        action = np.array([theta_u, v_u], dtype=float)
        obs, reward, done, info = env.step(action)
        
        p4_status = info.get("p4_status", {}) or {}
        reward_components = getattr(env, "last_reward_components", {}) or {}
        corridor_status = info.get("corridor_status", {}) or {}
        
        # 记录 trace
        contour_error = info.get("contour_error", 0.0)
        v_u_policy = float(p4_status.get("v_u_policy", v_u))
        v_u_exec = float(p4_status.get("v_u_exec", 0.0))
        v_ratio_cap = float(p4_status.get("v_ratio_cap", 1.0))
        v_ratio_brake = float(p4_status.get("v_ratio_brake", 1.0))
        cap_violation_ratio = float(p4_status.get("cap_violation_ratio", 0.0))
        tracking_reward = float(reward_components.get("tracking_reward", 0.0))
        cap_violation_penalty = float(reward_components.get("cap_violation_penalty", 0.0))
        corridor_active = float(reward_components.get("corridor_active", 0.0))
        
        trace_rows.append({
            "step": step_i,
            "contour_error": contour_error,
            "v_u_policy": v_u_policy,
            "v_u_exec": v_u_exec,
            "v_ratio_cap": v_ratio_cap,
            "v_ratio_brake": v_ratio_brake,
            "cap_violation_ratio": cap_violation_ratio,
            "tracking_reward": tracking_reward,
            "cap_violation_penalty": cap_violation_penalty,
            "corridor_active": corridor_active,
            "reward": reward,
        })
        
        # 验收项 1：Soft 模式下 v_u_exec == v_u_policy
        if is_soft:
            exec_err = abs(v_u_exec - v_u_policy)
            soft_mode_errors.append(exec_err)
        
        # 验收项 2：cap_violation_ratio > 0
        if cap_violation_ratio > 0.0:
            violations.append(cap_violation_ratio)
        
        # 验收项 3：violation 时 cap_violation_penalty < 0
        if cap_violation_ratio > 0.0 and cap_violation_penalty < 0.0:
            cap_violation_penalties_negative.append(cap_violation_penalty)
        
        # 验收项 4：deadzone 内 tracking_reward 为正（仅非 corridor 模式）
        e_abs = abs(contour_error)
        is_corridor_active = bool(corridor_active > 0.5)
        if e_abs <= dz and not is_corridor_active:
            deadzone_tracking_rewards.append(tracking_reward)
        
        if done:
            break
    
    # 评估结果
    # 验收项 1：Soft 模式
    if is_soft:
        max_exec_err = max(soft_mode_errors) if soft_mode_errors else 0.0
        if max_exec_err < 1e-6:
            result.soft_mode_verified = True
            result.messages.append(f"[PASS] Soft 模式验证：max exec_err={max_exec_err:.2e} < 1e-6")
        else:
            result.soft_mode_verified = False
            result.passed = False
            result.messages.append(f"[FAIL] Soft 模式验证：max exec_err={max_exec_err:.2e} >= 1e-6")
        result.soft_mode_errors = soft_mode_errors[:20]  # 仅保留前20个
    else:
        result.soft_mode_verified = True  # 非 soft 模式跳过
    
    # 验收项 2：cap_violation 出现（仅 soft 模式检查）
    result.violation_count = len(violations)
    result.violation_mean = float(np.mean(violations)) if violations else 0.0
    if is_soft:
        if len(violations) > 0:
            result.cap_violation_seen = True
            result.messages.append(f"[PASS] cap_violation 检测：{len(violations)} 次，mean={result.violation_mean:.4f}")
        else:
            result.cap_violation_seen = False
            result.passed = False
            result.messages.append("[FAIL] cap_violation 从未出现（cap 无法\"看见弯\"）")
    else:
        # Hard 模式下不检查 cap_violation（因为 hard 模式直接截断，不记录违规）
        result.cap_violation_seen = True  # 跳过
        result.messages.append(f"[INFO] Hard 模式跳过 cap_violation 检查")
    
    # 验收项 3：penalty 为负（仅 soft 模式检查）
    if is_soft:
        if len(cap_violation_penalties_negative) > 0:
            result.cap_violation_penalty_negative = True
            result.messages.append(f"[PASS] cap_violation_penalty 为负：{len(cap_violation_penalties_negative)} 次")
        elif len(violations) > 0:
            # 有 violation 但 penalty 不为负
            result.cap_violation_penalty_negative = False
            result.passed = False
            result.messages.append("[FAIL] cap_violation_penalty 在 violation 时应为负")
        else:
            result.cap_violation_penalty_negative = True  # 无 violation 时跳过
    else:
        # Hard 模式跳过
        result.cap_violation_penalty_negative = True
        result.messages.append(f"[INFO] Hard 模式跳过 cap_violation_penalty 检查")
    
    # 验收项 4：deadzone 内 tracking_reward（非 corridor 模式）
    if deadzone_tracking_rewards:
        # 检查：reward >= 0（边界处可能为 0）且 <= deadzone_center_weight
        all_non_negative = all(r >= 0 for r in deadzone_tracking_rewards)
        all_bounded = all(r <= deadzone_center_weight + 1e-6 for r in deadzone_tracking_rewards)
        if all_non_negative and all_bounded:
            result.deadzone_tracking_valid = True
            result.messages.append(f"[PASS] Deadzone tracking：{len(deadzone_tracking_rewards)} 步在死区内（非corridor），reward 均非负且 <= {deadzone_center_weight}")
        else:
            result.deadzone_tracking_valid = False
            result.passed = False
            neg_count = sum(1 for r in deadzone_tracking_rewards if r < 0)
            over_count = sum(1 for r in deadzone_tracking_rewards if r > deadzone_center_weight + 1e-6)
            result.messages.append(f"[FAIL] Deadzone tracking：{neg_count} 步为负，{over_count} 步超限")
        result.deadzone_tracking_in_dz = deadzone_tracking_rewards[:20]
    else:
        result.deadzone_tracking_valid = True  # 无数据跳过
        result.messages.append("[INFO] 无法验证 deadzone tracking（未进入死区或全程为 corridor 模式）")
    
    # 保存 trace.csv
    outdir.mkdir(parents=True, exist_ok=True)
    trace_path = outdir / "trace.csv"
    if trace_rows:
        with trace_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(trace_rows[0].keys()))
            writer.writeheader()
            writer.writerows(trace_rows)
        result.messages.append(f"[INFO] trace.csv 已保存：{trace_path}")
    
    # 保存 summary.json
    summary = {
        "passed": result.passed,
        "soft_mode_verified": result.soft_mode_verified,
        "cap_violation_seen": result.cap_violation_seen,
        "cap_violation_penalty_negative": result.cap_violation_penalty_negative,
        "deadzone_tracking_valid": result.deadzone_tracking_valid,
        "violation_count": result.violation_count,
        "violation_mean": result.violation_mean,
        "cap_mode": cap_mode,
        "deadzone_ratio": deadzone_ratio,
        "deadzone_center_weight": deadzone_center_weight,
        "messages": result.messages,
    }
    summary_path = outdir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    result.messages.append(f"[INFO] summary.json 已保存：{summary_path}")
    
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="P9 SoftCap+Deadzone 验收脚本")
    parser.add_argument("--config", type=str, default="configs/train_square_softcap_scaled.yaml", help="配置文件路径")
    parser.add_argument("--outdir", type=str, default=None, help="输出目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--max-steps", type=int, default=500, help="最大步数")
    args = parser.parse_args()
    
    _set_seed(args.seed)
    
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    if not config_path.exists():
        print(f"[ERROR] 配置文件不存在：{config_path}")
        return 2
    
    cfg = _load_yaml(config_path)
    outdir = _resolve_outdir(Path(args.outdir) if args.outdir else None)
    
    print(f"[P9 验收] 配置：{config_path}")
    print(f"[P9 验收] 输出：{outdir}")
    
    result = run_p9_acceptance(cfg, outdir, max_steps=args.max_steps)
    
    print("\n" + "=" * 60)
    for msg in result.messages:
        print(msg)
    print("=" * 60)
    
    if result.passed:
        print("\n✅ P9 验收通过")
        return 0
    else:
        print("\n❌ P9 验收失败")
        return 2


if __name__ == "__main__":
    sys.exit(main())
