"""P4.0 自动化验收：直线快进给/弯前降速/出弯加速/速度硬上限/停滞终止/视界打印。

设计原则（KISS）：
- 不做耗时训练，用可复现的启发式控制器验证环境侧逻辑是否生效；
- 输出与 00_README 一致的指标打印 + 关键可视化 PNG；
- 所有检查失败会返回非 0 退出码。
"""

from __future__ import annotations

import argparse
import math
import random
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

try:
    import matplotlib.pyplot as plt  # noqa: E402

    from src.environment import Env  # noqa: E402
except ImportError as exc:  # pragma: no cover
    print(f"[ERROR] 依赖缺失：{exc}. 请先安装依赖，例如: python -m pip install -r PPO_project/requirements.txt")
    raise


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_line(length: float, num_points: int, angle: float = 0.0) -> List[np.ndarray]:
    ts = np.linspace(0.0, 1.0, max(2, int(num_points)))
    dx = math.cos(float(angle))
    dy = math.sin(float(angle))
    return [np.array([float(length) * t * dx, float(length) * t * dy], dtype=float) for t in ts]


def build_open_square(side: float, num_points: int) -> List[np.ndarray]:
    if num_points < 10:
        raise ValueError("num_points too small")
    L = float(side)
    vertices = [
        np.array([0.0, 0.0], dtype=float),
        np.array([L, 0.0], dtype=float),
        np.array([L, L], dtype=float),
        np.array([0.0, L], dtype=float),
    ]
    per_edge = max(2, num_points // 3)

    def edge_points(p0: np.ndarray, p1: np.ndarray, n: int, *, include_start: bool) -> List[np.ndarray]:
        ts = np.linspace(0.0, 1.0, max(2, int(n)), endpoint=True)
        if not include_start:
            ts = ts[1:]
        return [p0 + t * (p1 - p0) for t in ts]

    pts: List[np.ndarray] = []
    pts.extend(edge_points(vertices[0], vertices[1], per_edge, include_start=True))
    pts.extend(edge_points(vertices[1], vertices[2], per_edge, include_start=False))
    pts.extend(edge_points(vertices[2], vertices[3], num_points - len(pts), include_start=False))
    return [np.array(p, dtype=float) for p in pts]


def build_s_shape(scale: float, num_points: int, amplitude: float, periods: float) -> List[np.ndarray]:
    t = np.linspace(0.0, 1.0, max(2, int(num_points)))
    x = float(scale) * t
    y = float(amplitude) * np.sin(2.0 * math.pi * float(periods) * t)
    return [np.array([float(x[i]), float(y[i])], dtype=float) for i in range(len(t))]


def _build_env(cfg: Dict, pm: Sequence[np.ndarray], *, p4_override: Optional[dict] = None) -> Env:
    env_cfg = cfg["environment"]
    kcm_cfg = cfg["kinematic_constraints"]
    reward_weights = dict(cfg.get("reward_weights", {}) or {})
    if p4_override is not None:
        reward_weights["p4"] = dict(p4_override)
    return Env(
        device="cpu",
        epsilon=float(env_cfg.get("epsilon", 0.5)),
        interpolation_period=float(env_cfg.get("interpolation_period", 0.1)),
        MAX_VEL=float(kcm_cfg.get("MAX_VEL", 1.0)),
        MAX_ACC=float(kcm_cfg.get("MAX_ACC", 2.0)),
        MAX_JERK=float(kcm_cfg.get("MAX_JERK", 3.0)),
        MAX_ANG_VEL=float(kcm_cfg.get("MAX_ANG_VEL", 1.5)),
        MAX_ANG_ACC=float(kcm_cfg.get("MAX_ANG_ACC", 3.0)),
        MAX_ANG_JERK=float(kcm_cfg.get("MAX_ANG_JERK", 5.0)),
        Pm=list(pm),
        max_steps=int(env_cfg.get("max_steps", 200)),
        lookahead_points=int(env_cfg.get("lookahead_points", 5)),
        reward_weights=reward_weights,
    )


def _controller_step(env: Env, corridor_status: Dict[str, object], *, kp_lat: float, v_policy: float) -> np.ndarray:
    half = float(getattr(env, "half_epsilon", 1.0))
    e_n = float(corridor_status.get("e_n", 0.0))
    theta = float(np.clip(kp_lat * (0.0 - e_n) / max(half, 1e-6), -1.0, 1.0))
    return np.array([theta, float(np.clip(v_policy, 0.0, 1.0))], dtype=float)


@dataclass
class Metrics:
    episodes: int
    success_rate: float
    oob_rate: float
    steps_mean: float
    v_mean: float
    rmse_mean: float


def _run_path_eval(
    env: Env,
    *,
    episodes: int,
    seed: int,
    kp_lat: float,
) -> Tuple[Metrics, Dict[str, List[float]]]:
    successes = 0
    oobs = 0
    steps: List[int] = []
    v_means: List[float] = []
    rmses: List[float] = []

    # 用于“速度 vs dist_to_turn”可视化（只记录首个 episode）
    trace: Dict[str, List[float]] = {
        "step": [],
        "dist_to_turn": [],
        "turn_severity": [],
        "kappa": [],
        "v_ratio_policy": [],
        "v_ratio_exec": [],
        "v_ratio_cap": [],
        "speed_target": [],
        "progress_multiplier": [],
    }

    for ep in range(int(episodes)):
        _set_seed(int(seed) + ep)
        env.reset()
        done = False
        info: Dict[str, object] = {}
        v_samples: List[float] = []
        e_samples: List[float] = []

        # 初始用“当前预计算 speed_target”
        p4_next = env._compute_p4_pre_step_status()  # type: ignore[attr-defined]
        action = np.array([0.0, float(p4_next.get("speed_target", 0.8))], dtype=float)

        while not done:
            _, _, done, info = env.step(action)
            v_samples.append(float(getattr(env, "velocity", 0.0)))
            e_samples.append(float(info.get("contour_error", 0.0)))

            corridor_status = info.get("corridor_status", {}) if isinstance(info, dict) else {}
            if not isinstance(corridor_status, dict):
                corridor_status = {}

            # 下一步速度目标：按环境同款计算（基于当前 s_{t+1}）
            p4_next = env._compute_p4_pre_step_status()  # type: ignore[attr-defined]
            v_policy = float(p4_next.get("speed_target", 0.8))
            action = _controller_step(env, corridor_status, kp_lat=float(kp_lat), v_policy=v_policy)

            # 仅记录首个 episode 的 trace（使用 env 返回的 p4_status：对应本步执行）
            if ep == 0 and isinstance(info, dict):
                p4 = info.get("p4_status", {})
                if isinstance(p4, dict):
                    trace["step"].append(float(p4.get("step", len(trace["step"]))))
                    trace["dist_to_turn"].append(float(p4.get("dist_to_turn", float("nan"))))
                    trace["turn_severity"].append(float(p4.get("turn_severity", float("nan"))))
                    trace["kappa"].append(float(p4.get("kappa", float("nan"))))
                    trace["v_ratio_policy"].append(float(p4.get("v_ratio_policy", float("nan"))))
                    trace["v_ratio_exec"].append(float(p4.get("v_ratio_exec", float("nan"))))
                    trace["v_ratio_cap"].append(float(p4.get("v_ratio_cap", float("nan"))))
                    trace["speed_target"].append(float(p4.get("speed_target", float("nan"))))
                    trace["progress_multiplier"].append(float(p4.get("progress_multiplier", 1.0)))

        reached_target = bool(getattr(env, "reached_target", False))
        contour_error = float(info.get("contour_error", 0.0)) if isinstance(info, dict) else float("inf")
        oob = bool(contour_error > float(getattr(env, "half_epsilon", 1.0)))

        if reached_target:
            successes += 1
        if oob:
            oobs += 1
        steps.append(int(getattr(env, "current_step", 0)))
        v_means.append(float(np.mean(v_samples)) if v_samples else 0.0)
        rmses.append(float(np.sqrt(np.mean(np.square(e_samples)))) if e_samples else 0.0)

    m = Metrics(
        episodes=int(episodes),
        success_rate=float(successes) / max(1, int(episodes)),
        oob_rate=float(oobs) / max(1, int(episodes)),
        steps_mean=float(np.mean(steps)) if steps else 0.0,
        v_mean=float(np.mean(v_means)) if v_means else 0.0,
        rmse_mean=float(np.mean(rmses)) if rmses else 0.0,
    )
    return m, trace


def _plot_speed_trace(trace: Dict[str, List[float]], out_path: Path, *, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.asarray(trace.get("dist_to_turn", []), dtype=float)
    v_exec = np.asarray(trace.get("v_ratio_exec", []), dtype=float)
    v_cap = np.asarray(trace.get("v_ratio_cap", []), dtype=float)
    v_tgt = np.asarray(trace.get("speed_target", []), dtype=float)

    ok = np.isfinite(x) & np.isfinite(v_exec)
    if ok.any():
        ax.scatter(x[ok], v_exec[ok], s=10, alpha=0.65, label="v_ratio_exec")
    ok_cap = np.isfinite(x) & np.isfinite(v_cap)
    if ok_cap.any():
        ax.scatter(x[ok_cap], v_cap[ok_cap], s=10, alpha=0.45, label="v_ratio_cap")
    ok_tgt = np.isfinite(x) & np.isfinite(v_tgt)
    if ok_tgt.any():
        ax.scatter(x[ok_tgt], v_tgt[ok_tgt], s=10, alpha=0.45, label="speed_target")

    ax.set_xlabel("dist_to_turn (state)")
    ax.set_ylabel("velocity ratio")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _check_speed_profile(trace: Dict[str, List[float]]) -> Tuple[bool, str]:
    v_exec = np.asarray(trace.get("v_ratio_exec", []), dtype=float)
    dist = np.asarray(trace.get("dist_to_turn", []), dtype=float)
    sev = np.asarray(trace.get("turn_severity", []), dtype=float)
    cap = np.asarray(trace.get("v_ratio_cap", []), dtype=float)
    mult = np.asarray(trace.get("progress_multiplier", []), dtype=float)

    if v_exec.size < 10:
        return False, "trace too short"

    # 1) 硬上限恒成立：v_exec <= v_cap
    ok_cap = np.isfinite(v_exec) & np.isfinite(cap)
    if ok_cap.any():
        if float(np.max(v_exec[ok_cap] - cap[ok_cap])) > 1e-6:
            return False, "v_ratio_exec exceeded v_ratio_cap"
        if float(np.min(cap[ok_cap])) >= 0.999:
            return False, "v_ratio_cap never dropped below 1 (cap not active)"

    # 2) 速度趋势：靠近急弯应更慢
    near_turn = np.isfinite(dist) & np.isfinite(sev) & (dist < np.nanpercentile(dist, 35)) & (sev > 0.2)
    straight = np.isfinite(sev) & (sev < 0.05)
    if near_turn.any() and straight.any():
        v_near = float(np.nanmean(v_exec[near_turn]))
        v_straight = float(np.nanmean(v_exec[straight]))
        if not (v_straight > v_near + 0.05):
            return False, f"speed profile weak: straight={v_straight:.3f} near_turn={v_near:.3f}"

    # 3) exit boost：至少出现过 progress_multiplier>1
    if not (np.isfinite(mult).any() and float(np.nanmax(mult)) > 1.01):
        return False, "exit boost not observed (progress_multiplier never > 1)"

    return True, "ok"


def _run_stall_test(cfg: Dict, *, seed: int) -> Tuple[bool, str]:
    pm = build_line(length=2.0, num_points=50, angle=0.0)
    p4_override = {
        "stall_enabled": True,
        "stall_steps": 20,
        "stall_progress_eps": 1e-4,
        "stall_v_eps": 0.05,
        "stall_penalty": -8.0,
        "time_penalty": -0.01,
        "speed_cap_enabled": True,
    }
    env = _build_env(cfg, pm, p4_override=p4_override)
    _set_seed(seed)
    env.reset()
    action = np.array([0.0, 0.0], dtype=float)
    done = False
    info: Dict[str, object] = {}
    while not done:
        _, _, done, info = env.step(action)
        if int(getattr(env, "current_step", 0)) > 2000:
            return False, "stall test did not terminate"
    p4 = info.get("p4_status", {}) if isinstance(info, dict) else {}
    triggered = False
    if isinstance(p4, dict):
        triggered = bool(float(p4.get("stall_triggered", 0.0)) > 0.5)
    if not triggered:
        return False, "stall_triggered not set in p4_status"
    if bool(getattr(env, "reached_target", False)):
        return False, "stall test unexpectedly reached target"
    return True, "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description="P4.0 自动化验收（speed/boost/cap/stall/done）")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "original_configs" / "smoke.yaml",
        help="用于读取环境/约束默认值的 YAML（默认 original_configs/smoke.yaml）。",
    )
    parser.add_argument("--episodes", type=int, default=20, help="每条路径评估 episode 数（默认 20）。")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kp-lat", type=float, default=1.0, help="横向误差 P 控制系数。")
    parser.add_argument("--outdir", type=Path, default=None)
    args = parser.parse_args()

    if not args.config.exists():
        print(f"[ERROR] missing config: {args.config}")
        raise SystemExit(2)

    cfg = _load_yaml(args.config)
    env_cfg = cfg.get("environment", {}) or {}
    path_cfg = cfg.get("path", {}) or {}
    ppo_cfg = cfg.get("ppo", {}) or {}

    dt = float(env_cfg.get("interpolation_period", 0.1))
    gamma = float(ppo_cfg.get("gamma", 0.99))

    # 验证 5：视界打印（调用工具脚本，顺便验证文件存在）
    try:
        tool = ROOT / "tools" / "print_effective_horizon.py"
        subprocess.run([sys.executable, str(tool), "--config", str(args.config)], check=True)
    except Exception as exc:
        print(f"[FAIL] print_effective_horizon failed: {exc}")
        raise SystemExit(2)

    scale = float(path_cfg.get("scale", 10.0))
    num_points = int(path_cfg.get("num_points", 200))
    s_cfg = path_cfg.get("s_shape", {}) if isinstance(path_cfg.get("s_shape", {}), dict) else {}
    amplitude = float(s_cfg.get("amplitude", scale / 2.0))
    periods = float(s_cfg.get("periods", 2.0))

    paths = {
        "line": build_line(length=scale, num_points=num_points, angle=float(path_cfg.get("line", {}).get("angle", 0.0)) if isinstance(path_cfg.get("line", {}), dict) else 0.0),
        "square": build_open_square(side=scale, num_points=num_points),
        "s_shape": build_s_shape(scale=scale, num_points=num_points, amplitude=amplitude, periods=periods),
    }

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = args.outdir or (REPO_ROOT / "logs" / "p4_0_accept" / stamp)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[RUN] seed={args.seed} dt={dt} gamma={gamma:.6f} episodes={int(args.episodes)} kp_lat={float(args.kp_lat)}")

    all_ok = True
    square_trace: Dict[str, List[float]] = {}

    for name, pm in paths.items():
        env = _build_env(cfg, pm)
        m, trace = _run_path_eval(env, episodes=int(args.episodes), seed=int(args.seed), kp_lat=float(args.kp_lat))
        print(
            f"[EVAL] path={name} success_rate={m.success_rate:.3f} oob_rate={m.oob_rate:.3f} "
            f"rmse_mean={m.rmse_mean:.4f} steps_mean={m.steps_mean:.1f} v_mean={m.v_mean:.3f}"
        )
        if m.success_rate < 0.95:
            all_ok = False
            print(f"[FAIL] success_rate < 0.95 on path={name}")
        if name == "square":
            square_trace = trace

    # 验证 2/3：速度模式 + 硬上限（用 square 首回合 trace）
    ok_profile, reason = _check_speed_profile(square_trace)
    if not ok_profile:
        all_ok = False
        print(f"[FAIL] speed profile/cap check: {reason}")
    else:
        print("[PASS] speed profile/cap check passed.")

    if square_trace:
        fig_path = outdir / "square_speed_vs_dist.png"
        _plot_speed_trace(square_trace, fig_path, title=f"Square speed profile (dt={dt})")
        print(f"[OUT] {fig_path}")

    # 验证 4：停滞机制
    ok_stall, stall_reason = _run_stall_test(cfg, seed=int(args.seed))
    if not ok_stall:
        all_ok = False
        print(f"[FAIL] stall termination: {stall_reason}")
    else:
        print("[PASS] stall termination passed.")

    if all_ok:
        print("[PASS] P4.0 acceptance passed.")
        raise SystemExit(0)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
