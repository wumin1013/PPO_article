"""P6.1 自动化验收：快又顺（抖动抑制 + 目标速度平滑器）。

按 12_P6.1 文档要求：
- 输出动作抖动：mean/p90(|Δu|)
- 输出 KCM 干预：mean/p90(kcm_intervention)
- 输出平顺性：jerk_mean / angular_jerk_mean
- 输出稳定性：success_rate / oob_rate / steps_mean / v_mean（line/square/s_shape，E=20）

提供 A/B 对比：
- A：关闭 r_du 与目标速度平滑器
- B：开启 r_du 与目标速度平滑器

说明：本脚本不做耗时训练；使用可复现的“带轻微抖动的启发式控制器”模拟策略 chattering，
用于验证 env 侧目标器能显著降低 KCM 干预与 jerk，同时不牺牲 success。
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
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


def _clip(x: float, lo: float, hi: float) -> float:
    return float(np.clip(float(x), float(lo), float(hi)))


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


def _build_env(cfg: Dict, pm: Sequence[np.ndarray], *, p6_1_override: Optional[dict]) -> Env:
    env_cfg = cfg.get("environment", {}) or {}
    kcm_cfg = cfg.get("kinematic_constraints", {}) or {}
    reward_weights = dict(cfg.get("reward_weights", {}) or {})

    p6_cfg = dict(reward_weights.get("p6_1", {}) or {}) if isinstance(reward_weights.get("p6_1", {}), dict) else {}
    if p6_1_override:
        p6_cfg.update(dict(p6_1_override))
    reward_weights["p6_1"] = p6_cfg

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


def _theta_p_controller(env: Env, corridor_status: Dict[str, object], *, kp: float) -> float:
    half = float(getattr(env, "half_epsilon", 1.0))
    e_n = float(corridor_status.get("e_n", 0.0))
    return _clip(float(kp) * (0.0 - float(e_n)) / max(half, 1e-6), -1.0, 1.0)


@dataclass
class EpisodeStats:
    success: bool
    oob: bool
    steps: int
    v_mean: float
    rmse: float
    du_mean: float
    du_p90: float
    kcm_mean: float
    kcm_p90: float
    jerk_mean: float
    ang_jerk_mean: float


@dataclass
class Summary:
    episodes: int
    success_rate: float
    oob_rate: float
    steps_mean: float
    v_mean: float
    rmse_mean: float
    du_mean: float
    du_p90: float
    kcm_mean: float
    kcm_p90: float
    jerk_mean: float
    ang_jerk_mean: float


def _percentile(x: Sequence[float], q: float) -> float:
    arr = np.asarray(list(x), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))


def _run_episode(
    env: Env,
    *,
    seed: int,
    kp_lat: float,
    jitter_v: float,
    jitter_theta: float,
    max_steps: int,
) -> EpisodeStats:
    _set_seed(seed)
    env.reset()

    done = False
    info: Dict[str, object] = {}
    v_samples: List[float] = []
    e_samples: List[float] = []
    du_samples: List[float] = []
    kcm_samples: List[float] = []
    jerk_samples: List[float] = []
    ang_jerk_samples: List[float] = []

    p4_next = env._compute_p4_pre_step_status()  # type: ignore[attr-defined]
    action = np.array([0.0, float(p4_next.get("speed_target", 0.8))], dtype=float)

    for t in range(int(max_steps)):
        _, _, done, info = env.step(action)
        if not isinstance(info, dict):
            info = {}

        v_samples.append(float(getattr(env, "velocity", 0.0)))
        e_samples.append(float(info.get("contour_error", 0.0)))
        kcm_samples.append(float(info.get("kcm_intervention", 0.0)))
        jerk_samples.append(float(getattr(env, "jerk", 0.0)))
        ang_jerk_samples.append(float(getattr(env, "angular_jerk", 0.0)))

        p4 = info.get("p4_status", {})
        if isinstance(p4, dict):
            du_samples.append(float(abs(float(p4.get("du_l1", 0.0)))))

        if done:
            break

        cs = info.get("corridor_status", {})
        if not isinstance(cs, dict):
            cs = {}
        theta_base = _theta_p_controller(env, cs, kp=float(kp_lat))

        p4_next = env._compute_p4_pre_step_status()  # type: ignore[attr-defined]
        v_base = float(p4_next.get("speed_target", 0.8))

        sign = 1.0 if (t % 2 == 0) else -1.0
        theta_u = _clip(theta_base + sign * float(jitter_theta), -1.0, 1.0)
        v_u = _clip(v_base + sign * float(jitter_v), 0.0, 1.0)
        action = np.array([theta_u, v_u], dtype=float)

    contour_error = float(info.get("contour_error", float("inf"))) if isinstance(info, dict) else float("inf")
    oob = bool(contour_error > float(getattr(env, "half_epsilon", 1.0)))
    success = bool(getattr(env, "reached_target", False))
    steps = int(getattr(env, "current_step", 0))

    v_mean = float(np.mean(v_samples)) if v_samples else 0.0
    rmse = float(np.sqrt(np.mean(np.square(e_samples)))) if e_samples else 0.0
    du_mean = float(np.mean(du_samples)) if du_samples else 0.0
    du_p90 = float(_percentile(du_samples, 90.0)) if du_samples else float("nan")
    kcm_mean = float(np.mean(kcm_samples)) if kcm_samples else 0.0
    kcm_p90 = float(_percentile(kcm_samples, 90.0)) if kcm_samples else float("nan")
    jerk_mean = float(np.mean(np.abs(jerk_samples))) if jerk_samples else 0.0
    ang_jerk_mean = float(np.mean(np.abs(ang_jerk_samples))) if ang_jerk_samples else 0.0

    return EpisodeStats(
        success=bool(success),
        oob=bool(oob),
        steps=int(steps),
        v_mean=float(v_mean),
        rmse=float(rmse),
        du_mean=float(du_mean),
        du_p90=float(du_p90),
        kcm_mean=float(kcm_mean),
        kcm_p90=float(kcm_p90),
        jerk_mean=float(jerk_mean),
        ang_jerk_mean=float(ang_jerk_mean),
    )


def _run_eval(
    env: Env,
    *,
    episodes: int,
    seed: int,
    kp_lat: float,
    jitter_v: float,
    jitter_theta: float,
    max_steps: int,
) -> Summary:
    results: List[EpisodeStats] = []
    for ep in range(int(episodes)):
        results.append(
            _run_episode(
                env,
                seed=int(seed) + ep,
                kp_lat=float(kp_lat),
                jitter_v=float(jitter_v),
                jitter_theta=float(jitter_theta),
                max_steps=int(max_steps),
            )
        )

    success_rate = float(np.mean([1.0 if r.success else 0.0 for r in results])) if results else 0.0
    oob_rate = float(np.mean([1.0 if r.oob else 0.0 for r in results])) if results else 0.0
    steps_mean = float(np.mean([r.steps for r in results])) if results else 0.0
    v_mean = float(np.mean([r.v_mean for r in results])) if results else 0.0
    rmse_mean = float(np.mean([r.rmse for r in results])) if results else 0.0

    du_samples = [float(v) for r in results for v in [r.du_mean] if math.isfinite(float(v))]
    kcm_samples = [float(v) for r in results for v in [r.kcm_mean] if math.isfinite(float(v))]
    jerk_samples = [float(v) for r in results for v in [r.jerk_mean] if math.isfinite(float(v))]
    ang_jerk_samples = [float(v) for r in results for v in [r.ang_jerk_mean] if math.isfinite(float(v))]

    return Summary(
        episodes=int(episodes),
        success_rate=float(success_rate),
        oob_rate=float(oob_rate),
        steps_mean=float(steps_mean),
        v_mean=float(v_mean),
        rmse_mean=float(rmse_mean),
        du_mean=float(np.mean(du_samples)) if du_samples else 0.0,
        du_p90=float(_percentile([r.du_p90 for r in results], 90.0)),
        kcm_mean=float(np.mean(kcm_samples)) if kcm_samples else 0.0,
        kcm_p90=float(_percentile([r.kcm_p90 for r in results], 90.0)),
        jerk_mean=float(np.mean(jerk_samples)) if jerk_samples else 0.0,
        ang_jerk_mean=float(np.mean(ang_jerk_samples)) if ang_jerk_samples else 0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="P6.1 平顺性报告/验收（Δu 惩罚 + v_target 平滑器）")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "original_configs" / "smoke.yaml",
        help="用于读取环境/约束默认值的 YAML（默认 original_configs/smoke.yaml）。",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=20, help="每条路径评估 episode 数（默认 20）。")
    parser.add_argument("--kp-lat", type=float, default=1.0, help="横向误差 P 控制系数。")
    parser.add_argument("--jitter-v", type=float, default=0.25, help="速度抖动幅度（模拟 chattering）。")
    parser.add_argument("--jitter-theta", type=float, default=0.0, help="转向抖动幅度（默认 0）。")
    parser.add_argument("--max-steps", type=int, default=None, help="每个 episode 的最大步数（默认用 YAML）。")
    args = parser.parse_args()

    if not args.config.exists():
        print(f"[ERROR] missing config: {args.config}")
        raise SystemExit(2)

    cfg = _load_yaml(args.config)
    env_cfg = cfg.get("environment", {}) or {}
    ppo_cfg = cfg.get("ppo", {}) or {}
    path_cfg = cfg.get("path", {}) or {}
    scale = float(path_cfg.get("scale", 10.0))
    num_points = int(path_cfg.get("num_points", 200))
    s_cfg = path_cfg.get("s_shape", {}) if isinstance(path_cfg.get("s_shape", {}), dict) else {}
    amplitude = float(s_cfg.get("amplitude", scale / 2.0))
    periods = float(s_cfg.get("periods", 2.0))

    max_steps = int(args.max_steps) if args.max_steps is not None else int(env_cfg.get("max_steps", 200))
    dt = float(env_cfg.get("interpolation_period", 0.1))
    gamma = float(ppo_cfg.get("gamma", 0.99))
    horizon_steps = float("inf") if gamma >= 1.0 else 1.0 / max(1e-12, 1.0 - gamma)
    horizon_time = float("inf") if gamma >= 1.0 else dt / max(1e-12, 1.0 - gamma)
    print(
        f"[RUN] seed={int(args.seed)} dt={dt} gamma={gamma:.6f} H_steps≈{horizon_steps:.1f} H_time≈{horizon_time:.3f} "
        f"episodes={int(args.episodes)} kp_lat={float(args.kp_lat)} "
        f"jitter_v={float(args.jitter_v)} jitter_theta={float(args.jitter_theta)}"
    )

    paths = {
        "line": build_line(
            length=scale,
            num_points=num_points,
            angle=float(path_cfg.get("line", {}).get("angle", 0.0))
            if isinstance(path_cfg.get("line", {}), dict)
            else 0.0,
        ),
        "square": build_open_square(side=scale, num_points=num_points),
        "s_shape": build_s_shape(scale=scale, num_points=num_points, amplitude=amplitude, periods=periods),
    }

    # A/B：P6.1 功能开关
    cfg_a = {"du_enabled": False, "v_target_smoother_enabled": False}
    cfg_b = {"du_enabled": True, "v_target_smoother_enabled": True}

    all_ok = True
    metrics_a: Dict[str, Summary] = {}
    metrics_b: Dict[str, Summary] = {}

    for name, pm in paths.items():
        env_a = _build_env(cfg, pm, p6_1_override=cfg_a)
        env_b = _build_env(cfg, pm, p6_1_override=cfg_b)

        m_a = _run_eval(
            env_a,
            episodes=int(args.episodes),
            seed=int(args.seed),
            kp_lat=float(args.kp_lat),
            jitter_v=float(args.jitter_v),
            jitter_theta=float(args.jitter_theta),
            max_steps=int(max_steps),
        )
        m_b = _run_eval(
            env_b,
            episodes=int(args.episodes),
            seed=int(args.seed),
            kp_lat=float(args.kp_lat),
            jitter_v=float(args.jitter_v),
            jitter_theta=float(args.jitter_theta),
            max_steps=int(max_steps),
        )

        metrics_a[name] = m_a
        metrics_b[name] = m_b

        print(
            f"[EVAL] path={name} mode=A(off) episodes={m_a.episodes} "
            f"success_rate={m_a.success_rate:.3f} oob_rate={m_a.oob_rate:.3f} steps_mean={m_a.steps_mean:.1f} v_mean={m_a.v_mean:.3f} "
            f"du_mean={m_a.du_mean:.4f} du_p90={m_a.du_p90:.4f} kcm_mean={m_a.kcm_mean:.4f} kcm_p90={m_a.kcm_p90:.4f} "
            f"jerk_mean={m_a.jerk_mean:.4f} angular_jerk_mean={m_a.ang_jerk_mean:.4f}"
        )
        print(
            f"[EVAL] path={name} mode=B(on)  episodes={m_b.episodes} "
            f"success_rate={m_b.success_rate:.3f} oob_rate={m_b.oob_rate:.3f} steps_mean={m_b.steps_mean:.1f} v_mean={m_b.v_mean:.3f} "
            f"du_mean={m_b.du_mean:.4f} du_p90={m_b.du_p90:.4f} kcm_mean={m_b.kcm_mean:.4f} kcm_p90={m_b.kcm_p90:.4f} "
            f"jerk_mean={m_b.jerk_mean:.4f} angular_jerk_mean={m_b.ang_jerk_mean:.4f}"
        )

        if m_b.success_rate + 1e-12 < m_a.success_rate:
            all_ok = False
            print(f"[FAIL] stability: {name} B success_rate decreased vs A")
        if m_b.oob_rate - 1e-12 > m_a.oob_rate:
            all_ok = False
            print(f"[FAIL] stability: {name} B oob_rate increased vs A")
        if m_b.success_rate < 0.95:
            all_ok = False
            print(f"[FAIL] quick_eval: {name} B success_rate < 0.95")
        if m_b.oob_rate > 0.10:
            all_ok = False
            print(f"[FAIL] quick_eval: {name} B oob_rate > 0.10")

    # 平顺性验收（square 作为主场景）
    square_a = metrics_a.get("square", None)
    square_b = metrics_b.get("square", None)
    if square_a is None or square_b is None:
        all_ok = False
        print("[FAIL] missing square metrics")
    else:
        improved = 0
        if square_b.kcm_mean <= square_a.kcm_mean * 0.90:
            improved += 1
        if square_b.ang_jerk_mean <= square_a.ang_jerk_mean * 0.90:
            improved += 1
        if square_b.jerk_mean <= square_a.jerk_mean * 0.90:
            improved += 1
        if square_b.du_mean <= square_a.du_mean * 0.90:
            improved += 1
        if improved < 2:
            all_ok = False
            print(f"[FAIL] smoothness: improved metrics < 2 (got {improved})")

    # 仍然快（硬指标）：line v_mean 不明显下降；square steps_mean 不明显上升
    line_a = metrics_a.get("line", None)
    line_b = metrics_b.get("line", None)
    if line_a is not None and line_b is not None:
        if line_b.v_mean + 1e-12 < 0.95 * line_a.v_mean:
            all_ok = False
            print(f"[FAIL] speed: line v_mean dropped too much (A={line_a.v_mean:.3f}, B={line_b.v_mean:.3f})")

    if square_a is not None and square_b is not None:
        if square_b.steps_mean > square_a.steps_mean + 5.0:
            all_ok = False
            print(f"[FAIL] speed: square steps_mean increased too much (A={square_a.steps_mean:.1f}, B={square_b.steps_mean:.1f})")

    if all_ok:
        print("[PASS] P6.1 smoothness acceptance passed.")
        raise SystemExit(0)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
