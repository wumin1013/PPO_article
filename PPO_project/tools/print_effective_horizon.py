"""P4.0 工具：打印 dt/gamma 的近似有效视界（time horizon）。

H_steps ≈ 1 / (1 - gamma)
H_time  ≈ dt / (1 - gamma)
并提示：若 dt 改变，可用 gamma_new = gamma_old ** (dt_new / dt_old) 保持“每单位时间”折扣一致。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    parser = argparse.ArgumentParser(description="打印 dt/gamma 有效视界 (P4.0)")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "original_configs" / "smoke.yaml",
        help="YAML 配置路径（默认 original_configs/smoke.yaml）。",
    )
    parser.add_argument(
        "--dt-new",
        type=float,
        default=None,
        help="可选：若计划将 dt 改为该值，输出保持时间折扣一致的 gamma_new 建议。",
    )
    args = parser.parse_args()

    if not args.config.exists():
        print(f"[ERROR] missing config: {args.config}")
        raise SystemExit(2)

    cfg = _load_yaml(args.config)
    env_cfg = cfg.get("environment", {}) or {}
    ppo_cfg = cfg.get("ppo", {}) or {}

    dt = float(env_cfg.get("interpolation_period", 0.0))
    gamma = ppo_cfg.get("gamma", None)
    gamma_f = float(gamma) if gamma is not None else None

    if gamma_f is None or not (0.0 < gamma_f < 1.0) or dt <= 0.0:
        print(f"[HORIZON] dt={dt} gamma={gamma} (no finite horizon or missing values)")
        print("[HORIZON] 提醒: 若要保持时间折扣一致，可用 gamma_new = gamma_old ** (dt_new / dt_old)")
        raise SystemExit(0)

    denom = max(1e-12, 1.0 - gamma_f)
    h_steps = 1.0 / denom
    h_time = dt / denom
    print(f"[HORIZON] dt={dt} gamma={gamma_f:.6f} H_steps≈{h_steps:.1f} H_time≈{h_time:.4f}")
    print("[HORIZON] dt 变化保持时间折扣一致：gamma_new = gamma_old ** (dt_new / dt_old)")

    if args.dt_new is not None and float(args.dt_new) > 0.0:
        dt_new = float(args.dt_new)
        gamma_new = gamma_f ** (dt_new / dt)
        print(f"[HORIZON] dt_old={dt} dt_new={dt_new} -> gamma_new≈{gamma_new:.6f}")

    raise SystemExit(0)


if __name__ == "__main__":
    main()

