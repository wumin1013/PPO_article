"""P2.5 偏移线/走廊自检：Pl/Pr 无自交、quad 无自交/退化、左右语义一致，并输出可视化 PNG。"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# 避免触发 `src/__init__.py` 的副作用（会导入 env 并依赖 rtree）。
sys.path.insert(0, str(ROOT / "src"))

try:
    import matplotlib.pyplot as plt  # noqa: E402
    import yaml  # noqa: E402

    from utils.geometry import (  # noqa: E402
        count_polyline_self_intersections,
        generate_offset_paths,
        left_normal,
        quad_is_degenerate,
        quad_self_intersects,
        right_normal,
    )
    from utils.path_generator import get_path_by_name  # noqa: E402
except ImportError as exc:  # pragma: no cover
    print(
        "[ERROR] 依赖缺失："
        f"{exc}. 请先安装依赖，例如: python -m pip install -r PPO_project/requirements.txt"
    )
    raise


@dataclass(frozen=True)
class CheckResult:
    path_name: str
    closed: bool
    pl_self_intersections: int
    pr_self_intersections: int
    quad_bad_count: int
    semantic_fail_indices: List[int]
    quad_count: int
    pm_core_len: int

    @property
    def ok(self) -> bool:
        return (
            self.pl_self_intersections == 0
            and self.pr_self_intersections == 0
            and self.quad_bad_count == 0
            and len(self.semantic_fail_indices) == 0
        )


def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v, dtype=float)
    return v / n


def _resolve_core(points: Sequence[Sequence[float]], eps: float = 1e-6) -> Tuple[List[np.ndarray], bool]:
    pts = [np.asarray(p, dtype=float) for p in points]
    if len(pts) > 2 and np.allclose(pts[0], pts[-1], atol=eps):
        return pts[:-1], True
    return pts, False


def _load_config(config_path: Path) -> Dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _pm_from_config(cfg: Dict) -> List[np.ndarray]:
    path_cfg = cfg.get("path", {})
    if path_cfg.get("type") == "waypoints":
        return [np.asarray(wp, dtype=float) for wp in path_cfg.get("waypoints", [])]
    path_type = path_cfg.get("type", "line")
    scale = float(path_cfg.get("scale", 10.0))
    num_points = int(path_cfg.get("num_points", 200))
    kwargs = path_cfg.get(path_type, {}) if isinstance(path_cfg.get(path_type, {}), dict) else {}
    return get_path_by_name(path_type, scale=scale, num_points=num_points, **kwargs)


def _build_quads(
    pm_core: Sequence[np.ndarray],
    pl_core: Sequence[np.ndarray],
    pr_core: Sequence[np.ndarray],
    closed: bool,
) -> List[List[np.ndarray]]:
    quads: List[List[np.ndarray]] = []
    if not pm_core:
        return quads
    if not closed:
        for i in range(len(pm_core) - 1):
            quads.append([pl_core[i], pl_core[i + 1], pr_core[i + 1], pr_core[i]])
        return quads
    m = len(pm_core)
    for i in range(m):
        j = (i + 1) % m
        quads.append([pl_core[i], pl_core[j], pr_core[j], pr_core[i]])
    return quads


def _check_semantics(
    pm_core: Sequence[np.ndarray],
    pl_core: Sequence[np.ndarray],
    pr_core: Sequence[np.ndarray],
    closed: bool,
    tol: float = 1e-9,
) -> List[int]:
    fails: List[int] = []
    n = len(pm_core)
    if n == 0:
        return fails
    for i in range(n):
        if closed:
            j = (i + 1) % n
            t = _unit(pm_core[j] - pm_core[i])
        else:
            if i == 0:
                t = _unit(pm_core[1] - pm_core[0]) if n > 1 else np.array([1.0, 0.0])
            elif i == n - 1:
                t = _unit(pm_core[-1] - pm_core[-2]) if n > 1 else np.array([1.0, 0.0])
            else:
                t = _unit(pm_core[i + 1] - pm_core[i - 1])
        if float(np.dot(pl_core[i] - pm_core[i], left_normal(t))) <= tol:
            fails.append(i)
            continue
        if float(np.dot(pr_core[i] - pm_core[i], right_normal(t))) <= tol:
            fails.append(i)
            continue
    return sorted(set(fails))


def check_path(
    path_name: str,
    pm: Sequence[Sequence[float]],
    half_width: float,
    out_path: Path,
) -> CheckResult:
    pm_core, closed = _resolve_core(pm)
    pl, pr = generate_offset_paths(pm, half_width, closed=closed)
    pl_core, _ = _resolve_core([p for p in pl if p is not None]) if closed else ([p for p in pl if p is not None], False)
    pr_core, _ = _resolve_core([p for p in pr if p is not None]) if closed else ([p for p in pr if p is not None], False)

    pl_self = count_polyline_self_intersections(pl_core, closed=closed)
    pr_self = count_polyline_self_intersections(pr_core, closed=closed)

    quads = _build_quads(pm_core, pl_core, pr_core, closed=closed)
    quad_bad = 0
    for quad in quads:
        if quad_is_degenerate(quad) or quad_self_intersects(quad):
            quad_bad += 1

    semantic_fails = _check_semantics(pm_core, pl_core, pr_core, closed=closed)

    _plot_debug(
        path_name=path_name,
        pm_core=pm_core,
        pl_core=pl_core,
        pr_core=pr_core,
        closed=closed,
        out_path=out_path,
        pl_self=pl_self,
        pr_self=pr_self,
        quad_bad=quad_bad,
        semantic_fails=semantic_fails,
    )

    return CheckResult(
        path_name=path_name,
        closed=closed,
        pl_self_intersections=pl_self,
        pr_self_intersections=pr_self,
        quad_bad_count=quad_bad,
        semantic_fail_indices=semantic_fails,
        quad_count=len(quads),
        pm_core_len=len(pm_core),
    )


def _plot_debug(
    path_name: str,
    pm_core: Sequence[np.ndarray],
    pl_core: Sequence[np.ndarray],
    pr_core: Sequence[np.ndarray],
    closed: bool,
    out_path: Path,
    pl_self: int,
    pr_self: int,
    quad_bad: int,
    semantic_fails: Sequence[int],
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)

    def plot_polyline(pts: Sequence[np.ndarray], color: str, label: str, linestyle: str = "-") -> None:
        if not pts:
            return
        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]
        if closed and len(pts) > 2:
            xs = xs + [xs[0]]
            ys = ys + [ys[0]]
        ax.plot(xs, ys, linestyle, color=color, linewidth=1.8, label=label)

    plot_polyline(pm_core, color="k", label="Pm", linestyle="-")
    plot_polyline(pl_core, color="g", label="Pl", linestyle="--")
    plot_polyline(pr_core, color="b", label="Pr", linestyle="--")

    if semantic_fails:
        pts = [pm_core[i] for i in semantic_fails if 0 <= i < len(pm_core)]
        if pts:
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], c="r", s=20, label="semantic_fail")
            for i in semantic_fails[:10]:
                p = pm_core[i]
                ax.annotate(f"{i}", (p[0], p[1]), textcoords="offset points", xytext=(4, 4), fontsize=8, color="r")

    ax.set_title(f"offset_debug: {path_name} | PlSelf={pl_self} PrSelf={pr_self} QuadBad={quad_bad} SemFail={len(semantic_fails)}")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="P2.5 偏移线/走廊自检（生成 PNG + 退出码）")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "tools" / "offset_debug_out",
        help="输出目录（默认：PPO_project/tools/offset_debug_out）",
    )
    parser.add_argument(
        "--configs",
        type=Path,
        default=ROOT / "configs",
        help="配置目录（默认：PPO_project/configs，用于读取 epsilon/2 作为默认半宽）",
    )
    parser.add_argument(
        "--half-width",
        type=float,
        default=None,
        help="单侧偏移距离 d（不指定则从 config 的 environment.epsilon/2 推导，或回退 0.5）",
    )
    parser.add_argument(
        "--include-sharp-angle",
        action="store_true",
        help="额外加入包含锐角拐点的折线用例（用于边界情况回归）。",
    )
    args = parser.parse_args()

    # P2.5 验收路径：line / square / S（这里用 s_shape_bspline 作为 S 路径，避免正弦极高曲率导致的理论必然自交）
    cases = [
        ("line", "line", False),
        ("square", "square", True),
        ("S", "s_shape_bspline", False),
    ]
    if args.include_sharp_angle:
        cases.append(("sharp_angle", "sharp_angle", False))

    default_half_width = 0.75
    train_line = args.configs / "train_line.yaml"
    if train_line.exists():
        try:
            cfg = _load_config(train_line)
            default_half_width = float(cfg.get("environment", {}).get("epsilon", 1.5)) / 2.0
        except Exception:
            default_half_width = 0.75

    all_ok = True
    for label, generator_name, expect_closed in cases:
        pm = get_path_by_name(generator_name, scale=10.0, num_points=200)
        if expect_closed and not np.allclose(pm[0], pm[-1], atol=1e-6):
            pm = list(pm) + [np.asarray(pm[0], dtype=float)]

        half_width = default_half_width if args.half_width is None else float(args.half_width)
        out_png = args.outdir / f"offset_debug_{label}.png"
        result = check_path(label, pm=pm, half_width=half_width, out_path=out_png)

        tag = "PASS" if result.ok else "FAIL"
        print(
            f"[{tag}] path={result.path_name} closed={result.closed} "
            f"A(PlSelf)={result.pl_self_intersections} B(PrSelf)={result.pr_self_intersections} "
            f"C(QuadBad)={result.quad_bad_count} D(SemFail)={len(result.semantic_fail_indices)} "
            f"| quads={result.quad_count} pm_core={result.pm_core_len} png={out_png}"
        )

        if result.path_name == "square":
            expected = result.pm_core_len
            if result.quad_count != expected:
                all_ok = False
                print(f"[FAIL] square quad_count mismatch: got={result.quad_count} expected={expected}")

        all_ok = all_ok and result.ok

    raise SystemExit(0 if all_ok else 2)


if __name__ == "__main__":
    main()
