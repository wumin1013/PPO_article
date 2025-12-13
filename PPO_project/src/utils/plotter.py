"""绘图与字体配置及实时可视化工具。"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def configure_chinese_font() -> None:
    """配置中文字体，避免图例乱码。"""
    try:
        system_fonts = ["Microsoft YaHei", "SimHei", "FangSong", "STSong"]
        linux_fonts = ["WenQuanYi Micro Hei", "AR PL UMing CN"]
        font_list = list(dict.fromkeys(system_fonts + linux_fonts))
        mpl.rcParams["font.sans-serif"] = font_list + mpl.rcParams["font.sans-serif"]
        mpl.rcParams["axes.unicode_minus"] = False
        test_font = mpl.font_manager.FontProperties(family=font_list)  # type: ignore[attr-defined]
        if not test_font.get_name():
            raise RuntimeError("字体配置失败")
    except Exception as exc:
        print(f"字体配置警告: {exc}")
        print("将使用默认字体显示，中文可能显示为方块")


def _clean_path(path):
    if path is None:
        return np.empty((0, 2))
    cleaned = [p for p in path if p is not None and not np.isnan(p).any()]
    return np.array(cleaned) if cleaned else np.empty((0, 2))


def _get_boundary(env, key: str):
    """Return Pl/Pr with fallbacks for legacy env cache."""
    if hasattr(env, key):
        return getattr(env, key)
    cache = getattr(env, "cache", {})
    return cache.get(key, [])


class TrajectoryPlotter:
    """交互式轨迹绘图器，用于训练时的实时刷新。"""

    def __init__(self, title: str = "Trajectory Tracking", pause: float = 0.01) -> None:
        configure_chinese_font()
        plt.ion()
        self.pause = pause
        self.fig, self.ax = plt.subplots(figsize=(8, 6), dpi=100)
        self.ref_line, = self.ax.plot([], [], "k--", linewidth=1.5, label="Reference Path")
        self.traj_line, = self.ax.plot([], [], "r-", linewidth=1.5, label="Actual Trajectory")
        self.left_line, = self.ax.plot([], [], "g--", linewidth=1.2, alpha=0.6, label="Left Boundary")
        self.right_line, = self.ax.plot([], [], "b--", linewidth=1.2, alpha=0.6, label="Right Boundary")
        self.sample_scatter = self.ax.scatter([], [], c="purple", s=25, alpha=0.6, edgecolor="white", label="Samples")
        self.ax.set_xlabel("X Coordinate")
        self.ax.set_ylabel("Y Coordinate")
        self.ax.set_title(title)
        self.ax.grid(True, linestyle=":", alpha=0.4)
        self.ax.legend(loc="upper left", framealpha=0.9)

    def _set_line(self, line, data: np.ndarray) -> None:
        if data.size == 0:
            line.set_data([], [])
        else:
            line.set_data(data[:, 0], data[:, 1])

    def _set_samples(self, trajectory: np.ndarray) -> None:
        if trajectory.size == 0:
            self.sample_scatter.set_offsets(np.empty((0, 2)))
            return
        step = max(1, len(trajectory) // 200)
        sampled = trajectory[::step]
        self.sample_scatter.set_offsets(sampled[:, :2])

    def update(
        self,
        reference_path: Optional[Iterable[Sequence[float]]] = None,
        trajectory: Optional[Iterable[Sequence[float]]] = None,
        left_boundary: Optional[Iterable[Sequence[float]]] = None,
        right_boundary: Optional[Iterable[Sequence[float]]] = None,
    ) -> None:
        """非阻塞刷新绘图，避免卡死。"""
        ref = _clean_path(reference_path)
        traj = _clean_path(trajectory)
        left = _clean_path(left_boundary)
        right = _clean_path(right_boundary)

        self._set_line(self.ref_line, ref)
        self._set_line(self.traj_line, traj)
        self._set_line(self.left_line, left)
        self._set_line(self.right_line, right)
        self._set_samples(traj)

        if traj.size > 0 or ref.size > 0:
            self.ax.relim()
            self.ax.autoscale_view()

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(self.pause)

    def close(self) -> None:
        plt.close(self.fig)


def visualize_final_path(env) -> None:
    """绘制参考路径与实际轨迹。"""
    plt.figure(figsize=(10, 6), dpi=100)

    pm = _clean_path(env.Pm)
    if pm.size == 0:
        print("warning: empty reference path, skip visualization.")
        return
    plt.plot(pm[:, 0], pm[:, 1], "k--", linewidth=2.5, label="Reference Path (Pm)")
    plt.scatter(pm[:, 0], pm[:, 1], c="black", marker="*", s=150, edgecolor="gold", zorder=3)

    pl = _clean_path(_get_boundary(env, "Pl"))
    pr = _clean_path(_get_boundary(env, "Pr"))
    if pl.size > 0 and pr.size > 0:
        plt.plot(pl[:, 0], pl[:, 1], "g--", linewidth=1.8, label="Left Boundary (Pl)", alpha=0.7)
        plt.plot(pr[:, 0], pr[:, 1], "b--", linewidth=1.8, label="Right Boundary (Pr)", alpha=0.7)

    pt = np.array(env.trajectory)
    plt.plot(pt[:, 0], pt[:, 1], "r-", linewidth=1.5, label="Actual Trajectory (Pt)")
    plt.scatter(
        pt[::20, 0],
        pt[::20, 1],
        c="purple",
        s=40,
        alpha=0.6,
        edgecolor="white",
        label="Sampled Points",
        zorder=2,
    )

    plt.annotate(
        f"Start\n({pm[0,0]:.1f}, {pm[0,1]:.1f})",
        xy=pm[0],
        xytext=(-20, -30),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="gray", alpha=0.6),
    )

    if len(pm) > 1:
        plt.annotate(
            f"End\n({pm[-1,0]:.1f}, {pm[-1,1]:.1f})",
            xy=pm[-1],
            xytext=(-40, 20),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="gray", alpha=0.6),
        )

    param_text = (
        f"ε = {env.epsilon:.2f}\n"
        f"MAX_VEL = {env.MAX_VEL:.1f}\n"
        f"Δt = {env.interpolation_period:.2f}s\n"
        f"Steps = {len(pt)}"
    )
    plt.gcf().text(
        0.88,
        0.85,
        param_text,
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.axis("equal")
    plt.xlabel("X Coordinate", fontsize=12)
    plt.ylabel("Y Coordinate", fontsize=12)
    plt.title("Final Trajectory Tracking Performance", fontsize=14, pad=20)
    plt.legend(loc="upper left", framealpha=0.9)
    plt.grid(True, color="gray", linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.show()


__all__ = ["configure_chinese_font", "visualize_final_path", "TrajectoryPlotter"]
