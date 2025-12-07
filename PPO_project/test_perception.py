"""
前瞻特征自检脚本：
1) 将路径设置为直线后10mm左转；
2) 打印并绘制局部坐标系下的预瞄点 (x_body, y_body, dκ/ds)。
"""

import numpy as np
import matplotlib.pyplot as plt
import torch

from PPO最终版 import Env


def build_turn_env() -> Env:
    """构造包含短距离左转的环境，便于检验坐标变换。"""
    path_points = [
        np.array([0.0, 0.0]),
        np.array([0.01, 0.0]),   # 10mm 直线
        np.array([0.01, 0.02]),  # 左转上行
    ]
    return Env(
        device=torch.device("cpu"),
        epsilon=0.5,
        interpolation_period=0.1,
        MAX_VEL=1.0,
        MAX_ACC=2.0,
        MAX_JERK=3.0,
        MAX_ANG_VEL=1.5,
        MAX_ANG_ACC=3.0,
        MAX_ANG_JERK=5.0,
        Pm=path_points,
        max_steps=50,
        lookahead_points=3,
    )


def main(show_plot: bool = True) -> None:
    env = build_turn_env()
    _ = env.reset()

    raw_features = env._compute_lookahead_features()
    lookahead = raw_features.reshape(-1, 3)

    print("局部坐标系预瞄特征 [x_body, y_body, dκ/ds]:")
    for i, (x_body, y_body, kappa_rate) in enumerate(lookahead, start=1):
        print(f"  P{i}: x={x_body:.4f}, y={y_body:.4f}, dκ/ds={kappa_rate:.6f}")

    # 断言左转预瞄点在车体坐标系的Y正方向偏转
    assert np.any(lookahead[:, 1] > 0), "左转预瞄点未在Y正方向偏转"
    assert lookahead[-1, 1] > 0, "末端预瞄点未体现左转偏移"

    if show_plot:
        plt.figure(figsize=(5, 4))
        plt.axhline(0, color="gray", linestyle="--", linewidth=1)
        plt.axvline(0, color="gray", linestyle="--", linewidth=1)
        plt.scatter(lookahead[:, 0], lookahead[:, 1], c="tab:blue", label="lookahead (body frame)")
        for i, (x_body, y_body, _) in enumerate(lookahead, start=1):
            plt.text(x_body, y_body, f"P{i}", fontsize=9)
        plt.xlabel("X_body (forward)")
        plt.ylabel("Y_body (left)")
        plt.legend()
        plt.title("Lookahead points in body frame")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
