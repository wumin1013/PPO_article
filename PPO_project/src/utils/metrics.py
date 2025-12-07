"""
论文指标统计工具。
"""
import numpy as np


class PaperMetrics:
    """收集论文实验相关指标。"""

    def __init__(self):
        self.reset()

    def reset(self):
        """重置episode级指标缓存。"""
        self.errors = []
        self.jerks = []
        self.velocities = []
        self.kcm_interventions = []

    def update(self, contour_error, jerk, velocity, kcm_intervention=0.0):
        """在每步后更新指标。"""
        self.errors.append(contour_error)
        self.jerks.append(abs(jerk))
        self.velocities.append(velocity)
        self.kcm_interventions.append(kcm_intervention)

    def compute(self):
        """计算一个episode结束后的统计量。"""
        if len(self.errors) == 0:
            return {
                "rmse_error": 0.0,
                "mean_jerk": 0.0,
                "roughness_proxy": 0.0,
                "mean_velocity": 0.0,
                "max_error": 0.0,
                "mean_kcm_intervention": 0.0,
                "steps": 0,
            }

        rmse_error = np.sqrt(np.mean(np.array(self.errors) ** 2))
        mean_jerk = np.mean(self.jerks)
        if len(self.jerks) > 1:
            jerk_diff = np.diff(self.jerks)
            roughness_proxy = np.sum(np.abs(jerk_diff))
        else:
            roughness_proxy = 0.0

        mean_velocity = np.mean(self.velocities)
        max_error = np.max(self.errors)
        mean_kcm_intervention = np.mean(self.kcm_interventions)

        return {
            "rmse_error": rmse_error,
            "mean_jerk": mean_jerk,
            "roughness_proxy": roughness_proxy,
            "mean_velocity": mean_velocity,
            "max_error": max_error,
            "mean_kcm_intervention": mean_kcm_intervention,
            "steps": len(self.errors),
        }


__all__ = ["PaperMetrics"]
