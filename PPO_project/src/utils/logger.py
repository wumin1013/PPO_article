"""
Data logger for inference runs.
Writes per-step state to CSV for paper figures and tables.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


class DataLogger:
    """CSV logger that records detailed states during inference."""

    required_columns = [
        "timestamp",
        "pos_x",
        "pos_y",
        "ref_x",
        "ref_y",
        "velocity",
        "acceleration",
        "jerk",
        "contour_error",
        "kcm_intervention",
        "reward_components",
    ]

    def __init__(
        self,
        log_dir: Optional[Path | str] = None,
        filename: str = "experiment_results.csv",
    ) -> None:
        project_root = Path(__file__).resolve().parents[2]
        self.log_dir = Path(log_dir) if log_dir else project_root / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / filename
        self._file = self.log_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.required_columns)
        self._writer.writeheader()
        self.current_time = 0.0

    def close(self) -> None:
        if not self._file.closed:
            self._file.flush()
            self._file.close()

    def log_step(
        self,
        dt: float,
        position: Sequence[float],
        reference_point: Sequence[float],
        velocity: float,
        acceleration: float,
        jerk: float,
        contour_error: float,
        kcm_intervention: float,
        reward_components: Optional[dict] = None,
    ) -> None:
        self.current_time += float(dt)
        ref_x, ref_y = reference_point
        pos_x, pos_y = position
        row = {
            "timestamp": round(self.current_time, 6),
            "pos_x": float(pos_x),
            "pos_y": float(pos_y),
            "ref_x": float(ref_x),
            "ref_y": float(ref_y),
            "velocity": float(velocity),
            "acceleration": float(acceleration),
            "jerk": float(jerk),
            "contour_error": float(contour_error),
            "kcm_intervention": float(kcm_intervention),
            "reward_components": json.dumps(reward_components or {}),
        }
        self._writer.writerow(row)

    @staticmethod
    def project_to_path(
        position: Sequence[float],
        path_points: Iterable[Sequence[float]],
        segment_index: int,
    ) -> Tuple[float, float]:
        """Project current position to the given segment to obtain ref point."""
        points = list(path_points)
        n = len(points)
        if n < 2:
            return float(position[0]), float(position[1])
        idx = max(0, min(segment_index, n - 2))
        p1 = np.asarray(points[idx], dtype=float)
        p2 = np.asarray(points[(idx + 1) % n], dtype=float)
        seg_vec = p2 - p1
        denom = float(np.dot(seg_vec, seg_vec))
        if denom < 1e-9:
            return float(p1[0]), float(p1[1])
        t = float(np.dot(np.asarray(position, dtype=float) - p1, seg_vec) / denom)
        t = float(np.clip(t, 0.0, 1.0))
        proj = p1 + t * seg_vec
        return float(proj[0]), float(proj[1])

    def __enter__(self) -> "DataLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = ["DataLogger"]
