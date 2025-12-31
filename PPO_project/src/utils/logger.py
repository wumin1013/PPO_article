"""
Logging utilities.
- ExperimentManager: create isolated experiment folders and config snapshots.
- CSVLogger: atomic CSV appends for training telemetry.
- DataLogger: inference-time trajectory logger.
"""
from __future__ import annotations

import csv
import json
import os
import time
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ExperimentManager:
    """Manage experiment directory structure and config snapshots."""

    def __init__(
        self,
        category: str,
        config_path: Path | str,
        experiment_dir: Path | str | None = None,
        config_data: Optional[Mapping] = None,
    ) -> None:
        if experiment_dir is not None:
            base_dir = Path(experiment_dir)
        else:
            override_root = os.environ.get("EXPERIMENT_DIR")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if override_root:
                base_dir = Path(override_root) / category / timestamp
            else:
                base_dir = PROJECT_ROOT / "saved_models" / category / timestamp

        self.experiment_dir = base_dir
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self.logs_dir = self.experiment_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.models_dir = self.experiment_dir / "checkpoints"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.config_copy_path = self.experiment_dir / "config.yaml"
        if config_data is not None:
            self._write_config(config_data)
        else:
            self._copy_config(config_path)

    def _copy_config(self, config_path: Path | str) -> None:
        source = Path(config_path)
        if not source.exists():
            return
        if self.config_copy_path.exists():
            return
        if source.resolve() == self.config_copy_path.resolve():
            return
        shutil.copyfile(source, self.config_copy_path)

    def _write_config(self, config_data: Mapping) -> None:
        """Persist the effective configuration snapshot for reproducibility."""
        with self.config_copy_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(config_data, f, allow_unicode=True)

    def create_logger(self, filename: str, fieldnames: Sequence[str]) -> "CSVLogger":
        return CSVLogger(self.logs_dir / filename, fieldnames)


class CSVLogger:
    """
    Atomic CSV logger.

    Each call opens the file, writes, flushes, and closes immediately to avoid
    descriptor contention with concurrent readers (e.g., Streamlit/Excel).
    """

    def __init__(self, path: Path | str, fieldnames: Sequence[str]) -> None:
        self.path = Path(path)
        self.fieldnames = list(fieldnames)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_header()

    def _ensure_header(self) -> None:
        if self.path.exists() and self.path.stat().st_size > 0:
            return
        self._write_row({}, write_header=True)

    def log(self, row: Mapping[str, object]) -> None:
        filtered_row = {name: row.get(name, "") for name in self.fieldnames}
        self._write_row(filtered_row)

    def _write_row(self, row: Mapping[str, object], write_header: bool = False) -> None:
        """写入一行，PermissionError 自动重试，缓解同步盘/杀毒占用。"""
        attempts = 3
        for attempt in range(attempts):
            try:
                mode = "w" if write_header else "a"
                with self.path.open(mode, newline="", encoding="utf-8") as file:
                    writer = csv.DictWriter(file, fieldnames=self.fieldnames)
                    if write_header:
                        writer.writeheader()
                    if row:
                        writer.writerow(row)
                    file.flush()
                return
            except PermissionError:
                if attempt == attempts - 1:
                    # Sync/AV may lock the file; skip this row to keep training running.
                    return
                time.sleep(0.1)

    def log_step(self, **row: object) -> None:
        self.log(row)


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
        self.log_dir = Path(log_dir) if log_dir else PROJECT_ROOT / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / filename
        self._ensure_header()
        self.current_time = 0.0

    def close(self) -> None:
        # kept for context manager compatibility
        return

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
        with self.log_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self.required_columns)
            writer.writerow(row)
            file.flush()

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

    def _ensure_header(self) -> None:
        if self.log_path.exists() and self.log_path.stat().st_size > 0:
            return
        with self.log_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self.required_columns)
            writer.writeheader()
            file.flush()


__all__ = ["CSVLogger", "DataLogger", "ExperimentManager"]
