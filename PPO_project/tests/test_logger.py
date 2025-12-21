from __future__ import annotations

import threading
import time
from pathlib import Path
import sys
import os

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.utils.logger import CSVLogger, DataLogger, ExperimentManager

CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "default.yaml"


def test_csvlogger_atomic_write(tmp_path: Path, monkeypatch) -> None:
    """模拟高频写与读，验证CSVLogger的原子写能力。"""
    monkeypatch.setenv("EXPERIMENT_DIR", str(tmp_path))
    manager = ExperimentManager("logger_test", CONFIG_PATH)
    log_path = manager.logs_dir / "training_logger_test.csv"
    logger = CSVLogger(log_path, ["episode_idx", "reward", "actor_loss", "critic_loss", "wall_time"])

    total_steps = 25
    errors: list[Exception] = []
    last_episode = -1
    stop_event = threading.Event()

    def writer() -> None:
        start = time.perf_counter()
        for step in range(total_steps):
            logger.log_step(
                episode_idx=step,
                reward=float(step),
                actor_loss=0.1 * step,
                critic_loss=0.2 * step,
                wall_time=time.perf_counter() - start,
            )
            time.sleep(0.005)
        stop_event.set()

    def reader() -> None:
        nonlocal last_episode
        while not stop_event.is_set():
            try:
                if log_path.exists() and log_path.stat().st_size > 0:
                    df = pd.read_csv(log_path)
                    if not df.empty:
                        last_episode = int(df.iloc[-1]["episode_idx"])
            except Exception as exc:
                errors.append(exc)
            time.sleep(0.005)

    writer_thread = threading.Thread(target=writer)
    reader_thread = threading.Thread(target=reader)
    writer_thread.start()
    reader_thread.start()
    writer_thread.join()
    stop_event.set()
    reader_thread.join()

    # 最后一读，确保拿到最新行
    if log_path.exists():
        df = pd.read_csv(log_path)
        if not df.empty:
            last_episode = int(df.iloc[-1]["episode_idx"])

    assert not errors, f"读取过程中出现错误: {errors}"
    assert last_episode == total_steps - 1, f"未读取到最新步: {last_episode}"


def test_datalogger_atomic_write(tmp_path: Path) -> None:
    """模拟推理日志的并发读写，验证DataLogger的原子性。"""
    log_path = tmp_path / "datalogger_test.csv"
    logger = DataLogger(log_dir=tmp_path, filename="datalogger_test.csv")

    errors: list[Exception] = []
    last_timestamp = 0.0
    stop_event = threading.Event()
    total_steps = 20

    def writer() -> None:
        for _ in range(total_steps):
            logger.log_step(
                dt=0.01,
                position=[1.0, 1.0],
                reference_point=[0.0, 0.0],
                velocity=0.5,
                acceleration=0.1,
                jerk=0.05,
                contour_error=0.01,
                kcm_intervention=0.0,
                reward_components={"r": 1.0},
            )
            time.sleep(0.005)
        stop_event.set()

    def reader() -> None:
        nonlocal last_timestamp
        while not stop_event.is_set():
            try:
                if log_path.exists() and log_path.stat().st_size > 0:
                    df = pd.read_csv(log_path)
                    if not df.empty:
                        last_timestamp = float(df.iloc[-1]["timestamp"])
            except Exception as exc:
                errors.append(exc)
            time.sleep(0.005)

    writer_thread = threading.Thread(target=writer)
    reader_thread = threading.Thread(target=reader)
    writer_thread.start()
    reader_thread.start()
    writer_thread.join()
    stop_event.set()
    reader_thread.join()

    if log_path.exists():
        df = pd.read_csv(log_path)
        if not df.empty:
            last_timestamp = float(df.iloc[-1]["timestamp"])

    assert not errors, f"读取过程中出现错误: {errors}"
    assert last_timestamp >= 0.01 * total_steps, f"未读取到最新时间戳: {last_timestamp}"
