#!/usr/bin/env python3
"""One-click launcher for the Streamlit dashboard."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = ROOT_DIR / "PPO_project"
PYTHON_CMD = ROOT_DIR / "python.cmd"


def resolve_python() -> str:
    """Prefer the repo-provided python.cmd on Windows, otherwise fall back to the current interpreter."""
    if PYTHON_CMD.exists():
        return str(PYTHON_CMD)
    return sys.executable or "python"


def main() -> int:
    python_exe = resolve_python()
    app_path = PROJECT_DIR / "app.py"
    if not app_path.exists():
        print(f"[launcher] 未找到应用入口: {app_path}")
        return 1

    cmd = [
        python_exe,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.headless",
        "false",
    ]

    print("[launcher] 启动 Streamlit 面板...")
    print(f"[launcher] 解释器: {python_exe}")
    print(f"[launcher] 工作目录: {PROJECT_DIR}")
    try:
        result = subprocess.run(cmd, cwd=str(PROJECT_DIR))
    except FileNotFoundError:
        print("[launcher] 未找到可用的 Python 或 streamlit，请检查依赖。")
        return 1
    except KeyboardInterrupt:
        print("[launcher] 已手动取消。")
        return 0

    if result.returncode != 0:
        print(f"[launcher] 运行结束，退出码: {result.returncode}")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
