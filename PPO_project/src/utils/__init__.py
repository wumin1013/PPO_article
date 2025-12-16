"""通用工具模块入口。

避免在 import 阶段就拉起全部工具模块（部分依赖 scipy/plotly 等重依赖），按需惰性导入。
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = ["geometry", "logger", "path_generator", "rl_utils", "checkpoint", "metrics", "plotter"]


def __getattr__(name: str) -> Any:  # pragma: no cover
    if name in set(__all__):
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
