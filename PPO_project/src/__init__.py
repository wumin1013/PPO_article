"""源代码包。

保持轻量：避免在 import 阶段就引入重依赖（如 scipy），通过惰性导入按需加载子模块。
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = ["algorithms", "environment", "utils"]


def __getattr__(name: str) -> Any:  # pragma: no cover
    if name == "algorithms":
        try:
            module = importlib.import_module(f"{__name__}.algorithms")
        except Exception:
            globals()["algorithms"] = None
            return None
        globals()["algorithms"] = module
        return module

    if name in {"environment", "utils"}:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
