"""源代码包"""
from . import environment
from . import utils

# 延迟导入 algorithms，避免在未安装 torch 时阻塞轻量脚本（如 parity_check）。
try:
    from . import algorithms  # type: ignore
    __all__ = ['algorithms', 'environment', 'utils']
except Exception:  # pragma: no cover - 仅作降级
    algorithms = None  # type: ignore
    __all__ = ['environment', 'utils']
