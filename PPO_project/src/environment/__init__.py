"""环境模块入口。"""
from __future__ import annotations

import inspect
from typing import Any, Dict

from .cnc_env import Env, create_environment_from_config, apply_kinematic_constraints
from .reward import RewardCalculator


def create_env_compatible(**kwargs: Any) -> Env:
    """按当前 Env.__init__ 签名过滤参数，兼容环境构造接口演进。"""
    allowed = set(inspect.signature(Env.__init__).parameters)
    allowed.discard("self")
    filtered: Dict[str, Any] = {key: value for key, value in kwargs.items() if key in allowed}
    return Env(**filtered)


__all__ = [
    "Env",
    "create_environment_from_config",
    "create_env_compatible",
    "apply_kinematic_constraints",
    "RewardCalculator",
]
