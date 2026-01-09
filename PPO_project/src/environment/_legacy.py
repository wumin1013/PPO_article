"""
Phase 20 遗留代码隔离
默认不被 __init__.py 导出
主流程禁止 import

此文件用于存放 Phase 20 清理过程中被移除但可能具有参考价值的旧代码。
"""

from typing import Dict, Any

def _legacy_p6_1_config_parser(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    (Deprecated) 解析 P6.1 动作变化率配置
    仅作参考，实际逻辑已在 Phase 20 中移除
    """
    return {
        "du_enabled": bool(cfg.get("du_enabled", True)),
        "w_du": float(cfg.get("w_du", 0.01)),
        "du_mode": str(cfg.get("du_mode", "l1")).lower(),
        "v_target_smoother_enabled": bool(cfg.get("v_target_smoother_enabled", True)),
        "v_target_mode": str(cfg.get("v_target_mode", "accel")).lower(),
    }
