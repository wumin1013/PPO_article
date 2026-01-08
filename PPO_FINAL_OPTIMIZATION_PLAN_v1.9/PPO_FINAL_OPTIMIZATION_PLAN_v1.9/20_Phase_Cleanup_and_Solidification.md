# Phase 20：代码清理与固化（Cleanup & Solidification）
版本日期：2026-01-08  
依赖：**无**（首个执行 Phase）  
**硬门槛**：必须通过 **Gate: Logic Equivalence Check** 后方可进入 Phase 21

---

## A. Scope / Non-Goals（范围与明确不做什么）

### A.1 允许改变（Refactoring Scope）
| 类别 | 允许的操作 |
|------|------------|
| 代码结构 | 函数拆分/合并、模块边界调整、调用顺序优化 |
| 命名 | 变量/函数/类重命名（需保持语义明确） |
| 参数传递 | 零散参数收敛为 context 对象、dataclass 或 TypedDict |
| 日志组织 | 日志级别调整、输出位置统一、trace 字段整理 |
| 死代码 | 删除未使用的函数、废弃的 flags、永不触发的分支 |
| 配置结构 | 合并冗余配置键、移除无效默认值 |

### A.2 禁止改变（Hard Constraints）

> [!CAUTION]
> 以下语义必须与清理前**完全等价**，任何偏离都必须通过 Gate 验证脚本证明为"可接受浮点误差"。

| 语义类别 | 约束说明 |
|----------|----------|
| **Physics** | `calculate_new_position`、运动学约束（KCM）、碰撞/越界检测的数值行为不变 |
| **Reward** | `calculate_reward` 在相同输入下返回相同数值（含各分项） |
| **Done/Success** | `reached_target`、`lap_completed`、`_p4_stall_triggered`、`max_steps` 等终止条件触发逻辑不变 |
| **action 接口** | `env.step(action)` 的输入维度、取值范围、裁剪逻辑不变 |
| **observation 接口** | `env.reset()` / `env.step()` 返回的状态向量维度与语义不变 |

### A.3 Phase 20 不做什么（Explicit Non-Goals）

```diff
- 不引入新的训练策略或超参数调整
- 不改变状态空间维度（这是 Phase 21 的任务）
- 不引入新的奖励项或修改现有奖励权重
- 不修改 P0/P0_L2 的配置文件（仅读取验证）
- 不优化性能瓶颈（除非重构自然带来优化）
```

---

## B. Cleanup Checklist（清理清单）

### B.1 Solidify（固化：从"可选开关"到"唯一主干"）

以下机制已被 P0 baseline 和 P0_L2 训练验证，应**移除 `if enabled` 开关**，使其成为默认核心逻辑：

| 机制标识 | 功能描述 | 当前代码位置 | 固化操作 |
|----------|----------|--------------|----------|
| `_p8_speed_cap` | 几何限速（turn scan + braking envelope） | `_init_p8_config` (L1317-1427) | 移除 `_p8_enabled` 检查，保留核心逻辑 |
| `_p4_stall` | 防停滞检测与终止 | `_init_p4_config` (L1156-1237), `step()` | 移除 `_p4_stall_enabled` 检查 |
| `_p4_exit_boost` | 出弯加速窗口 | `step()` (L687-703) | 移除 `_p4_exit_boost_enabled` 检查 |
| `_p1_corner_phase` | 拐角期标记（corner_mask） | `_init_p1_config` (L1114-1154) | 移除 `_p1_corner_phase_enabled` 检查 |
| `_p8_vcap` | 速度上限裁剪 | `step()`, `apply_action()` | 固化为默认行为 |

**固化代码模式**：
```python
# ---- 清理前（带开关） ----
if bool(getattr(self, "_p4_stall_enabled", True)):
    if progress_diff < stall_progress_eps and v_exec < stall_v_eps:
        self._p4_stall_counter += 1
    else:
        self._p4_stall_counter = 0

# ---- 清理后（无条件执行） ----
if progress_diff < self.stall_progress_eps and v_exec < self.stall_v_eps:
    self.stall_counter += 1
else:
    self.stall_counter = 0
```

### B.2 Remove Legacy / P.*（清除遗留代码）

#### B.2.1 遗留逻辑分类

| 分类 | 标识/函数 | 状态判定 | 处置策略 |
|------|-----------|----------|----------|
| **旧阶段实验** | `_p6_1_*` (动作变化率惩罚) | P0 已禁用 | 删除配置初始化，保留 reward.py 参数兼容签名 |
| **调试专用** | `_p7_3_dump_trace`, `_p7_3_trace_append` | 仅 NaN 诊断用 | 移至 `src/utils/debug.py`，step() 不再直接调用 |
| **废弃奖励分支** | `corridor_*` 系列参数 | VirtualCorridor 实验暂停 | 保留 reward.py 签名，简化 env 初始化 |
| **临时补丁** | `getattr(self, "_p*_xxx", default)` 模式 | 防御性编程残留 | 替换为直接属性访问 |
| **旧观测拼接** | lookahead 24维 | Phase 21 将移除 | Phase 20 保持不变（Phase 21 处理） |

#### B.2.2 删除/隔离策略

```
1. 删除条件：
   - 主流程中无任何调用路径引用该代码
   - 或：引用路径仅在 `if False` / `if disabled` 分支中
   
2. 隔离条件：
   - 代码仍有潜在价值（未来可能复用）
   - 或：短期无法确认无副作用
   
3. 隔离位置：
   - 创建 `src/environment/_legacy.py`
   - 被隔离代码默认不被 __init__.py 导出
   - 主流程禁止出现对 _legacy 的 import
```

#### B.2.3 待删除/隔离清单

| 代码位置 | 内容 | 操作 | 验证方法 |
|----------|------|------|----------|
| `_init_p6_1_config` | 动作变化率配置 | 删除 | 确认 reward 中 `du_enabled=False` |
| `_p7_3_trace_append` | 调试 trace 收集 | 隔离到 debug 模块 | 确认 step 无直接调用 |
| `_compute_lookahead_features` | 24维前瞻 | **保留**（Phase 21 处理） | - |
| `_init_corridor_config` 中的 P5.* 分支 | 旧走廊配置 | 简化，保留兼容键 | 确认默认配置无变化 |

### B.3 Perception Refactor（感知重构）

#### B.3.1 目标
将 `_scan_for_next_turn` 从内部辅助函数提升为**核心感知组件**。

#### B.3.2 `self.turn_info` 字段规范

| 字段名 | 类型 | 含义 | 单位 | 取值范围 | 来源函数 |
|--------|------|------|------|----------|----------|
| `in_turn_zone` | `bool` | 是否进入拐角相关区域 | - | `True`/`False` | `_scan_for_next_turn` |
| `corner_phase` | `bool` | 当前是否处于拐角期（用于 reward 的 corner_mask） | - | `True`/`False` | 基于 `dist_to_turn` 阈值 |
| `turn_sign` | `int` | 转向方向：+1=左转, -1=右转, 0=直行 | - | `{-1, 0, +1}` | `_scan_for_next_turn` |
| `dist_to_turn` | `float` | 到下一拐点的弧长距离 | mm | `[0, total_arc_length]` | `_scan_for_next_turn` |
| `dist_to_corner_entry` | `float` | 到转弯段起点（减速开始点）的距离 | mm | `[0, ∞)` | 基于 braking envelope |
| `turn_angle` | `float` | 拐角角度绝对值 | rad | `[0, π]` | `_scan_for_next_turn` |
| `turn_severity` | `float` | 急转程度代理量 = `|turn_angle| / π` | - | `[0, 1]` | 计算得出 |
| `curvature_proxy` | `float` | 曲率代理 = `turn_angle / segment_length` | 1/mm | `[0, ∞)` | 计算得出 |
| `recommended_v_cap` | `float` | 基于几何的建议速度上限 | mm/s | `[0, MAX_VEL]` | `_p8_speed_cap` 逻辑 |
| `is_isolated_corner` | `bool` | 是否为孤立拐角（非 S 弯复合） | - | `True`/`False` | `_scan_for_next_turn` |
| `turn_s` | `float` | 拐角顶点的弧长位置 | mm | `[0, total_arc_length]` | `_scan_for_next_turn` |

#### B.3.3 实施约束

> [!IMPORTANT]
> `turn_info` 在 Phase 20 **只做感知与记录**，不改变控制或奖励语义。
> 
> - 禁止在 Phase 20 中使用 `turn_info` 新增奖励项
> - 禁止在 Phase 20 中使用 `turn_info` 修改动作裁剪逻辑
> - `turn_info` 的目的是为 Phase 21 的 12 维状态提取提供稳定接口

#### B.3.4 调用位置重构

```python
# ---- 清理前：多处重复调用 ----
# reset() 中调用一次
# step() 中调用一次
# _compute_p4_pre_step_status() 中调用一次
# _compute_corridor_status() 中调用一次

# ---- 清理后：统一调用点 ----
def reset(self, ...):
    ...
    self.turn_info = self._compute_turn_info()  # 唯一调用点（reset）
    return state

def step(self, action):
    ...
    self.turn_info = self._compute_turn_info()  # 唯一调用点（step）
    ...
    return next_state, reward, done, info

def _compute_turn_info(self) -> Dict[str, Any]:
    """统一的 turn 感知计算，封装 _scan_for_next_turn 并补充派生字段"""
    s_now = self._get_current_arc_length()
    raw_scan = self._scan_for_next_turn(s_now)
    
    # 派生字段计算（不改变原有语义）
    turn_info = {
        "in_turn_zone": raw_scan["dist_to_turn"] < self._turn_zone_threshold,
        "corner_phase": self._determine_corner_phase(raw_scan),
        "turn_sign": int(raw_scan.get("turn_sign", 0)),
        "dist_to_turn": float(raw_scan.get("dist_to_turn", float("inf"))),
        # ... 其他字段
    }
    return turn_info
```

### B.4 Interface Refactor（接口重构）

#### B.4.1 `env.step()` 流程重构

**清理前流程**（约 230 行，分支复杂）：
```
┌─────────────────────────────────────────────────────────────┐
│ 1. _compute_p4_pre_step_status()  ← 调用 _scan_for_next_turn │
│ 2. 条件性跳过检查（多个 if enabled）                          │
│ 3. apply_action()                                            │
│ 4. _update_segment_info()                                    │
│ 5. _compute_corridor_status()     ← 再次调用 _scan_for_next_turn│
│ 6. 多处 getattr() 防御性取值                                  │
│ 7. calculate_reward()（参数列表过长）                         │
│ 8. 条件性诊断输出                                            │
│ 9. done 判定（多处分散）                                      │
│10. info 组装                                                 │
└─────────────────────────────────────────────────────────────┘
```

**清理后目标流程**：
```
┌─────────────────────────────────────────────────────────────┐
│ 1. apply_action(action)           → 更新 position/velocity   │
│ 2. update_turn_info()             → 更新 self.turn_info      │
│ 3. update_motion_state()          → 更新 errors/progress     │
│ 4. check_termination()            → 返回 (done, done_reason) │
│ 5. calculate_reward(context)      → 统一 context 对象传参    │
│ 6. build_info()                   → 组装 info dict           │
│ 7. build_next_state()             → 组装 observation         │
└─────────────────────────────────────────────────────────────┘
```

#### B.4.2 `reward.py` 参数收敛

**清理前签名**（40+ 参数）：
```python
def calculate_reward(
    self,
    contour_error: float,
    progress: float,
    velocity: float,
    heading_error: float,
    kcm_intervention: float,
    end_distance: float,
    jerk: float,
    angular_jerk: float,
    angular_acc: float = 0.0,
    du_theta_u: float = 0.0,
    du_v_u: float = 0.0,
    du_enabled: bool = False,
    du_weight: float = 0.0,
    du_mode: str = "l1",
    lap_completed: bool = False,
    is_closed: bool = False,
    corner_mask: bool = False,
    v_ratio_exec: float | None = None,
    speed_target: float | None = None,
    speed_weight: float = 6.0,
    time_penalty: float = 0.0,
    progress_multiplier: float = 1.0,
    stall_triggered: bool = False,
    stall_penalty: float = 0.0,
    corridor_enabled: bool = False,
    corridor_active: bool = False,
    # ... 20+ more corridor_* params
    **_: object,
) -> Tuple[float, Dict[str, float]]:
```

**清理后目标签名**：
```python
@dataclass
class RewardContext:
    """统一的 reward 计算上下文"""
    # Core tracking
    contour_error: float
    progress: float
    velocity: float
    heading_error: float
    
    # Motion state
    jerk: float
    angular_jerk: float
    angular_acc: float
    
    # Turn info (from self.turn_info)
    corner_mask: bool
    turn_sign: int
    
    # Termination
    stall_triggered: bool
    lap_completed: bool
    is_closed: bool
    
    # Legacy compat (deprecated, default to disabled)
    corridor_kwargs: Dict[str, Any] = field(default_factory=dict)

def calculate_reward(self, ctx: RewardContext) -> Tuple[float, Dict[str, float]]:
    """P0 reward: progress-dominant with tracking/heading/time penalties."""
    # 内部逻辑不变，仅参数传递方式改变
```

> [!WARNING]
> 参数收敛必须保证：`reward(old_params) == reward(ctx)`，由 Verification 证明。

---

## C. Logic Equivalence Verification（逻辑等价性验证）

### C.1 验证脚本：`verify_logic_equivalence.py`

**文件位置**：`PPO_project/tools/verify_logic_equivalence.py`

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 20 Gate: Logic Equivalence Verification

验证"清理前"与"清理后"代码的物理行为、奖励计算、终止条件完全等价。
这是进入 Phase 21 的硬门槛。

Usage:
    python tools/verify_logic_equivalence.py \
        --config configs/p0_l2_gold.yaml \
        --episodes 10 \
        --seed 42 \
        --out out/phase20_gate

Output:
    - out/phase20_gate/trace_before.csv
    - out/phase20_gate/trace_after.csv
    - out/phase20_gate/diff_report.json
    - out/phase20_gate/summary.json (PASS/FAIL)
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# 配置常量
# ============================================================
DEFAULT_SEED = 42
DEFAULT_EPISODES = 10
MAX_STEPS_PER_EPISODE = 2000
GOLDEN_CONFIG = "configs/p0_l2_gold.yaml"

# 容差标准（仅在完全一致失败时使用）
TOLERANCE = {
    "position": 1e-9,
    "velocity": 1e-9,
    "reward": 1e-9,
    "progress": 1e-9,
    "contour_error": 1e-9,
}


# ============================================================
# 工具函数
# ============================================================
def set_all_seeds(seed: int) -> None:
    """固定所有 RNG 源"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def hash_config(config: dict) -> str:
    """生成配置的哈希值，用于验证配置一致性"""
    config_str = json.dumps(config, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(config_str.encode("utf-8")).hexdigest()[:16]


# ============================================================
# Trace 收集
# ============================================================
@dataclass
class TraceRow:
    """单步 trace 记录"""
    episode: int
    step: int
    pos_x: float
    pos_y: float
    velocity: float
    heading: float
    contour_error: float
    progress: float
    reward: float
    # Reward 分项
    r_progress: float
    r_track: float
    r_dir: float
    r_time: float
    r_smooth: float
    # Turn info
    corner_phase: bool
    turn_sign: int
    dist_to_turn: float
    # Done info
    done: bool
    done_reason: str


def collect_trace(
    env,
    action_sequences: List[List[np.ndarray]],
    episodes: int,
) -> List[TraceRow]:
    """收集环境运行 trace"""
    trace: List[TraceRow] = []
    
    for ep in range(episodes):
        obs = env.reset()
        actions = action_sequences[ep]
        
        for step_idx, action in enumerate(actions):
            obs, reward, done, info = env.step(action)
            
            # 获取 reward 分项
            reward_components = getattr(env, "last_reward_components", {})
            
            # 获取 turn_info
            turn_info = getattr(env, "turn_info", {})
            if not turn_info:
                # 兼容旧版本：从 info 中提取
                turn_info = {
                    "corner_phase": info.get("corner_phase", False),
                    "turn_sign": info.get("turn_sign", 0),
                    "dist_to_turn": info.get("dist_to_turn", float("inf")),
                }
            
            # 确定 done_reason
            done_reason = "running"
            if done:
                if getattr(env, "reached_target", False):
                    done_reason = "success"
                elif getattr(env, "lap_completed", False):
                    done_reason = "lap_completed"
                elif getattr(env, "_p4_stall_triggered", False):
                    done_reason = "stall"
                elif env.current_step >= env.max_steps:
                    done_reason = "max_steps"
                else:
                    done_reason = "unknown"
            
            row = TraceRow(
                episode=ep,
                step=step_idx,
                pos_x=float(info.get("position", [0, 0])[0]),
                pos_y=float(info.get("position", [0, 0])[1]),
                velocity=float(info.get("velocity", 0)),
                heading=float(info.get("heading", 0)),
                contour_error=float(info.get("contour_error", 0)),
                progress=float(info.get("progress", 0)),
                reward=float(reward),
                r_progress=float(reward_components.get("r_progress", 0)),
                r_track=float(reward_components.get("r_track", 0)),
                r_dir=float(reward_components.get("r_dir", 0)),
                r_time=float(reward_components.get("r_time", 0)),
                r_smooth=float(reward_components.get("r_smooth", 0)),
                corner_phase=bool(turn_info.get("corner_phase", False)),
                turn_sign=int(turn_info.get("turn_sign", 0)),
                dist_to_turn=float(turn_info.get("dist_to_turn", float("inf"))),
                done=done,
                done_reason=done_reason,
            )
            trace.append(row)
            
            if done:
                break
    
    return trace


def save_trace_csv(trace: List[TraceRow], path: Path) -> None:
    """保存 trace 为 CSV"""
    import csv
    
    path.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = list(TraceRow.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in trace:
            writer.writerow(asdict(row))


# ============================================================
# Diff 比较
# ============================================================
@dataclass
class DiffResult:
    """Diff 比较结果"""
    passed: bool
    exact_match: bool
    total_rows: int
    mismatched_rows: int
    max_diffs: Dict[str, float]
    first_mismatch: Optional[Dict[str, Any]]
    tolerance_used: Dict[str, float]


def compare_traces(
    trace_before: List[TraceRow],
    trace_after: List[TraceRow],
    tolerance: Dict[str, float],
) -> DiffResult:
    """比较两个 trace 是否等价"""
    
    # 长度检查
    if len(trace_before) != len(trace_after):
        return DiffResult(
            passed=False,
            exact_match=False,
            total_rows=max(len(trace_before), len(trace_after)),
            mismatched_rows=abs(len(trace_before) - len(trace_after)),
            max_diffs={"trace_length": float(abs(len(trace_before) - len(trace_after)))},
            first_mismatch={"type": "length_mismatch", 
                           "before": len(trace_before), 
                           "after": len(trace_after)},
            tolerance_used=tolerance,
        )
    
    # 逐行比较
    max_diffs: Dict[str, float] = {}
    first_mismatch: Optional[Dict[str, Any]] = None
    mismatched_rows = 0
    exact_match = True
    
    float_fields = ["pos_x", "pos_y", "velocity", "heading", "contour_error", 
                    "progress", "reward", "r_progress", "r_track", "r_dir", 
                    "r_time", "r_smooth", "dist_to_turn"]
    
    for i, (before, after) in enumerate(zip(trace_before, trace_after)):
        row_has_diff = False
        
        for field in float_fields:
            v_before = getattr(before, field)
            v_after = getattr(after, field)
            
            if not math.isfinite(v_before) and not math.isfinite(v_after):
                continue  # 两边都是 NaN/Inf，视为一致
            
            diff = abs(v_before - v_after)
            max_diffs[field] = max(max_diffs.get(field, 0), diff)
            
            if diff > 0:
                exact_match = False
            
            # 获取该字段对应的容差
            tol_key = "position" if field in ["pos_x", "pos_y"] else field
            tol = tolerance.get(tol_key, tolerance.get("default", 1e-9))
            
            if diff > tol:
                row_has_diff = True
                if first_mismatch is None:
                    first_mismatch = {
                        "row": i,
                        "episode": before.episode,
                        "step": before.step,
                        "field": field,
                        "before": v_before,
                        "after": v_after,
                        "diff": diff,
                        "tolerance": tol,
                    }
        
        # 非浮点字段严格比较
        if before.done != after.done:
            row_has_diff = True
            exact_match = False
            if first_mismatch is None:
                first_mismatch = {
                    "row": i, "field": "done",
                    "before": before.done, "after": after.done,
                }
        
        if before.done_reason != after.done_reason:
            row_has_diff = True
            exact_match = False
            if first_mismatch is None:
                first_mismatch = {
                    "row": i, "field": "done_reason",
                    "before": before.done_reason, "after": after.done_reason,
                }
        
        if before.corner_phase != after.corner_phase:
            row_has_diff = True
            exact_match = False
        
        if before.turn_sign != after.turn_sign:
            row_has_diff = True
            exact_match = False
        
        if row_has_diff:
            mismatched_rows += 1
    
    # 判定是否通过
    passed = (first_mismatch is None)  # 容差内全部通过
    
    return DiffResult(
        passed=passed,
        exact_match=exact_match,
        total_rows=len(trace_before),
        mismatched_rows=mismatched_rows,
        max_diffs=max_diffs,
        first_mismatch=first_mismatch,
        tolerance_used=tolerance,
    )


# ============================================================
# 主流程
# ============================================================
def generate_deterministic_actions(seed: int, episodes: int, max_steps: int) -> List[List[np.ndarray]]:
    """生成确定性的动作序列"""
    rng = np.random.RandomState(seed)
    action_sequences = []
    for _ in range(episodes):
        actions = [
            np.array([rng.uniform(-1, 1), rng.uniform(0, 1)], dtype=np.float64)
            for _ in range(max_steps)
        ]
        action_sequences.append(actions)
    return action_sequences


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 20 Gate: Logic Equivalence Verification"
    )
    parser.add_argument("--config", type=str, default=GOLDEN_CONFIG,
                        help="黄金配置文件路径")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES,
                        help="验证 episode 数量")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="RNG 种子")
    parser.add_argument("--out", type=str, default="out/phase20_gate",
                        help="输出目录")
    parser.add_argument("--tolerance-scale", type=float, default=1.0,
                        help="容差倍数（1.0 = 默认容差）")
    args = parser.parse_args(argv)
    
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载配置
    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        return 1
    config = load_yaml(config_path)
    config_hash = hash_config(config)
    
    print("=" * 70)
    print("Phase 20 Gate: Logic Equivalence Verification")
    print("=" * 70)
    print(f"Config: {config_path}")
    print(f"Config Hash: {config_hash}")
    print(f"Seed: {args.seed}")
    print(f"Episodes: {args.episodes}")
    print(f"Tolerance Scale: {args.tolerance_scale}")
    print()
    
    # 设置种子
    set_all_seeds(args.seed)
    
    # 生成确定性动作序列
    action_sequences = generate_deterministic_actions(
        args.seed, args.episodes, MAX_STEPS_PER_EPISODE
    )
    
    # ========================================
    # 此处需要分别构建清理前/后的环境
    # 实际使用时，需要通过某种机制切换版本
    # 例如：git stash / git checkout / 环境变量
    # ========================================
    
    # 示例框架：
    # from src.environment import Env
    # env_before = build_env(config, version="pre_phase20")
    # env_after = build_env(config, version="post_phase20")
    
    # trace_before = collect_trace(env_before, action_sequences, args.episodes)
    # trace_after = collect_trace(env_after, action_sequences, args.episodes)
    
    # 由于当前无法同时加载两个版本，这里输出框架
    print("⚠️  完整验证需要分别运行清理前/后版本")
    print("   建议流程：")
    print("   1. git stash (保存清理后代码)")
    print("   2. git checkout pre_phase20 (切换到清理前)")
    print("   3. python tools/verify_logic_equivalence.py --mode before --out out/phase20_gate")
    print("   4. git checkout phase20 (切换到清理后)")
    print("   5. python tools/verify_logic_equivalence.py --mode after --out out/phase20_gate")
    print("   6. python tools/verify_logic_equivalence.py --mode compare --out out/phase20_gate")
    print()
    
    # 输出配置摘要
    summary = {
        "status": "FRAMEWORK_READY",
        "config_path": str(config_path),
        "config_hash": config_hash,
        "seed": args.seed,
        "episodes": args.episodes,
        "tolerance": TOLERANCE,
        "tolerance_scale": args.tolerance_scale,
    }
    
    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Summary written to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

### C.2 验收标准（可执行指标）

#### C.2.1 强制标准（优先检查）

| 指标 | 标准 | 检查方法 |
|------|------|----------|
| **Trace 长度** | 完全相同 | `len(trace_before) == len(trace_after)` |
| **Position** | 完全一致或 `|Δ| < 1e-9` | 逐行比较 `pos_x`, `pos_y` |
| **Velocity** | 完全一致或 `|Δ| < 1e-9` | 逐行比较 `velocity` |
| **Reward** | 完全一致或 `|Δ| < 1e-9` | 逐行比较 `reward` 及各分项 |
| **Progress** | 完全一致或 `|Δ| < 1e-9` | 逐行比较 `progress` |
| **Done/DoneReason** | 完全一致 | 严格字符串匹配 |
| **corner_phase** | 完全一致 | 严格布尔匹配 |
| **turn_sign** | 完全一致 | 严格整数匹配 |

#### C.2.2 容差说明（仅在完全一致失败时适用）

若出现不可避免的浮点差异，必须满足：

1. **最大差异** < 指定容差（默认 `1e-9`）
2. **差异来源**必须在报告中解释，例如：
   - 计算顺序改变导致的浮点累积差异
   - `np.float32` vs `np.float64` 精度差异
3. **差异影响分析**：证明该差异不影响：
   - 训练收敛性
   - 策略评估结果
   - 终止条件触发时机

#### C.2.3 Gate 通过/失败判定

```python
def gate_decision(diff_result: DiffResult) -> str:
    if diff_result.exact_match:
        return "✅ GATE PASSED (exact match)"
    elif diff_result.passed:
        return "✅ GATE PASSED (within tolerance)"
    else:
        return "❌ GATE FAILED"
```

**Gate 失败时**：
- 禁止进入 Phase 21
- 必须修复差异来源
- 重新运行验证直到通过

---

## D. Implementation Plan（实施步骤）

### 可执行 Checklist

```markdown
## Phase 20 实施进度追踪

### Step 0: 备份与分支准备
- [ ] 创建 pre_phase20 tag: `git tag pre_phase20`
- [ ] 创建工作分支: `git checkout -b phase20-cleanup`
- [ ] 确认 P0_L2 配置文件存在且可读: `configs/p0_l2_gold.yaml`
- [ ] 运行 baseline 验证: `python tools/acceptance_suite.py --phase p0_eval ...`

### Step 1: 盘点 Flags 与 P.* 遗留点
- [ ] 搜索所有 `_p[0-9]` 模式: `grep -rn "_p[0-9]" src/`
- [ ] 输出盘点表 (flag_inventory.csv)：
      | Flag 名 | 文件 | 行号 | 启用状态 | 处置建议 |
- [ ] 搜索所有 `getattr(self, "_p` 模式
- [ ] 标记每个 flag 的"固化/删除/隔离"决策

### Step 2: 固化核心逻辑
- [ ] 固化 `_p8_speed_cap`：移除 `_p8_enabled` 检查
- [ ] 固化 `_p4_stall`：移除 `_p4_stall_enabled` 检查
- [ ] 固化 `_p4_exit_boost`：移除 `_p4_exit_boost_enabled` 检查
- [ ] 固化 `_p1_corner_phase`：移除开关
- [ ] 将 `getattr(self, "_p*_xxx", default)` 替换为直接属性访问
- [ ] 运行单元测试（若有）：`pytest tests/`

### Step 3: 重构 _scan_for_next_turn → turn_info
- [ ] 创建 `_compute_turn_info()` 封装函数
- [ ] 在 `reset()` 末尾调用并存入 `self.turn_info`
- [ ] 在 `step()` 中更新 `self.turn_info`
- [ ] 删除 `step()` 中其他位置的 `_scan_for_next_turn` 重复调用
- [ ] 验证 `turn_info` 字段完整性

### Step 4: step() / reward.py 接口收敛
- [ ] 创建 `RewardContext` dataclass（或 TypedDict）
- [ ] 修改 `calculate_reward` 接收 context 对象
- [ ] 保留旧签名为 deprecated wrapper（可选）
- [ ] 简化 `step()` 主流程为 7 步结构
- [ ] 移除调试代码到 `src/utils/debug.py`

### Step 5: 运行验证脚本
- [ ] 准备 `verify_logic_equivalence.py` 脚本
- [ ] 切换到 pre_phase20 运行 before trace
- [ ] 切换到 phase20-cleanup 运行 after trace
- [ ] 运行 diff 比较
- [ ] 检查 diff_report.json，确认 PASSED
- [ ] 若 FAILED：定位差异 → 修复 → 重跑

### Step 6: P0/P0_L2 回归评估
- [ ] 使用 P0 模型运行 acceptance_suite (eval)
- [ ] 使用 P0_L2 模型运行 acceptance_suite (eval)
- [ ] 比较关键指标：success_rate, mean_progress, max_contour_error
- [ ] 确认指标一致（允许随机性导致的微小波动）

### Step 7: 收尾
- [ ] 更新 CHANGELOG.md
- [ ] Squash commit & merge to main
- [ ] 打 tag: `git tag phase20-done`
- [ ] 归档 `out/phase20_gate/` 到 artifacts
```

---

## E. Risks & Mitigations（风险与防护）

| 风险 | 等级 | 触发条件 | 缓解措施 |
|------|------|----------|----------|
| **确定性破坏** | 🔴 高 | RNG 状态未完全固定 | 使用 `set_all_seeds()` 固定所有 RNG 源；验证时强制固定配置 |
| **浮点累积差异** | 🟡 中 | 计算顺序改变、向量化替代循环 | 设定容差阈值；在报告中解释差异来源 |
| **日志字段漂移** | 🟡 中 | trace 字段重命名或删除 | 使用 dataclass 强制字段规范；版本化 trace schema |
| **配置漂移** | 🟡 中 | 默认值改变导致行为变化 | 使用黄金配置 `p0_l2_gold.yaml`；验证前后 config hash |
| **隐藏 flag 依赖** | 🟡 中 | 某些 flag 被间接引用 | 全局搜索 `_p[0-9]`；删除前确认无引用 |
| **import 循环** | 🟢 低 | 模块拆分导致循环依赖 | 遵循单向依赖原则；使用延迟 import |
| **测试覆盖不足** | 🟢 低 | 边界情况未被验证脚本覆盖 | 增加 episode 数量；使用多种路径类型 |

### 回滚策略

```powershell
# 如果 Phase 20 出现严重问题，立即回滚：
git checkout main
git branch -D phase20-cleanup  # 删除失败分支
git checkout pre_phase20 -- src/environment/  # 恢复文件
```

---

## 附录：文件修改摘要（预估）

| 文件 | 预估修改行数 | 主要改动 |
|------|--------------|----------|
| `cnc_env.py` | -200 ~ -400 | 移除开关、合并 init 函数、简化 step |
| `reward.py` | +20 ~ +50 | 添加 RewardContext，保留旧签名 |
| `_legacy.py` | +100 ~ +200 | 新建，隔离废弃代码 |
| `verify_logic_equivalence.py` | +300 ~ +400 | 新建验证脚本 |
