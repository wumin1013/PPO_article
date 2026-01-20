# Phase 21：状态空间精简与指标口径（FINAL v1.9）
版本日期：2026-01-07  
依赖：无（首个执行Phase）  
**阻断后续**：22/23/30 等 Phase 必须在本 Phase 通过后执行

---

## 0) 目标（一句话）
**精简状态空间到 12 维**，增加拐角感知特征，同时建立可计算的分段指标体系。

---

## 1) 状态空间重设计（36 → 12 维）

### 1.1 核心特征（8 维）
```python
core_keys = [
    "contour_error_norm",    # |e| / half_ε
    "e_n_norm",              # e_n / half_ε
    "heading_error_norm",    # τ / π
    "velocity_norm",         # v / MAX_VEL
    "acceleration_norm",     # a / MAX_ACC
    "angular_vel_norm",      # ω / MAX_ANG_VEL
    "overall_progress",      # [0, 1]
    "dist_to_turn_norm",     # dist_to_turn / dist_enter
]
```

### 1.2 拐角感知特征（4 维）
```python
corner_keys = [
    "turn_angle_norm",   # turn_angle / π
    "turn_sign",         # +1 左转 / -1 右转 / 0 直线
    "corner_phase",      # 1.0 在拐角期 / 0.0 否
    "inside_signed",     # turn_sign × e_n_norm
]
```

### 1.3 移除 Lookahead
- 原 24 维 Lookahead 对折线路径无效
- 拐角信息已通过 corner_keys 提供

---

## 2) 指标口径定义（原B1内容）

### 2.1 分段定义
| 分段 | 定义 |
|------|------|
| `corner_mask` | `corner_phase == 1` 的区间 |
| `corner_end` | `corner_mask` 结束后 N 步的窗口 |
| `non-corner` | 与 `corner_mask` 互补 |

### 2.2 必须记录的 trace 字段
- `e_n`, `contour_error`, `velocity`, `omega`
- `corner_phase`, `turn_sign`, `dist_to_turn`
- `inside_signed`

### 2.3 summary 分段指标
- **corner_mask 段**：`inside_ratio`, `corner_sharpness_index`, `v_drop_ratio`
- **non-corner 段**：`mean_velocity`, `steps`

---

## 3) 代码修改清单

### 3.1 `cnc_env.py` - `__init__`
```python
self.lookahead_points = 0
self.core_keys = [...]  # 8个
self.corner_keys = [...]  # 4个
self.observation_dim = 12
```

### 3.2 `cnc_env.py` - 新增 `_build_state`
（完整代码见原21文档）

### 3.3 修改 `reset` 和 `step` 返回值
使用 `_build_state()` 替代原有状态构建

---

## 4) 验收标准

### 4.1 状态空间验证
- [ ] `env.observation_space.shape[0] == 12`
- [ ] `turn_sign` 在拐角期非零
- [ ] `inside_signed` 有正/负变化

### 4.2 快检脚本
```powershell
conda activate PPO
python -c "
from src.environment import Env
env = Env(...)
state = env.reset()
print(f'State dim: {len(state)}')  # 期望: 12
"
```

---

## 5) Stop Rule
- `success_rate < 0.8`：检查状态归一化
- `steps` 暴增：检查关键特征是否丢失
