# Phase B1.5（31）：状态空间精简与拐角感知增强（FINAL v1.9）
版本日期：2026-01-06  
插入位置：`30_Phase_B2a_FINAL` 之后、`32_Phase_B2a1_ArcInBand...` 之前  
依赖：B1 已通过（主表列齐全、trace/summary 工作流稳定）  
**阻断后续**：32/35 等 Phase 必须在本 Phase 通过后执行

---

## 0) 这份 Phase 解决什么（一句话）
当前 36 维状态空间**冗余且缺乏拐角感知**。  
本 Phase **精简到 12 维**，移除无效的 Lookahead，增加关键拐角感知信号。

---

## 1) 当前状态空间的问题

### 1.1 现有 36 维分析
| 类别 | 维度 | 问题 |
|------|------|------|
| 基础 12 维 | 12 | 部分冗余（jerk/angular_jerk 等高阶量） |
| Lookahead | 24 = 8×3 | ❌ `kappa_rate` 对折线无效；信息与拐角特征高度重叠 |
| **缺失** | 0 | 关键拐角感知信息完全缺失 |

### 1.2 关键缺失
| 缺失特征 | 重要性 | 说明 |
|---------|-------|------|
| `e_n`（法向偏移） | 🔴 必需 | 策略需要知道当前在带内的位置 |
| `turn_sign` | 🔴 必需 | 左转 +1 / 右转 -1，决定内切方向 |
| `corner_phase` | 🟡 重要 | 是否在拐角期 |
| `inside_signed` | 🟡 重要 | turn_sign × e_n，内切深度直接指标 |

---

## 2) 新状态空间设计（12 维）

### 2.1 核心特征（8 维）
```python
core_keys = [
    "contour_error_norm",    # |e| / half_ε，误差比例 [0, 2+]
    "e_n_norm",              # e_n / half_ε，法向偏移 [-1, 1]
    "heading_error_norm",    # τ / π，航向误差 [-1, 1]
    "velocity_norm",         # v / MAX_VEL，[0, 1]
    "acceleration_norm",     # a / MAX_ACC，[-1, 1]
    "angular_vel_norm",      # ω / MAX_ANG_VEL，[-1, 1]
    "overall_progress",      # [0, 1]
    "dist_to_turn_norm",     # dist_to_turn / dist_enter，[0, 2]
]
```

### 2.2 拐角感知特征（4 维）
```python
corner_keys = [
    "turn_angle_norm",   # turn_angle / π，[-1, 1]，正=左转
    "turn_sign",         # +1 左转 / -1 右转 / 0 直线或S型
    "corner_phase",      # 1.0 在拐角期 / 0.0 否
    "inside_signed",     # turn_sign × e_n_norm，正=内侧，负=外侧
]
```

### 2.3 移除 Lookahead
原因：
- `dist_to_turn_norm` + `turn_angle_norm` 已提供"前方有拐角"的信息
- `turn_sign` 已提供转向方向
- Lookahead 的 `kappa_rate` 对折线路径无效

### 2.4 总计
$$\text{总维度} = 8 + 4 = 12$$

---

## 3) 代码修改清单

### 3.1 `cnc_env.py` - `__init__` 部分（~Line 55-75）
```python
# 完全移除 lookahead
self.lookahead_points = 0
self.lookahead_feature_size = 0

# 新定义
self.core_keys = [
    "contour_error_norm", "e_n_norm", "heading_error_norm",
    "velocity_norm", "acceleration_norm", "angular_vel_norm",
    "overall_progress", "dist_to_turn_norm"
]
self.corner_keys = [
    "turn_angle_norm", "turn_sign", "corner_phase", "inside_signed"
]
self.observation_dim = len(self.core_keys) + len(self.corner_keys)
# observation_dim = 8 + 4 = 12
```

### 3.2 `cnc_env.py` - 新增 `_build_state` 方法
```python
def _build_state(self):
    """构建 12 维状态向量"""
    # 获取投影和拐角信息
    proj, seg_idx, s_now, t_hat, n_hat = self._project_onto_path(self.current_position)
    e_n = float(np.dot(self.current_position - proj, n_hat))
    contour_err = abs(e_n)  # 或用 get_contour_error
    tau = self.calculate_direction_deviation(self.current_position)
    
    # 从 corridor_status 获取拐角信息
    corridor = getattr(self, "last_corridor_status", None) or self._compute_corridor_status()
    dist_to_turn = float(corridor.get("dist_to_turn", float("inf")))
    turn_angle = float(corridor.get("turn_angle", 0.0))
    turn_sign = float(corridor.get("turn_sign", 0))
    corner_phase = 1.0 if corridor.get("corner_phase", False) else 0.0
    
    # 归一化
    half_eps = max(self.half_epsilon, 1e-6)
    dist_enter = float(getattr(self, "_corridor_dist_enter", 6.0))
    
    e_n_norm = e_n / half_eps
    inside_signed = turn_sign * e_n_norm
    
    state = [
        # 核心 8 维
        contour_err / half_eps,                          # contour_error_norm
        e_n_norm,                                        # e_n_norm
        tau / math.pi,                                   # heading_error_norm
        self.velocity / max(self.MAX_VEL, 1e-6),        # velocity_norm
        self.acceleration / max(self.MAX_ACC, 1e-6),    # acceleration_norm
        self.angular_vel / max(self.MAX_ANG_VEL, 1e-6), # angular_vel_norm
        float(getattr(self, "_progress_ratio", 0.0)),   # overall_progress
        min(dist_to_turn / max(dist_enter, 1.0), 2.0),  # dist_to_turn_norm
        # 拐角 4 维
        turn_angle / math.pi,                           # turn_angle_norm
        turn_sign,                                      # turn_sign
        corner_phase,                                   # corner_phase
        inside_signed,                                  # inside_signed
    ]
    return np.array(state, dtype=np.float32)
```

### 3.3 `cnc_env.py` - 修改 `reset` 和 `step` 返回值
```python
# 在 reset() 末尾
return self._build_state()

# 在 step() 末尾
next_state = self._build_state()
return next_state, reward, done, info
```

### 3.4 删除或注释掉旧的 lookahead 相关代码
- `_compute_lookahead_features` 可保留但不调用
- `normalize_state` 可保留但不调用

---

## 4) 验收标准（必须全部通过）

### 4.1 基线不退化（Blocking）
| 指标 | 条件 |
|------|------|
| `success_rate` | ≥ 0.95 |
| `max_abs_contour_error` | ≤ baseline × 1.05 |
| `steps` | ≤ baseline × 1.10 |

### 4.2 状态空间验证
- [ ] `env.observation_space.shape[0] == 12`
- [ ] 在 corner_phase=True 期间，`state[9]`（turn_sign）≠ 0
- [ ] `state[11]`（inside_signed）在孤立拐角时有正/负变化

### 4.3 快检脚本
```powershell
conda activate PPO
python -c "
from src.environment import Env
env = Env(device='cpu', epsilon=1.5, interpolation_period=0.001,
          MAX_VEL=100, MAX_ACC=2000, MAX_JERK=20000,
          MAX_ANG_VEL=6.28, MAX_ANG_ACC=100, MAX_ANG_JERK=1000,
          Pm=[(0,0),(10,0),(10,10),(0,10),(0,0)], max_steps=1000)
state = env.reset()
print(f'State dim: {len(state)} (expected: 12)')
print(f'State names: contour_err, e_n, heading, v, a, omega, progress, dist_turn, turn_angle, turn_sign, corner_phase, inside')
print(f'State values: {[f\"{x:.3f}\" for x in state]}')
"
```

---

## 5) 训练与评估

### 5.1 配置文件 `train_square_b1_5.yaml`
基于 P0 gold 配置，只需确保不使用旧的 lookahead 相关参数。

### 5.2 训练命令
```powershell
conda activate PPO
python main.py --config configs/train_square_b1_5.yaml --mode train
```

### 5.3 评估命令
```powershell
python tools/acceptance_suite.py --config configs/train_square_b1_5.yaml \
    --phase p0_eval --model saved_models/B1_5/.../best_model.pth \
    --episodes 20 --deterministic --out artifacts/B1_5/eval
```

---

## 6) Stop Rule
- `success_rate < 0.8`：检查状态归一化（是否超出 ±2）
- `steps` 暴增（> 1.3× baseline）：可能移除了关键信息，检查 `dist_to_turn_norm`
- `inside_signed` 始终 ≈ 0：检查 `turn_sign` 和 `e_n` 计算

---

## 7) 对比总结

| 方案 | 维度 | Lookahead | 拐角感知 | 优势 |
|------|------|-----------|---------|------|
| 原始 | 36 | 8 点 × 3 | ❌ 无 | - |
| **新方案** | 12 | ❌ 移除 | ✅ 4 维 | 精简 67%，信息密度高 |
