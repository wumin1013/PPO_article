# PPO_project 目标文档（v3.0 前瞻引导 + 管道奖励）
版本日期：2026-01-18  
面向：RCIM 投稿 + 可复现 Artifact

---

## 0) 版本演进说明

### v2.1 → v3.0 的核心转变

| 维度 | v2.1 (曲率感知) | v3.0 (前瞻+管道) |
|------|-----------------|------------------|
| **核心问题诊断** | 状态缺少曲率信息 | 参考方向不连续 + 无切弯空间 |
| **参考方向** | 路径切线（拐角处跳变90°） | **前瞻向量（连续平滑）** |
| **误差惩罚** | 线性惩罚所有误差 | **管道内零惩罚** |
| **理论基础** | 曲率感知假设 | Preview Control + Constraint Relaxation |

### 为什么改变？

v2.1 失败的根本原因分析：
1. **θ_ref 不连续**：路径切线在拐角处瞬间跳变 90°，策略无法跟随
2. **ω×dt 太小**：每步最大偏转仅 0.36°，"转不动"
3. **误差恐惧**：线性惩罚导致策略不敢利用 tolerance band

**v3.0 解决方案**：
- **前瞻向量**：消除参考方向的不连续性
- **管道奖励**：释放带内几何自由度
- **运动学解锁**：提高 MAX_ANG_VEL，允许 >1.8°/step 的偏转能力

---

## 1) 总目标（系统层面）

在轮廓误差容许带（±ε/2）约束下，让策略 π **自主学习**：

- **直线段**：高进给、贴线
- **拐角段**：利用前瞻引导的连续参考 + 管道空间，**自主发现平滑过渡策略**
- **全程**：到达终点、无停滞、运动学约束由 KCM shielding 保证

> **v3.0 关键**：前瞻向量提供连续可微的参考方向，管道奖励释放几何自由度。

---

## 2) 核心假设（v3.0）

### 假设 H1''：前瞻引导使跟随可行

```
前瞻参考方向连续 → 策略可以用小 ω 跟随 → 轨迹自然平滑
```

### 假设 H2'：管道奖励释放自由度

```
误差 < TOLERANCE → 零惩罚 → 策略可以利用空间切弯
```

### 假设 H3：KCM 是充分的安全保证（继承）

运动学约束通过 action shielding 强制执行。

---

## 3) 技术设计

### 3.1 前瞻参考方向（Lookahead Vector Guidance）

```python
def _get_path_direction(self, position):
    # 1. 找到当前位置在路径上的弧长 s
    s_current = self._get_current_arc_length()
    
    # 2. 前瞻 L 距离
    s_target = s_current + LOOKAHEAD_DIST
    target_point = self._interpolate_point_at_s(s_target)
    
    # 3. 计算从当前位置指向目标点的向量角度
    dx = target_point[0] - position[0]
    dy = target_point[1] - position[1]
    theta_lookahead = math.atan2(dy, dx)
    
    return theta_lookahead
```

**参数**：`LOOKAHEAD_DIST = 3.0 × ε = 4.5mm`（可调）

**效果**：即使路径是方形，θ_ref 也会在拐角前开始平滑转向。

### 3.2 管道奖励（Tolerance Tube Reward）

```python
def _calculate_minimal_reward(self, ctx):
    # 管道内零惩罚
    TOLERANCE = 0.5 * self.half_epsilon  # 例如 0.375mm
    
    if abs(ctx.contour_error) < TOLERANCE:
        r_contour = 0.0
    else:
        excess_error = abs(ctx.contour_error) - TOLERANCE
        r_contour = -w_e * (excess_error / self.half_epsilon) ** 2
    
    # 其他奖励项保持不变
    reward = r_progress + r_contour + r_time + r_completion
```

**参数**：`TOLERANCE = 0.5 × ε/2 = 0.375mm`

**效果**：策略可以在管道内自由选择路径，追求速度最大化。

### 3.3 运动学解锁（Kinematic Unlock）

- **问题**：原 `MAX_ANG_VEL = 2π` (6.28) 配合 `dt=1ms`，每步仅能转 0.36°。对于 r=0.5mm 的拐角，通过速度被物理限制在 ~3mm/s。
- **对策**：提升 `MAX_ANG_VEL` 至 `10π` (~31.4)，即 1.8°/step。
- **原理**：RL 需要足够的控制权限（Control Authority）来执行高频调整，过严的物理限制会掩盖 Reward 的引导作用。

---

## 4) 状态空间（14维，继承 v2.1）

```python
# 8维核心特征
core_keys = [
    "contour_error_norm", "e_n_norm", "heading_error_norm", "velocity_norm",
    "acceleration_norm", "angular_vel_norm", "overall_progress", "dist_to_turn_norm",
]

# 4维拐角感知特征
corner_keys = [
    "turn_angle_norm", "turn_sign", "corner_phase", "inside_signed",
]

# 2维曲率感知特征（保留用于诊断）
curvature_keys = [
    "kappa_norm", "dkappa_ds_norm",
]
```

---

## 5) Phase 设计（v3.0 → v3.1）

```
Phase 30: 前瞻+管道基线训练（v3.0）
    ├─ 参考方向：前瞻向量（指令式）
    ├─ 奖励函数：管道 + 极简
    └─ 验收：success_rate ≥ 0.95, corner_peak_ω < 0.8×MAX

Phase 31: 前瞻作为观测（v3.1）← 学术改进
    ├─ 参考方向：路径切线（客观几何）
    ├─ 前瞻信息：作为状态观测（16维）
    ├─ 核心改变：策略自主学习如何利用前瞻信号
    └─ 验收：success_rate ≥ 0.95, 观察 lookahead_utilization

Phase 32: 多路径验证
    └─ 验证策略在不同路径上的泛化

Phase 33: 消融分析
    ├─ A: 无前瞻（对比）
    ├─ B: 前瞻作为指令 v3.0（对比）
    └─ C: 前瞻作为观测 v3.1（主方案）

Phase 40: 论文输出
```

---

## 6) 验收标准

### 必须项（MUST）

| 指标 | 条件 | 来源 |
|------|------|------|
| success_rate | ≥ 0.95 | summary.json |
| max_abs_e_n | ≤ ε/2 = 0.75mm | trace |
| steps | ≤ 1.2× baseline | 对比 P0_L2 |

### 涌现项（OBSERVE）

| 指标 | 期望 | 解读 |
|------|------|------|
| corner_peak_ω | < 0.8×MAX | 平滑涌现 |
| inside_ratio | > 0.5 | 内切涌现 |
| corner_min_v | > baseline | 高速过弯 |

---

## 7) 论文定位

### 创新点

1. **KCM Shielding**：运动学约束的安全保证
2. **前瞻作为观测**：Preview information 作为状态特征，策略自主学习利用（v3.1）
3. **管道奖励**：Constraint Relaxation 释放几何自由度

### 论文话术

> "Rather than prescribing lookahead as the reference direction, we provide it as **state observations**, enabling the agent to autonomously discover **when** to anticipate turns and **how** to adjust its behavior proactively. Combined with a **tolerance-based reward function**, this design preserves the RL paradigm of learning from experience while benefiting from predictive geometric information."

---

## 8) 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.9 | 2026-01-06 | 规则引导路线 |
| v2.0 | 2026-01-17 | 极简端到端路线 |
| v2.1 | 2026-01-17 | 曲率感知 + 极简奖励 |
| v3.0 | 2026-01-18 | 前瞻引导 + 管道奖励（前瞻作为指令） |
| **v3.1** | **2026-01-28** | **前瞻作为观测（策略自主学习利用）** |
