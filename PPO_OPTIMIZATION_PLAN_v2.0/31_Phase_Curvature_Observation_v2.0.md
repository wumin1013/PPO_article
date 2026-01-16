# Phase 31：曲率状态扩展（Curvature Observation）
版本日期：2026-01-17  
依赖：Phase 30 完成但未涌现平滑行为  
**触发条件**：仅当 Phase 30 的 corner_peak_omega 无显著改善时执行

---

## 0) 目标（一句话）

**让策略"看到"曲率信息**，假设：如果策略能感知自身轨迹的曲率，它更容易学会平滑行为。

---

## 1) 为什么需要这个 Phase？

### 1.1 Phase 30 失败的可能原因

Phase 30 的极简奖励依赖假设：
> 急转弯 → KCM降速 → 效率变低 → 策略自然避免急转弯

但如果这个反馈链条太长/太隐式，策略可能难以学习。

### 1.2 曲率状态的作用

```
原状态：策略只能"感知"位置、速度、误差
新状态：策略还能"感知"自身轨迹的曲率

曲率高 → 策略意识到"我在急转" → 可以主动调整
```

这**不是**在奖励中加规则，而是在状态中加信息。策略仍然通过效率压力学习，但有更多信息可用。

---

## 2) 状态空间扩展

### 2.1 当前状态（12 维，Phase 21）

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

corner_keys = [
    "turn_angle_norm",   # turn_angle / π
    "turn_sign",         # +1 / -1 / 0
    "corner_phase",      # 1.0 / 0.0
    "inside_signed",     # turn_sign × e_n_norm
]
```

### 2.2 新增曲率特征（2 维）

```python
curvature_keys = [
    "kappa_norm",        # κ / κ_max（当前轨迹曲率）
    "dkappa_ds_norm",    # (dκ/ds) / (dκ/ds)_max（曲率变化率）
]
```

### 2.3 曲率计算

```python
def compute_curvature_features(self) -> Tuple[float, float]:
    """从执行轨迹历史计算曲率特征"""
    
    # 使用最近 3 个轨迹点拟合圆弧
    if len(self.trajectory) < 3:
        return 0.0, 0.0
    
    p0, p1, p2 = self.trajectory[-3], self.trajectory[-2], self.trajectory[-1]
    
    # 三点定圆，计算曲率 κ = 1/R
    # 使用向量叉积方法
    v1 = p1 - p0
    v2 = p2 - p1
    
    cross = abs(v1[0] * v2[1] - v1[1] * v2[0])
    chord = np.linalg.norm(p2 - p0)
    
    if chord < 1e-9:
        kappa = 0.0
    else:
        # κ ≈ 2 * sin(θ) / chord ≈ 2 * |v1 × v2| / (|v1| * |v2| * chord)
        kappa = 2.0 * cross / (np.linalg.norm(v1) * np.linalg.norm(v2) * chord + 1e-9)
    
    # 曲率变化率（需要更多历史点）
    dkappa_ds = 0.0
    if hasattr(self, "_prev_kappa"):
        ds = np.linalg.norm(p2 - p1)
        if ds > 1e-9:
            dkappa_ds = (kappa - self._prev_kappa) / ds
    
    self._prev_kappa = kappa
    
    # 归一化
    kappa_max = 1.0 / (self.epsilon / 2)  # 允差带半径的倒数
    dkappa_max = kappa_max / (self.epsilon / 2)  # 经验值
    
    kappa_norm = np.clip(kappa / kappa_max, -1.0, 1.0)
    dkappa_ds_norm = np.clip(dkappa_ds / dkappa_max, -1.0, 1.0)
    
    return kappa_norm, dkappa_ds_norm
```

---

## 3) 配置文件

### 3.1 创建 `configs/train_square_curvature.yaml`

```yaml
seed: 42

environment:
  epsilon: 1.5
  interpolation_period: 0.001
  max_steps: 4000
  lookahead_points: 0
  
  # v2.0 Phase 31: 启用曲率状态
  curvature_observation:
    enabled: true
    kappa_max: 1.33     # 1 / (ε/2)
    dkappa_max: 1.78    # kappa_max / (ε/2)

kinematic_constraints:
  MAX_VEL: 100.0
  MAX_ACC: 2000.0
  MAX_JERK: 20000.0
  MAX_ANG_VEL: 6.283185307179586
  MAX_ANG_ACC: 100.0
  MAX_ANG_JERK: 1000.0

reward_weights:
  # 继续使用 v2.0 极简奖励（不变）
  minimal_mode: true
  w_s: 20.0
  w_e: 0.0
  w_tau: 0.0
  w_smooth: 0.0
  
  boundary:
    enabled: true
    penalty: -100.0
  
  completion:
    enabled: true
    reward: 50.0
  
  corridor:
    enabled: false
  
  p4:
    time_penalty: -0.02
    stall_enabled: true
    stall_steps: 300
    stall_penalty: -8.0

ppo:
  actor_lr: 2.0e-05
  critic_lr: 0.0001
  hidden_dim: 256  # 可能需要增加以处理更多状态
  gamma: 0.99
  lmbda: 0.95
  epochs: 10
  eps: 0.1
  ent_coef: 0.01

training:
  use_obs_normalizer: false
  num_episodes: 200
  smoothing_factor: 0.9
  save_interval: 100
  log_interval: 50

path:
  type: square
  closed: true
  scale: 10.0
  num_points: 200

experiment:
  mode: train
  name: curvature_v1
  enable_kcm: true
```

---

## 4) 代码修改清单

### 4.1 `src/environment/cnc_env.py` - `__init__`

```python
# Phase 31: 曲率观测
curvature_cfg = self.reward_weights.get("curvature_observation", {})
self.enable_curvature_obs = curvature_cfg.get("enabled", False)
if self.enable_curvature_obs:
    self.curvature_keys = ["kappa_norm", "dkappa_ds_norm"]
    self.observation_dim = 12 + 2  # = 14
else:
    self.curvature_keys = []
    self.observation_dim = 12
```

### 4.2 `src/environment/cnc_env.py` - `_build_state`

```python
def _build_state(self):
    """构建状态向量"""
    # 原 12 维状态
    state = self._build_core_state()  # 8 维
    state += self._build_corner_state()  # 4 维
    
    # Phase 31: 曲率状态
    if self.enable_curvature_obs:
        kappa_norm, dkappa_norm = self.compute_curvature_features()
        state += [kappa_norm, dkappa_norm]
    
    return state
```

### 4.3 神经网络输入维度

PPO 网络自动适配 `observation_dim`，无需额外修改。

---

## 5) 执行步骤

### Step 1：确认 Phase 30 结果

```powershell
# 确认 Phase 30 的 corner_peak_omega
python tools/b2a1_corner_evidence.py --candidate artifacts/minimal_v1 --baseline artifacts/P0_L2/P0_12d_gold_20260114_174552
```

如果 `corner_peak_omega` 接近 MAX_ANG_VEL，继续本 Phase。

### Step 2：代码修改（30 分钟）
1. 添加曲率计算函数
2. 修改状态构建逻辑
3. 更新 observation_dim

### Step 3：训练
```powershell
conda activate PPO
cd PPO_project
python main.py --config configs/train_square_curvature.yaml --mode train
```

### Step 4：对比评估
```powershell
# 与 Phase 30 对比
python tools/plot_bundle_compare.py --run1 artifacts/minimal_v1 --run2 artifacts/curvature_v1 --out artifacts/compare_30_31
```

---

## 6) 验收标准

### 6.1 成功标准

| 指标 | 条件 |
|------|------|
| corner_peak_omega | 相对 Phase 30 下降 ≥ 10% |
| success_rate | ≥ 0.95 |
| max_abs_e_n | ≤ ε/2 |

### 6.2 如果成功

```
曲率状态有效 → 进入 Phase 32（多路径验证）
在论文中说明：状态扩展帮助策略感知曲率，加速平滑行为学习
```

### 6.3 如果仍未改善

```
考虑最后手段：加入轻量曲率惩罚
r_curvature = -0.001 * |dκ/ds|
这仍然比 v1.9 的 15+ 权重简单得多
```

---

## 7) 备选方案：轻量曲率惩罚

**仅当 Phase 31 状态扩展无效时使用**

```python
# 在 _calculate_minimal_reward 中添加
curvature_penalty_cfg = self.weights.get("curvature_penalty", {})
if curvature_penalty_cfg.get("enabled", False):
    w_curv = float(curvature_penalty_cfg.get("weight", 0.001))
    # dkappa_ds 从 ctx 获取
    r_curvature = -w_curv * abs(ctx.dkappa_ds)
else:
    r_curvature = 0.0
```

配置：
```yaml
curvature_penalty:
  enabled: true
  weight: 0.001  # 极小权重
```

**注意**：这仍然是极简的，只有 1 个额外参数，比 v1.9 的 corridor 方案简单得多。

---

## 8) 交付物

| 文件 | 说明 |
|------|------|
| `configs/train_square_curvature.yaml` | 配置 |
| `artifacts/curvature_v1/` | Run Bundle |
| `artifacts/compare_30_31/` | 对比分析 |

---

## 9) 论文映射

如果 Phase 31 成功：

> "为了加速策略学习，我们在状态空间中加入了执行轨迹的曲率特征（κ 和 dκ/ds）。实验表明，这种状态扩展帮助策略更快地学习到平滑过渡行为，同时保持了端到端学习的特性（无显式几何规则）。"

如果需要使用轻量曲率惩罚：

> "为了进一步提升平滑性，我们引入了极轻量的曲率连续性惩罚（单一权重 0.001）。与传统多权重 reward shaping 不同，这种设计保持了奖励函数的简洁性和可解释性。"
