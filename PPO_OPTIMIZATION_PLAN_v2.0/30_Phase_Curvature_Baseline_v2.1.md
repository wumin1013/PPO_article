# Phase 30：曲率感知基线训练（v2.1）
版本日期：2026-01-17  
依赖：Phase 23 完成（P0_L2 基线可用）

---

## 0) 目标（一句话）

**用 14 维曲率感知状态 + 极简奖励验证平滑学习假设**：策略能"看到"曲率，从而学会避免急转弯。

---

## 1) 核心假设（v2.1）

```
策略感知曲率κ → 发现高曲率导致KCM限速 → 主动选择低曲率轨迹 → 维持高速 → 更高累积奖励
```

> **与 v2.0 的区别**：v2.0 假设策略能从"后果"反推低曲率更好，但反馈链太长。  
> v2.1 让策略直接"看到"曲率，缩短学习路径。

---

## 2) 状态空间（14维）

### 2.1 完整状态向量

```python
# 8维核心特征
core_keys = [
    "contour_error_norm",    # |e| / half_ε
    "e_n_norm",              # e_n / half_ε（带符号法向误差）
    "heading_error_norm",    # τ / π
    "velocity_norm",         # v / MAX_VEL
    "acceleration_norm",     # a / MAX_ACC
    "angular_vel_norm",      # ω / MAX_ANG_VEL
    "overall_progress",      # [0, 1]
    "dist_to_turn_norm",     # dist_to_turn / dist_enter
]

# 4维拐角感知特征
corner_keys = [
    "turn_angle_norm",   # turn_angle / π
    "turn_sign",         # +1 左转 / -1 右转 / 0 直线
    "corner_phase",      # 1.0 在拐角期 / 0.0 否
    "inside_signed",     # turn_sign × e_n_norm
]

# 2维曲率感知特征（v2.1 新增）
curvature_keys = [
    "kappa_norm",        # κ / κ_max（执行轨迹曲率）
    "dkappa_ds_norm",    # (dκ/ds) / (dκ/ds)_max（曲率变化率）
]
```

### 2.2 曲率计算

```python
def compute_curvature_features(self) -> Tuple[float, float]:
    """从执行轨迹历史计算曲率特征"""
    if len(self.trajectory) < 3:
        return 0.0, 0.0
    
    p0, p1, p2 = self.trajectory[-3], self.trajectory[-2], self.trajectory[-1]
    v1, v2 = p1 - p0, p2 - p1
    
    # 三点定圆
    cross = abs(v1[0] * v2[1] - v1[1] * v2[0])
    chord = np.linalg.norm(p2 - p0)
    kappa = 2.0 * cross / (np.linalg.norm(v1) * np.linalg.norm(v2) * chord + 1e-9)
    
    # 曲率变化率
    dkappa_ds = 0.0
    if hasattr(self, "_prev_kappa"):
        ds = np.linalg.norm(p2 - p1)
        if ds > 1e-9:
            dkappa_ds = (kappa - self._prev_kappa) / ds
    self._prev_kappa = kappa
    
    # 归一化
    kappa_max = 1.0 / (self.epsilon / 2)
    dkappa_max = kappa_max / (self.epsilon / 2)
    
    return (
        np.clip(kappa / kappa_max, -1.0, 1.0),
        np.clip(dkappa_ds / dkappa_max, -1.0, 1.0)
    )
```

---

## 3) 奖励函数（极简，继承 v2.0）

```python
reward = r_progress + r_boundary + r_time + r_completion
       = 20×Δs + (0 or -100) + (-0.02) + (0 or 50)
```

| 奖励项 | 作用 |
|--------|------|
| r_progress | 唯一正向激励，驱动前进 |
| r_boundary | 越带硬惩罚 |
| r_time | 效率压力 |
| r_completion | 完成奖励 |

> **关键**：奖励函数保持极简，改进仅在状态空间。

---

## 4) 配置文件

### 创建 `configs/train_square_curvature_v21.yaml`

```yaml
seed: 42

environment:
  epsilon: 1.5
  interpolation_period: 0.001
  max_steps: 4000
  lookahead_points: 0
  
  # v2.1: 曲率观测（必须启用）
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
  
  p6_1:
    du_enabled: false

ppo:
  actor_lr: 2.0e-05
  critic_lr: 0.0001
  hidden_dim: 256
  gamma: 0.99
  lmbda: 0.95
  epochs: 10
  eps: 0.1
  ent_coef: 0.01

training:
  use_obs_normalizer: false
  num_episodes: 500  # 增加训练量
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
  name: curvature_v21
  enable_kcm: true
```

---

## 5) 代码修改清单

### 5.1 `src/environment/cnc_env.py`

#### `__init__` 方法
```python
# v2.1: 曲率观测
self.curvature_cfg = self.reward_weights.get("curvature_observation", {})
self.enable_curvature_obs = self.curvature_cfg.get("enabled", False)
if self.enable_curvature_obs:
    self.curvature_keys = ["kappa_norm", "dkappa_ds_norm"]
    self.observation_dim = 14  # 12 + 2
else:
    self.curvature_keys = []
    self.observation_dim = 12
```

#### `_build_state` 方法
```python
def _build_state(self) -> np.ndarray:
    # 原12维状态...
    state = [...]  # 12个特征
    
    # v2.1: 曲率状态
    if self.enable_curvature_obs:
        kappa_norm, dkappa_norm = self.compute_curvature_features()
        state.extend([kappa_norm, dkappa_norm])
    
    return np.array(state, dtype=np.float32)
```

#### 新增 `compute_curvature_features` 方法
（见第2.2节）

---

## 6) 执行步骤

### Step 1：代码修改（30分钟）
1. 修改 `cnc_env.py`，添加曲率计算
2. 创建 `configs/train_square_curvature_v21.yaml`
3. 验证 `env.observation_dim == 14`

### Step 2：训练（2-3小时）
```powershell
conda activate PPO
cd PPO_project
python main.py --config configs/train_square_curvature_v21.yaml --mode train
```

### Step 3：评估
```powershell
python tools/acceptance_suite.py `
    --phase p0_eval `
    --config configs/train_square_curvature_v21.yaml `
    --model saved_models/curvature_v21/*/checkpoints/best_model.pth `
    --episodes 50 `
    --out out/curvature_v21_eval `
    --deterministic
```

### Step 4：涌现分析
```powershell
python tools/b2a1_corner_evidence.py `
    --candidate artifacts/curvature_v21 `
    --baseline artifacts/P0_L2/P0_12d_gold_20260114_174552
```

---

## 7) 验收标准

### 7.1 必须项（MUST）

| 指标 | 条件 |
|------|------|
| success_rate | ≥ 0.95 |
| max_abs_e_n | ≤ ε/2 = 0.75 |
| corner_peak_omega | < 0.9 × MAX_ANG_VEL |

### 7.2 涌现指标（OBSERVE）

| 指标 | 期望 | 解读 |
|------|------|------|
| corner_peak_omega | < 0.8 × MAX | 平滑涌现 |
| corner_min_v | > baseline | 高速过弯 |
| inside_ratio | > 0.5 | 内切涌现 |

---

## 8) 结果处理

### 8.1 如果成功
```
corner_peak_omega 显著下降（相对 P0_L2）
→ ✅ 假设验证，进入 Phase 32（多路径验证）
```

### 8.2 如果未涌现
```
corner_peak_omega 无变化
→ 尝试轻量曲率惩罚（单一参数 w=0.001）
→ 如仍无效，调整论文叙事（见 99_Fallback）
```

---

## 9) 交付物

| 文件 | 说明 |
|------|------|
| `configs/train_square_curvature_v21.yaml` | 配置 |
| `artifacts/curvature_v21/` | Run Bundle |
| `artifacts/curvature_v21/summary.json` | 指标汇总 |

---

## 10) 论文映射

如果成功：

> "我们在状态空间中加入了执行轨迹的曲率特征（κ 和 dκ/ds）。结合极简奖励设计（仅包含进度激励和效率惩罚），策略成功学习了拐角平滑过渡策略。这表明，适度的状态增强可以使端到端学习在运动规划任务中有效收敛。"
