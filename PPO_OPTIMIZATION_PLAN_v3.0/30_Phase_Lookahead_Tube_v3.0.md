# Phase 30：前瞻引导 + 管道奖励基线训练（v3.0）
版本日期：2026-01-18  
依赖：Phase 23 完成（P0_L2 基线可用）

---

## 0) 目标（一句话）

**用前瞻参考方向 + 管道奖励验证平滑学习假设**：消除参考方向跳变，释放带内自由度，让策略自主发现平滑过弯策略。

---

## 1) 核心假设（v3.0）

```
前瞻向量连续 → 策略可用小 ω 跟随
    +
管道内零惩罚 → 策略敢于利用空间切弯
    ↓
平滑轨迹自然涌现
```

---

## 2) 代码修改清单

### 2.1 修改 `src/environment/cnc_env.py`

#### 添加配置参数（`__init__`）

```python
# v3.0: 前瞻配置
lookahead_cfg = self.reward_weights.get("lookahead", {})
self.lookahead_enabled = bool(lookahead_cfg.get("enabled", True))
self.lookahead_dist = float(lookahead_cfg.get("distance", 3.0 * self.epsilon))
```

#### 修改 `_get_path_direction`

```python
def _get_path_direction(self, position, v_exec=None, record=False):
    """v3.0: 前瞻参考方向"""
    if not getattr(self, "lookahead_enabled", False):
        # 回退到原有切线逻辑
        return self._get_path_direction_legacy(position, v_exec, record)
    
    # 1. 获取当前弧长
    s_current = self._get_current_arc_length()
    
    # 2. 前瞻距离
    lookahead_dist = float(getattr(self, "lookahead_dist", 4.5))
    s_target = s_current + lookahead_dist
    
    # 3. 获取前瞻点
    target_point, _, _ = self._interpolate_point_at_s(s_target)
    
    # 4. 计算前瞻向量角度
    dx = float(target_point[0] - position[0])
    dy = float(target_point[1] - position[1])
    
    # 防止距离为零
    dist = math.sqrt(dx*dx + dy*dy)
    if dist < 1e-9:
        # 距离太近，使用切线方向
        return self._tangent_angle_at_s(s_current)
    
    theta_lookahead = math.atan2(dy, dx)
    
    if record:
        self._theta_ref_last = float(theta_lookahead)
    
    return float(theta_lookahead)

def _get_path_direction_legacy(self, position, v_exec=None, record=False):
    """原有切线逻辑（用于对比实验）"""
    # ... 保留原有代码 ...
```

### 2.2 修改 `src/environment/reward.py`

#### 修改 `_calculate_minimal_reward`

```python
def _calculate_minimal_reward(self, ctx: RewardContext) -> Tuple[float, Dict[str, float]]:
    """v3.0: 管道奖励 + 极简"""
    w_s = float(self.weights.get("w_s", 20.0))
    p4_cfg = self.weights.get("p4", {})
    time_penalty = float(p4_cfg.get("time_penalty", -0.02))
    stall_penalty = float(p4_cfg.get("stall_penalty", -8.0))
    boundary_cfg = self.weights.get("boundary", {})
    completion_cfg = self.weights.get("completion", {})
    
    # v3.0: 管道配置
    tube_cfg = self.weights.get("tube", {})
    tube_enabled = bool(tube_cfg.get("enabled", True))
    tube_ratio = float(tube_cfg.get("ratio", 0.5))  # 管道半径 = ratio × half_epsilon
    tube_tolerance = tube_ratio * self.half_epsilon
    
    # 1. Progress reward
    progress_now = float(ctx.progress)
    progress_diff = max(0.0, progress_now - float(self.last_progress))
    r_progress = w_s * progress_diff

    # 2. Contour penalty (v3.0: 管道奖励)
    r_contour = 0.0
    contour_error = abs(float(ctx.contour_error))
    
    if tube_enabled:
        if contour_error < tube_tolerance:
            r_contour = 0.0  # 管道内零惩罚
        else:
            # 超出管道部分惩罚
            excess = contour_error - tube_tolerance
            r_contour = -10.0 * (excess / self.half_epsilon) ** 2
    
    # 3. Boundary penalty (硬约束)
    r_boundary = 0.0
    if boundary_cfg.get("enabled", False):
        if contour_error > float(self.half_epsilon):
            r_boundary = float(boundary_cfg.get("penalty", -100.0))

    # 4. Time penalty
    r_time = time_penalty

    # 5. Completion reward
    r_completion = 0.0
    if completion_cfg.get("enabled", False) and ctx.lap_completed:
        r_completion = float(completion_cfg.get("reward", 50.0))

    # 6. Stall penalty
    r_stall = 0.0
    if ctx.stall_triggered:
        r_stall = stall_penalty

    total = r_progress + r_contour + r_boundary + r_time + r_completion + r_stall
    self.last_progress = progress_now

    return total, {
        "progress_diff": float(progress_diff),
        "r_progress": float(r_progress),
        "r_contour": float(r_contour),
        "r_boundary": float(r_boundary),
        "r_time": float(r_time),
        "r_completion": float(r_completion),
        "r_stall": float(r_stall),
        "total": float(total),
    }
```

---

## 3) 配置文件

### 创建 `configs/train_square_v30.yaml`

```yaml
seed: 42

environment:
  epsilon: 1.5
  interpolation_period: 0.001
  max_steps: 4000
  lookahead_points: 0

kinematic_constraints:
  MAX_VEL: 100.0
  MAX_ACC: 2000.0
  MAX_JERK: 20000.0
  MAX_ANG_VEL: 31.41592653589793  # 10 * pi (~1.8 deg/step)
  MAX_ANG_ACC: 500.0               # increased from 100
  MAX_ANG_JERK: 5000.0             # increased from 1000

reward_weights:
  minimal_mode: true
  w_s: 20.0
  w_e: 0.0
  w_tau: 0.0
  w_smooth: 0.0
  
  # v3.0: 前瞻配置
  lookahead:
    enabled: true
    distance: 4.5  # 3 × ε
  
  # v3.0: 管道配置
  tube:
    enabled: true
    ratio: 0.5  # 管道半径 = 0.5 × (ε/2)
  
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
  num_episodes: 500
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
  name: v30_lookahead_tube
  enable_kcm: true
```

---

## 4) 执行步骤

### Step 1：代码修改（30分钟）
1. 修改 `cnc_env.py` 添加前瞻逻辑
2. 修改 `reward.py` 添加管道奖励
3. 创建 `configs/train_square_v30.yaml`

### Step 2：Smoke Test（5分钟）
```powershell
conda activate PPO
cd PPO_project
python main.py --config configs/train_square_v30.yaml --mode train --episodes 5
```

验证：
- `env.lookahead_enabled == True`
- 前瞻方向连续变化
- 管道内误差不扣分

### Step 3：训练（2-3小时）
```powershell
python main.py --config configs/train_square_v30.yaml --mode train
```

### Step 4：评估
```powershell
python tools/acceptance_suite.py `
    --phase p0_eval `
    --config configs/train_square_v30.yaml `
    --model saved_models/v30_lookahead_tube/*/checkpoints/best_model.pth `
    --episodes 50 `
    --out out/v30_eval `
    --deterministic
```

---

## 5) 验收标准

### 必须项（MUST）

| 指标 | 条件 |
|------|------|
| success_rate | ≥ 0.95 |
| max_abs_e_n | ≤ 0.75mm |
| corner_peak_omega | < 0.85 × MAX_ANG_VEL |

### 涌现项（OBSERVE）

| 指标 | 期望 | 解读 |
|------|------|------|
| corner_peak_omega | < 0.7 × MAX | 显著平滑 |
| inside_ratio | > 0.6 | 明显内切 |
| corner_min_v | > 1.2 × baseline | 高速过弯 |

---

## 6) 结果处理

### 如果成功
```
corner_peak_omega 显著下降 + inside_ratio > 0.5
→ ✅ 假设验证，进入 Phase 32（多路径验证）
```

### 如果未涌现
```
corner_peak_omega 无变化
→ 检查前瞻距离参数
→ 检查管道半径参数
→ 如仍无效，考虑添加轻量角速度惩罚
```

---

## 7) 交付物

| 文件 | 说明 |
|------|------|
| `configs/train_square_v30.yaml` | 配置 |
| `artifacts/v30_lookahead_tube/` | Run Bundle |
| `artifacts/v30_lookahead_tube/summary.json` | 指标汇总 |
